"""Stage-3: cross-bag /map consistency via per-bag residual FIELDS (well-posed).

For each bag: global rigid map->ENU fit (w_g,o_g); residual field
  d(t) = enu(t) - (w_g*map_xy(t) + o_g)      [position warp residual, ~0.4 m p50]
  r(t) = yaw_map(t) - bearing_enu(t)         [yaw residual vs RTK path tangent]
Then interpolate d,r of each bag onto matched ENU positions along the loop:
  cross-bag position inconsistency = |d_A(enu) - d_B(enu)|
  cross-bag yaw inconsistency      = |r_A(enu) - r_B(enu)|
Shared RTK warp/tangent noise cancels in the DIFFERENCE at the same location;
what remains = genuine cross-bag /map inconsistency = the GT noise floor of the
relpose eval (labels AND metrics both come from cross-bag /map relative poses).
"""
import numpy as np, json, math
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from scipy.spatial.transform import Rotation as Rot
from scipy.ndimage import uniform_filter1d

BAGS = {
 "db_064638": ("/data/data/reLocationData/zc_data/20260921_064638.mcap", "lawn"),
 "q_062224": ("/data/data/reLocationData/zc_data/20260921_062224.mcap", "lawn"),
 "q_081533": ("/data/data/reLocationData/zc_data/20260920_081533.mcap", "lawn"),
 "db9_055425": ("/data/data/reLocationData/zc_data/研发大楼数据/20260923_055425_real_robot_record_external_lz4/bag_0.mcap", "bldg"),
 "q9_060415": ("/data/data/reLocationData/zc_data/研发大楼数据/20260923_060415_real_robot_record_external_lz4/bag_0.mcap", "bldg"),
}
# common ENU origin per group (first bag's first gnss sample) so bags are comparable
GROUP_ORIGIN = {}

def read_bag(path):
    mt, mp, myaw, gnss = [], [], [], []
    with open(path,'rb') as f:
        rd = make_reader(f, decoder_factories=[DecoderFactory()])
        for s,ch,m,dec in rd.iter_decoded_messages(topics=["/map","/ll/position/gps/gnss_fix"]):
            if ch.topic=="/map":
                p=dec.pose.pose
                mt.append(m.log_time*1e-9)
                mp.append((p.position.x,p.position.y))
                R=Rot.from_quat((p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w)).as_matrix()
                myaw.append(math.atan2(R[1,0],R[0,0]))
            else:
                gnss.append((m.log_time*1e-9, dec.latitude, dec.longitude))
    return dict(t=np.array(mt), p=np.array(mp), yaw=np.unwrap(np.array(myaw)), gnss=np.array(gnss))

R_e=6378137.0; f_i=1/298.257223563; e2=f_i*(2-f_i)
def enu_at(lat, lon, lat0, lon0):
    lat0r, lon0r = math.radians(lat0), math.radians(lon0)
    N0=R_e/np.sqrt(1-e2*math.sin(lat0r)**2)
    M=R_e*(1-e2)/(1-e2*math.sin(lat0r)**2)**1.5
    east=(np.radians(lon)-lon0r)*N0*math.cos(lat0r)
    north=(np.radians(lat)-lat0r)*M
    return east, north

def fit_transform(src, dst):
    a=src[:,0]+1j*src[:,1]; b=dst[:,0]+1j*dst[:,1]
    mu_a,mu_b=a.mean(),b.mean()
    w=((a-mu_a).conj()*(b-mu_b)).sum(); w/=abs(w)
    return w, mu_b-w*mu_a

def process(name, path, group):
    d = read_bag(path)
    g = d['gnss']
    if group not in GROUP_ORIGIN:
        GROUP_ORIGIN[group] = (g[0,1], g[0,2])
    lat0, lon0 = GROUP_ORIGIN[group]
    e, n = enu_at(g[:,1], g[:,2], lat0, lon0)
    enu = np.stack([e,n],1)
    # sample map at gnss times
    idx = np.clip(np.searchsorted(d['t'], g[:,0]), 0, len(d['t'])-1)
    mp = d['p'][idx]; myaw = d['yaw'][idx]
    w, o = fit_transform(mp, enu)
    fitted = np.stack([(w*(mp[:,0]+1j*mp[:,1])+o).real, (w*(mp[:,0]+1j*mp[:,1])+o).imag],1)
    dres = enu - fitted                      # position residual field
    dres_c = dres[:,0]+1j*dres[:,1]
    # bearing of motion from smoothed ENU (±1.5 m arc): complex velocity
    W = 61  # ~3 s at 20 Hz
    vel = np.gradient(enu, g[:,0], axis=0)  # m/s
    vel = np.stack([uniform_filter1d(vel[:,k], W) for k in range(2)],1)
    speed = np.linalg.norm(vel,axis=1)
    bearing = np.arctan2(vel[:,1], vel[:,0])
    # map yaw transformed into ENU frame: yaw_enu = yaw_map + arg(w)
    yaw_enu = myaw + math.atan2(w.imag, w.real)
    rres = np.angle(np.exp(1j*(yaw_enu - bearing)))  # yaw residual field
    good = speed > 0.15
    return dict(name=name, group=group, enu=enu[good], d=dres_c[good], r=rres[good],
                dw=float(np.median(np.abs(dres_c[good]))), speed=speed[good])

def interp_field(bagA, bagB, axis):
    """sample bagA's field at bagB's positions (nearest within 0.6 m), return diffs"""
    from scipy.spatial import cKDTree
    tr = cKDTree(bagA['enu'])
    dist, ind = tr.query(bagB['enu'], k=1)
    m = dist < 0.6
    if axis=='d':
        diff = np.abs(bagB['d'][m] - bagA['d'][ind[m]])
    else:
        diff = np.abs(np.angle(np.exp(1j*(bagB['r'][m] - bagA['r'][ind[m]]))))
    return diff, int(m.sum())

def _pct(x, p):
    return round(float(np.percentile(x, p)), 4) if len(x) else None

def main():
    bags = {}
    for name,(path,group) in BAGS.items():
        bags[name] = process(name, path, group)
        b=bags[name]
        print(f"{name}: n={len(b['enu'])} |d| p50={_pct(np.abs(b['d']),50)} p90={_pct(np.abs(b['d']),90)} m | "
              f"r p50={np.degrees(_pct(np.abs(b['r']),50) or 0):.3f} deg", flush=True)
    out={}
    pairs = [('db_064638','q_062224'),('db_064638','q_081533'),('q_062224','q_081533'),('db9_055425','q9_060415')]
    for a,b in pairs:
        dd,nd = interp_field(bags[a],bags[b],'d')
        rr,nr = interp_field(bags[a],bags[b],'r')
        out[f'{a}_vs_{b}'] = dict(
            n_pos=nd, dpos_p50_p90_cm=[round(float(np.median(dd))*100,1), round(_pct(dd,90)*100,1)] if nd else None,
            n_yaw=nr, dyaw_p50_p90_deg=[round(float(np.degrees(np.median(rr))),3), round(float(np.degrees(_pct(rr,90))),3)] if nr else None)
        print(a,'vs',b,json.dumps(out[f'{a}_vs_{b}']), flush=True)
    json.dump(out, open('/home/yr/.hermes/cache/scratch/gt_analysis3.json','w'), indent=1)

if __name__=='__main__':
    main()
