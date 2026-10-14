"""Stage-4 v2 (fast): cross-bag /map relative-displacement GT error vs RTK-ENU, binned by baseline.
Subsample both sides to ~1 Hz; same math as v1."""
import numpy as np, json, math
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from scipy.spatial.transform import Rotation as Rot

LAWN = {
 "db_064638": "/data/data/reLocationData/zc_data/20260921_064638.mcap",
 "q_062224": "/data/data/reLocationData/zc_data/20260921_062224.mcap",
 "q_081533": "/data/data/reLocationData/zc_data/20260920_081533.mcap",
}
BLDG = {
 "db9_055425": "/data/data/reLocationData/zc_data/研发大楼数据/20260923_055425_real_robot_record_external_lz4/bag_0.mcap",
 "q9_060415": "/data/data/reLocationData/zc_data/研发大楼数据/20260923_060415_real_robot_record_external_lz4/bag_0.mcap",
}
R_e = 6378137.0; f_i = 1/298.257223563; e2 = f_i*(2-f_i)

def read_bag(path):
    mt, mp, myaw, gnss, ht, hy = [], [], [], [], [], []
    with open(path,'rb') as f:
        rd = make_reader(f, decoder_factories=[DecoderFactory()])
        for s,ch,m,dec in rd.iter_decoded_messages(topics=["/map","/ll/position/gps/gnss_fix","/ll/position/gps/heading_deg_stamped"]):
            if ch.topic=="/map":
                p=dec.pose.pose
                mt.append(m.log_time*1e-9); mp.append((p.position.x,p.position.y))
                R=Rot.from_quat((p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w)).as_matrix()
                myaw.append(math.atan2(R[1,0],R[0,0]))
            elif ch.topic.endswith("gnss_fix"):
                gnss.append((m.log_time*1e-9, dec.latitude, dec.longitude))
            else:
                ht.append(m.log_time*1e-9); hy.append(float(dec.vector.y))
    return dict(t=np.array(mt), p=np.array(mp), yaw=np.array(myaw),
                gnss=np.array(gnss), ht=np.array(ht), hy=np.array(hy))

def enu_at(lat, lon, lat0, lon0):
    lat0r, lon0r = math.radians(lat0), math.radians(lon0)
    N0 = R_e/np.sqrt(1-e2*math.sin(lat0r)**2)
    M  = R_e*(1-e2)/(1-e2*math.sin(lat0r)**2)**1.5
    return np.stack([(np.radians(lon)-lon0r)*N0*math.cos(lat0r), (np.radians(lat)-lat0r)*M], 1)

def prep(bags):
    lat0 = min(b['gnss'][0,1] for b in bags.values()); lon0 = min(b['gnss'][0,2] for b in bags.values())
    out={}
    for name,d in bags.items():
        g=d['gnss']; e=enu_at(g[:,1],g[:,2],lat0,lon0)
        idx=np.clip(np.searchsorted(d['t'],g[:,0]),0,len(d['t'])-1)
        sub=slice(0,None,20)  # ~1 Hz
        out[name]=dict(map=d['p'][idx][sub], enu=e[sub])
    return out

def joint_fit(P):
    src=np.concatenate([v['map'] for v in P.values()]); dst=np.concatenate([v['enu'] for v in P.values()])
    a=src[:,0]+1j*src[:,1]; b=dst[:,0]+1j*dst[:,1]
    mu_a,mu_b=a.mean(),b.mean()
    w=((a-mu_a).conj()*(b-mu_b)).sum(); w/=abs(w)
    return w, mu_b-w*mu_a

BINS=[(0.3,1.0),(1.0,2.0),(2.0,3.0),(3.0,5.0),(5.0,10.0),(10.0,20.0),(20.0,60.0)]
def rel_err(P,w,a,b):
    A,B=P[a],P[b]
    dm=(B['map'][:,None,0]-A['map'][None,:,0])+1j*(B['map'][:,None,1]-A['map'][None,:,1])
    de=(B['enu'][:,None,0]-A['enu'][None,:,0])+1j*(B['enu'][:,None,1]-A['enu'][None,:,1])
    dm*=w; dm=dm.ravel(); de=de.ravel()
    base=np.abs(de); err=np.abs(dm-de)
    u=np.where(base>1e-9, de/np.maximum(base,1e-9), 0)
    d=dm-de
    along=np.abs((d*np.conj(u)).real); cross=np.abs((d*np.conj(u)).imag)
    res={}
    for lo,hi in BINS:
        m=(base>=lo)&(base<hi)
        if m.sum()<50: continue
        res[f'{lo}-{hi}m']=dict(n=int(m.sum()),
            err_p50_p90_cm=[round(float(np.median(err[m]))*100,1), round(float(np.percentile(err[m],90))*100,1)],
            err_mean_cm=round(float(err[m].mean())*100,1),
            along_p50_cm=round(float(np.median(along[m]))*100,1),
            cross_p50_cm=round(float(np.median(cross[m]))*100,1))
    return res

def run(paths,label,pairs):
    bags={k:read_bag(p) for k,p in paths.items()}
    P=prep(bags); w,o=joint_fit(P)
    print(f'[{label}] w={math.degrees(math.atan2(w.imag,w.real)):.3f}deg', flush=True)
    for n,v in P.items():
        r=np.abs((v['map'][:,0]+1j*v['map'][:,1])*w+o-(v['enu'][:,0]+1j*v['enu'][:,1]))
        print(f'  {n}: global resid p50/p90={np.median(r):.3f}/{np.percentile(r,90):.3f} m', flush=True)
    out={}
    for a,b in pairs:
        out[f'{a}_vs_{b}']=rel_err(P,w,a,b)
        print(f'  {a}->{b}:', flush=True)
        for k,v in out[f'{a}_vs_{b}'].items(): print('    ',k,v, flush=True)
    return out

res={}
res['lawn']=run(LAWN,'lawn',[('db_064638','q_062224'),('db_064638','q_081533')])
res['bldg']=run(BLDG,'bldg',[('db9_055425','q9_060415')])
json.dump(res, open('/home/yr/.hermes/cache/scratch/gt_analysis4.json','w'), indent=1)
print('DONE')
