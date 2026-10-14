"""Stage-2 GT analysis: cross-bag relative-GT noise floor + /map yaw consistency vs gyro.

(1) Per-bag per-window (10s) rigid map->ENU fit (w,o) from stage-1 logic; bin windows by
    physical ENU position (4 m grid); compare bags sharing the map frame at the same place:
    |o_i - o_j| = cross-bag position GT inconsistency, angle(w_i/w_j) = cross-bag yaw GT
    inconsistency. This is the ACTUAL noise floor of inv(P_db)@P_q labels/metrics.
(2) /map relative rotation angle over 1-10 s lags vs lidar gyro integration (angle is
    frame-invariant, mounting rotation cancels): |a_map - a_gyro| = /map local yaw noise.
"""
import numpy as np, json, math, sys
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, '/home/yr/.hermes/cache/scratch')

BAGS = {
 "db_064638": "/data/data/reLocationData/zc_data/20260921_064638.mcap",
 "q_062224": "/data/data/reLocationData/zc_data/20260921_062224.mcap",
 "q_081533": "/data/data/reLocationData/zc_data/20260920_081533.mcap",
 "q9_060415": "/data/data/reLocationData/zc_data/研发大楼数据/20260923_060415_real_robot_record_external_lz4/bag_0.mcap",
}

def read_bag(path):
    mt, mp, mR, gnss = [], [], [], []
    with open(path,'rb') as f:
        rd = make_reader(f, decoder_factories=[DecoderFactory()])
        for s,ch,m,dec in rd.iter_decoded_messages(topics=["/map","/ll/position/gps/gnss_fix"]):
            if ch.topic=="/map":
                p=dec.pose.pose
                mt.append(m.log_time*1e-9)
                mp.append((p.position.x,p.position.y,p.position.z))
                mR.append(Rot.from_quat((p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w)).as_matrix())
            else:
                gnss.append((m.log_time*1e-9, dec.latitude, dec.longitude))
    return dict(t=np.array(mt), p=np.array(mp), R=np.array(mR), gnss=np.array(gnss) if gnss else None)

def enu(gnss):
    lat0, lon0 = gnss[0,1], gnss[0,2]
    R_e=6378137.0; f=1/298.257223563; e2=f*(2-f)
    lat=np.radians(gnss[:,1])
    lat0r, lon0r = math.radians(lat0), math.radians(lon0)
    N0=R_e/np.sqrt(1-e2*math.sin(lat0r)**2)
    M=R_e*(1-e2)/(1-e2*math.sin(lat0r)**2)**1.5
    east=(np.radians(gnss[:,2])-lon0r)*N0*math.cos(lat0r)
    north=(lat-lat0r)*M
    return np.stack([east,north],1)

def rigid_fit_resid(src, dst):
    a=src[:,0]+1j*src[:,1]; b=dst[:,0]+1j*dst[:,1]
    mu_a, mu_b = a.mean(), b.mean()
    w=((a-mu_a).conj()*(b-mu_b)).sum(); w/=abs(w)
    pred=w*(a-mu_a)+mu_b
    return np.abs(pred-b)


def fit_transform(src, dst):
    a = src[:,0] + 1j*src[:,1]; b = dst[:,0] + 1j*dst[:,1]
    mu_a, mu_b = a.mean(), b.mean()
    w = ((a-mu_a).conj()*(b-mu_b)).sum(); w /= abs(w)
    return w, mu_b - w*mu_a

def window_fits(d):
    """per 10s window: rigid map->ENU transform + window centroid (ENU)."""
    g = d['gnss']; e = enu(g)
    idx = np.clip(np.searchsorted(d['t'], g[:,0]), 0, len(d['t'])-1)
    mp = d['p'][idx][:,:2]
    out = []
    t0, t1 = g[:,0].min(), g[:,0].max()
    tt = t0
    while tt + 10 < t1:
        m = (g[:,0] >= tt) & (g[:,0] < tt+10)
        if m.sum() > 60:
            w, o = fit_transform(mp[m], e[m])
            c = e[m].mean(0)
            # window shape quality: residual after fit
            r = rigid_fit_resid(mp[m], e[m])
            out.append(dict(w=w, o=o, c=c, resid=float(np.median(r)), n=int(m.sum())))
        tt += 5
    return out

# ---- gyro read (lidar imu) ----
def read_gyro(path):
    gt, gw = [], []
    with open(path, 'rb') as f:
        rd = make_reader(f, decoder_factories=[DecoderFactory()])
        for s, ch, m, dec in rd.iter_decoded_messages(topics=['/lidar_imu']):
            a = dec.angular_velocity
            gt.append(m.log_time*1e-9)
            gw.append((a.x, a.y, a.z))
    return np.array(gt), np.array(gw)

def yaw_consistency(d, gt, gw, lags=(1.0, 3.0, 10.0)):
    """For each lag, sample times: angle of /map attitude delta vs gyro-integrated delta."""
    t, R = d['t'], d['R']
    # subsample map to ~20Hz for speed
    step = max(1, len(t)//8000)
    t, R = t[::step], R[::step]
    res = {}
    for lag in lags:
        diffs = []
        for i in range(0, len(t)-1, max(1, len(t)//400)):
            tA = t[i]
            j = np.searchsorted(t, tA+lag)
            if j >= len(t): continue
            tB = t[j]
            a_map = Rot.from_matrix(R[i].T @ R[j]).magnitude()
            mg = (gt >= tA) & (gt <= tB)
            if mg.sum() < 10: continue
            # integrate gyro (rotation vector)
            dt = np.diff(gt[mg])
            rv = (gw[mg][1:] * dt[:, None]).sum(0)
            a_gyro = np.linalg.norm(rv)
            diffs.append(np.degrees(abs(a_map - a_gyro)))
        diffs = np.array(diffs)
        res[f'lag{lag}s'] = dict(n=len(diffs), med_deg=round(float(np.median(diffs)), 4),
                                 p90_deg=round(float(np.percentile(diffs, 90)), 4)) if len(diffs) else None
    return res

def main():
    data = {}
    fits = {}
    for name, path in BAGS.items():
        d = read_bag(path)
        data[name] = d
        fits[name] = window_fits(d)
        print(name, 'windows:', len(fits[name]), flush=True)
    # gyro consistency (lawn query + db)
    out = {}
    for name in ('db_064638', 'q_062224', 'q_081533', 'q9_060415'):
        gt, gw = read_gyro(BAGS[name])
        out['gyro_' + name] = yaw_consistency(data[name], gt, gw)
        print(name, out['gyro_' + name], flush=True)
    # cross-bag window comparison, binned by ENU cell
    CELL = 4.0
    def cell(c): return (int(c[0]//CELL), int(c[1]//CELL))
    groups = [('lawn', ['db_064638', 'q_062224', 'q_081533']), ('bldg', ['q9_060415'])]
    for gname, bags in groups:
        bins = {}
        for b in bags:
            for wf in fits[b]:
                bins.setdefault(cell(wf['c']), []).append((b, wf))
        dpos, dyaw = [], []
        for k, v in bins.items():
            bl = list({b for b, _ in v})
            if len(bl) < 2: continue
            for i in range(len(v)):
                for j in range(i+1, len(v)):
                    (b1, f1), (b2, f2) = v[i], v[j]
                    if b1 == b2: continue
                    if f1['resid'] > 0.15 or f2['resid'] > 0.15: continue  # bad-fit windows (RTK glitch)
                    dpos.append(abs(f1['o'] - f2['o']))
                    dy = math.degrees(math.atan2((f1['w']*np.conj(f2['w'])).imag, (f1['w']*np.conj(f2['w'])).real))
                    dyaw.append(abs((dy+180) % 360 - 180))
        dpos, dyaw = np.array(dpos), np.array(dyaw)
        out['crossbag_' + gname] = dict(n_pairs=len(dpos), n_cells=len(bins),
            dpos_med_p50_p90_cm=[round(float(np.median(dpos))*100, 1), round(float(np.percentile(dpos, 50))*100, 1), round(float(np.percentile(dpos, 90))*100, 1)] if len(dpos) else None,
            dyaw_med_p90_deg=[round(float(np.median(dyaw)), 3), round(float(np.percentile(dyaw, 90)), 3)] if len(dyaw) else None)
        print(gname, out['crossbag_' + gname], flush=True)
    json.dump(out, open('/home/yr/.hermes/cache/scratch/gt_analysis2.json', 'w'), indent=1)

if __name__ == '__main__':
    main()
