"""Stage-6: cross-bag YAW GT consistency (the angle target's floor) — two independent measures.

(a) /map-only: match samples of bag A and bag B at the same ENU position AND the same travel
    bearing (<0.4 m, <5 deg) => the physical heading is the same, so yaw_A - yaw_B is pure
    cross-bag /map yaw inconsistency. No dual-antenna needed.
(b) dual-antenna: per-bag constant offset (/map yaw) - (-heading_deg), bootstrap CI, and
    within-bag time drift of that offset (is it a bias or a drift?).
"""
import numpy as np, json, math
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial import cKDTree

LAWN = {
 "db_064638": "/data/data/reLocationData/zc_data/20260921_064638.mcap",
 "q_062224": "/data/data/reLocationData/zc_data/20260921_062224.mcap",
 "q_081533": "/data/data/reLocationData/zc_data/20260920_081533.mcap",
}
BLDG = {
 "db9_055425": "/data/data/reLocationData/zc_data/研发大楼数据/20260923_055425_real_robot_record_external_lz4/bag_0.mcap",
 "q9_060415": "/data/data/reLocationData/zc_data/研发大楼数据/20260923_060415_real_robot_record_external_lz4/bag_0.mcap",
}
R_e=6378137.0; f_i=1/298.257223563; e2=f_i*(2-f_i)

def read_bag(path):
    mt,mp,myaw,gnss,ht,hy=[],[],[],[],[],[]
    with open(path,'rb') as f:
        rd=make_reader(f, decoder_factories=[DecoderFactory()])
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
    return dict(t=np.array(mt),p=np.array(mp),yaw=np.array(myaw),gnss=np.array(gnss),
                ht=np.array(ht),hy=np.array(hy))

def enu_at(lat,lon,lat0,lon0):
    lat0r,lon0r=math.radians(lat0),math.radians(lon0)
    N0=R_e/np.sqrt(1-e2*math.sin(lat0r)**2); M=R_e*(1-e2)/(1-e2*math.sin(lat0r)**2)**1.5
    return np.stack([(np.radians(lon)-lon0r)*N0*math.cos(lat0r),(np.radians(lat)-lat0r)*M],1)

def wrap(a): return np.angle(np.exp(1j*a))

def prep(paths):
    bags={k:read_bag(p) for k,p in paths.items()}
    lat0=min(b['gnss'][0,1] for b in bags.values()); lon0=min(b['gnss'][0,2] for b in bags.values())
    P={}
    for n,d in bags.items():
        g=d['gnss']; e=enu_at(g[:,1],g[:,2],lat0,lon0)
        idx=np.clip(np.searchsorted(d['t'],g[:,0]),0,len(d['t'])-1)
        sub=slice(0,None,5)  # ~4 Hz
        enu=e[sub]; yaw=d['yaw'][idx][sub]; t=g[:,0][sub]
        # travel bearing from smoothed ENU
        v=np.gradient(enu, t, axis=0)
        W=41
        from scipy.ndimage import uniform_filter1d
        v=np.stack([uniform_filter1d(v[:,k],W) for k in range(2)],1)
        sp=np.linalg.norm(v,axis=1)
        bear=np.arctan2(v[:,1],v[:,0])
        m=sp>0.2
        P[n]=dict(enu=enu[m],yaw=yaw[m],bear=bear[m],t=t[m],raw=d,ok=m)
    return P

def crossbag_yaw(P,a,b,dpos=0.4,dbear=math.radians(5.0)):
    A,B=P[a],P[b]
    tr=cKDTree(A['enu'])
    dist,ind=tr.query(B['enu'],k=1)
    m=dist<dpos
    db_=wrap(B['bear']-A['bear'][ind])
    m &= np.abs(db_)<dbear
    dy=wrap(B['yaw'][m]-A['yaw'][ind[m]])
    return dict(n=int(m.sum()),
                dyaw_med_deg=round(float(np.degrees(np.median(dy))),3),
                dyaw_p50abs_deg=round(float(np.degrees(np.median(np.abs(dy)))),3),
                dyaw_p90abs_deg=round(float(np.degrees(np.percentile(np.abs(dy),90))),3),
                dyaw_std_deg=round(float(np.degrees(np.std(dy))),3))

def dualant(P,n):
    v=P[n]; d=v['raw']
    if len(d['ht'])<20: return None
    idx=np.clip(np.searchsorted(d['t'],d['ht']),0,len(d['t'])-1)
    ym=d['yaw'][idx]; off_raw=wrap(ym-np.radians(-d['hy']))
    off=float(np.median(off_raw)); r=wrap(off_raw-off)
    rng=np.random.default_rng(0)
    meds=[np.median(r[rng.integers(0,len(r),len(r))]) for _ in range(400)]
    # drift: split in 4 time quarters
    q=[round(float(np.degrees(np.median(r[i::4]))),3) for i in range(4)]
    return dict(off_deg=round(math.degrees(off),3), resid_std_deg=round(float(np.degrees(np.std(r))),3),
                resid_p50abs_deg=round(float(np.degrees(np.median(np.abs(r)))),3),
                boot_se_deg=round(float(np.degrees(np.std(meds))),4),
                quarter_resid_deg=q, n=len(r))

def run(paths,label,pairs):
    P=prep(paths)
    out={'dualant':{n:dualant(P,n) for n in P}}
    print(f'[{label}] dual-antenna offsets:', json.dumps(out['dualant'],ensure_ascii=False), flush=True)
    for a,b in pairs:
        out[f'{a}_vs_{b}']=crossbag_yaw(P,a,b)
        print(f'  cross-bag yaw {a} vs {b}: {out[f"{a}_vs_{b}"]}', flush=True)
        if out['dualant'][a] and out['dualant'][b]:
            d1=out['dualant'][a]['off_deg']; d2=out['dualant'][b]['off_deg']
            se=math.hypot(out['dualant'][a]['boot_se_deg'],out['dualant'][b]['boot_se_deg'])
            out[f'offdiff_{a}_{b}']=dict(diff_deg=round(d1-d2,3), se_deg=round(se,4))
            print(f'  dual-antenna offset diff {a}-{b} = {d1-d2:+.3f} deg (SE {se:.4f})', flush=True)
    return out

res={}
res['lawn']=run(LAWN,'lawn',[('db_064638','q_062224'),('db_064638','q_081533')])
res['bldg']=run(BLDG,'bldg',[('db9_055425','q9_060415')])
json.dump(res, open('/home/yr/.hermes/cache/scratch/gt_analysis6.json','w'), indent=1, ensure_ascii=False)
print('DONE')
