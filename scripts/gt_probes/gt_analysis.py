"""GT feasibility analysis for targets: angle<0.2deg, along<10cm, cross<3cm.
Reads /map (+covariance), gnss_fix, heading from each ZC bag; measures:
1) /map self-reported covariance (position sigma, yaw sigma from quaternion->cov)
2) global rigid map<-ENU fit residual (known ~0.39m p50)
3) sliding-window (5s) rigid fit residual -> LOCAL noise floor (what relpose eval actually sees)
4) high-freq jitter: /map pos minus local quadratic fit
5) /map yaw vs dual-antenna heading_deg_stamped -> yaw noise bound
6) cross-bag: per-bag windowed-fit drift series comparison (same-day vs db)
"""
import numpy as np, json, math
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from scipy.spatial.transform import Rotation as Rot

BAGS = {
 "db_064638": "/data/data/reLocationData/zc_data/20260921_064638.mcap",
 "q_062224": "/data/data/reLocationData/zc_data/20260921_062224.mcap",
 "q_081533": "/data/data/reLocationData/zc_data/20260920_081533.mcap",
 "q9_060415": "/data/data/reLocationData/zc_data/研发大楼数据/20260923_060415_real_robot_record_external_lz4/bag_0.mcap",
}

def read_bag(path):
    mt, mp, mR, cov, gnss, ht, hy = [], [], [], [], [], [], []
    with open(path,'rb') as f:
        rd = make_reader(f, decoder_factories=[DecoderFactory()])
        for s,ch,m,dec in rd.iter_decoded_messages(topics=["/map","/ll/position/gps/gnss_fix","/ll/position/gps/heading_deg_stamped"]):
            if ch.topic=="/map":
                p=dec.pose.pose; c=dec.pose.covariance
                mt.append(m.log_time*1e-9)
                mp.append((p.position.x,p.position.y,p.position.z))
                mR.append(Rot.from_quat((p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w)).as_matrix())
                cov.append(np.array(c).reshape(6,6))
            elif ch.topic.endswith("gnss_fix"):
                gnss.append((m.log_time*1e-9, dec.latitude, dec.longitude))
            else:
                ht.append(m.log_time*1e-9); hy.append(float(dec.vector.y))
    return dict(t=np.array(mt), p=np.array(mp), R=np.array(mR), cov=np.array(cov),
                gnss=np.array(gnss) if gnss else None, ht=np.array(ht), hy=np.array(hy))

def enu(gnss):
    lat0, lon0 = gnss[0,1], gnss[0,2]
    R_e=6378137.0; f=1/298.257223563; e2=f*(2-f)
    lat=np.radians(gnss[:,1]); lon=np.radians(gnss[:,2])
    N=R_e/np.sqrt(1-e2*np.sin(lat)**2)
    x=(N)*np.cos(lat)*np.cos(lon); y=(N)*np.cos(lat)*np.sin(lon)
    # local ENU about lat0/lon0
    lat0r, lon0r = math.radians(lat0), math.radians(lon0)
    N0=R_e/np.sqrt(1-e2*math.sin(lat0r)**2)
    x0=N0*math.cos(lat0r)*math.cos(lon0r); y0=N0*math.cos(lat0r)*math.sin(lon0r)
    # simple equirectangular projection around origin (site ~100m, error negligible)
    east=(lon-np.radians(lon0))*N0*math.cos(lat0r)
    north=(lat-np.radians(lat0))*R_e*(1-e2)/(1-e2*math.sin(lat0r)**2)**1.5*N0/R_e*R_e  # meridian radius
    M=R_e*(1-e2)/(1-e2*math.sin(lat0r)**2)**1.5
    north=(lat-lat0r)*M
    return np.stack([east,north],1)

def rigid_fit_resid(src, dst):
    """2D complex LS rigid fit src->dst, return residuals"""
    a=src[:,0]+1j*src[:,1]; b=dst[:,0]+1j*dst[:,1]
    mu_a, mu_b = a.mean(), b.mean()
    w=((a-mu_a).conj()*(b-mu_b)).sum(); w/=abs(w)
    pred=w*(a-mu_a)+mu_b
    return np.abs(pred-b)

def yaw_of(R):
    # heading of body x-axis in map frame
    return np.arctan2(R[:,1,0], R[:,0,0])

def analyze(name, d):
    t,p,R,cov = d['t'],d['p'],d['R'],d['cov']
    out={'bag':name,'n_map':len(t)}
    # 1) covariance self-report
    pos_sig=np.sqrt(np.clip(np.einsum('nii->ni',cov[:,:3,:3]),0,None))
    out['cov_pos_sigma_med']=[round(float(np.median(pos_sig[:,i])),4) for i in range(3)]
    rot_sig=np.sqrt(np.clip(np.einsum('nii->ni',cov[:,3:,3:]),0,None))
    out['cov_rot_sigma_med_rad']=[round(float(np.median(rot_sig[:,i])),5) for i in range(3)]
    out['cov_rot_sigma_med_deg']=[round(float(np.degrees(np.median(rot_sig[:,i]))),4) for i in range(3)]
    # 2) global + 3) windowed rigid fit vs RTK ENU
    if d['gnss'] is not None and len(d['gnss'])>100:
        g=d['gnss']; e=enu(g)
        # sample /map at gnss times
        idx=np.clip(np.searchsorted(t,g[:,0]),0,len(t)-1)
        mp2=p[idx]
        r_global=rigid_fit_resid(mp2[:,:2], e)
        out['rtk_global_resid_p50_p90']=[round(float(np.median(r_global)),3), round(float(np.percentile(r_global,90)),3)]
        # windowed: 10s windows with 5s step, rigid fit each window
        wins=[]
        t0,t1=g[:,0].min(),g[:,0].max()
        tt=t0
        res_w=[]
        centers=[]
        while tt+10 < t1:
            m=(g[:,0]>=tt)&(g[:,0]<tt+10)
            if m.sum()>50:
                rw=rigid_fit_resid(mp2[m][:,:2], e[m])
                res_w.append(np.median(rw)); centers.append(tt-t0)
            tt+=5
        res_w=np.array(res_w)
        out['rtk_win10s_resid_p50_med']=[round(float(np.median(res_w)),4), round(float(np.percentile(res_w,50)),4)] if len(res_w) else None
        out['rtk_win10s_resid_p50_list_head']=[round(x,3) for x in res_w[:8]] if len(res_w) else None
        out['n_win']=len(res_w)
        # 4) high-freq jitter of /map itself: pos minus local quadratic (1s window centered)
        jj=[]
        step=max(1,int(len(t)/3000))
        for i in range(50,len(t)-50,step):
            m=(t>t[i]-0.5)&(t<t[i]+0.5)
            if m.sum()<20: continue
            dt=t[m]-t[i]
            for ax in range(2):
                c=np.polyfit(dt,p[m,ax],2)
                jj.append(abs(p[i,ax]-np.polyval(c,0.0)))
        jj=np.array(jj)
        out['map_hf_jitter_p50_p90_cm']=[round(float(np.median(jj))*100,2), round(float(np.percentile(jj,90))*100,2)]
        # 5) yaw vs dual-antenna heading
        if len(d['ht'])>10:
            hy=np.unwrap(np.radians(d['hy']))
            iy=np.clip(np.searchsorted(t,d['ht']),0,len(t)-1)
            ymap=yaw_of(R[iy])
            diff=(hy-ymap+np.pi)%(2*np.pi)-np.pi
            # heading convention may be CW-from-north vs CCW; test both signs & offsets
            best=None
            for s in (1,-1):
                dd=(s*hy-ymap+np.pi)%(2*np.pi)-np.pi
                off=np.median(dd); dd=(dd-off+np.pi)%(2*np.pi)-np.pi
                sc=np.median(np.abs(dd))
                if best is None or sc<best[0]: best=(sc,s,float(np.degrees(off)),dd)
            sc,s,off,dd=best
            out['yaw_vs_dualantenna']={'sign':int(s),'offset_deg':round(off,2),
                'resid_med_deg':round(float(np.degrees(np.median(np.abs(dd)))),3),
                'resid_p90_deg':round(float(np.degrees(np.percentile(np.abs(dd),90))),3),
                'resid_std_deg':round(float(np.degrees(np.std(dd))),3),'n':len(dd)}
    return out

res=[]
data={}
for name,path in BAGS.items():
    d=read_bag(path); data[name]=d
    r=analyze(name,d); res.append(r)
    print(json.dumps(r,ensure_ascii=False))

# 6) cross-bag consistency: same-day q vs db — compare each bag's map-frame vs RTK-ENU fit transform
# if both bags share one map frame and RTK is consistent, per-bag global rigid transforms should agree
def fit_transform(src,dst):
    a=src[:,0]+1j*src[:,1]; b=dst[:,0]+1j*dst[:,1]
    mu_a,mu_b=a.mean(),b.mean()
    w=((a-mu_a).conj()*(b-mu_b)).sum(); w/=abs(w)
    return w, mu_b - w*mu_a
tf={}
for name,d in data.items():
    if d['gnss'] is None or len(d['gnss'])<100: continue
    g=d['gnss']; e=enu(g)
    idx=np.clip(np.searchsorted(d['t'],g[:,0]),0,len(d['t'])-1)
    w,o=fit_transform(d['p'][idx][:,:2], e)
    tf[name]=(w,o)
    print(name, 'rot_deg=%.3f'%math.degrees(math.atan2(w.imag,w.real)), 'offset=(%.3f,%.3f)'%(o.real,o.imag))
print(json.dumps(res,ensure_ascii=False), file=open('/home/yr/.hermes/cache/scratch/gt_analysis.json','w'))
