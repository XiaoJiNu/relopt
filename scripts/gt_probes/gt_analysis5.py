"""Stage-5 (decisive, lever-corrected): cross-bag /map relative GT error.

Model per bag:  enu = w * (p_map + R_yaw*v) + o
  v = RTK antenna lever arm in body frame (unmodelled in stage 1/4 -> inflated the
  "0.39 m GT noise floor" with a heading-dependent term).
Given w (rotation), the rest is LINEAR least squares in (o, w*v) = 4 unknowns.
Fit w by coarse search, then per-bag (o,u). Report:
  (a) per-bag residual after lever modelling  -> true /map-vs-RTK position error
  (b) cross-bag relative displacement error binned by baseline -> what the eval consumes
  (c) per-bag yaw bias vs dual-antenna heading + cross-bag bias difference -> yaw GT floor
"""
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

def fit_lever(pmap, yaw, enu, th):
    """enu = e^{i th}*(p + R_yaw v) + o ; linear LS in (o, u=e^{i th} v)."""
    w = complex(math.cos(th), math.sin(th))
    # R_yaw applied to v in 2D: R(yaw) v ; after w rotation: w*R(yaw)*v = R(yaw)*w*v? No:
    # w*(R(yaw) v) as complex: R(yaw)v = e^{i yaw} v (2D rotation) -> w*e^{i yaw}*v = e^{i yaw}*(w v)
    # so with u = w*v:  enu = w*p + e^{i yaw} * u + o   (linear in u,o)
    A = np.zeros((len(pmap)*2, 4)); b = np.zeros(len(pmap)*2)
    pc = pmap[:,0]+1j*pmap[:,1]
    ey = np.exp(1j*yaw)
    rhs = enu[:,0]+1j*enu[:,1] - w*pc
    A[0::2, 0] = ey.real; A[0::2, 1] = -ey.imag; A[0::2, 2] = 1.0
    A[1::2, 0] = ey.imag; A[1::2, 1] =  ey.real; A[1::2, 3] = 1.0
    b[0::2] = rhs.real; b[1::2] = rhs.imag
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    u = complex(sol[0], sol[1]); o = complex(sol[2], sol[3])
    pred = w*pc + ey*u + o
    resid = np.abs(pred - (enu[:,0]+1j*enu[:,1]))
    return w, u, o, resid

def joint_lever_fit(P):
    """shared w (map->ENU rotation) across bags of one group; per-bag (u,o)."""
    best = None
    for thd in np.arange(85.0, 95.01, 0.02):
        th = math.radians(thd); w = complex(math.cos(th), math.sin(th))
        tot = []
        for v in P.values():
            _,_,_,r = fit_lever(v['map'], v['yaw'], v['enu'], th)
            tot.append(r)
        m = np.median(np.concatenate(tot))
        if best is None or m < best[0]:
            best = (m, thd)
    _, thd = best
    w = complex(math.cos(math.radians(thd)), math.sin(math.radians(thd)))
    out = {}
    for n, v in P.items():
        _, u, o, r = fit_lever(v['map'], v['yaw'], v['enu'], math.radians(thd))
        out[n] = dict(u=u, o=o, resid=r)
    return thd, w, out

BINS=[(0.3,1.0),(1.0,2.0),(2.0,3.0),(3.0,5.0),(5.0,10.0),(10.0,60.0)]
def rel_err(P, w, F, a, b):
    """relative displacement of ANTENNA positions is what RTK measures; the eval consumes
    BODY(/map camera) positions. Body_A->Body_B displacement in ENU:
      body_enu = w*(p+v_rot) - w*R yaw v ... careful: antenna_enu = w*p + e^{i yaw} u + o
      => body_enu = antenna_enu - e^{i yaw} u  (remove lever)
    """
    A,B=P[a],P[b]
    ba = w*(A['map'][:,0]+1j*A['map'][:,1]) + A['o_']       # body in ENU (lever removed), o_ per bag
    bb = w*(B['map'][:,0]+1j*B['map'][:,1]) + B['o_']
    ea, eb = A['enu_c'], B['enu_c']                          # antenna ENU (RTK truth)
    # antenna->body in ENU = -e^{i yaw} u
    la = -np.exp(1j*A['yaw'])*A['u_']; lb = -np.exp(1j*B['yaw'])*B['u_']
    bodyA = ea + la; bodyB = eb + lb                          # RTK-derived body truth
    sub = slice(0,None,10)
    dmap = bb[:,None]-ba[sub][None,:]
    dtru = bodyB[:,None]-bodyA[sub][None,:]
    err = np.abs(dmap-dtru).ravel(); base=np.abs(dtru).ravel(); d=(dmap-dtru).ravel()
    uu = np.where(base>1e-9, dtru.ravel()/np.maximum(base,1e-9), 0)
    along=np.abs((d*np.conj(uu)).real); cross=np.abs((d*np.conj(uu)).imag)
    res={}
    for lo,hi in BINS:
        m=(base>=lo)&(base<hi)
        if m.sum()<80: continue
        res[f'{lo}-{hi}m']=dict(n=int(m.sum()),
            err_p50_p90_cm=[round(float(np.median(err[m]))*100,1), round(float(np.percentile(err[m],90))*100,1)],
            err_mean_cm=round(float(err[m].mean())*100,1),
            along_p50_p90_cm=[round(float(np.median(along[m]))*100,1), round(float(np.percentile(along[m],90))*100,1)],
            cross_p50_p90_cm=[round(float(np.median(cross[m]))*100,1), round(float(np.percentile(cross[m],90))*100,1)])
    return res

def yaw_bias(d, w):
    """per-bag yaw bias vs dual-antenna heading. heading_deg is CW-from-north (sign -1, offset ~-92deg
    measured in stage 1) => yaw_map_equiv = -heading - 90deg (mod). Report median residual after removing
    the constant offset; std = local yaw noise; median per bag = per-bag yaw bias."""
    if len(d['ht'])<20: return None
    idx=np.clip(np.searchsorted(d['t'],d['ht']),0,len(d['t'])-1)
    ym=d['yaw'][idx]; hy=np.radians(-d['hy'])
    diff=np.angle(np.exp(1j*(ym-hy)))
    off=float(np.median(diff)); r=np.angle(np.exp(1j*(diff-off)))
    return dict(off_deg=round(math.degrees(off),3), bias_resid_med_deg=round(float(np.degrees(np.median(r))),4),
                resid_std_deg=round(float(np.degrees(np.std(r))),3),
                resid_p50_abs_deg=round(float(np.degrees(np.median(np.abs(r)))),3), n=len(r))

def run(paths,label,pairs):
    bags={k:read_bag(p) for k,p in paths.items()}
    lat0=min(b['gnss'][0,1] for b in bags.values()); lon0=min(b['gnss'][0,2] for b in bags.values())
    P={}
    for n,d in bags.items():
        g=d['gnss']; e=enu_at(g[:,1],g[:,2],lat0,lon0)
        idx=np.clip(np.searchsorted(d['t'],g[:,0]),0,len(d['t'])-1)
        sub=slice(0,None,10)
        P[n]=dict(map=d['p'][idx][sub], yaw=d['yaw'][idx][sub], enu=e[sub],
                  enu_c=e[sub][:,0]+1j*e[sub][:,1], raw=bags[n])
    thd,w,F=joint_lever_fit(P)
    print(f'[{label}] shared map->ENU rotation = {thd:.3f} deg')
    yb={}
    for n,v in P.items():
        f=F[n]; v['u_']=f['u']; v['o_']=f['o']
        r=f['resid']
        print(f'  {n}: lever u=({f["u"].real:+.3f},{f["u"].imag:+.3f}) m |o|={abs(f["o"]):.2f} '
              f'resid p50/p90={np.median(r):.3f}/{np.percentile(r,90):.3f} m')
        yb[n]=yaw_bias(v['raw'], w)
        print(f'      yaw vs dual-antenna: {yb[n]}')
    out={'theta_deg':thd,'per_bag':{n:dict(lever=[round(F[n]['u'].real,3),round(F[n]['u'].imag,3)],
        resid_p50_p90=[round(float(np.median(F[n]['resid'])),3),round(float(np.percentile(F[n]['resid'],90)),3)],
        yaw=yb[n]) for n in P}}
    for a,b in pairs:
        out[f'{a}_vs_{b}']=rel_err(P,w,F,a,b)
        print(f'  {a}->{b} (lever-corrected):')
        for k,v in out[f'{a}_vs_{b}'].items(): print('    ',k,v)
    # cross-bag yaw bias difference
    for a,b in pairs:
        if yb[a] and yb[b]:
            print(f'  yaw bias diff {a} vs {b} = {abs(yb[a]["bias_resid_med_deg"]-yb[b]["bias_resid_med_deg"]):.4f} deg')
            out[f'yawbiasdiff_{a}_{b}']=round(abs(yb[a]['bias_resid_med_deg']-yb[b]['bias_resid_med_deg']),4)
    return out

res={}
res['lawn']=run(LAWN,'lawn',[('db_064638','q_062224'),('db_064638','q_081533')])
res['bldg']=run(BLDG,'bldg',[('db9_055425','q9_060415')])
json.dump(res, open('/home/yr/.hermes/cache/scratch/gt_analysis5.json','w'), indent=1)
print('DONE')
