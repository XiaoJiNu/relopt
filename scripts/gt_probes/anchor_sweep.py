"""Anchor-sweep on dumped per-candidate relpose (FT weights, odd subset, K=10). Zero GPU.
bm3's anchor = median of implied candidate positions is rank-blind; sweep anchoring on the
retrieval top-1 (MegaLoc most-similar db frame) instead. Product: anchor_sweep.json.
Input: reloc3r/data/zc_sim3/cands_<key>_K10_ft_odd.npz (from dump_cands_ft.py)."""
import numpy as np, json, math
from scipy.spatial.transform import Rotation as Rot

def rot_avg(Rs):
    q = Rot.from_matrix(Rs).as_quat()
    q = q * np.sign(q @ q[0] + 1e-9)[:, None]
    qm = q.mean(0); qm /= np.linalg.norm(qm)
    return Rot.from_quat(qm).as_matrix()

def tri(lines):
    A=np.zeros((3,3)); b=np.zeros(3)
    for p0,p1 in lines:
        d=p1-p0; d/=(np.linalg.norm(d)+1e-12); M=np.eye(3)-np.outer(d,d); A+=M; b+=M@p0
    return np.linalg.solve(A+1e-9*np.eye(3),b)

def agg(pdb,pq2d,keep):
    qR=[];lines=[]
    for i in keep:
        pq=pdb[i]@pq2d[i]; qR.append(pq[:3,:3])
        lines.append((pdb[i][:3,3],(pdb[i]@pq2d[i][:4,3])[:3]))
    Rt=np.eye(4);Rt[:3,:3]=rot_avg(np.array(qR));Rt[:3,3]=tri(lines);return Rt

def implied(pdb,pq2d,n):
    return np.array([(pdb[k]@pq2d[k])[:3,3] for k in range(n)])

res={}
for key in ('062224','081533','0923'):
    d=np.load(f'/data/code/reLocation-all/reloc3r/data/zc_sim3/cands_{key}_K10_ft_odd.npz')
    pdb,pq2d,gt=d['pdb'],d['pq2d'],d['gt']
    cfgs={}
    variants=[
        ('K5_bm3', lambda ip,n: (lambda rough: [k for k in range(min(5,n)) if np.linalg.norm(ip[k]-rough)<3.0] or list(range(min(5,n))))(np.median(ip[:min(5,n)],0))),
        ('K5_anchor1_r1.5', lambda ip,n: [k for k in range(min(5,n)) if np.linalg.norm(ip[k]-ip[0])<1.5]),
        ('K5_anchor1_r3', lambda ip,n: [k for k in range(min(5,n)) if np.linalg.norm(ip[k]-ip[0])<3.0]),
        ('K10_anchor1_r1.5', lambda ip,n: [k for k in range(n) if np.linalg.norm(ip[k]-ip[0])<1.5]),
        ('K10_mode_r1.5', lambda ip,n: (lambda c:[k for k in range(n) if np.linalg.norm(ip[k]-ip[c[0]])<1.5])(sorted(range(n), key=lambda k:(-sum(np.linalg.norm(ip[j]-ip[k])<1.5 for j in range(n)), k)))),
    ]
    for nm,sel in variants:
        T=[]
        for i in range(len(gt)):
            n=pdb[i].shape[0]; ip=implied(pdb[i],pq2d[i],n)
            keep=sel(ip,n) or [0]
            Rt=agg(pdb[i],pq2d[i],keep)
            T.append(np.linalg.norm(Rt[:3,3]-gt[i][:3,3]))
        T=np.array(T)
        cfgs[nm]=dict(ep_mean=round(float(T.mean()),4),ep_med=round(float(np.median(T)),4),
                      ep_p90=round(float(np.percentile(T,90)),4),n_gt0p5=int((T>0.5).sum()),
                      top3=[round(float(x),1) for x in np.sort(T)[::-1][:3]])
    res[key]=cfgs
    print(key)
    for nm,s in cfgs.items(): print('  ',nm,s)
json.dump(res,open('/home/yr/.hermes/cache/scratch/anchor_sweep.json','w'),indent=1)
