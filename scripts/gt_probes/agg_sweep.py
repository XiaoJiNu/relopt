"""Offline aggregation sweep on dumped per-candidate relpose (FT weights, odd subset, K=10).
Zero GPU. Tests: K choice, bm3 radius, pairwise-consistency RANSAC, rotation-median gating,
oracle-best-K (upper bound of what aggregation can buy).
"""
import numpy as np, json, itertools, sys
from scipy.spatial.transform import Rotation as Rot

SC = {'062224': ('/data/code/reLocation-all/reloc3r/data/zc_fix', '20260921_062224', '20260921_064638'),
      '081533': ('/data/code/reLocation-all/reloc3r/data/zc_fix', '20260920_081533', '20260921_064638'),
      '0923': ('/data/code/reLocation-all/reloc3r/data/zc_0923',
               '20260923_060415_real_robot_record_external_lz4',
               '20260923_055425_real_robot_record_external_lz4')}

def rot_avg(Rs):
    q = Rot.from_matrix(Rs).as_quat()
    ref = q[0]
    q = q * np.sign(q @ ref.T + 1e-9)[:, None] if q.ndim == 2 else q
    qm = q.mean(0); qm /= np.linalg.norm(qm)
    return Rot.from_quat(qm).as_matrix()

def tri(lines):
    """least-squares intersection of camera-center rays (Reloc3r camera_center_triangulation)"""
    A = np.zeros((3, 3)); b = np.zeros(3)
    for p0, p1 in lines:
        d = p1 - p0; d = d / (np.linalg.norm(d) + 1e-12)
        M = np.eye(3) - np.outer(d, d)
        A += M; b += M @ p0
    try:
        return np.linalg.solve(A + 1e-9*np.eye(3), b)
    except np.linalg.LinAlgError:
        return p0

def aggregate(pdb, pq2d, keep):
    qR, lines = [], []
    for i in keep:
        pose_q = pdb[i] @ pq2d[i]
        qR.append(pose_q[:3, :3])
        lines.append((pdb[i][:3, 3], (pdb[i] @ pq2d[i][:4, 3])[:3]))
    Rt = np.eye(4); Rt[:3, :3] = rot_avg(np.array(qR)); Rt[:3, 3] = tri(lines)
    return Rt

def err(Rt, gt):
    d = Rt[:3, 3] - gt[:3, 3]
    te = float(np.linalg.norm(d))
    re = float(np.degrees(Rot.from_matrix(Rt[:3, :3].T @ gt[:3, :3]).magnitude()))
    return te, re, d

def bm3_set(pdb, pq2d, n, rad):
    ct = np.array([(pdb[k] @ pq2d[k])[:3, 3] for k in range(n)])
    rough = np.median(ct, 0)
    keep = [k for k in range(n) if np.linalg.norm(pdb[k][:3, 3] - rough) < rad]
    return keep if len(keep) >= 2 else list(range(n))

def consistency_set(pdb, pq2d, n, pos_thr, rot_thr):
    """pairwise-consistency RANSAC-lite: score each candidate by how many others agree."""
    cand = np.array([(pdb[k] @ pq2d[k])[:3, 3] for k in range(n)])
    rots = np.array([Rot.from_matrix((pdb[k] @ pq2d[k])[:3, :3]).as_rotvec() for k in range(n)])
    sc = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            if np.linalg.norm(cand[i] - cand[j]) < pos_thr and \
               np.degrees(np.linalg.norm(rots[i] - rots[j])) < rot_thr:
                sc[i] += 1
    best = int(sc.argmax())
    keep = [k for k in range(n) if np.linalg.norm(cand[k] - cand[best]) < pos_thr
            and np.degrees(np.linalg.norm(rots[k] - rots[best])) < rot_thr]
    return keep if len(keep) >= 2 else [best]

def stats(T, R):
    return dict(n=len(T), ep_mean=round(float(T.mean()), 4), ep_med=round(float(np.median(T)), 4),
                ep_p90=round(float(np.percentile(T, 90)), 4),
                eth_mean=round(float(R.mean()), 4), eth_med=round(float(np.median(R)), 4),
                eth_p90=round(float(np.percentile(R, 90)), 4))

def main():
    out = {}
    for key in SC:
        f = f'/data/code/reLocation-all/reloc3r/data/zc_sim3/cands_{key}_K10_ft_odd.npz'
        d = np.load(f)
        pdb, pq2d, gt, names = d['pdb'], d['pq2d'], d['gt'], d['names']
        root, q, db = SC[key]
        # travel dir per query for along/cross split
        qn_all = sorted(n[:-10] for n in __import__('os').listdir(f'{root}/{q}/seq-01') if n.endswith('.color.png'))
        pos = {n: np.loadtxt(f'{root}/{q}/seq-01/{n}.pose.txt')[:3, 3] for n in qn_all}
        tdirs = []
        for nm in names:
            i = qn_all.index(str(nm))
            a, b = max(0, i-2), min(len(qn_all)-1, i+2)
            v = pos[qn_all[b]] - pos[qn_all[a]]; tdirs.append(v / (np.linalg.norm(v) + 1e-9))
        tdirs = np.array(tdirs)
        cfgs = {}
        variants = []
        for K in (3, 5, 7, 10):
            variants.append((f'K{K}_all', lambda p, q2, n, K=K: list(range(K))))
            variants.append((f'K{K}_bm3', lambda p, q2, n, K=K: bm3_set(p, q2, min(K, n), 3.0)))
            variants.append((f'K{K}_bm1.5', lambda p, q2, n, K=K: bm3_set(p, q2, min(K, n), 1.5)))
        variants.append(('K10_cons_1m_10deg', lambda p, q2, n: consistency_set(p, q2, min(10, n), 1.0, 10.0)))
        variants.append(('K10_cons_0.5m_5deg', lambda p, q2, n: consistency_set(p, q2, min(10, n), 0.5, 5.0)))
        variants.append(('K10_cons_2m_15deg', lambda p, q2, n: consistency_set(p, q2, min(10, n), 2.0, 15.0)))
        for nm, fn in variants:
            T, R, AL, CR = [], [], [], []
            for i in range(len(gt)):
                n = pdb[i].shape[0]
                keep = fn(pdb[i], pq2d[i], n)
                Rt = aggregate(pdb[i], pq2d[i], keep)
                te, re, dv = err(Rt, gt[i])
                T.append(te); R.append(re)
                a = dv @ tdirs[i]; AL.append(abs(a)); CR.append(float(np.linalg.norm(dv - a*tdirs[i])))
            cfgs[nm] = dict(**stats(np.array(T), np.array(R)),
                            along_mean=round(float(np.mean(AL)), 4), along_med=round(float(np.median(AL)), 4),
                            along_p90=round(float(np.percentile(AL, 90)), 4),
                            cross_mean=round(float(np.mean(CR)), 4), cross_med=round(float(np.median(CR)), 4),
                            cross_p90=round(float(np.percentile(CR, 90)), 4))
        # oracle: best single candidate (GT-leaked upper bound of per-pair quality)
        T, R = [], []
        for i in range(len(gt)):
            n = pdb[i].shape[0]
            best = None
            for k in range(n):
                Rt = aggregate(pdb[i], pq2d[i], [k])
                te, re, _ = err(Rt, gt[i])
                if best is None or te < best[0]: best = (te, re)
            T.append(best[0]); R.append(best[1])
        cfgs['ORACLE_best1of10'] = stats(np.array(T), np.array(R))
        # oracle best subset of 3
        T, R = [], []
        for i in range(len(gt)):
            n = min(5, pdb[i].shape[0]); best = None
            for comb in itertools.combinations(range(n), 3):
                Rt = aggregate(pdb[i], pq2d[i], list(comb))
                te, re, _ = err(Rt, gt[i])
                if best is None or te < best[0]: best = (te, re)
            T.append(best[0]); R.append(best[1])
        cfgs['ORACLE_best3of5'] = stats(np.array(T), np.array(R))
        out[key] = cfgs
        print(key, flush=True)
        for nm, s in cfgs.items():
            print('  ', nm, s, flush=True)
    json.dump(out, open('/home/yr/.hermes/cache/scratch/agg_sweep.json', 'w'), indent=1)

if __name__ == '__main__':
    main()
