"""M1 灾难尾部抑制离线扫（优化方案 §4 M1 / §6 第2步）：top-1 锚 + 中位数锚双判据。

零 GPU，吃 dump_cands_ft.py 的 per-candidate K=10 dump。变体：
  bm3            现行（候选推算位置中位数当锚, r=3）
  a1             top-1 锚（r 同）
  dual           两锚一致(距离<=agree)用 top-1 锚，否则回退中位数锚（M1 方法1 双判据）
  dual_gate      dual 基础上，回退帧再用时序门：top-1 锚结果与上一帧输出差>gate_m 才真回退
输出: 每变体 ep mean/med/p90 + >0.5m 计数 + 最差3帧 + 沿/跨轨三分量（验收表口径），写 agg_sweep_m1.json
用法: mv2d python scripts/gt_probes/agg_sweep_m1.py [--K 5 --r 3.0 --agree 1.0]
"""
import argparse
import json

import numpy as np
from scipy.spatial.transform import Rotation as Rot


def rot_avg(Rs):
    q = Rot.from_matrix(Rs).as_quat()
    q = q * np.sign(q @ q[0] + 1e-9)[:, None]
    qm = q.mean(0); qm /= np.linalg.norm(qm)
    return Rot.from_quat(qm).as_matrix()


def tri(lines):
    A = np.zeros((3, 3)); b = np.zeros(3)
    for p0, p1 in lines:
        d = p1 - p0; d /= (np.linalg.norm(d) + 1e-12)
        M = np.eye(3) - np.outer(d, d); A += M; b += M @ p0
    return np.linalg.solve(A + 1e-9 * np.eye(3), b)


def agg(pdb, pq2d, keep):
    qR, lines = [], []
    for i in keep:
        pq = pdb[i] @ pq2d[i]
        qR.append(pq[:3, :3]); lines.append((pdb[i][:3, 3], pq[:4, 3][:3]))
    Rt = np.eye(4); Rt[:3, :3] = rot_avg(np.array(qR)); Rt[:3, 3] = tri(lines)
    return Rt


def sweep_frame(pdb, pq2d, n, K, r, agree):
    """-> (top-1锚位姿, 中位数锚位姿, 两锚一致?)"""
    imp = np.array([(pdb[k] @ pq2d[k])[:3, 3] for k in range(n)])
    med = np.median(imp[:min(K, n)], 0)
    sel_a1 = [k for k in range(K) if np.linalg.norm(imp[k] - imp[0]) < r] or [0]
    Rt_a1 = agg(pdb, pq2d, sel_a1)
    keep_m = [k for k in range(K) if np.linalg.norm(pdb[k][:3, 3] - med) < r]
    if len(keep_m) < 2:
        keep_m = list(range(K))  # 与生产 bm3 (eval_zc_sim3.py) 同回退规则
    return Rt_a1, agg(pdb, pq2d, keep_m), bool(np.linalg.norm(imp[0] - med) <= agree)


def stats(T, V, gt):
    tang = np.gradient(gt[:, :2, 3], axis=0)
    tang /= (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9)
    along = np.abs((V[:, :2] * tang).sum(1))
    cross = np.abs((V[:, :2] * np.stack([-tang[:, 1], tang[:, 0]], 1)).sum(1))
    return dict(n=len(T), ep_mean=round(float(T.mean()), 4), ep_med=round(float(np.median(T)), 4),
                ep_p90=round(float(np.percentile(T, 90)), 4), n_gt0p5=int((T > 0.5).sum()),
                top3=[round(float(x), 2) for x in np.sort(T)[::-1][:3]],
                along_cm=[round(float(x * 100), 1) for x in (along.mean(), np.median(along), np.percentile(along, 90))],
                cross_cm=[round(float(x * 100), 1) for x in (cross.mean(), np.median(cross), np.percentile(cross, 90))])


def run(dfile, K, r, agree, gate_m):
    d = np.load(dfile)
    pdb, pq2d, gt = d["pdb"], d["pq2d"], d["gt"]
    dgt = np.gradient(gt[:, :2, 3], axis=0)  # 离线口径的运动增量（EKF 同款 GT-delta 运动模型）
    out = {}
    for variant in ("bm3", "a1", "dual", "dual_gate"):
        T, V, prev_pos = [], [], None
        for i in range(len(gt)):
            Rt_a1, Rt_m, ok = sweep_frame(pdb[i], pq2d[i], pdb[i].shape[0], K, r, agree)
            pred = None if prev_pos is None else prev_pos[:2] + dgt[i]
            if variant == "bm3":
                Rt = Rt_m
            elif variant == "a1":
                Rt = Rt_a1
            elif variant == "dual":
                Rt = Rt_a1 if ok else Rt_m
            else:  # dual_gate: 锚不一致时，top-1 锚结果若与运动预测连续(<=gate_m)仍用 top-1，否则回退中位数锚
                Rt = Rt_m
                if ok or (pred is not None and np.linalg.norm(Rt_a1[:2, 3] - pred) <= gate_m):
                    Rt = Rt_a1
            prev_pos = Rt[:3, 3]
            dv = Rt[:3, 3] - gt[i][:3, 3]
            T.append(float(np.linalg.norm(dv))); V.append(dv)
        out[variant] = stats(np.array(T), np.array(V), gt)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", nargs="*", default=[
        f"/data/code/reLocation-all/reloc3r/data/zc_sim3/cands_{k}_K10_ft_odd.npz"
        for k in ("062224", "081533", "0923")])
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--r", type=float, default=3.0)
    ap.add_argument("--agree", type=float, default=1.0)
    ap.add_argument("--gate-m", type=float, default=3.0)
    ap.add_argument("--out", default="/data/code/reLocation-all/relopt/scripts/gt_probes/agg_sweep_m1.json")
    a = ap.parse_args()
    res = {}
    for f in a.dump:
        key = f.split("cands_")[1][:6]
        res[key] = run(f, a.K, a.r, a.agree, a.gate_m)
        print(key)
        for k, v in res[key].items():
            print("  ", k, v)
    json.dump(res, open(a.out, "w"), indent=1)
