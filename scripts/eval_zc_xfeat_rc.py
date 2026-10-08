"""方案 B · P0-1：XFeat 旋转修复 + 逐帧 dump（优化路线文档 §3.1a/b、§4 P0-1/P1-1）。

一次前向同时算两套口径，保证可比：
  base  = 现行归档口径（query 朝向 = db 帧朝向，snap top-1 / top-3 圆均值）
  rc    = 修复口径：query 朝向 = db 朝向 ⊕ recoverPose 的 R（上游 geom() 早已返回 R，
          eval_zc_xfeat.py:117 把它丢弃）；旋转另给 top-3 内点加权圆均值（circ3）。
验收门槛（§4）：eθ p90 三场景 7.95° 显著下降（<5°）；ep 六列不劣化（±0.02 m 内）——
base 列必须与归档 outputs/zc_xfeat_*.json 对齐（同口径回归），rc 列只许旋转项变好。

用法（mv2d）：
  python scripts/eval_zc_xfeat_rc.py --scene lawn_sameday --dump --out results/zc_xfeat_rc_lawn_sameday.json
"""
import argparse
import glob
import json
import os
import sys
import time

ROOT = "/data/code/reLocation-all"
for p in (f"{ROOT}/accelerated_features", f"{ROOT}/accelerated_features/scripts", f"{ROOT}/AsymLoc/scripts"):
    sys.path.insert(0, p)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from eval_zc_netvlad import SCENES, pose_path, ls_solve, med_solve, heading, circmean  # noqa: E402
from modules.xfeat import XFeat  # noqa: E402


def geom_all(pts_q, pts_d):
    """eval_zc_xfeat.geom 的守卫延后版：返回 (R, t̂, ninl, ang)，不在 25° 处丢弃——
    守卫在聚合处按 --guard-deg 施加，逐候选 dump 支持离线守卫扫（§3.1c）。"""
    F, inl = cv2.findFundamentalMat(pts_q, pts_d, cv2.FM_RANSAC, 1.0, 0.999)
    if F is None or inl is None or inl.sum() < 6:
        return None
    from eval_zc_netvlad import K  # 同一内参口径
    ok, R, t, _ = cv2.recoverPose(K.T @ F @ K, pts_q[inl.ravel() == 1], pts_d[inl.ravel() == 1], K)
    if not ok or ok < 6:
        return None
    ang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    return R, t.ravel() / np.linalg.norm(t), int(inl.sum()), ang

AGGS = ("snap", "ls", "med")          # 位置聚合口径，与归档一致
VARIANTS = ("base", "rc", "circ3")    # 朝向口径：snap base/rc；circ3 = rc 的 top-3 内点圆均值


def yaw_err(h, h_gt):
    return float(np.degrees(abs((h - h_gt + np.pi) % (2 * np.pi) - np.pi)))


def rel_yaw(A_db, R, fallback):
    """query 朝向 = db 朝向 ⊕ R（recoverPose: R = A_db^T A_q ⇒ A_q = A_db @ R）。"""
    A_q = A_db @ R
    return np.arctan2(A_q[1, 2], A_q[0, 2])


def stats(arr):
    a = np.asarray(arr, dtype=float)
    finite = np.isfinite(a[:, 0]) & np.isfinite(a[:, 1])
    d = {"n": int(a.shape[0]), "n_ok": int(finite.sum())}
    if finite.any():
        ep, eth = a[finite, 0], a[finite, 1]
        d.update(ep_mean=round(float(ep.mean()), 3), ep_med=round(float(np.median(ep)), 3),
                 ep_p90=round(float(np.percentile(ep, 90)), 3),
                 eth_mean=round(float(eth.mean()), 2), eth_med=round(float(np.median(eth)), 2),
                 eth_p90=round(float(np.percentile(eth, 90)), 2))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, choices=sorted(SCENES))
    ap.add_argument("--matcher", default="xfeat", choices=["xfeat", "xfeat-star"])
    ap.add_argument("--top-k", type=int, default=0, help="0=按 matcher 默认 (4096/10000)，§3.1d sweep 用")
    ap.add_argument("--guard-deg", type=float, default=25.0, help="§3.1c 守卫角 sweep")
    ap.add_argument("--max-query", type=int, default=0)
    ap.add_argument("--dump", action="store_true", help="逐帧 npz（P1-1 EKF 输入）")
    ap.add_argument("--dump-cands", action="store_true", help="逐候选 npz（§3.1c 守卫角/阈值离线扫，零重推理）")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    xfeat = XFeat(top_k=args.top_k or (10_000 if args.matcher == "xfeat-star" else 4096))
    db_root, q_root, pair_file = SCENES[args.scene]
    pairs = np.load(pair_file, allow_pickle=True)
    print(f"{args.scene}: {len(pairs)} queries x top-10 ({args.matcher}, top_k={xfeat.top_k}, guard={args.guard_deg})", flush=True)

    img_cache, sp_cache = {}, {}

    def resolve(path):
        p = path.replace("./data/", f"{ROOT}/reloc3r/data/")
        if not os.path.exists(p):
            key = path.split("seq-01/")[-1]
            cand = (glob.glob(f"{db_root}/seq-01/{key}") or glob.glob(f"{q_root}/seq-01/{key}"))
            p = cand[0] if cand else None
        return p

    def img(p):
        if p not in img_cache:
            img_cache[p] = cv2.imread(p)
        return img_cache[p]

    def sparse(p):
        if p not in sp_cache:
            out = xfeat.detectAndCompute(img(p), top_k=xfeat.top_k)[0]
            sp_cache[p] = (out["keypoints"].cpu().numpy(), out["descriptors"])
        return sp_cache[p]

    acc = {(a, v): [] for a in AGGS for v in VARIANTS}
    dump = {"name": [], "gt": [], "ninl": [], "ncand": [], "solve": [],
            "pred_xy": [], "pred_h": [], "pred_h_rc": []}
    cdump = {k: [] for k in ("qi", "c", "d", "ninl", "ang", "h_db", "h_rc")}
    for k in acc:
        dump[f"ep_{'_'.join(k)}"] = []
        dump[f"eth_{'_'.join(k)}"] = []
    n_query, t_ms = 0, []

    for qi in range(len(pairs)):
        if args.max_query and qi >= args.max_query:
            break
        tq = time.time()
        row = pairs[qi]
        qp = resolve(row[0]["query_path"])
        Tgt = np.loadtxt(pose_path(qp)) if qp and os.path.exists(pose_path(qp)) else None
        if qp is None or Tgt is None:
            continue
        n_query += 1
        dump["name"].append(os.path.basename(qp))
        dump["gt"].append(Tgt)
        h_gt = heading(Tgt[:3, :3])
        cands = []
        for c in row:
            dp = resolve(c["db_path"])
            Tdb = np.loadtxt(pose_path(dp)) if dp and os.path.exists(pose_path(dp)) else None
            if dp is None or Tdb is None:
                continue
            if args.matcher == "xfeat":
                kq, dq = sparse(qp)
                kd, dd = sparse(dp)
                idx0, idx1 = xfeat.match(dq, dd, min_cossim=-1)
                pts_q, pts_d = kq[idx0.cpu().numpy()], kd[idx1.cpu().numpy()]
            else:
                pts_q, pts_d = xfeat.match_xfeat_star(img(qp), img(dp))
            if len(pts_q) < 60:
                continue
            tv = geom_all(pts_q, pts_d)
            if tv is None:
                continue
            R, t, ninl, ang = tv
            A_db = Tdb[:3, :3]
            h_db = heading(A_db)
            h_rc = rel_yaw(A_db, R, h_db)
            if args.dump_cands:  # dump 不施守卫：守卫角 sweep 离线做（§3.1c）
                cdump["qi"].append(n_query - 1)
                cdump["c"].append(Tdb[:3, 3])
                cdump["d"].append(A_db @ t)
                cdump["ninl"].append(ninl)
                cdump["ang"].append(ang)
                cdump["h_db"].append(h_db)
                cdump["h_rc"].append(h_rc)
            if ang > args.guard_deg:   # 聚合级守卫 = 上游 geom() 的 25°（base 与归档一致的前提）
                continue
            cands.append(dict(c=Tdb[:3, 3], d=A_db @ t, ninl=ninl, h_db=h_db, h_rc=h_rc, ang=ang))
        t_ms.append((time.time() - tq) * 1000)
        dump["ncand"].append(len(cands))
        if not cands:
            dump["solve"].append(0)
            dump["ninl"].append(0)
            dump["pred_xy"].append([np.nan, np.nan])
            dump["pred_h"].append(np.nan)
            dump["pred_h_rc"].append(np.nan)
            for k in acc:
                acc[k].append((np.inf, np.inf))
                dump[f"ep_{'_'.join(k)}"].append(np.inf)
                dump[f"eth_{'_'.join(k)}"].append(np.inf)
            continue
        dump["solve"].append(1)
        best = max(cands, key=lambda x: x["ninl"])
        dump["ninl"].append(best["ninl"])
        top = sorted(cands, key=lambda x: -x["ninl"])[:3]
        ey = {}
        ey[("snap", "base")] = yaw_err(best["h_db"], h_gt)
        ey[("snap", "rc")] = yaw_err(best["h_rc"], h_gt)
        ey[("snap", "circ3")] = yaw_err(circmean([c["h_rc"] for c in top], [c["ninl"] for c in top]), h_gt)
        ey[("ls", "base")] = ey[("med", "base")] = yaw_err(circmean([c["h_db"] for c in top], [c["ninl"] for c in top]), h_gt)
        ey[("ls", "rc")] = ey[("med", "rc")] = ey[("ls", "circ3")] = ey[("med", "circ3")] = ey[("snap", "circ3")]
        pos = {}
        for mode, solver in (("ls", ls_solve), ("med", med_solve)):
            xy = solver(cands) if len(cands) >= 2 else None
            pos[mode] = float(np.linalg.norm(xy - Tgt[:3, 3][:2])) if xy is not None else np.inf
        pos["snap"] = float(np.linalg.norm(best["c"][:2] - Tgt[:3, 3][:2]))
        dump["pred_xy"].append(best["c"][:2])          # EKF 平移观测（snap）
        dump["pred_h"].append(best["h_db"])            # EKF 旋转观测备选（base）
        dump["pred_h_rc"].append(best["h_rc"])         # EKF 旋转观测备选（rc）
        for a in AGGS:
            for v in VARIANTS:
                e = ey[(a, v)]
                acc[(a, v)].append((pos[a], e))
                dump[f"ep_{a}_{v}"].append(pos[a])
                dump[f"eth_{a}_{v}"].append(e)
        dump["ninl"].append(best["ninl"])
        if qi % 200 == 0:
            print(f"{qi}/{n_query} cands={len(cands)}", flush=True)

    out = {"scene": args.scene, "matcher": args.matcher, "top_k": xfeat.top_k,
           "guard_deg": args.guard_deg, "n_query": n_query,
           "solve_rate": round(sum(dump["solve"]) / max(n_query, 1), 3),
           "query_ms_mean": round(float(np.mean(t_ms)), 1) if t_ms else 0}
    for k, arr in acc.items():
        out[f"{k[0]}_{k[1]}"] = stats(arr)
    print(json.dumps(out, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
    if args.dump:
        d = {k: (np.array(v, dtype=float) if k == "gt" else np.array(v)) for k, v in dump.items()
             if k not in ("name", "gt")}
        d["name"] = np.array(dump["name"])
        d["gt"] = np.array(dump["gt"])
        p = (args.out or f"results/zc_xfeat_rc_{args.scene}.json").replace(".json", ".npz")
        np.savez_compressed(p, **d)
        print("dumped ->", p)
    if args.dump_cands:
        c = {k: np.array(v, dtype=float) for k, v in cdump.items()}
        c["gt"] = np.array(dump["gt"])
        np.savez_compressed((args.out or f"results/zc_xfeat_rc_{args.scene}.json").replace(".json", "_cands.npz"), **c)
        print(f"dumped cands -> {c['qi'].shape[0]} 候选")


if __name__ == "__main__":
    main()
