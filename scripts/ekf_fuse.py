"""P1-1：EKF 离线融合原型（优化路线文档 §3.3）。零 GPU——只吃归档逐帧观测。

状态 (x, y, yaw)，两路独立滤波（ponytail: xy/yaw 解耦标量化；若沿轨误差与时序相关需回
3 状态联合 EKF 时再加，PF gate3 的增益本来就分别体现在 ep 与 eθ 上）：
  平移观测: XFeat snap xy（results/zc_xfeat_rc_<scene>.npz pred_xy，每帧）
  旋转观测: XFeat snap 朝向（pred_h）恒有；Reloc3r sim3 yaw（--reloc3r <npz>，由
            eval_zc_sim3_cache.py 的 pred 输出提供，权重更高 r_h2）
  运动模型: GT 增量 + 噪声（与 Topometric PF / COLD 离线原型同口径，文档 §3.3 明示）
  门控: gate3 判据（观测 vs belief 平移差 >3 m 或 yaw 差 >30° 丢弃该观测）
注意：归档 res_*_sim3.npz 只有 terr/rerr（无逐帧 pred yaw），与文档 §3.3"有逐帧 pred"
的说法不符——已由 eval_zc_sim3_cache.py 重跑补 dump。

用法: python scripts/ekf_fuse.py [--reloc3r key=path ...]  （mv2d，纯 CPU，秒级）
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "/data/code/reLocation-all/AsymLoc/scripts")
FIX = "/data/code/reLocation-all/reloc3r/data"
SCENE = {"lawn_sameday": ("062224", "zc_fix/20260921_062224"),
         "lawn_crossday": ("081533", "zc_fix/20260920_081533"),
         "0923": ("0923", "zc_0923/20260923_060415_real_robot_record_external_lz4")}


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def run_ekf(obs_xy, obs_h, obs_h2, gt_xy, gt_h, q_xy=0.25, r_xy=0.30, q_h=0.01,
            r_h=2.0, r_h2=1.5, gate_m=3.0, gate_d=30.0):
    """离线 EKF 原型：平移 2D 线性 Kalman（观测 XFeat snap xy），旋转独立标量 Kalman
    （观测 = XFeat 朝向 + 可选 Reloc3r yaw obs_h2，低频/有则用）。运动模型 = GT 增量 +
    噪声（文档 §3.3 离线口径）。门控 = gate3 判据：观测-belief 平移差>gate_m 或 yaw 差
    >gate_d 则丢弃该观测。obs_h2=None 时跳过第二路旋转观测。"""
    n = len(obs_xy)
    x = obs_xy[0] if np.isfinite(obs_xy[0]).all() else gt_xy[0].copy()
    P = np.eye(2) * r_xy ** 2
    h = obs_h[0] if np.isfinite(obs_h[0]) else gt_h[0]
    V = np.radians(10.0) ** 2
    fe, fh = [], []
    for i in range(n):
        if i:
            dxy, dh = gt_xy[i] - gt_xy[i - 1], wrap(gt_h[i] - gt_h[i - 1])
            x = x + dxy          # 运动模型输入 = 世界系 GT 增量（离线原型口径；上真 /odom 时换成 body 系旋入）
            h = wrap(h + dh)
            P = P + q_xy ** 2 * np.eye(2)
            V += q_h ** 2
        if np.isfinite(obs_xy[i]).all() and np.linalg.norm(obs_xy[i] - x) < gate_m:
            Kk = P @ np.linalg.inv(P + r_xy ** 2 * np.eye(2))
            x = x + Kk @ (obs_xy[i] - x)
            P = (np.eye(2) - Kk) @ P
        for o, rv in ((obs_h[i], r_h ** 2), (None if obs_h2 is None else obs_h2[i], r_h2 ** 2)):
            if o is not None and np.isfinite(o) and abs(wrap(o - h)) < np.radians(gate_d):
                g = wrap(o - h)
                h = wrap(h + V / (V + rv) * g)
                V = V * rv / (V + rv)
        fe.append(np.linalg.norm(x - gt_xy[i]))
        fh.append(abs(np.degrees(wrap(h - gt_h[i]))))
    return np.array(fe), np.array(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q-xy", type=float, default=0.25)
    ap.add_argument("--r-xy", type=float, default=0.30)
    ap.add_argument("--r-h", type=float, default=2.0)
    ap.add_argument("--gate-m", type=float, default=3.0)
    ap.add_argument("--reloc3r", action="append", default=[], metavar="scene=npz",
                    help="Reloc3r pred 位姿 dump（eval_zc_sim3_cache.py --out 产物）")
    ap.add_argument("--out", default="results/ekf_fuse.json")
    args = ap.parse_args()
    r3map = dict(a.split("=", 1) for a in args.reloc3r)

    res = {}
    for scene, (key, sub) in SCENE.items():
        d = np.load(f"results/zc_xfeat_rc_{scene}.npz")
        names = d["name"].astype(str)
        gt = d["gt"]
        obs_xy = np.array([np.asarray(p, float) for p in d["pred_xy"]])
        obs_h = np.array(d["pred_h"], float)
        obs_h2 = None
        if scene in r3map:
            r3 = np.load(r3map[scene])
            h3 = np.arctan2(r3["preds"][:, 1, 2], r3["preds"][:, 0, 2])
            obs_h2 = np.full(len(obs_h), np.nan)
            m = {n.replace(".color.png", ""): i for i, n in enumerate(r3["names"].astype(str))}
            for i, nm in enumerate(names):
                j = m.get(nm.replace(".color.png", ""))
                if j is not None:
                    obs_h2[i] = h3[j]
        gt_xy = gt[:, :2, 3]
        gt_h = np.arctan2(gt[:, 1, 2], gt[:, 0, 2])
        fe, fh = run_ekf(obs_xy, obs_h, obs_h2, gt_xy, gt_h, q_xy=args.q_xy, r_xy=args.r_xy,
                         r_h=args.r_h, gate_m=args.gate_m)
        single_ep = np.linalg.norm(obs_xy - gt_xy, axis=1)
        single_eh = np.degrees(np.abs(wrap(obs_h - gt_h)))
        ok = np.isfinite(fe)
        res[scene] = {"n": int(ok.sum()),
                      "ekf_ep_mean": round(float(fe[ok].mean()), 3), "ekf_ep_med": round(float(np.median(fe[ok])), 3),
                      "ekf_ep_p90": round(float(np.percentile(fe[ok], 90)), 3),
                      "ekf_eh_mean": round(float(fh[ok].mean()), 2), "ekf_eh_med": round(float(np.median(fh[ok])), 2),
                      "ekf_eh_p90": round(float(np.percentile(fh[ok], 90)), 2),
                      "single_ep_med": round(float(np.median(single_ep[np.isfinite(single_ep)])), 3),
                      "single_ep_p90": round(float(np.percentile(single_ep[np.isfinite(single_ep)], 90)), 3),
                      "single_eh_med": round(float(np.median(single_eh[np.isfinite(single_eh)])), 2),
                      "single_eh_p90": round(float(np.percentile(single_eh[np.isfinite(single_eh)], 90)), 2)}
        print(scene, json.dumps(res[scene]))
    json.dump(res, open(args.out, "w"), indent=1)
    print("->", args.out)


if __name__ == "__main__":
    main()
