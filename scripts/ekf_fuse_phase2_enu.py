#!/usr/bin/env python3
"""P2: 时序 EKF 接 Phase2 ENU 观测（离线运动模型 = GT 增量）。

从 eval_zc_sim3 的 npz 还原 pred = gt + dvec，再跑与 ekf_fuse.py 同构的
2D 平移 Kalman + gate；输出滤波前/后 terr 与 along/cross。

口径红线：运动模型依赖离线 GT 增量（非真 /odom）；报告须标明。
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

FIX = "/data/code/reLocation-all/reloc3r/data"
SIM3 = f"{FIX}/zc_sim3"
SC = {
    "062224": (f"{FIX}/zc_fix_enu", "20260921_062224"),
    "081533": (f"{FIX}/zc_fix_enu", "20260920_081533"),
    "0923": (
        f"{FIX}/zc_0923_enu",
        "20260923_060415_real_robot_record_external_lz4",
    ),
}
MIN_STEP = 1e-3


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def six(x: np.ndarray) -> dict:
    return {
        "mean": round(float(x.mean()), 4),
        "med": round(float(np.median(x)), 4),
        "p90": round(float(np.percentile(x, 90)), 4),
    }


def along_cross(err_xyz: np.ndarray, gt_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tang = np.gradient(gt_xy, axis=0)
    nrm = np.linalg.norm(tang, axis=1, keepdims=True)
    nrm = np.maximum(nrm, MIN_STEP)
    u = tang / nrm
    e_xy = err_xyz[:, :2]
    along = np.abs((e_xy * u).sum(1))
    resid = e_xy - ((e_xy * u).sum(1, keepdims=True) * u)
    cross = np.linalg.norm(resid, axis=1)
    return along, cross


def run_ekf_xy(obs_xy, gt_xy, q_xy=0.25, r_xy=0.30, gate_m=3.0):
    """离线 EKF：运动 = GTΔ，观测 = Reloc3r ENU xy，gate3 平移门。"""
    n = len(obs_xy)
    x = obs_xy[0].copy() if np.isfinite(obs_xy[0]).all() else gt_xy[0].copy()
    P = np.eye(2) * r_xy ** 2
    fe = []
    filt_xy = []
    for i in range(n):
        if i:
            dxy = gt_xy[i] - gt_xy[i - 1]
            x = x + dxy
            P = P + q_xy ** 2 * np.eye(2)
        if np.isfinite(obs_xy[i]).all() and np.linalg.norm(obs_xy[i] - x) < gate_m:
            Kk = P @ np.linalg.inv(P + r_xy ** 2 * np.eye(2))
            x = x + Kk @ (obs_xy[i] - x)
            P = (np.eye(2) - Kk) @ P
        filt_xy.append(x.copy())
        fe.append(float(np.linalg.norm(x - gt_xy[i])))
    return np.array(filt_xy), np.array(fe)


def process(key: str, npz_tag: str, q_xy: float, r_xy: float, gate_m: float) -> dict:
    root, bag = SC[key]
    npz_p = f"{SIM3}/res_{key}_{npz_tag}.npz"
    d = np.load(npz_p)
    names = [str(n) for n in d["names"]]
    dvec = np.asarray(d["dvec"], float)
    terr0 = np.asarray(d["terr"], float)
    rerr0 = np.asarray(d["rerr"], float) if "rerr" in d.files else None

    gt_xyz = []
    for n in names:
        P = np.loadtxt(f"{root}/{bag}/seq-01/{n}.pose.txt")
        gt_xyz.append(P[:3, 3])
    gt_xyz = np.asarray(gt_xyz, float)
    gt_xy = gt_xyz[:, :2]
    pred_xy = gt_xy + dvec[:, :2]

    filt_xy, fe = run_ekf_xy(pred_xy, gt_xy, q_xy=q_xy, r_xy=r_xy, gate_m=gate_m)
    # 竖直分量保持观测 dvec z（滤波只动水平）
    err_f = np.column_stack([filt_xy - gt_xy, dvec[:, 2]])
    terr_f = np.linalg.norm(err_f, axis=1)

    along0, cross0 = along_cross(dvec, gt_xy)
    along_f, cross_f = along_cross(err_f, gt_xy)

    out = {
        "key": key,
        "npz": npz_p,
        "n": len(names),
        "motion_model": "offline_GT_delta",
        "q_xy": q_xy,
        "r_xy": r_xy,
        "gate_m": gate_m,
        "before": {
            "terr": six(terr0),
            "along": six(along0),
            "cross": six(cross0),
            "n_gt0p5": int((terr0 > 0.5).sum()),
        },
        "after_ekf": {
            "terr": six(terr_f),
            "along": six(along_f),
            "cross": six(cross_f),
            "n_gt0p5": int((terr_f > 0.5).sum()),
        },
    }
    if rerr0 is not None:
        out["before"]["rerr"] = six(rerr0)
    return out


def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(os.environ.get("OUT_DIR", "/home/yr/grok-bot-work/logs_zc_opt_20261023"))
    out_dir.mkdir(parents=True, exist_ok=True)
    tags = os.environ.get(
        "NPZ_TAGS",
        "sim5_bm3_zcOnlyENU_odd,sim5_dg_zcOnlyENU_odd_rej2",
    ).split(",")
    q_xy = float(os.environ.get("EKF_Q_XY", "0.25"))
    r_xy = float(os.environ.get("EKF_R_XY", "0.30"))
    gate_m = float(os.environ.get("EKF_GATE_M", "3.0"))
    keys = os.environ.get("KEYS", "062224,081533,0923").split(",")

    results = {"stamp": stamp, "note": "offline GT-delta motion; ENU GT; pred=gt+dvec"}
    for tag in tags:
        tag = tag.strip()
        if not tag:
            continue
        results[tag] = {}
        for key in keys:
            key = key.strip()
            try:
                results[tag][key] = process(key, tag, q_xy, r_xy, gate_m)
                b = results[tag][key]["before"]["terr"]
                a = results[tag][key]["after_ekf"]["terr"]
                print(
                    f"{tag} {key}: terr before {b['mean']}/{b['med']}/{b['p90']} "
                    f"-> after {a['mean']}/{a['med']}/{a['p90']}"
                )
            except Exception as e:
                results[tag][key] = {"error": str(e)}
                print(f"{tag} {key}: ERROR {e}")

    jp = out_dir / f"p2_ekf_phase2_enu_{stamp}.json"
    jp.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("->", jp)


if __name__ == "__main__":
    main()
