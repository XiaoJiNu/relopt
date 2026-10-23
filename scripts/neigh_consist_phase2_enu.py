#!/usr/bin/env python3
"""P4: 轻量邻域一致性 / 短窗平滑，接 Phase2 ENU dualgate+REJ 输出（零 GPU）。

输入: res_*_sim5_dg_zcOnlyENU_odd_rej2.npz（dvec/terr/rerr/names）
pred_xyz = gt_xyz + dvec

变体（均不重跑推理）:
  base          — dg+REJ 原样
  med_w{W}      — pred xy 短窗中位数平滑（W=半窗，全窗=2W+1）
  clamp_w{W}_t{T} — 相对邻域中位数残差 > T m 则回退到邻域中位（不含自身）
  blend_w{W}_a{A} — pred = (1-a)*pred + a*邻域中位（固定混合）

口径: terr = ||pred-gt||；along/cross 同 probe_along_cross_phase2_enu（GT 切向 2D）。
红线: 平滑用预测邻域，不引入新 GT 运动模型（与 EKF 离线 GTΔ 不同）。
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
NPZ_TAG = os.environ.get("NPZ_TAG", "sim5_dg_zcOnlyENU_odd_rej2")
MIN_STEP = 1e-3


def six(x: np.ndarray) -> dict:
    return {
        "mean": round(float(x.mean()), 4),
        "med": round(float(np.median(x)), 4),
        "p90": round(float(np.percentile(x, 90)), 4),
    }


def load_gt_xyz(root: str, bag: str, names: list[str]) -> np.ndarray:
    out = []
    for n in names:
        P = np.loadtxt(f"{root}/{bag}/seq-01/{n}.pose.txt")
        out.append(P[:3, 3])
    return np.asarray(out, float)


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


def neighbor_median_xy(pred_xy: np.ndarray, i: int, W: int, exclude_self: bool) -> np.ndarray:
    lo, hi = max(0, i - W), min(len(pred_xy), i + W + 1)
    sl = list(range(lo, hi))
    if exclude_self and i in sl and len(sl) > 1:
        sl.remove(i)
    return np.median(pred_xy[sl], axis=0)


def apply_med(pred_xyz: np.ndarray, W: int) -> np.ndarray:
    out = pred_xyz.copy()
    for i in range(len(pred_xyz)):
        out[i, :2] = neighbor_median_xy(pred_xyz[:, :2], i, W, exclude_self=False)
    return out


def apply_clamp(pred_xyz: np.ndarray, W: int, thr: float) -> np.ndarray:
    out = pred_xyz.copy()
    for i in range(len(pred_xyz)):
        med = neighbor_median_xy(pred_xyz[:, :2], i, W, exclude_self=True)
        if np.linalg.norm(pred_xyz[i, :2] - med) > thr:
            out[i, :2] = med
    return out


def apply_blend(pred_xyz: np.ndarray, W: int, a: float) -> np.ndarray:
    out = pred_xyz.copy()
    for i in range(len(pred_xyz)):
        med = neighbor_median_xy(pred_xyz[:, :2], i, W, exclude_self=True)
        out[i, :2] = (1.0 - a) * pred_xyz[i, :2] + a * med
    return out


def metrics(pred_xyz: np.ndarray, gt_xyz: np.ndarray) -> dict:
    err = pred_xyz - gt_xyz
    terr = np.linalg.norm(err, axis=1)
    al, cr = along_cross(err, gt_xyz[:, :2])
    return {
        "terr": six(terr),
        "along": six(al),
        "cross": six(cr),
        "n_gt0p5": int((terr > 0.5).sum()),
        "n_gt0p3": int((terr > 0.3).sum()),
        "terr_max": round(float(terr.max()), 4),
    }


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(os.environ.get("OUT_DIR", "/home/yr/grok-bot-work/logs_zc_opt_p47_20261023"))
    out_dir.mkdir(parents=True, exist_ok=True)
    scenes = os.environ.get("SCENES", "062224,081533,0923").split(",")

    # variants: (name, fn)
    variants = [("base", lambda p: p)]
    for W in (2, 3, 5, 7):
        variants.append((f"med_w{W}", lambda p, W=W: apply_med(p, W)))
    for W, T in ((3, 0.25), (3, 0.40), (5, 0.25), (5, 0.40), (5, 0.60)):
        variants.append((f"clamp_w{W}_t{T:.2f}", lambda p, W=W, T=T: apply_clamp(p, W, T)))
    for W, A in ((3, 0.3), (5, 0.3), (5, 0.5)):
        variants.append((f"blend_w{W}_a{A:.1f}", lambda p, W=W, A=A: apply_blend(p, W, A)))

    report = {
        "stamp": stamp,
        "npz_tag": NPZ_TAG,
        "note": "neighborhood consistency on dg+REJ pred; no GT motion model in filter",
        "scenes": {},
    }
    lines = [f"P4 neigh_consist stamp={stamp} npz={NPZ_TAG}"]

    for key in scenes:
        root, bag = SC[key]
        d = np.load(f"{SIM3}/res_{key}_{NPZ_TAG}.npz")
        names = [str(n) for n in d["names"]]
        dvec = d["dvec"].astype(float)
        gt = load_gt_xyz(root, bag, names)
        pred0 = gt + dvec
        # sanity: terr from npz vs recomputed
        terr0 = np.linalg.norm(dvec, axis=1)
        assert np.allclose(terr0, d["terr"], atol=1e-5), key

        scene_out = {}
        base_m = None
        for name, fn in variants:
            pred = fn(pred0)
            m = metrics(pred, gt)
            scene_out[name] = m
            if name == "base":
                base_m = m
            dt = m["terr"]["mean"] - base_m["terr"]["mean"]
            lines.append(
                f"{key:6s} {name:18s} terr {m['terr']['mean']:.4f}/{m['terr']['med']:.4f}/{m['terr']['p90']:.4f}"
                f"  cross {m['cross']['mean']:.4f}/{m['cross']['med']:.4f}/{m['cross']['p90']:.4f}"
                f"  >0.5={m['n_gt0p5']}  dmean={dt:+.4f}"
            )
        report["scenes"][key] = scene_out

        # case study: disaster frames under best clamp
        bad = np.where(terr0 > 0.5)[0]
        if len(bad):
            clamp = apply_clamp(pred0, 5, 0.40)
            lines.append(f"  case>0.5m base→clamp_w5_t0.40:")
            for i in bad:
                tb = float(np.linalg.norm(pred0[i] - gt[i]))
                tc = float(np.linalg.norm(clamp[i] - gt[i]))
                lines.append(f"    {names[i]} terr {tb:.3f}→{tc:.3f}")

    # pick best by 081533 terr mean among non-base
    if "081533" in report["scenes"]:
        sc = report["scenes"]["081533"]
        ranked = sorted(
            ((k, v["terr"]["mean"], v["cross"]["p90"]) for k, v in sc.items() if k != "base"),
            key=lambda x: (x[1], x[2]),
        )
        report["best_on_081533_by_terr_mean"] = [
            {"name": n, "terr_mean": tm, "cross_p90": cp} for n, tm, cp in ranked[:5]
        ]
        lines.append("best@081533: " + ", ".join(f"{n}({tm:.4f})" for n, tm, cp in ranked[:5]))

    out_json = out_dir / f"p4_neigh_consist_{stamp}.json"
    out_log = out_dir / f"p4_neigh_consist_{stamp}.log"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    out_log.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
