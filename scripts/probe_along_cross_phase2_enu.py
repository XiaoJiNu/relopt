#!/usr/bin/env python3
"""Phase2 ZC-only ENU: along-track / cross-track from existing eval_zc_sim3 npz.

口径（复用验收 / A2，与 terr 同套样本）:
  - e = dvec = pred_xyz - gt_xyz（eval_zc_sim3 已在 POSE_TAG=enu 下，绝对位姿误差，单位 m）
  - 前进方向 u = GT 查询轨迹水平切向（相邻帧 cam xy 差分单位向量；与
    eval_rtk_rescore / gen_reloc3r_finetune_report.a2_split / alongcross.py 一致）
  - along = |dot(e_xy, u)|
  - cross = ||e_xy - signed_along * u||   （仅水平面 2D，不含竖直；同 rescore）
  - 另附 cross3d = ||e - signed_along_3 * u3|| 供对照（u3=[ux,uy,0]）

不重跑推理；读 Phase2 已存 res_*_sim5_bm3_zcOnlyENU_odd.npz。
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
NPZ_TAG = "sim5_bm3_zcOnlyENU_odd"
MIN_STEP = 1e-3  # m；过短切向跳过


def pose_xy(root: str, bag: str, name: str) -> np.ndarray:
    P = np.loadtxt(f"{root}/{bag}/seq-01/{name}.pose.txt")
    return P[:2, 3].astype(float)


def heading_unit(root: str, bag: str, name: str, all_names: list[str]) -> np.ndarray | None:
    i = all_names.index(name)
    j = min(i + 1, len(all_names) - 1)
    k = max(i - 1, 0)
    if j == k:
        return None
    v = pose_xy(root, bag, all_names[j]) - pose_xy(root, bag, all_names[k])
    n = float(np.linalg.norm(v))
    if n < MIN_STEP:
        return None
    return v / n


def six(x: np.ndarray) -> str:
    return f"{x.mean():.4f}/{np.median(x):.4f}/{np.percentile(x, 90):.4f}"


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(os.environ.get("OUT_DIR", "/home/yr/grok-bot-work/logs_zc_enu"))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"phase2_along_cross_{stamp}.log"
    json_path = out_dir / f"phase2_along_cross_{stamp}.json"

    lines: list[str] = []
    def log(s: str = "") -> None:
        print(s)
        lines.append(s)

    log(f"stamp={stamp}")
    log("definition=GT_path_tangent_2D_on_dvec (reuse A2/rescore; same samples as terr)")
    log("POSE_TAG=enu; dvec already ENU absolute error from eval_zc_sim3 motion-averaging")
    log(f"npz_tag={NPZ_TAG}")
    log("")

    results: dict = {
        "stamp": stamp,
        "definition": {
            "error": "dvec = pred[:3,3]-gt[:3,3] from eval_zc_sim3 (ENU absolute, post motion-averaging)",
            "along_axis": "GT query trajectory horizontal tangent (neighbor-frame xy delta unit)",
            "along": "|dot(e_xy, u)|",
            "cross": "||e_xy - signed_along*u|| (2D horizontal)",
            "cross3d_note": "||e - signed_along*u3|| with u3=[ux,uy,0] (includes vertical)",
            "aligned_with_terr": True,
            "not_relative_t_gt": (
                "Not using pair-wise relative t_gt; acceptance口径 is absolute ep along/cross "
                "vs GT path heading (eval_rtk_rescore / a2_split)."
            ),
        },
        "scenes": {},
    }

    for key, (root, bag) in SC.items():
        npz_p = f"{SIM3}/res_{key}_{NPZ_TAG}.npz"
        arr = np.load(npz_p)
        assert "dvec" in arr.files and "terr" in arr.files
        names = [str(x) for x in arr["names"]]
        dvec = arr["dvec"].astype(float)
        terr = arr["terr"].astype(float)
        all_names = sorted(
            n[:-10]
            for n in os.listdir(f"{root}/{bag}/seq-01")
            if n.endswith(".color.png")
        )

        along, cross, cross3d = [], [], []
        skipped = 0
        for dv, nm, te in zip(dvec, names, terr):
            u = heading_unit(root, bag, nm, all_names)
            if u is None:
                skipped += 1
                continue
            e_xy = dv[:2]
            signed = float(e_xy @ u)
            along.append(abs(signed))
            resid = e_xy - signed * u
            cross.append(float(np.linalg.norm(resid)))
            u3 = np.array([u[0], u[1], 0.0])
            s3 = float(dv @ u3)
            cross3d.append(float(np.linalg.norm(dv - s3 * u3)))

        al = np.asarray(along)
        cr = np.asarray(cross)
        c3 = np.asarray(cross3d)
        # sanity: for 2D, hypot(along, cross) should ≈ ||e_xy|| ≤ terr
        e_xy_norm = np.linalg.norm(dvec[:, :2], axis=1)
        # only on kept frames — rebuild mask
        kept_idx = []
        for i, nm in enumerate(names):
            if heading_unit(root, bag, nm, all_names) is not None:
                kept_idx.append(i)
        kept_idx = np.asarray(kept_idx)
        hyp = np.hypot(al, cr)
        xy_kept = e_xy_norm[kept_idx]
        hyp_ok = float(np.max(np.abs(hyp - xy_kept))) < 1e-9

        scene = {
            "n": int(len(terr)),
            "n_along": int(len(al)),
            "skipped_no_heading": int(skipped),
            "terr_mean_med_p90_m": [float(terr.mean()), float(np.median(terr)), float(np.percentile(terr, 90))],
            "along_mean_med_p90_m": [float(al.mean()), float(np.median(al)), float(np.percentile(al, 90))],
            "cross_mean_med_p90_m": [float(cr.mean()), float(np.median(cr)), float(np.percentile(cr, 90))],
            "cross3d_mean_med_p90_m": [float(c3.mean()), float(np.median(c3)), float(np.percentile(c3, 90))],
            "hypot_along_cross_matches_exy": hyp_ok,
            "npz": npz_p,
            "data_root": root,
            "query_bag": bag,
        }
        results["scenes"][key] = scene

        log(f"==== {key} ====")
        log(f"npz={npz_p}")
        log(f"n={scene['n']} n_along={scene['n_along']} skipped={skipped} hyp_ok={hyp_ok}")
        log(f"terr  mean/med/p90 = {six(terr)} m")
        log(f"along mean/med/p90 = {six(al)} m")
        log(f"cross mean/med/p90 = {six(cr)} m  (2D horizontal)")
        log(f"cross3d mean/med/p90 = {six(c3)} m  (incl. vertical)")
        # acceptance对照（客观）
        log(
            f"vs target along<0.10 / cross<0.03 m: "
            f"along_med={np.median(al):.4f} ({'OK' if np.median(al)<0.10 else 'NOT'}) "
            f"cross_med={np.median(cr):.4f} ({'OK' if np.median(cr)<0.03 else 'NOT'}) "
            f"along_p90={np.percentile(al,90):.4f} cross_p90={np.percentile(cr,90):.4f}"
        )
        log("")

    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    log_path.write_text("\n".join(lines) + "\n")
    log(f"JSON={json_path}")
    log(f"LOG={log_path}")


if __name__ == "__main__":
    main()
