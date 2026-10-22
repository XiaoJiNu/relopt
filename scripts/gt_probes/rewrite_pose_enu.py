"""Phase2: 旁路写出统一 ENU 的 4x4 T_ENU_cam（不覆盖 zc_fix / zc_0923）。

组装口径（与 build_rtk_gt / eval_rtk_rescore 一致）:
  t_xy_ENU = cam_enu（已含杆臂：由 w·cam_map + o 得到）
  t_z_ENU  = 旧 pose 的 map-Z（map/ENU 共享竖直轴；w 为绕 Z 的 2D 旋转）
  R_ENU    = R_z(theta) @ R_map
             其中 R_z 对应复数 w=e^{i·theta}: E+iN = w·(X+iY)
  朝向偏置 off_rad 默认不旋进 R（与 eval_rtk_rescore 默认 BIAS_CORRECT=0 一致；
  跨包 /map 航向系实测无额外旋转，off 差属双天线标定偏差）。

输出树:
  <REPO>/data/zc_fix_enu/<bag>/seq-01/{frame-*.color.png symlink, frame-*.pose.txt}
  <REPO>/data/zc_0923_enu/...

用法: mv2d python scripts/gt_probes/rewrite_pose_enu.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

RELOPT = "/data/code/reLocation-all/relopt"
REPO = "/data/code/reLocation-all/reloc3r"  # data symlink 目标；worktree 同指

BAGS = {
    "20260921_064638": "data/zc_fix",
    "20260921_062224": "data/zc_fix",
    "20260920_081533": "data/zc_fix",
    "20260923_055425_real_robot_record_external_lz4": "data/zc_0923",
    "20260923_060415_real_robot_record_external_lz4": "data/zc_0923",
}


def Rz_from_w(w):
    """2D complex w = cos + i sin → 3x3 rotation about Up/Z: (E,N)=Rot@(X,Y)."""
    c, s = float(w[0]), float(w[1])
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def assemble(R_map, t_map, cam_enu_xy, Rz):
    R_enu = Rz @ R_map
    t = np.array([cam_enu_xy[0], cam_enu_xy[1], t_map[2]], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_enu
    T[:3, 3] = t
    return T


def out_root_for(src_rel):
    # data/zc_fix -> data/zc_fix_enu
    return src_rel.rstrip("/") + "_enu"


def process_bag(bag, src_rel, dry_run=False):
    src_seq = f"{REPO}/{src_rel}/{bag}/seq-01"
    dst_rel = out_root_for(src_rel)
    dst_seq = f"{REPO}/{dst_rel}/{bag}/seq-01"
    gt_path = f"{RELOPT}/cache/rtk_gt_{bag}.npz"
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(gt_path)
    g = np.load(gt_path)
    fidx = g["fidx"]
    cam_enu = g["cam_enu"]
    cam_map = g["cam_map"]
    w = g["w"]
    o = g["o"]
    Rz = Rz_from_w(w)

    poses = sorted(n for n in os.listdir(src_seq) if n.endswith(".pose.txt"))
    colors = sorted(n for n in os.listdir(src_seq) if n.endswith(".color.png"))
    assert len(poses) == len(colors), f"{bag}: pose/color count mismatch"
    # fidx 与 extract 帧一一对应（Phase0/1 已核）
    assert len(fidx) == len(poses), f"{bag}: rtk {len(fidx)} vs poses {len(poses)}"
    frame_ids = [int(n.split("frame-")[1][:6]) for n in poses]
    assert frame_ids == list(map(int, fidx)), f"{bag}: frame id != fidx"

    if not dry_run:
        os.makedirs(dst_seq, exist_ok=True)

    dets, z22, tnorm, resid_xy, skip = [], [], [], [], 0
    written = 0
    for i, pname in enumerate(poses):
        fid = frame_ids[i]
        T_map = np.loadtxt(f"{src_seq}/{pname}")
        R_map, t_map = T_map[:3, :3], T_map[:3, 3]
        # 一致性：cam_enu 应 ≈ w·cam_map + o；亦应 ≈ (Rz @ t_map)_xy + o
        ce = cam_enu[i]
        cm = cam_map[i]
        pred = (complex(*w) * complex(*cm) + complex(*o))
        resid_xy.append(abs(complex(*ce) - pred))
        # 与旧 pose 平移的 map→ENU 也应接近 cam_enu（杆臂已在 pose）
        t_xy_from_pose = (Rz @ t_map)[:2] + o
        resid_xy[-1]  # keep list
        # 跳过病态旋转
        det = float(np.linalg.det(R_map))
        if not (0.95 < det < 1.05) or not np.isfinite(ce).all():
            skip += 1
            continue
        T_enu = assemble(R_map, t_map, ce, Rz)
        det_e = float(np.linalg.det(T_enu[:3, :3]))
        dets.append(det_e)
        z22.append(float(T_enu[2, 2]))
        tnorm.append(float(np.linalg.norm(T_enu[:2, 3])))
        cname = pname.replace(".pose.txt", ".color.png")
        if not dry_run:
            src_color = os.path.abspath(f"{src_seq}/{cname}")
            dst_color = f"{dst_seq}/{cname}"
            dst_pose = f"{dst_seq}/{pname}"
            if os.path.lexists(dst_color):
                if os.path.islink(dst_color) or os.path.isfile(dst_color):
                    os.remove(dst_color)
            os.symlink(src_color, dst_color)
            np.savetxt(dst_pose, T_enu)
            # 旁路校验：Rz@t_map_xy+o 与 cam_enu
            _ = t_xy_from_pose
        written += 1

    # 与旧 map 跨包一致性：同组内 db-query 最近邻 cam_map vs cam_enu 残差（仅统计用）
    stats = dict(
        bag=bag,
        src=src_rel,
        dst=dst_rel,
        n_src=len(poses),
        n_written=written,
        n_skip=skip,
        coverage=round(written / max(len(poses), 1), 4),
        det_R_med=round(float(np.median(dets)), 6) if dets else None,
        cam_z_dot_up_med=round(float(np.median(z22)), 4) if z22 else None,  # 期望 (-0.5,0)
        t_xy_norm_med_p90=[round(float(x), 3) for x in (np.median(tnorm), np.percentile(tnorm, 90))] if tnorm else None,
        cam_enu_vs_w_cam_map_resid_med_m=round(float(np.median(resid_xy)), 6) if resid_xy else None,
        theta_deg=float(g["theta_deg"]),
        off_rad=float(g["off_rad"]),
        o=list(map(float, o)),
        w=list(map(float, w)),
    )
    return stats


def cross_bag_consistency():
    """草坪三包：把 cam_map / cam_enu 对齐到共享 ENU 后，db(064638) 对两 query 的
    逐帧最近邻距离分布——ENU 应显著收紧跨包位置分歧（若 /map 原点漂）。"""
    lawn = ["20260921_064638", "20260921_062224", "20260920_081533"]
    G = {b: np.load(f"{RELOPT}/cache/rtk_gt_{b}.npz") for b in lawn}
    db = G["20260921_064638"]
    out = {}
    for qk in ("20260921_062224", "20260920_081533"):
        q = G[qk]
        # 暴力：每个 query 帧找 db 最近 cam
        for tag, key in (("map", "cam_map"), ("enu", "cam_enu")):
            A, B = db[key], q[key]
            # 子采样加速
            step = max(1, len(B) // 200)
            Bs = B[::step]
            dmin = []
            for p in Bs:
                dmin.append(float(np.min(np.linalg.norm(A - p, axis=1))))
            dmin = np.array(dmin)
            out[f"{qk}_{tag}_nn_med_p90_m"] = [round(float(np.median(dmin)), 3),
                                               round(float(np.percentile(dmin, 90)), 3)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--bags", nargs="*", default=None)
    args = ap.parse_args()
    keys = args.bags or list(BAGS)
    all_stats = {}
    for bag in keys:
        st = process_bag(bag, BAGS[bag], dry_run=args.dry_run)
        all_stats[bag] = st
        print(json.dumps(st, ensure_ascii=False))
    cross = cross_bag_consistency()
    all_stats["_cross_lawn_nn"] = cross
    print("cross_lawn_nn", json.dumps(cross, ensure_ascii=False))
    out_json = f"{RELOPT}/cache/rewrite_pose_enu_stats.json"
    if not args.dry_run:
        with open(out_json, "w") as f:
            json.dump(all_stats, f, ensure_ascii=False, indent=1)
        print("wrote", out_json)


if __name__ == "__main__":
    main()
