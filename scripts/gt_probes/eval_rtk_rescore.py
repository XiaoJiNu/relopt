"""M2 验收: 用 RTK 锚定 + 朝向去偏真值重新评分候选 dump（零 GPU、零推理）。

评测口径转换（§1.1 的"两包 /map 分歧进指标"在这里被扣除）：
  位置: pred(map系,锚在 db 包) -> ENU: w_db·p + o_db；GT: query 包 cam_enu（拟合直读 pose.txt）
        ep_rtk = |w_db·pred + o_db - cam_enu_q|
  朝向: eθ_rtk = wrap(θ_pred - θ_gt) - (off_db - off_q)   （跨包 /map 朝向基准差校正）
变体: bm3(现行) / dualgate(M1) — 与 agg_sweep_m1.py 同一聚合实现。
用法: mv2d python scripts/gt_probes/eval_rtk_rescore.py [--dvec res_*.npz ...]
"""
import argparse
import json
import math
import os

import numpy as np
from scipy.spatial.transform import Rotation as Rot

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agg_sweep_m1 import agg, sweep_frame  # 同一实现，保证与离线扫口径一致

RELOPT = "/data/code/reLocation-all/relopt"
SC = {  # scene_key: (cands dump, db bag, query bag)
    "062224": ("062224", "20260921_064638", "20260921_062224"),
    "081533": ("081533", "20260921_064638", "20260920_081533"),
    "0923": ("0923", "20260923_055425_real_robot_record_external_lz4",
             "20260923_060415_real_robot_record_external_lz4"),
}
FIXMAP = {"062224": "data/zc_fix", "081533": "data/zc_fix", "0923": "data/zc_0923"}


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def yaw_of(R):
    """相机位姿 3x3 -> 光轴水平方位角（与 pose.txt 同链，跨包只差 off）。"""
    return math.atan2(R[1, 2], R[0, 2])


def rescore(dfile, K=5, r=3.0, agree=1.0, gate=3.0, rej=2.0):
    key = os.path.basename(dfile).split("cands_")[1].split("_K")[0]
    _, db_bag, q_bag = SC[key]
    d = np.load(dfile)
    pdb, pq2d, gt = d["pdb"], d["pq2d"], d["gt"]
    names = [int(n.split("frame-")[-1][:6]) for n in d["names"]]
    gtg = np.load(f"{RELOPT}/cache/rtk_gt_{q_bag}.npz")
    fidx = np.array(names)
    pos = {int(v): i for i, v in enumerate(gtg["fidx"])}
    assert all(f in pos for f in fidx), f"{key}: 帧号缺失于 rtk_gt"
    rows = np.array([pos[f] for f in fidx])
    dbgt = np.load(f"{RELOPT}/cache/rtk_gt_{db_bag}.npz")
    w_db = complex(*dbgt["w"]); o_db = complex(*dbgt["o"])
    # off 差（双天线口径）默认不扣：实测原始 eθ 分布中心≈0，证明跨包 /map 航向系并无 1.3° 旋转
    # （若真有，/map 口径 raw eθ 会整体偏 1.3°；位置共享 w 拟合 6cm 也证明两包 map 轴系一致）。
    # 1.31° 差异属于双天线航向的逐包标定偏差，不是评测消费的跨包基准差。--bias-correct 可强行扣。
    dw_off = (dbgt["off_rad"] - gtg["off_rad"]) if os.environ.get("BIAS_CORRECT") == "1" else 0.0
    cam_enu_q = gtg["cam_enu"][rows]
    gt_yaw = np.array([yaw_of(R[:3, :3]) for R in gt])
    out = {}
    for variant in ("bm3", "dualgate"):
        prev_pos = None
        dgt = np.gradient(gt[:, :2, 3], axis=0)
        EP, ET = [], []
        DE = []
        for i in range(len(gt)):
            Rt_a1, Rt_m, ok = sweep_frame(pdb[i], pq2d[i], pdb[i].shape[0], K, r, agree)
            pred = None if prev_pos is None else prev_pos[:2] + dgt[i]
            if variant == "bm3":
                Rt = Rt_m
            else:  # dualgate = eval_zc_sim3 SIM3_AGG=dualgate + REJ 同语义
                Rt = Rt_a1 if (ok or (pred is not None and np.linalg.norm(Rt_a1[:2, 3] - pred) <= gate)) else Rt_m
                if rej > 0 and pred is not None and np.linalg.norm(Rt[:2, 3] - pred) > rej:
                    Rt = Rt.copy(); Rt[:2, 3] = pred
            prev_pos = Rt[:3, 3]
            pe = w_db * (Rt[0, 3] + 1j * Rt[1, 3]) + o_db
            de = pe - (cam_enu_q[i, 0] + 1j * cam_enu_q[i, 1])
            EP.append(float(abs(de)))
            DE.append(de)
            ET.append(abs(math.degrees(wrap(yaw_of(Rt[:3, :3]) - gt_yaw[i]) - dw_off)))
        EP, ET = np.array(EP), np.array(ET)
        DE = np.array(DE)
        tang = np.gradient(cam_enu_q, axis=0)
        tang = tang[:, 0] + 1j * tang[:, 1]
        tang /= (np.abs(tang) + 1e-9)
        al = np.abs((DE * np.conj(tang)).real); cr = np.abs((DE * np.conj(tang)).imag)
        out[variant] = dict(ep_mean=round(float(EP.mean()), 4), ep_med=round(float(np.median(EP)), 4),
                            ep_p90=round(float(np.percentile(EP, 90)), 4), n_gt0p5=int((EP > 0.5).sum()),
                            et_mean=round(float(ET.mean()), 3), et_med=round(float(np.median(ET)), 3),
                            et_p90=round(float(np.percentile(ET, 90)), 3),
                            along_cm=[round(float(x * 100), 1) for x in (al.mean(), np.median(al), np.percentile(al, 90))],
                            cross_cm=[round(float(x * 100), 1) for x in (cr.mean(), np.median(cr), np.percentile(cr, 90))])
        # 同帧集 /map 口径对照（不扣真值误差）——证明"扣掉的是真值分歧不是模型误差"
    out["_note"] = f"yaw_off_db_off_q_deg={math.degrees(dw_off):.3f}"
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--tag", default="ft")  # 权重臂: ft | ftM | official
    a = ap.parse_args()
    res = {}
    for k in SC:
        f = f"/data/code/reLocation-all/reloc3r/data/zc_sim3/cands_{k}_K10_{a.tag}_odd.npz"
        res[k] = rescore(f, K=a.K)
        print(k, json.dumps(res[k], ensure_ascii=False))
    json.dump(res, open(f"{RELOPT}/scripts/gt_probes/eval_rtk_rescore_{a.tag}.json", "w"), indent=1)
