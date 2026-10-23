"""减税路径实测: MegaLoc 检索相似度前置预筛 top-3 -> 仅 3 对 relpose -> 运动平均。

候选集 = MegaLoc snap 存档的 order[:, :3]（全库相似度排名, GT-free）。
对照: 同数据 top-10 全推(+bm3) 已存档。relpose 计时按 batch=3 实测折算单帧。
用法: python eval_zc_sim3.py (repo root) <scene_key>
  env SIM3_K=3/5  SIM3_BM3=0/1 （同前）
  env WEIGHTS=/path/to/ft.pth 微调权重（{'model':...} pth；缺省=官方 siyan824/reloc3r-512）
  env EVAL_PARITY=even|odd|all 查询帧奇偶子集（微调防泄漏协议 train=even/eval=odd，同子集对比基线）
  env OUT_TAG=suffix 输出文件后缀（默认按权重/parity 自动命名，不覆盖归档 sim3 原文件）
"""
import os, sys, time
import numpy as np
sys.path.insert(0, '/data/code/reLocation-all/reloc3r')
from reloc3r.reloc3r_relpose import Reloc3rRelpose, setup_reloc3r_relpose_model, inference_relpose
from reloc3r.reloc3r_visloc import Reloc3rVisloc
from reloc3r.utils.metric import get_rot_err
from reloc3r.utils.image import load_images

ML = "/data/code/reLocation-all/MegaLoc/outputs/eval_zc_megaloc"
FIX = "/data/code/reLocation-all/reloc3r/data"
OUT = f"{FIX}/zc_sim3"
# Phase2: POSE_TAG=enu 或 DATA_ROOT_LAWN/DATA_ROOT_BLDG 指向旁路 ENU 树
_POSE_TAG = os.environ.get("POSE_TAG", "").strip().lower()
_ENU = _POSE_TAG in ("enu", "enu_rtk", "1", "true", "yes")
_LAWN = os.environ.get("DATA_ROOT_LAWN") or (FIX + ("/zc_fix_enu" if _ENU else "/zc_fix"))
_BLDG = os.environ.get("DATA_ROOT_BLDG") or (FIX + ("/zc_0923_enu" if _ENU else "/zc_0923"))
SC = {
 "062224": (_LAWN, "20260921_064638", "20260921_062224", ML + "/res__mowing__same_day_062224.npz"),
 "081533": (_LAWN, "20260921_064638", "20260920_081533", ML + "/res__mowing__cross_day_081533.npz"),
 "0923": (_BLDG, "20260923_055425_real_robot_record_external_lz4",
          "20260923_060415_real_robot_record_external_lz4", ML + "/res__building_0923__query_060415.npz"),
}
K = int(os.environ.get("SIM3_K", "3"))
USE_BM3 = os.environ.get("SIM3_BM3", "0") == "1"
AGG = os.environ.get("SIM3_AGG", "")  # ""=现行 | dualgate(M1 双锚+时序门)
REJ = float(os.environ.get("SIM3_REJ_M", "2.0"))  # dualgate 跳变拒绝阈值 m（0=关；离线扫 2.0 最优）
WEIGHTS = os.environ.get("WEIGHTS", "")           # 微调 pth；空=官方
EVAL_PARITY = os.environ.get("EVAL_PARITY", "all")  # even|odd|all（查询帧奇偶）

def _parity(path, par):
    return par == "all" or int(os.path.basename(path).split("frame-")[-1][:6]) % 2 == (par == "odd")

def main(key):
    import torch
    root, db, q, mlf = SC[key]
    _tag = _POSE_TAG or "map"
    _gt = "ENU/RTK" if _ENU else "/map"
    print(f"[eval_zc_sim3] key={key} DATA_ROOT={root} POSE_TAG={_tag} GT口径={_gt}")
    order = np.load(mlf)["order"]
    dbn = sorted(n[:-10] for n in os.listdir(f"{root}/{db}/seq-01") if n.endswith(".color.png"))
    qn_all = sorted(n[:-10] for n in os.listdir(f"{root}/{q}/seq-01") if n.endswith(".color.png"))
    qn = [n for n in qn_all if _parity(n, EVAL_PARITY)]
    qsel = np.array([i for i, n in enumerate(qn_all) if _parity(n, EVAL_PARITY)])  # order 行索引
    device = torch.device("cuda")
    # WEIGHTS 路径：本地构图+load_state_dict，避免 from_pretrained 访问 huggingface
    if WEIGHTS:
        model = Reloc3rRelpose(img_size=512).to(device)
        model.eval()
        ck = torch.load(WEIGHTS, map_location=device, weights_only=False)  # 本地可信 ckpt（含 optimizer/args）
        ck = ck.get("model", ck)
        ck = {k.replace("dec_blocks2", "dec_blocks"): v for k, v in ck.items()} if any(
            k.startswith("dec_blocks2") for k in ck) else dict(ck)
        print(model.load_state_dict(ck, strict=False))
        print(f"finetuned weights loaded: {WEIGHTS}")
    else:
        model = setup_reloc3r_relpose_model("Reloc3rRelpose(img_size=512)", device)
    visloc = Reloc3rVisloc()
    db_imgs = load_images([f"{root}/{db}/seq-01/{n}.color.png" for n in dbn], size=512, verbose=False)
    q_imgs = load_images([f"{root}/{q}/seq-01/{n}.color.png" for n in qn], size=512, verbose=False)
    # P5: APPLY_OFF=1 → 仅旋入「相对本场景 db 包」的 Δoff（双天线逐包标定差，~0–1.3°）
    # 禁止把绝对 off_rad(~93°) 旋入：会破坏运动平均射线几何（实测 terr 炸到米级）。
    _APPLY_OFF = os.environ.get("APPLY_OFF", "0") == "1"
    _OFF_CACHE = {}
    def _off_rad(bag):
        if bag not in _OFF_CACHE:
            g = np.load(f"/data/code/reLocation-all/relopt/cache/rtk_gt_{bag}.npz")
            _OFF_CACHE[bag] = float(g["off_rad"])
        return _OFF_CACHE[bag]
    def _Rz(yaw):
        c, s = float(np.cos(yaw)), float(np.sin(yaw))
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    _off_db0 = _off_rad(db) if _APPLY_OFF else 0.0
    def pose(s, n):
        T = np.loadtxt(f"{root}/{s}/seq-01/{n}.pose.txt")
        if _APPLY_OFF:
            d_off = _off_rad(s) - _off_db0  # db 包 Δ=0；query 包 ≈ off_q-off_db
            if abs(d_off) > 1e-12:
                T = T.copy(); T[:3, :3] = _Rz(d_off) @ T[:3, :3]
        return T
    if _APPLY_OFF:
        print(f"[eval_zc_sim3] APPLY_OFF=1 Δoff vs db={db}: "
              f"db0={_off_db0:.6f} q_doff={_off_rad(q)-_off_db0:.6f} rad "
              f"({(_off_rad(q)-_off_db0)*180/3.14159265:.3f} deg)")

    T, R, V, t_infer = [], [], [], 0.0
    prev_pos = None
    DGT = np.gradient(np.array([pose(q, n)[:2, 3] for n in qn]), axis=0)  # 时序门运动增量（离线口径，上真机换 /odom）
    for qi in range(len(qn)):
        sel = [int(j) for j in order[qsel[qi], :K]]
        n = len(sel)
        v1 = {"img": torch.cat([db_imgs[j]["img"] for j in sel], 0),
              "true_shape": torch.as_tensor(np.stack([db_imgs[j]["true_shape"] for j in sel])).reshape(n, 2)}
        v2 = {"img": q_imgs[qi]["img"].repeat(n, 1, 1, 1),
              "true_shape": torch.as_tensor(q_imgs[qi]["true_shape"]).repeat(n, 1)}
        t0 = time.time()
        with torch.no_grad():
            q2d = inference_relpose([v1, v2], model, device, use_amp=True)
        torch.cuda.synchronize()
        t_infer += time.time() - t0
        pd = [pose(db, dbn[j]) for j in sel]
        pq = [p.cpu().numpy() for p in q2d]
        AGG = os.environ.get("SIM3_AGG", "")
        if AGG == "dualgate":
            # M1: top-1 锚 + 中位数锚双判据 + 时序门（优化方案 §4 M1 方法1；离线扫=agg_sweep_m1 dual_gate）
            r_bm3 = float(os.environ.get("SIM3_R", "3.0"))
            agree = float(os.environ.get("SIM3_AGREE", "1.0"))
            gate = float(os.environ.get("SIM3_GATE_M", "3.0"))
            imp = np.array([(pd[kk] @ pq[kk])[:3, 3] for kk in range(n)])
            med = np.median(imp[:n], 0)
            sel_a1 = [kk for kk in range(n) if np.linalg.norm(imp[kk] - imp[0]) < r_bm3] or [0]
            keep_m = [kk for kk in range(n) if np.linalg.norm(pd[kk][:3, 3] - med) < r_bm3]
            if len(keep_m) < 2:
                keep_m = list(range(n))
            ok = np.linalg.norm(imp[0] - med) <= agree
            Rt_a1 = visloc.motion_averaging([pd[i2] for i2 in sel_a1], [pq[i2] for i2 in sel_a1])
            cont = prev_pos is not None and np.linalg.norm(Rt_a1[:2, 3] - (prev_pos + DGT[qi])) <= gate
            keep = sel_a1 if (ok or cont) else keep_m
            pd, pq = [pd[i2] for i2 in keep], [pq[i2] for i2 in keep]
        elif USE_BM3:
            cand_t = np.array([(pd[kk] @ pq[kk])[:3, 3] for kk in range(n)])
            rough = np.median(cand_t, axis=0)
            keep = [kk for kk in range(n) if np.linalg.norm(pd[kk][:3, 3] - rough) < 3.0]
            if len(keep) < 2:
                keep = list(range(n))
            pd, pq = [pd[i] for i in keep], [pq[i] for i in keep]
        Rt = visloc.motion_averaging(pd, pq)
        if AGG == "dualgate" and REJ > 0 and prev_pos is not None \
                and np.linalg.norm(Rt[:2, 3] - (prev_pos + DGT[qi])) > REJ:
            Rt[:2, 3] = prev_pos + DGT[qi]  # M1 方法3-lite: 聚合跳变>REJ m 冻结为预测位（离线扫 reject=2.0 最优）
        prev_pos = Rt[:2, 3].copy()
        gt = pose(q, qn[qi])
        dvec = Rt[:3, 3] - gt[:3, 3]
        T.append(float(np.linalg.norm(dvec)))
        R.append(get_rot_err(Rt[:3, :3], gt[:3, :3]))
        V.append(dvec)  # A2 分量分析用（沿/跨 GT 轨迹方向）
    T, R = np.array(T), np.array(R)
    V = np.array(V)
    tag = f"_{os.environ.get('FT_TAG', 'ft' if WEIGHTS else 'official')}_{EVAL_PARITY}" if (WEIGHTS or EVAL_PARITY != "all" or os.environ.get('FT_TAG')) else ""
    tag += os.environ.get("OUT_TAG", "")
    aggt = "_dg" if AGG == "dualgate" else ""
    np.savez(f"{OUT}/res_{key}_sim{K}{'_bm3' if USE_BM3 else ''}{aggt}{tag}.npz",
             terr=T, rerr=R, dvec=V, names=np.array(qn))
    print(f"{key} sim{K}{'_bm3' if USE_BM3 else ''}{aggt}{tag}: n={len(T)} "
          f"terr mean/med/p90 = {T.mean():.2f}/{np.median(T):.2f}/{np.percentile(T,90):.2f} | "
          f"rerr mean/med/p90 = {R.mean():.2f}/{np.median(R):.2f}/{np.percentile(R,90):.2f} | "
          f"relpose {(t_infer/len(qn))*1000:.0f} ms/帧")

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    main(sys.argv[1])
