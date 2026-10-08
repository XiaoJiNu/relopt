"""P0-2 (b)：encoder 特征缓存版 sim3 评测（对照 eval_zc_sim3.py，逐位一致性验证 + 计时）。

缓存口径（§1 疑问1 实测）：
  完整 forward K=3 = 每对重复 encode query + 在线 encode db；
  缓存路径      = db 特征查表 + query 每帧 encode 一次复用 K 次 → decoder+head。
数学恒等前提：encoder 逐图独立（CroCo ViT 自注意力 + RoPE）。--verify 用实测证伪/证实。

用法（mv2d）：
  python scripts/eval_zc_sim3_cache.py --key 062224 --verify          # 等价性门槛（先跑）
  python scripts/eval_zc_sim3_cache.py --key 062224 --out results/...  # 全量 eval + 计时
  SIM3_K=5 SIM3_BM3=1 同参跑 P1-2（sim5+bm3）。
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/data/code/reLocation-all/reloc3r")
from reloc3r.reloc3r_relpose import setup_reloc3r_relpose_model, inference_relpose  # noqa: E402
from reloc3r.reloc3r_visloc import Reloc3rVisloc  # noqa: E402
from reloc3r.utils.metric import get_rot_err  # noqa: E402
from reloc3r.utils.image import load_images  # noqa: E402
from eval_zc_sim3 import SC, ML, FIX  # noqa: E402

CACHE = "/data/code/reLocation-all/relopt/cache"
K = int(os.environ.get("SIM3_K", "3"))
USE_BM3 = os.environ.get("SIM3_BM3", "0") == "1"


def forward_cached(model, feat1, pos1, shape1, feat2, pos2, shape2):
    """复制 Reloc3rRelpose.forward 的 decoder+head 段（reloc3r_relpose.py:154-163），
    encoder 输出改为缓存/单算——恒等性由 --verify 实测把关。"""
    dec1, dec2 = model._decoder(feat1, pos1, feat2, pos2)
    with torch.cuda.amp.autocast(enabled=False):
        pose2 = model._downstream_head([tok.float() for tok in dec2], shape2)
    return pose2["pose"]


@torch.no_grad()
def encode_one(model, img, ts, device):
    with torch.cuda.amp.autocast(enabled=True):
        x, pos, _ = model._encode_image(img.to(device), ts.to(device))
    return x, pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True, choices=sorted(SC))
    ap.add_argument("--verify", action="store_true", help="前 N 帧双路径逐位对比，不跑全量")
    ap.add_argument("--n-verify", type=int, default=20)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root, db, q, mlf = SC[args.key]
    order = np.load(mlf)["order"]
    dbn = sorted(n[:-10] for n in os.listdir(f"{root}/{db}/seq-01") if n.endswith(".color.png"))
    qn = sorted(n[:-10] for n in os.listdir(f"{root}/{q}/seq-01") if n.endswith(".color.png"))
    device = torch.device("cuda")
    model = setup_reloc3r_relpose_model("Reloc3rRelpose(img_size=512)", device)
    visloc = Reloc3rVisloc()
    pose = lambda s, n: np.loadtxt(f"{root}/{s}/seq-01/{n}.pose.txt")

    ck = np.load(f"{CACHE}/enc_{db}.npz")
    assert list(ck["names"]) == dbn, "缓存 db 帧表与目录不一致"
    feat_db = torch.from_numpy(ck["feat"]).to(device)  # (Ndb, tk, C) fp16 —— 存的就是 autocast 下的 fp16 输出
    pos_db = torch.from_numpy(ck["pos"]).to(device)
    ts_db = ck["true_shape"]
    q_imgs = load_images([f"{root}/{q}/seq-01/{n}.color.png" for n in qn], size=512, verbose=False)

    n_iter = min(len(qn), args.n_verify) if args.verify else len(qn)
    diffs, T, R, t_cache, t_full, PREDS = [], [], [], 0.0, 0.0, []
    db_imgs_sel = None
    for qi in range(n_iter):
        sel = [int(j) for j in order[qi, :K]]
        n = len(sel)
        shape2 = torch.as_tensor(q_imgs[qi]["true_shape"]).repeat(n, 1)
        t0 = time.time()
        f2, p2 = encode_one(model, q_imgs[qi]["img"], q_imgs[qi]["true_shape"], device)
        f2 = f2.repeat(n, 1, 1)
        p2 = p2.repeat(n, 1, 1)
        f1 = feat_db[sel]
        p1 = pos_db[None].repeat(n, 1, 1)
        s1 = torch.as_tensor(np.stack([ts_db] * n))  # db 全同形状(640x480→同一 true_shape)
        with torch.cuda.amp.autocast(enabled=True):
            q2d = forward_cached(model, f1, p1, s1, f2, p2, shape2)
        torch.cuda.synchronize()
        t_cache += time.time() - t0
        pq = [p.cpu().numpy() for p in q2d]

        if args.verify:  # 完整 forward 对照（与 eval_zc_sim3.py 相同调用式）
            if db_imgs_sel is None:
                db_imgs_sel = load_images([f"{root}/{db}/seq-01/{n}.color.png" for n in dbn], size=512, verbose=False)
            v1 = {"img": torch.cat([db_imgs_sel[j]["img"] for j in sel], 0),
                  "true_shape": torch.as_tensor(np.stack([db_imgs_sel[j]["true_shape"][0] for j in sel]))}
            v2 = {"img": q_imgs[qi]["img"].repeat(n, 1, 1, 1), "true_shape": shape2}
            t0 = time.time()
            q2d_full = inference_relpose([v1, v2], model, device, use_amp=True)
            torch.cuda.synchronize()
            t_full += time.time() - t0
            diffs.append(max(float((a.cpu() - b.cpu()).abs().max()) for a, b in zip(q2d, q2d_full)))

        pd = [pose(db, dbn[j]) for j in sel]
        if USE_BM3:
            cand_t = np.array([(pd[kk] @ pq[kk])[:3, 3] for kk in range(n)])
            rough = np.median(cand_t, axis=0)
            keep = [kk for kk in range(n) if np.linalg.norm(pd[kk][:3, 3] - rough) < 3.0]
            if len(keep) < 2:
                keep = list(range(n))
            pd, pq = [pd[i] for i in keep], [pq[i] for i in keep]
        Rt = visloc.motion_averaging(pd, pq)
        gt = pose(q, qn[qi])
        T.append(float(np.linalg.norm(Rt[:3, 3] - gt[:3, 3])))
        R.append(get_rot_err(Rt[:3, :3], gt[:3, :3]))
        PREDS.append(Rt)

    T, R = np.array(T), np.array(R)
    tag = f"sim{K}{'_bm3' if USE_BM3 else ''}"
    print(f"{args.key} {tag} cache: n={len(T)} terr mean/med/p90 = {T.mean():.2f}/{np.median(T):.2f}/{np.percentile(T,90):.2f} | "
          f"rerr mean/med/p90 = {R.mean():.2f}/{np.median(R):.2f}/{np.percentile(R,90):.2f} | "
          f"{(t_cache/n_iter)*1000:.1f} ms/帧" + (f" | full-forward {(t_full/n_iter)*1000:.1f} ms/帧" if args.verify else ""))
    if args.verify:
        print(f"等价性: {len(diffs)} 帧双路径 pose 最大绝对差 = {max(diffs):.3e}（逐位一致 <1e-6 / 容差通过 <1e-2 需注明）")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        np.savez(args.out, terr=T, rerr=R, names=np.array(qn[:len(T)]),
                 preds=np.array(PREDS), ms_per_frame=t_cache / n_iter * 1000)
        print("saved ->", args.out)


if __name__ == "__main__":
    main()
