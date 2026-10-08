"""P0-2 (a)：db 侧 Reloc3r encoder 特征一次性预计算并缓存。

依据（优化路线文档 §1 疑问1）：Reloc3r = CroCo 架构，encoder 逐图独立（ViT 自注意力 +
LayerNorm + RoPE，无批间耦合），db 图建图后固定 ⇒ 每对 relpose 里对 db 图的重复 encode
是纯浪费。缓存路径与完整 forward 数学恒等。

存储口径 = 推理同口径：inference_relpose 在 autocast 下 forward，encoder 输出本就是 fp16，
故直接存 fp16（768 token × 1024 维 ≈ 1.5 MB/帧），解码时原样喂回 ⇒ 逐位一致（非近似）。
pos / true_shape 只依赖图像形状（整包同尺寸），各存一份。

用法（mv2d, 需 GPU）：
  python scripts/export_reloc3r_enc.py --key 062224   # db=20260921_064638
  python scripts/export_reloc3r_enc.py --key 0923
产物: cache/enc_<dbbag>.pt
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/data/code/reLocation-all/reloc3r")
from reloc3r.reloc3r_relpose import setup_reloc3r_relpose_model  # noqa: E402
from reloc3r.utils.image import load_images  # noqa: E402
from eval_zc_sim3 import SC  # noqa: E402  # 复用上游场景表（root/db/query/megaloc）

CACHE = "/data/code/reLocation-all/relopt/cache"


@torch.no_grad()
def encode_set(model, imgs, ts, device, batch=64):
    """imgs: list of load_images dicts (每张 img=(1,3,H,W))。返回 feat (N,tk,C) fp16-cpu, pos。"""
    feats, pos = [], None
    for i in range(0, len(imgs), batch):
        v = torch.cat([x["img"] for x in imgs[i:i + batch]]).to(device)
        shp = torch.as_tensor(np.concatenate([x["true_shape"] for x in imgs[i:i + batch]])).to(device)
        with torch.cuda.amp.autocast(enabled=True):
            x, pos, _ = model._encode_image(v, shp)
        feats.append(x.float().half().cpu())
        if i == 0:
            pos_cpu = pos[0].cpu()  # 同形状所有图 pos 相同
    return torch.cat(feats), pos_cpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True, choices=sorted(SC))
    args = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    root, db, q, _ = SC[args.key]
    device = torch.device("cuda")
    model = setup_reloc3r_relpose_model("Reloc3rRelpose(img_size=512)", device)
    dbn = sorted(n[:-10] for n in os.listdir(f"{root}/{db}/seq-01") if n.endswith(".color.png"))
    t0 = time.time()
    imgs = load_images([f"{root}/{db}/seq-01/{n}.color.png" for n in dbn], size=512, verbose=False)
    feat, pos = encode_set(model, imgs, None, device)
    ts = torch.as_tensor(np.stack([im["true_shape"][0] for im in imgs])).to(torch.int32)[:1]  # (1,2) 全同
    out = f"{CACHE}/enc_{db}.npz"
    np.savez(out, names=np.array(dbn), feat=feat.numpy(), pos=pos.numpy(),
             true_shape=np.asarray(imgs[0]["true_shape"][0], dtype=np.int32))
    print(f"{db}: {len(dbn)} 帧 feat={tuple(feat.shape)} {feat.dtype} "
          f"{os.path.getsize(out)/1e9:.2f} GB in {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
