"""遗忘检查: 官方 vs 微调权重在 MegaDepth-1500 后750对(与训练前750不相交)上的 relpose 误差。

用法: (cd reloc3r) python scripts/eval_forgetting.py [WEIGHTS_pth|'']
判定(方案§7): 微调后 terr/rerr 中位 ≤ 官方 +10%。产物: results/forgetting_{official|ft}.json
"""
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, '/data/code/reLocation-all/reloc3r')
from reloc3r.datasets.megadepth_valid import MegaDepth_valid  # noqa: E402
from reloc3r.reloc3r_relpose import Reloc3rRelpose, inference_relpose  # noqa: E402
from reloc3r.loss import Reloc3rPoseLoss  # noqa: E402

N = int(os.environ.get("N_EVAL", "200"))  # 200对≈1-2min；N_EVAL=750 全量更稳


def run(weights, tag):
    dev = torch.device('cuda')
    model = Reloc3rRelpose(img_size=512)
    # 缺省=官方 reloc3r-512（本地转换 pth）；绝非随机初始化
    w = weights or 'checkpoints/reloc3r-512.pth'
    ck = torch.load(w, map_location='cpu', weights_only=False)
    ck = ck.get('model', ck)
    print(model.load_state_dict(ck, strict=False))
    model.to(dev).eval()
    ds = MegaDepth_valid(split='test', resolution=(512, 384), pair_subset=0, seed=777)
    pl = Reloc3rPoseLoss()
    te, re_ = [], []
    with torch.no_grad():
        for i in range(min(N, len(ds))):
            v1, v2 = ds[i]
            gt = np.linalg.inv(np.asarray(v1['camera_pose'], dtype=np.float64)) @ np.asarray(v2['camera_pose'], dtype=np.float64)
            batch = [{'img': v['img'][None], 'true_shape': torch.as_tensor(v['true_shape'])[None]} for v in (v1, v2)]
            pr = inference_relpose(batch, model, dev, use_amp=True)[0]  # tensor (B,4,4)? B=1 per view
            gtT = torch.as_tensor(gt, dtype=torch.float32)[None].to(dev)
            p = pr[None] if pr.dim() == 2 else pr
            te.append(float(pl.transl_ang_loss(p[:, :3, 3], gtT[:, :3, 3]) * 180 / np.pi))
            re_.append(float(pl.rot_ang_loss(p[:, :3, :3], gtT[:, :3, :3]) * 180 / np.pi))
    out = {'tag': tag, 'n': len(te), 'terr_deg_med': float(np.median(te)),
           'rerr_deg_med': float(np.median(re_))}
    os.makedirs('results', exist_ok=True)
    path = f'results/forgetting_{tag}.json'
    json.dump(out, open(path, 'w'), indent=1)
    print(out, '->', path)
    return out


if __name__ == '__main__':
    # ZC-only 微调：公开集遗忘不作硬门禁（方案 Phase1）；仍可手动跑对照。
    if os.environ.get('ZC_ONLY', '0') in ('1', 'true', 'True', 'yes'):
        print('ZC_ONLY=1: skip MegaDepth forgetting hard gate (optional side note only).')
        tag = (sys.argv[2] if len(sys.argv) > 2 else 'zc_only_skip')
        out = {'tag': tag, 'skipped': True, 'reason': 'ZC_ONLY'}
        os.makedirs('results', exist_ok=True)
        path = f'results/forgetting_{tag}.json'
        json.dump(out, open(path, 'w'), indent=1)
        print(out, '->', path)
        raise SystemExit(0)
    w = sys.argv[1] if len(sys.argv) > 1 else ''
    run(w, (sys.argv[2] if len(sys.argv) > 2 else ('ft' if w else 'official')))
