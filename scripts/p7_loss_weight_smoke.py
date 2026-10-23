#!/usr/bin/env python3
"""P7 (a): 沿轨/横向损失加权草稿 + 极短 smoke（几 step，不写 ckpt）。

动机（方向文档）: Phase2 best 已收敛；再训期望有限。本脚本仅证明
  LOSS_ALONG_W / LOSS_CROSS_W / LOSS_ROT_W 可接到 Reloc3rPoseLoss 变体并反传。

不替代满训；/data 仅余 ~25G 且 ckpt 目录已 15G → 跳过 ≤3–5 epoch 续训。
"""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

# repo on path
sys.path.insert(0, "/data/code/reLocation-all/reloc3r-grok-bot")
sys.path.insert(0, "/data/code/reLocation-all/reloc3r")

from reloc3r.loss import Reloc3rPoseLoss  # noqa: E402


class Reloc3rPoseLossAniso(nn.Module):
    """相对平移拆成沿 GT 方向 / 横向，可分别加权；旋转保持角误差。

    env:
      LOSS_ALONG_W (default 1.0)
      LOSS_CROSS_W (default 1.0)  — 想压 cross 可 >1
      LOSS_ROT_W   (default 1.0)
    """

    def __init__(self, along_w=None, cross_w=None, rot_w=None):
        super().__init__()
        self.along_w = float(os.environ.get("LOSS_ALONG_W", along_w if along_w is not None else 1.0))
        self.cross_w = float(os.environ.get("LOSS_CROSS_W", cross_w if cross_w is not None else 1.0))
        self.rot_w = float(os.environ.get("LOSS_ROT_W", rot_w if rot_w is not None else 1.0))
        self._base = Reloc3rPoseLoss()

    def forward(self, pose_pred, pose_gt, eps=1e-6):
        t = pose_pred[:, 0:3, -1]
        tgt = pose_gt[:, 0:3, -1]
        R = pose_pred[:, :3, :3]
        Rgt = pose_gt[:, :3, :3]

        # unit GT translation direction in pair frame
        tgt_n = torch.norm(tgt, dim=1, keepdim=True).clamp_min(eps)
        u = tgt / tgt_n
        e = t - tgt
        along = (e * u).sum(1)  # signed
        cross_vec = e - along[:, None] * u
        cross = torch.norm(cross_vec, dim=1)

        # angular transl (legacy) kept for细节对比
        trans_ang = self._base.transl_ang_loss(t, tgt)
        rot = self._base.rot_ang_loss(R, Rgt)

        # metric along/cross in meters (batch mean)
        along_loss = along.abs().mean()
        cross_loss = cross.mean()
        loss = self.along_w * along_loss + self.cross_w * cross_loss + self.rot_w * rot
        details = {
            "along_m": float(along_loss),
            "cross_m": float(cross_loss),
            "rot_rad": float(rot),
            "trans_ang_rad": float(trans_ang),
            "along_w": self.along_w,
            "cross_w": self.cross_w,
            "rot_w": self.rot_w,
        }
        return loss, details


def smoke(steps=5, bs=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # random SE3-ish poses
    def rand_pose(b):
        P = torch.eye(4, device=device).repeat(b, 1, 1)
        P[:, :3, 3] = torch.randn(b, 3, device=device) * 0.5
        # small random rot via axis-angle
        aa = torch.randn(b, 3, device=device) * 0.1
        theta = torch.norm(aa, dim=1, keepdim=True).clamp_min(1e-8)
        k = aa / theta
        K = torch.zeros(b, 3, 3, device=device)
        K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
        K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
        K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
        R = torch.eye(3, device=device).repeat(b, 1, 1) + torch.sin(theta)[:, :, None] * K + (1 - torch.cos(theta))[:, :, None] * (K @ K)
        P[:, :3, :3] = R
        return P

    crit = Reloc3rPoseLossAniso().to(device)
    # tiny linear so we have params to step
    head = nn.Linear(16, 16, device=device)
    opt = torch.optim.Adam(list(crit.parameters()) + list(head.parameters()), lr=1e-3)
    # crit has no params — just prove loss backward through a dummy that mixes poses
    # Use pose_pred as leaf with grad
    hist = []
    for i in range(steps):
        gt = rand_pose(bs)
        pred = gt.clone().detach()
        pred[:, :3, 3] = pred[:, :3, 3] + 0.05 * torch.randn_like(pred[:, :3, 3])
        pred = pred.detach().requires_grad_(True)
        loss, det = crit(pred, gt)
        # also touch head so optimizer has work
        dummy = head(pred[:, :4, :4].reshape(bs, -1)).square().mean()
        (loss + 0.0 * dummy).backward()
        opt.step(); opt.zero_grad()
        hist.append({"step": i, "loss": float(loss.detach()), **det})
        print(f"step{i}: loss={loss.item():.6f} along={det['along_m']:.4f} cross={det['cross_m']:.4f} rot={det['rot_rad']:.4f}")
    return hist


if __name__ == "__main__":
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(os.environ.get("OUT_DIR", "/home/yr/grok-bot-work/logs_zc_opt_p47_20261023"))
    out.mkdir(parents=True, exist_ok=True)
    # two weight settings
    reports = {}
    for tag, env in [
        ("default_1_1_1", {"LOSS_ALONG_W": "1", "LOSS_CROSS_W": "1", "LOSS_ROT_W": "1"}),
        ("cross2x", {"LOSS_ALONG_W": "1", "LOSS_CROSS_W": "2", "LOSS_ROT_W": "1"}),
    ]:
        for k, v in env.items():
            os.environ[k] = v
        print(f"==== {tag} {env} ====")
        reports[tag] = smoke(steps=int(os.environ.get("P7_STEPS", "5")))
    payload = {
        "stamp": stamp,
        "note": "P7(a) loss-weight smoke only; skipped short continue train due to /data ~25G free & 15G ckpt dir",
        "disk_reason": "/data 99% used (~25G free); checkpoint-best ~2.6G each; ≤3–5 ep would risk fill",
        "reports": reports,
    }
    (out / f"p7_loss_smoke_{stamp}.json").write_text(json.dumps(payload, indent=2))
    print("wrote", out / f"p7_loss_smoke_{stamp}.json")
