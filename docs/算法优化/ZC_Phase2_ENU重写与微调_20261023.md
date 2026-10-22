# ZC Phase2：ENU 旁路重写 + ZC-only 微调（进行中）

- **日期**：2026-10-23（Asia/Shanghai）
- **机器**：拯救者 `252f7d1f-95fc-44c6-94d1-8f76153ee6af`
- **依据**：审核稿 §Phase2；讨论结论「统一 ENU」；用户已批准继续 Phase2
- **红线**：不覆盖 `data/zc_fix` / `zc_0923` pose；不 sudo；不 push；绝对指标必须标 **GT=ENU/RTK**；诚实写 RTK 底噪可能挡 cross&lt;3cm

---

## 0. 开工状态

| 项 | 状态 |
| :--- | :--- |
| A 旁路 ENU pose | ✅ 完成 |
| B MegaLoc 对（ENU） | ✅ 完成 |
| C SMOKE 微调 | ✅ 通过（1 epoch） |
| C 满训 30 epoch | 🔄 进行中 |
| D 自测（ENU 口径） | ⏳ 待满训结束 |
| E 本文档 | 🔄 进行中 → 收工补全 |

---

## 1. A. ENU 旁路 pose

### 1.1 组装口径

脚本：`relopt/scripts/gt_probes/rewrite_pose_enu.py`

```
t_xy = cam_enu          # 已含杆臂（build_rtk_gt: w·cam_map + o）
t_z  = pose_map[2,3]    # map/ENU 共享竖直轴
R    = R_z(theta) @ R_map
```

- `off_rad` **未**旋入 R（与 `eval_rtk_rescore` 默认 `BIAS_CORRECT=0` 一致）
- 图像：symlink 原 `color.png`；仅新写 `pose.txt`（省磁盘）

### 1.2 输出路径

| 旁路根 | 包数/帧 | 覆盖率 |
| :--- | ---: | ---: |
| `/data/code/reLocation-all/reloc3r/data/zc_fix_enu/` | 3 包 / 2189 帧 | 100% |
| `/data/code/reLocation-all/reloc3r/data/zc_0923_enu/` | 2 包 / 1709 帧 | 100% |

（worktree `reloc3r-grok-bot/data` → 同目录 symlink）

### 1.3 冒烟

- `det(R)≈1.000001`；`cam_z·up = pose[2,2]` 中位 ≈ −0.17～−0.18（守卫期望 (−0.5,0)）
- 包内相对位姿与 `/map` 相对位姿差 &lt;1e-14（同组共享 w 时刚体变换保持相对不变）
- 草坪局部 `cam_enu` 量级：数米～数十米；大楼：更大（ENU 原点不同）
- 统计：`relopt/cache/rewrite_pose_enu_stats.json`

### 1.4 跨包一致性（诚实）

草坪 db(064638)→query 最近邻距离 **map vs ENU 几乎相同**（med≈0.15–0.20 m）：

- 同组共享 `w`、`o` 近似 → ENU 对草坪近似全局刚体，**不显著收紧**跨包 NN
- 该 ~15–20 cm NN 反映 **/map↔RTK 拟合残差 + 轨迹采样**，是绝对尺子的底噪量级
- **因此 cross&lt;3cm 仍可能被 RTK/拟合底噪挡住**；禁止用旧 `/map` 尺子谎称达标

---

## 2. B. MegaLoc 对（ENU）

```bash
DATA_TAG=enu python relopt/scripts/gt_probes/regen_pairs_megaloc.py
```

| 场景 | 输出 npy | n_query | kept_frac | kept baseline med |
| :--- | :--- | ---: | ---: | ---: |
| 062224 | `_db-q_pair_info/zcME_20260921_062224_...npy` | 591 | 0.837 | 1.34 m |
| 081533 | `zcME_20260920_081533_...npy` | 823 | 0.967 | 1.32 m |
| 0923 | `zc9ME_20260923_060415_...npy` | 756 | 0.885 | 1.54 m |

- 基线过滤仍 0.15–10 m（ENU 相对位移）
- 与 map 版 MegaLoc top-10 重叠 med=10（检索序相同、相对基线几乎不变）
- 统计：`_db-q_pair_info/regen_pairs_megaloc_enu_stats.json`
- **未覆盖**旧 `zcM_*` / `zc9M_*`

---

## 3. C. 训练

### 3.1 入口

- `reloc3r-grok-bot/scripts/finetune_zc_only_enu.sh`
- `OUT_DIR=checkpoints/zc_finetune_ZC_only_enu`（不覆盖 Phase1）
- `pretrained=checkpoints/reloc3r-512.pth`（**官方**，与 Phase1 公平对比）
- 协议同 Phase1：N=4000×3、30 epoch、BS=8、LR=1e-5、`freeze_encoder`、train even / val odd

### 3.2 SMOKE

- `SMOKE=1` → `checkpoints/_smoke_zc_only_enu`，1 epoch 通过
- 日志：`/home/yr/grok-bot-work/logs_zc_enu/smoke_zc_only_enu.log`

### 3.3 满训（进行中）

| 项 | 值 |
| :--- | :--- |
| PID | **200373**（bash 包装 200370） |
| 日志 | `/home/yr/grok-bot-work/logs_zc_enu/finetune_zc_only_enu_20261023_002551.log` |
| 开始 | 2026-10-23 00:25 CST |
| 节奏 | ~1500 it/epoch × ~0.48 s ≈ 12 min 训 + ~1 min eval |
| ETA | **约 06:00–06:30 CST**（30 epoch ≈ 5.5–6.5 h） |
| GPU | RTX 5090 Laptop，占用约 11–18 GB |

盯盘建议（给主助手 / routine）：

```bash
# 进度
tail -5 /home/yr/grok-bot-work/logs_zc_enu/finetune_zc_only_enu_20261023_002551.log
# 是否结束
test -f /data/code/reLocation-all/reloc3r-grok-bot/checkpoints/zc_finetune_ZC_only_enu/checkpoint-final.pth && echo DONE
# 结束后三场景 ENU 评测
cd /data/code/reLocation-all/reloc3r-grok-bot
W=checkpoints/zc_finetune_ZC_only_enu/checkpoint-best.pth
for k in 062224 081533 0923; do
  POSE_TAG=enu SIM3_K=5 SIM3_BM3=1 EVAL_PARITY=odd WEIGHTS=$W FT_TAG=zcOnlyENU \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    /data/software/conda/anaconda3/envs/mv2d/bin/python eval_zc_sim3.py $k
done
```

---

## 4. D. 自测（待满训）

- `eval_zc_sim3.py` 已支持 `POSE_TAG=enu` → `data/zc_*_enu`
- 场景：062224 / 081533 / 0923，`EVAL_PARITY=odd`
- 报告 terr/rerr，**口径标签 = ENU/RTK**
- 对照 Phase1 `/map` 表（见 `ZC_Phase1_ZC-only微调_20261022.md` §3.1）
- 可选：对 res npz 再跑 `eval_rtk_rescore` 得 along/cross cm（pred 已在 ENU 时 ep 应接近 sim3 terr）

---

## 5. 改动文件

| 路径 | 说明 |
| :--- | :--- |
| `relopt/scripts/gt_probes/rewrite_pose_enu.py` | **新建** ENU pose 旁路 |
| `relopt/scripts/gt_probes/regen_pairs_megaloc.py` | `DATA_TAG=enu` → zcME_/zc9ME_ |
| `reloc3r-grok-bot/scripts/finetune_zc_only_enu.sh` | **新建** ENU 微调入口 |
| `reloc3r-grok-bot/eval_zc_sim3.py` | `POSE_TAG=enu` / DATA_ROOT_* |
| `reloc3r/data/zc_fix_enu/`、`zc_0923_enu/` | 旁路数据（symlink 图 + 新 pose） |
| `reloc3r/_db-q_pair_info/zcME_*`、`zc9ME_*` | 新 pair npy |

---

## 6. 未解决问题（≤5）

1. 满训未完成 → 自测数字待补；完成后更新本文「收工」节。
2. 草坪 ENU 与 `/map` 相对 GT 几乎相同 → Phase2 训练收益主要在**绝对验收尺子统一**，相对监督改善有限（需在收工对照表中写明）。
3. RTK 拟合 resid p50≈5–7 cm + 跨包 NN≈15–20 cm → **cross&lt;3cm 可能不可达**；勿宣称 cm 达标。
4. `/data` 磁盘约 40G 可用；满训多 ckpt 需盯空间（勿删原始数据，可清 `_smoke_*`）。
5. `off_rad` 未写入 R；若后续要双天线航向绝对对齐，需单独开关并重评。

---

## 7. 收工占位

（满训 + 三场景 ENU 评测完成后填写：best epoch、terr/rerr 表、与 Phase1 `/map` 对照、结论。）
