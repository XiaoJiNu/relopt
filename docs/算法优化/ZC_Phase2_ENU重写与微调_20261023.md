# ZC Phase2：ENU 旁路重写 + ZC-only 微调（已完成）

- **日期**：2026-10-23（Asia/Shanghai）
- **机器**：拯救者 `252f7d1f-95fc-44c6-94d1-8f76153ee6af`
- **依据**：审核稿 §Phase2；讨论结论「统一 ENU」；用户已批准继续 Phase2
- **红线**：不覆盖 `data/zc_fix` / `zc_0923` pose；不 sudo；不 push；绝对指标必须标 **GT=ENU/RTK**；诚实写 RTK 底噪可能挡 cross&lt;3cm；**勿宣称 cross&lt;3cm 已达标**

---

## 0. 开工状态

| 项 | 状态 |
| :--- | :--- |
| A 旁路 ENU pose | ✅ 完成 |
| B MegaLoc 对（ENU） | ✅ 完成 |
| C SMOKE 微调 | ✅ 通过（1 epoch） |
| C 满训 30 epoch | ✅ 完成（Training time 6:37:32，00:25→07:03 CST） |
| D 自测（ENU 口径） | ✅ 完成（三场景 `eval_zc_sim3` + §4.4 along/cross） |
| E 本文档 | ✅ 收工 |

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

### 3.3 满训（已完成）

| 项 | 值 |
| :--- | :--- |
| PID | 200373（已结束） |
| 日志 | `/home/yr/grok-bot-work/logs_zc_enu/finetune_zc_only_enu_20261023_002551.log` |
| 开始 | 2026-10-23 **00:25** CST |
| 结束 | 2026-10-23 **07:03** CST |
| 总时长 | **Training time 6:37:32** |
| GPU | RTX 5090 Laptop，训时约 11 GB |
| OUT | `/data/code/reLocation-all/reloc3r-grok-bot/checkpoints/zc_finetune_ZC_only_enu/` |

**best / final（训练侧 val loss；口径=相对位姿监督，非绝对 along/cross）：**

| ckpt | 时间 (CST) | 对应 | val loss med / avg |
| :--- | :--- | :--- | :--- |
| **checkpoint-best.pth** | **05:57** | **Test25**（log `epoch=25`；用户侧记 epoch≈24） | **med≈0.080814 / avg≈0.089811** |
| checkpoint-final.pth | 07:03 | epoch=30 | （权重落盘；末轮 Test30 见下） |
| checkpoint-30.pth / last | 07:02–07:03 | epoch=30 | — |
| 末轮 Test30 | 07:03 | epoch=30 | med≈0.083865 / avg≈0.090801（未刷新 best） |

复现训练：

```bash
cd /data/code/reLocation-all/reloc3r-grok-bot
bash scripts/finetune_zc_only_enu.sh
# 日志示例：/home/yr/grok-bot-work/logs_zc_enu/finetune_zc_only_enu_*.log
```

---

## 4. D. 自测（三场景 ENU，已完成）

- 脚本：`bash /home/yr/grok-bot-work/logs_zc_enu/run_eval_after_train.sh`
- 权重默认：`checkpoints/zc_finetune_ZC_only_enu/checkpoint-best.pth`
- `POSE_TAG=enu`，`SIM3_K=5`，`SIM3_BM3=1`，`EVAL_PARITY=odd`，`FT_TAG=zcOnlyENU`
- **GT 口径 = ENU/RTK**（`DATA_ROOT=.../zc_fix_enu` 或 `zc_0923_enu`）
- 评测日志：`/home/yr/grok-bot-work/logs_zc_enu/phase2_eval_three_scenes_20261023_070527.log`（exit 0，≈07:05–07:09 CST）

### 4.1 主表（原文照抄；terr/rerr = mean/med/p90；平移 m、旋转 °）

权重 = `checkpoint-best.pth`；tag = `sim5_bm3_zcOnlyENU_odd`：

| 场景 | n | terr mean/med/p90 (m) | rerr mean/med/p90 (°) | relpose |
| :--- | ---: | :--- | :--- | :--- |
| 草坪同天 062224 | 338 | **0.06 / 0.05 / 0.11** | **0.41 / 0.35 / 0.73** | 160 ms/帧 |
| 草坪跨天 081533 | 411 | **0.20 / 0.05 / 0.12** | **0.44 / 0.36 / 0.76** | 161 ms/帧 |
| 大楼 0923 | 407 | **0.06 / 0.04 / 0.10** | **0.38 / 0.21 / 0.79** | 158 ms/帧 |

原文行：

```
062224 sim5_bm3_zcOnlyENU_odd: n=338 terr mean/med/p90 = 0.06/0.05/0.11 | rerr mean/med/p90 = 0.41/0.35/0.73 | relpose 160 ms/帧
081533 sim5_bm3_zcOnlyENU_odd: n=411 terr mean/med/p90 = 0.20/0.05/0.12 | rerr mean/med/p90 = 0.44/0.36/0.76 | relpose 161 ms/帧
0923 sim5_bm3_zcOnlyENU_odd: n=407 terr mean/med/p90 = 0.06/0.04/0.10 | rerr mean/med/p90 = 0.38/0.21/0.79 | relpose 158 ms/帧
```

### 4.2 与 Phase1 `/map` zcOnly 对照（仅供参考；GT 坐标系不同，不可直接宣称绝对 cm 达标）

Phase1（GT=/map，见 `ZC_Phase1_ZC-only微调_20261022.md` §3.1）zcOnly terr med：062224 **0.053** / 081533 **0.051** / 0923 **0.074** m。  
Phase2（GT=ENU）terr med：062224 **0.05** / 081533 **0.05** / 0923 **0.04** m。相对量级接近；**本表不是 along/cross 绝对验收**，且 **勿宣称 cross&lt;3cm 已达标**。

### 4.3 复现评测

```bash
bash /home/yr/grok-bot-work/logs_zc_enu/run_eval_after_train.sh
# 或：
cd /data/code/reLocation-all/reloc3r-grok-bot
W=checkpoints/zc_finetune_ZC_only_enu/checkpoint-best.pth
for k in 062224 081533 0923; do
  POSE_TAG=enu SIM3_K=5 SIM3_BM3=1 EVAL_PARITY=odd WEIGHTS=$W FT_TAG=zcOnlyENU \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    /data/software/conda/anaconda3/envs/mv2d/bin/python eval_zc_sim3.py $k
done
```

ckpt 路径：

- best：`/data/code/reLocation-all/reloc3r-grok-bot/checkpoints/zc_finetune_ZC_only_enu/checkpoint-best.pth`
- final：`/data/code/reLocation-all/reloc3r-grok-bot/checkpoints/zc_finetune_ZC_only_enu/checkpoint-final.pth`

---

### 4.4 沿轨/横向平移误差（along-track / cross-track）

**定义（复用既有验收 / A2 口径，与 §4.1 terr 同套样本）**

- 误差向量：`e = dvec = pred[:3,3] − gt[:3,3]`，来自 `eval_zc_sim3` 在 `POSE_TAG=enu` 下的 **绝对位姿**（MegaLoc top-5 + bm3 motion averaging 之后），**非** pairwise 相对 `t` 再做 Sim(3) 对齐。
- 前进方向 `u`：GT 查询轨迹 **水平切向**（邻帧 cam xy 差分单位向量），与 `eval_rtk_rescore` / `gen_reloc3r_finetune_report.a2_split` / `docs/zc数据集上测试/tools/alongcross.py` 一致。
- `along = |dot(e_xy, u)|`；`cross = ||e_xy − signed_along·u||`（**仅水平面 2D**，不含竖直）。
- 说明：任务备选「相对 `t_gt` 单位向量」未采用——本评测 terr 是绝对 ep；验收 along&lt;10cm / 横向&lt;3cm 历来按轨迹切向分解绝对误差。

**命令（零 GPU，读已有 npz）**

```bash
/data/software/conda/anaconda3/envs/mv2d/bin/python \
  /data/code/reLocation-all/relopt/scripts/probe_along_cross_phase2_enu.py
# 产物：
#   /home/yr/grok-bot-work/logs_zc_enu/phase2_along_cross_20261023_071704.log
#   /home/yr/grok-bot-work/logs_zc_enu/phase2_along_cross_20261023_071704.json
# 输入 npz：reloc3r/data/zc_sim3/res_{062224,081533,0923}_sim5_bm3_zcOnlyENU_odd.npz
```

**三场景表（m；mean/med/p90；n_along = 有可用航向的帧）**

| 场景 | n / n_along | terr mean/med/p90 | along mean/med/p90 | cross mean/med/p90 (2D) |
| :--- | ---: | :--- | :--- | :--- |
| 草坪同天 062224 | 338 / 331 | 0.0641 / 0.0510 / 0.1088 | **0.0468 / 0.0320 / 0.0836** | **0.0272 / 0.0204 / 0.0584** |
| 草坪跨天 081533 | 411 / 397 | 0.1966 / 0.0477 / 0.1162 | **0.1268 / 0.0313 / 0.0925** | **0.1258 / 0.0174 / 0.0588** |
| 大楼 0923 | 407 / 401 | 0.0567 / 0.0374 / 0.0972 | **0.0471 / 0.0264 / 0.0855** | **0.0161 / 0.0111 / 0.0319** |

（附 cross3d 含竖直分量 mean/med/p90：062224 0.0349/0.0301/0.0670；081533 0.1330/0.0248/0.0659；0923 0.0226/0.0168/0.0382 — **验收对照用上表 2D cross**。）

**与验收目标对照（客观，不夸大）**

| 目标 | 观察 |
| :--- | :--- |
| 沿轨 &lt;10 cm | 三场景 **med** 均 &lt;10 cm（≈3.2 / 3.1 / 2.6 cm）；**p90** 亦均 &lt;10 cm（≈8.4 / 9.2 / 8.6 cm）。mean 在 081533 被少数大误差拉高（0.13 m）。 |
| 横向 &lt;3 cm | 三场景 **med** 约 2.0 / 1.7 / 1.1 cm；但 **p90** 约 **5.8 / 5.9 / 3.2 cm**，仍高于 3 cm。**勿宣称 cross&lt;3cm 已达标**。跨包 NN≈15–20 cm 与 RTK 拟合底噪仍可能限制绝对尺子。 |

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
| `relopt/scripts/probe_along_cross_phase2_enu.py` | **新建** Phase2 along/cross 探针（读 ENU npz） |

---

## 6. 未解决问题（≤5）

1. **勿宣称 cross&lt;3cm 已达标**：§4.4 已给出与 terr 同套样本的沿轨/横向（2D）；三场景 cross **med**≈2.0/1.7/1.1 cm，但 **p90**≈5.8/5.9/3.2 cm 仍高于 3 cm；RTK 拟合 resid + 跨包 NN≈15–20 cm 底噪仍可能挡绝对尺子。
2. 草坪 ENU 与 `/map` 相对 GT 几乎相同 → Phase2 训练收益主要在**绝对验收尺子统一**，相对监督改善有限；terr 表与 Phase1 对照仅供参考（GT 系不同）。
3. 未另跑完整 `eval_rtk_rescore`（需 cands dump / dualgate 变体）；当前 along/cross 直接分解 ENU `dvec`（pred 已在 ENU，与 rescore 主路径等价于跳过 w·p+o 变换）。若要比对 dualgate，需另开 rescore。
4. 大楼 0923 rerr p90≈0.79° 仍偏高（与 Phase1 zcOnly≈0.817° 同量级）；可择机扫 LR/epoch 或短续训。
5. `off_rad` 未写入 R；若后续要双天线航向绝对对齐，需单独开关并重评。

---

## 7. 收工

| 项 | 结论 |
| :--- | :--- |
| 训练 | ✅ 30 epoch 完成；时长 **6:37:32**（00:25→07:03 CST） |
| best | **Test25**，val med≈**0.080814** / avg≈**0.089811**，`checkpoint-best.pth` @05:57 |
| final | `checkpoint-final.pth` @07:03；末轮 Test30 med≈0.083865 / avg≈0.090801 |
| 评测 | ✅ 三场景 ENU exit 0；terr med ≈ **0.05 / 0.05 / 0.04** m；§4.4 along/cross 已记 |
| 口径 | **GT=ENU/RTK**；along/cross=绝对 dvec×GT 轨迹切向（2D）；**勿宣称 cross&lt;3cm 已达标** |
| 日志 | 训：`.../logs_zc_enu/finetune_zc_only_enu_20261023_002551.log`；评：`.../phase2_eval_three_scenes_20261023_070527.log`；沿轨：`.../phase2_along_cross_20261023_071704.log` |
