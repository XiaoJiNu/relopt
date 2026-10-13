# Reloc3r 与 XFeat 模型训练与数据方案

（2026-10-09。面向综合排名表第 2 行 **Reloc3r+MegaLoc+sim5+bm3（本仓新增，缓存）** 与第 7 行
**XFeat 稀疏 snap** 的模型侧训练需求。依据 = `docs/汇总/优化实施记录_202610081555.md` §5、
`docs/算法优化/量产重定位方案问题分析与优化路线_202610081143.md` §2–§3，
以及两个仓库训练代码的实际审读（reloc3r/train.py + datasets/、accelerated_features/modules/training/train.py）。
所有路径均为本机实测路径；「未来命令」= 尚未执行，需按标注补前置条件后运行。）

## 结论先行（判定）

| 问题                  | Reloc3r（sim5+bm3 行的 relpose 模型）                                                                                                                                                 | XFeat                                                                                                          |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| 训练代码是否已有      | ✅ 仓库自带`train.py`，官方脚本 `scripts/train_small.sh`（单卡 3090 可跑）                                                                                                        | ✅ 仓库自带`modules/training/train.py`（`--training_type xfeat_synthetic` 可纯合成训练）                   |
| ZC 数据适配层是否已有 | ✅**已写好**：`reloc3r/datasets/zc_retrieval.py` 的 `ZcRelpose`（评测在用，训练只差一行注册）                                                                               | ❌ 需新写 ~150 行 Dataset（MegaDepth npz 格式或合成 warp 直读 ZC 帧）                                          |
| 需要多少数据          | 微调：**≥5k 对**即可启动，目标 2–5 万对；现有 ZC 已产 **1.5 万对**                                                                                                      | 合成 warp：**0 标注，≥2k 张图**；MegaDepth 监督路径：需深度图，≥5k 对                                  |
| 需要人工标注吗        | **完全不需要**。监督信号 = 相机位姿对，ZC 链已由 /map 融合位姿+标定自动生成（`eval_zc.py extract`）                                                                           | **完全不需要**。合成 warp 自监督（homography 已知）；深度监督路径的深度来自深度相机/LiDAR 投影，位姿同上 |
| 标签内容              | 每帧：图像 + 内参 K +`T_map_cam`（c2w 4×4，米制）；成对监督由 loss 现场算相对位姿                                                                                                  | 每帧：图像 +（可选）深度图 + K + c2w；或纯图像（合成 warp 时对应点在线生成）                                   |
| 主要风险              | ① CC BY-NC-SA 4.0**非商用许可**（量产必须换路线，见 §5）；② ZC 仅 2 场景，纯 ZC 微调会灾难性遗忘公开域能力，需混公开数据；③ A2 along-track 尺度问题是架构性的，微调只能缓解 | 许可 Apache 2.0 干净；ZC 域微调收益待验证（当前短板 B1 旋转尾部已被 rc 修复+EKF 消化，训练优先级低于 Reloc3r） |

优先级判定：**要训练先训 Reloc3r**（sim5+bm3 行的 ep p90 1.11 m 尾部、A2 along-track 误差都压在 relpose
回归头上；XFeat 的已知问题全部是无成本的口径/组合修复，且已有 EKF 融合兜底）。MegaLoc **不可训**
（官方 README 明确不放训练代码与数据集，`gmberton/MegaLoc` issue #5/#9），sim5+bm3 行里它只做检索，
检索 hit 率已 99%，不是瓶颈。

---

## 1. sim5+bm3（缓存）行的结构：哪部分可训、哪部分不可训

该链路 = MegaLoc 检索 top-5 预筛 → Reloc3r relpose 回归（encoder 缓存）→ base-max3 位姿投票。三个组件：

| 组件                       | 占比（端到端 ~107 ms） | 可训练性                                                                              |
| -------------------------- | ---------------------- | ------------------------------------------------------------------------------------- |
| MegaLoc 全局描述子（检索） | 37 ms                  | ❌ 官方不放训练代码/数据。替代：EigenPlaces 同门有训练链，或 GSV-Cities 自训（§4.4） |
| Reloc3r relpose 回归       | 69.7 ms（缓存后）      | ✅**训练主战场**。仓库 `train.py` + DUSt3R 数据框架齐全                       |
| bm3 投票 / sim5 预筛       | ≈0（查表）            | 无参数，纯策略层                                                                      |

encoder 缓存（P0-2）只是推理加速（db 侧特征预计算，数学恒等已证 0.00e+00 差），
训练时不需要它——训练用官方 `train.py` 的标准双图前向。

### 1.1 Reloc3r 的训练监督是什么（决定标签格式）

`reloc3r/loss.py::RelativeCameraPoseRegression`：输入一对图 (view1, view2)，各带
`camera_intrinsics`（3×3）与 `camera_pose`（**c2w 4×4，米制**），GT 相对位姿由 loss 现场计算
`inv(pose1) @ pose2`，损失 = 平移**方向角**误差 + 旋转角误差（L21 变体）。

→ 标签就是每帧一个位姿，**没有任何逐像素/逐点人工标注**。这与 ZC 现有数据链完全同构：
`reloc3r/data/zc_fix|zc_0923/<bag>/seq-01/frame-XXXXXX.{color.png,pose.txt}` 正是
（去畸变 640×480 图 + c2w 4×4 map 系位姿），pose 由 `eval_zc.py extract` 从 mcap 的 `/map`
融合位姿 + 定稿外参链自动导出（`T_map_cam = T_map_lidar @ R_BL @ T_cam_lidar`；R_BL=雷达系→车体系旋转，
四路证据定稿见 `reloc3r/eval_zc_qwen.py` 头注。旧 glm 链漏乘 R_BL、位姿差 ~2 m，其产物目录
`data/zc/` 已于 2026-10-10 删除，`eval_zc.py extract` 已内嵌定稿链，现全树唯一口径 = `data/zc_fix|zc_0923`）。

### 1.2 ZC 训练对已经存在（零新代码即可开训）

- `reloc3r/datasets/zc_retrieval.py::ZcRelpose`：从 `_db-q_pair_info/zc_<q>_q-retrieval_db-step=1_topk=10.npy`
  读 (query, db) 对，现场读 png+pose.txt——**这个类当前被评测脚本用，训练只需在
  `reloc3r/datasets/__init__.py` 加一行 import 注册**（实测确认：目前 `__init__.py` 未注册，
  `train.py --train_dataset` 字符串 eval 找不到 `ZcRelpose`）。
- 现有对数（实测 `_db-q_pair_info/`）：
  - 081533 跨天查询 × 064638 db：**823 query × top10 = 8 230 对**
  - 062224 同天查询 × 064638 db：**676 × 10 = 6 760 对**
  - 合计 **≈1.5 万对**，且是检索命中对（有共视、贴近部署分布——比随机对质量高）。
- 0923 大楼场景的对文件**已存在**（2026-10-10 实测更正，原文档"缺口"判断作废）：
  `_db-q_pair_info/zc9_20260923_060415_..._q-retrieval_db-step=1_topk=10.npy` = 815 query × top10
  = **8 150 对**。三场景合计 ≈ **2.31 万对**，全部可直接用于训练。
  若需换 topk/db_step 重生成 = 复用 `eval_zc.py` 的 TopkRetrieval 段（GPU ~10 min/场景）。

**关键口径提醒（决定这些对是"训练"还是"考试"）**：当前 2.31 万对与评测同分布同数据。
直接拿它们微调再在原三场景评测 = 数据泄漏，测出的提升不可信。正确做法二选一：
(a) **按 bag 切分**：081533 的 8 230 对全部用于训练，评测只报 062224/0923（跨天场景变 held-out）；
(b) **采新 bag**：训练数据来自新采集行程（§3 采集方案），现有三场景永远只做评测。
量产路线必须 (b)；快速验证可行性可先 (a)。

---

## 2. Reloc3r 训练：数据、量级、命令

### 2.1 数据格式（每个场景一份，`ZcRelpose` 已实现）

```
data/zc_train/<bag>/seq-01/frame-XXXXXX.color.png   # 去畸变 RGB（640×480，ZC 现口径）
data/zc_train/<bag>/seq-01/frame-XXXXXX.pose.txt    # 4×4 c2w，map 系，米制
_db-q_pair_info/zc_<q>_q-retrieval_db-step=1_topk=10.npy   # 检索对列表（query_path/db_path）
```

内参不用逐帧存：ZC 前视针孔 K 全包恒定（`zc_retrieval.py` 已内置 INTRINSICS，来自 bag 内
camera_info 实测核对）。换新平台/新相机时把该常量改成按 bag 读 calib 目录。

### 2.2 需要多少数据

| 档位               | 对数                                          | 依据                                                                        | 预期                                                                                                   |
| ------------------ | --------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| 可行性验证         | 现有 1.5 万（按 §1.2(a) 切分后 ~8k 训练）    | `train_small.sh` 官方样例就是 10k ScanNetpp 对单卡跑通                    | 能跑完、能看 terr/rerr 曲线是否下降；不承诺 ZC 指标提升                                                |
| 域微调（推荐起点） | 2–5 万对，≥3 个场景/db 包、含跨光照对       | 域微调经验量级；DUSt3R 系冻结 encoder 只训 decoder+head，参数少、数据需求低 | ZC ep p90 尾部（当前 1.11 m）与 A2 along-track 有望收窄；**必须混 20–50% 公开数据防灾难性遗忘** |
| 官方全量复现       | 35 万对/epoch × 100 epoch（7 数据集 × 50k） | `scripts/train.sh`：8×H800                                               | 不在本方案范围（单卡 5090 24G 不现实，也无必要——我们不是从零训）                                     |

单卡 RTX 5090 Laptop 24G 预算（`train_small.sh` 口径，batch 8、freeze_encoder、512×384）：
1.5 万对 ≈ 1 900 iter/epoch，100 epoch 数天量级；**微调 20–30 epoch 足够**（域适配不是从头学）。

### 2.3 训练命令（未来命令，未执行）

前置：① `reloc3r/datasets/__init__.py` 注册 `from .zc_retrieval import ZcRelpose`；
② 下载 DUSt3R 预训练权重 `checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth`（HF
`naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt`，~1.4 GB；注意当前 /data 只剩 53G，权重放得下，
MegaDepth 500G 放不下——§2.4）；③ conda env `mv2d`。

```bash
# 混合微调 = ZC 对 + 公开域（ScanNetpp 需先跑 datasets_preprocess/preprocess_scannetpp.py）
cd /data/code/reLocation-all/reloc3r
python train.py \
  --train_dataset "8_000 @ ZcRelpose(scene='20260921_064638', query_scene='20260920_081533', resolution=[(512,384)], transform=ColorJitter) \
                 + 4_000 @ ScanNetpp(split='train', resolution=[(512,384)], transform=ColorJitter)" \
  --test_dataset "1_000 @ ZcRelpose(scene='20260921_064638', query_scene='20260921_062224', resolution=(512,384), seed=777)" \
  --model "Reloc3rRelpose(img_size=512)" \
  --pretrained "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" \
  --lr 1e-5 --warmup_epochs 2 --epochs 30 --batch_size 8 --freeze_encoder \
  --output_dir checkpoints/zc_finetune
# 若从 reloc3r-512 官方权重续训（推荐，起点更高）：--pretrained 换成 HF 缓存里的
# siyan824/reloc3r-512 safetensors（需一次转 pth，from_pretrained 可读）
```

验收 = 两级：训练侧 terr/rerr 曲线下降且 ScanNetpp 测试集不崩（遗忘检查）；
产品侧用 `relopt/scripts/eval_zc_sim3_cache.py` 换 `--pretrained` 重跑三场景，
对比归档 `res_*_sim5bm3.npz` 六列（ep p90 1.68/1.10/0.56 → 目标 <1.2/0.9/0.5）。

### 2.4 公开域混入数据的选择（磁盘现实约束）

/data 只剩 53G，MegaDepth（500G）、DL3DV、CO3Dv2 全量不现实。按预处理体积排序：
**ScanNetpp**（`preprocess_scannetpp.py`，室内手持，与机器人视角差异大但量大质好）或
**RealEstate10K**（视频帧，~60G 压缩，720p 够用可只取子集）。二选一混 20–50% 即可——
混入的目的是锚住 decoder 的通用几何先验，不是复刻官方配比。

---

## 3. 数据采集与标注（两模型共用，ZC 平台）

### 3.1 采集 = 复用既有清单，训练无新增硬件需求

`topometric/docs/数据采集清单_重定位_20260920_093345.md` 已定稿（R2 平台：前视针孔 10 Hz +
RTK + 激光 + 激光内置 IMU + /odom + TF 全链）。**训练数据与评测数据用同一套采集规范**，
差别只在用途切分。训练专用增采建议：

| 项         | 要求                                                                                                                                            | 为什么                                                                                                                                                                                                                                                                                                  |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 场景数     | ≥3 个不同场地（现草坪+大楼=2，至少再补 1 个）                                                                                                  | 单场景微调必然过拟合场景纹理                                                                                                                                                                                                                                                                            |
| 每场景行程 | db 行程 1 条 + 查询行程 ≥4 条（同天/跨天/夜晚/逆光/雨后各一）                                                                                  | relpose 的域差鲁棒性靠"同地点不同外观"对喂出来；现有跨天对只有 1 条查询                                                                                                                                                                                                                                 |
| 帧率       | **图像 2 Hz 即可**（extract IMG_STEP=5，与现 db 库同粒度）；其它传感器照常（IMU ≥200 Hz、odom ≥20 Hz、LiDAR 10 Hz、/map、RTK、TF 全链） | 训练基线多样性来自选帧配对（2 Hz 下可配 0.57/1.1/5.7 m 任意基线），不来自录制帧率；10 Hz 多出的 0.1 m 级小基线恰是已证实的病态区（B4/A2），训练不需要。位姿标签由 /map（68.8 Hz）最近邻插值，与图像帧率无关。**例外**：留 1–2 条 10 Hz 图像查询包，专用于 10 Hz 在线 EKF 实测与离线 VIO 备份真值 |
| 单条时长   | 5–15 min（既有清单口径）                                                                                                                       | —                                                                                                                                                                                                                                                                                                      |
| 同步       | 相机-/map 时差 <100 ms（extract 已按此丢帧）                                                                                                    | 位姿标签质量 = 同步质量                                                                                                                                                                                                                                                                                 |

### 3.2 标注 = 全自动，零人工

「标签」只有两样，都来自传感器链而非人工：

1. **每帧 c2w 位姿**（两模型共用的核心标签）：`T_map_cam = T_map_lidar(/map 融合位姿) @ R_BL @ T_cam_lidar(标定)`（定稿链，含雷达→车体换轴；旧漏乘 R_BL 的链已删）。
   工具已有：`reloc3r/eval_zc.py extract`（mcap → png+pose.txt，含去畸变与同步丢帧），逐 bag 跑即可。
   标定链是四路证据定稿的既有资产（自研数据测试文档 §链路）。
2. **训练对列表**（哪些 (query, db) 组成一对）：MegaLoc/NetVLAD 检索 top-k 自动生成
   （`reloc3r/image_retrieval/topk_retrieval.py`，评测已在用）。不需要人工挑对。

**如果没有 /map（RTK 融合位姿）怎么办**——退级顺序：
(a) RTK 原始 fix + LIO 离线融合（采集清单里的备份真值路线，质量≈/map）；
(b) LiDAR SLAM 离线建图位姿（无 RTK 锚定则尺度仍米制、绝对位置漂，对 relpose **相对位姿**监督够用——
loss 只吃 inv(pose1)@pose2，全局原点无关）；
(c) 都不行 = COLMAP/离线 SfM 对 db 行程重建（成本最高，最后手段）。
XFeat 的深度监督路径另需**每帧深度**：优先 LiDAR 投影（权威米制，`AirSLAM/rosless/extract_zc.py`
已有 depth-source=lidar 的投影实现可搬），次选深度相机原始流（/camera_01/depth，bag 里有）。

### 3.3 数据量换算（采多少 bag 够）

一条 10 min @ 2 Hz 图像 bag = 1 200 帧（mcap 体积也随图像帧率降 ~5×，/data 97% 满的现状下更现实）；每查询帧取 top-10 = 一条查询 bag 产 ~1.2 万对。
**每场景 1 db + 4 查询行程 ≈ 5 万对**，已超 §2.2 的 2–5 万目标（对多不怕，采样上限控制）。
真正稀缺的是**场景数与条件覆盖**，不是帧数——3 场景 × 5 行程（每场景 1 db + 4 条件查询）
即够一轮认真的域微调；存储 ~60–100 G mcap/包（对照现有 10 Hz 录制的 5 包 302 G，2 Hz 图像后降数倍），/data（97% 满）清出空间即可。

---

## 4. XFeat 训练：数据与两条路径

### 4.1 官方训练配方（仓库自带，Apache 2.0）

`modules/training/train.py`，`xfeat_default` = **60% MegaDepth 监督 + 40% COCO 合成 warp 自监督**：

- 合成 warp 路径：从图像目录（默认 coco_20k，2 万张普通照片即可）在线生成
  homography/TPS 形变对 + FAST 关键点 + 解析 warp 对应点（`modules/dataset/augmentation.py`）——
  **标签是变换矩阵本身，零标注**。
- MegaDepth 路径：LoFTR 格式 scene_info npz（`pair_infos`(含 central_matches)、`intrinsics`、
  `poses`、`depth_paths`），对应点由 `megadepth_warper.spvis_coarse` 用深度+相对位姿在线投影生成
  ——**标签 = 深度图 + 位姿，同样零人工**。全量 MegaDepth ~500 G（本机放不下，见 §4.3 替代）。

监督信号本质：稠密关键点对应（8× 下采样格点上的 match/非 match 二分类 + 描述子对比），
不是位姿回归——所以 XFeat 训练**不直接吃 pose.txt**，吃的是"对应点"，对应点由
（位姿+深度）投影或（合成变换）解析产生。

> 疑问（2026-10-09 批注）：XFeat 训练这里训练用的关键点是什么？这里的合成 warp 路径和 MegaDepth 路径，它们最终训练的标签是一样的吗？但是这里好像不一样？

**答**：分两类"关键点"，两路径**最终喂进 loss 的标签同构，只是生成方式不同**——这就是"看起来不一样"的来源。

1. **主监督 = 稠密对应点 `positives_c`**（不是人工标的关键点，是"同一 3D 点在两图里的像素坐标对"）。统一格式 `[K, 4]`：前两列 = 图1 的 (x,y)，后两列 = 图2 的 (x,y)，都在 8× 下采样的 coarse 格点坐标（`train.py:224` `pts1, pts2 = positives_c[b][:,:2], positives_c[b][:,2:]`）。两条路径产的都是这个格式，`train.py:200` 直接 `positives_c = positives_s_coarse + positives_md_coarse` 拼成一个 list 混训，下游 4 个 loss（`dual_softmax` / `coordinate_classification` / `keypoint_loss` / `alike_distill`）对二者一视同仁——**标签含义与格式完全一样**。
   - 合成 warp 路径：`augmentation.py::get_correspondences` 用**已知的 homography/TPS 变换矩阵 H** 把网格点从图2 解析 warp 回图1 → 对应点。标签来自"我自己施加的 2D 变换"，零深度零位姿。
   - MegaDepth 路径：`megadepth_warper.py::spvis_coarse` 用**深度图 + 相对位姿** `warp_kpts` 把图1 网格点反投影到 3D、变换、再投到图0，做双向重投影一致性检查（`dist < 1.5`）→ 对应点。标签来自"3D 几何投影"。
   - 一句话：**来源不同（2D 变换 vs 3D 投影），产物同构（都是 (pt1,pt2) 对应点对）**。差异在"怎么算出这对点"，不在"标签长什么样"。
2. **辅助监督 = ALIKE 蒸馏关键点**（`losses.py::alike_distill_loss`）：用 ALIKE 检测器在**同一张图**上现提取关键点当 keypoint-位置的真值（65 类分类：64 亚像素偏移 bin + 1 非关键点）。这条与走哪条路径无关，两路径共用，是对当前 batch 图像在线算的。

→ 所以：训练"关键点"= ①对应点（主，两路径同构）②ALIKE 蒸馏点（辅，路径无关）。§4.2 路径 A 只提供①的合成版，路径 B 提供①的几何版，两者可与官方 xfeat.pt 权重无缝混训。

### 4.2 ZC 域微调的两条路径（按成本排序）

**路径 A（先做，零标注零深度）：ZC 帧当合成 warp 底图。**
把 `reloc3r/data/zc_fix/*/seq-01/*.color.png`（~3.9k 帧， Lawn+大楼纹理都在）软链到一个目录当
`--synthetic_root_path`，`--training_type xfeat_synthetic` 直接开训。
局限（要如实预期）：单应/TPS 形变 ≠ 真实 3D 视角变化，学到的是光照/纹理鲁棒性，
**不会改善大视差匹配**；对 B 系列已知短板（B1 旋转尾部已被 rc 修复）收益有限，属于低成本试错。

```bash
# 未来命令：ZC 合成 warp 微调（~1 天/160k step 单卡）
cd /data/code/reLocation-all/accelerated_features
mkdir -p data/zc_frames && ln -s ../../reloc3r/data/zc_fix/*/seq-01/*.color.png data/zc_frames/
python3 -m modules.training.train --training_type xfeat_synthetic \
  --synthetic_root_path data/zc_frames --ckpt_save_path ckpts/zc_synth \
  --n_steps 40000 --batch_size 10
# 验收：scripts/eval_zc_xfeat.py 换 --weights 重跑三场景 vs 归档 zc_xfeat_*.json
#（可解率 99.9–100% 已近满，主要看内点数与 ep med 0.23 是否下探）
```

**路径 B（认真做，需深度）：ZC 版 MegaDepth-format npz。**
每帧补 LiDAR 投影深度（§3.2(c)）后，写一个 ~150 行的 npz 生成器：
`image_paths/depth_paths/intrinsics/poses` 直接来自 zc_fix 布局，`pair_infos` 用检索 top-k 对 +
`warp_kpts` 预计算 overlap_score 与 central_matches（`megadepth_warper.py` 函数可整体复用）。
生成的 npz 放 `data/zc_md/`，`MegaDepthDataset(root_dir=..., npz_path=...)` 即可直接加载，
训练循环零改动。数据量：≥5k 对起步（合成:ZC = 4:6 混训防遗忘，同 §2.2 逻辑）。

> 疑问（2026-10-09 批注）：如果机器人没有 Lidar，是否可以用高精度的深度估计模型用于得到深度图来作为训练的标签？

**答**：**技术上能跑通，但要分清"深度当什么用"——当 XFeat 的几何对应点标签可以（有兜底），当度量真值不行。且 ZC 场景有个已实测的硬伤，务必先读。**

1. **能用的地方**：XFeat 路径 B 里深度只进 `warp_kpts` 生成对应点，而 `spvis_coarse` 自带**双向重投影一致性检查**（`dist < 1.5 px` 才保留）。深度估计模型的误差会直接反映为重投影不一致 → 坏对应点被自动丢弃。所以用 DA3/Depth-Anything/Metric3D 的伪深度当标签，**质量差的点会被过滤，不会污染训练**——这是它比"当度量真值"安全的根本原因。深度只需**局部相对形状大致对**（相邻像素的相对远近），不需绝对米制精度。
2. **ZC 场景的已知硬伤（必须先说）**：本仓 topometric 复现的既有结论是——**DA3 公开权重在小基线平移下失效**（自研数据测试文档 §偏差与修复、优化路线 A2；对齐塌缩到参考质心）。XFeat 路径 B 的对是"检索命中的 db-query 对"，ZC db 帧距 0.57 m、跨天对基线更小，**正落在 DA3 失效区**。伪深度在这些对上产出的对应点会系统性偏，一致性检查能滤掉一部分，但留下的也带偏差。
3. **务实排序（无 LiDAR 时）**：
   - **首选**：ZC 平台若有**深度相机**（bag 里 `/camera_01/depth` 就有，AirSLAM 复现已用它做虚拟双目）——直接取深度相机原始流，比任何单目估计都准，且是传感器实测非模型猜测。
   - **次选**：多帧 **SfM/COLMAP 稀疏重建**深度（离线、准、慢），或双目/多目三角化深度。
   - **兜底才用单目深度模型**：且优先**度量深度模型**（Metric3D v2 / UniDepth / Depth-Anything-**Metric**）而非相对深度（DA3 属相对，尺度自由，ZC 小基线又失效）。用前必须在 ZC 几帧上与深度相机/LiDAR 投影对比验证尺度与形状，不能盲信。
   - **或者干脆绕开深度**：走 §4.2 **路径 A**（合成 warp）——零深度也能训，只是学不到大视差匹配。**若无 LiDAR 且深度相机也不可靠，路径 A + Reloc3r 位姿微调（§2，位姿标签不吃深度）是更稳的组合**，把几何监督的担子交给 Reloc3r（它直接回归相对位姿，本就不需要逐帧深度图当输入标签）。
4. **一句话结论**：单目深度估计可作 XFeat 路径 B 的**兜底**标签（一致性检查兜住质量），但在 ZC 小基线 + DA3 已知失效的双重不利下，**优先级排在深度相机、SfM 之后**；真要无深度，就把重心放 Reloc3r 位姿微调 + XFeat 路径 A，不硬凑伪深度。

### 4.3 是否要用公开 MegaDepth 从头训？——不要

从零训 XFeat（160k step 官方配方）需要 500 G MegaDepth + COCO 20k，本机磁盘不够，
且官方权重 `weights/xfeat.pt` 已在 MegaDepth-1500/ScanNet-1500 复现中对齐论文 Table 1/2
（本仓既有结论）——公开域能力已验证，缺的只是 ZC 域适配。微调（路径 A/B）即正解。
若未来必须补公开域数据，ScanNet-1500 已在本机（`data/scannet1500/`，2.5 G）可作验证集。

### 4.4 MegaLoc（检索头）想训怎么办

官方无训练代码。若检索真成瓶颈（当前 hit 99%，不是），替代品 = **EigenPlaces**（MegaLoc 同作者
前身，训练链完整）或 GSV-Cities 框架自训 DINOv2+聚合头；训练标签 = GPS 位置（地点分类/度量学习），
ZC 场景可由 /map 位姿自动产地点标签——同样零人工。仅备忘，不建议现在做。

> 疑问（2026-10-09 批注）：MegaLoc 的训练标签是什么，监督信号是什么？机器人没有 GPS，只有 1 个相机、IMU、轮速计等传感器，应该怎么办，是否可以用定位得到的相机在 map 坐标系下的信息来作为标签？

**答**：**可以，而且你的 /map 位姿比 GPS 更适合**。但要先纠正一个概念——这类模型的标签**不是坐标值本身（不是回归 xy），而是"地点身份"（place identity：哪些图属于同一地点）**。

1. **监督信号 = 地点级度量学习（place recognition）**。MegaLoc/EigenPlaces/SALAD 同门，官方都用 **GSV-Cities** 训：它把街景图按 **GPS 距离聚类成 ~562 个 "place"，每 place ~10 张图**；训练时**同 place 的图互为正样本、不同 place 为负样本**，用 MultiSimilarity/对比损失把全局描述子按地点拉开。→ **GPS 只是"产生 place 标签"的一种手段，不是模型输入，也不是回归目标**。模型学的是"同一地点不同外观 → 描述子相近"。
2. **没有 GPS 完全不影响**——place 标签的本质是"把图像按物理位置分组"，任何能提供**米制全局位置的源**都行：
   - 你的 **`/map` 融合位姿（RTK+LIO 锚定）就是最优源**：室内室外都能用、无 RTK 遮挡丢星问题、精度 0.4 m 级远好于民用 GPS。
   - 退级同 §3.2：无 /map 时用 LIO/SLAM 离线位姿（place 聚类只要**相对位置**分得开，全局原点无关，尺度米制即可）。
   - 单相机 + IMU + 轮速计本身给不了全局米制位姿（会漂），但**离线 VIO/LIO 融合**能产出足够聚类的轨迹——采集清单里的"自洽真值"路线就是干这个的。
3. **具体怎么产标签（复用本仓已有代码，~0 新算法）**：
   - topometric `scripts/build_map.py` 的 **HDBSCAN 已经把 map 位姿聚成"地点"了**（论文 §3.1，ZC 上实测 mcs=5→33 地点、mcs=12→9 地点）。**直接拿这套聚类输出的簇标签当 MegaLoc 的 place id**：同簇帧 = 同 place = 正样本对。
   - 或更简单：按 map 位置**网格化/半径聚类**（每 N 米一个 place），一行 sklearn 搞定，GSV-Cities 本质就是这么做的。
   - 帧 → place id 的映射表就是全部标签，**零人工**。
4. **两个必须满足的前提（否则训不出跨条件鲁棒性）**：
   - **同一地点要有多条件观测**：place recognition 的价值在"白天/夜晚/跨天/雨后同地点仍判为同 place"。单行程数据里一个 place 只有单一光照，模型学不到不变性。→ 正好呼应 §3.1 采集方案（每场景 db + 4 条件查询行程），**多行程同地点**是训练必需，不是可选。
   - **place 粒度要平衡**：簇太粗（一个 place 跨几十米）→ 正样本含不该匹配的远景；太细 → 同地点被拆成多 place 变负样本，互相打架。topometric 的 mcs 扫描（§关键发现1）已给出 ZC 经验值可参考。
5. **但仍维持 §4.4 的口径：现在别训 MegaLoc**。理由不变——检索 hit 率已 99%（自研数据测试文档），**不是瓶颈**；sim5+bm3 行的精度压在 relpose（Reloc3r）和几何上，不在检索召回。真要动检索，先用现成 EigenPlaces 权重换掉 MegaLoc 测一版（零训练成本），确认检索确实是短板后，再用上面这套 map-位姿→place 标签自训。**优先级：Reloc3r 位姿微调 > XFeat > MegaLoc 自训。**

> 补充对照（三模型标签一张表）：
> | 模型 | 监督信号 | 标签来源 | ZC 自动产标签的工具 |
> | ---- | ---- | ---- | ---- |
> | Reloc3r | 相对位姿回归（平移方向角+旋转角） | 每帧 c2w 位姿 | `eval_zc.py extract`（/map×标定） |
> | XFeat | 稠密对应点 + ALIKE 蒸馏 | 深度+位姿投影 / 合成变换 | `megadepth_warper` / `augmentation` |
> | MegaLoc | 地点级度量学习（place id） | 全局位置聚类 | `topometric/build_map.py` HDBSCAN（已有） |
> 三者标签**全部可由传感器链自动生成，无一需要人工标注**——差别只在"从位姿派生出什么"（相对位姿 / 对应点 / 地点簇）。

---

## 5. 许可与合规（量产前必须过的一关）

| 组件        | 许可                                        | 量产含义                                                                                                                                                                                                          |
| ----------- | ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Reloc3r     | **CC BY-NC-SA 4.0（非商用）**         | 微调产物同样受 NC 约束。量产二选一：① 用其思想自训（CroCo 预训练同为 NC，需从干净底座重起，成本高）；② 采购商业授权/找替代（如 DUSt3R 系商用授权的衍生）。**研究/内测阶段无碍，立项量产前必须法务过一遍** |
| XFeat       | Apache 2.0                                  | 干净，可商用可微调可闭源发布                                                                                                                                                                                      |
| MegaLoc     | 代码 MIT 但训练数据不可再分发、官方无训练链 | 推理使用合规；自训走 EigenPlaces/GSV-Cities                                                                                                                                                                       |
| ZC 自采数据 | 公司内部                                    | 训练产物含 ZC 数据信息，模型权重按内部资产管理，不入公开仓（本次推送的三私有仓口径一致）                                                                                                                          |

## 6. 执行顺序建议（最短路径）

1. **今天就可做（零采集）**：Reloc3r 注册 `ZcRelpose` → 用现有 8 230 对（081533）+ 062224 做
   held-out，按 §2.3 跑 30 epoch 微调，验证"ZC 域微调是否有效"这一根本假设。成本 = 1 行注册 +
   DUSt3R 权重下载 + 单卡数天。
2. **并行（零采集）**：XFeat 路径 A 合成 warp 微调（§4.2），~1 天，看 ep med 是否下探 0.23。
3. **假设 1 成立后再采集**：按 §3.1 补 1 个新场景 + 每场景 4 条件查询行程（图像 2 Hz 即可，另留
   1–2 条 10 Hz 查询包给在线 EKF 实测），走 §3.2 自动标注链，扩到 2–5 万对正式微调；同时用新 bag 的深度走 XFeat 路径 B。
4. **量产决策点**：Reloc3r 许可问题（§5）在步骤 3 之前必须有答案，否则训得越好沉没成本越大。

## 附：本文档事实来源索引

- 训练框架/损失/数据集类：`reloc3r/train.py`、`reloc3r/loss.py::RelativeCameraPoseRegression`、
  `reloc3r/datasets/zc_retrieval.py`、`reloc3r/datasets/__init__.py`（ZcRelpose 未注册，实测）、
  `reloc3r/scripts/train{,_small}.sh`、`reloc3r/README.md` §Training（8×H800/3090 口径）。
- XFeat 训练：`accelerated_features/modules/training/train.py`（xfeat_default/synthetic/megadepth 三档、
  60/40 混比）、`modules/dataset/augmentation.py`（合成 warp 全解析）、
  `modules/dataset/megadepth/{megadepth.py,megadepth_warper.py}`（LoFTR npz 格式、spvis_coarse）、
  `modules/dataset/download.py`（MegaDepth 500G 警示）。MegaLoc 不训：`MegaLoc/README.md` L44。
- ZC 数据现状（全部实测）：帧数 zc_fix 2 189 + zc_0923 1 709；对文件 823×10 + 676×10；
  mcap 5 包 302 G；/data 余 53 G；relopt 归档 `results/res_*_sim5bm3.npz`（含 preds/terr/rerr/names）。
- 基线指标：优化实施记录 §5 综合表（sim5+bm3 ep 0.57/0.17/1.11，eθ 1.58/1.46/2.59，~107 ms；
  XFeat snap ep 0.33/0.23/0.59，eθ p90 7.95）；A2 along-track p90 4.4 m = 优化路线 §2 表。
- 采集规范：`topometric/docs/数据采集清单_重定位_20260920_093345.md`（本文 §3 全部沿用其口径）。
