# ZC Phase 1：ZC-only 微调（收工）

- 日期：2026-10-22（Asia/Shanghai）
- 状态：**满训完成 + 三场景自测完成**
- 承接：`ZC_Phase0盘点_20261022.md`、审核稿 `ZC仅自采数据_Reloc3r处理与训练方案_审核稿_20261022.md`

---

## 1. 分支与改动

| 项 | 值 |
| :--- | :--- |
| worktree | `/data/code/reLocation-all/reloc3r-grok-bot` |
| 分支 | `grok-bot` |
| commit | `13dcfe0c4bff43de0461d411c4955053cf78b79a`（基线 `46a7752`） |
| 训中改动 | `scripts/finetune_zc_only.sh`（新）、`reloc3r/datasets/zc_retrieval.py`（`data_root`）、`scripts/eval_forgetting.py`（`ZC_ONLY=1`） |
| 收工补丁 | `eval_zc_sim3.py`：有 `WEIGHTS` 时本地 `Reloc3rRelpose(img_size=512)` + `load_state_dict`，避开 HF `from_pretrained`（本机 huggingface.co 超时） |
| 未 push | 是（仅本地） |

---

## 2. 满训结果

```bash
PAIRSET=M bash scripts/finetune_zc_only.sh
# OUT_DIR=checkpoints/zc_finetune_ZC_only  EP=30  N=4000×3  freeze_encoder
```

| 项 | 值 |
| :--- | :--- |
| 启动 / 结束 | 2026-10-22 15:26:35 → **22:03:45** Asia/Shanghai |
| 墙钟 | **Training time 6:37:04** |
| bash / python PID | `90451` / `90454`（已退出） |
| 日志 | `/home/yr/grok-bot-work/logs_zc_only/full_train_20261022_152635.log` |
| 末 train loss | Epoch29 avg ≈ **0.0489** |
| **best** | **Test Epoch 29**（val loss med≈**0.100** / avg≈**0.180**），`checkpoint-best.pth` @ 21:50（2.6G） |
| 其它 ckpt | `checkpoint-{10,20,30,last,final}.pth` 均写出 |
| 峰值显存 | max mem ≈ 17698 MiB，无 OOM |
| 预训练 | `checkpoints/reloc3r-512.pth`；无公开集混训 |

> best 按训练侧 val **median** 刷新；Test30 med≈0.111 未刷新。

---

## 3. 三场景自测（部署口径）

协议：`SIM3_K=5 SIM3_BM3=1 EVAL_PARITY=odd FT_TAG=zcOnly`，权重=`checkpoints/zc_finetune_ZC_only/checkpoint-best.pth`。

```bash
cd /data/code/reLocation-all/reloc3r-grok-bot
PY=/data/software/conda/anaconda3/envs/mv2d/bin/python
W=checkpoints/zc_finetune_ZC_only/checkpoint-best.pth
for k in 062224 081533 0923; do
  SIM3_K=5 SIM3_BM3=1 EVAL_PARITY=odd WEIGHTS=$W FT_TAG=zcOnly \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    $PY eval_zc_sim3.py $k
done
```

评测日志：`/home/yr/grok-bot-work/logs_zc_only/phase1_eval_three_scenes_20261022_221428.log`  
产物：`reloc3r/data/zc_sim3/res_{062224,081533,0923}_sim5_bm3_zcOnly_odd.npz`（worktree `data/zc_sim3/` 同步有一份）。

### 3.1 主表（terr / rerr = mean / med / p90；单位与历史报告一致：平移 m、旋转 °）

| 场景 | 权重 | n | terr mean/med/p90 | rerr mean/med/p90 |
| :--- | :--- | ---: | :--- | :--- |
| 草坪同天 062224 | 官方 | 338 | 1.012 / 0.203 / 1.654 | 1.480 / 1.291 / 2.543 |
| 草坪同天 062224 | mix50 | 338 | 0.076 / 0.047 / 0.113 | 0.459 / 0.408 / 0.729 |
| 草坪同天 062224 | **zcOnly** | 338 | **0.070 / 0.053 / 0.121** | **0.443 / 0.406 / 0.735** |
| 草坪跨天 081533 | 官方 | 411 | 0.599 / 0.205 / 1.094 | 2.396 / 2.388 / 3.454 |
| 草坪跨天 081533 | mix50 | 411 | 0.246 / 0.052 / 0.133 | 0.445 / 0.370 / 0.787 |
| 草坪跨天 081533 | **zcOnly** | 411 | **0.195 / 0.051 / 0.110** | **0.437 / 0.362 / 0.753** |
| 大楼 0923 | 官方 | 407 | 0.268 / 0.111 / 0.539 | 0.831 / 0.660 / 1.733 |
| 大楼 0923 | mix50 | 407 | 0.110 / 0.072 / 0.214 | 0.313 / 0.253 / 0.572 |
| 大楼 0923 | **zcOnly** | 407 | **0.106 / 0.074 / 0.207** | **0.415 / 0.239 / 0.817** |

相对官方：三场景 terr/rerr 全面改善。相对 mix50：平移整体持平或略好（跨天 081533 p90 0.110 vs 0.133）；大楼 rerr p90 略差于 mix50（0.817 vs 0.572），med 仍略好（0.239 vs 0.253）。relpose ≈ **160 ms/帧**。

### 3.2 口径红线（必读）

- 上表是 **MegaLoc top-K + Reloc3r relpose + bm3** 相对 `/map` pose 的部署差分，**不是** RTK+ENU 统一后的绝对沿轨/跨轨验收。
- **禁止**宣称跨包 `/map` 已达沿轨 &lt;10 cm / 跨轨 &lt;3 cm / 角 &lt;0.2°；该尺子需 Phase2 ENU 重写后再验。
- 本 Phase 仍用 `/map` pose；跨包 `/map`  alone 不是厘米级细尺子。

---

## 4. 训练规模（Phase0 复核，未变）

- MegaLoc even 唯一 query：1089；候选对：10832
- 每 epoch 采样槽：12000（有放回）；无公开集

---

## 5. 未解决问题 / 后续（≤5）

1. Phase2：全 bag 相机位姿统一到同一 ENU（RTK+lever-arm），再报沿轨/跨轨/角绝对指标。
2. 大楼 0923 的 rerr p90 相对 mix50 偏高，可择机扫 LR/epoch 或短续训对照。
3. worktree 内 `croco` typechange、`data/checkpoints/_db-q_pair_info` symlink 仍未提交。
4. 可把本收工 md + `eval_zc_sim3.py` 离线加载补丁正式 commit 进 `relopt`/`reloc3r-grok-bot`（未 push）。
5. 遗忘评测（ScanNet/MegaDepth）未在本 Phase 跑；若需旁注官方对照，事后手动 `eval_forgetting`（勿设 `ZC_ONLY` 硬跳过时另议）。

---

## 6. 验收结论

| 项 | 结果 |
| :--- | :--- |
| 满训 30 epoch 正常结束 | ✅ |
| `checkpoint-best.pth` 可用 | ✅（Test29） |
| 三场景 odd+sim5_bm3 自测 | ✅ |
| 相对官方全面改善 | ✅ |
| 相对 mix50 平移竞争力 | ✅（跨天略优）；旋转大楼 p90 仍弱一截 |
| 绝对 cm / 0.2° 验收 | ❌ 未做（口径不允许用本表宣称） |
