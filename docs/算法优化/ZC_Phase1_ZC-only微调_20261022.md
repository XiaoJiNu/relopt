# ZC Phase 1：ZC-only 微调（开训卡 / 进行中）

- 日期：2026-10-22（Asia/Shanghai）
- 状态：**满训进行中**（SMOKE 已通过）
- 承接：`ZC_Phase0盘点_20261022.md`、审核稿 `ZC仅自采数据_Reloc3r处理与训练方案_审核稿_20261022.md`

---

## 1. 分支与改动

| 项 | 值 |
| :--- | :--- |
| worktree | `/data/code/reLocation-all/reloc3r-grok-bot` |
| 分支 | `grok-bot` |
| commit | `13dcfe0c4bff43de0461d411c4955053cf78b79a`（基线 `46a7752`） |
| 改动文件 | `scripts/finetune_zc_only.sh`（新）、`reloc3r/datasets/zc_retrieval.py`（`data_root`）、`scripts/eval_forgetting.py`（`ZC_ONLY=1` 跳过公开集硬门禁） |
| 未 push | 是（仅本地 commit） |

---

## 2. SMOKE（已通过）

```bash
cd /data/code/reLocation-all/reloc3r-grok-bot
PAIRSET=M SMOKE=1 bash scripts/finetune_zc_only.sh
```

| 项 | 结果 |
| :--- | :--- |
| 配置 | N=200×3，EP=1，LR=1e-5，BS=8，freeze_encoder，amp，无 MegaDepth |
| OUT | `checkpoints/_smoke_zc_only/` |
| loss | 首步 0.327 → epoch 均 0.351；窗口 loss 末段约 0.31 |
| 显存 | max mem ≈ 17698 MiB，无 OOM |
| ckpt | `checkpoint-last/best/final.pth` 均写出 |
| 日志 | `/home/yr/grok-bot-work/logs_zc_only/smoke_*.log` |

---

## 3. 满训（进行中）

```bash
PAIRSET=M bash scripts/finetune_zc_only.sh
# OUT_DIR=checkpoints/zc_finetune_ZC_only  EP=30  N=4000×3
```

| 项 | 值 |
| :--- | :--- |
| 启动 | 2026-10-22 15:26:35 Asia/Shanghai |
| bash PID | `90451`（文件：`/home/yr/grok-bot-work/logs_zc_only/full_train.pid`） |
| python PID | `90454`（GPU ~11 GiB / peak train mem ~17.7 GiB） |
| 日志 | `/home/yr/grok-bot-work/logs_zc_only/full_train_20261022_152635.log` |
| Train length | 1500 iter/epoch（12000 samples / BS=8） |
| Epoch0 趋势 | loss 0.591 → ~0.28（窗口）；avg ~0.35；eta≈10 min/epoch 训练段 |
| **ETA** | ≈ **12–14 min/epoch × 30 ≈ 6–7 h** → 约 **22:00–23:00 Asia/Shanghai** 量级（含每 epoch odd 测试） |
| 预训练 | `checkpoints/reloc3r-512.pth` |
| 未覆盖 | `zc_finetune_mix50*` / `zc_finetune_M_mix50_lr5e6` 等旧目录 |

---

## 4. 满训后自测（待跑）

训完 `checkpoint-best.pth` 后：

```bash
cd /data/code/reLocation-all/reloc3r-grok-bot
PY=/data/software/conda/anaconda3/envs/mv2d/bin/python
W=checkpoints/zc_finetune_ZC_only/checkpoint-best.pth
for k in 062224 081533 0923; do
  SIM3_K=5 SIM3_BM3=1 EVAL_PARITY=odd WEIGHTS=$W FT_TAG=zcOnly \
    $PY eval_zc_sim3.py $k
done
```

对照已有产物（同协议 odd + sim5_bm3）：`data/zc_sim3/res_*_sim5_bm3_{official,ft,mix50,ftM}_odd.npz`。

**口径红线**：禁止宣称跨包/map 已达 cm / 0.2°；只报相对/部署差分与相对官方或旧 mix50 的对比。

---

## 5. 训练规模（Phase0 复核）

- MegaLoc even 唯一 query：1089；候选对：10832
- 每 epoch 采样槽：12000（有放回）；无公开集

---

## 6. 未解决问题 / 后续

1. 满训结束后跑三场景 sim3 自测并回填本报告数值表。
2. Phase2 ENU 重写未做（本 Phase 仍用 `/map` pose）。
3. worktree 内 `croco` 以 symlink 指向主仓（未提交 typechange）；`data/checkpoints/_db-q_pair_info` 同为 symlink。
4. 可择机把审核稿/结论 md 正式 commit 进 `relopt` `grok-bot`。
5. 遗忘评测对 ZC-only 仅跳过硬门禁；若需旁注官方对照，可事后手动跑 `eval_forgetting`（勿设 `ZC_ONLY`）。
