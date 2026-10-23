#!/bin/bash
# ZC-only finetune Reloc3r：仅自采三场景（无 MegaDepth / 无公开集混训）。
# 方案: relopt/docs/算法优化/ZC仅自采数据_Reloc3r处理与训练方案_审核稿_20261022.md
# 冒烟: PAIRSET=M SMOKE=1 bash scripts/finetune_zc_only.sh
# 满训: PAIRSET=M bash scripts/finetune_zc_only.sh
# 红线: OUT_DIR 默认 checkpoints/zc_finetune_ZC_only（勿覆盖旧 mix50/官方族 ckpt）
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/data/software/conda/anaconda3/envs/mv2d/bin/python}
PAIRSET=${PAIRSET:-M}
ZC9="./_db-q_pair_info/zc9_{}_q-retrieval_db-step=1_topk=10.npy"
if [ "$PAIRSET" = "M" ]; then
  ZCM="./_db-q_pair_info/zcM_{}_q-retrieval_db-step=1_topk=10.npy"
  ZC9="./_db-q_pair_info/zc9M_{}_q-retrieval_db-step=1_topk=10.npy"
  LMASK=", pairs_info_file_mask='$ZCM'"
else
  LMASK=""
fi
LR=${LR:-1e-5}
PAR=${PARITY:-even}
EVALP=${EVALP:-odd}
BS=${BS:-8}
if [ "${SMOKE:-0}" = 1 ]; then
  N=${N:-200}
  EP=${EP:-1}
  OUT=checkpoints/_smoke_zc_only
else
  N=${N:-4000}
  EP=${EP:-30}
  OUT=checkpoints/zc_finetune_ZC_only
fi
OUT=${OUT_DIR:-$OUT}
export ZC_ONLY=1
echo "[zc_only] PAIRSET=$PAIRSET PARITY=$PAR EVALP=$EVALP N=$N EP=$EP LR=$LR BS=$BS OUT=$OUT SMOKE=${SMOKE:-0}"
$PY train.py \
  --train_dataset "$N @ ZcRelpose(scene='20260921_064638', query_scene='20260920_081533', query_parity='$PAR'$LMASK, resolution=[(512,384)], transform=ColorJitter) \
                 + $N @ ZcRelpose(scene='20260921_064638', query_scene='20260921_062224', query_parity='$PAR'$LMASK, resolution=[(512,384)], transform=ColorJitter) \
                 + $N @ ZcRelpose(scene='20260923_055425_real_robot_record_external_lz4', query_scene='20260923_060415_real_robot_record_external_lz4', pairs_info_file_mask='$ZC9', query_parity='$PAR', resolution=[(512,384)], transform=ColorJitter)" \
  --test_dataset "2000 @ ZcRelpose(scene='20260921_064638', query_scene='20260920_081533', query_parity='$EVALP', resolution=(512,384), seed=777)" \
  --model "Reloc3rRelpose(img_size=512)" \
  --pretrained "checkpoints/reloc3r-512.pth" \
  --lr $LR --warmup_epochs 2 --epochs $EP --batch_size $BS --freeze_encoder --amp 1 \
  --save_freq 5 --keep_freq 10 --eval_freq 1 \
  --output_dir "$OUT"
