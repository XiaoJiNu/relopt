#!/bin/bash
# Phase2 ZC-only + ENU pose 微调。旁路数据 data/zc_*_enu + pair zcME_/zc9ME_。
# 冒烟: SMOKE=1 bash scripts/finetune_zc_only_enu.sh
# 满训: bash scripts/finetune_zc_only_enu.sh
# 红线: OUT_DIR 默认 checkpoints/zc_finetune_ZC_only_enu（勿覆盖 Phase1）
# pretrained 默认官方 reloc3r-512.pth（与 Phase1 公平对比）
set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."
PY=${PY:-/data/software/conda/anaconda3/envs/mv2d/bin/python}
ZCM="./_db-q_pair_info/zcME_{}_q-retrieval_db-step=1_topk=10.npy"
ZC9="./_db-q_pair_info/zc9ME_{}_q-retrieval_db-step=1_topk=10.npy"
LMASK=", pairs_info_file_mask='$ZCM'"
LR=${LR:-1e-5}
PAR=${PARITY:-even}
EVALP=${EVALP:-odd}
BS=${BS:-8}
PRETRAINED=${PRETRAINED:-checkpoints/reloc3r-512.pth}
if [ "${SMOKE:-0}" = 1 ]; then
  N=${N:-200}
  EP=${EP:-1}
  OUT=checkpoints/_smoke_zc_only_enu
else
  N=${N:-4000}
  EP=${EP:-30}
  OUT=checkpoints/zc_finetune_ZC_only_enu
fi
OUT=${OUT_DIR:-$OUT}
export ZC_ONLY=1
echo "[zc_only_enu] PARITY=$PAR EVALP=$EVALP N=$N EP=$EP LR=$LR BS=$BS OUT=$OUT PRETRAINED=$PRETRAINED SMOKE=${SMOKE:-0}"
$PY train.py \
  --train_dataset "$N @ ZcRelpose(scene='20260921_064638', query_scene='20260920_081533', query_parity='$PAR'$LMASK, resolution=[(512,384)], transform=ColorJitter) \
                 + $N @ ZcRelpose(scene='20260921_064638', query_scene='20260921_062224', query_parity='$PAR'$LMASK, resolution=[(512,384)], transform=ColorJitter) \
                 + $N @ ZcRelpose(scene='20260923_055425_real_robot_record_external_lz4', query_scene='20260923_060415_real_robot_record_external_lz4', pairs_info_file_mask='$ZC9', query_parity='$PAR', resolution=[(512,384)], transform=ColorJitter)" \
  --test_dataset "2000 @ ZcRelpose(scene='20260921_064638', query_scene='20260920_081533', query_parity='$EVALP', pairs_info_file_mask='$ZCM', resolution=(512,384), seed=777)" \
  --model "Reloc3rRelpose(img_size=512)" \
  --pretrained "$PRETRAINED" \
  --lr $LR --warmup_epochs 2 --epochs $EP --batch_size $BS --freeze_encoder --amp 1 \
  --save_freq 5 --keep_freq 10 --eval_freq 1 \
  --output_dir "$OUT"
