#!/bin/bash
# 优化方案 M3+M4 臂（PAIRSET=M 检索统一 + ND=12000 MD混50% + LR=5e-6 降学习率）训完后的自动链：
# 等待 checkpoint-best -> 遗忘检查 -> 三场景 odd 回测（现行 bm3 + M1 dualgate 两种聚合）
# 全部产物落 reloc3r/data/zc_sim3 与 results/finetune，日志 M3M4_auto.log。
set -u; cd /data/code/reLocation-all/reloc3r
PY=/data/software/conda/anaconda3/envs/mv2d/bin/python
D=checkpoints/zc_finetune_M_mix50_lr5e6
FT=$D/checkpoint-best.pth
SCOUT=/data/code/reLocation-all/relopt/results/finetune
log(){ echo "[$(date +%m-%d\ %H:%M)] $*" >> M3M4_auto.log; }
while [ ! -f "$D/checkpoint-final.pth" ]; do
  if ! pgrep -f "zc_finetune_M_mix50_lr5e6" >/dev/null; then
    log "train dead, relaunch resume"
    PAIRSET=M ND=12000 LR=5e-6 EP=30 OUT_DIR=$D bash scripts/finetune_zc.sh >> logs_mix50_lr5e6_M.log 2>&1
  fi
  sleep 300
done
log "train done"
sleep 60
[ -f results/forgetting_M_mix50_lr5e6.json ] || \
  $PY scripts/eval_forgetting.py "$FT" M_mix50_lr5e6 >> M3M4_auto.log 2>&1
for k in 062224 081533 0923; do
  [ -f data/zc_sim3/res_${k}_sim5_bm3_ftM_odd.npz ] || \
    SIM3_K=5 SIM3_BM3=1 WEIGHTS=$FT EVAL_PARITY=odd FT_TAG=ftM \
    $PY eval_zc_sim3.py $k >> M3M4_auto.log 2>&1
  [ -f data/zc_sim3/res_${k}_sim5_dg_ftM_odd.npz ] || \
    SIM3_K=5 SIM3_AGG=dualgate SIM3_REJ_M=2.0 WEIGHTS=$FT EVAL_PARITY=odd FT_TAG=ftM \
    $PY eval_zc_sim3.py $k >> M3M4_auto.log 2>&1
done
cp results/forgetting_*.json "$SCOUT/" 2>/dev/null
log "ALL_DONE"
