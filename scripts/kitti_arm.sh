#!/usr/bin/env bash
# Frozen KITTI split and 60+40 recipe. Usage: DATA_DIR=... IMG_CACHE=... scripts/kitti_arm.sh PCM3 run_name GPU [ablation args]
set -euo pipefail
cd "$(dirname "$0")/.."
P=${1:?preset}; RUN=${2:?run name}; GPU=${3:?physical GPU}; shift 3
PY=${PYTHON:-python}
: "${DATA_DIR:?set DATA_DIR to prepared KITTI}"
: "${IMG_CACHE:?set IMG_CACHE to the 256x512 training cache}"
FLOWNET=${FLOWNET:-pretrain_models/flownets_bn_EPE2.459.pth.tar}
[[ -f "$FLOWNET" ]] || { echo "Missing FlowNet initialization: $FLOWNET" >&2; exit 1; }
[[ ! -e "results/$RUN" ]] || { echo "Run already exists: $RUN" >&2; exit 1; }
export VIO_SEG_LENGTHS=100,200,300,400,500,600,700,800 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg
unset PYTHONPATH
mkdir -p "results/$RUN"
"$PY" train_crvio.py --preset "$P" --stage both --data_dir "$DATA_DIR" --img_cache "$IMG_CACHE" \
  --train_seq 00 01 02 04 06 08 --val_seq 09 --test_seq 05 07 10 --dt 0.1 \
  --epochs_stage1 60 --epochs_stage2 40 --workers 8 --gpu 0 --amp bf16 --batch_size 32 \
  --lr_stage1 3.5e-4 --pretrain_flownet "$FLOWNET" --experiment_name "$RUN" "$@" \
  > "results/$RUN/stdout.log" 2>&1
"$PY" tools/benchmark.py --dataset kitti --ckpt "results/$RUN/checkpoints/stage1_last.pth" \
  --data-dir "$DATA_DIR" --gpu 0 --name "${RUN}_s1_benchmark"
"$PY" tools/benchmark.py --dataset kitti --ckpt "results/$RUN/checkpoints/stage2_last.pth" \
  --data-dir "$DATA_DIR" --gpu 0 --name "${RUN}_s2_benchmark" --include-policy
