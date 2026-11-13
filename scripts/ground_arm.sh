#!/usr/bin/env bash
# Frozen ground stage-1 recipe. Usage: DATA_DIR=... IMG_CACHE=... scripts/ground_arm.sh PCM3 run_name GPU [ablation args]
set -euo pipefail
cd "$(dirname "$0")/.."
P=${1:?preset}; RUN=${2:?run name}; GPU=${3:?physical GPU}; shift 3
PY=${PYTHON:-python}
: "${DATA_DIR:?set DATA_DIR to the prepared ground dataset}"
: "${IMG_CACHE:?set IMG_CACHE to the 256x512 training cache}"
FLOWNET=${FLOWNET:-pretrain_models/flownets_bn_EPE2.459.pth.tar}
[[ -f "$FLOWNET" ]] || { echo "Missing FlowNet initialization: $FLOWNET" >&2; exit 1; }
[[ ! -e "results/$RUN" ]] || { echo "Run already exists: $RUN" >&2; exit 1; }
export VIO_SEG_LENGTHS=5,10,15,20 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg
unset PYTHONPATH
mkdir -p "results/$RUN"
"$PY" train_crvio.py --preset "$P" --stage 1 --data_dir "$DATA_DIR" \
  --train_seq m2_bridge2 m2_street1 m2_switch m3_dynamic01 m3_shaturn01 m3_wheelfloat01 \
  --val_seq m2_tree --test_seq m2_building2 m2_street2 m3_varyingillu01 m3_occlusion01 \
  --img_cache "$IMG_CACHE" --dt 0.0666667 --epochs_stage1 40 --epochs_stage2 40 \
  --workers 8 --gpu 0 --amp bf16 --batch_size 32 --lr_stage1 3.5e-4 \
  --visual_types noise photometric occlusion blur blank --pretrain_flownet "$FLOWNET" \
  --experiment_name "$RUN" "$@" > "results/$RUN/stdout.log" 2>&1
"$PY" tools/benchmark.py --dataset ground --ckpt "results/$RUN/checkpoints/stage1_last.pth" \
  --data-dir "$DATA_DIR" --gpu 0 --name "${RUN}_benchmark"
"$PY" tools/pcm_diag.py --ckpt "results/$RUN/checkpoints/stage1_last.pth" --data_dir "$DATA_DIR" \
  --gpu 0 --out "results/$RUN/diag_last.json" > "results/$RUN/diag_last.log" 2>&1
