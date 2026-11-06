#!/usr/bin/env bash
# E0 baseline: U-Net (ResNet34), one month of S1 in, raw Dynamic World label out.
# Runs on GPU 0 only (GPU 1 is used by someone else). Start inside tmux:
#   tmux new -s e0      then      bash run_e0.sh
set -euo pipefail
cd "$(dirname "$0")"
source ../.venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
python3 train.py \
  --prep-dir ../data/tiles_v0/prepared \
  --label dw \
  --name E0 \
  --epochs 40 --batch-size 32 --lr 1e-3 \
  2>&1 | tee runs_E0.log
