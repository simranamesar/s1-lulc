#!/usr/bin/env bash
# All runs for the report's model comparison, one after another on GPU 0.
# Input = single raw Sentinel-1 scenes (no averaging); E5 adds a Lee speckle filter.
# Start inside tmux:   tmux new -s exp    then    bash run_experiments.sh
# Each run writes runs/<ID>/ ; a summary table is printed at the end (compare_runs.py).
set -euo pipefail
cd "$(dirname "$0")"
[ -f ../.venv/bin/activate ] && source ../.venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1   # print progress immediately, also into log files
P=../data/tiles_v0/prepared
run() { echo "=== $1 ==="; python3 train.py --prep-dir "$P" --name "$@" 2>&1 | tee "runs_$1.log"; }

# E1 first: it is the main candidate for the prototype, so it is ready even if later runs are cut short
run E1  --label hyb_s                                 # hybrid + smoothed monthly labels (buildings, seasonal fix)
run E0  --label dw                                    # baseline: raw Dynamic World labels
run E2b --label hyb_s --encoder-weights none          # no transfer learning: trained from scratch
run E2a --label hyb_s --freeze-encoder                # transfer learning: ImageNet encoder frozen (feature extraction)
run E3  --label hyb_s --upsample transposed           # decoder upsampling by transposed convolution
run E5  --label hyb_s --lee                           # 5x5 Lee speckle filter on every raw scene
run E1y --label hyb_y                                 # hybrid ANNUAL label (yearly majority), stability baseline

python3 compare_runs.py runs E1 E0 E2b E2a E3 E5 E1y
