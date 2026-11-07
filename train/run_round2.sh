#!/usr/bin/env bash
# Round 2 (2026-11-06): repeat runs with another seed (noise), the regularised E6, and 3-month smoothing.
# Start:  cd ~/vault/s1-lulc && bash train/run_round2.sh      (runs in the background, log: train/round2_log.txt)
cd "$(dirname "$0")"
[ -f ../.venv/bin/activate ] && source ../.venv/bin/activate
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
P=../data/tiles_v0/prepared
nohup bash -c "
set -x
python3 train.py --prep-dir $P --name E3s1 --label hyb_s --upsample transposed --seed 1 > runs_E3s1.log 2>&1
python3 train.py --prep-dir $P --name E1s1 --label hyb_s --seed 1 > runs_E1s1.log 2>&1
python3 train.py --prep-dir $P --name E6 --label hyb_s --upsample transposed --dropout 0.2 --jitter-db 1.5 --noise-db 0.5 --lovasz 0.5 > runs_E6.log 2>&1
for r in E3s1 E1s1 E6; do python3 eval_change.py --run runs/\$r --prep-dir $P --label hyb_s; done
python3 eval_ensemble.py --prep-dir $P --runs runs/E3 runs/E1 runs/E5 runs/E2b --no-tta --out ensemble_eval_round2.csv
python3 eval_ensemble.py --prep-dir $P --runs runs/E6 runs/E3 runs/E1 runs/E5 runs/E2b --no-tta --out ensemble_eval_with_E6.csv
python3 compare_runs.py runs E1 E1s1 E3 E3s1 E6 E0 E2b E2a E5 E1y E1half
echo ROUND2_DONE
" > round2_log.txt 2>&1 &
echo "Round 2 started (about 2.5 hours). Progress: tail -f ~/vault/s1-lulc/train/round2_log.txt"
