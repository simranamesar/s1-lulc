#!/usr/bin/env bash
# 2026-11-07: one more training run that records how the model learns (for notebooks/04_training_progress.ipynb):
# E3v = the E3 setting (transposed-conv upsampling, hyb_s labels) with seed 2; saves the loss of every batch
# (steps.csv), per-class validation IoU per epoch (history.csv) and the predictions of 6 fixed validation scenes
# after every epoch (snapshots/). Also runs the pending ensemble evaluation with the moderate change rule (0.5/0.35).
# Start:  cd ~/vault/s1-lulc && bash train/run_viz.sh      (background, ~45 min; log: train/viz_log.txt)
cd "$(dirname "$0")"
[ -f ../.venv/bin/activate ] && source ../.venv/bin/activate
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
P=../data/tiles_v0/prepared
nohup bash -c "
set -x
python3 train.py --prep-dir $P --name E3v --label hyb_s --upsample transposed --seed 2 --snapshots 6 > runs_E3v.log 2>&1
python3 eval_ensemble.py --prep-dir $P --runs runs/E3 runs/E1 runs/E5 runs/E2b --no-tta --thresholds 0.5:0.2,0.5:0.35,0.6:0.2,0.7:0.3 --out ensemble_eval_round2.csv
cd ../notebooks && python3 build_notebooks.py && for nb in 01_data 02_training_evaluation 04_training_progress; do jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 \$nb.ipynb; done
echo VIZ_DONE
" > viz_log.txt 2>&1 &
echo "Started. Progress: tail -f ~/vault/s1-lulc/train/viz_log.txt   (training lines: tail -f ~/vault/s1-lulc/train/runs_E3v.log)"
