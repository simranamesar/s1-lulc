#!/usr/bin/env bash
# Starts all training runs in the background on GPU 0. Keeps running after you close the browser tab.
#   bash setup/start_training.sh  start
#   tail -f train/train_log.txt   watch progress (Ctrl-C only stops watching, not training)
cd "$(dirname "$0")/../train"
nohup bash run_experiments.sh > train_log.txt 2>&1 &
echo "Training started (process $!). Progress: tail -f train/train_log.txt"
echo "E1 (the main model) runs first; results land in train/runs/E1/"
