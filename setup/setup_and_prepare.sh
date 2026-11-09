#!/usr/bin/env bash
# One-time setup + data preparation on the server (DataLab / JupyterLab terminal).
# Run from the project folder:   bash setup/setup_and_prepare.sh
# Everything it prints is also saved to setup/setup_log.txt.
set -uo pipefail
cd "$(dirname "$0")/.."
exec > >(tee setup/setup_log.txt) 2>&1
step() { echo; echo "=== $* ==="; }

step "1/6 Find the data folder (tiles_v0)"
if [ ! -d data/tiles_v0 ]; then
  found=$(find .. -maxdepth 4 -type d -name tiles_v0 2>/dev/null | head -1)
  if [ -n "$found" ]; then
    ln -s "$(cd "$found" && pwd)" data/tiles_v0 && echo "linked data/tiles_v0 -> $found"
  else
    echo "ERROR: no tiles_v0 folder found. Put it inside s1-lulc/data/ and run again."; exit 1
  fi
fi
echo "tiles: $(tail -n +2 data/tiles_v0/tiles.csv | wc -l) | label files: $(ls data/tiles_v0/raw/*.tif | wc -l) | scene files: $(ls data/tiles_v0/scenes/*.npz | wc -l)"

step "2/6 Python environment"
if [ ! -f .venv/bin/activate ]; then
  python3 -m venv .venv || echo "venv not available here: installing into the user's Python instead"
fi
[ -f .venv/bin/activate ] && source .venv/bin/activate
PIP="python3 -m pip install -q"; [ -z "${VIRTUAL_ENV:-}" ] && PIP="$PIP --user"
python3 --version

step "3/6 Install packages (needs internet; a few minutes the first time)"
python3 -c "import torch" 2>/dev/null || $PIP torch --index-url https://download.pytorch.org/whl/cu121 \
  || { echo "ERROR: could not install PyTorch (no internet from this machine?)"; exit 1; }
$PIP -r data/requirements.txt -r train/requirements-train.txt scipy \
  || { echo "ERROR: package install failed"; exit 1; }

step "4/6 GPU check"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv || true
CUDA_VISIBLE_DEVICES=0 python3 -c "import torch; print('PyTorch', torch.__version__, '| GPU available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

step "5/6 Prepare the data (10-20 min)"
cd data
python3 prepare_data.py --tiles tiles_v0/tiles.csv --raw-dir tiles_v0/raw --scenes-dir tiles_v0/scenes \
  --out-dir tiles_v0/prepared || { echo "ERROR in prepare_data"; exit 1; }

step "6/6 Balance check"
cd ../train
python3 balance.py --prep-dir ../data/tiles_v0/prepared --label hyb_s || { echo "ERROR in balance"; exit 1; }

echo; echo "ALL DONE. Log: setup/setup_log.txt. Then start training with:  bash setup/start_training.sh"
