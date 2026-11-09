# Step 3–4 — Prepare data and train E0 on the server

Folder layout on the server (inside the vault folder):

```
s1lulc/
  .venv/
  data/    sample_tiles.py  export_tiles.py  check_tile.py  prepare_data.py  requirements.txt
           tiles_v0/tiles.csv  tiles_v0/raw/*.tif  tiles_v0/scenes/*.npz   <- labels from the Mac, scenes from export_scenes.py
  train/   train.py  run_e0.sh  requirements-train.txt
```

## Setup (once)

```bash
cd /path/to/vault/s1lulc
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r data/requirements.txt -r train/requirements-train.txt
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"   # expect: True 2
```

## Prepare (≈ 2–5 min, CPU)

```bash
cd data
python3 prepare_data.py --tiles tiles_v0/tiles.csv --raw-dir tiles_v0/raw --scenes-dir tiles_v0/scenes --out-dir tiles_v0/prepared
```

Prints the S1 normalisation stats, the month-to-month label flip rate (raw vs smoothed), built pixels
Dynamic World vs hybrid per settlement type, and the class shares. Send this output for the log.

## Check the balance (≈ 1 min, CPU)

```bash
cd ../train
python3 balance.py --prep-dir ../data/tiles_v0/prepared --label hyb_s
```

Writes `balance_report.md`, `.csv` and `.png` into the prepared folder. It checks three things:

| Check | Balanced means | Fixed by |
| --- | --- | --- |
| Geography | every world region and settlement type gets an equal share of training draws | balanced sampler (automatic) |
| Classes | every class gets at least 2% of the training signal | balanced sampler + class weights in the loss (automatic) |
| Diversity | every class appears in at least 10 train tiles; effective sample size above 50% | **more tiles**: `sample_tiles.py --extend` |

Val and test are never re-weighted: they keep the real distribution, and we report per-class and
per-settlement metrics instead. The report's **Warnings** list says what to add next.

## Train E0 (GPU 0 only)

```bash
cd ../train
tmux new -s e0
bash run_e0.sh            # detach: Ctrl-b d · reattach: tmux attach -t e0
```

Results land in `train/runs/E0/`:

| File | What it holds |
| --- | --- |
| `history.csv`, `curves.png` | loss and mIoU per epoch (for the notebook) |
| `best.pt` | weights of the best validation epoch (not in the repository; `best_fp16.pt`, the same weights in half precision, is included for E1, E2b, E3, E5) |
| `test_metrics.json` | OA, mIoU, macro F1 (6 classes), per settlement type and region, village built-up, flip rate, built-up precision / recall / false change rate, new built-up with the prototype's change rule |
| `test_per_class.csv`, `test_confusion.csv` | per-class IoU and confusion matrix |

Training uses the balanced sampler by default; `--sampler uniform` turns it off (useful as a comparison run).

## All experiments for the report

`run_experiments.sh` trains every variant in turn on GPU 0 and prints a comparison table (`runs/comparison.csv`):

| Run | What changes | Question it answers |
| --- | --- | --- |
| E0 | raw Dynamic World labels | baseline |
| E1 | hybrid + smoothed monthly labels | do building footprints and seasonal smoothing help villages and change numbers? |
| E1y | hybrid annual label (yearly majority) | are stable, WorldCover-style annual labels better than monthly ones? |
| E2a | E1 with the ImageNet encoder frozen | transfer learning as feature extraction vs fine-tuning |
| E2b | E1 trained from scratch | does ImageNet help at all, given photos and radar differ (negative transfer)? |
| E3 | E1 with transposed-conv upsampling | learned "inverse of convolution" vs interpolation |
| E5 | E1 with a 5×5 Lee speckle filter | does speckle filtering help on raw single scenes? |

```bash
tmux new -s exp
bash run_experiments.sh
```

Built-up change metrics (the prototype's headline), printed after the test:

- **false change rate**: share of pixels whose predicted built-up status flips between consecutive months where the
  label's built-up status stays the same. Each flip would be fake construction or demolition.
- **new built-up (prototype rule)**: predicted vs label new built-up in ha per km², the false part, and its precision
  and recall, between the first 3 and the last 3 months of 2023. `eval_ensemble.py` scores the final setting (ensemble,
  3-month smoothing) with the probability rules the app uses.

`compare_runs.py` puts `built_P`, `built_R`, `built_false_change` and `false_new_ha_km2` next to mIoU, so the run
chosen for the prototype is the one that measures change best, not only the one with the best mIoU.

The network is defined once in `model.py` (U-Net: ResNet34 encoder, decoder, 1×1 conv output) so the prototype
loads exactly the same model. Training stops early after `--patience` epochs (default 10) without val mIoU gain.
