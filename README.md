# Sentinel-1 land use / land cover monitor

Deep-learning semantic segmentation of land use and land cover from **Sentinel-1 radar only**, trained on 600 places
on every continent, and a prototype that shows **how much was built in a year** at any coordinates.

Exam project, Data Analytics 3 (SRH Hochschule Heidelberg), 2026.

| | |
| --- | --- |
| Input | single raw Sentinel-1 IW scenes (VV, VH, VH−VV in dB), terrain-flattened γ⁰, 10 m |
| Output | 6 classes: water, trees, grass & shrub, crops, built-up, bare & snow |
| Labels | Dynamic World monthly maps + Open Buildings roof cover (hybrid), seasonally smoothed |
| Data | 600 places of 2.56 × 2.56 km, stratified by region × biome × settlement × land cover; 34,981 scenes of 2023 |
| Model | U-Net, ResNet34 encoder (ImageNet, fine-tuned), CE + Dice loss, balanced sampler |
| Prototype | Streamlit app: point + start month → 12 monthly maps, class %, changes, built-up change, independent checks |

## Results (130 test places never seen in training)

| | mIoU | OA | Built-up F1 | Month-to-month flicker |
| --- | --- | --- | --- | --- |
| E1, one scene per month | 0.487 | 0.632 | 0.82 | 25 % |
| E1, month mean of all scenes | 0.522 | 0.662 | 0.84 | 15 % |
| Ensemble of 4 models, month mean | 0.544 | 0.682 | 0.84 | 12 % |
| **Prototype: ensemble + 3-month smoothing** | **0.549** | | | **5.9 %** |

Built-up change compares the first 3 months with the same 3 months one year later, so a change of season cannot
look like construction. A pixel counts as new built-up when its built-up probability is ≥ 0.5 at the end and < 0.35
in every start month ("moderate" rule; a strict and a loose rule are shown as lower and upper bound). On the 130 test
places this rule wrongly marks about **0.04 ha of new built-up per km² per year** (strict rule ≈ 0.005, loose rule
≈ 0.59; measured between the first and last 3 months of 2023, the year with labels). The app compares every result
with this error level and reports *clear growth* only when the new built-up is at least 3 times larger. The variant
comparison (labels, transfer learning, upsampling, speckle filter, annual labels, learning curve) is in
`notebooks/02_training_evaluation.ipynb`; what was tried, what failed and why is in the report.

## Repository

```
data/        Earth Engine download and preparation (sampling, labels, scenes, cleaning, statistics)  -> data/README.md
train/       model, training, evaluation, variant runs and their results (train/runs/*)              -> train/README.md
prototype/   Streamlit app, pipeline, pre-computed gallery of example places                         -> prototype/README.md
notebooks/   01 data · 02 training and evaluation · 03 case studies · 04 training step by step
             (all executed, with outputs)
```

Model weights: `train/runs/<run>/best_fp16.pt` (half precision, ~50 MB each) for the four prototype models
E3, E1, E5, E2b; the app loads them automatically. Settings, training history, test metrics and logs of all runs are
in `train/runs/`.

## Dataset

600 places, 34,981 Sentinel-1 scenes of 2023 with monthly labels (13.5 GB), CC BY 4.0:
**https://doi.org/10.5281/zenodo.24053842**. Unpack the three `prepared_*.tar` into `data/tiles_v0/` and the files of
`metadata.zip` into `data/tiles_v0/prepared/` to retrain or evaluate (see the dataset README for the file format).

## Run the prototype

```bash
cd prototype
pip install torch                      # see requirements.txt for the GPU variant
pip install -r requirements.txt
earthengine authenticate               # Google Earth Engine account with a Cloud project (non-commercial)
streamlit run app.py
```

The app opens on a **Home** page. **Live run** works for any coordinates (2–5 min; each result is saved in
`prototype/cache/` and opens instantly the next time). **Gallery** shows 14 pre-computed places with notes on what
was found; these result files are large and not in the repository, so create them once with
`python3 case_studies.py --project <your Earth Engine project>` (3–5 min per place). See `prototype/README.md`.

## Reproduce the data and training

1. `data/`: `sample_tiles.py` → `export_tiles.py` → `export_scenes.py` → `prepare_data.py` (see `data/README.md`).
2. `train/`: `balance.py`, then `run_experiments.sh` (all variants on one GPU), `eval_change.py`, `eval_ensemble.py`.
3. `notebooks/`: run 01 and 02 for the figures and tables, 04 for the training curves (03 needs the gallery files).

## Data sources

Sentinel-1 GRD (ESA Copernicus, via Google Earth Engine) · Dynamic World V1 (Google / WRI) · Open Buildings v3 and
2.5D Temporal (Google) · GHSL SMOD and BUILT-S (EU JRC) · RESOLVE Ecoregions 2017 · ESA WorldCover 2021 ·
Copernicus DEM GLO-30 · ESRI 10 m Land Cover (Impact Observatory).
