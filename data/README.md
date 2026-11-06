# Step 1 — Sample training tiles

`sample_tiles.py` picks a globally stratified list of 2.56 km tiles for the Sentinel-1-only LULC model.
It does not download imagery yet; it only decides *where* the training and test data will come from.

## One-time setup

1. Earth Engine must be registered for noncommercial use on Cloud project `scm-ksns`.
2. On the machine that runs the script (laptop or cluster login node, needs internet):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
earthengine authenticate          # opens a browser link; on a headless node use: earthengine authenticate --auth_mode=notebook
```

## Run

```bash
python sample_tiles.py --project scm-ksns --target 300 --out-dir tiles_v0
```

It takes a few minutes. Outputs in `tiles_v0/`:

| File | What it holds |
| --- | --- |
| `tiles.csv` | one row per tile: id, lon/lat, region, biome, settlement type, split, S1 months, UTM EPSG |
| `tiles.geojson` | the tile squares, for QGIS or geojson.io |
| `tiles_map.html` | open in a browser: colour = settlement type, black ring = test tile |

## How tiles are chosen

- **Strata**: 7 world regions × 14 biomes (RESOLVE 2017) × settlement type (GHSL Degree of Urbanisation + built-up surface) × dominant land cover (ESA WorldCover 2021, 1 km).
- **Shares**: village 35%, none 25%, town 20%, city 20%. Villages are oversampled because they are the hardest case.
- **Equal turns**: inside each settlement type, every region takes a turn before any region gets a second tile; inside a region, turns rotate over land cover × biome.
- **Growing the set** (`--extend tiles_v0/tiles.csv --target 600 --out-dir tiles_v1`): keeps every existing tile (same id and split) and gives new tiles to the thinnest strata first. Point `export_tiles.py --out-dir` at the same `raw` folder and it only downloads the new tiles.
- **Sentinel-1 check**: a tile is kept only if it has VV+VH IW data in at least 10 months of `--year` (default 2023).
- **Splits**: train / val / test are assigned per 5° block (≈ 550 km), so test regions never sit inside training areas. Roughly 70 / 10 / 20.

## If something goes wrong

| Symptom | Fix |
| --- | --- |
| `WARNING: only N tiles` | raise `--points-per-region` (e.g. 8000) or `--oversample` (e.g. 3) |
| Earth Engine timeout on step 1 | lower `--points-per-region` to 2000 and run again |
| Too few villages in Africa / Asia | lower `--built-threshold` (e.g. 10) |
| `Asset not found` for a GHSL layer | the catalog version moved; check the ID in the Earth Engine catalog and update `stratum_image()` |

---

# Step 2 — Download Sentinel-1 composites and labels

`export_tiles.py` writes one GeoTIFF per tile (256 × 256 px, 10 m, local UTM, int16, ~10 MB).
Each file holds, for every month of `--year`:

| Band | Content | Used as |
| --- | --- | --- |
| `mMM_VV`, `mMM_VH` | Sentinel-1 monthly median, dB × 100 | **model input** |
| `mMM_VHVV` | VH − VV ratio, dB × 100 | **model input** |
| `mMM_N` | number of S1 scenes that month (0 = none) | quality flag |
| `mMM_DW` | Dynamic World label for the month (0–8, −1 = no clear S2 image) | label (E0) |
| `mMM_HYB` | Dynamic World with "built" forced where buildings are dense | label (E1) |
| `BFRAC` | building roof cover % per pixel (Open Buildings v3; 0 outside its coverage) | analysis |

Dynamic World classes: 0 water, 1 trees, 2 grass, 3 flooded vegetation, 4 crops, 5 shrub, 6 built, 7 bare, 8 snow/ice.

## Run

Test on 3 tiles first:

```bash
pip install -r requirements.txt
python3 export_tiles.py --project scm-ksns --tiles tiles_v0/tiles.csv --out-dir tiles_v0/raw --limit 3
python3 check_tile.py tiles_v0/raw/t00000.tif
```

`check_tile.py` prints a month-by-month coverage table and writes `t00000_check.png`
(S1 VV, Dynamic World label, hybrid label, roof cover side by side).

If those look right, download everything (≈ 3 GB, roughly 30–90 min):

```bash
python3 export_tiles.py --project scm-ksns --tiles tiles_v0/tiles.csv --out-dir tiles_v0/raw
```

Rerunning the same command skips tiles already on disk and retries only failures (listed in `manifest.csv`).


---

# Step 2b — Download single raw Sentinel-1 scenes (model input)

Since 2026-11-03 the model input is **single raw scenes**, not monthly medians. `export_tiles.py` is still needed for
the monthly **labels** (Dynamic World, hybrid, roof cover); `export_scenes.py` downloads every Sentinel-1 acquisition
of the year on the same 256 × 256 px, 10 m grid:

```bash
python3 export_scenes.py --project scm-ksns --tiles tiles_v0/tiles.csv --out-dir tiles_v0/scenes --limit 3
python3 export_scenes.py --project scm-ksns --tiles tiles_v0/tiles.csv --out-dir tiles_v0/scenes
```

One `.npz` per tile: all scenes (VV, VH, VH−VV in dB × 100), date, orbit pass, relative orbit and coverage.
Scenes covering less than half the tile are dropped. Same-day slices of one pass are joined (not averaged).

**Terrain flattening (on by default).** Earth Engine's S1_GRD is already calibrated, thermal-noise-cleaned and
geometrically terrain-corrected (sigma0), but slopes facing the satellite still look brighter than slopes facing
away. `s1_composite.flatten()` converts each scene to **gamma0** with the volume-scattering model of Vollrath et al.
(2020) on the Copernicus GLO-30 DEM and masks layover and radar-shadow pixels (they count as gaps). On a 10° slope
this removes about 1.5–1.8 dB of bias. `--no-flatten` keeps sigma0 for a comparison run; the flag is stored in each
`.npz`, in `stats.json`, and the prototype reads it from the model folder, so training and prediction always match.

Expect 30–60 minutes and 5–10 GB for 300 tiles (40–60 scenes per tile). Run the 3-tile test first.

Then prepare with both inputs:

```bash
python3 prepare_data.py --tiles tiles_v0/tiles.csv --raw-dir tiles_v0/raw --scenes-dir tiles_v0/scenes --out-dir tiles_v0/prepared
```

Every scene becomes one training sample, labelled with its month's map. `stats.json` holds normalisation values for
raw scenes and for the Lee-filtered variant (E5).
