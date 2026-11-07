# S1-LULC-600: Sentinel-1 single scenes with monthly land-cover labels, 600 places worldwide (2023)

Training and evaluation data of the project *Sentinel-1 land cover monitor* (Data Analytics 3, SRH Hochschule
Heidelberg, 2026). Code: https://github.com/simranamesar/s1-lulc

600 places of 2.56 km × 2.56 km (256 × 256 pixels of 10 m), drawn stratified by world region (7), biome (14),
settlement type (city / town / village / none) and land cover, with **every Sentinel-1 IW acquisition of 2023**
(34,981 scenes) and **monthly land-cover labels** (6 classes), aligned pixel by pixel on a local UTM grid.
Train / validation / test = 409 / 61 / 130 places, split by 5° × 5° geographic blocks.

## Files

| File | Contents |
| --- | --- |
| `prepared_train.tar`, `prepared_val.tar`, `prepared_test.tar` | one `<tile_id>.npz` per place (input + labels, see below), grouped by split |
| `metadata.zip` | `tiles.csv` (one row per place: id, lat/lon, region, biome, settlement type, split, UTM grid), `tiles.geojson` (place outlines), `index.csv` (per place: scenes, months with data, label flip rates, class pixel counts), `stats.json` (normalisation mean / std from the train split, class counts), `balance_report.*` |
| `raw_labels.tar` | the GeoTIFFs as downloaded from Google Earth Engine, one per place: monthly Dynamic World and hybrid labels, Open Buildings roof cover (`BFRAC`), monthly Sentinel-1 composites (not used by the final models) |

## One place: `<tile_id>.npz` (numpy, `np.load`)

| Key | Shape, type | Meaning |
| --- | --- | --- |
| `s1` | (k, 3, 256, 256) int16 | the k Sentinel-1 scenes of 2023: VV, VH, VH − VV in dB × 100 (divide by 100); −32768 = no data (border, layover / shadow) |
| `month`, `day` | (k,) int8 | acquisition month (0 = January) and day of each scene |
| `orbit` | (k,) str | ASCENDING or DESCENDING |
| `valid` | (k,) float | share of the place covered by the scene |
| `n` | (12,) int16 | scenes per month |
| `best` | (12,) int | index of the scene with the best coverage nearest mid-month (−1 = none) |
| `dw` | (12, 256, 256) int8 | monthly Dynamic World label merged to 6 classes |
| `hyb` | (12, 256, 256) int8 | `dw` + built-up where Open Buildings roof cover ≥ 15 % |
| `hyb_s` | (12, 256, 256) int8 | `hyb` seasonally smoothed (trees/shrub fix + 3-month weighted vote): **the training label** |
| `hyb_y` | (12, 256, 256) int8 | annual majority of `hyb`, repeated for every month |
| `bfrac` | (256, 256) uint8 | Open Buildings roof cover, % of the pixel |

Labels: 0 water, 1 trees, 2 grass & shrub (incl. flooded vegetation), 3 crops, 4 built-up, 5 bare & snow;
−1 = no label (clouds, or snow outside cold biomes).

## How it was made

1. Places: stratified random sampling in Google Earth Engine (GHSL settlement model, RESOLVE Ecoregions 2017,
   ESA WorldCover 2021), plus extra points inside GHSL towns and cities.
2. Sentinel-1 GRD, IW mode, VV + VH; border noise masked; radiometric terrain flattening to γ⁰ (Vollrath et al.
   2020, Copernicus DEM GLO-30) with layover and shadow masked; converted to dB. Slices of one pass are merged;
   scenes covering < 50 % of the place are dropped.
3. Labels: Dynamic World V1, most frequent class per pixel and month; Open Buildings v3 roof cover; merged to
   6 classes; snow removed outside cold biomes; seasonal smoothing.

Scripts: `data/sample_tiles.py`, `export_tiles.py`, `export_scenes.py`, `prepare_data.py` in the code repository.

## Licence and attribution

Released under **CC BY 4.0**. Derived from:

* Contains modified Copernicus Sentinel data 2023 (ESA), processed in Google Earth Engine.
* Dynamic World V1, Google and World Resources Institute (CC BY 4.0).
* Open Buildings v3, Google (CC BY 4.0 / ODbL v1.0).
* GHSL (EU Joint Research Centre), RESOLVE Ecoregions 2017 (CC BY 4.0), ESA WorldCover 2021 (CC BY 4.0),
  Copernicus DEM GLO-30, used for sampling and terrain correction only.

Known limits: labels are model outputs (Dynamic World) and still change class in about 10 % of pixels from month
to month after smoothing; small buildings are under-represented outside the Open Buildings regions (Africa,
South & Southeast Asia, Latin America).
