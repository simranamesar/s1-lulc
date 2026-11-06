# Sentinel-1 LULC monitor (prototype)

Enter a location (latitude/longitude, or click the map) and a start month. The app:

1. **Retrieves** one **raw Sentinel-1 scene per month** (no averaging) from Google Earth Engine for a square around
   the point (2.56, 5.12 or 7.68 km): the scene that covers the whole area and was acquired closest to mid-month.
   It is prepared with exactly the same recipe as the training data (`../data/s1_composite.py`): IW scene with
   VV + VH → terrain flattening to gamma0 (Copernicus GLO-30 DEM, layover/shadow masked) → border-noise mask →
   VV, VH, VH−VV in dB. Earth Engine has already applied calibration, thermal-noise removal and geometric terrain
   correction. No speckle filter, unless the chosen model was trained with the Lee filter (E5).
2. **Predicts** a 6-class land-cover map for every month with the trained U-Net (`../train/model.py`),
   normalised with the training statistics. A month without Sentinel-1 data is left empty.
3. **Calculates**, per month, the % of the area covered by each class (pixels of class / pixels with data × 100),
   the **month-to-month change** (this month − previous month) and the **total change** (last month with data −
   first month with data), both in **percentage points**.
4. **Measures built-up change** (the headline tab): built-up % at the start and the end, new / lost / net hectares
   and a change map. To stop single-month errors from reading as construction, the start and end are the **first 3
   and last 3 months with data**; a pixel is *new built-up* only if it is built-up in most of the last 3 months and
   in **none** of the first 3, and patches smaller than 4 pixels (400 m², adjustable) are dropped. The same rule
   gives a robust change table for every class.
5. **Shows** the 12 maps (slider, grid, animated GIF) with the date and orbit pass of each scene, the % and change
   tables, a % chart and the Sentinel-1 input.
6. **Checks** the result against independent maps, never used in training:
   - built-up change against **Open Buildings 2.5D Temporal** (Google, yearly building presence 2016–2023, dated
     30 June; Africa, South & Southeast Asia, Latin America) and against **ESRI 10 m land cover** of the start and
     end years. The table also shows how much of our new built-up the reference confirms;
   - the class map against ESRI on the same 6 classes (its *rangeland* = our grass & shrub).

The default example is the Lekki–Ajah corridor in Lagos, July 2022 – June 2023, which lines up with the Open
Buildings snapshots of June 2022 and June 2023. Other presets: Nairobi (Kitengela), Dhaka (Purbachal) and
Heidelberg as a place with little change.

Optional: **Smooth predictions across months** applies the same 3-month majority and trees/shrub seasonal fix used for
the training labels, to show how much of the change is classification flicker.

This Streamlit app plays the role of the Flask prediction miniapp from lab 14, with a map instead of a file upload.

## Accounts and keys

- A **Google Earth Engine** account with a Cloud project registered for noncommercial use (we used `scm-ksns`).
  No API key file is needed: authenticate once per machine with `earthengine authenticate`.

## Setup

```bash
cd prototype
python3 -m venv .venv && source .venv/bin/activate
pip install torch                      # see requirements.txt for the GPU variant
pip install -r requirements.txt
earthengine authenticate
```

The app needs a trained model folder from `train/runs/` (contains `best.pt`, `config.json`, `stats.json`).
It defaults to `../train/runs/E1`; change it in the sidebar.

## Run

```bash
streamlit run app.py                  # opens http://localhost:8501
EE_PROJECT=scm-ksns streamlit run app.py
```

On the server, run it there and open it from your laptop with an SSH tunnel:
`ssh -L 8501:localhost:8501 user@server`, then browse to http://localhost:8501.

`S1LULC_DEMO=1 streamlit run app.py` runs the interface on synthetic data (no Earth Engine, no model), only for
checking the UI.

## Files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit interface |
| `pipeline.py` | grid, retrieval, prediction, smoothing, class % and change, ESRI check, rendering; usable without the UI via `pipeline.run(...)` |
