# Sentinel-1 land cover monitor (prototype)

A Streamlit app: type a location and a start month, and it draws **12 monthly land-cover maps from Sentinel-1 radar
only**, the share of every class per month, and **how much land was newly built in one year**, checked against two
independent maps.

## What you see

- **Home**: a rotating globe with the 600 dataset places (small dots) and the places shown in the app (large dots),
  the key numbers of the project and two buttons: *Open the gallery* and *Run a place of your choice*.
- **Live run**: type latitude and longitude (or click the map), choose the area size (2.56, 5.12 or 7.68 km) and the
  start month, press **Run**.
- **Gallery**: 14 places computed in advance (fast-growing city edges, a stable control, hard cases), each with the
  facts of the run and a short note on what was found.

The results have four tabs:

| Tab | Shows |
| --- | --- |
| **Built-up change** | built-up share, new and lost built-up in hectares, a change map (pink = new) and a verdict box |
| **Monthly maps** | the land-cover map of each month next to the radar image the model received; all 12 months and an animated GIF |
| **Class shares** | the share of each of the 6 classes per month, and the change between the start and the end |
| **Checks** | the built-up change of Open Buildings 2.5D Temporal and ESRI 10 m land cover on the same pixels, how much of our new built-up they confirm, and our map of the year next to ESRI's |

## How a run works (`pipeline.py`)

1. **Grid**: a square of 10 m pixels around the point, in the local UTM zone.
2. **Download**: every Sentinel-1 IW scene of the 12 months, plus the same 3 months one year later, from Google Earth
   Engine, prepared exactly like the training data (`../data/s1_composite.py`: VV + VH, border-noise mask, terrain
   flattening to γ⁰ with layover / shadow masked, VV, VH, VH−VV in dB).
3. **Predict**: every scene with the 4 trained models E3, E1, E5, E2b (`../train/runs/*/best_fp16.pt`); their class
   probabilities are averaged per scene and then over all scenes of a month.
4. **Smooth**: each month's probabilities are averaged with the previous and next month (halves the month-to-month
   flicker on the test places), then the most likely class is taken.
5. **Class shares**: % of the pixels with data per class and month; changes in percentage points.
6. **Built-up change**: the first 3 months are compared with **the same 3 months one year later**, so a change of
   season cannot look like construction. A pixel is *new built-up* when its built-up probability is ≥ 0.5 at the end
   and < 0.35 in every start month ("moderate" rule); patches under 4 pixels (400 m²) are dropped.
7. **Is it real?** On the 130 test places (regions never used in training) this rule wrongly marks about
   0.04 ha of new built-up per km² per year. The app scales this to the area (about 1.1 ha for 5.12 km × 5.12 km):
   at least 3 times more = *clear growth*, 1.5–3 times = *some growth*, otherwise *no change detectable*.
8. **Check**: Open Buildings 2.5D Temporal (Google; yearly building presence, Africa, South & Southeast Asia, Latin
   America) and ESRI 10 m annual land cover, neither used in training.

The default setting is stored in `pipeline.PROTOTYPE`. *Advanced settings* in the sidebar can choose single models,
test-time augmentation, the within-year comparison, a stricter or looser change rule, majority smoothing and the
smallest change patch.

**Saved results.** Every live run with the default settings is saved in `cache/`; typing the same coordinates again
(same start month and area) opens it instantly, with a grey note saying when it was computed. Gallery and village
test results are reused the same way.

## Setup

A Google Earth Engine account with a Cloud project registered for non-commercial use (no key file; authenticate once
per machine).

```bash
cd prototype
python3 -m venv .venv && source .venv/bin/activate
pip install torch                      # see requirements.txt for the GPU variant
pip install -r requirements.txt
earthengine authenticate
```

The four model files are in `../train/runs/{E3,E1,E5,E2b}/best_fp16.pt`; the app finds them automatically.

## Run

```bash
EE_PROJECT=<your-project> streamlit run app.py      # opens http://localhost:8501
```

Gallery files are large (about 20 MB per place) and not in the repository. Create them once:

```bash
python3 case_studies.py --project <your-project>                 # 14 places, 3–5 min each
python3 case_studies.py --project <your-project> --only lagos    # one place
python3 summarize_cases.py                                       # print their numbers
```

`S1LULC_DEMO=1 streamlit run app.py` runs the interface on synthetic data (no Earth Engine, no model), only to check
the layout. On a server, use an SSH tunnel: `ssh -L 8501:localhost:8501 user@server`.

## Files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit interface (gallery, live run, result tabs, saved results) |
| `landing.py` | Home page with the globe |
| `pipeline.py` | grid, download, prediction, smoothing, class shares, built-up change, verdict, checks; `PROTOTYPE` holds the default setting |
| `case_studies.py` | computes the 14 gallery places and their notes (`FINDINGS`) |
| `summarize_cases.py` | prints the numbers of the gallery places |
| `village_candidates.py` | the village tests (Nile delta and others), saved to `village_candidates/` |
| `check_coverage.py` | counts Sentinel-1 scenes over candidate places (used to replace a desert place without any scenes) |
| `static/places.json` | latitude / longitude of the 600 dataset places, for the globe |
| `static/cobe.min.js` | the globe library [cobe](https://github.com/shuding/cobe) 0.6.3 (MIT, Shu Ding), bundled so the page works offline |

The globe follows the design of the *Globe* component of Magic UI (MIT), rewritten as plain HTML and JavaScript.
