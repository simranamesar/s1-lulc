"""Pre-compute the demo gallery: the full prototype pipeline for a list of places, saved for instant replay.

Each place runs exactly what the app's "Run" button does (all raw Sentinel-1 scenes of 12 months, month-mean
prediction, ESRI + Open Buildings Temporal checks) and is saved to case_studies/<id>.pkl.gz. The app's
"Gallery" mode opens them without waiting for Earth Engine, which keeps the screen recording smooth.

The list deliberately includes places where the model is expected to struggle; the notes say why.

Usage (on a machine with Earth Engine access, e.g. the Mac):
    python3 case_studies.py --run ../train/runs/E1 --project scm-ksns            # all places
    python3 case_studies.py --run ../train/runs/E1 --project scm-ksns --only lagos,heidelberg
"""

from __future__ import annotations

import argparse
import gzip
import pickle
import time
from pathlib import Path

import pipeline as P

START = "2022-07"     # July 2022 - June 2023: lines up with the Open Buildings snapshots of 30 June 2022 / 2023

CASES = [
    # id, title, lat, lon, kind, what to look for
    ("lagos", "Lagos, Lekki-Ajah corridor (Nigeria)", 6.468, 3.630, "fast growth",
     "One of the fastest-growing urban edges in Africa: new estates on former bush and wetland. Expect clear new "
     "built-up (magenta) along the corridor. Open Buildings Temporal covers Nigeria, so our new built-up can be "
     "checked against an independent yearly building map."),
    ("nairobi", "Nairobi, Kitengela (Kenya)", -1.474, 36.958, "fast growth",
     "Peri-urban sprawl south of Nairobi on dry grassland. Expect scattered new built-up patches. Small single "
     "houses are near the 10 m resolution limit, so some growth may be missed."),
    ("dhaka", "Dhaka, Purbachal new town (Bangladesh)", 23.835, 90.520, "fast growth",
     "A planned new town on former paddy fields and wetland. Monsoon flooding (June-September) changes the radar "
     "signal strongly, so watch for water / crops flicker next to real construction."),
    ("accra", "Kasoa, west of Accra (Ghana)", 5.530, -0.420, "fast growth",
     "One of West Africa's fastest-growing towns: dense single-storey housing spreading over farmland. Many houses "
     "are small, so this tests how much of the growth a 10 m radar can see. Checked against Open Buildings Temporal."),
    ("addis", "Addis Ababa, eastern edge (Ethiopia)", 9.030, 38.880, "fast growth",
     "Condominium blocks and new roads on the city's expanding edge, in the highlands at 2,300 m. Large buildings "
     "should be detected well; the rainy season (June-September) changes the soil signal."),
    ("dar", "Dar es Salaam, Kigamboni (Tanzania)", -6.880, 39.320, "fast growth",
     "Coastal peri-urban growth on former bush and farmland across the harbour. Expect scattered new houses; sea "
     "and mangroves test the water class."),
    ("cairo", "New Administrative Capital (Egypt)", 30.020, 31.760, "fast growth, desert",
     "A new capital city being built in the desert. Real construction is large, but dry sand and new concrete can "
     "look alike to radar: a test of built-up vs bare ground."),
    ("phoenix", "Phoenix, Queen Creek (USA)", 33.260, -111.640, "fast growth",
     "Suburban growth on farmland and desert. Outside Open Buildings Temporal coverage, so only ESRI checks it."),
    ("heidelberg", "Heidelberg (Germany)", 49.4094, 8.6942, "stable (control)",
     "An old, almost fully built city with forest and river: very little real change in a year. Any new built-up "
     "reported here is mostly model error, so this shows the error level directly."),
    ("niamey", "Niamey outskirts (Niger)", 13.470, 2.200, "village-scale",
     "Small mud-brick houses spreading into the Sahel. Village houses are the hardest case (E1 finds about 40% of "
     "village built-up on the test set), so expect under-counting."),
    ("mekong", "Mekong delta rice fields (Vietnam)", 10.300, 105.550, "hard: seasonal water",
     "Rice paddies that are flooded, planted and harvested several times a year. Radar sees flooded fields as "
     "water and young rice as crops, so the monthly maps change a lot without any land-use change."),
    ("alps", "Zermatt, Swiss Alps", 46.020, 7.750, "hard: mountains and snow",
     "Steep slopes, glaciers and seasonal snow. Terrain causes radar layover and shadow (masked, shown as no data) "
     "and wet snow looks like water. A deliberately difficult place."),
    ("erg_chebbi", "Erg Chebbi dunes and Merzouga (Morocco)", 31.125, -4.000, "hard: smooth sand",
     "A Saharan sand sea with the edge of the village of Merzouga. Smooth dry sand reflects almost no radar signal "
     "back, like calm water, so expect water / bare confusion on the dunes; the village tests whether small "
     "desert houses stand out from the sand. (Replaces the Grand Erg Occidental in Algeria: Sentinel-1 recorded "
     "no scene there in 2022-23.)"),
    ("amazon", "Rondonia deforestation front (Brazil)", -9.750, -63.100, "real non-built change",
     "Forest cleared for pasture. Little built-up change, but trees -> grass & shrub should show in the "
     "All-class change tab: change that is real but not construction."),
]


# What the gallery showed (2026-11-07; moderate rule, July-Sept 2022 vs July-Sept 2023; "references" = ESRI 10 m land
# cover and Open Buildings Temporal (OBT), neither used in training). Shown in the app and in notebook 03.
FINDINGS = {
    "lagos": "Built-up 73 % (ESRI 72 %). 15 ha of new built-up (strict rule 0.5 ha, loose rule 69 ha); the references "
             "report 79-113 ha. The new estates are found, but much of the growth is houses filling gaps inside areas that "
             "are already built-up, which a land-cover map cannot show.",
    "nairobi": "Built-up 62 % (ESRI 78 %): under-counted, single houses on large plots are near the 10 m limit. 5 ha new. "
               "Comparing the same season halved the loose-rule result (210 -> 109 ha): much of it was rainy vs dry season.",
    "dhaka": "Built-up 36 % (ESRI 64 %): under-counted, dense low-rise and flooded land are read as water or crops. 11 ha "
             "new. The monsoon effect is visible: the loose rule gave 301 ha when comparing different seasons, 153 ha for "
             "the same season. The references disagree strongly (ESRI 286 ha, OBT 25 ha).",
    "accra": "The whole area is already built-up (ours 99 %, ESRI 100 %), so new houses only fill gaps: no land-cover change "
             "(ours 0 ha, ESRI 0.1 ha), while OBT counts 195 ha of new buildings. A limitation of mapping built-up as a "
             "class: densification is invisible.",
    "addis": "Built-up 86 % (ESRI 87 %). 8 ha new but 31 ha lost: the loss is probably not demolition but differences "
             "between the two rainy seasons and cleared construction sites. References: ESRI 19 ha, OBT 90 ha new.",
    "dar": "Built-up 62 % (ESRI 68 %). 37 ha of new built-up, the clearest agreement with both references (ESRI 74 ha, "
           "OBT 65 ha): scattered new houses on former bush and farmland are detected.",
    "cairo": "Built-up 13 % (ESRI 37 %): under-counted, dry sand and fresh concrete look alike to radar. 37 ha new and 32 ha "
             "lost: construction sites switch between sand, concrete and buildings. The references differ 7x (ESRI 195 ha, "
             "OBT 27 ha).",
    "phoenix": "Built-up 71 % (ESRI 78 %). 39 ha of new built-up in the new subdivisions; ESRI reports 146 ha (outside OBT "
               "coverage).",
    "heidelberg": "Control place: built-up 50 % (ESRI 52 %), 0.0 ha new, 0.2 ha lost. The headline rule reports no change "
                  "where there is none; the loose rule would report 5 ha new and 13 ha lost, all noise.",
    "niamey": "Built-up 30 % (ESRI 63 %): mud-brick houses are strongly under-counted. 3 ha new, 22 ha lost: probably the "
              "very wet 2022 rainy season (floods in Niger) vs a drier 2023, same months but different weather. References "
              "disagree 5x (ESRI 242 ha, OBT 48 ha).",
    "mekong": "Built-up 26 % (ESRI 28 %). Only 0.8 ha new despite fields that flood and dry several times a year: comparing "
              "the same season and requiring a persistent signal removes the rice-paddy flicker.",
    "alps": "Built-up 5 % (ESRI 4 %), 0.1 ha new: snow, glaciers and steep slopes produce no false growth (slopes the radar "
            "cannot see are masked as no data).",
    "erg_chebbi": "A clear failure case, as expected. The dunes are mapped as bare, but 9-44 % of the area is called "
                  "water depending on the month (smooth dry sand sends the radar signal away like calm water; the share "
                  "changes with wind ripples and sand moisture). The mud-brick houses of Merzouga are missed (built-up "
                  "0.1 %, ESRI 4.7 %), as in Niamey. No built-up change is reported, which is correct for the dunes. "
                  "(ESRI itself calls 82 % of the area rangeland, so it is a weak reference here too.)",
    "amazon": "No built-up change (0.1 %), as expected. The forest-to-pasture change is only partly seen: 53 ha of trees "
              "turned into grass & shrub, but the model calls 84 % of the area trees (ESRI 52 %) and even shows more trees "
              "in 2023 than in 2022, while ESRI reports 101 ha of tree loss. Pasture with scattered trees and regrowth "
              "looks like forest to C-band radar: a known limit for mapping deforestation with Sentinel-1 alone.",
}


def with_timeout(fn, seconds: int = 240):
    """Run an Earth Engine download in a background thread; give up after `seconds` (a request that never
    returns would otherwise block the whole gallery run). The place is then saved without that reference."""
    import threading
    box = {}

    def run():
        try:
            box["v"] = fn()
        except Exception as e:  # noqa: BLE001
            box["e"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(seconds)
    if t.is_alive():
        raise TimeoutError(f"no answer from Earth Engine after {seconds} s")
    if "e" in box:
        raise box["e"]
    return box["v"]


def compute(models, device, lat, lon, start_ym, size, tta=False, log=print, smooth3=False) -> dict:
    """The app's processing, without the UI. Returns the same dict the app keeps in session_state.result."""
    months = P.month_list(start_ym)
    months_all = P.month_list(start_ym, 12 + P.EXTRA_MONTHS)    # + the same 3 months one year later
    log(f"  downloading all raw Sentinel-1 scenes ({len(months_all)} months) ...")
    g = P.area_grid(lat, lon, size)
    stats = models[0][1]
    scenes_all, n_all, info, s1_all = P.fetch_s1_all(g, start_ym, terrain_flatten=bool(stats.get("terrain_flattened", True)),
                                                     n_months=len(months_all))
    log(f"  {int(n_all.sum())} scenes; predicting ...")
    pred_all, pb_all = P.predict_month_probs(models, scenes_all, device, tta, smooth3=smooth3)
    part = P.split_result(months_all, n_all, info, s1_all, pred_all, pb_all)
    log("  prediction done; downloading the reference maps (ESRI, Open Buildings Temporal) ...")
    ref = obt = esri_ab = None
    y0, y1 = int(months[0][:4]), int(months[-1][:4])
    try:
        ref = with_timeout(lambda: P.fetch_esri(g, y0))
        esri_ab = (y0, y1, ref, with_timeout(lambda: P.fetch_esri(g, y1))) if y1 != y0 else None
    except Exception as e:  # noqa: BLE001
        log(f"  ESRI not available: {e}")
    try:
        obt = with_timeout(lambda: P.fetch_obt_built(g, months[0], months[-1]))
    except Exception as e:  # noqa: BLE001
        log(f"  Open Buildings Temporal not available: {e}")
    return {"g": g, **part, "ref": ref, "ref_year": y0, "obt": obt, "esri_ab": esri_ab, "smoothed": False,
            "lat": lat, "lon": lon}


def pbuilt_or_none(res):
    return res.get("pbuilt")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="", help="trained model folder(s), comma-separated (several = ensemble); "
                                               "default: the prototype setting in pipeline.PROTOTYPE")
    ap.add_argument("--tta", action="store_true", default=None, help="test-time augmentation (default: PROTOTYPE)")
    ap.add_argument("--project", default="scm-ksns")
    ap.add_argument("--out", default="case_studies")
    ap.add_argument("--size", type=int, default=512, help="pixels of 10 m (512 = 5.12 km)")
    ap.add_argument("--start", default=START)
    ap.add_argument("--only", default="", help="comma-separated ids, e.g. lagos,heidelberg")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    import ee
    ee.Initialize(project=args.project)
    runs_root = Path(__file__).resolve().parents[1] / "train" / "runs"
    run_dirs = [r.strip() for r in args.run.split(",") if r.strip()] or [str(runs_root / r) for r in P.PROTOTYPE["runs"]]
    tta = P.PROTOTYPE["tta"] if args.tta is None else args.tta
    models, device = P.load_models(run_dirs)
    s3 = bool(P.PROTOTYPE.get("smooth3", False))
    floors = {r: P.noise_floor(run_dirs, r, tta, P.PROTOTYPE["hi"], P.PROTOTYPE["lo"], s3) for r in ("argmax", "moderate", "confident")}
    floor = floors[P.PROTOTYPE["change_rule"]]
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    only = set(args.only.split(",")) if args.only else None
    print(f"Models {[Path(r).name for r in run_dirs]} (TTA {tta}) on {device}; error level {floor:.2f} ha/km2/year")
    for cid, title, lat, lon, kind, note in CASES:
        if only and cid not in only:
            continue
        path = out / f"{cid}.pkl.gz"
        if path.exists() and not args.overwrite:
            print(f"{cid}: exists, skipped")
            continue
        print(f"{cid}: {title}")
        t0 = time.time()
        try:
            res = compute(models, device, lat, lon, args.start, args.size, tta, smooth3=s3)
        except Exception as e:  # noqa: BLE001 - keep going with the other places
            import traceback
            print(f"  FAILED: {e}")
            traceback.print_exc()
            continue
        res["case"] = {"id": cid, "title": title, "kind": kind, "note": note, "model": "+".join(Path(r).name for r in run_dirs) + (" TTA" if tta else "") + (" smooth3" if s3 else ""),
                       "floor": floor, "floors": floors}
        pr, pb, mo, fi, la, used = P.change_inputs(res, P.PROTOTYPE["change_mode"])
        bc = P.built_change(pr, mo, pbuilt=pb, rule=P.PROTOTYPE["change_rule"], first=fi, last=la)
        print(f"  change compares {used.replace('_', ' ')}: {mo[(fi or [0])[0]]}... vs ...{mo[(la or [-1])[-1]]}")
        if bc is None:
            print("  not enough months with Sentinel-1 data: no change can be measured")
        if bc:
            v, why, _ = P.change_verdict(bc, floor)
            res["case"]["verdict"] = v
            print(f"  built-up {bc['before_pct']:.1f}% -> {bc['after_pct']:.1f}%, new {bc['gain_ha']:.1f} ha, "
                  f"lost {bc['loss_ha']:.1f} ha -> {v}")
        with gzip.open(path, "wb") as f:
            pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  saved {path} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
