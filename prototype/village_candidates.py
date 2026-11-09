"""Try a few village areas before the screen recording and summarise how the model does there.

Runs the same processing as the app (and case_studies.py) for each candidate, saves the result to
village_candidates/<id>.pkl.gz (NOT the gallery) and prints built-up share, new built-up and the two references.

    python3 village_candidates.py --project scm-ksns
    python3 village_candidates.py --project scm-ksns --only nile,punjab
"""
import argparse
import gzip
import pickle
import time
from pathlib import Path

import pipeline as P
from case_studies import START, compute

CANDIDATES = [
    ("nile", "Villages in the Nile delta between Tanta and Shibin el-Kom (Egypt)", 30.700, 30.950),
    ("punjab", "Villages and fields near Ludhiana, Punjab (India)", 30.850, 75.700),
    ("bogura", "Rural villages near Bogura (Bangladesh)", 24.850, 89.350),
    ("kakamega", "Dense rural settlement near Kakamega (Kenya)", 0.250, 34.750),
    ("bahirdar", "Highland villages near Bahir Dar (Ethiopia)", 11.550, 37.450),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default="scm-ksns")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    import ee
    ee.Initialize(project=args.project)
    runs_root = Path(__file__).resolve().parents[1] / "train" / "runs"
    run_dirs = [str(runs_root / r) for r in P.PROTOTYPE["runs"]]
    models, device = P.load_models(run_dirs)
    s3 = bool(P.PROTOTYPE.get("smooth3", False))
    floor = P.noise_floor(run_dirs, P.PROTOTYPE["change_rule"], P.PROTOTYPE["tta"], smooth3=s3)
    out = Path(__file__).resolve().parent / "village_candidates"
    out.mkdir(exist_ok=True)
    only = set(args.only.split(",")) if args.only else None
    rows = []
    for cid, title, lat, lon in CANDIDATES:
        if only and cid not in only:
            continue
        print(f"{cid}: {title}")
        t0 = time.time()
        try:
            res = compute(models, device, lat, lon, START, 512, P.PROTOTYPE["tta"], smooth3=s3)
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {e}")
            continue
        res["case"] = {"id": cid, "title": title, "kind": "village test", "note": "", "floor": floor}
        with gzip.open(out / f"{cid}.pkl.gz", "wb") as f:
            pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
        pr, pb, mo, fi, la, used = P.change_inputs(res, "same_season")
        bc = P.built_change(pr, mo, pbuilt=pb, rule="moderate", first=fi, last=la)
        row = {"place": cid, "scenes": int(res["n"].sum())}
        if bc is not None:
            v, _, _ = P.change_verdict(bc, floor)
            row.update({"built % (ours)": round(bc["before_pct"], 1), "new ha": round(bc["gain_ha"], 1),
                        "lost ha": round(bc["loss_ha"], 1), "verdict": v})
            if res.get("esri_ab"):
                y0, y1, ea, eb = res["esri_ab"]
                r = P.reference_change("esri", ea == P.BUILT, eb == P.BUILT, (ea >= 0) & (eb >= 0), bc)
                row.update({"built % (ESRI)": r["built-up start (%)"], "ESRI new ha": r["new built-up (ha)"]})
            if res.get("obt"):
                o = res["obt"]
                r = P.reference_change("obt", o["built_a"], o["built_b"], o["valid"], bc)
                row.update({"OBT built %": r["built-up start (%)"], "OBT new ha": r["new built-up (ha)"]})
        rows.append(row)
        print(f"  {row}  ({time.time() - t0:.0f} s)")
    import pandas as pd
    pd.set_option("display.width", 220)
    print("\n" + pd.DataFrame(rows).set_index("place").to_string())


if __name__ == "__main__":
    main()
