"""Print every number of the pre-computed gallery places (no downloading, reads case_studies/*.pkl.gz).

    python3 summarize_cases.py                 # all places
    python3 summarize_cases.py lagos nairobi   # some
"""
import gzip
import pickle
import sys
from pathlib import Path

import pandas as pd

import pipeline as P

rows = []
for f in sorted(Path(__file__).resolve().parent.joinpath("case_studies").glob("*.pkl.gz")):
    cid = f.name.split(".")[0]
    if len(sys.argv) > 1 and cid not in sys.argv[1:]:
        continue
    with gzip.open(f, "rb") as fh:
        res = pickle.load(fh)
    row = {"place": cid, "scenes": int(res["n"].sum())}
    arg = None
    for mode, tag in (("within_year", "W"), ("same_season", "S")):
        pr, pb, mo, fi, la, used = P.change_inputs(res, mode)
        if used != mode:
            row[f"{tag}: note"] = "extra months missing"
            continue
        for rule in ("argmax", "moderate", "confident"):
            bc = P.built_change(pr, mo, pbuilt=pb, rule=rule, first=fi, last=la)
            if bc is None:
                continue
            if rule == "argmax":
                row[f"{tag}: built start/end %"] = f"{bc['before_pct']:.1f} / {bc['after_pct']:.1f}"
                if mode == "within_year":
                    arg = bc
            row[f"{tag}: new / lost ha ({rule})"] = f"{bc['gain_ha']:.1f} / {bc['loss_ha']:.1f}"
    if arg is None:
        row["note"] = "no Sentinel-1 data in enough months"
        rows.append(row)
        continue
    if res.get("obt"):
        o = res["obt"]
        r = P.reference_change("obt", o["built_a"], o["built_b"], o["valid"], arg)
        row.update({"OBT years": f"{o['date_a'][:4]}-{o['date_b'][:4]}", "OBT built start %": r["built-up start (%)"],
                    "OBT new ha": r["new built-up (ha)"], "OBT confirms argmax-new %": r["our new built-up that it confirms (%)"]})
    if res.get("esri_ab"):
        y0, y1, ea, eb = res["esri_ab"]
        r = P.reference_change("esri", ea == P.BUILT, eb == P.BUILT, (ea >= 0) & (eb >= 0), arg)
        row.update({"ESRI built start %": r["built-up start (%)"], "ESRI new ha": r["new built-up (ha)"]})
    rows.append(row)

pd.set_option("display.width", 250)
print(pd.DataFrame(rows).set_index("place").round(1).T.to_string())
