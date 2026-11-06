"""Print one comparison table over finished runs (for the report and the Project Log).

Usage:  python3 compare_runs.py runs E0 E1 E2a E2b E3
"""
import json
import sys
from pathlib import Path

import pandas as pd

root, names = Path(sys.argv[1]), sys.argv[2:]
rows = []
for n in names:
    f = root / n / "test_metrics.json"
    if not f.exists():
        print(f"{n}: no test_metrics.json yet")
        continue
    m = json.loads(f.read_text())
    cfg = json.loads((root / n / "config.json").read_text())
    o = m["overall"]
    row = {"run": n, "label": cfg["label"], "weights": cfg["encoder_weights"],
           "frozen": cfg.get("freeze_encoder", False), "upsample": cfg.get("upsample", "interp"),
           "lee": cfg.get("lee", False),
           "OA": o.get("OA", o.get("pixel_acc")), "mIoU": o["mIoU"], "macro_F1": o.get("macro_F1")}
    for k, v in m.get("per_settlement", {}).items():
        row[f"mIoU_{k}"] = v["mIoU"]
    vb = m.get("village_built", {})
    row["village_built_recall"] = vb.get("recall")
    bu = m.get("built", {})
    row["built_P"], row["built_R"], row["built_false_change"] = bu.get("precision"), bu.get("recall"), bu.get("false_change_rate")
    row["false_new_ha_km2"] = m.get("built_gain", {}).get("false_new_ha_per_km2")
    fr = m.get("flip_rate", {})
    row["flip_pred"], row["flip_label"] = fr.get("prediction"), fr.get("label")
    row["best_epoch"] = m.get("best_epoch")
    rows.append(row)
df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(df.round(3).to_string(index=False))
df.to_csv(root / "comparison.csv", index=False)
print(f"\nWrote {root}/comparison.csv")
