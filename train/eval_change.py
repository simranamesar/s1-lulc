"""How should the 12 monthly maps be built from single raw scenes? Compared on the test tiles, no retraining.

The model always reads ONE raw scene at a time. What changes is how its per-scene predictions become one map per
month (the prototype shows 12 maps and measures change between them):

    A  best scene      one scene per month (full coverage, nearest mid-month)  <- what E1 was tested with
    D  best, 1 orbit   as A, but only scenes from the tile's most common orbit direction (asc or desc)
    B  month mean      class probabilities averaged over ALL scenes of the month, then argmax
    C  mean, 1 orbit   as B, only the most common orbit direction

For each: monthly OA / mIoU vs the monthly label, built-up precision / recall, month-to-month flip rate,
built-up false change rate, and new built-up over 12 months with the prototype's rule (first 3 vs last 3 months).

Usage (GPU 0, about 5 min):
    python3 eval_change.py --run runs/E1 --prep-dir ../data/tiles_v0/prepared
Writes runs/E1/change_eval.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from classes import BUILT, N_CLASSES  # noqa: E402
from model import build_from_config  # noqa: E402
from s1_composite import best_scene_per_month, lee_filter_db  # noqa: E402
from train import S1_NODATA, built_gain, confusion, summarise_cm  # noqa: E402

STRATEGIES = {"A best scene": ("best", False), "D best, 1 orbit": ("best", True),
              "B month mean": ("mean", False), "C mean, 1 orbit": ("mean", True)}


@torch.no_grad()
def scene_probs(model, s1, mean, std, device, lee, batch=16):
    """Softmax per scene, float16 (S, C, H, W); nodata pixels -> NaN-free zeros (masked later)."""
    out = []
    for i in range(0, len(s1), batch):
        xs = []
        for k in range(i, min(i + batch, len(s1))):
            sc = lee_filter_db(s1[k]) if lee else s1[k]
            x = (sc.astype(np.float32) / 100 - mean) / std
            x[sc == S1_NODATA] = 0
            xs.append(x)
        x = torch.from_numpy(np.stack(xs)).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            p = model(x).float().softmax(1)
        out.append(p.half().cpu().numpy())
    return np.concatenate(out)


def monthly_maps(probs, nod, month, day, valid, orbit, how, one_orbit):
    S, C, H, W = probs.shape
    use = np.ones(S, bool)
    if one_orbit:
        vals, cnt = np.unique(orbit, return_counts=True)
        use = orbit == vals[cnt.argmax()]
    maps = np.full((12, H, W), -1, np.int16)
    if how == "best":
        sub = np.where(use)[0]
        best = best_scene_per_month(month[sub], day[sub], valid[sub])
        for m in range(12):
            if best[m] >= 0:
                k = sub[best[m]]
                maps[m] = np.where(nod[k], -1, probs[k].argmax(0))
    else:
        for m in range(12):
            ks = np.where(use & (month == m))[0]
            if not len(ks):
                continue
            ok = ~nod[ks]                                            # (n, H, W)
            acc = (probs[ks].astype(np.float32) * ok[:, None]).sum(0)
            have = ok.any(0)
            maps[m] = np.where(have, acc.argmax(0), -1)
    return maps


def score(maps, lbl):
    cm = np.zeros((N_CLASSES, N_CLASSES), np.int64)
    flips = flip_px = bfl = bpx = 0
    for m in range(12):
        ok = (maps[m] >= 0) & (lbl[m] >= 0)
        if ok.any():
            cm += confusion(maps[m][ok].astype(np.int64), lbl[m][ok].astype(np.int64))
    for m in range(11):
        a, b, ya, yb = maps[m], maps[m + 1], lbl[m], lbl[m + 1]
        both = (a >= 0) & (b >= 0)
        flips += int(((a != b) & both).sum()); flip_px += int(both.sum())
        st = both & (ya >= 0) & (yb >= 0) & ((ya == BUILT) == (yb == BUILT))
        bfl += int((((a == BUILT) != (b == BUILT)) & st).sum()); bpx += int(st.sum())
    seq = np.where(lbl >= 0, maps, -1)
    gp, gl = built_gain(seq), built_gain(np.where(seq >= 0, lbl, -1))
    g = (0, 0, 0, 0) if gp is None or gl is None else (int(gp.sum()), int((gp & gl).sum()), int(gl.sum()), gp.size)
    return cm, (flips, flip_px, bfl, bpx), g


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--prep-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    run, prep = Path(args.run), Path(args.prep_dir)
    cfg = json.loads((run / "config.json").read_text())
    stats = json.loads((run / "stats.json").read_text())
    lee = bool(cfg.get("lee", False))
    mean = np.asarray(stats["mean_lee" if lee else "mean"], np.float32)[:, None, None]
    std = np.asarray(stats["std_lee" if lee else "std"], np.float32)[:, None, None]
    label = args.label or cfg.get("label", "hyb_s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_from_config(cfg)
    model.load_state_dict(torch.load(run / "best.pt", map_location=device))
    model.to(device).eval()

    idx = pd.read_csv(prep / "index.csv")
    tiles = idx[(idx["status"] == "ok") & (idx["split"] == args.split)]
    print(f"{run.name}: {len(tiles)} {args.split} tiles, label {label}, lee {lee}")
    acc = {s: [np.zeros((N_CLASSES, N_CLASSES), np.int64), np.zeros(4, np.int64), np.zeros(4, np.int64)]
           for s in STRATEGIES}
    for i, r in enumerate(tiles.itertuples(), 1):
        z = np.load(prep / f"{r.tile_id}.npz")
        s1, lbl = z["s1"], z[label]
        month, day, valid = z["month"].astype(int), z["day"].astype(int), z["valid"]
        orbit = z["orbit"].astype(str)
        probs = scene_probs(model, s1, mean, std, device, lee)
        nod = (s1 == S1_NODATA).any(1)
        for s, (how, one) in STRATEGIES.items():
            cm, fl, g = score(monthly_maps(probs, nod, month, day, valid, orbit, how, one), lbl)
            acc[s][0] += cm; acc[s][1] += fl; acc[s][2] += g
        if i % 20 == 0:
            print(f"  {i}/{len(tiles)} tiles")

    rows = []
    for s, (cm, fl, g) in acc.items():
        o = summarise_cm(cm)
        km2 = g[3] * 1e-4
        rows.append({"strategy": s, "OA": o["OA"], "mIoU": o["mIoU"],
                     "built_P": cm[BUILT, BUILT] / max(cm[:, BUILT].sum(), 1),
                     "built_R": cm[BUILT, BUILT] / max(cm[BUILT].sum(), 1),
                     "flip_rate": fl[0] / max(fl[1], 1), "built_false_change": fl[2] / max(fl[3], 1),
                     "new_built_pred_ha_km2": g[0] * 0.01 / max(km2, 1e-9),
                     "new_built_label_ha_km2": g[2] * 0.01 / max(km2, 1e-9),
                     "false_new_ha_km2": (g[0] - g[1]) * 0.01 / max(km2, 1e-9),
                     "change_P": g[1] / g[0] if g[0] else float("nan"),
                     "change_R": g[1] / g[2] if g[2] else float("nan")})
    df = pd.DataFrame(rows).set_index("strategy")
    df.to_csv(run / "change_eval.csv")
    pd.set_option("display.width", 200)
    print("\n" + df.round(3).to_string())
    print(f"\nSaved {run / 'change_eval.csv'}")


if __name__ == "__main__":
    main()
