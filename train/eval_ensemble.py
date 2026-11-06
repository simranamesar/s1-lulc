"""Improvements without retraining, measured on the test places (all with month-mean maps):

    single        one model                                   (baseline: E3, the prototype model)
    + TTA         test-time augmentation: each scene also predicted flipped / rotated, probabilities averaged
    ensemble      probabilities of several models averaged (different models make different mistakes)
    ensemble+TTA  both

and two rules for NEW built-up over the 12 months (first 3 vs last 3 months with data):

    argmax rule      built in most end months, in none of the start months (current prototype rule)
    confident rule   mean built-up probability >= HI over the end months and < LO in every start month

Everything is scored against the same labels (hyb_s). Usage (GPU, ~20-40 min):
    python3 eval_ensemble.py --prep-dir ../data/tiles_v0/prepared --runs runs/E3 runs/E1 runs/E5 runs/E2b
Writes runs/ensemble_eval.csv
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
from s1_composite import lee_filter_db  # noqa: E402
from train import S1_NODATA, built_gain, confusion, summarise_cm  # noqa: E402
from eval_change import score  # noqa: E402

TTA = [lambda t: t, lambda t: t.flip(-1), lambda t: t.flip(-2), lambda t: t.flip(-1).flip(-2)]   # each its own inverse


def load(run: Path, device):
    cfg = json.loads((run / "config.json").read_text())
    stats = json.loads((run / "stats.json").read_text())
    lee = bool(cfg.get("lee", False))
    m = build_from_config(cfg)
    m.load_state_dict(torch.load(run / "best.pt", map_location=device))
    mean = np.asarray(stats["mean_lee" if lee else "mean"], np.float32)[:, None, None]
    std = np.asarray(stats["std_lee" if lee else "std"], np.float32)[:, None, None]
    return {"name": run.name, "model": m.to(device).eval(), "lee": lee, "mean": mean, "std": std}


@torch.no_grad()
def month_probs(M, s1, month, device, tta: bool, batch=16):
    """Month-mean class probabilities (12, C, H, W) float32 and a data mask (12, H, W)."""
    H, W = s1.shape[-2:]
    acc = np.zeros((12, N_CLASSES, H, W), np.float32)
    cnt = np.zeros((12, H, W), np.float32)
    nod_all = (s1 == S1_NODATA).any(1)
    for i in range(0, len(s1), batch):
        ks = range(i, min(i + batch, len(s1)))
        xs = []
        for k in ks:
            sc = lee_filter_db(s1[k]) if M["lee"] else s1[k]
            x = (sc.astype(np.float32) / 100 - M["mean"]) / M["std"]
            x[sc == S1_NODATA] = 0
            xs.append(x)
        x = torch.from_numpy(np.stack(xs)).to(device)
        p = 0
        for f in (TTA if tta else TTA[:1]):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = M["model"](f(x)).float()
            p = p + f(out).softmax(1)
        p = (p / (len(TTA) if tta else 1)).cpu().numpy()
        for j, k in enumerate(ks):
            ok = ~nod_all[k]
            acc[month[k]] += p[j] * ok
            cnt[month[k]] += ok
    have = cnt > 0
    return acc / np.maximum(cnt, 1)[:, None], have


def confident_gain(pb, have, lo, hi, window=3, min_px=4):
    """New built-up from probabilities: mean P(built) >= hi over the last months, < lo in every first month."""
    from scipy import ndimage
    months = [m for m in range(12) if have[m].any()]
    w = min(window, len(months) // 2)
    if not w:
        return None
    first, last = months[:w], months[-w:]
    ok = np.all([have[m] for m in first + last], axis=0)
    end = np.mean([pb[m] for m in last], axis=0)
    start = np.max([pb[m] for m in first], axis=0)
    gain = (end >= hi) & (start < lo) & ok
    lab, n = ndimage.label(gain, structure=np.ones((3, 3), bool))
    if n:
        keep = np.bincount(lab.ravel()) >= min_px
        keep[0] = False
        gain = keep[lab]
    return gain


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prep-dir", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="first = the single-model baseline")
    ap.add_argument("--split", default="test")
    ap.add_argument("--label", default="hyb_s")
    ap.add_argument("--thresholds", default="0.5:0.2,0.6:0.2,0.7:0.3", help="HI:LO pairs for the confident rule")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prep = Path(args.prep_dir)
    models = [load(Path(r), device) for r in args.runs]
    base = models[0]["name"]
    thr = [tuple(float(v) for v in t.split(":")) for t in args.thresholds.split(",")]
    variants = {f"{base} single": ([0], False), f"{base} + TTA": ([0], True),
                f"ensemble ({'+'.join(m['name'] for m in models)})": (list(range(len(models))), False),
                "ensemble + TTA": (list(range(len(models))), True)}

    idx = pd.read_csv(prep / "index.csv")
    tiles = idx[(idx["status"] == "ok") & (idx["split"] == args.split)]
    print(f"{len(tiles)} {args.split} tiles; models {[m['name'] for m in models]}")
    acc = {v: {"cm": np.zeros((N_CLASSES, N_CLASSES), np.int64), "fl": np.zeros(4, np.int64),
               "g": np.zeros(4, np.int64), **{t: np.zeros(3, np.int64) for t in thr}} for v in variants}
    for i, r in enumerate(tiles.itertuples(), 1):
        z = np.load(prep / f"{r.tile_id}.npz")
        s1, lbl, month = z["s1"], z[args.label], z["month"].astype(int)
        cache = {}
        for v, (ms, tta) in variants.items():
            ps = []
            for mi in ms:
                if (mi, tta) not in cache:
                    cache[(mi, tta)] = month_probs(models[mi], s1, month, device, tta)
                ps.append(cache[(mi, tta)])
            prob = np.mean([p for p, _ in ps], axis=0)
            have = ps[0][1]
            maps = np.where(have, prob.argmax(1), -1).astype(np.int16)
            cm, fl, g = score(maps, lbl)
            a = acc[v]
            a["cm"] += cm; a["fl"] += fl; a["g"] += g
            gl = built_gain(np.where((lbl >= 0) & (maps >= 0), lbl, -1))
            for t in thr:
                gc = confident_gain(prob[:, BUILT], have & (lbl >= 0), t[1], t[0])
                if gc is not None and gl is not None:
                    a[t] += [int(gc.sum()), int((gc & gl).sum()), int(gc.size)]
        if i % 10 == 0:
            print(f"  {i}/{len(tiles)} tiles")

    rows = []
    for v, a in acc.items():
        o = summarise_cm(a["cm"]); cm, fl, g = a["cm"], a["fl"], a["g"]
        km2 = max(g[3], 1) * 1e-4
        base_row = {"variant": v, "OA": o["OA"], "mIoU": o["mIoU"],
                    "built_P": cm[BUILT, BUILT] / max(cm[:, BUILT].sum(), 1),
                    "built_R": cm[BUILT, BUILT] / max(cm[BUILT].sum(), 1),
                    "flip_rate": fl[0] / max(fl[1], 1), "built_false_change": fl[2] / max(fl[3], 1)}
        rows.append({**base_row, "change_rule": "argmax", "new_ha_km2": g[0] * 0.01 / km2,
                     "false_new_ha_km2": (g[0] - g[1]) * 0.01 / km2, "change_P": g[1] / g[0] if g[0] else np.nan})
        for t in thr:
            n, tp, px = a[t]
            k2 = max(px, 1) * 1e-4
            rows.append({**base_row, "change_rule": f"confident {t[0]}/{t[1]}", "new_ha_km2": n * 0.01 / k2,
                         "false_new_ha_km2": (n - tp) * 0.01 / k2, "change_P": tp / n if n else np.nan})
    df = pd.DataFrame(rows)
    out = Path(args.runs[0]).parent / "ensemble_eval.csv"
    df.to_csv(out, index=False)
    pd.set_option("display.width", 220)
    print("\nMaps (one row per variant):")
    print(df[df.change_rule == "argmax"].set_index("variant")[["OA", "mIoU", "built_P", "built_R", "flip_rate",
                                                               "built_false_change"]].round(3).to_string())
    print("\nNew built-up over 12 months (vs labels; 'false' = not new in the labels):")
    print(df.set_index(["variant", "change_rule"])[["new_ha_km2", "false_new_ha_km2", "change_P"]].round(3).to_string())
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
