"""Turn exported tiles into training-ready .npz files, clean the labels, compute statistics.

Input  : <raw-dir>/<tile_id>.tif from export_tiles.py (monthly labels), tiles.csv, and
         <scenes-dir>/<tile_id>.npz from export_scenes.py (single raw Sentinel-1 scenes = model input)
Output : <out-dir>/<tile_id>.npz with
             s1     int16 (S, 3, 256, 256)   one raw scene each: VV, VH, VH-VV in dB x 100, -32768 = no data
             month  int8  (S,)               month index 0-11 of each scene (its label = that month's label)
             day    int8  (S,)               day of month
             orbit  str   (S,)               ASCENDING / DESCENDING
             valid  float (S,)               share of the tile covered
             best   int32 (12,)              scene used for each month's map (full coverage, nearest mid-month)
             n      int16 (12,)              scenes per month
             dw     int8  (12, 256, 256)     Dynamic World label, cleaned        (E0 label)
             hyb    int8  (12, 256, 256)     hybrid label (DW + buildings), cleaned
             hyb_s  int8  (12, 256, 256)     hybrid label, seasonally smoothed  (E1 label)
             hyb_y  int8  (12, 256, 256)     annual hybrid label: each pixel's most frequent class over the year,
                                             the same in every month (stable, WorldCover-style; variant E1y)
             bfrac  uint8 (256, 256)         roof cover % (Open Buildings)
         <out-dir>/index.csv   one row per tile with coverage and label-flip statistics
         <out-dir>/stats.json  normalisation mean/std (train split) and class counts per label type

Classes: the 9 Dynamic World classes are merged into 6 (data/classes.py): water, trees, grass & shrub
(+ flooded vegetation), crops, built-up, bare & snow.

Label cleaning (all label types)
    Dynamic World snow/ice is set to "ignore" (-1) outside cold biomes (stray snow pixels in the tropics),
    then the 9 classes are merged into 6.

Seasonal smoothing (hyb_s only)
    1. Seasonal pairs: a pixel whose labelled months are only trees and grass & shrub (both present) gets its
       yearly majority of the two in every labelled month. Deciduous forest losing leaves is not land-cover change.
    2. 3-month window: each labelled month takes the most common class of (previous, current, next) month,
       with the current month weighted 1.5 so ties keep the original label. Months without a label stay -1.

Statistics: per-band mean/std over all train scenes, raw ("mean"/"std") and after the 5 x 5 Lee filter
("mean_lee"/"std_lee", from every 4th train scene) for the speckle-filter variant.

Without --scenes-dir the old monthly-median composites from the .tif are used (earlier approach).

Usage
    python3 prepare_data.py --tiles tiles_v0/tiles.csv --raw-dir tiles_v0/raw --scenes-dir tiles_v0/scenes \
                            --out-dir tiles_v0/prepared
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from classes import BUILT, CLASS_NAMES, DW_SNOW as SNOW, GRASS_SHRUB, N_CLASSES, TREES, from_dw
S1_NODATA = -32768
# RESOLVE biomes where snow is plausible: temperate forests, boreal, temperate grassland, montane, tundra.
COLD_BIOMES = {4, 5, 6, 8, 10, 11}
SEASONAL_PAIRS = ((TREES, GRASS_SHRUB),)   # deciduous / dry forest read as shrub in the leafless season


def read_tile(path: Path) -> dict[str, np.ndarray]:
    import rasterio
    from export_tiles import all_band_names

    with rasterio.open(path) as src:
        a = src.read()
        names = list(src.descriptions)
    if not all(names):
        names = all_band_names()
    idx = {n: i for i, n in enumerate(names)}

    def band(month: int, key: str) -> np.ndarray:
        return a[idx[f"m{month:02d}_{key}"]]

    months = range(1, 13)
    return {
        "s1": np.stack([np.stack([band(m, k) for k in ("VV", "VH", "VHVV")]) for m in months]).astype(np.int16),
        "n": np.array([band(m, "N").max() for m in months], dtype=np.int16),
        "dw": np.stack([band(m, "DW") for m in months]).astype(np.int8),
        "hyb": np.stack([band(m, "HYB") for m in months]).astype(np.int8),
        "bfrac": np.clip(a[idx["BFRAC"]], 0, 100).astype(np.uint8),
    }


def drop_snow(lbl: np.ndarray, biome: int) -> np.ndarray:
    if biome in COLD_BIOMES:
        return lbl
    out = lbl.copy()
    out[out == SNOW] = -1
    return out


def smooth_labels(lbl: np.ndarray, pairs=SEASONAL_PAIRS, centre_weight: float = 1.5) -> np.ndarray:
    """Seasonal-pair fix, then a 3-month weighted mode. lbl: int8 (T, H, W), -1 = no label."""
    T = lbl.shape[0]
    valid = lbl >= 0
    out = lbl.copy()

    n_valid = valid.sum(0)
    for a, b in pairs:
        ca = (lbl == a).sum(0)
        cb = (lbl == b).sum(0)
        only_pair = (ca + cb == n_valid) & (ca > 0) & (cb > 0)
        majority = np.where(ca >= cb, a, b).astype(np.int8)
        for t in range(T):
            m = only_pair & valid[t]
            out[t][m] = majority[m]

    smoothed = out.copy()
    for t in range(T):
        votes = np.zeros((N_CLASSES,) + lbl.shape[1:], dtype=np.float32)
        for s, w in ((t - 1, 1.0), (t, centre_weight), (t + 1, 1.0)):
            if 0 <= s < T:
                lab = out[s]
                for c in range(N_CLASSES):
                    votes[c] += (lab == c) * w
        mode = votes.argmax(0).astype(np.int8)
        smoothed[t] = np.where(valid[t], mode, -1)
    return smoothed


def annual_label(lbl: np.ndarray) -> np.ndarray:
    """Most frequent class per pixel over the labelled months, repeated for all months (-1 if never labelled)."""
    counts = np.stack([(lbl == c).sum(0) for c in range(N_CLASSES)])
    year = counts.argmax(0).astype(np.int8)
    year[counts.sum(0) == 0] = -1
    return np.broadcast_to(year, lbl.shape).copy()


def flip_rate(lbl: np.ndarray) -> float:
    """Share of pixels whose label changes between consecutive months (where both months are labelled)."""
    changed = total = 0
    for t in range(lbl.shape[0] - 1):
        both = (lbl[t] >= 0) & (lbl[t + 1] >= 0)
        changed += int(((lbl[t] != lbl[t + 1]) & both).sum())
        total += int(both.sum())
    return changed / total if total else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--scenes-dir", default="", help="single-scene .npz from export_scenes.py (model input)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--keep-snow", action="store_true", help="do not drop snow/ice outside cold biomes")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    tiles = pd.read_csv(args.tiles)
    raw, out = Path(args.raw_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    s1_sum = np.zeros(3)
    s1_sq = np.zeros(3)
    s1_cnt = np.zeros(3)
    lee_sum, lee_sq, lee_cnt = np.zeros(3), np.zeros(3), np.zeros(3)
    flattened_flags: set[bool] = set()
    scenes_dir = Path(args.scenes_dir) if args.scenes_dir else None
    if scenes_dir is not None:
        from s1_composite import best_scene_per_month, lee_filter_db
    class_counts = {k: np.zeros(N_CLASSES, dtype=np.int64) for k in ("dw", "hyb", "hyb_s", "hyb_y")}

    for i, r in enumerate(tiles.itertuples(), 1):
        tif = raw / f"{r.tile_id}.tif"
        if not tif.exists():
            rows.append({"tile_id": r.tile_id, "status": "missing tif"})
            continue
        t = read_tile(tif)
        if scenes_dir is not None:
            sc_path = scenes_dir / f"{r.tile_id}.npz"
            if not sc_path.exists():
                rows.append({"tile_id": r.tile_id, "status": "missing scenes"})
                continue
            sc = np.load(sc_path)
            dates = sc["date"].astype(str)
            month = np.array([int(d[5:7]) - 1 for d in dates], dtype=np.int8)
            day = np.array([int(d[8:10]) for d in dates], dtype=np.int8)
            t["s1"], t["month"], t["day"] = sc["s1"], month, day
            t["orbit"], t["valid"] = sc["orbit"].astype(str), sc["valid"]
            flattened_flags.add(bool(sc["flattened"]) if "flattened" in sc.files else False)
            t["n"] = np.bincount(month, minlength=12)[:12].astype(np.int16)
            t["best"] = best_scene_per_month(month, day, sc["valid"])
        for key in ("dw", "hyb"):
            if not args.keep_snow:
                t[key] = drop_snow(t[key], int(r.biome))    # still Dynamic World ids here
            t[key] = from_dw(t[key])                        # 9 Dynamic World classes -> 6 classes
        # No S1 that month -> nothing to learn from; keep label but the loader skips the month.
        t["hyb_s"] = smooth_labels(t["hyb"])
        t["hyb_y"] = annual_label(t["hyb"])

        np.savez_compressed(out / f"{r.tile_id}.npz", **t)
        n_scenes = len(t["s1"]) if "month" in t else int((t["n"] > 0).sum())

        s1_months = int((t["n"] > 0).sum())
        label_months = int(((t["dw"] >= 0).mean(axis=(1, 2)) >= 0.2).sum())
        has_s1 = t["n"] > 0
        cls_counts = {}
        for key in ("dw", "hyb_s"):
            lab = t[key][has_s1]
            bc = np.bincount(lab[lab >= 0].ravel(), minlength=N_CLASSES)[:N_CLASSES]
            cls_counts.update({f"px_{key}_{c}": int(v) for c, v in enumerate(bc)})
        rows.append({
            "tile_id": r.tile_id, "status": "ok", "split": r.split, "settlement": r.settlement,
            "region": r.region, "biome": r.biome,
            "s1_months": s1_months, "label_months": label_months, "scenes": n_scenes,
            "roof_cover_pct": float(t["bfrac"].mean()),
            "built_px_dw": int((t["dw"] == BUILT).sum()), "built_px_hyb": int((t["hyb"] == BUILT).sum()),
            "flip_dw": flip_rate(t["dw"]), "flip_hyb": flip_rate(t["hyb"]), "flip_hyb_s": flip_rate(t["hyb_s"]),
            **cls_counts,
        })

        if r.split == "train":
            # samples = scenes (each with its month's label), or months for the old composites
            samples = ([(k, int(t["month"][k])) for k in range(len(t["s1"]))] if "month" in t
                       else [(m, m) for m in range(12) if t["n"][m] > 0])
            for j, (k, m) in enumerate(samples):
                x = t["s1"][k].astype(np.float64) / 100
                ok = t["s1"][k] != S1_NODATA
                for c in range(3):
                    v = x[c][ok[c]]
                    s1_sum[c] += v.sum()
                    s1_sq[c] += (v ** 2).sum()
                    s1_cnt[c] += v.size
                if scenes_dir is not None and j % 4 == 0:
                    xl = lee_filter_db(t["s1"][k]).astype(np.float64) / 100
                    for c in range(3):
                        v = xl[c][ok[c]]
                        lee_sum[c] += v.sum()
                        lee_sq[c] += (v ** 2).sum()
                        lee_cnt[c] += v.size
                for key in class_counts:
                    lab = t[key][m][ok.all(0)]
                    class_counts[key] += np.bincount(lab[lab >= 0], minlength=N_CLASSES)[:N_CLASSES]

        if i % 25 == 0 or i == len(tiles):
            print(f"  {i}/{len(tiles)} tiles prepared")

    idx = pd.DataFrame(rows)
    idx.to_csv(out / "index.csv", index=False)

    mean = s1_sum / np.maximum(s1_cnt, 1)
    std = np.sqrt(np.maximum(s1_sq / np.maximum(s1_cnt, 1) - mean ** 2, 1e-6))
    stats = {
        "channels": ["VV_dB", "VH_dB", "VHVV_dB"],
        "mean": mean.round(4).tolist(), "std": std.round(4).tolist(),
        "input": "single_scenes" if scenes_dir is not None else "monthly_median",
        "terrain_flattened": (flattened_flags == {True}),
        "class_names": CLASS_NAMES,
        "class_counts_train": {k: v.tolist() for k, v in class_counts.items()},
        "snow_dropped_outside_cold_biomes": not args.keep_snow,
    }
    if lee_cnt.sum():
        lm = lee_sum / np.maximum(lee_cnt, 1)
        stats["mean_lee"] = lm.round(4).tolist()
        stats["std_lee"] = np.sqrt(np.maximum(lee_sq / np.maximum(lee_cnt, 1) - lm ** 2, 1e-6)).round(4).tolist()
    (out / "stats.json").write_text(json.dumps(stats, indent=2))

    ok = idx[idx["status"] == "ok"]
    print(f"\nPrepared {len(ok)} of {len(tiles)} tiles -> {out}")
    if len(flattened_flags) > 1:
        print("WARNING: some scene files are terrain-flattened and some are not. Re-export them consistently.")
    print(f"Terrain flattening (gamma0): {'yes' if stats['terrain_flattened'] else 'no'}")
    print(f"Input: {stats['input']}; scenes per tile: median {ok['scenes'].median():.0f}, total {ok['scenes'].sum():.0f}")
    print(f"S1 mean (dB): {stats['mean']}  std: {stats['std']}")
    if "std_lee" in stats:
        print(f"After Lee 5x5 filter: mean {stats['mean_lee']}  std {stats['std_lee']} (less speckle = lower std)")
    print("\nMonthly label flip rate (share of pixels changing class month to month), mean over tiles:")
    print(f"  Dynamic World raw : {ok['flip_dw'].mean():.3f}")
    print(f"  hybrid raw        : {ok['flip_hyb'].mean():.3f}")
    print(f"  hybrid smoothed   : {ok['flip_hyb_s'].mean():.3f}")
    print("\nBuilt pixels, Dynamic World vs hybrid, per settlement type (train+val+test):")
    print(ok.groupby("settlement")[["built_px_dw", "built_px_hyb"]].sum().to_string())
    tot = class_counts["hyb_s"]
    print("\nTrain class shares (hyb_s): " + ", ".join(
        f"{n} {c / max(tot.sum(), 1) * 100:.1f}%" for n, c in zip(CLASS_NAMES, tot)))


if __name__ == "__main__":
    main()
