"""Prototype processing: location + start month -> 12 monthly LULC maps, class %, changes, independent check.

Steps (the report's "Prototype" section follows this order):
    1. grid      a square of --size px x 10 m centred on the point, in the local UTM zone
    2. fetch     for each of the 12 months EVERY raw Sentinel-1 scene over the area (no input averaging), prepared
                 with the SAME recipe as the training data (data/s1_composite.py)
    3. predict   the trained U-Net (train/model.py) reads each scene on its own, exactly as in training; the
                 month's map = class with the highest probability averaged over that month's scenes
                 ("month mean"). On the 130 test tiles this beat one scene per month: mIoU 0.492 -> 0.522,
                 month-to-month flip rate 25% -> 15%, built-up false change 4.4% -> 2.2% (train/eval_change.py).
                 (Lee speckle filter only if that model was trained with --lee)
    4. smooth    optional: 3-month majority + trees/shrub seasonal fix (same rule as the training labels)
    5. compute   % of valid pixels per class per month; month-to-month change and first-to-last change in
                 percentage points (pp)
    6. check     ESRI 10 m annual land cover (never used in training) vs our year-majority map, on the same 6 classes
    7. built-up  the headline (end goal): robust built-up change over the 12 months. A pixel is NEW built-up only
                 if it is built-up in most of the last 3 months with data and in none of the first 3; LOST
                 built-up the other way round. This ignores one-month flicker, which would otherwise look like
                 construction. Checked against two independent yearly products: Open Buildings 2.5D Temporal
                 (Africa, South / South-East Asia, Latin America; yearly 2016-2023) and ESRI 10 m annual maps.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "data"), str(ROOT / "train")]

from s1_composite import (S1_BANDS, S1_NODATA, best_scene_per_month, dem_slope_aspect, lee_filter_db,  # noqa: E402
                          s1_scene_image, s1_scene_images, s1_scene_table, valid_fraction)

from classes import BUILT, CLASS_COLOURS, DISPLAY_NAMES as CLASS_NAMES, ESRI_ORIGINAL, ESRI_REMAPPED, N_CLASSES  # noqa: E402,E501
PIXEL_M = 10

ESRI_COLLECTION = "projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS"
OBT_COLLECTION = "GOOGLE/Research/open-buildings-temporal/v1"
PIXEL_HA = PIXEL_M * PIXEL_M / 10_000       # one 10 m pixel = 0.01 ha

# Case-study presets (approximate centres; fast-growing city edges are where yearly built-up change is visible)
PRESETS = {
    "Lagos, Lekki\u2013Ajah corridor (Nigeria)": (6.468, 3.630),
    "Nairobi, Kitengela (Kenya)": (-1.474, 36.958),
    "Dhaka, Purbachal (Bangladesh)": (23.835, 90.520),
    "Heidelberg (Germany, little change)": (49.4094, 8.6942),
}
# ESRI codes -> our 6 classes: see data/classes.py (ESRI_ORIGINAL; ESRI_REMAPPED for catalog versions storing 1..9)


# ----------------------------------------------------------------------------- 1. grid

def utm_epsg(lon: float, lat: float) -> int:
    return (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1


def area_grid(lat: float, lon: float, size_px: int) -> dict:
    """Square grid of size_px x size_px 10 m pixels centred on (lat, lon), snapped to 10 m in local UTM."""
    from pyproj import Transformer
    epsg = utm_epsg(lon, lat)
    x, y = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform(lon, lat)
    half = size_px * PIXEL_M / 2
    x0 = round((x - half) / PIXEL_M) * PIXEL_M
    y_top = round((y + half) / PIXEL_M) * PIXEL_M
    return {"epsg": epsg, "x0": x0, "y_top": y_top, "size": size_px}


def month_list(start_ym: str, n: int = 12) -> list[str]:
    """'2024-03' -> ['2024-03', ..., '2025-02'] (n consecutive months)."""
    y, m = map(int, start_ym.split("-"))
    out = []
    for i in range(n):
        mm = m - 1 + i
        out.append(f"{y + mm // 12:04d}-{mm % 12 + 1:02d}")
    return out


# ----------------------------------------------------------------------------- 2. fetch

def _ee_grid(g: dict) -> dict:
    return {"dimensions": {"width": g["size"], "height": g["size"]},
            "affineTransform": {"scaleX": PIXEL_M, "shearX": 0, "translateX": g["x0"],
                                "shearY": 0, "scaleY": -PIXEL_M, "translateY": g["y_top"]},
            "crsCode": f"EPSG:{g['epsg']}"}


def _region(g: dict):
    import ee
    s = g["size"] * PIXEL_M
    return ee.Geometry.Rectangle([g["x0"], g["y_top"] - s, g["x0"] + s, g["y_top"]],
                                 ee.Projection(f"EPSG:{g['epsg']}"), False)


def fetch_s1(g: dict, start_ym: str, workers: int = 6,
             terrain_flatten: bool = True) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """One raw scene per month (terrain-flattened like the training data unless the model was trained without).

    Returns s1 int16 (12, 3, H, W) in dB x 100 (nodata -32768; all nodata for a month without a scene),
    n (12,) number of scenes available that month, and per-month info {date, orbit} of the scene used.
    """
    import ee

    region = _region(g)
    months = month_list(start_ym)

    def one(ym: str):
        y, m = map(int, ym.split("-"))
        start = ee.Date.fromYMD(y, m, 1)
        end = start.advance(1, "month")
        mid = start.advance(14, "day").millis()
        col = s1_scene_images(region, start, end, terrain_flatten=terrain_flatten)
        def score(im):
            # share of the area with data (a mosaicked scene has no usable footprint, so measure it)
            cover = ee.Number(im.select("VV").neq(S1_NODATA)
                              .reduceRegion(ee.Reducer.mean(), region, 100, bestEffort=True).get("VV"))
            full = cover.gte(0.999)
            return im.set("score", ee.Number(ee.Algorithms.If(full, 0, 1e13))
                          .add(ee.Number(im.get("system:time_start")).subtract(mid).abs()))

        scored = col.map(score)
        best = ee.Image(scored.sort("score").first())
        info = ee.Dictionary({"n": col.size(), "date": best.get("date"), "orbit": best.get("pass")}).getInfo()
        if not info["n"]:
            empty = np.full((3, g["size"], g["size"]), S1_NODATA, np.int16)
            return empty, 0, {"date": None, "orbit": None}
        arr = ee.data.computePixels({"expression": best.select(S1_BANDS), "fileFormat": "NUMPY_NDARRAY",
                                     "grid": _ee_grid(g)})
        return (np.stack([arr[b] for b in S1_BANDS]).astype(np.int16), int(info["n"]),
                {"date": info["date"], "orbit": info["orbit"]})

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        res = list(pool.map(one, months))
    return (np.stack([r[0] for r in res]), np.array([r[1] for r in res], dtype=np.int16),
            [r[2] for r in res])


def fetch_s1_all(g: dict, start_ym: str, workers: int = 6, terrain_flatten: bool = True,
                 min_valid: float = 0.5, chunk: int = 6, n_months: int = 12):
    """EVERY raw scene of each of the 12 months (same recipe as training), for the month-mean prediction.

    Returns scenes: list of 12 int16 arrays (k, 3, H, W) in dB x 100 (k = 0 for a month without data),
    n (12,) scenes used per month, info: per month {date, orbit (of the best scene, for display), dates, orbits},
    and s1_show (12, 3, H, W): the best scene per month (full coverage, nearest mid-month) for the input tab.
    """
    import datetime as dt
    import ee

    region = _region(g)
    months = month_list(start_ym, n_months)
    dem = None if not terrain_flatten else dem_slope_aspect(region)
    empty = np.full((3, g["size"], g["size"]), S1_NODATA, np.int16)

    def one(ym: str):
        y, m = map(int, ym.split("-"))
        start = f"{y:04d}-{m:02d}-01"
        end = (dt.date(y, m, 28) + dt.timedelta(days=4)).replace(day=1).isoformat()
        table = s1_scene_table(region, start, end)
        arrs = []
        for i in range(0, len(table), chunk):
            part = table[i:i + chunk]
            img = ee.Image.cat([s1_scene_image(region, start, end, sc["ids"], terrain_flatten, dem)
                                .rename([f"s{j}_{b}" for b in S1_BANDS]) for j, sc in enumerate(part)])
            arr = ee.data.computePixels({"expression": img, "fileFormat": "NUMPY_NDARRAY", "grid": _ee_grid(g)})
            arrs += [np.stack([arr[f"s{j}_{b}"] for b in S1_BANDS]).astype(np.int16) for j in range(len(part))]
        if not arrs:
            return np.zeros((0, 3, g["size"], g["size"]), np.int16), [], empty
        s1 = np.stack(arrs)
        valid = valid_fraction(s1)
        keep = valid >= min_valid
        table = [sc for sc, k in zip(table, keep) if k]
        s1, valid = s1[keep], valid[keep]
        if not len(s1):
            return s1, [], empty
        days = np.array([int(sc["date"][8:10]) for sc in table])
        b = best_scene_per_month(np.zeros(len(s1), int), days, valid, n_months=1)[0]
        return s1, [table[b]] + [sc for k, sc in enumerate(table) if k != b], s1[b]

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        res = list(pool.map(one, months))
    scenes = [r[0] for r in res]
    n = np.array([len(r[0]) for r in res], dtype=np.int16)
    info = [{"date": r[1][0]["date"] if r[1] else None, "orbit": r[1][0]["pass"] if r[1] else None,
             "dates": sorted(sc["date"] for sc in r[1]), "orbits": sorted({sc["pass"] for sc in r[1]})}
            for r in res]
    return scenes, n, info, np.stack([r[2] for r in res])


def fetch_esri(g: dict, year: int) -> np.ndarray:
    """ESRI 10 m annual land cover for `year` in our 6 classes (H, W), -1 = clouds / no data."""
    import ee
    img = (ee.ImageCollection(ESRI_COLLECTION).filterDate(f"{year}-01-01", f"{year + 1}-01-01")
           .mosaic().rename("lc").unmask(0).toInt16())
    arr = ee.data.computePixels({"expression": img, "fileFormat": "NUMPY_NDARRAY", "grid": _ee_grid(g)})["lc"]
    return remap_esri(arr)


def fetch_obt_built(g: dict, start_ym: str, end_ym: str, presence: float = 0.5,
                    min_cover: float = 0.25) -> dict | None:
    """Open Buildings 2.5D Temporal: built-up (bool, H x W) at the yearly snapshots bracketing the 12 months.

    A 10 m pixel counts as built when >= `min_cover` of it has building presence >= `presence` (4 m product).
    Snapshots are dated 30 June each year (2016-2023). Returns None where the product has no coverage.
    """
    import ee
    region = _region(g)
    col = ee.ImageCollection(OBT_COLLECTION).filterBounds(region)
    times = sorted(set(col.aggregate_array("system:time_start").getInfo() or []))
    if not times:
        return None
    import datetime as dt
    to_date = lambda t: dt.datetime.fromtimestamp(t / 1000, dt.timezone.utc).date()  # noqa: E731
    start = dt.date(int(start_ym[:4]), int(start_ym[5:7]), 1)
    end = dt.date(int(end_ym[:4]), int(end_ym[5:7]), 28)
    before = [t for t in times if to_date(t) <= start] or times[:1]
    after = [t for t in times if to_date(t) >= end] or times[-1:]
    ta, tb = before[-1], after[0]
    if ta == tb:
        return None

    def built(t):
        imgs = col.filter(ee.Filter.eq("system:time_start", t))
        pres = (imgs.mosaic().select("building_presence").gte(presence)
                .setDefaultProjection(imgs.first().projection()))
        cover = pres.reduceResolution(ee.Reducer.mean(), maxPixels=1024).unmask(-1).rename("b")
        arr = ee.data.computePixels({"expression": cover, "fileFormat": "NUMPY_NDARRAY", "grid": _ee_grid(g)})["b"]
        return arr

    a, b = built(ta), built(tb)
    valid = (a >= 0) & (b >= 0)
    if valid.mean() < 0.5:
        return None
    return {"date_a": str(to_date(ta)), "date_b": str(to_date(tb)), "valid": valid,
            "built_a": (a >= min_cover) & valid, "built_b": (b >= min_cover) & valid}


def remap_esri(codes: np.ndarray) -> np.ndarray:
    vals = set(np.unique(codes).tolist())
    table = ESRI_REMAPPED if (vals & {3, 6}) and not (vals & {10, 11}) else ESRI_ORIGINAL
    out = np.full(codes.shape, -1, dtype=np.int8)
    for k, v in table.items():
        out[codes == k] = v
    return out


# ----------------------------------------------------------------------------- 3-4. predict, smooth

def load_model(run_dir: str | Path, device: str | None = None):
    """Model + normalisation stats + config from a training run folder (best.pt, config.json, stats.json)."""
    import torch
    from model import build_from_config

    run = Path(run_dir)
    cfg = json.loads((run / "config.json").read_text())
    stats_path = run / "stats.json"
    if not stats_path.exists():   # runs made before stats were copied into the run folder
        stats_path = Path(cfg["prep_dir"]) / "stats.json"
    stats = json.loads(stats_path.read_text())
    if device is None:
        mps = getattr(torch.backends, "mps", None)
        device = "cuda" if torch.cuda.is_available() else ("mps" if mps is not None and mps.is_available() else "cpu")
    model = build_from_config(cfg)
    weights = run / "best.pt" if (run / "best.pt").exists() else run / "best_fp16.pt"   # fp16 copy for GitHub
    sd = torch.load(weights, map_location="cpu")
    model.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()})
    return model.to(device).eval(), stats, cfg, device


def predict(model, stats: dict, s1: np.ndarray, n: np.ndarray, device: str, lee: bool = False) -> np.ndarray:
    """int8 (12, H, W) class per pixel; -1 where that month has no Sentinel-1 data."""
    import torch
    mean = np.asarray(stats["mean_lee" if lee else "mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(stats["std_lee" if lee else "std"], dtype=np.float32)[:, None, None]
    out = np.full((s1.shape[0],) + s1.shape[2:], -1, dtype=np.int8)
    with torch.no_grad():
        for m in range(s1.shape[0]):
            if n[m] == 0:
                continue
            scene = lee_filter_db(s1[m]) if lee else s1[m]
            nod = scene == S1_NODATA
            x = (scene.astype(np.float32) / 100 - mean) / std
            x[nod] = 0
            p = model(torch.from_numpy(x)[None].to(device)).argmax(1)[0].cpu().numpy().astype(np.int8)
            p[nod.any(0)] = -1
            out[m] = p
    return out


def predict_month_mean(model, stats: dict, scenes: list, device: str, lee: bool = False,
                       batch: int = 8) -> np.ndarray:
    """int8 (12, H, W): per month, the class with the highest probability averaged over all of its scenes.
    Each scene is predicted on its own (as in training); a pixel is averaged only over scenes with data there."""
    import torch
    mean = np.asarray(stats["mean_lee" if lee else "mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(stats["std_lee" if lee else "std"], dtype=np.float32)[:, None, None]
    H, W = scenes[0].shape[-2:] if len(scenes[0]) else next(s for s in scenes if len(s)).shape[-2:]
    out = np.full((len(scenes), H, W), -1, dtype=np.int8)
    with torch.no_grad():
        for m, s1 in enumerate(scenes):
            if not len(s1):
                continue
            acc, have = None, np.zeros((H, W), bool)
            for i in range(0, len(s1), batch):
                xs, oks = [], []
                for sc in s1[i:i + batch]:
                    sc = lee_filter_db(sc) if lee else sc
                    nod = sc == S1_NODATA
                    x = (sc.astype(np.float32) / 100 - mean) / std
                    x[nod] = 0
                    xs.append(x); oks.append(~nod.any(0))
                p = model(torch.from_numpy(np.stack(xs)).to(device)).softmax(1).cpu().numpy()
                ok = np.stack(oks)
                part = (p * ok[:, None]).sum(0)
                acc = part if acc is None else acc + part
                have |= ok.any(0)
            out[m] = np.where(have, acc.argmax(0), -1).astype(np.int8)
    return out


TTA_FLIPS = ((), (-1,), (-2,), (-1, -2))     # identity, horizontal, vertical, both (each is its own inverse)

# What the prototype uses; chosen on the 130 test places (train/eval_change.py, train/eval_ensemble.py).
# Measured 2026-11-05 (eval_ensemble.py, 130 test places, month-mean maps):
#   E3 alone          mIoU 0.522, flicker 13.3%, false new built-up 0.39 ha/km2/yr (argmax rule)
#   ensemble of 4     mIoU 0.544, flicker 11.6%, 0.41 (argmax) -> 0.004 with the confident rule 0.5/0.2
#   TTA adds only +0.003 mIoU at 4x the time, so it is off.
# Rules for NEW built-up from probabilities: mean P(built) >= hi over the end months and < lo in EVERY start month.
RULES = {"confident": (0.5, 0.2), "moderate": (0.5, 0.35)}
EXTRA_MONTHS = 3     # same-season mode: the first 3 months are compared with the same 3 months one year later

# Round 2 (2026-11-06): 3-month smoothing halves flicker (11.6% -> 5.9%) at +0.005 mIoU (0.549); E6 (regularised)
# did not help and is left out; a second seed showed run-to-run noise of ~0.01 mIoU.
# Gallery (2026-11-07, 13 places, same season): "moderate" is the headline rule. It reports ~0 ha in the stable
# places (Heidelberg 0.0, Alps 0.1, Mekong 0.8) and tens of ha where both references see growth (Dar es Salaam 37,
# Phoenix 39, Lagos 15); "confident" missed obvious growth (Lagos 0.5 ha) and "argmax" reports change in stable
# places (Heidelberg 5 ha, Mekong 19 ha). The app shows argmax / confident as the upper / lower bound.
PROTOTYPE = {"runs": ["E3", "E1", "E5", "E2b"], "tta": False, "smooth3": True,
             "change_rule": "moderate", "hi": 0.5, "lo": 0.35, "change_mode": "same_season"}


def load_models(run_dirs) -> list:
    """[(model, stats, cfg), ...] and the device, for one or several run folders (ensemble)."""
    out, device = [], None
    for r in run_dirs:
        model, stats, cfg, device = load_model(r, device)
        out.append((model, stats, cfg))
    return out, device


def predict_month_probs(models: list, scenes: list, device: str, tta: bool = False, batch: int = 8,
                        smooth3: bool = False):
    """Month-mean class probabilities, averaged over all scenes of a month, over TTA flips and over models.
    smooth3: then average each month with the previous and next month (no wrap-around) before the argmax.

    Returns pred int8 (12, H, W) (argmax, -1 = no data) and p_built float16 (12, H, W)."""
    import torch
    shape = next((s.shape[-2:] for s in scenes if len(s)), (1, 1))
    H, W = shape
    pred = np.full((len(scenes), H, W), -1, np.int8)
    pbuilt = np.zeros((len(scenes), H, W), np.float16)
    probs = np.zeros((len(scenes), N_CLASSES, H, W), np.float32)
    haves = np.zeros((len(scenes), H, W), bool)
    flips = TTA_FLIPS if tta else TTA_FLIPS[:1]
    with torch.no_grad():
        for m, s1 in enumerate(scenes):
            if not len(s1):
                continue
            acc, cnt = np.zeros((N_CLASSES, H, W), np.float32), np.zeros((H, W), np.float32)
            for model, stats, cfg in models:
                lee = bool(cfg.get("lee", False))
                mean = np.asarray(stats["mean_lee" if lee else "mean"], np.float32)[:, None, None]
                std = np.asarray(stats["std_lee" if lee else "std"], np.float32)[:, None, None]
                for i in range(0, len(s1), batch):
                    xs, oks = [], []
                    for sc in s1[i:i + batch]:
                        nod = sc == S1_NODATA
                        sc = lee_filter_db(sc) if lee else sc
                        x = (sc.astype(np.float32) / 100 - mean) / std
                        x[nod] = 0
                        xs.append(x); oks.append(~nod.any(0))
                    x = torch.from_numpy(np.stack(xs)).to(device)
                    p = 0
                    for f in flips:
                        out = model(x.flip(f) if f else x)
                        p = p + (out.flip(f) if f else out).softmax(1)
                    p = (p / len(flips)).float().cpu().numpy()
                    ok = np.stack(oks)
                    acc += (p * ok[:, None]).sum(0)
                    cnt += ok.sum(0)
            haves[m] = cnt > 0
            probs[m] = acc / np.maximum(cnt, 1)
    if smooth3:
        probs = smooth_probs(probs, haves)
    for m in range(len(scenes)):
        pred[m] = np.where(haves[m], probs[m].argmax(0), -1)
        pbuilt[m] = probs[m, BUILT]
    return pred, pbuilt


def smooth_probs(prob: np.ndarray, have: np.ndarray) -> np.ndarray:
    """Mean of months m-1, m, m+1 that have data, per pixel (no wrap-around: the 12 months are consecutive).
    Halved month-to-month flicker on the test places (11.6% -> 5.9%, ensemble) at +0.005 mIoU."""
    out = np.zeros_like(prob)
    for m in range(prob.shape[0]):
        ks = [k for k in (m - 1, m, m + 1) if 0 <= k < prob.shape[0]]
        w = np.stack([have[k] for k in ks]).astype(np.float32)
        out[m] = sum(prob[k] * w[i][None] for i, k in enumerate(ks)) / np.maximum(w.sum(0), 1)[None]
    return out


def smooth(pred: np.ndarray) -> np.ndarray:
    from prepare_data import smooth_labels
    return smooth_labels(pred)


# ----------------------------------------------------------------------------- 5. compute

def class_percentages(pred: np.ndarray, months: list[str], names=CLASS_NAMES) -> pd.DataFrame:
    """% of valid pixels per class, one row per month; a month without data is all NaN."""
    k = len(names)
    rows = []
    for m in range(pred.shape[0]):
        v = pred[m][pred[m] >= 0]
        rows.append(np.bincount(v, minlength=k)[:k] / v.size * 100 if v.size else np.full(k, np.nan))
    return pd.DataFrame(rows, index=months, columns=names)


def month_to_month(pct: pd.DataFrame) -> pd.DataFrame:
    """Change vs the previous month, in percentage points (first month and gaps = NaN)."""
    return pct.diff()


def total_change(pct: pd.DataFrame) -> pd.Series:
    """Last month with data minus first month with data, in percentage points."""
    valid = pct.dropna(how="all")
    if len(valid) < 2:
        return pd.Series(np.nan, index=pct.columns)
    return (valid.iloc[-1] - valid.iloc[0]).rename(f"{valid.index[0]} → {valid.index[-1]}")


def year_majority(pred: np.ndarray) -> np.ndarray:
    """Most frequent class per pixel over the months (-1 if never observed)."""
    k = N_CLASSES
    counts = np.stack([(pred == c).sum(0) for c in range(k)])
    out = counts.argmax(0).astype(np.int8)
    out[counts.sum(0) == 0] = -1
    return out


# ----------------------------------------------------------------------------- 7. built-up change (headline)

def _periods(pred: np.ndarray, window: int = 3) -> tuple[list[int], list[int]]:
    have = [m for m in range(pred.shape[0]) if (pred[m] >= 0).any()]
    w = min(window, len(have) // 2)
    return (have[:w], have[-w:]) if w else ([], [])


def _drop_small(mask: np.ndarray, min_px: int) -> np.ndarray:
    """Remove connected patches (8-neighbour) smaller than min_px pixels: real construction comes in patches."""
    if min_px <= 1 or not mask.any():
        return mask
    from scipy import ndimage
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), bool))
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_px
    keep[0] = False
    return keep[lab]


def built_change(pred: np.ndarray, months: list[str], window: int = 3, min_px: int = 4,
                 pbuilt: np.ndarray | None = None, rule: str = "argmax", hi: float = 0.6, lo: float = 0.2,
                 first: list[int] | None = None, last: list[int] | None = None) -> dict | None:
    """Robust built-up change between the first and last `window` months with data.

    built before / after = built-up in the majority of that period's months
    new built-up         = built-up in the majority of the last months and in NONE of the first months
    lost built-up        = the reverse
    Requiring persistence ignores single-month misclassifications, which would otherwise read as construction;
    patches smaller than `min_px` pixels (default 4 = 400 m2) are dropped as noise.
    """
    if first is None or last is None:
        first, last = _periods(pred, window)
    else:   # given periods (same-season mode): keep the months that have data
        first = [m for m in first if (pred[m] >= 0).any()]
        last = [m for m in last if (pred[m] >= 0).any()]
        if len(first) < 2 or len(last) < 2:
            return None
    if not first:
        return None
    if rule in RULES:
        hi, lo = RULES[rule]
    need = len(first) // 2 + 1
    bf = np.stack([pred[m] == BUILT for m in first]).sum(0)
    bl = np.stack([pred[m] == BUILT for m in last]).sum(0)
    vf = np.stack([pred[m] >= 0 for m in first]).sum(0)
    vl = np.stack([pred[m] >= 0 for m in last]).sum(0)
    valid = (vf >= need) & (vl >= need)
    before, after = (bf >= need) & valid, (bl >= need) & valid
    if rule in RULES and pbuilt is not None:
        # new = mean P(built) >= hi over the end months and < lo in every start month (and the reverse for lost)
        pf = np.stack([np.where(pred[m] >= 0, pbuilt[m].astype(np.float32), np.nan) for m in first])
        pl = np.stack([np.where(pred[m] >= 0, pbuilt[m].astype(np.float32), np.nan) for m in last])
        import warnings
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)   # pixels without data in all start / end months
            gain = (np.nanmean(pl, 0) >= hi) & (np.nanmax(pf, 0) < lo) & valid
            loss = (np.nanmean(pf, 0) >= hi) & (np.nanmax(pl, 0) < lo) & valid
        gain, loss = _drop_small(gain, min_px), _drop_small(loss, min_px)
    else:
        gain = _drop_small(after & (bf == 0), min_px)
        loss = _drop_small(before & (bl == 0), min_px)
    n = max(int(valid.sum()), 1)
    return {"first": [months[m] for m in first], "last": [months[m] for m in last], "valid": valid,
            "before": before, "after": after, "gain": gain, "loss": loss,
            "before_pct": before.sum() / n * 100, "after_pct": after.sum() / n * 100,
            "change_pp": (after.sum() - before.sum()) / n * 100,
            "gain_ha": gain.sum() * PIXEL_HA, "loss_ha": loss.sum() * PIXEL_HA,
            "net_ha": (gain.sum() - loss.sum()) * PIXEL_HA, "area_ha": n * PIXEL_HA}


def period_change(pred: np.ndarray, months: list[str], window: int = 3) -> pd.DataFrame | None:
    """Robust total change for every class: majority map of the first vs the last `window` months with data."""
    first, last = _periods(pred, window)
    if not first:
        return None
    a, b = year_majority(pred[first]), year_majority(pred[last])
    ok = (a >= 0) & (b >= 0)
    pa = np.bincount(a[ok], minlength=N_CLASSES)[:N_CLASSES] / max(ok.sum(), 1) * 100
    pb = np.bincount(b[ok], minlength=N_CLASSES)[:N_CLASSES] / max(ok.sum(), 1) * 100
    return pd.DataFrame({f"start ({months[first[0]]}\u2026)": pa,
                         f"end (\u2026{months[last[-1]]})": pb, "change (pp)": pb - pa}, index=CLASS_NAMES)


def reference_change(name: str, a: np.ndarray, b: np.ndarray, valid: np.ndarray, ours: dict) -> dict:
    """Built-up change of a reference product on the same pixels, plus how much of our new built-up it confirms."""
    v = valid & ours["valid"]
    n = max(int(v.sum()), 1)
    gain_ref = b & ~a & v
    ours_gain = ours["gain"] & v
    confirmed = (ours_gain & b).sum() / max(ours_gain.sum(), 1) * 100 if ours_gain.any() else float("nan")
    return {"source": name, "built-up start (%)": (a & v).sum() / n * 100, "built-up end (%)": (b & v).sum() / n * 100,
            "change (pp)": ((b & v).sum() - (a & v).sum()) / n * 100, "new built-up (ha)": gain_ref.sum() * PIXEL_HA,
            "our new built-up that it confirms (%)": confirmed}


def built_change_table(ours: dict, refs: list[dict]) -> pd.DataFrame:
    rows = [{"source": f"our model ({ours['first'][0]}\u2026 vs \u2026{ours['last'][-1]})",
             "built-up start (%)": ours["before_pct"], "built-up end (%)": ours["after_pct"],
             "change (pp)": ours["change_pp"], "new built-up (ha)": ours["gain_ha"],
             "our new built-up that it confirms (%)": float("nan")}] + refs
    return pd.DataFrame(rows).set_index("source")


DEFAULT_FALSE_NEW_HA_KM2 = 0.5    # E1 on the 130 test tiles, month-mean maps (train/eval_change.py)


MIN_FLOOR_HA_KM2 = 0.02   # never claim to detect less than 2 ha per 100 km2 a year
CONFIDENT_FLOOR_HA_KM2 = 0.02   # measured 0.004-0.005 for the ensemble (eval_ensemble.py), floored at MIN


def noise_floor(run_dirs, rule: str = "argmax", tta: bool = False, hi: float = 0.5, lo: float = 0.2,
                smooth3: bool = False) -> float:
    """False new built-up the setting produces on unseen test places, ha per km2 per year.

    Read from train/runs/ensemble_eval.csv (eval_ensemble.py) for this model set / TTA / rule, else from the first
    run's change_eval.csv (argmax rule), else the measured E1 value."""
    run_dirs = [run_dirs] if isinstance(run_dirs, (str, Path)) else list(run_dirs)
    first = Path(run_dirs[0])
    if rule in RULES:
        hi, lo = RULES[rule]
    key = "argmax" if rule == "argmax" else f"confident {hi}/{lo}"
    if len(run_dirs) > 1:
        prefixes = (["ensemble + smooth3"] if smooth3 else []) + ["ensemble + TTA" if tta else "ensemble ("]
    else:
        prefixes = ([f"{first.name} + smooth3"] if smooth3 else []) + [f"{first.name} {'+ TTA' if tta else 'single'}"]
    for fname in ("ensemble_eval_round2.csv", "ensemble_eval.csv"):
        for pre in prefixes:
            try:
                df = pd.read_csv(first.parent / fname)
                v = df[df.variant.str.startswith(pre)]
                val = float(v.loc[v.change_rule == key, "false_new_ha_km2"].iloc[0])
                return max(val, MIN_FLOOR_HA_KM2)
            except Exception:  # noqa: BLE001 - file or variant missing: try the next one
                continue
    if rule != "argmax":
        return CONFIDENT_FLOOR_HA_KM2
    try:
        df = pd.read_csv(first / "change_eval.csv", index_col=0)
        return float(df.loc[df.index.str.startswith("B"), "false_new_ha_km2"].iloc[0])
    except Exception:  # noqa: BLE001 - file missing (older run): use the measured E1 value
        return DEFAULT_FALSE_NEW_HA_KM2


def change_verdict(bc: dict, floor_ha_km2: float) -> tuple[str, str, float]:
    """Compare new built-up with what the model produces by error on unseen places of the same size.

    Returns (verdict, explanation, expected false ha). >= 3x the error level = clear growth."""
    expected = floor_ha_km2 * bc["area_ha"] / 100
    r = bc["gain_ha"] / max(expected, 1e-9)
    if r >= 3:
        v = "Clear growth"
    elif r >= 1.5:
        v = "Some growth, partly within the error level"
    else:
        v = "No change detectable"
    why = (f"On 130 unseen test places the model reports about {floor_ha_km2:.2f} ha of false new built-up per km2 "
           f"per year, i.e. about {expected:.1f} ha for this area. Found here: {bc['gain_ha']:.1f} ha "
           f"({r:.1f}x the error level).")
    return v, why, expected


def split_result(months_all, n, info, s1_show, pred, pbuilt, n_main: int = 12) -> dict:
    """Split a 12 + EXTRA_MONTHS prediction into the 12 months shown and the extra same-season months."""
    out = {"months": list(months_all[:n_main]), "n": n[:n_main], "scenes": info[:n_main], "s1": s1_show[:n_main],
           "pred": pred[:n_main], "pred_raw": pred[:n_main], "pbuilt": pbuilt[:n_main], "extra": None}
    if len(months_all) > n_main:
        out["extra"] = {"months": list(months_all[n_main:]), "n": n[n_main:], "pred": pred[n_main:],
                        "pbuilt": pbuilt[n_main:], "scenes": info[n_main:]}
    return out


def change_inputs(res: dict, mode: str = "same_season"):
    """Arrays and periods for the built-up change of a result.

    same_season : the first 3 months vs the SAME 3 calendar months one year later (res["extra"], downloaded on
                  top of the 12 months), so wet / dry seasons cannot read as construction.
    within_year : the first 3 vs the last 3 of the 12 months (different seasons).
    Returns pred, pbuilt, months, first, last, mode actually used (falls back to within_year when the extra
    months are missing, e.g. too recent)."""
    pred, pb, months = res["pred"], res.get("pbuilt"), res["months"]
    ex = res.get("extra")
    if mode == "same_season" and ex is not None and int((np.asarray(ex["n"]) > 0).sum()) >= 2:
        k = len(ex["months"])
        p2 = np.concatenate([pred[:k], ex["pred"]])
        b2 = None if pb is None or ex.get("pbuilt") is None else np.concatenate([pb[:k], ex["pbuilt"]])
        return p2, b2, list(months[:k]) + list(ex["months"]), list(range(k)), list(range(k, 2 * k)), "same_season"
    return pred, pb, months, None, None, "within_year"


def change_rgb(bc: dict) -> np.ndarray:
    """Map: built-up that stayed = dark grey, new built-up = magenta, lost built-up = cyan, rest = light."""
    rgb = np.full(bc["valid"].shape + (3,), 238, dtype=np.uint8)
    rgb[~bc["valid"]] = 255
    rgb[bc["before"] & bc["after"]] = (90, 90, 90)
    rgb[bc["gain"]] = (214, 32, 181)
    rgb[bc["loss"]] = (0, 170, 200)
    return rgb


# ----------------------------------------------------------------------------- 6. independent check

def compare_with_reference(pred_year: np.ndarray, ref: np.ndarray) -> tuple[float, pd.DataFrame]:
    """Agreement with the reference map on the 6 classes: overall agreement and class % side by side."""
    ok = (pred_year >= 0) & (ref >= 0)
    agree = float((pred_year[ok] == ref[ok]).mean() * 100) if ok.any() else float("nan")
    k = N_CLASSES
    a = np.bincount(pred_year[ok], minlength=k)[:k] / max(ok.sum(), 1) * 100
    b = np.bincount(ref[ok], minlength=k)[:k] / max(ok.sum(), 1) * 100
    table = pd.DataFrame({"our model (%)": a, "ESRI 10 m (%)": b, "difference (pp)": a - b}, index=CLASS_NAMES)
    return agree, table


# ----------------------------------------------------------------------------- rendering

def colourise(lbl: np.ndarray, colours=CLASS_COLOURS) -> np.ndarray:
    rgb = np.full(lbl.shape + (3,), 235, dtype=np.uint8)   # light grey = no data
    for c, hexcol in enumerate(colours):
        rgb[lbl == c] = [int(hexcol[i:i + 2], 16) for i in (1, 3, 5)]
    return rgb


def grid_figure(pred: np.ndarray, months: list[str], n: np.ndarray):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 4, figsize=(13, 10.4))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(colourise(pred[i]), interpolation="nearest")
        ax.set_title(months[i] if n[i] else f"{months[i]} (no S1 data)", fontsize=10)
        ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CLASS_COLOURS]
    fig.legend(handles, CLASS_NAMES, loc="lower center", ncol=N_CLASSES, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def animation_gif(pred: np.ndarray, months: list[str], scale: int = 2) -> bytes:
    import io
    from PIL import Image, ImageDraw
    frames = []
    for i in range(pred.shape[0]):
        im = Image.fromarray(colourise(pred[i])).resize((pred.shape[2] * scale, pred.shape[1] * scale), Image.NEAREST)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 92, 22], fill=(255, 255, 255))
        d.text((6, 5), months[i], fill=(20, 20, 20))
        frames.append(im)
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], duration=700, loop=0)
    return buf.getvalue()


def run(run_dir: str, lat: float, lon: float, start_ym: str, size_px: int = 512,
        smooth_months: bool = False, ref_year: int | None = None) -> dict:
    """Whole pipeline without a UI (used by the app and for scripted case studies)."""
    model, stats, cfg, device = load_model(run_dir)
    g = area_grid(lat, lon, size_px)
    months = month_list(start_ym)
    scenes_all, n, scenes, s1 = fetch_s1_all(g, start_ym, terrain_flatten=stats.get("terrain_flattened", True))
    pred_raw = predict_month_mean(model, stats, scenes_all, device, cfg.get("lee", False))
    pred = smooth(pred_raw) if smooth_months else pred_raw
    pct = class_percentages(pred, months)
    out = {"grid": g, "months": months, "n": n, "s1": s1, "scenes": scenes,
           "built": built_change(pred, months), "pred_raw": pred_raw, "pred": pred, "pct": pct,
           "mom": month_to_month(pct), "total": total_change(pct), "cfg": cfg}
    if ref_year:
        ref = fetch_esri(g, ref_year)
        out["ref"] = ref
        out["agreement"], out["ref_table"] = compare_with_reference(year_majority(pred), ref)
    return out
