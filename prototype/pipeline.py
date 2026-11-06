"""Prototype processing: location + start month -> 12 monthly LULC maps, class %, changes, independent check.

Steps (the report's "Prototype" section follows this order):
    1. grid      a square of --size px x 10 m centred on the point, in the local UTM zone
    2. fetch     for each of the 12 months ONE raw Sentinel-1 scene (no averaging), prepared with the SAME recipe
                 as the training data (data/s1_composite.py). Chosen scene: covers the whole area, closest to
                 mid-month (same rule as the training evaluation)
    3. predict   the trained U-Net (train/model.py) on each month's scene, normalised with the training statistics
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

from s1_composite import S1_BANDS, S1_NODATA, lee_filter_db, s1_scene_images  # noqa: E402

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


def month_list(start_ym: str) -> list[str]:
    """'2024-03' -> ['2024-03', ..., '2025-02']."""
    y, m = map(int, start_ym.split("-"))
    out = []
    for i in range(12):
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
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_from_config(cfg)
    model.load_state_dict(torch.load(run / "best.pt", map_location=device))
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


def built_change(pred: np.ndarray, months: list[str], window: int = 3, min_px: int = 4) -> dict | None:
    """Robust built-up change between the first and last `window` months with data.

    built before / after = built-up in the majority of that period's months
    new built-up         = built-up in the majority of the last months and in NONE of the first months
    lost built-up        = the reverse
    Requiring persistence ignores single-month misclassifications, which would otherwise read as construction;
    patches smaller than `min_px` pixels (default 4 = 400 m2) are dropped as noise.
    """
    first, last = _periods(pred, window)
    if not first:
        return None
    need = len(first) // 2 + 1
    bf = np.stack([pred[m] == BUILT for m in first]).sum(0)
    bl = np.stack([pred[m] == BUILT for m in last]).sum(0)
    vf = np.stack([pred[m] >= 0 for m in first]).sum(0)
    vl = np.stack([pred[m] >= 0 for m in last]).sum(0)
    valid = (vf >= need) & (vl >= need)
    before, after = (bf >= need) & valid, (bl >= need) & valid
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
    s1, n, scenes = fetch_s1(g, start_ym, terrain_flatten=stats.get("terrain_flattened", True))
    pred_raw = predict(model, stats, s1, n, device, cfg.get("lee", False))
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
