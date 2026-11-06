"""Download one GeoTIFF per tile: 12 monthly Sentinel-1 composites + two label maps per month.

Input  : tiles.csv from sample_tiles.py
Output : <out-dir>/<tile_id>.tif   (256 x 256 px, 10 m, local UTM zone, int16)
         <out-dir>/manifest.csv    (one row per tile: status, error)

Bands (73 per file), for each month MM = 01..12 of --year:
    mMM_VV     Sentinel-1 VV median, dB x 100          nodata -32768
    mMM_VH     Sentinel-1 VH median, dB x 100          nodata -32768
    mMM_VHVV   VH - VV (dB ratio) x 100                nodata -32768
    mMM_N      number of S1 scenes in the median       0 = no data that month
    mMM_DW     Dynamic World label (argmax of the month's mean probabilities)   nodata -1
    mMM_HYB    hybrid label: DW, with "built" (6) forced where buildings cover   nodata -1
               >= --built-frac of the 3x3 neighbourhood (Open Buildings v3)
plus one static band:
    BFRAC      building roof fraction per 10 m pixel x 100 (0-100), Open Buildings v3

Only the S1 bands are model input. DW, HYB and BFRAC are labels / analysis layers.

Dynamic World class ids:
    0 water, 1 trees, 2 grass, 3 flooded_vegetation, 4 crops,
    5 shrub_and_scrub, 6 built, 7 bare, 8 snow_and_ice

Usage
    python3 export_tiles.py --project scm-ksns --tiles tiles_v0/tiles.csv --out-dir tiles_v0/raw --limit 3
    python3 export_tiles.py --project scm-ksns --tiles tiles_v0/tiles.csv --out-dir tiles_v0/raw
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import time
from pathlib import Path

import pandas as pd
import requests

TILE_PX = 256
PIXEL_M = 10
HALF_M = TILE_PX * PIXEL_M / 2

DW_BANDS = ["water", "trees", "grass", "flooded_vegetation", "crops",
            "shrub_and_scrub", "built", "bare", "snow_and_ice"]
DW_BUILT = 6

S1_NODATA = -32768
LABEL_NODATA = -1


def month_band_names(month: int) -> list[str]:
    p = f"m{month:02d}_"
    return [p + b for b in ("VV", "VH", "VHVV", "N", "DW", "HYB")]


def all_band_names() -> list[str]:
    names = []
    for m in range(1, 13):
        names += month_band_names(m)
    return names + ["BFRAC"]


def tile_grid(x_utm: float, y_utm: float) -> list[float]:
    """crs_transform for a 256 x 256 grid of 10 m pixels centred on (x, y), snapped to whole metres x10."""
    x0 = round((x_utm - HALF_M) / PIXEL_M) * PIXEL_M
    y_top = round((y_utm + HALF_M) / PIXEL_M) * PIXEL_M
    return [PIXEL_M, 0, x0, 0, -PIXEL_M, y_top]


# ----------------------------------------------------------------------------- Earth Engine part

def utm_coords(tiles: pd.DataFrame) -> list[tuple[float, float]]:
    import ee
    feats = [ee.Feature(ee.Geometry.Point([r.lon, r.lat]).transform(f"EPSG:{r.utm_epsg}", 0.01))
             for r in tiles.itertuples()]
    out = ee.FeatureCollection(feats).getInfo()["features"]
    return [tuple(f["geometry"]["coordinates"]) for f in out]


def build_image(region, year: int, built_frac: float, min_conf: float):
    """73-band int16 image for one tile (see module docstring)."""
    import ee

    from s1_composite import s1_collection, s1_month

    s1_all = s1_collection(region)
    dw_all = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(region).select(DW_BANDS)

    # Building roof fraction at 10 m: paint footprints at 2 m, average to 10 m.
    buildings = (ee.FeatureCollection("GOOGLE/Research/open-buildings/v3/polygons")
                 .filterBounds(region)
                 .filter(ee.Filter.gte("confidence", min_conf)))
    proj = region.projection()
    frac = (ee.Image(0).byte().paint(buildings, 1)
            .reproject(crs=proj, scale=2)
            .reduceResolution(ee.Reducer.mean(), maxPixels=64)
            .reproject(crs=proj, scale=PIXEL_M))
    built_nb = frac.reduceNeighborhood(ee.Reducer.mean(), ee.Kernel.square(1)).gte(built_frac)

    empty_lbl = ee.Image.constant(0).rename("DW").updateMask(0)

    bands = []
    for m in range(1, 13):
        start = ee.Date.fromYMD(year, m, 1)
        end = start.advance(1, "month")

        s1m = s1_month(s1_all, start, end)          # VV, VH, VHVV (dB x 100) + N, shared with the prototype

        dw = dw_all.filterDate(start, end)
        lbl = ee.Image(ee.Algorithms.If(
            dw.size().gt(0),
            dw.mean().toArray().arrayArgmax().arrayGet(0).rename("DW"),
            empty_lbl))
        hyb = lbl.where(built_nb, DW_BUILT).updateMask(lbl.mask()).rename("HYB")

        month = (s1m
                 .addBands(lbl.unmask(LABEL_NODATA))
                 .addBands(hyb.unmask(LABEL_NODATA))
                 .toInt16()
                 .rename(month_band_names(m)))
        bands.append(month)

    stack = ee.Image.cat(bands + [frac.multiply(100).round().toInt16().rename("BFRAC")])
    return stack


def download_tile(row, xy, args) -> tuple[str, str]:
    import ee

    out = Path(args.out_dir) / f"{row.tile_id}.tif"
    if out.exists() and out.stat().st_size > 0 and not args.overwrite:
        return row.tile_id, "skipped (exists)"

    crs = f"EPSG:{row.utm_epsg}"
    transform = tile_grid(*xy)
    x0, y_top = transform[2], transform[5]
    region = ee.Geometry.Rectangle(
        [x0, y_top - TILE_PX * PIXEL_M, x0 + TILE_PX * PIXEL_M, y_top], ee.Projection(crs), False)

    last_err = ""
    for attempt in range(1, args.retries + 1):
        try:
            img = build_image(region, args.year, args.built_frac, args.min_confidence)
            url = img.getDownloadURL({
                "crs": crs, "crs_transform": transform,
                "dimensions": f"{TILE_PX}x{TILE_PX}", "format": "GEO_TIFF",
            })
            r = requests.get(url, timeout=600)
            r.raise_for_status()
            tmp = out.with_suffix(".part")
            tmp.write_bytes(r.content)
            tmp.rename(out)
            return row.tile_id, "ok"
        except Exception as e:  # noqa: BLE001 - report any failure in the manifest
            last_err = str(e).splitlines()[0][:300]
            time.sleep(5 * attempt)
    return row.tile_id, f"failed: {last_err}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True)
    ap.add_argument("--tiles", required=True, help="tiles.csv from sample_tiles.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--built-frac", type=float, default=0.15,
                    help="roof share in the 3x3 neighbourhood above which the hybrid label becomes built")
    ap.add_argument("--min-confidence", type=float, default=0.75, help="Open Buildings confidence cut-off")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="only the first N tiles (for a test run)")
    ap.add_argument("--split", default="", help="only tiles of this split (train/val/test)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    import ee
    ee.Initialize(project=args.project)

    tiles = pd.read_csv(args.tiles)
    if args.split:
        tiles = tiles[tiles["split"] == args.split]
    if args.limit:
        tiles = tiles.head(args.limit)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print(f"Converting {len(tiles)} tile centres to UTM ...")
    xys = utm_coords(tiles)

    print(f"Downloading {len(tiles)} tiles with {args.workers} workers (year {args.year}) ...")
    results = []
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(download_tile, r, xy, args) for r, xy in zip(tiles.itertuples(), xys)]
        for i, f in enumerate(cf.as_completed(futs), 1):
            tid, status = f.result()
            results.append({"tile_id": tid, "status": status})
            if i % 10 == 0 or i == len(futs) or not status.startswith(("ok", "skipped")):
                print(f"  [{i}/{len(futs)}] {tid}: {status}   ({time.time() - t0:.0f} s)")

    man = pd.DataFrame(results).merge(tiles, on="tile_id")
    man.to_csv(Path(args.out_dir) / "manifest.csv", index=False)
    ok = man["status"].str.startswith(("ok", "skipped")).sum()
    print(f"\nDone: {ok} of {len(man)} tiles on disk. Failures are listed in manifest.csv; "
          "rerun the same command to retry only those.")


if __name__ == "__main__":
    main()
