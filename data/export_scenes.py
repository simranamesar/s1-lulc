"""Download every single Sentinel-1 acquisition (raw scene, no averaging) for each tile in one year.

Input  : tiles.csv from sample_tiles.py
Output : <out-dir>/<tile_id>.npz   with
             s1        int16 (S, 3, 256, 256)  VV, VH, VH-VV in dB x 100, -32768 = no data
             date      str   (S,)              acquisition date YYYY-MM-DD
             orbit     str   (S,)              ASCENDING / DESCENDING
             rel_orbit int16 (S,)              relative orbit number
             valid     float (S,)              share of the tile covered by the scene
             flattened bool                    True = terrain-flattened gamma0 (default), False = sigma0
         <out-dir>/manifest.csv
Scenes covering less than --min-valid of the tile are dropped (swath edges; layover / shadow masks count as gaps).
Terrain flattening (gamma0) is on by default; --no-flatten keeps sigma0 for a comparison run.
Same 256 x 256, 10 m, local-UTM grid as export_tiles.py, so scenes line up pixel by pixel with the monthly labels.

Usage
    python3 export_scenes.py --project scm-ksns --tiles tiles_v0/tiles.csv --out-dir tiles_v0/scenes --limit 3
    python3 export_scenes.py --project scm-ksns --tiles tiles_v0/tiles.csv --out-dir tiles_v0/scenes
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import time
from pathlib import Path

import numpy as np
import pandas as pd

from export_tiles import PIXEL_M, TILE_PX, tile_grid, utm_coords
from s1_composite import S1_BANDS, dem_slope_aspect, s1_scene_image, s1_scene_table, valid_fraction


def fetch_tile(row, xy, args) -> tuple[str, str, int]:
    import ee

    out = Path(args.out_dir) / f"{row.tile_id}.npz"
    if out.exists() and not args.overwrite:
        return row.tile_id, "skipped (exists)", -1

    crs = f"EPSG:{row.utm_epsg}"
    tf = tile_grid(*xy)
    x0, y_top = tf[2], tf[5]
    span = TILE_PX * PIXEL_M
    region = ee.Geometry.Rectangle([x0, y_top - span, x0 + span, y_top], ee.Projection(crs), False)
    grid = {"dimensions": {"width": TILE_PX, "height": TILE_PX},
            "affineTransform": {"scaleX": PIXEL_M, "shearX": 0, "translateX": x0,
                                "shearY": 0, "scaleY": -PIXEL_M, "translateY": y_top},
            "crsCode": crs}

    last = ""
    for attempt in range(1, args.retries + 1):
        try:
            start, end = f"{args.year}-01-01", f"{args.year + 1}-01-01"
            scenes = s1_scene_table(region, start, end)
            if not scenes:
                return row.tile_id, "no scenes", 0
            dem = None if args.no_flatten else dem_slope_aspect(region)
            chunks = []
            for i in range(0, len(scenes), args.chunk):
                part = scenes[i:i + args.chunk]
                img = ee.Image.cat([
                    s1_scene_image(region, start, end, sc["ids"], not args.no_flatten, dem)
                    .rename([f"s{j}_{b}" for b in S1_BANDS])
                    for j, sc in enumerate(part)])
                arr = ee.data.computePixels({"expression": img, "fileFormat": "NUMPY_NDARRAY", "grid": grid})
                chunks.append(np.stack([np.stack([arr[f"s{j}_{b}"] for b in S1_BANDS]) for j in range(len(part))]))
            s1 = np.concatenate(chunks).astype(np.int16)
            valid = valid_fraction(s1)
            keep = valid >= args.min_valid
            np.savez_compressed(out, s1=s1[keep], date=np.array([sc["date"] for sc in scenes])[keep],
                                orbit=np.array([sc["pass"] for sc in scenes])[keep],
                                rel_orbit=np.array([sc["rel_orbit"] for sc in scenes], dtype=np.int16)[keep],
                                valid=valid[keep],
                                flattened=np.array(not args.no_flatten))
            return row.tile_id, "ok", int(keep.sum())
        except Exception as e:  # noqa: BLE001 - reported in the manifest
            last = " ".join(str(e).split())[:400]   # whole message on one line (the cause is often at the end)
            time.sleep(5 * attempt)
    return row.tile_id, f"failed: {last}", -1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True)
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--min-valid", type=float, default=0.5, help="drop scenes covering less of the tile than this")
    ap.add_argument("--chunk", type=int, default=24, help="scenes per Earth Engine request (request size limit)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated tile ids, e.g. t00059,t00056 (for debugging)")
    ap.add_argument("--no-flatten", action="store_true", help="keep sigma0 (no terrain flattening), for comparison")
    args = ap.parse_args()

    import ee
    ee.Initialize(project=args.project)
    tiles = pd.read_csv(args.tiles)
    if args.only:
        tiles = tiles[tiles.tile_id.isin(args.only.split(","))]
    if args.limit:
        tiles = tiles.head(args.limit)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print(f"Converting {len(tiles)} tile centres to UTM ...")
    xys = utm_coords(tiles)
    print(f"Downloading single scenes for {len(tiles)} tiles, {args.year}, {args.workers} workers ...")
    res, t0 = [], time.time()
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(fetch_tile, r, xy, args) for r, xy in zip(tiles.itertuples(), xys)]
        for i, f in enumerate(cf.as_completed(futs), 1):
            tid, status, n = f.result()
            res.append({"tile_id": tid, "status": status, "scenes": n})
            if i % 10 == 0 or i == len(futs) or not status.startswith(("ok", "skipped")):
                print(f"  [{i}/{len(futs)}] {tid}: {status} {'' if n < 0 else f'({n} scenes)'}  ({time.time() - t0:.0f} s)")
    man = pd.DataFrame(res)
    man.to_csv(Path(args.out_dir) / ("manifest_only.csv" if args.only else "manifest.csv"), index=False)
    ok = man[man.status == "ok"]
    print(f"\nDone: {man.status.str.startswith(('ok', 'skipped')).sum()} of {len(man)} tiles on disk.")
    if len(ok):
        print(f"Scenes per tile: median {ok.scenes.median():.0f}, min {ok.scenes.min()}, max {ok.scenes.max()}; "
              f"total {ok.scenes.sum()}")


if __name__ == "__main__":
    main()