"""Sample a globally stratified set of training tiles for Sentinel-1-only LULC.

Each tile is a 2.56 km x 2.56 km square (256 x 256 px at 10 m) around a sampled centre point.

Strata
    region      7 broad geographic boxes (spreads tiles over all continents)
    biome       RESOLVE Ecoregions 2017, 14 biomes
    settlement  city / town / village / none, from GHSL Degree of Urbanisation (SMOD, 1 km)
                plus GHSL built-up surface, so scattered rural villages are not counted as "none"
    landcover   dominant ESA WorldCover 2021 class in the 1 km cell (so water, wetland, bare,
                grass and shrub landscapes get their turn, not only forest and cropland)

Steps
    1. Draw random points in each region box and read biome + settlement at each point (Earth Engine).
    2. Pick a balanced candidate set: fixed share per settlement type; inside it, regions take equal
       turns, and inside a region the turns rotate over land cover x biome.
    3. Drop candidates without enough Sentinel-1 IW VV+VH coverage in the chosen year.
    4. Assign train / val / test by 5-degree blocks, so test regions are never mixed with training.
    5. Write tiles.csv, tiles.geojson and tiles_map.html (open in a browser to inspect).

Usage
    earthengine authenticate          # once
    python sample_tiles.py --project scm-ksns --target 300 --out-dir tiles_v0
    # grow an existing set to 600 tiles, keeping its 300 tiles and filling the thinnest strata first:
    python sample_tiles.py --project scm-ksns --target 600 --extend tiles_v0/tiles.csv --out-dir tiles_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

SETTLEMENT_NAMES = {0: "none", 1: "village", 2: "town", 3: "city"}

# Share of the final tile set per settlement type. Villages are oversampled on purpose.
DEFAULT_SHARES = {"village": 0.35, "none": 0.25, "town": 0.20, "city": 0.20}

BIOME_NAMES = {
    1: "Tropical moist broadleaf forests",
    2: "Tropical dry broadleaf forests",
    3: "Tropical coniferous forests",
    4: "Temperate broadleaf & mixed forests",
    5: "Temperate conifer forests",
    6: "Boreal forests / taiga",
    7: "Tropical grasslands, savannas & shrublands",
    8: "Temperate grasslands, savannas & shrublands",
    9: "Flooded grasslands & savannas",
    10: "Montane grasslands & shrublands",
    11: "Tundra",
    12: "Mediterranean forests, woodlands & scrub",
    13: "Deserts & xeric shrublands",
    14: "Mangroves",
}

LANDCOVER_NAMES = {10: "trees", 20: "shrub", 30: "grass", 40: "crops", 50: "built", 60: "bare",
                   70: "snow", 80: "water", 90: "wetland", 95: "mangrove", 100: "moss"}

# [west, south, east, north] in degrees. Boxes only spread the sampling; they are not exact continents.
REGIONS = {
    "europe": [-25, 34, 45, 72],
    "africa_mideast": [-20, -36, 52, 34],
    "asia_north": [45, 34, 180, 75],
    "asia_south": [52, -11, 155, 34],
    "oceania": [110, -48, 180, -11],
    "north_america": [-170, 13, -50, 72],
    "south_america": [-93, -56, -33, 13],
}

TILE_SIZE_M = 2560


# ----------------------------------------------------------------------------- Earth Engine part

def stratum_image(built_threshold: float):
    """Image with bands 'biome' (1-14) and 'settle' (0-3) at 1 km, masked over sea and missing data."""
    import ee

    smod = ee.Image("JRC/GHSL/P2023A/GHS_SMOD_V2-0/2020").select("smod_code")
    # Mean built-up surface (m2 per 100 m cell) inside each 1 km SMOD cell.
    built = (
        ee.Image("JRC/GHSL/P2023A/GHS_BUILT_S/2020")
        .select("built_surface")
        .reduceResolution(ee.Reducer.mean(), maxPixels=1024)
        .reproject(smod.projection())
    )
    rural = smod.gte(11).And(smod.lte(12))
    settle = (
        ee.Image(0)
        .where(rural.And(built.gt(built_threshold)), 1)  # scattered village inside "rural" SMOD
        .where(smod.eq(13), 1)                           # SMOD rural cluster = village
        .where(smod.gte(21).And(smod.lte(23)), 2)        # suburban / semi-dense / dense cluster = town
        .where(smod.eq(30), 3)                           # urban centre = city
        .updateMask(smod.gte(10))                        # 10 = water (kept: lakes, coasts), negative = no data
        .rename("settle")
        .toInt()
    )
    eco = ee.FeatureCollection("RESOLVE/ECOREGIONS/2017").filter(
        ee.Filter.rangeContains("BIOME_NUM", 1, 14)
    )
    biome = eco.reduceToImage(["BIOME_NUM"], ee.Reducer.first()).toInt().rename("biome")
    # Land-cover stratum: WorldCover read at the 1 km sampling scale through Earth Engine's own pyramid
    # (mode of the 10 m classes). An explicit reduceResolution + reproject of 10 m data to 1 km failed with
    # "Reprojection output too large" over continent-sized regions.
    lc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").rename("lc").toInt()
    return settle.addBands(biome).addBands(lc).updateMask(biome.mask())


def sample_candidates(points_per_region: int, seed: int, built_threshold: float,
                      urban_points: int = 0) -> pd.DataFrame:
    import ee

    img = stratum_image(built_threshold)
    rows = []
    for i, (name, box) in enumerate(REGIONS.items()):
        geom = ee.Geometry.BBox(*box)
        pts = ee.FeatureCollection.randomPoints(geom, points_per_region, seed + i)
        fc = img.sampleRegions(collection=pts, scale=1000, geometries=True, tileScale=4)
        feats = fc.getInfo()["features"]
        for f in feats:
            lon, lat = f["geometry"]["coordinates"]
            p = f["properties"]
            if "lc" not in p:
                continue
            rows.append({"region": name, "lon": lon, "lat": lat, "biome": int(p["biome"]),
                         "settle": int(p["settle"]), "landcover": LANDCOVER_NAMES.get(int(p["lc"]), "other")})
        print(f"  {name:15s} {len(feats):5d} land points")
        if urban_points:
            # Random land points almost never fall in cities (1 of ~12,000 in the 2026-11-05 run), so towns
            # and cities are also drawn directly from the GHSL urban classes (stratified sample, 1 km cells).
            smod = ee.Image("JRC/GHSL/P2023A/GHS_SMOD_V2-0/2020").select("smod_code")
            urb = (ee.Image(0).where(smod.gte(21).And(smod.lte(23)), 2).where(smod.eq(30), 3)
                   .rename("settle").toInt().selfMask())
            urb_img = urb.addBands(img.select(["biome", "lc"])).updateMask(img.select("biome").mask())
            fc = urb_img.stratifiedSample(numPoints=urban_points, classBand="settle", region=geom, scale=1000,
                                          seed=seed + 100 + i, classValues=[2, 3],
                                          classPoints=[urban_points // 2, urban_points],
                                          geometries=True, tileScale=8)
            uf = fc.getInfo()["features"]
            for f in uf:
                lon, lat = f["geometry"]["coordinates"]
                q = f["properties"]
                if "lc" not in q or "biome" not in q:
                    continue
                rows.append({"region": name, "lon": lon, "lat": lat, "biome": int(q["biome"]),
                             "settle": int(q["settle"]),
                             "landcover": LANDCOVER_NAMES.get(int(q["lc"]), "other")})
            print(f"  {name:15s} {len(uf):5d} extra town / city points")
    df = pd.DataFrame(rows)
    df["settlement"] = df["settle"].map(SETTLEMENT_NAMES)
    return df


def add_s1_coverage(df: pd.DataFrame, year: int, chunk: int = 200) -> pd.DataFrame:
    """Count months (and scenes) in `year` with Sentinel-1 IW dual-pol VV+VH data over each point."""
    import ee

    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
    )

    def count(feat):
        col = s1.filterBounds(feat.geometry())
        months = (
            col.aggregate_array("system:time_start")
            .map(lambda t: ee.Date(t).get("month"))
            .distinct()
            .size()
        )
        return feat.set({"s1_months": months, "s1_scenes": col.size()})

    months, scenes = [], []
    for start in range(0, len(df), chunk):
        part = df.iloc[start:start + chunk]
        fc = ee.FeatureCollection(
            [ee.Feature(ee.Geometry.Point([r.lon, r.lat]), {"k": int(k)})
             for k, r in zip(part.index, part.itertuples())]
        ).map(count)
        props = [f["properties"] for f in fc.getInfo()["features"]]
        props.sort(key=lambda p: p["k"])
        months += [p["s1_months"] for p in props]
        scenes += [p["s1_scenes"] for p in props]
    out = df.copy()
    out["s1_months"] = months
    out["s1_scenes"] = scenes
    return out


# ----------------------------------------------------------------------------- offline part

def cell_key(df: pd.DataFrame, cell_deg: float = 0.05) -> pd.Series:
    return (np.floor(df["lon"] / cell_deg)).astype(int).astype(str) + "_" + (
        np.floor(df["lat"] / cell_deg)).astype(int).astype(str)


def dedupe(df: pd.DataFrame, existing: pd.DataFrame | None = None) -> pd.DataFrame:
    """Keep one point per ~5 km grid cell so tiles never overlap (also against `existing` tiles)."""
    key = cell_key(df)
    keep = ~key.duplicated()
    if existing is not None and len(existing):
        keep &= ~key.isin(set(cell_key(existing)))
    return df.loc[keep].reset_index(drop=True)


def _existing_count(df: pd.DataFrame, existing: pd.DataFrame | None, cols: list[str]) -> np.ndarray:
    if existing is None or not len(existing):
        return np.zeros(len(df), dtype=int)
    c = existing.groupby(cols).size().rename("_n").reset_index()
    return df[cols].merge(c, on=cols, how="left")["_n"].fillna(0).astype(int).to_numpy()


def select_balanced(df: pd.DataFrame, n_new: int, shares: dict[str, float], seed: int,
                    existing: pd.DataFrame | None = None) -> pd.DataFrame:
    """Pick `n_new` points so that existing + new tiles are as balanced as possible.

    Settlement types get quotas (share x total). Inside a settlement type, regions take equal turns,
    and inside a region the turns rotate over land cover x biome. Strata that already hold tiles
    (from `existing`) wait their turn, so an extension fills the thinnest strata first.
    If a settlement type runs out of points, the shortfall is filled from the others.
    """
    df = df.copy()
    if "landcover" not in df:
        df["landcover"] = "unknown"
    ex = None
    if existing is not None and len(existing):
        ex = existing.copy()
        if "landcover" not in ex:
            ex["landcover"] = "unknown"
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    inner = ["settlement", "region", "landcover", "biome"]
    df["r_in"] = df.groupby(inner).cumcount().to_numpy() + _existing_count(df, ex, inner)
    df = df.sort_values("r_in", kind="stable").reset_index(drop=True)
    outer = ["settlement", "region"]
    df["r_reg"] = df.groupby(outer).cumcount().to_numpy() + _existing_count(df, ex, outer)
    df = df.sort_values(["r_reg", "r_in"], kind="stable")

    total = n_new + (0 if ex is None else len(ex))
    picked = []
    for name, share in shares.items():
        have = 0 if ex is None else int((ex["settlement"] == name).sum())
        need = max(round(total * share) - have, 0)
        picked.append(df[df["settlement"] == name].head(need))
    out = pd.concat(picked)
    if len(out) > n_new:
        out = out.sort_values(["r_reg", "r_in"], kind="stable").head(n_new)
    short = n_new - len(out)
    if short > 0:
        out = pd.concat([out, df.drop(out.index).head(short)])
    return out.drop(columns=["r_in", "r_reg"]).reset_index(drop=True)


def assign_splits(df: pd.DataFrame, seed: int, block_deg: float = 5.0,
                  val: float = 0.10, test: float = 0.20) -> pd.Series:
    """Split by geographic block: every tile in a block gets the same split."""
    def one(lon, lat):
        key = f"{seed}:{math.floor(lon / block_deg)}:{math.floor(lat / block_deg)}"
        u = int(hashlib.md5(key.encode()).hexdigest(), 16) % 10_000 / 10_000
        return "test" if u < test else ("val" if u < test + val else "train")
    return pd.Series([one(lo, la) for lo, la in zip(df["lon"], df["lat"])], index=df.index)


def utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def tile_square(lon: float, lat: float, size_m: float = TILE_SIZE_M) -> list[list[float]]:
    """Approximate lon/lat square for display; exports use the exact UTM square."""
    half_lat = size_m / 2 / 111_320
    half_lon = half_lat / max(math.cos(math.radians(lat)), 1e-6)
    return [[lon - half_lon, lat - half_lat], [lon + half_lon, lat - half_lat],
            [lon + half_lon, lat + half_lat], [lon - half_lon, lat + half_lat],
            [lon - half_lon, lat - half_lat]]


def write_outputs(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "tiles.csv", index=False)

    feats = [{
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [tile_square(r.lon, r.lat)]},
        "properties": {k: (v.item() if hasattr(v, "item") else v) for k, v in r._asdict().items()
                       if k != "Index"},
    } for r in df.itertuples()]
    (out_dir / "tiles.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))

    try:
        import folium
    except ImportError:
        print("folium not installed: skipped tiles_map.html")
        return
    colours = {"city": "#d62728", "town": "#ff7f0e", "village": "#9467bd", "none": "#2ca02c"}
    m = folium.Map(location=[20, 10], zoom_start=2, tiles="OpenStreetMap")
    for r in df.itertuples():
        folium.CircleMarker(
            [r.lat, r.lon], radius=5 if r.split == "train" else 7,
            color="#000000" if r.split == "test" else colours[r.settlement],
            weight=2 if r.split == "test" else 1,
            fill=True, fill_color=colours[r.settlement], fill_opacity=0.8,
            popup=(f"{r.tile_id}<br>{r.settlement} · {r.split}<br>{r.biome_name}"
                   f"<br>land cover: {getattr(r, 'landcover', 'n/a')}"
                   f"<br>S1 months: {getattr(r, 's1_months', 'n/a')}"),
        ).add_to(m)
    legend = ("<div style='position:fixed;bottom:20px;left:20px;z-index:999;background:white;"
              "padding:8px;font:12px sans-serif;border:1px solid #999'>"
              + "".join(f"<span style='color:{c}'>&#9679;</span> {n}<br>" for n, c in colours.items())
              + "black ring = test, larger = val/test</div>")
    m.get_root().html.add_child(folium.Element(legend))
    m.save(str(out_dir / "tiles_map.html"))


def summarise(df: pd.DataFrame) -> None:
    print("\nTiles per settlement type and split:")
    print(pd.crosstab(df["settlement"], df["split"], margins=True))
    print("\nTiles per region:")
    print(df["region"].value_counts().to_string())
    print(f"\nBiomes covered: {df['biome'].nunique()} of 14")
    if "landcover" in df:
        print("\nTiles per dominant land cover (1 km, ESA WorldCover):")
        print(df["landcover"].value_counts().to_string())


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True, help="Google Cloud project registered for Earth Engine")
    ap.add_argument("--target", type=int, default=300, help="number of tiles to keep")
    ap.add_argument("--year", type=int, default=2023, help="year whose Sentinel-1 coverage is checked")
    ap.add_argument("--points-per-region", type=int, default=4000)
    ap.add_argument("--built-threshold", type=float, default=20.0,
                    help="mean built-up m2 per 100 m cell above which a rural 1 km cell counts as village")
    ap.add_argument("--min-s1-months", type=int, default=10)
    ap.add_argument("--oversample", type=float, default=2.0,
                    help="candidates checked for S1 coverage, as a multiple of --target")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="tiles_v0")
    ap.add_argument("--extend", default="",
                    help="existing tiles.csv to keep; --target is then the total after adding new tiles")
    ap.add_argument("--shares", default="",
                    help="settlement shares of the NEW tiles, e.g. city=0.7,town=0.3 (default: the standard mix)")
    ap.add_argument("--urban-points", type=int, default=300,
                    help="extra candidate points per region drawn inside GHSL towns / cities (0 = off)")
    args = ap.parse_args()

    import ee
    ee.Initialize(project=args.project)

    existing = pd.read_csv(args.extend) if args.extend else None
    n_have = 0 if existing is None else len(existing)
    n_new = args.target - n_have
    if n_new <= 0:
        raise SystemExit(f"--target {args.target} must be larger than the {n_have} existing tiles")
    if existing is not None:
        print(f"Extending {args.extend}: keeping {n_have} tiles, adding {n_new}")

    print("1/4 Sampling random land points and reading strata ...")
    shares = DEFAULT_SHARES
    if args.shares:
        shares = {k: float(v) for k, v in (kv.split("=") for kv in args.shares.split(","))}
        tot = sum(shares.values())
        shares = {k: shares.get(k, 0.0) / tot for k in DEFAULT_SHARES}
        print(f"Shares of the new tiles: {shares}")
    cand = dedupe(sample_candidates(args.points_per_region, args.seed, args.built_threshold,
                                    args.urban_points), existing)
    print(f"    {len(cand)} candidates; per settlement: {cand['settlement'].value_counts().to_dict()}")

    print(f"2/4 Picking a balanced candidate set ({args.oversample}x the new tiles) ...")
    pre = select_balanced(cand, int(n_new * args.oversample), shares, args.seed, existing)

    print(f"3/4 Checking Sentinel-1 coverage in {args.year} ...")
    pre = add_s1_coverage(pre, args.year)
    ok = pre[pre["s1_months"] >= args.min_s1_months]
    print(f"    {len(ok)} of {len(pre)} have >= {args.min_s1_months} months of VV+VH data")

    print("4/4 Final selection and splits ...")
    new = select_balanced(ok, n_new, shares, args.seed, existing)
    if len(new) < n_new:
        print(f"    WARNING: only {len(new)} new tiles (wanted {n_new}). "
              "Raise --points-per-region or --oversample, or lower --min-s1-months.")
    new["split"] = assign_splits(new, args.seed)
    new["biome_name"] = new["biome"].map(BIOME_NAMES)
    new["utm_epsg"] = [utm_epsg(lo, la) for lo, la in zip(new["lon"], new["lat"])]
    new = new.drop(columns="settle")
    start = 0 if existing is None else int(existing["tile_id"].str[1:].astype(int).max()) + 1
    new.insert(0, "tile_id", [f"t{i:05d}" for i in range(start, start + len(new))])
    tiles = new if existing is None else pd.concat([existing, new], ignore_index=True)

    write_outputs(tiles, Path(args.out_dir))
    summarise(tiles)
    print(f"\nWrote {args.out_dir}/tiles.csv, tiles.geojson, tiles_map.html")


if __name__ == "__main__":
    main()