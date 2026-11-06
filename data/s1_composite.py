"""Sentinel-1 input recipe: the ONE definition used by the training export and the prototype.

If the prototype prepared Sentinel-1 differently from the training data, the model would see unfamiliar input,
so both call these functions.

Model input = SINGLE RAW SCENES (decided 2026-11-03), no temporal averaging.

Pre-processing done by Earth Engine for COPERNICUS/S1_GRD (before our code):
    thermal noise removal, radiometric calibration to sigma0, terrain correction (orthorectification
    with SRTM 30 m / ASTER DEM), conversion to dB.
Our steps for one scene (s1_scene_images):
    1. keep Interferometric Wide swath scenes with both VV and VH
    2. mask border noise and extreme lows (VV <= -30 dB or VH <= -35 dB)
    3. radiometric terrain flattening (decided 2026-11-03): sigma0 -> gamma0 with the angular-based volume model
       of Vollrath, Mullissa & Reiche (2020), Remote Sensing 12(11):1867, on the Copernicus GLO-30 DEM;
       layover and radar-shadow pixels masked. Without it, slopes facing the radar look as bright as buildings
       and slopes facing away as dark as water. The DEM is preprocessing only: the model input stays Sentinel-1.
    4. one acquisition = all GRD slices of the same day and relative orbit, mosaicked (a tile can straddle two
       consecutive slices of the same pass; this joins them, it does not average different dates)
    5. add the VH - VV ratio band (dB difference)
    6. store as int16 dB x 100; -32768 = no data
No speckle filter by default; lee_filter_db() is the optional 5 x 5 Lee filter (experiment variant).

s1_month() is the earlier monthly-median composite (kept so old exports stay readable; no longer model input).
"""

from __future__ import annotations

import numpy as np

S1_NODATA = -32768
S1_BANDS = ["VV", "VH", "VHVV"]
PREPROCESSING = "S1_GRD IW, border mask, terrain-flattened gamma0 (Vollrath 2020 volume model, GLO-30), dB"


# ----------------------------------------------------------------------------- Earth Engine

def s1_collection(region):
    import ee
    return (ee.ImageCollection("COPERNICUS/S1_GRD")
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .filterBounds(region)
            .select(["VV", "VH", "angle"]))


def dem_slope_aspect(region):
    """Terrain slope and aspect (radians) from the Copernicus GLO-30 DEM around `region`."""
    import ee
    dems = ee.ImageCollection("COPERNICUS/DEM/GLO30_2024_1").filterBounds(region).select("DEM")
    dem = dems.mosaic().setDefaultProjection(dems.first().projection())
    deg = np.pi / 180
    return ee.Terrain.slope(dem).multiply(deg), ee.Terrain.aspect(dem).multiply(deg)


def flatten(img, slope, aspect, region=None):
    """sigma0 (dB, VV + VH + angle) -> terrain-flattened gamma0 (dB, VV + VH), layover / shadow masked.

    Volume model (Vollrath et al. 2020):
        alpha_r = atan(tan(slope) * cos(look azimuth - terrain aspect))     slope component in range direction
        gamma0  = sigma0 / cos(theta)                                       theta = ellipsoid incidence angle
        gamma0_flat = gamma0 * tan(90 deg - theta) / tan(90 deg - theta + alpha_r)
    Layover where alpha_r >= theta, shadow where alpha_r <= theta - 90 deg.
    Azimuths: Earth Engine's aspect points downhill. The incidence angle grows away from the sensor, so the
    aspect of the angle field points TOWARD the sensor; a slope facing the sensor then has aspect ~ look azimuth,
    alpha_r > 0 and is darkened (factor < 1), a slope facing away is brightened.
    """
    import ee
    deg = np.pi / 180
    theta = img.select("angle").multiply(deg)
    # Look direction, measured only around the area of interest: it is practically constant over a few km.
    # (Reducing over the whole 250 km scene for every scene exceeded Earth Engine's memory limit on some tiles.)
    # The angle band is read unmasked (water / border masks would leave nothing to measure on coasts),
    # and if the tile area still gives no value, a coarse estimate over the whole scene is used.
    ang_aspect = ee.Terrain.aspect(img.select("angle").unmask(img.select("angle"), False))
    whole = ang_aspect.reduceRegion(ee.Reducer.mean(), img.geometry(), 5000, bestEffort=True, maxPixels=1e6).get("aspect")
    if region is None:
        look = whole
    else:
        local = ang_aspect.reduceRegion(ee.Reducer.mean(), region.buffer(3000, 100), 300,
                                        bestEffort=True, maxPixels=1e6).get("aspect")
        look = ee.Algorithms.If(ee.Algorithms.IsEqual(local, None), whole, local)
    phi_r = ee.Image.constant(look).multiply(deg).subtract(aspect)
    alpha_r = slope.tan().multiply(phi_r.cos()).atan()
    ninety = ee.Image.constant(np.pi / 2)
    gamma0 = ee.Image(10).pow(img.select(["VV", "VH"]).divide(10)).divide(theta.cos())
    flat = gamma0.multiply(ninety.subtract(theta).tan()).divide(ninety.subtract(theta).add(alpha_r).tan())
    ok = alpha_r.lt(theta).And(alpha_r.gt(theta.subtract(ninety)))
    return ee.Image(flat.log10().multiply(10).updateMask(ok).copyProperties(img, img.propertyNames()))


def _clean(img):
    return img.updateMask(img.select("VV").gt(-30).And(img.select("VH").gt(-35)))


def _to_int16(img):
    """VV, VH (dB) -> VV, VH, VHVV as int16 dB x 100, nodata -32768."""
    full = img.select(["VV", "VH"]).addBands(img.select("VH").subtract(img.select("VV")).rename("VHVV"))
    return full.select(S1_BANDS).multiply(100).round().unmask(S1_NODATA).toInt16()


def s1_scene_images(region, start, end, terrain_flatten: bool = True):
    """ImageCollection with ONE image per acquisition over `region` in [start, end).

    Each image: VV, VH, VHVV int16 (dB x 100). Properties: key ('YYYY-MM-dd_<relorbit>'), date ('YYYY-MM-dd'),
    pass ('ASCENDING' / 'DESCENDING'), rel_orbit, system:time_start.
    """
    import ee

    def keyed(im):
        date = im.date().format("YYYY-MM-dd")
        rel = ee.Number(im.get("relativeOrbitNumber_start")).format("%d")
        return im.set({"key": date.cat("_").cat(rel), "date": date})

    col = s1_collection(region).filterDate(start, end).map(keyed)
    keys = col.aggregate_array("key").distinct().sort()
    if terrain_flatten:
        slope, aspect = dem_slope_aspect(region)

    def prep(im):
        keep = im.select("VV").gt(-30).And(im.select("VH").gt(-35))      # same rule as _clean, on sigma0
        out = flatten(im, slope, aspect, region) if terrain_flatten else im.select(["VV", "VH"])
        return out.updateMask(keep)

    def one(k):
        sub = col.filter(ee.Filter.eq("key", k))
        first = ee.Image(sub.first())
        return (_to_int16(sub.map(prep).mosaic())
                .set({"key": k, "date": first.get("date"),
                      "pass": first.get("orbitProperties_pass"),
                      "rel_orbit": first.get("relativeOrbitNumber_start"),
                      "system:time_start": first.get("system:time_start")}))

    return ee.ImageCollection(keys.map(one))


def s1_scene_table(region, start: str, end: str) -> list[dict]:
    """Client-side list of acquisitions over `region` in [start, end): one entry per date + relative orbit,
    with the ids of its slices. Only image properties are read, so this stays light where Sentinel-1 is dense
    (Europe: several hundred slices a year). Building every scene image just to read its properties
    (s1_scene_images) exceeded Earth Engine's memory limit there."""
    import datetime as dt
    import ee
    rows = (s1_collection(region).filterDate(start, end)
            .reduceColumns(ee.Reducer.toList(4), ["system:time_start", "relativeOrbitNumber_start",
                                                  "orbitProperties_pass", "system:index"])
            .get("list").getInfo()) or []
    scenes: dict[str, dict] = {}
    for t, rel, pas, idx in rows:
        date = dt.datetime.fromtimestamp(t / 1000, dt.timezone.utc).strftime("%Y-%m-%d")
        key = f"{date}_{int(rel)}"
        sc = scenes.setdefault(key, {"key": key, "date": date, "pass": pas, "rel_orbit": int(rel),
                                     "time": t, "ids": []})
        sc["ids"].append(idx)
        sc["time"] = min(sc["time"], t)
    return [scenes[k] for k in sorted(scenes)]


def s1_scene_image(region, start: str, end: str, ids: list[str], terrain_flatten: bool = True, dem=None):
    """One acquisition (its same-day slices joined) prepared exactly like s1_scene_images: VV, VH, VHVV int16.
    `dem` = dem_slope_aspect(region), passed in so it is built once per tile."""
    import ee
    sub = s1_collection(region).filterDate(start, end).filter(ee.Filter.inList("system:index", ids))
    if terrain_flatten:
        slope, aspect = dem if dem is not None else dem_slope_aspect(region)

    def prep(im):
        keep = im.select("VV").gt(-30).And(im.select("VH").gt(-35))
        out = flatten(im, slope, aspect, region) if terrain_flatten else im.select(["VV", "VH"])
        return out.updateMask(keep)

    return _to_int16(sub.map(prep).mosaic())


def s1_month(s1_all, start, end):
    """(Earlier approach) monthly median composite: VV, VH, VHVV (dB x 100) + N scenes, int16."""
    import ee
    s1 = s1_all.filterDate(start, end).map(_clean).select(["VV", "VH"])
    has = s1.size().gt(0)
    med0 = s1.median()
    empty = ee.Image.constant([0, 0, 0]).rename(S1_BANDS).updateMask(0)
    med = ee.Image(ee.Algorithms.If(
        has, med0.addBands(med0.select("VH").subtract(med0.select("VV")).rename("VHVV")), empty))
    n = ee.Image(ee.Algorithms.If(
        has, s1.select("VV").count().unmask(0).rename("N"), ee.Image.constant(0).rename("N")))
    s1_i16 = med.select(S1_BANDS).multiply(100).round().unmask(S1_NODATA)
    return s1_i16.addBands(n).toInt16()


# ----------------------------------------------------------------------------- numpy helpers

def valid_fraction(s1: np.ndarray) -> np.ndarray:
    """Share of pixels with data, per scene. s1: (S, 3, H, W) or (3, H, W)."""
    ok = (s1 != S1_NODATA).all(axis=-3)
    return ok.reshape(ok.shape[:-2] + (-1,)).mean(-1)


def best_scene_per_month(months: np.ndarray, days: np.ndarray, valid: np.ndarray, n_months: int = 12) -> np.ndarray:
    """Index of the scene used for each month's map (-1 if none): full coverage first, then closest to mid-month.

    The prototype applies the same rule in Earth Engine (pipeline.fetch_s1).
    """
    out = np.full(n_months, -1, dtype=np.int32)
    for m in range(n_months):
        idx = np.flatnonzero(months == m)
        if idx.size:
            score = (valid[idx] < 0.999) * 1000 + np.abs(days[idx] - 15)
            out[m] = idx[np.argmin(score)]
    return out


def _box_mean(a: np.ndarray, k: int) -> np.ndarray:
    """k x k moving average with edge padding (k odd), via cumulative sums."""
    p = k // 2
    a = np.pad(a, p, mode="edge")
    c = np.cumsum(np.cumsum(a, 0), 1)
    c = np.pad(c, ((1, 0), (1, 0)))
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def lee_filter_db(s1: np.ndarray, size: int = 5, looks: float = 4.4) -> np.ndarray:
    """Lee speckle filter on one scene (3, H, W) int16 dB x 100; returns the same format.

    Works in linear power (as speckle is multiplicative), on VV and VH; VH-VV is recomputed.
    `looks` = equivalent number of looks of Sentinel-1 IW GRD (about 4.4).
    """
    out = s1.copy()
    nod = (s1 == S1_NODATA).any(0)
    cu2 = 1.0 / looks
    for b in (0, 1):
        lin = 10 ** (s1[b].astype(np.float64) / 1000)          # dB x 100 -> linear power
        lin[nod] = np.nan
        fill = np.nanmean(lin) if np.isfinite(lin).any() else 1.0
        lin = np.where(nod, fill, lin)
        mean = _box_mean(lin, size)
        var = np.maximum(_box_mean(lin ** 2, size) - mean ** 2, 0)
        w = np.clip((var - mean ** 2 * cu2) / np.maximum(var * (1 + cu2), 1e-12), 0, 1)
        filt = mean + w * (lin - mean)
        out[b] = np.round(1000 * np.log10(np.maximum(filt, 1e-6))).astype(np.int16)
    out[2] = out[1] - out[0]
    out[:, nod] = S1_NODATA
    return out