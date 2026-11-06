"""Streamlit prototype: Sentinel-1 land use / land cover for any location, 12 months from a start month.

Run:  streamlit run app.py
Plays the role of the Flask "prediction miniapp" from lab 14, with a map to pick the location.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import pipeline as P

ROOT = Path(__file__).resolve().parents[1]
DEMO = os.environ.get("S1LULC_DEMO") == "1"   # offline UI check with synthetic data, no Earth Engine / model

st.set_page_config(page_title="Sentinel-1 LULC monitor", layout="wide")


# ----------------------------------------------------------------------------- cached resources

@st.cache_resource(show_spinner="Connecting to Earth Engine ...")
def ee_init(project: str):
    import ee
    ee.Initialize(project=project)
    return True


@st.cache_resource(show_spinner="Loading the trained model ...")
def get_model(run_dir: str):
    return P.load_model(run_dir)


@st.cache_data(show_spinner=False)
def get_s1(lat: float, lon: float, start_ym: str, size: int, terrain_flatten: bool):
    g = P.area_grid(lat, lon, size)
    s1, n, scenes = P.fetch_s1(g, start_ym, terrain_flatten=terrain_flatten)
    return g, s1, n, scenes


@st.cache_data(show_spinner=False)
def get_obt(g: dict, start_ym: str, end_ym: str):
    return P.fetch_obt_built(g, start_ym, end_ym)


@st.cache_data(show_spinner=False)
def get_ref(g: dict, year: int):
    return P.fetch_esri(g, year)


def demo_data(size: int, start_ym: str):
    """Synthetic predictions for checking the UI only."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:size, 0:size]
    base = np.full((size, size), 3, np.int8)                       # crops
    base[xx < size * 0.35] = 1                                         # trees
    base[(yy - size / 2) ** 2 + (xx - size * 0.65) ** 2 < (size * 0.18) ** 2] = 4   # built-up
    base[yy > size * 0.85] = 0                                         # water
    site = (np.abs(yy - size * 0.3) < size * 0.06) & (np.abs(xx - size * 0.6) < size * 0.08)   # new estate
    pred = np.stack([base] * 12)
    for m in range(12):
        noise = rng.random((size, size)) < 0.03 + 0.02 * m
        pred[m][noise & (base == 3)] = 5 if 3 <= m <= 8 else 2
        if m >= 6:                                                     # built up from month 7 on
            pred[m][site & (xx < size * (0.52 + 0.03 * (m - 5)))] = 4
    n = np.array([4, 5, 0, 5, 6, 4, 5, 5, 4, 5, 6, 5], np.int16)
    pred[n == 0] = -1
    ref = base.copy()
    ref[(yy < size * 0.1)] = 2
    ok = np.ones((size, size), bool)
    obt = {"date_a": "2022-06-30", "date_b": "2023-06-30", "valid": ok,
           "built_a": base == 4, "built_b": (base == 4) | (site & (xx < size * 0.66))}
    return pred, n, ref, obt


# ----------------------------------------------------------------------------- sidebar

with st.sidebar:
    st.header("Settings")
    run_dir = st.text_input("Trained model folder", str(ROOT / "train" / "runs" / "E1"),
                            help="A training run folder containing best.pt, config.json and stats.json")
    project = st.text_input("Earth Engine project", os.environ.get("EE_PROJECT", "scm-ksns"))
    size_label = st.selectbox("Area size", ["2.56 km × 2.56 km", "5.12 km × 5.12 km", "7.68 km × 7.68 km"],
                              index=1)
    size = {"2.56": 256, "5.12": 512, "7.68": 768}[size_label[:4]]
    smooth_months = st.checkbox("Smooth predictions across months", value=False,
                                help="3-month majority + trees/shrub seasonal fix, the same rule used for the "
                                     "training labels. Reduces fake change from classification flicker.")
    do_ref = st.checkbox("Check against independent maps", value=True,
                         help="Never used in training. Built-up change: Open Buildings 2.5D Temporal (buildings, "
                              "Africa / South & Southeast Asia / Latin America, 2016-2023) and ESRI 10 m land cover "
                              "for the start and end years. Class map: ESRI on the same 6 classes.")
    min_px = st.slider("Smallest change patch counted (pixels of 10 m × 10 m)", 1, 25, 4,
                       help="New or lost built-up patches smaller than this are treated as noise. 4 px = 400 m².")


# ----------------------------------------------------------------------------- input

st.title("Sentinel-1 land use and land cover monitor")
st.caption("Pick a location and a start month. For each of 12 consecutive months the model reads one raw "
           "Sentinel-1 radar scene (no averaging) and maps 6 land-cover classes.")

PRESET_NAMES = list(P.PRESETS)
if "lat" not in st.session_state:
    st.session_state.lat, st.session_state.lon = P.PRESETS[PRESET_NAMES[0]]   # Lagos, fast-growing


def _use_preset():
    name = st.session_state.preset
    if name in P.PRESETS:
        st.session_state.lat, st.session_state.lon = P.PRESETS[name]


st.selectbox("Example places (or click the map / type coordinates)", PRESET_NAMES + ["Custom location"],
             key="preset", on_change=_use_preset)

left, right = st.columns([3, 2])
with left:
    import folium
    from streamlit_folium import st_folium
    fmap = folium.Map(location=[st.session_state.lat, st.session_state.lon], zoom_start=12, tiles="OpenStreetMap")
    half_deg = size * P.PIXEL_M / 2 / 111_320
    half_lon = half_deg / max(np.cos(np.radians(st.session_state.lat)), 1e-6)
    folium.Rectangle([[st.session_state.lat - half_deg, st.session_state.lon - half_lon],
                      [st.session_state.lat + half_deg, st.session_state.lon + half_lon]],
                     color="#c4281b", weight=2, fill=False).add_to(fmap)
    clicked = st_folium(fmap, height=380, use_container_width=True, key="map")
    if clicked and clicked.get("last_clicked"):
        c = clicked["last_clicked"]
        if (round(c["lat"], 5), round(c["lng"], 5)) != (round(st.session_state.lat, 5), round(st.session_state.lon, 5)):
            st.session_state.lat, st.session_state.lon = c["lat"], c["lng"]
            st.session_state.preset = "Custom location"
            st.rerun()
    st.caption("Click the map to move the analysis area (red square).")

with right:
    lat = st.number_input("Latitude", -60.0, 75.0, float(st.session_state.lat), format="%.5f")
    lon = st.number_input("Longitude", -180.0, 180.0, float(st.session_state.lon), format="%.5f")
    st.session_state.lat, st.session_state.lon = lat, lon
    today = dt.date.today()
    c1, c2 = st.columns(2)
    years = list(range(2017, today.year + 1))[::-1]
    year = c1.selectbox("Start year", years, index=years.index(2022),
                        help="July 2022 to June 2023 lines up with the Open Buildings snapshots (30 June each year)")
    month = c2.selectbox("Start month", list(range(1, 13)), index=6,
                         format_func=lambda m: dt.date(2000, m, 1).strftime("%B"))
    start_ym = f"{year:04d}-{month:02d}"
    months = P.month_list(start_ym)
    st.write(f"Months analysed: **{months[0]} to {months[-1]}**")
    go = st.button("Run", type="primary", use_container_width=True)


# ----------------------------------------------------------------------------- processing

if go:
    try:
        with st.status("Running ...", expanded=True) as status:
            refs, ref, obt, esri_ab = [], None, None, None
            if DEMO:
                pred_raw, n, ref, obt = demo_data(size, start_ym)
                g, s1 = P.area_grid(lat, lon, size), None
                scenes = [{"date": f"{mm}-1{i % 9}", "orbit": "ASCENDING" if i % 2 else "DESCENDING"} if n[i] else
                          {"date": None, "orbit": None} for i, mm in enumerate(months)]
                st.write("Demo mode: synthetic data")
            else:
                ee_init(project)
                model, stats, cfg, device = get_model(run_dir)
                st.write(f"Model: U-Net {cfg.get('encoder')} ({Path(run_dir).name}), running on {device}")
                st.write("Picking one raw Sentinel-1 scene per month in Earth Engine and downloading it ...")
                g, s1, n, scenes = get_s1(lat, lon, start_ym, size, bool(stats.get("terrain_flattened", True)))
                st.write("Scenes used: " + ", ".join(sc["date"] or "none" for sc in scenes))
                st.write("Predicting" + (" (Lee filter, as in training) ..." if cfg.get("lee") else " ..."))
                pred_raw = P.predict(model, stats, s1, n, device, cfg.get("lee", False))
                if do_ref:
                    y0, y1 = int(months[0][:4]), int(months[-1][:4])
                    st.write(f"Fetching ESRI land cover {y0}" + (f" and {y1}" if y1 != y0 else "") + " ...")
                    try:
                        ref = get_ref(g, y0)
                        esri_ab = (y0, y1, ref, get_ref(g, y1)) if y1 != y0 else None
                    except Exception as e:  # noqa: BLE001 - a missing reference year should not stop the app
                        st.write(f"ESRI not available: {e}")
                    st.write("Fetching Open Buildings 2.5D Temporal snapshots ...")
                    try:
                        obt = get_obt(g, months[0], months[-1])
                    except Exception as e:  # noqa: BLE001
                        st.write(f"Open Buildings Temporal not available: {e}")
            pred = P.smooth(pred_raw) if smooth_months else pred_raw
            status.update(label="Done", state="complete", expanded=False)
        st.session_state.result = {"g": g, "months": months, "n": n, "s1": s1, "scenes": scenes,
                                   "pred": pred, "pred_raw": pred_raw,
                                   "ref": ref, "ref_year": int(months[0][:4]), "obt": obt, "esri_ab": esri_ab,
                                   "smoothed": smooth_months, "lat": lat, "lon": lon}
    except Exception as e:  # noqa: BLE001 - show the reason in the app
        st.error(f"Could not finish: {e}")
        st.info("Check the model folder, that the Earth Engine project is registered for noncommercial use, "
                "and that `earthengine authenticate` has been run on this machine.")


# ----------------------------------------------------------------------------- results

res = st.session_state.get("result")
if not res:
    st.stop()

months, n, pred = res["months"], res["n"], res["pred"]
pct = P.class_percentages(pred, months)
mom = P.month_to_month(pct)
tot = P.total_change(pct)
km = res["g"]["size"] * P.PIXEL_M / 1000

st.divider()
st.subheader(f"Results · {res['lat']:.4f}, {res['lon']:.4f} · {km:.2f} km × {km:.2f} km")
m1, m2, m3 = st.columns(3)
m1.metric("Months with a Sentinel-1 scene", f"{int((n > 0).sum())} of 12")
m2.metric("Predictions", "Smoothed" if res["smoothed"] else "Raw",
          help="Raw = each month predicted on its own; smoothed = 3-month majority + seasonal fix")
if res["ref"] is not None:
    agree, ref_table = P.compare_with_reference(P.year_majority(pred), res["ref"])
    m3.metric(f"Agreement with ESRI {res['ref_year']}", f"{agree:.1f} %", help="6 merged classes, pixels valid in both")

def fmt(df: pd.DataFrame) -> pd.DataFrame:
    """One decimal; empty cells shown as a dash."""
    return df.apply(lambda col: col.map(lambda v: "–" if pd.isna(v) else f"{v:.1f}"))


legend = " ".join(f"<span style='display:inline-block;width:11px;height:11px;background:{c};"
                  f"margin:0 4px 0 10px;vertical-align:middle'></span>{nme}"
                  for nme, c in zip(P.CLASS_NAMES, P.CLASS_COLOURS))

tab_built, tab_maps, tab_pct, tab_change, tab_ref, tab_input = st.tabs(
    ["Built-up change", "Maps", "Class % per month", "All-class change", "Independent check", "Sentinel-1 input"])

with tab_built:
    bc = P.built_change(pred, months, window=3, min_px=min_px)
    if bc is None:
        st.warning("Fewer than 2 months with Sentinel-1 data: no change can be measured.")
    else:
        st.markdown(f"How much **built-up area** (buildings, roads, paved ground) changed between the start "
                    f"(**{', '.join(bc['first'])}**) and the end (**{', '.join(bc['last'])}**) of the 12 months.")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(f"Built-up (start {bc['before_pct']:.1f} %)", f"{bc['after_pct']:.1f} %",
                  f"{bc['change_pp']:+.2f} pp", help="Share of the area with data at the end, and the change since "
                                                      "the start in percentage points (pp)")
        k2.metric("New built-up", f"{bc['gain_ha']:.1f} ha", help="Built in most end months, in none of the start months")
        k3.metric("Built-up lost", f"{bc['loss_ha']:.1f} ha", help="The reverse; usually misclassification or demolition")
        k4.metric("Net change", f"{bc['net_ha']:+.1f} ha", help=f"New minus lost, of {bc['area_ha']:.0f} ha analysed")
        a, b = st.columns([3, 2])
        a.image(P.change_rgb(bc), use_container_width=True, caption="Built-up change map")
        with b:
            sw = lambda rgb, t: (f"<span style='display:inline-block;width:12px;height:12px;background:rgb{rgb};"  # noqa: E731
                                 f"border:1px solid #999;margin-right:6px;vertical-align:middle'></span>{t}<br>")
            st.markdown(sw((214, 32, 181), "new built-up") + sw((0, 170, 200), "built-up lost")
                        + sw((90, 90, 90), "built-up at start and end") + sw((238, 238, 238), "not built-up")
                        + sw((255, 255, 255), "no data"), unsafe_allow_html=True)
            st.markdown("**Built-up % per month**")
            st.bar_chart(pct[P.CLASS_NAMES[P.BUILT]], color=P.CLASS_COLOURS[P.BUILT], height=200)

        st.markdown("**Checked against independent sources** (never used in training)")
        refs = []
        if res.get("obt"):
            o = res["obt"]
            refs.append(P.reference_change(f"Open Buildings Temporal ({o['date_a']} vs {o['date_b']})",
                                           o["built_a"], o["built_b"], o["valid"], bc))
        if res.get("esri_ab"):
            y0, y1, ea, eb = res["esri_ab"]
            refs.append(P.reference_change(f"ESRI land cover ({y0} vs {y1})", ea == P.BUILT, eb == P.BUILT,
                                           (ea >= 0) & (eb >= 0), bc))
        st.dataframe(fmt(P.built_change_table(bc, refs)), use_container_width=True)
        notes = []
        if not res.get("obt"):
            notes.append("Open Buildings Temporal does not cover this place (it covers Africa, South & Southeast "
                         "Asia and Latin America, 2016-2023) or was not fetched.")
        if not res.get("esri_ab"):
            notes.append("ESRI change needs a 12-month window that spans two calendar years (e.g. start in July)."
                         if months[0][:4] == months[-1][:4] else "ESRI land cover was not fetched for this run.")
        for t in notes:
            st.caption(t)
        with st.expander("How built-up change is measured, and why it is robust"):
            st.markdown(
                "- **Built-up at start / end** = pixels predicted *built-up* in the majority of the first / last "
                "3 months with data, as % of pixels observed in both periods.\n"
                "- **New built-up** = built-up in the majority of the last 3 months **and in none** of the first 3. "
                "A pixel misclassified in a single month therefore never counts as construction.\n"
                f"- Patches smaller than **{min_px} pixels** ({min_px * 100} m²) are dropped as noise; one pixel "
                "is 10 m × 10 m = 0.01 ha.\n"
                "- *Our new built-up that it confirms* = share of our new built-up pixels that the reference "
                "also shows as built at the end. Open Buildings maps buildings only (no roads or paved yards), "
                "so it will read lower than our built-up class.\n"
                "- Known causes of false change in radar: wet soil after rain, crops with strong double-bounce, "
                "and scenes from a different orbit direction. The map tab shows the orbit of each scene.")
        st.download_button("Download built-up change (CSV)",
                           P.built_change_table(bc, refs).round(3).to_csv().encode(), "built_up_change.csv",
                           "text/csv")

with tab_maps:
    pick = st.select_slider("Month", options=months, value=months[0])
    i = months.index(pick)
    a, b = st.columns([3, 2])
    sc = res["scenes"][i]
    a.image(P.colourise(pred[i]), use_container_width=True,
            caption=(f"{pick} · scene of {sc['date']} ({sc['orbit'].lower()} pass) · {int(n[i])} scenes that month"
                     if n[i] else f"{pick}: no Sentinel-1 scene"))
    with b:
        st.markdown(legend, unsafe_allow_html=True)
        st.dataframe(pct.loc[[pick]].T.rename(columns={pick: "% of area"}).round(1), use_container_width=True)
    st.markdown("**All 12 months**")
    st.pyplot(P.grid_figure(pred, months, n))
    gif = P.animation_gif(pred, months, scale=1)
    g1, g2 = st.columns([2, 3])
    g1.image(gif, caption="Animation over the 12 months", use_container_width=True)
    g2.download_button("Download animation (GIF)", gif, "lulc_12_months.gif", "image/gif")

with tab_pct:
    st.markdown("Share of the area covered by each class, in % of pixels with data that month.")
    st.dataframe(fmt(pct), use_container_width=True)
    st.line_chart(pct, color=P.CLASS_COLOURS)
    st.download_button("Download class % (CSV)", pct.round(3).to_csv().encode(), "class_percent.csv", "text/csv")

with tab_change:
    st.markdown("**Month-to-month change** (percentage points): this month's % minus the previous month's %.")
    st.dataframe(fmt(mom), use_container_width=True)
    st.caption("– = no Sentinel-1 data that month, or none the month before.")
    st.markdown(f"**Total change, first to last month with data** ({tot.name}, percentage points)")
    st.dataframe(fmt(tot.to_frame("change (pp)").T), use_container_width=True)
    pc = P.period_change(pred, months)
    if pc is not None:
        st.markdown("**Robust total change**: majority map of the first 3 vs the last 3 months with data "
                    "(single-month flicker cancels out)")
        st.dataframe(fmt(pc), use_container_width=True)
    with st.expander("How the numbers are calculated, and what can inflate them"):
        st.markdown(
            "- **Class %** for month *m*, class *c* = pixels predicted *c* / pixels with data × 100.\n"
            "- **Month-to-month change** = %(*m*) − %(*m*−1), in percentage points (pp).\n"
            "- **Total change** = % in the last month with data − % in the first month with data.\n"
            "- A month without Sentinel-1 data has no values and is skipped.\n"
            "- Classification errors that flicker between months add up as fake change; seasonal effects "
            "(leaf loss, harvest, flooding) can also look like land-use change. Compare with "
            "**Smooth predictions across months** in the sidebar.")
    pr = P.class_percentages(res["pred_raw"], months)
    flicker = float(np.nanmean(np.abs(P.month_to_month(pr)).sum(axis=1)))
    st.caption(f"Average total |month-to-month change| over all classes, raw predictions: {flicker:.1f} pp per month.")

with tab_ref:
    if res["ref"] is None:
        st.info("Turn on the ESRI check in the sidebar and run again.")
    else:
        st.markdown(f"Our map (most frequent class over the 12 months) against **ESRI 10 m land cover "
                    f"{res['ref_year']}**, an independent Sentinel-2 product not used in training, on the same "
                    "6 classes (ESRI's *rangeland* = our grass & shrub).")
        ours = P.year_majority(pred)
        a, b = st.columns(2)
        a.image(P.colourise(ours), caption="Our model (year majority)", use_container_width=True)
        b.image(P.colourise(res["ref"]), caption=f"ESRI {res['ref_year']}", use_container_width=True)
        st.markdown(legend, unsafe_allow_html=True)
        st.dataframe(fmt(ref_table), use_container_width=True)

with tab_input:
    if res["s1"] is None:
        st.info("Demo mode: no Sentinel-1 input.")
    else:
        st.markdown("The raw Sentinel-1 scene used for each month: VV backscatter (dB), the model's main input.")
        cols = st.columns(6)
        for i in range(12):
            vv = res["s1"][i, 0].astype(float) / 100
            vv[res["s1"][i, 0] == P.S1_NODATA] = np.nan
            img = np.clip((vv + 25) / 25, 0, 1)
            img = np.nan_to_num(img, nan=0.92)
            sc = res["scenes"][i]
            cols[i % 6].image((img * 255).astype(np.uint8), use_container_width=True,
                              caption=f"{sc['date']} · {sc['orbit'][:3].lower()}" if sc["date"] else f"{months[i]} · none")
