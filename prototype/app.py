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
def get_models(run_dirs: tuple):
    return P.load_models(run_dirs)


@st.cache_data(show_spinner=False)
def get_s1(lat: float, lon: float, start_ym: str, size: int, terrain_flatten: bool, n_months: int = 12):
    g = P.area_grid(lat, lon, size)
    scenes_all, n, scenes, s1_show = P.fetch_s1_all(g, start_ym, terrain_flatten=terrain_flatten, n_months=n_months)
    return g, scenes_all, n, scenes, s1_show


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
    mode = st.radio("Mode", ["Gallery (pre-computed examples)", "Live run (any location)"],
                    help="Gallery opens saved results instantly; Live run downloads Sentinel-1 for any point "
                         "(2-5 min, needs Earth Engine on this machine).")
    LIVE = mode.startswith("Live")
    project, size = os.environ.get("EE_PROJECT", "scm-ksns"), 512
    if LIVE:
        size_label = st.selectbox("Area size", ["2.56 km × 2.56 km", "5.12 km × 5.12 km", "7.68 km × 7.68 km"],
                                  index=1)
        size = {"2.56": 256, "5.12": 512, "7.68": 768}[size_label[:4]]
        project = st.text_input("Earth Engine project", project)
    with st.expander("Advanced settings"):
        runs_txt = st.text_input("Trained model(s)", ", ".join(P.PROTOTYPE["runs"]),
                                 help="Run folder name(s) in train/runs (or full paths), comma-separated. "
                                      "Several = ensemble: their probabilities are averaged.")
        use_tta = st.checkbox("Test-time augmentation", value=P.PROTOTYPE["tta"],
                              help="Also predict every scene flipped (4 views) and average. Slower, steadier.")
        use_s3 = st.checkbox("3-month probability smoothing", value=P.PROTOTYPE.get("smooth3", False),
                             help="Each month's probabilities averaged with the previous and next month before "
                                  "choosing the class. Halved month-to-month flicker on the test places.")
        mode = st.radio("Built-up change compares", ["same_season", "within_year"],
                        index=["same_season", "within_year"].index(P.PROTOTYPE.get("change_mode", "same_season")),
                        format_func=lambda m: {"same_season": "same 3 months, one year later",
                                               "within_year": "first vs last 3 of the 12 months"}[m],
                        help="Same season: the first 3 months are compared with the same calendar months a year "
                             f"later ({P.EXTRA_MONTHS} extra months are downloaded), so wet and dry seasons cannot "
                             "look like construction. Within the year: first 3 vs last 3 months.")
        rule = st.radio("New built-up rule", ["moderate", "confident", "argmax"],
                        index=["moderate", "confident", "argmax"].index(P.PROTOTYPE["change_rule"]),
                        help="moderate (default) / confident: mean built-up probability >= 0.5 at the end and below "
                             "0.35 / 0.2 in every start month. confident = lower bound; argmax = built-up in most end "
                             "months and none of the start months = upper bound.")
        smooth_months = st.checkbox("Majority smoothing of the maps", value=False,
                                    help="3-month majority + trees/shrub seasonal fix, the rule used for the training "
                                         "labels.")
        do_ref = st.checkbox("Check against independent maps (live run)", value=True,
                             help="Open Buildings 2.5D Temporal and ESRI 10 m land cover, never used in training.")
        min_px = st.slider("Smallest change patch (pixels)", 1, 25, 4,
                           help="New or lost built-up patches smaller than this are treated as noise. 4 px = 400 m².")
    run_dirs = tuple(str(r) if Path(r).is_absolute() else str(ROOT / "train" / "runs" / r)
                     for r in (x.strip() for x in runs_txt.split(",")) if r)
    run_dir = run_dirs[0]


# ----------------------------------------------------------------------------- input

st.title("Sentinel-1 land use and land cover monitor")
st.caption("Land cover from Sentinel-1 radar only: 12 monthly maps for any place, and how much was built in a year.")

GALLERY_DIR = Path(__file__).resolve().parent / "case_studies"


@st.cache_data(show_spinner="Opening the example ...")
def load_case(path: str) -> dict:
    import gzip
    import pickle
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


if not LIVE:
    files = sorted(GALLERY_DIR.glob("*.pkl.gz"))
    try:                                   # the order of case_studies.CASES: fast growth first, hard places last
        from case_studies import CASES
        _order = [c[0] for c in CASES]
        files.sort(key=lambda f: _order.index(f.name.split(".")[0]) if f.name.split(".")[0] in _order else 99)
    except ImportError:
        pass
    if not files:
        st.warning("No pre-computed examples yet. Run `python3 case_studies.py --run ../train/runs/E1` "
                   "on a machine with Earth Engine access, or switch to Live run in the sidebar.")
        st.stop()
    cases = {f: load_case(str(f)) for f in files}
    pick_f = st.selectbox("Example place", files, format_func=lambda f: f"{cases[f]['case']['title']}  ·  "
                                                                       f"{cases[f]['case']['kind']}")
    case = cases[pick_f]
    c = case["case"]
    try:
        from case_studies import FINDINGS
        found = FINDINGS.get(c.get("id", ""), "")
    except ImportError:
        found = ""
    if found:
        st.info(f"**What we found:** {found}")
    with st.expander("About this place"):
        st.write(c["note"])
    st.session_state.result = case

if LIVE:
    PRESET_NAMES = list(P.PRESETS)
    if "lat" not in st.session_state:
        st.session_state.lat, st.session_state.lon = P.PRESETS[PRESET_NAMES[0]]   # Lagos, fast-growing


    def _use_preset():
        name = st.session_state.preset
        if name in P.PRESETS:
            st.session_state.lat, st.session_state.lon = P.PRESETS[name]


    if st.session_state.pop("_to_custom", False):      # set by a map click on the previous run
        st.session_state.preset = "Custom location"    # (a widget's value may only be set before it is drawn)
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
            key = (round(c["lat"], 5), round(c["lng"], 5))
            if key != st.session_state.get("_last_click"):     # only a NEW click moves the area
                st.session_state._last_click = key
                st.session_state.lat, st.session_state.lon = c["lat"], c["lng"]
                st.session_state._to_custom = True
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
                refs, ref, obt, esri_ab, pbuilt, extra = [], None, None, None, None, None
                if DEMO:
                    pred_raw, n, ref, obt = demo_data(size, start_ym)
                    g, s1 = P.area_grid(lat, lon, size), None
                    scenes = [{"date": f"{mm}-1{i % 9}", "orbit": "ASCENDING" if i % 2 else "DESCENDING"} if n[i] else
                              {"date": None, "orbit": None} for i, mm in enumerate(months)]
                    st.write("Demo mode: synthetic data")
                else:
                    ee_init(project)
                    models, device = get_models(run_dirs)
                    stats, cfg = models[0][1], models[0][2]
                    st.write(f"Model: U-Net {cfg.get('encoder')} ({' + '.join(Path(r).name for r in run_dirs)}"
                             f"{', with TTA' if use_tta else ''}), running on {device}")
                    n_all = 12 + P.EXTRA_MONTHS
                    st.write(f"Downloading every raw Sentinel-1 scene of the 12 months (+ {P.EXTRA_MONTHS} months one "
                             "year later, for the same-season change) from Earth Engine ...")
                    g, scenes_all, n_a, info_a, s1_a = get_s1(lat, lon, start_ym, size,
                                                              bool(stats.get("terrain_flattened", True)), n_all)
                    months_all = P.month_list(start_ym, n_all)
                    st.write(f"{int(n_a.sum())} scenes: " + ", ".join(f"{mm} {int(k)}" for mm, k in zip(months_all, n_a)))
                    st.write("Predicting every scene and averaging per month ...")
                    pr_all, pb_all = P.predict_month_probs(models, scenes_all, device, use_tta, smooth3=use_s3)
                    part = P.split_result(months_all, n_a, info_a, s1_a, pr_all, pb_all)
                    n, scenes, s1, pred_raw, pbuilt, extra = (part["n"], part["scenes"], part["s1"], part["pred"],
                                                              part["pbuilt"], part["extra"])
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
                                       "ref": ref, "ref_year": int(months[0][:4]), "obt": obt, "esri_ab": esri_ab, "pbuilt": pbuilt,
                                       "extra": extra,
                                       "smoothed": smooth_months, "lat": lat, "lon": lon}
        except Exception as e:  # noqa: BLE001 - show the reason in the app
            st.error(f"Could not finish: {e}")
            st.info("Check the model folder, that the Earth Engine project is registered for noncommercial use, "
                    "and that `earthengine authenticate` has been run on this machine.")


# ----------------------------------------------------------------------------- results

res = st.session_state.get("result")
if not res:
    st.stop()

months, n = res["months"], res["n"]
if int(np.asarray(n).sum()) == 0 or np.asarray(res["pred_raw"]).shape[-1] < 2:
    st.warning("**No Sentinel-1 scenes here in these 12 months**, so no map and no change can be made. Sentinel-1 "
               "does not record every spot on Earth (after Sentinel-1B failed in Dec 2021, remote areas such as parts "
               "of the Sahara were left out of the acquisition plan). Try another place or period.")
    st.stop()
pred = P.smooth(res["pred_raw"]) if smooth_months else res["pred_raw"]
pct = P.class_percentages(pred, months)
mom = P.month_to_month(pct)
tot = P.total_change(pct)
km = res["g"]["size"] * P.PIXEL_M / 1000

if res["ref"] is not None:
    agree, ref_table = P.compare_with_reference(P.year_majority(pred), res["ref"])
st.divider()
st.subheader("Results")
st.caption(f"{res['lat']:.4f}, {res['lon']:.4f} · {km:.2f} km × {km:.2f} km · {months[0]} to {months[-1]} · "
           f"radar data in {int((n > 0).sum())} of 12 months"
           + (f" · agrees with ESRI {res['ref_year']} on {agree:.0f} % of pixels" if res["ref"] is not None else ""))


def fmt(df: pd.DataFrame) -> pd.DataFrame:
    """One decimal; empty cells shown as a dash."""
    return df.apply(lambda col: col.map(lambda v: "–" if pd.isna(v) else f"{v:.1f}"))


legend = " ".join(f"<span style='display:inline-block;width:11px;height:11px;background:{c};"
                  f"margin:0 4px 0 10px;vertical-align:middle'></span>{nme}"
                  for nme, c in zip(P.CLASS_NAMES, P.CLASS_COLOURS))

# built-up change: shown in the first tab, its comparison with the references in "Checks"
res_c = {**res, "pred": pred}
c_pred, c_pb, c_months, c_first, c_last, used = P.change_inputs(res_c, mode)
bc = P.built_change(c_pred, c_months, window=3, min_px=min_px, pbuilt=c_pb, rule=rule, first=c_first, last=c_last)
refs = []
if bc is not None:
    if res.get("obt"):
        o = res["obt"]
        refs.append(P.reference_change(f"Open Buildings Temporal ({o['date_a']} vs {o['date_b']})",
                                       o["built_a"], o["built_b"], o["valid"], bc))
    if res.get("esri_ab"):
        y0, y1, ea, eb = res["esri_ab"]
        refs.append(P.reference_change(f"ESRI land cover ({y0} vs {y1})", ea == P.BUILT, eb == P.BUILT,
                                       (ea >= 0) & (eb >= 0), bc))

tab_built, tab_maps, tab_pct, tab_ref = st.tabs(["Built-up change", "Monthly maps", "Class shares", "Checks"])

with tab_built:
    if bc is None:
        st.warning("Fewer than 2 months with Sentinel-1 data: no change can be measured.")
    else:
        period = f"{bc['first'][0]} to {bc['first'][-1]} vs {bc['last'][0]} to {bc['last'][-1]}"
        st.markdown(f"Built-up area (buildings, roads, paved ground), **{period}**"
                    + (" · same season one year apart" if used == "same_season" else " · different seasons"))
        k1, k2, k3 = st.columns(3)
        k1.metric("Built-up share", f"{bc['after_pct']:.1f} %", f"{bc['change_pp']:+.1f} pp",
                  help=f"Share of the area at the end; change since the start ({bc['before_pct']:.1f} %) in "
                       "percentage points")
        k2.metric("New built-up", f"{bc['gain_ha']:.1f} ha", help="See 'How this is measured' below")
        k3.metric("Built-up lost", f"{bc['loss_ha']:.1f} ha",
                  help="Usually misclassification (wet soil, construction sites), rarely demolition")
        floor = P.noise_floor(run_dirs, rule, use_tta, P.PROTOTYPE["hi"], P.PROTOTYPE["lo"], use_s3)
        verdict, why, _ = P.change_verdict(bc, floor)
        (st.success if verdict.startswith("Clear") else st.warning if verdict.startswith("Some") else st.info)(
            f"**{verdict}.** {why}")
        rows, span = [], {}
        if c_pb is not None:
            for m_ in ("same_season", "within_year"):
                p_, b_, mo_, f_, l_, u_ = P.change_inputs(res_c, m_)
                if u_ != m_:
                    continue
                for r_ in ("confident", "moderate", "argmax"):
                    x_ = P.built_change(p_, mo_, window=3, min_px=min_px, pbuilt=b_, rule=r_, first=f_, last=l_)
                    if x_ is not None:
                        rows.append({"compares": {"same_season": "same months, +1 year",
                                                  "within_year": "first vs last 3 months"}[m_],
                                     "rule": r_, "new ha": x_["gain_ha"], "lost ha": x_["loss_ha"]})
                        if m_ == used:
                            span[r_] = x_["gain_ha"]
        if "confident" in span and "argmax" in span:
            st.caption(f"Range: {span['confident']:.1f} ha with the strict rule to {span['argmax']:.1f} ha with the "
                       "loose rule. A wide range means an uncertain result.")
        a, b = st.columns([3, 2])
        a.image(P.change_rgb(bc), use_container_width=True)
        with b:
            sw = lambda rgb, t: (f"<span style='display:inline-block;width:12px;height:12px;background:rgb{rgb};"  # noqa: E731
                                 f"border:1px solid #999;margin-right:6px;vertical-align:middle'></span>{t}<br>")
            st.markdown(sw((214, 32, 181), "new built-up") + sw((0, 170, 200), "built-up lost")
                        + sw((90, 90, 90), "built-up throughout") + sw((238, 238, 238), "not built-up")
                        + sw((255, 255, 255), "no data"), unsafe_allow_html=True)
            st.markdown("**Built-up % per month**")
            st.bar_chart(pct[P.CLASS_NAMES[P.BUILT]], color=P.CLASS_COLOURS[P.BUILT], height=200)
        with st.expander("How this is measured"):
            st.markdown(
                "- **Start and end** = the first 3 months and the same 3 months one year later (or the last 3 of the "
                "12 months), so a wet or dry season cannot look like construction.\n"
                "- **New built-up (moderate rule)** = mean built-up probability ≥ 0.5 over the end months and below "
                "0.35 in every start month. The strict rule uses 0.2; the loose rule counts any pixel built-up in "
                "most end months and none of the start months.\n"
                f"- Patches smaller than **{min_px} pixels** ({min_px * 100} m²) are dropped as noise; one pixel is "
                "10 m × 10 m = 0.01 ha.\n"
                "- **Verdict**: the new area compared with what the model reports by error on 130 unseen test places "
                "of the same size: at least 3× = clear growth, below 1.5× = no change detectable.")
            if rows:
                st.dataframe(pd.DataFrame(rows).set_index(["compares", "rule"]).round(1), use_container_width=True)

with tab_maps:
    pick = st.select_slider("Month", options=months, value=months[0])
    i = months.index(pick)
    sc = res["scenes"][i]
    a, b = st.columns(2)
    a.image(P.colourise(pred[i]), use_container_width=True,
            caption=(f"Map · average of {int(n[i])} radar images" if n[i] else f"{pick}: no Sentinel-1 image"))
    if res.get("s1") is not None and n[i]:
        vv = res["s1"][i, 0].astype(float) / 100
        vv[res["s1"][i, 0] == P.S1_NODATA] = np.nan
        img = np.nan_to_num(np.clip((vv + 25) / 25, 0, 1), nan=0.92)
        b.image((img * 255).astype(np.uint8), use_container_width=True,
                caption=f"Radar input (VV) · {sc['date']} · {str(sc['orbit']).lower()} orbit")
    st.markdown(legend, unsafe_allow_html=True)
    with st.expander("All 12 months and animation"):
        st.pyplot(P.grid_figure(pred, months, n))
        gif = P.animation_gif(pred, months, scale=1)
        g1, g2 = st.columns([2, 3])
        g1.image(gif, use_container_width=True)
        g2.download_button("Download animation (GIF)", gif, "lulc_12_months.gif", "image/gif")

with tab_pct:
    st.markdown("Share of the area covered by each class per month (% of pixels with radar data).")
    st.line_chart(pct, color=P.CLASS_COLOURS)
    pc = P.period_change(pred, months)
    if pc is not None:
        st.markdown("**Change over the period** (first 3 vs last 3 months with data, percentage points)")
        st.dataframe(fmt(pc), use_container_width=True)
    with st.expander("Monthly numbers and month-to-month change"):
        st.dataframe(fmt(pct), use_container_width=True)
        st.markdown("**Month-to-month change** (percentage points): this month's % minus the previous month's %.")
        st.dataframe(fmt(mom), use_container_width=True)
        st.markdown(f"**First to last month with data** ({tot.name})")
        st.dataframe(fmt(tot.to_frame("change (pp)").T), use_container_width=True)
        pr = P.class_percentages(res["pred_raw"], months)
        flicker = float(np.nanmean(np.abs(P.month_to_month(pr)).sum(axis=1)))
        st.caption(f"– = no radar data that month or the month before. Seasonal effects (leaf loss, harvest, floods) "
                   f"and classification errors also show up as change; average total month-to-month change here: "
                   f"{flicker:.1f} pp.")
        st.download_button("Download class % (CSV)", pct.round(3).to_csv().encode(), "class_percent.csv", "text/csv")

with tab_ref:
    st.markdown("Two maps that were **never used in training**.")
    if bc is not None:
        st.markdown("**Built-up change compared**")
        st.dataframe(fmt(P.built_change_table(bc, refs)), use_container_width=True)
        if not res.get("obt"):
            st.caption("Open Buildings Temporal does not cover this place (it covers Africa, South & Southeast Asia "
                       "and Latin America).")
        st.caption("Open Buildings maps buildings only (no roads or yards), so its numbers read lower than a "
                   "built-up class. The two references often disagree on one-year change too.")
    if res["ref"] is not None:
        st.markdown(f"**Land cover compared with ESRI {res['ref_year']}** (same 6 classes; ESRI *rangeland* = "
                    "our grass & shrub)")
        a, b = st.columns(2)
        a.image(P.colourise(P.year_majority(pred)), caption="Our model (most frequent class of the year)",
                use_container_width=True)
        b.image(P.colourise(res["ref"]), caption=f"ESRI {res['ref_year']}", use_container_width=True)
        st.markdown(legend, unsafe_allow_html=True)
        st.dataframe(fmt(ref_table), use_container_width=True)
    if bc is not None:
        st.download_button("Download built-up change (CSV)",
                           P.built_change_table(bc, refs).round(3).to_csv().encode(), "built_up_change.csv",
                           "text/csv")
