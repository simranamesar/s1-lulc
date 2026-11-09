"""Landing page of the app: a rotating globe with the places we tested, and two ways in (gallery or live run).

The globe follows the "Globe" component of Magic UI (MIT licence), drawn with the cobe library (MIT licence,
bundled in static/cobe.min.js so the page also works offline), rewritten here as plain HTML + JavaScript because the app is Streamlit, not React.
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


def tested_places() -> list[dict]:
    """[lat, lon] markers: the gallery places (bigger) and the village tests."""
    marks = []
    try:
        from case_studies import CASES
        marks += [{"location": [c[2], c[3]], "size": 0.07} for c in CASES]
    except ImportError:
        pass
    try:
        from village_candidates import CANDIDATES
        marks += [{"location": [c[2], c[3]], "size": 0.05} for c in CANDIDATES]
    except ImportError:
        pass
    return marks


GLOBE_HTML = """
<div id="wrap" style="position:relative;height:466px;overflow:hidden;border:1px solid #e5e7eb;border-radius:12px;
     box-sizing:border-box;background:#fff;display:flex;justify-content:center;
     font-family:'Source Sans Pro',system-ui,sans-serif">
  <div style="position:absolute;top:6px;width:100%;text-align:center;pointer-events:none;z-index:1;
       font-size:clamp(40px,7vw,84px);font-weight:600;line-height:1;letter-spacing:-0.02em;
       background:linear-gradient(to bottom,#111827,rgba(209,213,219,0.8));
       -webkit-background-clip:text;background-clip:text;color:transparent">Watch the world grow</div>
  <canvas id="globe" style="position:absolute;top:96px;width:min(600px,92vw);aspect-ratio:1/1;opacity:0;
          transition:opacity .6s;cursor:grab;contain:layout paint size"></canvas>
  <div style="position:absolute;inset:0;pointer-events:none;
       background:radial-gradient(circle at 50% 200%,rgba(0,0,0,0.12),rgba(255,255,255,0))"></div>
</div>
<script>__COBE__</script>
<script>
const createGlobe = window.createGlobe;
const canvas = document.getElementById("globe");
let phi = 0, width = 0, r = 0, pointer = null, moved = 0;
const resize = () => { width = canvas.offsetWidth; };
window.addEventListener("resize", resize); resize();
const globe = createGlobe(canvas, {
  devicePixelRatio: 2, width: width * 2, height: width * 2, phi: 0, theta: 0.3, dark: 0, diffuse: 0.4,
  mapSamples: 16000, mapBrightness: 1.2, baseColor: [1, 1, 1], markerColor: [196/255, 40/255, 27/255],
  glowColor: [1, 1, 1], markers: __MARKERS__,
  onRender: (state) => {
    if (pointer === null) phi += 0.004;
    state.phi = phi + r; state.width = width * 2; state.height = width * 2;
  },
});
canvas.addEventListener("pointerdown", (e) => { pointer = e.clientX - moved; canvas.style.cursor = "grabbing"; });
const release = () => { pointer = null; canvas.style.cursor = "grab"; };
canvas.addEventListener("pointerup", release); canvas.addEventListener("pointerout", release);
canvas.addEventListener("mousemove", (e) => { if (pointer !== null) { moved = e.clientX - pointer; r = moved / 200; } });
canvas.addEventListener("touchmove", (e) => { if (pointer !== null && e.touches[0]) { moved = e.touches[0].clientX - pointer; r = moved / 200; } });
setTimeout(() => { canvas.style.opacity = "1"; });
</script>
"""


def show(mode_key: str, gallery_label: str, live_label: str) -> None:
    """Draw the landing page. The two buttons switch the sidebar mode radio (session key `mode_key`)."""
    def go(label: str):
        st.session_state[mode_key] = label

    components.html(GLOBE_HTML.replace("__MARKERS__", json.dumps(tested_places()))
                    .replace("__COBE__", (Path(__file__).resolve().parent / "static" / "cobe.min.js").read_text()), height=470)
    st.markdown(
        "<h2 style='text-align:center;margin:0.2rem 0 0'>Sentinel-1 land cover monitor</h2>"
        "<p style='text-align:center;font-size:1.1rem;color:#555;max-width:760px;margin:0.4rem auto 1.2rem'>"
        "Type a location and a start month. From radar images alone, which see through clouds, the app draws "
        "12 monthly land-cover maps and measures how much land was newly built over one year. "
        "It works in cities and in villages. The red dots are the places tested in this project.</p>",
        unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training places", "600", help="2.56 km squares on every inhabited continent, all of 2023")
    c2.metric("Land-cover classes", "6", help="water, trees, grass & shrub, crops, built-up, bare & snow")
    c3.metric("Radar scenes in training", "34,981", help="every Sentinel-1 scene of 2023 over the 600 places")
    c4.metric("Test mIoU", "0.549", help="130 test places in regions never used for training")

    st.write("")
    _, b1, b2, _ = st.columns([1, 2, 2, 1])
    b1.button("Open the gallery (saved examples)", use_container_width=True,
              on_click=go, args=(gallery_label,))
    b2.button("Run a place of your choice", type="primary", use_container_width=True,
              on_click=go, args=(live_label,))
    st.caption("<div style='text-align:center'>Copernicus Sentinel-1 data, processed in Google Earth Engine · "
               "U-Net with a ResNet34 encoder, ensemble of 4 models</div>", unsafe_allow_html=True)
