"""Quick look at one exported tile: per-month coverage table + a PNG of S1 VV, DW label and hybrid label.

Usage
    python3 check_tile.py tiles_v0/raw/t00000.tif            # writes t00000_check.png next to it
    python3 check_tile.py tiles_v0/raw/t00000.tif --month 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DW_NAMES = ["water", "trees", "grass", "flooded veg", "crops", "shrub", "built", "bare", "snow/ice"]
DW_COLOURS = ["#419bdf", "#397d49", "#88b053", "#7a87c6", "#e49635",
              "#dfc35a", "#c4281b", "#a59b8f", "#b39fe1"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tif")
    ap.add_argument("--month", type=int, default=0, help="month to plot (default: the one with most S1 scenes)")
    args = ap.parse_args()

    import rasterio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    with rasterio.open(args.tif) as src:
        names = list(src.descriptions)
        data = src.read()
        if not all(names):  # band names not stored in the file: use the export order
            from export_tiles import all_band_names
            names = all_band_names()
        print(f"{Path(args.tif).name}: {src.width}x{src.height} px, {src.count} bands, {src.crs}")
    band = {n: data[i] for i, n in enumerate(names)}

    print(f"\n{'month':>5} {'S1 scenes':>9} {'S1 valid%':>9} {'DW valid%':>9}  top DW classes")
    for m in range(1, 13):
        p = f"m{m:02d}_"
        n = int(band[p + "N"].max())
        s1_ok = (band[p + "VV"] != -32768).mean() * 100
        dw = band[p + "DW"]
        dw_ok = (dw >= 0).mean() * 100
        top = ""
        if dw_ok > 0:
            vals, counts = np.unique(dw[dw >= 0], return_counts=True)
            order = np.argsort(-counts)[:3]
            top = ", ".join(f"{DW_NAMES[vals[i]]} {counts[i] / dw.size * 100:.0f}%" for i in order)
        print(f"{m:>5} {n:>9} {s1_ok:>8.0f}% {dw_ok:>8.0f}%  {top}")
    print(f"\nBuilding roof cover (Open Buildings): {band['BFRAC'].mean():.1f}% of tile")

    month = args.month or int(np.argmax([band[f"m{m:02d}_N"].max() for m in range(1, 13)])) + 1
    p = f"m{month:02d}_"
    vv = np.where(band[p + "VV"] == -32768, np.nan, band[p + "VV"] / 100)
    cmap = ListedColormap(DW_COLOURS)

    fig, ax = plt.subplots(1, 4, figsize=(16, 4.4))
    ax[0].imshow(vv, cmap="gray", vmin=-25, vmax=0); ax[0].set_title(f"S1 VV dB, month {month}")
    for a, key, title in [(ax[1], "DW", "Dynamic World label"), (ax[2], "HYB", "Hybrid label")]:
        lbl = np.ma.masked_less(band[p + key], 0)
        a.imshow(lbl, cmap=cmap, vmin=-0.5, vmax=8.5, interpolation="nearest"); a.set_title(title)
    ax[3].imshow(band["BFRAC"], cmap="Reds", vmin=0, vmax=100); ax[3].set_title("Roof cover %")
    for a in ax:
        a.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in DW_COLOURS]
    fig.legend(handles, DW_NAMES, loc="lower center", ncol=9, fontsize=8, frameon=False)
    out = Path(args.tif).with_name(Path(args.tif).stem + "_check.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
