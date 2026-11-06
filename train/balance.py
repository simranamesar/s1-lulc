"""Measure and control how balanced the training data is.

"Balanced" in this project means three things, checked separately:

  1. Geography   every world region and every settlement type (city / town / village / none)
                 gets a fair share of training samples, so the model does not learn "world = Asia" or
                 "world = big city".
  2. Classes     every land-cover class is seen often enough to be learned. Pixel shares can never be
                 equal (built-up is ~1% of land), so we raise rare classes until each is clearly present.
  3. Diversity   rare classes come from many different tiles, not from one tile repeated a hundred times.

The model never sees a perfectly balanced world; instead, training draws each (tile, month)
sample with a weight:

    class_weight = sum over classes of (share of that class in the sample) x (1 / global share) ^ class_power
    then raked (iterative proportional fitting) so every region AND every settlement type gets an
    equal share of training draws, then clipped to at most --max-ratio x the median.

On top of sampling, the loss gives each class a mild extra weight (1 / share seen)^0.5, which is the only
way to lift a class that always comes mixed with others in the same image (built-up inside farmland).

Validation and test are NOT re-weighted: they keep the real distribution, and we report
metrics per class and per settlement type instead (mIoU already weights every class equally).

Report (no GPU needed):
    python3 balance.py --prep-dir ../data/tiles_v0/prepared --label hyb_s
writes balance_report.md, balance_report.csv and balance_report.png into the prepared folder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from classes import CLASS_NAMES, N_CLASSES  # noqa: E402
S1_NODATA = -32768
CNT_COLS = [f"c{c}" for c in range(N_CLASSES)]

# Thresholds used for the warnings in the report.
MIN_CLASS_SHARE = 0.02      # a class below 2% of training pixels (after balancing) is under-trained
MIN_CLASS_TILES = 10        # a class should appear (>=1% of a tile-month) in at least this many train tiles
MAX_GEO_RATIO = 2.0         # region / settlement shares within 2x of an equal share


# ----------------------------------------------------------------------------- per-sample table

def item_table(prep_dir: Path, tiles: pd.DataFrame, label_key: str, min_label_frac: float = 0.2) -> pd.DataFrame:
    """One row per usable sample: metadata + labelled pixel count per class.

    A sample is one raw Sentinel-1 scene (`sample` = scene index) labelled with its month's map
    (`month`). Older prepared folders without scenes use one monthly composite per month (sample = month).
    A sample is usable when at least `min_label_frac` of its pixels have both radar data and a label.
    """
    rows = []
    for r in tiles.itertuples():
        z = np.load(prep_dir / f"{r.tile_id}.npz")
        s1, n, lbl = z["s1"], z["n"], z[label_key]
        if "month" in z.files:
            pairs = [(k, int(m)) for k, m in enumerate(z["month"])]
        else:
            pairs = [(m, m) for m in range(12) if n[m] > 0]
        for k, m in pairs:
            lab = lbl[m].astype(np.int64)
            lab[(s1[k] == S1_NODATA).any(0)] = -1
            valid = lab >= 0
            if valid.mean() < min_label_frac:
                continue
            bc = np.bincount(lab[valid].ravel(), minlength=N_CLASSES)[:N_CLASSES]
            rows.append({"tile_id": r.tile_id, "sample": k, "month": m, "split": r.split, "region": r.region,
                         "settlement": r.settlement, "biome": r.biome,
                         **{c: int(v) for c, v in zip(CNT_COLS, bc)}})
    return pd.DataFrame(rows)


def class_counts(items: pd.DataFrame) -> np.ndarray:
    return items[CNT_COLS].to_numpy(np.float64)


# ----------------------------------------------------------------------------- weights

def rake(w: np.ndarray, groups: list[np.ndarray], iters: int = 50) -> np.ndarray:
    """Iterative proportional fitting: scale w until every group in every grouping has an equal total."""
    w = w.astype(np.float64).copy()
    for _ in range(iters):
        for g in groups:
            codes, inv = np.unique(g, return_inverse=True)
            tot = np.bincount(inv, weights=w, minlength=len(codes))
            target = w.sum() / len(codes)
            w *= (target / np.maximum(tot, 1e-12))[inv]
    return w


def sample_weights(items: pd.DataFrame, geo_power: float = 1.0, class_power: float = 0.5,
                   max_ratio: float = 10.0) -> np.ndarray:
    """Sampling probability per row of `items` (sums to 1). See module docstring.

    1. class weight per sample = its pixel mix x (1 / global class share) ^ class_power,
       divided by the tile's number of scenes so each place counts equally
    2. rake so every region and every settlement type gets an equal share of draws
       (geo_power 1 = fully equal, 0 = keep the raw geography, in between = partial)
    3. clip extreme weights to max_ratio x median (limits repeating the same few samples), re-rake
    """
    cnt = class_counts(items)
    frac = cnt / np.maximum(cnt.sum(1, keepdims=True), 1)
    share = cnt.sum(0) / max(cnt.sum(), 1)
    cw = np.where(share > 0, (1.0 / np.maximum(share, 1e-12)) ** class_power, 0.0)
    cw = cw / max((share * cw).sum(), 1e-12)          # average pixel keeps weight 1
    w_cls = np.maximum(frac @ cw, 1e-6)
    if "tile_id" in items:
        # every PLACE counts equally, however many scenes it has (Europe: up to ~180 a year, elsewhere ~30)
        w_cls = w_cls / items.groupby("tile_id")["tile_id"].transform("size").to_numpy()

    groups = [items["region"].astype(str).to_numpy(), items["settlement"].astype(str).to_numpy()]
    w_bal = rake(w_cls, groups)
    w = w_cls * (w_bal / w_cls) ** geo_power
    for _ in range(3):
        med = np.median(w)
        w = np.clip(w, med / max_ratio, med * max_ratio)
        if geo_power > 0:
            w = w_cls * (rake(w, groups) / w_cls) ** geo_power
    return w / w.sum()


def effective_class_shares(items: pd.DataFrame, w: np.ndarray | None = None) -> np.ndarray:
    """Share of each class among the pixels the model sees, given sampling weights (None = uniform)."""
    cnt = class_counts(items)
    if w is None:
        w = np.full(len(items), 1.0 / max(len(items), 1))
    tot = cnt.sum(1)
    # per draw, a sample contributes its pixel counts; normalise per sample so big and small months count alike
    per = cnt / np.maximum(tot[:, None], 1)
    eff = (w[:, None] * per).sum(0)
    return eff / max(eff.sum(), 1e-12)


def group_shares(items: pd.DataFrame, col: str, w: np.ndarray | None = None) -> pd.Series:
    w = np.full(len(items), 1.0) if w is None else w
    s = pd.Series(w, index=items.index).groupby(items[col]).sum()
    return s / s.sum()


def effective_sample_size(w: np.ndarray) -> float:
    return float(w.sum() ** 2 / max((w ** 2).sum(), 1e-12))


def loss_class_weights(shares: np.ndarray, power: float = 0.5, cap: float = 10.0) -> np.ndarray:
    """Mild extra weighting in the loss, from the shares the model sees AFTER balanced sampling."""
    w = np.where(shares > 0, (1.0 / np.maximum(shares, 1e-12)) ** power, 0.0)
    w = w / w[shares > 0].mean()
    return np.clip(w, 0, cap)


# ----------------------------------------------------------------------------- report

def build_report(items_all: pd.DataFrame, index: pd.DataFrame, w: np.ndarray, label: str) -> tuple[str, pd.DataFrame, list[str]]:
    tr = items_all[items_all["split"] == "train"].reset_index(drop=True)
    warnings: list[str] = []
    out = [f"# Training data balance report (label: `{label}`)", ""]

    # 1. Geography: tiles
    ok = index[index["status"] == "ok"]
    out += ["## 1. Tiles per settlement type and split", "",
            pd.crosstab(ok["settlement"], ok["split"], margins=True).to_markdown(), ""]
    out += ["## 2. Train tiles per region and settlement type", "",
            pd.crosstab(ok[ok.split == "train"]["region"], ok[ok.split == "train"]["settlement"],
                        margins=True).to_markdown(), ""]
    n_biomes = ok[ok.split == "train"]["biome"].nunique()
    out += [f"Train tiles cover **{n_biomes} of 14 biomes**.", ""]
    if n_biomes < 14:
        warnings.append(f"Only {n_biomes}/14 biomes in the train split.")

    # 2. Geography: sample shares before / after balanced sampling
    geo = []
    for col in ("region", "settlement"):
        raw, bal = group_shares(tr, col), group_shares(tr, col, w)
        equal = 1.0 / len(raw)
        for k in raw.index:
            geo.append({"axis": col, "group": k, "raw_%": raw[k] * 100, "balanced_%": bal[k] * 100,
                        "equal_share_%": equal * 100})
            if bal[k] < equal / MAX_GEO_RATIO or bal[k] > equal * MAX_GEO_RATIO:
                warnings.append(f"{col} '{k}' gets {bal[k] * 100:.1f}% of training draws "
                                f"(equal share {equal * 100:.1f}%) even after balancing.")
    geo = pd.DataFrame(geo)
    out += ["## 3. Share of training draws, uniform vs balanced sampling", "",
            geo.round(1).to_markdown(index=False), ""]

    # 3. Classes
    raw_c = effective_class_shares(tr)
    bal_c = effective_class_shares(tr, w)
    split_c = {s: effective_class_shares(items_all[items_all["split"] == s]) for s in ("val", "test")}
    frac = class_counts(tr) / np.maximum(class_counts(tr).sum(1, keepdims=True), 1)
    tiles_with = [tr.loc[frac[:, c] >= 0.01, "tile_id"].nunique() for c in range(N_CLASSES)]
    lw = loss_class_weights(bal_c)
    with_loss = bal_c * lw / max((bal_c * lw).sum(), 1e-12)
    cls = pd.DataFrame({
        "class": CLASS_NAMES,
        "train_raw_%": raw_c * 100, "train_balanced_%": bal_c * 100, "train_with_loss_%": with_loss * 100,
        "val_%": split_c["val"] * 100, "test_%": split_c["test"] * 100,
        "train_tiles_with_class": tiles_with,
    })
    out += ["## 4. Class shares (pixels the model sees)", "",
            "Raw = uniform sampling. Balanced = expected share per draw with the balanced sampler. "
            "With loss = share of the loss signal once the class weights in the loss are added "
            "(what the model is pushed to learn). Val/test keep the real distribution on purpose.", "",
            cls.round(2).to_markdown(index=False), ""]
    for c, name in enumerate(CLASS_NAMES):
        if with_loss[c] < MIN_CLASS_SHARE:
            warnings.append(f"Class '{name}' gets only {with_loss[c] * 100:.2f}% of the training signal after balancing.")
        if tiles_with[c] < MIN_CLASS_TILES:
            warnings.append(f"Class '{name}' appears in only {tiles_with[c]} train tiles (want >= {MIN_CLASS_TILES}): "
                            "add tiles that contain it rather than up-weighting the same few.")
        if raw_c[c] > 0 and split_c["test"][c] == 0:
            warnings.append(f"Class '{name}' has no test pixels: its test IoU cannot be measured.")

    # 4. Diversity
    ess = effective_sample_size(w)
    out += ["## 5. Diversity", "",
            f"- Train samples (single scenes): **{len(tr)}** from **{tr['tile_id'].nunique()}** tiles",
            f"- Effective sample size with balanced sampling: **{ess:.0f}** "
            f"({ess / max(len(tr), 1) * 100:.0f}% of samples). Below ~50% means a few samples are repeated a lot.",
            ""]
    if ess < 0.5 * len(tr):
        warnings.append(f"Effective sample size is {ess / len(tr) * 100:.0f}% of the samples: weights are very uneven; "
                        "the real fix is more tiles in the thin strata.")

    out += ["## 6. Warnings", ""] + ([f"- {x}" for x in warnings] or ["- None. The training data is balanced on all three checks."]) + [""]
    return "\n".join(out), cls, warnings


def plot_classes(cls: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#1f1f1e", "#6b6a64", "#e4e3dd"
    blue, orange = "#2a78d6", "#eb6834"
    order = cls.sort_values("train_raw_%").reset_index(drop=True)
    y = np.arange(len(order))
    h = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(y + h / 2 + 0.01, order["train_raw_%"], h, color=blue, label="uniform sampling")
    ax.barh(y - h / 2 - 0.01, order["train_with_loss_%"], h, color=orange, label="balanced sampling + loss weights")
    ax.axvline(MIN_CLASS_SHARE * 100, color=muted, linewidth=1, linestyle="--")
    ax.text(MIN_CLASS_SHARE * 100, len(order) - 0.4, f" {MIN_CLASS_SHARE * 100:.0f}% target floor",
            color=muted, fontsize=8, va="bottom")
    ax.set_yticks(y, order["class"].str.replace("_", " "), color=ink, fontsize=9)
    ax.set_xlabel("share of the training signal (%)", color=muted)
    ax.set_title("Balanced sampling raises rare classes; common ones shrink", color=ink, fontsize=11, loc="left")
    ax.grid(axis="x", color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.tick_params(colors=muted, length=0)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prep-dir", required=True)
    ap.add_argument("--label", default="hyb_s", choices=["dw", "hyb", "hyb_s", "hyb_y"])
    ap.add_argument("--geo-power", type=float, default=1.0)
    ap.add_argument("--class-power", type=float, default=0.5)
    ap.add_argument("--max-ratio", type=float, default=10.0)
    ap.add_argument("--min-label-frac", type=float, default=0.2)
    args = ap.parse_args()

    prep = Path(args.prep_dir)
    index = pd.read_csv(prep / "index.csv")
    ok = index[index["status"] == "ok"]
    print(f"Reading {len(ok)} prepared tiles ...")
    items = item_table(prep, ok, args.label, args.min_label_frac)
    tr = items[items["split"] == "train"].reset_index(drop=True)
    w = sample_weights(tr, args.geo_power, args.class_power, args.max_ratio)

    text, cls, warnings = build_report(items, index, w, args.label)
    (prep / "balance_report.md").write_text(text)
    cls.to_csv(prep / "balance_report.csv", index=False)
    plot_classes(cls, prep / "balance_report.png")
    print(text)
    print(f"Wrote {prep}/balance_report.md, .csv, .png")


if __name__ == "__main__":
    main()
