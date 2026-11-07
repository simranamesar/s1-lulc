"""Writes the five report notebooks (00 read me first + 01-04) (run once; then open them in Jupyter and Run All).

    python3 build_notebooks.py
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent


def nb(cells):
    book = nbf.v4.new_notebook()
    book.cells = [nbf.v4.new_markdown_cell(c[3:]) if c.startswith("md:") else nbf.v4.new_code_cell(c.strip("\n"))
                  for c in cells]
    book.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return book


SETUP = r'''
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display
warnings.filterwarnings("ignore")
ROOT = Path.cwd().resolve().parent                  # the s1-lulc folder
sys.path[:0] = [str(ROOT / "data"), str(ROOT / "train"), str(ROOT / "prototype")]
DATA, PREP, RUNS = ROOT / "data" / "tiles_v0", ROOT / "data" / "tiles_v0" / "prepared", ROOT / "train" / "runs"
from classes import CLASS_NAMES, DISPLAY_NAMES, CLASS_COLOURS, N_CLASSES, BUILT
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
print("project folder:", ROOT)
'''

# --------------------------------------------------------------------------------------------- 01 data
NB1 = [
"""md:# 1 · Data: where, when, what

Sentinel-1 land use / land cover segmentation, **Sentinel-1 as the only model input**. This notebook documents the
dataset: where the 600 places are, how balanced they are, what the radar input and the labels look like, and how
noisy the monthly labels are. All numbers are computed from the files in `data/tiles_v0/` (see `data/README.md`
for how they were downloaded from Google Earth Engine).""",
SETUP,
"""md:## 1.1 The 600 places

Places were drawn at random **stratified by world region × biome × settlement type × land cover** (`data/sample_tiles.py`), not
picked by hand, so the model sees villages in the Sahel as well as German cities. Towns and cities were also drawn
directly from GHSL urban areas, because random points almost never land in a city (Project Log, iteration 18).
Train / val / test are split by **5° geographic blocks**: test places are never next to training places.""",
r'''
tiles = pd.read_csv(DATA / "tiles.csv")
print(f"{len(tiles)} places, 2.56 km x 2.56 km each (256 x 256 pixels of 10 m)")
display(pd.crosstab(tiles.region, tiles.settlement, margins=True))
display(pd.crosstab(tiles.settlement, tiles.split, margins=True))
''',
r'''
colours = {"city": "#c4281b", "town": "#e49635", "village": "#7b4ba3", "none": "#397d49"}
fig, ax = plt.subplots(figsize=(13, 6.2))
for s, c in colours.items():
    d = tiles[tiles.settlement == s]
    ax.scatter(d.lon, d.lat, s=14, c=c, label=f"{s} ({len(d)})", alpha=0.85, edgecolors="none")
for sp, m in [("val", "s"), ("test", "^")]:
    d = tiles[tiles.split == sp]
    ax.scatter(d.lon, d.lat, s=40, facecolors="none", edgecolors="k", linewidths=0.5, marker=m, label=f"{sp} place")
ax.set_xlim(-180, 180); ax.set_ylim(-60, 80); ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
ax.grid(alpha=0.3); ax.legend(loc="lower left", ncol=3, fontsize=9)
ax.set_title("The 600 places (colour = settlement type; outline = validation / test)")
plt.show()
''',
r'''
display(tiles.biome_name.value_counts().rename("places").to_frame())
''',
"""md:## 1.2 Sentinel-1 input: every raw scene of 2023

For every place, **every Sentinel-1 IW acquisition of 2023** was downloaded (no averaging), prepared in Earth Engine
with the recipe below. Every scene is one training sample, labelled with the map of its month.""",
r'''
import s1_composite
print("Recipe:", s1_composite.PREPROCESSING)
stats = json.loads((PREP / "stats.json").read_text())
idx = pd.read_csv(PREP / "index.csv"); ok = idx[idx.status == "ok"]
print(f"prepared places: {len(ok)} of {len(idx)} | scenes: {int(ok.scenes.sum())} | terrain flattened: {stats.get('terrain_flattened')}")
print("normalisation (train scenes, dB): mean", np.round(stats["mean"], 2), "std", np.round(stats["std"], 2))
fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))
ok.boxplot(column="scenes", by="region", ax=ax[0], rot=30); ax[0].set_title("Scenes per place in 2023"); ax[0].set_xlabel("")
ok.groupby("split").scenes.sum().plot.bar(ax=ax[1], rot=0, color="#419bdf"); ax[1].set_title("Scenes per split")
plt.suptitle(""); plt.tight_layout(); plt.show()
print("Europe has up to ~180 scenes per place (many overlapping orbits) vs ~30 elsewhere; training and test scores are")
print("therefore weighted per PLACE, not per scene (Project Log, iteration 17).")
''',
"""md:## 1.3 Classes and their distribution

Dynamic World's 9 classes are merged into **6 classes** that Sentinel-1 can tell apart (`data/classes.py`).""",
r'''
from classes import DW_TO_CLASS
dw_names = ["water", "trees", "grass", "flooded veg.", "crops", "shrub & scrub", "built", "bare", "snow & ice"]
display(pd.DataFrame({"Dynamic World class": dw_names, "our class": [DISPLAY_NAMES[c] for c in DW_TO_CLASS]}))
rows = {}
for sp in ["train", "val", "test"]:
    d = ok[ok.split == sp]
    px = np.array([d[f"px_hyb_s_{c}"].sum() for c in range(N_CLASSES)], float)
    rows[sp] = px / px.sum() * 100
dist = pd.DataFrame(rows, index=DISPLAY_NAMES).round(1)
display(dist.rename(columns=lambda c: f"{c} %"))
dist.plot.bar(figsize=(10, 3.5), rot=0, color=["#419bdf", "#e49635", "#7b4ba3"]); plt.ylabel("% of labelled pixels")
plt.title("Class distribution per split (label hyb_s)"); plt.show()
''',
"""md:## 1.4 Balance of the training data

Uniform sampling would let Europe (27% of scenes) and villages dominate. The **balanced sampler** (`train/balance.py`)
re-weights every scene so each world region and each settlement type gets an equal share of training draws, each place
counts equally regardless of its number of scenes, and rare classes get more weight in the loss.""",
r'''
md = PREP / "balance_report.md"
if md.exists():
    display(Markdown(md.read_text()))
png = PREP / "balance_report.png"
if png.exists():
    display(Image(filename=str(png)))
''',
"""md:## 1.5 How noisy are the monthly labels?

Labels come from Dynamic World (Sentinel-2) **per month**, so the model can learn change, plus Open Buildings roof
cover for small houses (hybrid). A pixel that changes class from one month to the next is mostly label noise
(clouds, leaf loss, harvest), not real land-use change. Seasonal smoothing (`hyb_s`) removes part of it.""",
r'''
flip = ok[["flip_dw", "flip_hyb", "flip_hyb_s"]].mean().rename({"flip_dw": "Dynamic World raw (E0)",
        "flip_hyb": "hybrid raw", "flip_hyb_s": "hybrid smoothed (E1)"})
display((flip * 100).round(1).rename("% pixels changing class month to month").to_frame())
built = ok.groupby("settlement")[["built_px_dw", "built_px_hyb"]].sum()
built["added by buildings %"] = (built.built_px_hyb / built.built_px_dw - 1) * 100
display(built.round(1))
''',
"""md:## 1.6 One example place

Radar input (VV backscatter of the scene used for each month), the raw Dynamic World label and the smoothed hybrid label.""",
r'''
cand = ok[(ok.split == "test") & ok.settlement.isin(["town", "city"])].sort_values("scenes", ascending=False)
tid = (cand if len(cand) else ok).iloc[0].tile_id
z = np.load(PREP / f"{tid}.npz")
row = tiles.set_index("tile_id").loc[tid]
print(f"{tid}: {row.settlement} in {row.region} ({row.lat:.3f}, {row.lon:.3f}), {row.biome_name}; {len(z['s1'])} scenes")
from pipeline import colourise
best, months = z["best"], [f"{m + 1:02d}" for m in range(12)]
fig, axes = plt.subplots(3, 12, figsize=(18, 5.2))
for m in range(12):
    if best[m] >= 0:
        vv = z["s1"][best[m], 0].astype(float) / 100; vv[z["s1"][best[m], 0] == -32768] = np.nan
        axes[0, m].imshow(vv, cmap="gray", vmin=-25, vmax=0)
    axes[1, m].imshow(colourise(z["dw"][m])); axes[2, m].imshow(colourise(z["hyb_s"][m]))
    axes[0, m].set_title(f"month {months[m]}", fontsize=8)
for a in axes.ravel(): a.axis("off")
for r, t in enumerate(["S1 VV (dB)", "Dynamic World", "hybrid smoothed"]):
    axes[r, 0].text(-20, 128, t, rotation=90, va="center", ha="right", fontsize=9)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CLASS_COLOURS]
fig.legend(handles, DISPLAY_NAMES, loc="lower center", ncol=6, frameon=False, fontsize=9)
plt.tight_layout(rect=(0.02, 0.05, 1, 1)); plt.show()
''',
r'''
from s1_composite import lee_filter_db
k = int(best[best >= 0][0])
raw = z["s1"][k]; lee = lee_filter_db(raw)
fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
for a, im, t in [(ax[0], raw, "raw scene (model input)"), (ax[1], lee, "5x5 Lee speckle filter (variant E5)")]:
    v = im[0].astype(float) / 100; v[im[0] == -32768] = np.nan
    a.imshow(v, cmap="gray", vmin=-25, vmax=0); a.set_title(t); a.axis("off")
lbl = z["hyb_s"][int(z["month"][k])]
for c in range(N_CLASSES):
    v = raw[0][(lbl == c) & (raw[0] != -32768)] / 100
    if v.size: ax[2].hist(v, bins=60, range=(-30, 5), histtype="step", color=CLASS_COLOURS[c], label=DISPLAY_NAMES[c], density=True)
ax[2].set_xlabel("VV backscatter (dB)"); ax[2].set_title("VV per class in this scene"); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.show()
print("Water is dark (smooth surface), built-up is bright (double bounce from walls); vegetation classes overlap,")
print("which is why grass, shrub, crops and bare ground are the hardest classes for a radar-only model.")
''',
"""md:## 1.7 The data pipeline step by step

What the data looks like after each script, for the example place above:
**sampler → label download → scene download → label cleaning → prepared file → training sample**.""",
r'''
print("STEP 0 · sample_tiles.py -> tiles.csv (one row per place)")
display(tiles.head(3))
display(tiles[tiles.tile_id == tid].T.rename(columns=lambda c: tid))
''',
r'''
print("STEP 1 · export_tiles.py -> raw/<tile>.tif (monthly labels from Earth Engine, 256 x 256 px, 10 m, UTM)")
tif = DATA / "raw" / f"{tid}.tif"
if tif.exists():
    import rasterio
    with rasterio.open(tif) as src:
        print(f"  {src.count} bands, {src.dtypes[0]}, {src.width} x {src.height} px, CRS {src.crs}, pixel {src.res[0]:.0f} m")
        names = list(src.descriptions)
        if not all(names):          # Earth Engine GeoTIFFs carry no band names: use the fixed export order
            from export_tiles import all_band_names
            names = all_band_names()
        print("  bands of one month:", [n for n in names if n and n.startswith("m07_")], "+ static", names[-1])
        dw7 = src.read(names.index("m07_DW") + 1); hyb7 = src.read(names.index("m07_HYB") + 1)
        bfrac = src.read(names.index("BFRAC") + 1)
    print("  Dynamic World class ids in July:", dict(zip(*np.unique(dw7, return_counts=True))))
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].imshow(dw7, cmap="tab10", vmin=-1, vmax=9); ax[0].set_title("m07_DW: Dynamic World ids 0-8 (-1 = no label)")
    ax[1].imshow(hyb7, cmap="tab10", vmin=-1, vmax=9); ax[1].set_title("m07_HYB: + built where roofs cover >= 15%")
    im = ax[2].imshow(bfrac, cmap="magma", vmin=0, vmax=60); ax[2].set_title("BFRAC: roof cover % (Open Buildings)")
    plt.colorbar(im, ax=ax[2], fraction=0.046)
    for a in ax: a.axis("off")
    plt.tight_layout(); plt.show()
else:
    print("  (raw tif not found)")
''',
r'''
print("STEP 2 · export_scenes.py -> scenes/<tile>.npz (every raw Sentinel-1 acquisition of 2023)")
sc_path = DATA / "scenes" / f"{tid}.npz"
if sc_path.exists():
    sc = np.load(sc_path)
    display(pd.DataFrame({k: [str(sc[k].dtype), sc[k].shape] for k in sc.files}, index=["dtype", "shape"]).T)
    display(pd.DataFrame({"date": sc["date"], "orbit": sc["orbit"], "relative orbit": sc["rel_orbit"],
                          "coverage": np.round(sc["valid"], 3)}).head(10))
    v = sc["s1"][0]
    print("  stored as int16 = dB x 100 (-32768 = no data). Top-left 3 x 3 pixels of the first scene:")
    for b, name in enumerate(["VV", "VH", "VH-VV"]):
        print(f"    {name:6s}", v[b, :3, :3].tolist(), "->", np.round(v[b, :3, :3] / 100, 2).tolist(), "dB")
else:
    print("  (scenes file not found)")
''',
r'''
print("STEP 3 · prepare_data.py: label cleaning, one month and one pixel at every stage")
from prepare_data import read_tile, drop_snow, smooth_labels, annual_label
from classes import from_dw
if tif.exists():
    t = read_tile(tif)
    stages = {"1 Dynamic World (9 ids)": t["dw"]}
    s = drop_snow(t["dw"], int(row.biome)); stages["2 snow dropped outside cold biomes"] = s
    stages["3 merged to 6 classes (E0 label)"] = from_dw(s)
    h = from_dw(drop_snow(t["hyb"], int(row.biome))); stages["4 + building footprints (hybrid)"] = h
    stages["5 seasonally smoothed (E1 label)"] = smooth_labels(h)
    stages["6 annual majority (E1y label)"] = annual_label(h)
    m = 6
    fig, ax = plt.subplots(1, 6, figsize=(19, 3.6))
    for a, (k, v) in zip(ax, stages.items()):
        a.imshow(v[m], cmap="tab10", vmin=-1, vmax=9) if k.startswith(("1", "2")) else a.imshow(colourise(v[m]))
        a.set_title(k, fontsize=9); a.axis("off")
    plt.suptitle(f"month {m + 1}", y=1.02); plt.tight_layout(); plt.show()
    lab6 = stages["3 merged to 6 classes (E0 label)"]
    changes = (np.diff(lab6, axis=0) != 0).sum(0); yy, xx = np.unravel_index(changes.argmax(), changes.shape)
    seq = pd.DataFrame({k: [DISPLAY_NAMES[c] if (c >= 0 and not k.startswith(("1", "2"))) else (dw_names[c] if c >= 0 else "-")
                            for c in v[:, yy, xx]] for k, v in stages.items()}, index=[f"month {i + 1}" for i in range(12)]).T
    print(f"  The most 'flickering' pixel of this place (row {yy}, col {xx}) through the 12 months:")
    display(seq)
''',
r'''
print("STEP 4 · prepared/<tile>.npz: everything the training needs, aligned pixel by pixel")
display(pd.DataFrame({k: [str(z[k].dtype), z[k].shape] for k in z.files}, index=["dtype", "shape"]).T)
print("  scenes per month (n):", z["n"].tolist())
print("  scene used for each month's map (best):", z["best"].tolist())
''',
r'''
print("STEP 5 · balance.py + train.py: from file to training sample")
from balance import item_table, sample_weights
items = item_table(PREP, tiles[tiles.tile_id == tid].assign(biome=row.biome), "hyb_s")
print(f"  {len(items)} training samples from this place (one per scene with enough labelled pixels):")
display(items.head(5))
from train import TileMonths
ds = TileMonths(items.head(4), PREP, "hyb_s", stats["mean"], stats["std"], augment=False)
x, y = ds[0]
raw = z["s1"][items.iloc[0]["sample"]].astype(np.float32) / 100
stat = pd.DataFrame({"raw dB mean": raw.reshape(3, -1).mean(1), "raw dB std": raw.reshape(3, -1).std(1),
                     "normalised mean": x.reshape(3, -1).mean(1).numpy(), "normalised std": x.reshape(3, -1).std(1).numpy()},
                    index=["VV", "VH", "VH-VV"])
print(f"  input tensor x: {tuple(x.shape)} {x.dtype};  label tensor y: {tuple(y.shape)} {y.dtype}, values {sorted(set(y.unique().tolist()))} (-1 = ignored)")
display(stat.round(2))
ds_aug = TileMonths(items.head(1), PREP, "hyb_s", stats["mean"], stats["std"], augment=True)
fig, ax = plt.subplots(2, 4, figsize=(13, 6.4))
np.random.seed(3)
for j in range(4):
    xa, ya = ds_aug[0] if j else (x, y)
    ax[0, j].imshow(xa[0], cmap="gray", vmin=-2.5, vmax=2.5); ax[1, j].imshow(colourise(ya.numpy()))
    ax[0, j].set_title("original" if j == 0 else f"augmented {j} (rotate / flip)", fontsize=9)
for a in ax.ravel(): a.axis("off")
plt.tight_layout(); plt.show()
''',
]

# --------------------------------------------------------------------------------------------- 02 training
NB2 = [
"""md:# 2 · Model, training and evaluation

A **U-Net** (CNN encoder + decoder + 1×1 output convolution) for 6-class segmentation of single raw Sentinel-1 scenes.
All variants were trained with `train/train.py` on GPU 0 of the university server (`train/run_experiments.sh`); this
notebook shows their training logs and curves, evaluates them on the **130 test places** and compares them.
Set `RUN_TRAINING = True` below to train E1 again inside this notebook (about 25 minutes on a GPU).""",
SETUP,
r'''
RUN_TRAINING = False        # True = train E1 again here (writes runs/E1_notebook, ~25 min on a GPU)
if RUN_TRAINING:
    import subprocess
    cmd = [sys.executable, "-u", str(ROOT / "train" / "train.py"), "--prep-dir", str(PREP), "--name", "E1_notebook",
           "--label", "hyb_s", "--runs-dir", str(RUNS)]
    with subprocess.Popen(cmd, cwd=ROOT / "train", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as p:
        for line in p.stdout:
            print(line, end="")
''',
"""md:## 2.1 The network

ResNet34 encoder (ImageNet start, fine-tuned), decoder with 5 upsampling blocks and skip connections, 1×1 convolution
to 6 class scores per pixel (`train/model.py`). Input: 3 channels = VV, VH, VH−VV in dB, normalised.""",
r'''
import torch
from model import build_model
net = build_model("resnet34", "none").eval()
shapes = []
hooks = [m.register_forward_hook(lambda m, i, o, n=n: shapes.append((n, tuple(o.shape[1:]))))
         for n, m in [("encoder stage %d" % k, s) for k, s in enumerate([net.encoder.layer1, net.encoder.layer2,
                       net.encoder.layer3, net.encoder.layer4], 1)] + [("decoder block %d" % k, b) for k, b in enumerate(net.decoder, 1)]
                      + [("1x1 output", net.head)]]
with torch.no_grad():
    net(torch.zeros(1, 3, 256, 256))
for h in hooks: h.remove()
display(pd.DataFrame(shapes, columns=["layer", "output (channels, H, W)"]))
n_enc = sum(p.numel() for p in net.encoder.parameters()); n_all = sum(p.numel() for p in net.parameters())
print(f"parameters: {n_all / 1e6:.1f} M (encoder {n_enc / 1e6:.1f} M, decoder + head {(n_all - n_enc) / 1e6:.1f} M)")
''',
"""md:### What goes in and what comes out

One test scene through the trained E1: the input channels, the 6 class scores (logits) per pixel, the softmax
probabilities, and the final map = the class with the highest probability.""",
r'''
from pipeline import load_model, colourise
model, stats, cfg, device = load_model(RUNS / "E1")
idx = pd.read_csv(PREP / "index.csv"); tt = idx[(idx.status == "ok") & (idx.split == "test")]
zz = np.load(PREP / f"{tt[tt.settlement.isin(['town', 'city'])].iloc[0].tile_id if (tt.settlement.isin(['town', 'city'])).any() else tt.iloc[0].tile_id}.npz")
k = int(zz["best"][zz["best"] >= 0][0]); sc = zz["s1"][k]
mean = np.asarray(stats["mean"], np.float32)[:, None, None]; std = np.asarray(stats["std"], np.float32)[:, None, None]
x = (sc.astype(np.float32) / 100 - mean) / std; x[sc == -32768] = 0
with torch.no_grad():
    logits = model(torch.from_numpy(x)[None].to(device)).float().cpu()[0]
prob = logits.softmax(0).numpy()
print(f"input {tuple(x.shape)} -> logits {tuple(logits.shape)} -> softmax (sums to 1 per pixel) -> argmax map {prob.shape[1:]}")
c0 = prob.shape[1] // 2
print(f"one pixel ({c0}, {c0}):")
display(pd.DataFrame({"logit": [round(float(v), 2) for v in logits[:, c0, c0]],
                      "probability": [round(float(v), 3) for v in prob[:, c0, c0]]}, index=DISPLAY_NAMES).T)
print("-> predicted class:", DISPLAY_NAMES[int(prob[:, c0, c0].argmax())])
fig, ax = plt.subplots(2, 6, figsize=(19, 6.6))
for b, nme in enumerate(["VV", "VH", "VH-VV"]):
    ax[0, b].imshow(x[b], cmap="gray", vmin=-2.5, vmax=2.5); ax[0, b].set_title(f"input: {nme} (normalised)", fontsize=9)
ax[0, 3].imshow(colourise(zz["hyb_s"][int(zz["month"][k])])); ax[0, 3].set_title("label", fontsize=9)
ax[0, 4].imshow(colourise(prob.argmax(0))); ax[0, 4].set_title("prediction (argmax)", fontsize=9)
ax[0, 5].imshow(prob.max(0), cmap="viridis", vmin=0.2, vmax=1); ax[0, 5].set_title("confidence (max probability)", fontsize=9)
for c in range(6):
    ax[1, c].imshow(prob[c], cmap="viridis", vmin=0, vmax=1); ax[1, c].set_title(f"P({DISPLAY_NAMES[c]})", fontsize=9)
for a in ax.ravel(): a.axis("off")
plt.tight_layout(); plt.show()
''',
"""md:## 2.2 The variants

| Run | What changes vs E1 | Question |
| --- | --- | --- |
| E1 | main model: hybrid + smoothed monthly labels, ImageNet fine-tuned, interpolation upsampling | |
| E0 | raw Dynamic World labels | do building footprints + seasonal smoothing help? |
| E1y | annual label (yearly majority) | are stable annual labels better than monthly ones? |
| E2a | ImageNet encoder frozen | transfer learning as feature extraction (lecture 08) |
| E2b | no pretraining (from scratch) | does ImageNet help radar at all? |
| E3 | transposed-convolution upsampling | learned "inverse of convolution" vs interpolation (lecture 07) |
| E5 | 5×5 Lee speckle filter on every scene | does speckle filtering help raw single scenes? |
| E1half | only half of the training places | is the dataset big enough? (learning curve) |
| E1s1, E3s1 | E1 / E3 with another random seed | how large is run-to-run noise? |
| E6 | E3 + Dropout 0.2 + random ±1.5 dB VV/VH offset and 0.5 dB noise + Lovász loss | does regularisation against moisture / overfitting help? |

Common setup: CE + Dice loss with class weights, AdamW (lr 1e-3, cosine schedule), batch 32, bf16, random 90°
rotations and flips, balanced sampler, early stopping after 10 epochs without val mIoU gain.""",
r'''
ORDER = ["E1", "E1s1", "E0", "E1y", "E2a", "E2b", "E3", "E3s1", "E5", "E6", "E1half"]
runs = [r for r in ORDER if (RUNS / r / "test_metrics.json").exists()]
print("finished runs:", runs)
''',
"""md:## 2.3 Training logs (as printed during training)""",
r'''
for r in runs:
    log = ROOT / "train" / f"runs_{r}.log"
    if log.exists():
        text = log.read_text()
        text = text[text.find(f"=== {r}"):] if f"=== {r}" in text else text
        print(text.split("Saved to")[0].strip()[-6000:], "\n" + "-" * 100)
''',
"""md:## 2.4 Training curves""",
r'''
ncol = 6
nrow = -(-len(runs) // ncol)
fig, axes = plt.subplots(2 * nrow, ncol, figsize=(3.2 * ncol, 3.0 * 2 * nrow), squeeze=False)
for a in axes.ravel(): a.axis("off")
for j, r in enumerate(runs):
    h = pd.read_csv(RUNS / r / "history.csv")
    best = h.val_miou.idxmax()
    al, am = axes[2 * (j // ncol), j % ncol], axes[2 * (j // ncol) + 1, j % ncol]
    for a in (al, am): a.axis("on")
    al.plot(h.epoch, h.train_loss, label="train"); al.plot(h.epoch, h.val_loss, label="val")
    am.plot(h.epoch, h.train_miou, label="train"); am.plot(h.epoch, h.val_miou, label="val")
    am.axvline(h.epoch[best], color="grey", ls=":", lw=1)
    al.set_title(r); am.set_ylim(0.3, 0.85); am.set_xlabel("epoch")
    if j % ncol == 0:
        al.set_ylabel("loss"); am.set_ylabel("mIoU")
axes[0, 0].legend(); axes[1, 0].legend()
plt.tight_layout(); plt.show()
print("Dotted line = best validation epoch (checkpoint used for testing). Train keeps improving while val flattens:")
print("the model starts to memorise training places and label noise; early stopping keeps the best epoch.")
''',
"""md:## 2.5 Test results (130 unseen places)

Each run scored against its own training label, with one scene per month (as printed at the end of training).""",
r'''
rows = []
for r in runs:
    m = json.loads((RUNS / r / "test_metrics.json").read_text())
    o, b = m["overall"], m.get("built", {})
    rows.append({"run": r, "OA": o["OA"], "mIoU": o["mIoU"], "macro F1": o["macro_F1"],
                 "built P": b.get("precision"), "built R": b.get("recall"),
                 "village built R": m.get("village_built", {}).get("recall"),
                 "flip rate": m.get("flip_rate", {}).get("prediction"), "best epoch": m.get("best_epoch")})
test = pd.DataFrame(rows).set_index("run")
display(test.round(3))
iou = pd.DataFrame({r: json.loads((RUNS / r / "test_metrics.json").read_text())["per_class_iou"] for r in runs})
iou.index = DISPLAY_NAMES
display(iou.round(3).rename_axis("IoU per class"))
''',
r'''
MAIN = "E1"
m = json.loads((RUNS / MAIN / "test_metrics.json").read_text())
f1 = pd.Series(m["per_class_f1"]); f1.index = DISPLAY_NAMES
cm = pd.read_csv(RUNS / MAIN / "test_confusion.csv", index_col=0).to_numpy(float)
cmn = cm / cm.sum(1, keepdims=True)
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw={"width_ratios": [1.2, 1, 1]})
im = ax[0].imshow(cmn, cmap="Blues", vmin=0, vmax=1)
ax[0].set_xticks(range(6), DISPLAY_NAMES, rotation=35, ha="right"); ax[0].set_yticks(range(6), DISPLAY_NAMES)
for i in range(6):
    for j in range(6):
        ax[0].text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=8, color="w" if cmn[i, j] > .5 else "k")
ax[0].set_xlabel("predicted"); ax[0].set_ylabel("label"); ax[0].set_title(f"{MAIN}: confusion (row-normalised)")
pd.Series({k: v["mIoU"] for k, v in m["per_region"].items()}).sort_values().plot.barh(ax=ax[1], color="#419bdf")
ax[1].set_title("mIoU per world region"); ax[1].set_xlim(0, 0.7)
pd.Series({k: v["mIoU"] for k, v in m["per_settlement"].items()}).plot.bar(ax=ax[2], rot=0, color="#e49635")
ax[2].set_title("mIoU per settlement type"); ax[2].set_ylim(0, 0.7)
plt.tight_layout(); plt.show()
tp = np.diag(cm); prec = tp / np.maximum(cm.sum(0), 1); rec = tp / np.maximum(cm.sum(1), 1)
f1c = 2 * prec * rec / np.maximum(prec + rec, 1e-9); sup = cm.sum(1)
rep_ = pd.DataFrame({"precision": prec, "recall": rec, "f1-score": f1c, "support (pixels)": sup}, index=DISPLAY_NAMES)
rep_.loc["macro avg"] = [prec.mean(), rec.mean(), f1c.mean(), sup.sum()]
rep_.loc["weighted avg"] = [np.average(prec, weights=sup), np.average(rec, weights=sup), np.average(f1c, weights=sup), sup.sum()]
display(Markdown(f"**{MAIN}: classification report** (as `sklearn.metrics.classification_report` in the labs, pixel level, "
                 f"each place weighted equally; accuracy = {tp.sum() / cm.sum():.3f})"))
display(rep_.round(3).astype({"support (pixels)": int}))
''',
"""md:## 2.6 From single scenes to 12 monthly maps

The model reads one raw scene at a time. The prototype needs one map per month; we compared four ways of combining a
month's scene predictions (`train/eval_change.py`), all scored against the **same** labels (hyb_s) so the runs are
comparable. *Month mean* = class probabilities averaged over all scenes of the month.""",
r'''
import subprocess
RUN_MISSING_EVAL = True     # runs eval_change.py (~5 min per run on a GPU) where change_eval.csv is missing
for r in runs:
    if RUN_MISSING_EVAL and not (RUNS / r / "change_eval.csv").exists():
        print(f"evaluating {r} ...")
        subprocess.run([sys.executable, str(ROOT / "train" / "eval_change.py"), "--run", str(RUNS / r),
                        "--prep-dir", str(PREP), "--label", "hyb_s"], cwd=ROOT / "train", check=False)
ce = {r: pd.read_csv(RUNS / r / "change_eval.csv", index_col=0) for r in runs if (RUNS / r / "change_eval.csv").exists()}
if "E1" in ce:
    display(Markdown("**E1: four ways to build the monthly maps**")); display(ce["E1"].round(3))
cols = ["OA", "mIoU", "built_P", "built_R", "flip_rate", "built_false_change", "false_new_ha_km2"]
fair = pd.DataFrame({r: d.loc[d.index.str.startswith("B")].iloc[0] for r, d in ce.items()}).T
if len(fair):
    display(Markdown("**All runs, month-mean maps, same reference labels (hyb_s): the fair comparison**"))
    display(fair[cols].round(3))
''',
r'''
if len(fair):
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
    fair["mIoU"].plot.bar(ax=ax[0], rot=0, color="#419bdf"); ax[0].set_ylim(0.4, 0.6); ax[0].set_title("mIoU (month mean)")
    (fair["flip_rate"] * 100).plot.bar(ax=ax[1], rot=0, color="#e49635"); ax[1].set_title("month-to-month flicker (%)")
    fair["false_new_ha_km2"].plot.bar(ax=ax[2], rot=0, color="#c4281b"); ax[2].set_title("false new built-up (ha/km² per year)")
    plt.tight_layout(); plt.show()
    print("Differences of about +-0.02 mIoU are within run-to-run noise.")
''',
"""md:## 2.7 Is the dataset big enough? (learning curve)""",
r'''
if "E1half" in ce and "E1" in ce:
    lc = pd.DataFrame({"E1half (50% of training places)": ce["E1half"].iloc[2], "E1 (100%)": ce["E1"].iloc[2]}).T
    display(lc[cols].round(3))
elif "E1half" in runs:
    display(test.loc[[r for r in ["E1half", "E1"] if r in runs]])
else:
    print("E1half not trained yet:  python3 train.py --prep-dir ../data/tiles_v0/prepared --name E1half --label hyb_s --train-frac 0.5")
''',
"""md:## 2.8 Final setting for the prototype: ensemble and change rule

No retraining (`train/eval_ensemble.py`): the probabilities of the 4 best models are averaged (ensemble), test-time
augmentation (flipped copies) is tried, and two rules for **new built-up** are compared: *argmax* (built in most end
months, in none of the start months) and *confident* (mean built-up probability ≥ HI at the end and < LO in every
start month).""",
r'''
f = RUNS / "ensemble_eval.csv"
if f.exists():
    ee_ = pd.read_csv(f)
    display(Markdown("**Maps**"))
    display(ee_[ee_.change_rule == "argmax"].set_index("variant")[["OA", "mIoU", "built_P", "built_R", "flip_rate", "built_false_change"]].round(3))
    display(Markdown("**New built-up over 12 months** (false = not new in the labels; the labels themselves are noisy)"))
    display(ee_.set_index(["variant", "change_rule"])[["new_ha_km2", "false_new_ha_km2", "change_P"]].round(3))
    try:
        from pipeline import PROTOTYPE
    except ImportError:   # older prototype/pipeline.py on this machine
        PROTOTYPE = {"runs": ["E3", "E1", "E5", "E2b"], "tta": False, "change_rule": "confident", "hi": 0.5, "lo": 0.2}
    display(Markdown(f"**Prototype uses:** models {' + '.join(PROTOTYPE['runs'])}, TTA {PROTOTYPE['tta']}, "
                     f"{PROTOTYPE['change_rule']} rule (HI {PROTOTYPE['hi']}, LO {PROTOTYPE['lo']}). The ensemble adds about "
                     "+2 mIoU points and less flicker; TTA adds almost nothing at 4x the time; the confident rule removes "
                     "almost all false new built-up but is strict, so the app also shows the argmax result as an upper bound."))
else:
    print("ensemble_eval.csv not found: run train/eval_ensemble.py")
''',
"""md:## 2.9 Round 2: run-to-run noise, regularisation (E6), 3-month smoothing

Three follow-ups after comparing with a fellow student's approach (ideas only): repeat runs with a second seed,
a regularised model (E6), and averaging each month's probabilities with its neighbours ("smooth3").""",
r'''
if all((RUNS / r / "change_eval.csv").exists() for r in ["E1", "E1s1", "E3", "E3s1"]):
    seeds = pd.DataFrame({r: pd.read_csv(RUNS / r / "change_eval.csv", index_col=0).iloc[2] for r in ["E1", "E1s1", "E3", "E3s1", "E6"]
                          if (RUNS / r / "change_eval.csv").exists()}).T
    display(Markdown("**Same settings, different random seed (month-mean maps):** differences = run-to-run noise"))
    display(seeds[["OA", "mIoU", "built_P", "built_R", "flip_rate", "false_new_ha_km2"]].round(3))
for fn, title in [("ensemble_eval_round2.csv", "3-month smoothing (ensemble of E3, E1, E5, E2b)"),
                  ("ensemble_eval_with_E6.csv", "Adding E6 to the ensemble")]:
    f = RUNS / fn
    if f.exists():
        d = pd.read_csv(f)
        display(Markdown(f"**{title}**"))
        display(d[d.change_rule == "argmax"].set_index("variant")[["OA", "mIoU", "built_P", "built_R", "flip_rate", "built_false_change"]].round(3))
        display(d[d.change_rule.isin(["confident 0.5/0.2", "confident 0.5/0.35"])].set_index(["variant", "change_rule"])[["new_ha_km2", "false_new_ha_km2", "change_P"]].round(3))
display(Markdown("**Reading:** a second seed moves mIoU by about 0.01 and the argmax false new built-up by up to 0.17 ha/km², "
                 "so differences of that size between variants are noise. E6 did not help (mIoU 0.501 vs 0.522-0.527, "
                 "bare & snow IoU 0.36 -> 0.26): absolute backscatter level carries class information that the brightness "
                 "shifts destroy. 3-month smoothing halves the flicker (11.6% -> 5.9%) and slightly raises mIoU, so the "
                 "prototype uses ensemble + smooth3; built-up change uses the moderate rule (0.5 / 0.35, chosen on the "
                 "gallery places, notebook 03)."))
''',
"""md:## 2.10 Example predictions on test places

Label, one scene per month, and the month-mean map (what the prototype shows).""",
r'''
from pipeline import load_model, colourise
from eval_change import scene_probs, monthly_maps
model, stats, cfg, device = load_model(RUNS / "E1")
mean = np.asarray(stats["mean"], np.float32)[:, None, None]; std = np.asarray(stats["std"], np.float32)[:, None, None]
idx = pd.read_csv(PREP / "index.csv"); test_tiles = idx[(idx.status == "ok") & (idx.split == "test")]
def well_labelled(df, min_frac=0.5):
    """first place of df that has at least one month with >= min_frac labelled pixels and radar data"""
    for r in df.itertuples():
        zz_ = np.load(PREP / f"{r.tile_id}.npz")
        if any(zz_["n"][m] > 0 and (zz_["hyb_s"][m] >= 0).mean() >= min_frac for m in range(12)):
            return df[df.tile_id == r.tile_id]
    return df.head(1)
picks = pd.concat([well_labelled(test_tiles[test_tiles.settlement == s]) for s in ["city", "town", "village", "none"]])
fig, axes = plt.subplots(len(picks), 4, figsize=(13, 3.3 * len(picks)), squeeze=False)
for i, r in enumerate(picks.itertuples()):
    z = np.load(PREP / f"{r.tile_id}.npz")
    probs = scene_probs(model, z["s1"], mean, std, torch.device(device), bool(cfg.get("lee", False)))
    nod = (z["s1"] == -32768).any(1)
    a = monthly_maps(probs, nod, z["month"].astype(int), z["day"].astype(int), z["valid"], z["orbit"].astype(str), "best", False)
    b = monthly_maps(probs, nod, z["month"].astype(int), z["day"].astype(int), z["valid"], z["orbit"].astype(str), "mean", False)
    labelled = np.array([(z["hyb_s"][mm] >= 0).mean() if z["n"][mm] > 0 else -1 for mm in range(12)])
    m = int(np.argmax(labelled))          # the month with the most labelled pixels (and radar data)
    vv = z["s1"][z["best"][m], 0].astype(float) / 100
    axes[i, 0].imshow(vv, cmap="gray", vmin=-25, vmax=0); axes[i, 1].imshow(colourise(z["hyb_s"][m]))
    axes[i, 2].imshow(colourise(a[m])); axes[i, 3].imshow(colourise(b[m]))
    axes[i, 0].set_ylabel(f"{r.settlement}, {r.region}", fontsize=9)
    for j, t in enumerate(["S1 VV", "label", "one scene", f"mean of {int(z['n'][m])} scenes"]):
        axes[i, j].set_title(f"{t} (month {m + 1})", fontsize=9); axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CLASS_COLOURS]
fig.legend(handles, DISPLAY_NAMES, loc="lower center", ncol=6, frameon=False)
plt.tight_layout(rect=(0, 0.03, 1, 1)); plt.show()
''',
]

# --------------------------------------------------------------------------------------------- 03 case studies
NB3 = [
"""md:# 3 · Prototype case studies

The prototype (`prototype/app.py`) takes a location and a start month, downloads **every raw Sentinel-1 scene** of
12 months (plus the same 3 months one year later), and predicts one map per month with the final setting: an
ensemble of 4 models (E3, E1, E5, E2b), month-mean probabilities and 3-month smoothing.

**Built-up change** compares the first 3 months with the **same 3 calendar months one year later** (same season, so
monsoon or dry-season effects cannot read as construction). Three rules: *argmax* (built-up in most end months, in
none of the start months), *moderate* and *confident* (built-up probability ≥ 0.5 at the end and below 0.35 / 0.2 in
every start month). **Moderate is the headline rule**: in the gallery it reported almost nothing in stable places
and tens of hectares where both references see growth, while confident missed obvious growth and argmax reported
change in stable places. The verdict compares the new built-up with the model's error level measured on 130 unseen test
places. Every place is checked against two products never used in training: **Open Buildings 2.5D Temporal** and
**ESRI 10 m land cover**.

Places computed with `prototype/case_studies.py` (July 2022 – June 2023 + July–Sept 2023, 5.12 km × 5.12 km); several
were chosen *because* they are hard for radar.""",
SETUP,
r'''
import gzip, pickle, base64
from IPython.display import HTML
import pipeline as P
CASES = sorted((ROOT / "prototype" / "case_studies").glob("*.pkl.gz"))
cases = []
for f in CASES:
    with gzip.open(f, "rb") as fh:
        cases.append(pickle.load(fh))
from case_studies import CASES as CASE_LIST, FINDINGS
order = [c[0] for c in CASE_LIST]
cases.sort(key=lambda r: order.index(r["case"]["id"]) if r["case"]["id"] in order else 99)
RULE, MODE = P.PROTOTYPE["change_rule"], P.PROTOTYPE["change_mode"]
print(len(cases), "places:", ", ".join(c["case"]["title"] for c in cases))
print(f"prototype setting: models {'+'.join(P.PROTOTYPE['runs'])}, 3-month smoothing {P.PROTOTYPE['smooth3']}, "
      f"change rule {RULE}, comparison {MODE.replace('_', ' ')}")
''',
r'''
def change(res, mode=MODE, rule=RULE):
    pr, pb, mo, fi, la, used = P.change_inputs(res, mode)
    return P.built_change(pr, mo, pbuilt=pb, rule=rule, first=fi, last=la), used

def built_refs(res, bc):
    refs = []
    if bc is None:
        return refs
    if res.get("obt"):
        o = res["obt"]
        refs.append(P.reference_change(f"Open Buildings Temporal ({o['date_a'][:4]} vs {o['date_b'][:4]})",
                                       o["built_a"], o["built_b"], o["valid"], bc))
    if res.get("esri_ab"):
        y0, y1, ea, eb = res["esri_ab"]
        refs.append(P.reference_change(f"ESRI land cover ({y0} vs {y1})", ea == P.BUILT, eb == P.BUILT,
                                       (ea >= 0) & (eb >= 0), bc))
    return refs

RUN_DIRS = [ROOT / "train" / "runs" / r for r in P.PROTOTYPE["runs"]]
def floor_of(res, rule=RULE):
    """Error level of this rule, measured on the test places (computed now from train/runs if available,
    else the value stored when the place was computed)."""
    if all(d.exists() for d in RUN_DIRS):
        return P.noise_floor(RUN_DIRS, rule, P.PROTOTYPE["tta"], smooth3=P.PROTOTYPE["smooth3"])
    c = res["case"]
    return c.get("floors", {}).get(rule, c.get("floor", P.MIN_FLOOR_HA_KM2))

summary = []
for res in cases:
    c = res["case"]; row = {"place": c["title"], "type": c["kind"], "scenes": int(res["n"].sum())}
    bc, used = change(res)
    if bc is None:
        row["verdict"] = "not enough Sentinel-1 data"; summary.append(row); continue
    v, why, exp = P.change_verdict(bc, floor_of(res))
    row.update({"compares": used.replace("_", " "), "built-up start %": bc["before_pct"], "end %": bc["after_pct"],
                f"new ha ({RULE})": bc["gain_ha"], "error level ha": exp, "verdict": v})
    for rule in ("argmax", "moderate"):
        b2, _ = change(res, rule=rule)
        row[f"new ha ({rule})"] = b2["gain_ha"] if b2 else np.nan
    for r in built_refs(res, bc):
        key = "OBT" if r["source"].startswith("Open") else "ESRI"
        row[f"{key} new ha"] = r["new built-up (ha)"]
    summary.append(row)
display(pd.DataFrame(summary).set_index("place").round(1))
''',
"""md:## Each place in detail

For every place: the 12 monthly radar images, maps and built-up probability (how the whole area changes through the
year), an animation of the monthly maps, the built-up change map and the class shares per month.""",
r'''
def month_strip(res):
    months, n = res["months"], res["n"]
    pb = res.get("pbuilt")
    rows = 3 if pb is not None else 2
    fig, ax = plt.subplots(rows, 12, figsize=(24, 2.2 * rows + 0.4), squeeze=False)
    for m in range(12):
        if res.get("s1") is not None:
            vv = res["s1"][m, 0].astype(float) / 100; vv[res["s1"][m, 0] == P.S1_NODATA] = np.nan
            ax[0, m].imshow(vv, cmap="gray", vmin=-25, vmax=0)
        ax[1, m].imshow(P.colourise(res["pred"][m]), interpolation="nearest")
        if pb is not None:
            b = pb[m].astype(float); b = b / 255 if b.max() > 1.5 else b
            ax[2, m].imshow(np.where(res["pred"][m] >= 0, b, np.nan), cmap="magma", vmin=0, vmax=1)
        ax[0, m].set_title(f"{months[m]}  ({int(n[m])} scenes)", fontsize=8)
    for a, t in zip(ax[:, 0], ["radar VV", "map", "P(built-up)"]):
        a.set_ylabel(t, fontsize=9)
    for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
    plt.tight_layout(); plt.show()

for res in cases:
    c = res["case"]; months = res["months"]
    bc, used = change(res)
    found = FINDINGS.get(c["id"], "")
    display(Markdown(f"### {c['title']}  ·  *{c['kind']}*\n\n**What to look for:** {c['note']}"
                     + (f"\n\n**What we found:** {found}" if found else "")))
    if bc is None:
        display(Markdown("**No result:** too few months with Sentinel-1 data here.")); continue
    v, why, _ = P.change_verdict(bc, floor_of(res))
    display(Markdown(f"**Verdict ({RULE} rule, {used.replace('_', ' ')}): {v}.** {why}"))
    month_strip(res)
    gif = P.animation_gif(res["pred"], months)
    display(HTML(f'<img src="data:image/gif;base64,{base64.b64encode(gif).decode()}" width="360"> '
                 f'<span style="font-size:12px">monthly maps as an animation</span>'))
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw={"width_ratios": [1, 1, 1.5]})
    ax[0].imshow(P.colourise(P.year_majority(res["pred"]))); ax[0].set_title("year majority (our model)")
    ax[1].imshow(P.change_rgb(bc)); ax[1].set_title(f"built-up {bc['first'][0]}… vs …{bc['last'][-1]}\nmagenta new, cyan lost", fontsize=10)
    pct = P.class_percentages(res["pred"], months)
    for k, col in enumerate(pct.columns):
        ax[2].plot(range(12), pct[col], color=CLASS_COLOURS[k], label=DISPLAY_NAMES[k], marker="o", ms=3)
    ax[2].set_xticks(range(12), [m[2:] for m in months], rotation=45, fontsize=8); ax[2].set_ylabel("% of area")
    ax[2].legend(fontsize=7, ncol=2, frameon=False); ax[2].set_title("class % per month")
    for a in ax[:2]: a.axis("off")
    plt.tight_layout(); plt.show()
    display(P.built_change_table(bc, built_refs(res, bc)).round(2))
''',
]


# --------------------------------------------------------------------------------------------- 04 training progress
NB4 = [
"""md:# 4 · Training, step by step

Everything `train/train.py` printed or saved while the networks learned, turned into tables and pictures:

* **4.1–4.2** the setup each run printed and the epoch-by-epoch log of one run as a table,
* **4.3–4.4** loss, mIoU, accuracy and learning rate over the epochs, for one run and for all 11 runs,
* **4.5** test results of all runs as heatmaps (per class, per world region, per settlement type),
* **4.6** **watching the model learn**: a run that saved its predictions for 6 fixed validation scenes after
  every epoch (from the untrained network, epoch 0, to the end), the loss after every batch, and per-class IoU
  per epoch: how the output maps change while the network trains.

Sources: `train/runs_<run>.log` (printed text), `train/runs/<run>/history.csv` (one row per epoch),
`steps.csv` (one row per batch) and `snapshots/` (predictions per epoch).""",
SETUP,
r'''
import re, io, base64
from IPython.display import HTML
from pipeline import colourise
ORDER = ["E1", "E1s1", "E0", "E1y", "E2a", "E2b", "E3", "E3s1", "E3v", "E5", "E6", "E1half"]
runs = [r for r in ORDER if (RUNS / r / "history.csv").exists()]
runs += sorted(d.name for d in RUNS.iterdir() if (d / "history.csv").exists() and d.name not in runs)
hist = {r: pd.read_csv(RUNS / r / "history.csv") for r in runs}
WATCH = next((r for r in ["E3v"] + runs if (RUNS / r / "snapshots" / "examples.npz").exists()), None)
print("runs with a training history:", ", ".join(runs))
print("run with prediction snapshots (section 4.6):", WATCH or "none yet")
def log_text(r):
    f = ROOT / "train" / f"runs_{r}.log"
    return f.read_text(errors="ignore") if f.exists() else ""
''',
"""md:## 4.1 What each run printed before training

At the start, `train.py` prints the number of training and validation scenes, how strongly the balanced sampler
reweights them (*effective sample size*: how many independent scenes the weighted draw is worth), the network size,
and the class weights of the loss. The table collects these lines from the log files; the chart shows how the
balanced sampler changes the class mix the network sees.""",
r'''
def grab(pat, t, f=float):
    m = re.search(pat, t)
    return f(m[1]) if m else np.nan
rows = []
for r in runs:
    t, h = log_text(r), hist[r]
    cfg = json.loads((RUNS / r / "config.json").read_text()) if (RUNS / r / "config.json").exists() else {}
    best = h.val_miou.idxmax()
    rows.append({"run": r, "label": cfg.get("label"), "encoder": "frozen" if cfg.get("freeze_encoder") else cfg.get("encoder_weights"),
                 "upsampling": cfg.get("upsample"), "train scenes": grab(r"train (\d+), val", t, int),
                 "val scenes": grab(r"val (\d+);", t, int), "effective sample size": grab(r"effective sample size (\d+)", t, int),
                 "parameters (M)": grab(r"([\d.]+)M parameters", t), "trainable (M)": grab(r"\(([\d.]+)M trainable\)", t),
                 "epochs run": len(h), "best epoch": int(h.epoch[best]), "best val mIoU": h.val_miou[best],
                 "early stopped": "yes" if "Early stopping" in t else ("no" if t else "?"),
                 "time (min)": h.seconds.sum() / 60, "s / epoch": h.seconds.mean()})
setup = pd.DataFrame(rows).set_index("run")
display(setup.round(3))
w = {}
for r in runs:
    m = re.search(r"Class weights: (.*)", log_text(r))
    if m:
        w[r] = {k: float(v) for k, v in (x.rsplit(" ", 1) for x in m[1].split(", "))}
if w:
    display(Markdown("**Class weights of the loss** (printed as `Class weights: ...`; larger = rarer class in what the sampler draws)"))
    display(pd.DataFrame(w).T.round(2))
r0 = "E1" if "E1" in runs else runs[0]
f = RUNS / r0 / "train_class_shares.csv"
if f.exists():
    cs = pd.read_csv(f).set_index("class")
    ax = cs.plot.barh(figsize=(8, 3.6), color=["#bbbbbb", "#419bdf"])
    ax.set_xlabel("% of training pixels"); ax.set_title(f"{r0}: class mix, plain draw vs balanced sampler")
    ax.legend(["every scene equally likely", "balanced sampler (what the network sees)"], frameon=False)
    plt.tight_layout(); plt.show()
''',
"""md:## 4.2 The epoch log as a table

During training every epoch printed one line such as
`ep  12  loss 0.981/1.204  mIoU 0.612/0.531  OA 0.790/0.683  51s  *best`.
Here are those lines of one run (the prototype's main model, E3) as a table. Colours: greener = better val mIoU,
bars = loss; ★ marks a new best validation mIoU, the checkpoint kept for testing.""",
r'''
RUN = "E3" if "E3" in runs else runs[0]
h = hist[RUN]
t = log_text(RUN)
lines = [l for l in t.splitlines() if l.startswith("ep ")]
if lines:
    print(f"As printed ({RUN}, first and last 3 epochs):")
    print("\n".join(lines[:3] + ["   ..."] + lines[-3:]), "\n")
cols = [c for c in ["epoch", "train_loss", "val_loss", "train_miou", "val_miou", "train_oa", "val_oa", "lr", "seconds"] if c in h]
tab = h[cols].copy()
tab["new best"] = np.where(h.val_miou >= h.val_miou.cummax(), "★", "")
try:
    sty = (tab.style.hide(axis="index")
           .format({c: "{:.3f}" for c in cols if c not in ("epoch", "lr", "seconds")} | {"lr": "{:.1e}", "seconds": "{:.0f}"})
           .background_gradient(subset=["val_miou"], cmap="Greens")
           .bar(subset=["train_loss", "val_loss"], color="#f4a582"))
    display(sty)
except Exception:          # pandas Styler needs jinja2
    display(tab.round(3))
''',
"""md:## 4.3 One run: loss, mIoU, accuracy and learning rate per epoch

* **Loss** (cross-entropy + Dice) should fall for both train and validation; when validation stops falling while
  training keeps falling, the network starts memorising the training places.
* **mIoU**: the shaded gap between train and validation is the generalisation gap.
* **Accuracy** (OA): share of correctly classified pixels.
* **Learning rate**: one warm-up epoch, then a cosine decay (`train.py`), which is why the curves calm down at the end.
* Dotted line = best epoch (checkpoint kept); the grey band = the 10 epochs of patience before early stopping.""",
r'''
def dashboard(r):
    h = hist[r]; best = int(h.epoch[h.val_miou.idxmax()])
    fig, ax = plt.subplots(1, 4, figsize=(18, 3.6))
    ax[0].plot(h.epoch, h.train_loss, "o-", ms=3, label="train"); ax[0].plot(h.epoch, h.val_loss, "o-", ms=3, label="val")
    ax[0].set_title("loss (CE + Dice)")
    ax[1].plot(h.epoch, h.train_miou, "o-", ms=3, label="train"); ax[1].plot(h.epoch, h.val_miou, "o-", ms=3, label="val")
    ax[1].fill_between(h.epoch, h.val_miou, h.train_miou, color="grey", alpha=0.15, label="gap")
    ax[1].set_title("mIoU")
    if "train_oa" in h: ax[2].plot(h.epoch, h.train_oa, "o-", ms=3, label="train")
    ax[2].plot(h.epoch, h.val_oa, "o-", ms=3, label="val", color="C1"); ax[2].set_title("overall accuracy")
    ax[3].plot(h.epoch, h.lr, color="k"); ax[3].set_yscale("log"); ax[3].set_title("learning rate")
    for a in ax:
        a.axvline(best, color="grey", ls=":", lw=1.2); a.set_xlabel("epoch")
        if len(h) > best: a.axvspan(best, h.epoch.max(), color="grey", alpha=0.07)
    for a in ax[:3]: a.legend(frameon=False, fontsize=8)
    fig.suptitle(f"{r}: best val mIoU {h.val_miou.max():.3f} at epoch {best} of {len(h)}  "
                 f"({h.seconds.sum() / 60:.0f} min)", y=1.03)
    plt.tight_layout(); plt.show()
dashboard(RUN)
''',
"""md:Every other run the same way (scroll):""",
r'''
for r in runs:
    if r != RUN:
        dashboard(r)
''',
"""md:## 4.4 All runs on one chart

The same curves overlaid: how fast each variant learns and where it levels off. Frozen encoder (E2a) and no
pretraining (E2b) start lower; E1half has half the training places.""",
r'''
fig, ax = plt.subplots(1, 3, figsize=(18, 4.4))
cmap = plt.get_cmap("tab20")
for i, r in enumerate(runs):
    h = hist[r]; c = cmap(i % 20)
    ax[0].plot(h.epoch, h.val_miou, color=c, label=r); ax[1].plot(h.epoch, h.val_loss, color=c, label=r)
    ax[2].scatter(h.seconds.sum() / 60, h.val_miou.max(), color=c, s=40)
    ax[2].annotate(r, (h.seconds.sum() / 60, h.val_miou.max()), xytext=(4, 3), textcoords="offset points", fontsize=8)
ax[0].set_title("validation mIoU per epoch"); ax[1].set_title("validation loss per epoch")
for a in ax[:2]: a.set_xlabel("epoch")
ax[0].legend(ncol=2, fontsize=7, frameon=False)
ax[2].set_xlabel("training time (min)"); ax[2].set_ylabel("best validation mIoU"); ax[2].set_title("result vs cost")
plt.tight_layout(); plt.show()
''',
"""md:## 4.5 Test results of all runs as heatmaps

From `test_metrics.json` (130 unseen places, one scene per month, each run against its own label). Rows = runs,
columns = classes / world regions / settlement types; greener = better. Notebook 02 has the fair comparison
on the same labels.""",
r'''
def heat(df, title, ax, vmin=0, vmax=0.8):
    im = ax.imshow(df.to_numpy(float), cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(df.shape[1]), df.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(df.shape[0]), df.index, fontsize=8)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            v = df.iat[i, j]
            if v == v: ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title(title)
tm = {r: json.loads((RUNS / r / "test_metrics.json").read_text()) for r in runs if (RUNS / r / "test_metrics.json").exists()}
if tm:
    cls = pd.DataFrame({r: m["per_class_iou"] for r, m in tm.items()}).T; cls.columns = DISPLAY_NAMES
    cls["mIoU"] = [m["overall"]["mIoU"] for m in tm.values()]
    reg = pd.DataFrame({r: {k: v["mIoU"] for k, v in m["per_region"].items()} for r, m in tm.items()}).T
    st_ = pd.DataFrame({r: {k: v["mIoU"] for k, v in m["per_settlement"].items()} for r, m in tm.items()}).T
    fig, ax = plt.subplots(1, 3, figsize=(20, 0.42 * len(tm) + 2), gridspec_kw={"width_ratios": [7, len(reg.columns), 4]})
    heat(cls, "IoU per class (test)", ax[0]); heat(reg, "mIoU per world region", ax[1]); heat(st_, "mIoU per settlement type", ax[2])
    plt.tight_layout(); plt.show()
''',
"""md:## 4.6 Watching the model learn

The run below (`E3v`: the E3 setting with another seed) saved, after every epoch, the predicted map, the confidence
and the built-up probability for 6 fixed validation scenes (`snapshots/`), and the training loss of every batch
(`steps.csv`). Epoch 0 is the network before any training: its decoder is random, so its maps are meaningless (often one class
everywhere).""",
r'''
if WATCH is None:
    print("No run with snapshots yet. On the server:  bash train/run_viz.sh   (about 40 min), then rerun this notebook.")
else:
    W = RUNS / WATCH; hw = hist[WATCH]
    ex = np.load(W / "snapshots" / "examples.npz")
    eps = sorted(int(f.stem[2:]) for f in (W / "snapshots").glob("ep*.npz"))
    snap = {e: np.load(W / "snapshots" / f"ep{e:03d}.npz") for e in eps}
    lab = ex["label"].astype(int); okl = lab >= 0
    acc = pd.Series({e: (snap[e]["pred"][okl] == lab[okl]).mean() for e in eps})
    best_ep = int(hw.epoch[hw.val_miou.idxmax()])
    print(f"{WATCH}: {len(eps) - 1} epochs, snapshots of {len(lab)} validation scenes:")
    display(pd.DataFrame({"place": [f"{s}, {r}" for s, r in zip(ex["settlement"], ex["region"])],
                          "tile": ex["tile_id"], "month of the label": ex["month"] + 1}))
''',
"""md:### Loss after every batch

Each grey point is the loss of one batch of 32 scenes; the line is a moving average over 200 batches. Batches differ
(a city scene is harder than a desert), so single values jump; the trend is what matters. Red = learning rate.""",
r'''
if WATCH and (W / "steps.csv").exists():
    stp = pd.read_csv(W / "steps.csv")
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(stp.step, stp.loss, ".", color="grey", ms=1.5, alpha=0.3)
    ax.plot(stp.step, stp.loss.rolling(200, min_periods=20).mean(), color="C0", lw=2, label="moving average (200 batches)")
    for e, g in stp.groupby("epoch"):
        ax.axvline(g.step.max(), color="k", lw=0.3, alpha=0.3)
    ax.set_xlabel("training step (batch)"); ax.set_ylabel("training loss"); ax.set_ylim(top=stp.loss.quantile(0.995))
    ax2 = ax.twinx(); ax2.plot(stp.step, stp.lr, color="C3", lw=1); ax2.set_ylabel("learning rate", color="C3")
    sec = ax.secondary_xaxis("top", functions=(lambda s: s / stp.groupby("epoch").step.max().iloc[0],
                                               lambda e: e * stp.groupby("epoch").step.max().iloc[0]))
    sec.set_xlabel("epoch"); ax.legend(loc="upper right", frameon=False)
    plt.tight_layout(); plt.show()
    print(f"{len(stp)} batches; loss first 100 batches {stp.loss[:100].mean():.2f} -> last 100 {stp.loss[-100:].mean():.2f}")
''',
"""md:### Which classes are learned first

Validation IoU of every class after every epoch. Classes with a clear radar signature (water, built-up) are learned
in the first epochs; classes that look alike to radar (grass & shrub vs crops vs bare) need longer and stay lower.""",
r'''
if WATCH:
    ic = [c for c in hw.columns if c.startswith("val_iou_")]
    if ic:
        fig, ax = plt.subplots(1, 2, figsize=(17, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
        for k, c in enumerate(ic):
            ax[0].plot(hw.epoch, hw[c], "o-", ms=3, color=CLASS_COLOURS[k], label=DISPLAY_NAMES[k])
        ax[0].plot(hw.epoch, hw.val_miou, "k--", lw=2, label="mIoU"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("validation IoU")
        ax[0].legend(ncol=2, fontsize=8, frameon=False); ax[0].set_title("per-class IoU per epoch")
        im = ax[1].imshow(hw[ic].to_numpy().T, aspect="auto", cmap="viridis", vmin=0, vmax=0.9,
                          extent=(hw.epoch.min() - .5, hw.epoch.max() + .5, len(ic) - .5, -.5))
        ax[1].set_yticks(range(len(ic)), DISPLAY_NAMES); ax[1].set_xlabel("epoch"); ax[1].set_title("same as a heatmap")
        plt.colorbar(im, ax=ax[1], label="IoU"); plt.tight_layout(); plt.show()
''',
"""md:### How the predicted maps change during training

Rows = the 6 fixed validation scenes, columns = the radar input, the label, and the network's prediction after
selected epochs. The number above each map is its pixel accuracy against the label.""",
r'''
if WATCH:
    show_eps = [e for e in dict.fromkeys([0, 1, 2, 3, 5, 10, 20, best_ep, eps[-1]]) if e in snap]
    n = len(lab)
    fig, axes = plt.subplots(n, len(show_eps) + 2, figsize=(2.2 * (len(show_eps) + 2), 2.3 * n), squeeze=False)
    for i in range(n):
        vv = ex["s1"][i, 0].astype(float) / 100; vv[ex["s1"][i, 0] == -32768] = np.nan
        axes[i, 0].imshow(vv, cmap="gray", vmin=-25, vmax=0); axes[i, 1].imshow(colourise(lab[i]))
        axes[i, 0].set_ylabel(f"{ex['settlement'][i]}\n{ex['region'][i]}", fontsize=8)
        for j, e in enumerate(show_eps):
            pr = snap[e]["pred"][i].astype(int); ok = lab[i] >= 0
            axes[i, j + 2].imshow(colourise(np.where(lab[i] >= -1, pr, -1)))
            axes[i, j + 2].set_title(f"{'best ' if e == best_ep else ''}epoch {e}\nacc {(pr[ok] == lab[i][ok]).mean():.2f}", fontsize=8)
    axes[0, 0].set_title("radar VV (dB)", fontsize=8); axes[0, 1].set_title("label", fontsize=8)
    for a in axes.ravel(): a.set_xticks([]); a.set_yticks([])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CLASS_COLOURS]
    fig.legend(handles, DISPLAY_NAMES, loc="lower center", ncol=6, frameon=False)
    plt.tight_layout(rect=(0, 0.03, 1, 1)); plt.show()
''',
"""md:### How much the maps still change from one epoch to the next

Left: pixel accuracy on the 6 scenes per epoch. Right: share of pixels whose predicted class changed since the
previous epoch. Large at first (the network is still finding the classes), small at the end (the learning rate is
low and the maps only get refined): this is what convergence looks like in the output images.""",
r'''
if WATCH:
    chg = pd.Series({e: (snap[e]["pred"] != snap[p]["pred"]).mean() for p, e in zip(eps[:-1], eps[1:])})
    fig, ax = plt.subplots(1, 2, figsize=(15, 3.6))
    acc.plot(ax=ax[0], marker="o", ms=3); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("pixel accuracy (6 scenes)")
    ax[0].axvline(best_ep, color="grey", ls=":"); ax[0].set_ylim(0, 1)
    (chg * 100).plot.bar(ax=ax[1], color="#e49635", width=0.8); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("% pixels changed")
    ax[1].set_xticks(range(0, len(chg), max(1, len(chg) // 15)), chg.index[::max(1, len(chg) // 15)], rotation=0)
    plt.tight_layout(); plt.show()
    print(f"epoch 0 -> 1: {chg.iloc[0] * 100:.0f}% of pixels changed; last epoch: {chg.iloc[-1] * 100:.1f}%")
''',
"""md:### Confidence and built-up probability while learning

For the first (city) scene: the network's confidence (highest class probability; dark = unsure) and its built-up
probability after the same epochs. Early on everything is uncertain; later the network is confident inside
homogeneous areas and unsure along class borders, which is where most errors are.""",
r'''
if WATCH:
    i = 0
    fig, axes = plt.subplots(2, len(show_eps) + 1, figsize=(2.2 * (len(show_eps) + 1), 4.8), squeeze=False)
    axes[0, 0].imshow(colourise(lab[i])); axes[0, 0].set_title("label", fontsize=8)
    axes[1, 0].imshow(lab[i] == BUILT, cmap="magma", vmin=0, vmax=1); axes[1, 0].set_title("label built-up", fontsize=8)
    for j, e in enumerate(show_eps):
        axes[0, j + 1].imshow(snap[e]["conf"][i] / 255, cmap="viridis", vmin=0.2, vmax=1)
        axes[1, j + 1].imshow(snap[e]["p_built"][i] / 255, cmap="magma", vmin=0, vmax=1)
        axes[0, j + 1].set_title(f"confidence, ep {e}", fontsize=8); axes[1, j + 1].set_title(f"P(built), ep {e}", fontsize=8)
    for a in axes.ravel(): a.axis("off")
    plt.tight_layout(); plt.show()
''',
"""md:### Animation: all 6 scenes through every epoch

Top row: labels; bottom row: the prediction after the epoch shown in the corner (one frame per epoch).""",
r'''
if WATCH:
    from PIL import Image as PILImage, ImageDraw
    top = np.concatenate([colourise(l) for l in lab], axis=1)
    frames = []
    for e in eps:
        bot = np.concatenate([colourise(p.astype(int)) for p in snap[e]["pred"]], axis=1)
        im = PILImage.fromarray(np.concatenate([top, np.full((6, top.shape[1], 3), 255, np.uint8), bot]))
        im = im.resize((im.width // 2, im.height // 2), PILImage.NEAREST)
        d = ImageDraw.Draw(im); d.rectangle([0, im.height // 2, 120, im.height // 2 + 18], fill=(255, 255, 255))
        d.text((5, im.height // 2 + 3), f"epoch {e}  acc {acc[e]:.2f}", fill=(0, 0, 0))
        frames.append(im)
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], duration=[1200] + [450] * (len(frames) - 2) + [2500], loop=0)
    (W / "snapshots" / "learning.gif").write_bytes(buf.getvalue())
    display(HTML(f'<img src="data:image/gif;base64,{base64.b64encode(buf.getvalue()).decode()}" style="max-width:100%">'))
    print(f"saved {W / 'snapshots' / 'learning.gif'}")
''',
]


def with_explanations(key, cells):
    """Insert the long "how and why" cells of explanations.py next to the cells they belong to."""
    import explanations as X
    out = list(cells)
    for nb_, anchor, pos, md in X.E:
        if nb_ != key:
            continue
        hits = [i for i, c in enumerate(out) if anchor in c and not c.startswith("md:### How")]
        if not hits:
            raise SystemExit(f"explanations.py: anchor not found in {key}: {anchor!r}")
        i = hits[0]
        out.insert(i + 1 if pos == "after" else i, md)
    return out


if __name__ == "__main__":
    import explanations as X
    for name, key, cells in [("00_read_me_first.ipynb", "NB0", X.NB0), ("01_data.ipynb", "NB1", NB1),
                             ("02_training_evaluation.ipynb", "NB2", NB2), ("03_case_studies.ipynb", "NB3", NB3),
                             ("04_training_progress.ipynb", "NB4", NB4)]:
        cells = with_explanations(key, cells)
        nbf.write(nb(cells), HERE / name)
        print("wrote", HERE / name)
