"""Writes the three report notebooks (run once; then open them in Jupyter and Run All).

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
Set `RUN_TRAINING = True` below to train E1 again inside this notebook (about 30 minutes on a GPU).""",
SETUP,
r'''
RUN_TRAINING = False        # True = train E1 again here (writes runs/E1_notebook, ~30 min on a GPU)
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
print(f"one pixel ({c0}, {c0}): logits", np.round(logits[:, c0, c0].numpy(), 2).tolist(),
      "-> probabilities", np.round(prob[:, c0, c0], 3).tolist(), "->", DISPLAY_NAMES[int(prob[:, c0, c0].argmax())])
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

Common setup: CE + Dice loss with class weights, AdamW (lr 1e-3, cosine schedule), batch 32, bf16, random 90°
rotations and flips, balanced sampler, early stopping after 10 epochs without val mIoU gain.""",
r'''
ORDER = ["E1", "E0", "E1y", "E2a", "E2b", "E3", "E5", "E1half"]
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
fig, axes = plt.subplots(2, len(runs), figsize=(3.3 * len(runs), 6.2), squeeze=False)
for j, r in enumerate(runs):
    h = pd.read_csv(RUNS / r / "history.csv")
    best = h.val_miou.idxmax()
    axes[0, j].plot(h.epoch, h.train_loss, label="train"); axes[0, j].plot(h.epoch, h.val_loss, label="val")
    axes[1, j].plot(h.epoch, h.train_miou, label="train"); axes[1, j].plot(h.epoch, h.val_miou, label="val")
    axes[1, j].axvline(h.epoch[best], color="grey", ls=":", lw=1)
    axes[0, j].set_title(r); axes[1, j].set_ylim(0.3, 0.75); axes[1, j].set_xlabel("epoch")
axes[0, 0].set_ylabel("loss (CE + Dice)"); axes[1, 0].set_ylabel("mIoU"); axes[0, 0].legend(); axes[1, 0].legend()
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
display(f1.round(3).rename("F1").to_frame().T)
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
"""md:## 2.8 Example predictions on test places

Label, one scene per month, and the month-mean map (what the prototype shows).""",
r'''
from pipeline import load_model, colourise
from eval_change import scene_probs, monthly_maps
model, stats, cfg, device = load_model(RUNS / "E1")
mean = np.asarray(stats["mean"], np.float32)[:, None, None]; std = np.asarray(stats["std"], np.float32)[:, None, None]
idx = pd.read_csv(PREP / "index.csv"); test_tiles = idx[(idx.status == "ok") & (idx.split == "test")]
picks = pd.concat([test_tiles[test_tiles.settlement == s].head(1) for s in ["city", "town", "village", "none"]])
fig, axes = plt.subplots(len(picks), 4, figsize=(13, 3.3 * len(picks)), squeeze=False)
for i, r in enumerate(picks.itertuples()):
    z = np.load(PREP / f"{r.tile_id}.npz")
    probs = scene_probs(model, z["s1"], mean, std, torch.device(device), bool(cfg.get("lee", False)))
    nod = (z["s1"] == -32768).any(1)
    a = monthly_maps(probs, nod, z["month"].astype(int), z["day"].astype(int), z["valid"], z["orbit"].astype(str), "best", False)
    b = monthly_maps(probs, nod, z["month"].astype(int), z["day"].astype(int), z["valid"], z["orbit"].astype(str), "mean", False)
    m = int(np.argmax(z["n"]))
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

The prototype (`prototype/app.py`) takes a location and a start month, downloads every raw Sentinel-1 scene of 12
months, predicts a map per month with E1 (month mean), and measures **built-up change** with a robust rule (built in
most of the last 3 months and in none of the first 3; patches under 400 m² dropped). Each place is checked against two
independent products never used in training: **Open Buildings 2.5D Temporal** (yearly building presence; Africa,
South / Southeast Asia, Latin America) and **ESRI 10 m land cover**. The verdict compares the new built-up with the
model's measured error level on unseen test places.

The examples were computed with `prototype/case_studies.py` (July 2022 – June 2023, 5.12 km × 5.12 km) and include
places chosen *because* they are hard for radar.""",
SETUP,
r'''
import gzip, pickle
import pipeline as P
CASES = sorted((ROOT / "prototype" / "case_studies").glob("*.pkl.gz"))
cases = []
for f in CASES:
    with gzip.open(f, "rb") as fh:
        cases.append(pickle.load(fh))
print(len(cases), "places:", ", ".join(c["case"]["title"] for c in cases))
''',
r'''
def built_refs(res, bc):
    refs = []
    if res.get("obt"):
        o = res["obt"]
        refs.append(P.reference_change(f"Open Buildings Temporal ({o['date_a'][:4]} vs {o['date_b'][:4]})",
                                       o["built_a"], o["built_b"], o["valid"], bc))
    if res.get("esri_ab"):
        y0, y1, ea, eb = res["esri_ab"]
        refs.append(P.reference_change(f"ESRI land cover ({y0} vs {y1})", ea == P.BUILT, eb == P.BUILT,
                                       (ea >= 0) & (eb >= 0), bc))
    return refs

summary = []
for res in cases:
    c = res["case"]; bc = P.built_change(res["pred"], res["months"])
    v, why, exp = P.change_verdict(bc, c["floor"])
    refs = built_refs(res, bc)
    row = {"place": c["title"], "type": c["kind"], "scenes": int(res["n"].sum()),
           "built-up start %": bc["before_pct"], "end %": bc["after_pct"], "new ha": bc["gain_ha"],
           "lost ha": bc["loss_ha"], "error level ha": exp, "verdict": v}
    for r in refs:
        key = "OBT" if r["source"].startswith("Open") else "ESRI"
        row[f"{key} new ha"] = r["new built-up (ha)"]; row[f"{key} confirms %"] = r["our new built-up that it confirms (%)"]
    summary.append(row)
display(pd.DataFrame(summary).set_index("place").round(1))
''',
"""md:## Each place in detail""",
r'''
from IPython.display import display
for res in cases:
    c = res["case"]; months = res["months"]; bc = P.built_change(res["pred"], months)
    v, why, _ = P.change_verdict(bc, c["floor"])
    display(Markdown(f"### {c['title']}  ·  *{c['kind']}*\n\n**What to look for:** {c['note']}\n\n**Verdict: {v}.** {why}"))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1, 1, 1.4]})
    ax[0].imshow(P.colourise(P.year_majority(res["pred"]))); ax[0].set_title("year majority (our model)")
    ax[1].imshow(P.change_rgb(bc)); ax[1].set_title("built-up change: magenta new, cyan lost")
    pct = P.class_percentages(res["pred"], months)
    for k, col in enumerate(pct.columns):
        ax[2].plot(range(12), pct[col], color=CLASS_COLOURS[k], label=col, marker="o", ms=3)
    ax[2].set_xticks(range(12), [m[2:] for m in months], rotation=45, fontsize=8); ax[2].set_ylabel("% of area")
    ax[2].legend(fontsize=7, ncol=2); ax[2].set_title("class % per month")
    for a in ax[:2]: a.axis("off")
    plt.tight_layout(); plt.show()
    display(P.built_change_table(bc, built_refs(res, bc)).round(2))
    pc = P.period_change(res["pred"], months)
    if pc is not None:
        display(pc.round(1))
''',
]


if __name__ == "__main__":
    for name, cells in [("01_data.ipynb", NB1), ("02_training_evaluation.ipynb", NB2), ("03_case_studies.ipynb", NB3)]:
        nbf.write(nb(cells), HERE / name)
        print("wrote", HERE / name)
