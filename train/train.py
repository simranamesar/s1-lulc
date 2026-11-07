"""Train a U-Net that maps ONE RAW Sentinel-1 scene (VV, VH, VH-VV) to a 9-class LULC map.

One sample = one single scene of a tile, labelled with that month's map, with at least --min-label-frac
labelled pixels. No speckle filter unless --lee (5 x 5 Lee filter, experiment variant).
Training draws samples with the balanced sampler from balance.py (region x settlement and class rarity);
the loss adds a mild class weight computed from the class shares the model actually sees.

Evaluation (val each epoch, test once at the end with the best checkpoint):
    - overall accuracy (OA), per-class IoU / F1 / precision / recall, mIoU and macro F1 over 6 classes
    - mIoU per settlement type (city / town / village / none) and per world region
      (performance on regions unlike the rest of the training data)
    - built-up recall and IoU on village tiles
    - month-to-month flip rate of predictions vs labels on test tiles (lower = steadier change numbers)
    - built-up precision / recall, built-up false change rate (built status flips where the label is stable)
    - new built-up over 12 months with the prototype's change rule: predicted vs label, false ha per km2

Training stops early when val mIoU has not improved for --patience epochs (lecture 05).
Model variants (model.py): --encoder-weights imagenet|none, --freeze-encoder, --upsample interp|transposed.

Outputs in runs/<name>/: config.json, history.csv (one row per epoch: loss, mIoU, accuracy, per-class IoU, lr),
steps.csv (training loss after every batch), snapshots/ (predictions of fixed validation scenes after every epoch,
for watching the model learn), best.pt, last.pt, curves.png, test_metrics.json, test_per_class.csv, test_confusion.csv

Usage
    CUDA_VISIBLE_DEVICES=0 python3 train.py --prep-dir ../data/tiles_v0/prepared --label dw --name E0
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from s1_composite import best_scene_per_month, lee_filter_db  # noqa: E402
from classes import BUILT, CLASS_NAMES, N_CLASSES  # noqa: E402
from model import build_model
from balance import (effective_class_shares, effective_sample_size, item_table,
                     loss_class_weights, sample_weights)

S1_NODATA = -32768
IGNORE = -1


# ----------------------------------------------------------------------------- data

class TileMonths(Dataset):
    """Samples = rows of balance.item_table: (tile, scene index, month of its label)."""

    def __init__(self, items: pd.DataFrame, prep_dir: Path, label_key: str, mean, std, augment: bool,
                 lee: bool = False, jitter_db: float = 0.0, noise_db: float = 0.0):
        self.mean = np.asarray(mean, dtype=np.float32)[:, None, None]
        self.std = np.asarray(std, dtype=np.float32)[:, None, None]
        self.augment, self.lee = augment, lee
        self.jitter_db, self.noise_db = jitter_db, noise_db
        self.data = {}
        for tid in items["tile_id"].unique():
            z = np.load(prep_dir / f"{tid}.npz")
            self.data[tid] = (z["s1"], z[label_key])
        self.items = list(zip(items["tile_id"], items["sample"], items["month"]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        tid, k, m = self.items[i]
        s1, lbl = self.data[tid]
        scene = lee_filter_db(s1[k]) if self.lee else s1[k]
        x, y = to_tensors(scene, lbl[m], self.mean, self.std)
        if self.augment and (self.jitter_db or self.noise_db):
            # radiometric augmentation (E6): soil moisture, rain and calibration shift backscatter by ~1-2 dB.
            # VV and VH get their own offset (+ pixel noise); VH-VV follows from them. Applied in dB, then scaled.
            ok = ~(scene == S1_NODATA).any(0)
            a, b = np.random.uniform(-self.jitter_db, self.jitter_db, 2)
            H, W = ok.shape
            na = np.random.normal(0, self.noise_db, (H, W)) if self.noise_db else 0
            nb = np.random.normal(0, self.noise_db, (H, W)) if self.noise_db else 0
            dv, dh = a + na, b + nb
            for c, d in ((0, dv), (1, dh), (2, dh - dv)):
                x[c] = np.where(ok, x[c] + d / self.std[c, 0, 0], x[c])
        if self.augment:
            k = np.random.randint(4)
            x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k)
            if np.random.rand() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))


def to_tensors(s1_month: np.ndarray, lbl_month: np.ndarray, mean, std):
    """int16 dBx100 (3,H,W) -> normalised float32; nodata -> 0 and label ignored there."""
    nod = s1_month == S1_NODATA
    x = (s1_month.astype(np.float32) / 100 - mean) / std
    x[nod] = 0
    y = lbl_month.astype(np.int64)
    y[nod.any(0)] = IGNORE
    return x, y


# ----------------------------------------------------------------------------- metrics

def confusion(pred: np.ndarray, y: np.ndarray, n: int = N_CLASSES) -> np.ndarray:
    ok = y >= 0
    return np.bincount(y[ok] * n + pred[ok], minlength=n * n).reshape(n, n)


def iou_per_class(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm).astype(float)
    union = cm.sum(0) + cm.sum(1) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, tp / union, np.nan)


def f1_per_class(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-class F1, precision, recall from a confusion matrix (rows = label, columns = prediction)."""
    tp = np.diag(cm).astype(float)
    pred, true = cm.sum(0).astype(float), cm.sum(1).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(pred > 0, tp / pred, np.nan)
        rec = np.where(true > 0, tp / true, np.nan)
        f1 = np.where(pred + true > 0, 2 * tp / (pred + true), np.nan)
    return f1, prec, rec


def miou(cm: np.ndarray) -> float:
    support = cm.sum(1) > 0  # only classes that occur in the labels
    v = iou_per_class(cm)[support]
    return float(np.nanmean(v)) if v.size else float("nan")


def summarise_cm(cm: np.ndarray) -> dict:
    acc = np.diag(cm).sum() / max(cm.sum(), 1)
    support = cm.sum(1) > 0
    f1 = f1_per_class(cm)[0][support]
    return {"OA": float(acc), "mIoU": miou(cm), "macro_F1": float(np.nanmean(f1)) if f1.size else float("nan"),
            "pixels": int(cm.sum())}


# ----------------------------------------------------------------------------- model / loss

class CEDice(nn.Module):
    """Cross-entropy (class-weighted) + Dice, optionally + lovasz_w x Lovasz-softmax (a direct IoU surrogate)."""

    def __init__(self, weights: torch.Tensor, lovasz_w: float = 0.0):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.ce = nn.CrossEntropyLoss(weight=weights, ignore_index=IGNORE)
        self.dice = smp.losses.DiceLoss(mode="multiclass", ignore_index=IGNORE)
        self.lovasz_w = lovasz_w
        self.lovasz = smp.losses.LovaszLoss(mode="multiclass", ignore_index=IGNORE) if lovasz_w > 0 else None

    def forward(self, logits, y):
        loss = self.ce(logits, y) + self.dice(logits, y)
        if self.lovasz is not None:
            loss = loss + self.lovasz_w * self.lovasz(logits, y)
        return loss


# ----------------------------------------------------------------------------- loops

def run_epoch(model, loader, loss_fn, device, opt=None, sched=None, amp_dtype=None, step_log=None):
    """One pass over loader; returns the mean loss and the confusion matrix. step_log (list): training loss and
    learning rate of every batch are appended to it."""
    train = opt is not None
    model.train(train)
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train), torch.autocast(
                device_type=device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
            logits = model(x)
            loss = loss_fn(logits.float(), y)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step_log is not None:
                step_log.append((loss.item(), opt.param_groups[0]["lr"]))
            sched.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
        cm += confusion(logits.argmax(1).cpu().numpy().ravel(), y.cpu().numpy().ravel())
    return total / max(n, 1), cm


@torch.no_grad()
def built_gain(seq: np.ndarray, window: int = 3, min_px: int = 4) -> np.ndarray | None:
    """New built-up between the first and last `window` months with data: the same rule as the prototype
    (prototype/pipeline.py built_change): built in the majority of the last months, in none of the first,
    patches smaller than min_px dropped. seq = (12, H, W) class ids, -1 = no data."""
    from scipy import ndimage
    have = [m for m in range(seq.shape[0]) if (seq[m] >= 0).any()]
    w = min(window, len(have) // 2)
    if not w:
        return None
    first, last = have[:w], have[-w:]
    need = w // 2 + 1
    bf = sum((seq[m] == BUILT).astype(int) for m in first)
    bl = sum((seq[m] == BUILT).astype(int) for m in last)
    ok = (sum((seq[m] >= 0).astype(int) for m in first) >= need) & (sum((seq[m] >= 0).astype(int) for m in last) >= need)
    gain = (bl >= need) & (bf == 0) & ok
    lab, k = ndimage.label(gain, structure=np.ones((3, 3), bool))
    if k:
        keep = np.bincount(lab.ravel()) >= min_px
        keep[0] = False
        gain = keep[lab]
    return gain


def evaluate_tiles(model, tiles: pd.DataFrame, prep_dir: Path, label_key: str, mean, std, device, amp_dtype,
                   lee: bool = False):
    """Every test scene vs its month's label (confusion overall / per settlement / per region) + flip rate of
    the monthly map sequence (best scene per month, as in the prototype)."""
    model.eval()
    mean = np.asarray(mean, dtype=np.float32)[:, None, None]
    std = np.asarray(std, dtype=np.float32)[:, None, None]
    cm_all = np.zeros((N_CLASSES, N_CLASSES), dtype=np.float64)
    cm_by, cm_reg = {}, {}
    flips = []
    bflip = [0, 0]                 # built status flips where the label's built status is stable: [flips, pixels]
    gain = [0, 0, 0, 0]            # predicted new built-up px, of which also new in the label, label new px, area px
    for r in tiles.itertuples():
        z = np.load(prep_dir / f"{r.tile_id}.npz")
        s1, n, lbl = z["s1"], z["n"], z[label_key]
        if "month" in z.files:                       # single scenes
            smonth = z["month"].astype(int)
            best = z["best"] if "best" in z.files else best_scene_per_month(smonth, z["day"], z["valid"])
        else:                                        # old monthly composites
            smonth = np.arange(12)
            best = np.where(n > 0, np.arange(12), -1)
        idx = [k for k in range(len(s1)) if (n[smonth[k]] > 0)]
        if not idx:
            continue
        preds = {}
        for b0 in range(0, len(idx), 16):
            ks = idx[b0:b0 + 16]
            xs, ys = zip(*[to_tensors(lee_filter_db(s1[k]) if lee else s1[k], lbl[smonth[k]], mean, std) for k in ks])
            x = torch.from_numpy(np.stack(xs)).to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype or torch.float32,
                                enabled=amp_dtype is not None):
                p = model(x).argmax(1).cpu().numpy()
            for k, pk, yk in zip(ks, p, ys):
                preds[k] = pk
                # every scene is scored against its month's label; weighted so each tile counts as 12 scenes
                # (dense European tiles have up to ~180 scenes and would otherwise dominate the test metrics)
                cm = confusion(pk.ravel(), yk.ravel()).astype(np.float64) * (12.0 / len(idx))
                cm_all += cm
                cm_by.setdefault(r.settlement, np.zeros_like(cm))
                cm_by[r.settlement] += cm
                cm_reg.setdefault(r.region, np.zeros_like(cm))
                cm_reg[r.region] += cm

        # flip rate on the monthly map sequence the prototype would show (one scene per month)
        pf = lf = pt = lt = 0
        for m in range(11):
            a, b = int(best[m]), int(best[m + 1])
            if a < 0 or b < 0 or a not in preds or b not in preds:
                continue
            pa, pb = preds[a], preds[b]
            pf += int((pa != pb).sum()); pt += pa.size
            ya, yb = lbl[m], lbl[m + 1]
            both = (ya >= 0) & (yb >= 0)
            lf += int(((ya != yb) & both).sum()); lt += int(both.sum())
            stable = both & ((ya == BUILT) == (yb == BUILT))
            bflip[0] += int((((pa == BUILT) != (pb == BUILT)) & stable).sum()); bflip[1] += int(stable.sum())
        if pt:
            flips.append((pf / pt, lf / lt if lt else np.nan))

        # built-up change with the prototype's rule: predicted new built-up vs new built-up in the labels
        seq = np.full(lbl.shape, -1, np.int16)
        for m in range(12):
            if int(best[m]) in preds:
                seq[m] = np.where(lbl[m] >= 0, preds[int(best[m])], -1)
        gp, gl = built_gain(seq), built_gain(np.where(seq >= 0, lbl, -1))
        if gp is not None and gl is not None:
            gain[0] += int(gp.sum()); gain[1] += int((gp & gl).sum()); gain[2] += int(gl.sum())
            gain[3] += int(gp.size)

    f1 = f1_per_class(cm_all)[0]
    out = {"overall": summarise_cm(cm_all),
           "per_class_iou": dict(zip(CLASS_NAMES, np.round(iou_per_class(cm_all), 4).tolist())),
           "per_class_f1": dict(zip(CLASS_NAMES, np.round(f1, 4).tolist())),
           "per_settlement": {k: summarise_cm(v) for k, v in sorted(cm_by.items())},
           "per_region": {k: summarise_cm(v) for k, v in sorted(cm_reg.items())}}
    if "village" in cm_by:
        v = cm_by["village"]
        out["village_built"] = {
            "recall": float(v[BUILT, BUILT] / max(v[BUILT].sum(), 1)),
            "iou": float(iou_per_class(v)[BUILT]) if v[BUILT].sum() + v[:, BUILT].sum() else float("nan"),
            "label_pixels": int(v[BUILT].sum()),
        }
    out["built"] = {"precision": float(cm_all[BUILT, BUILT] / max(cm_all[:, BUILT].sum(), 1)),
                    "recall": float(cm_all[BUILT, BUILT] / max(cm_all[BUILT].sum(), 1)),
                    "false_change_rate": float(bflip[0] / bflip[1]) if bflip[1] else float("nan")}
    if gain[3]:
        km2 = gain[3] * 1e-4                                       # 10 m pixels -> km2
        out["built_gain"] = {"pred_new_ha_per_km2": gain[0] * 0.01 / km2, "label_new_ha_per_km2": gain[2] * 0.01 / km2,
                             "false_new_ha_per_km2": (gain[0] - gain[1]) * 0.01 / km2,
                             "precision": gain[1] / gain[0] if gain[0] else float("nan"),
                             "recall": gain[1] / gain[2] if gain[2] else float("nan")}
    if flips:
        f = np.array(flips, dtype=float)
        out["flip_rate"] = {"prediction": float(np.nanmean(f[:, 0])), "label": float(np.nanmean(f[:, 1]))}
    return out, cm_all


def pick_snapshot_items(items: pd.DataFrame, k: int):
    """k fixed validation scenes to watch during training: one per settlement type (city, town, village, none,
    then again), different world regions where possible, each a well-labelled scene with several classes."""
    from balance import CNT_COLS
    it = items.copy()
    cnt = it[CNT_COLS].to_numpy(float)
    tot = cnt.sum(1)
    # score: labelled pixels x number of classes covering >= 5 % (interesting maps, not all one class)
    it["score"] = tot * ((cnt / np.maximum(tot, 1)[:, None]) >= 0.05).sum(1)
    order = ["city", "town", "village", "none"]
    picks, used_tiles, used_reg = [], set(), set()
    for i in range(k):
        cand = it[(it.settlement == order[i % len(order)]) & ~it.tile_id.isin(used_tiles)]
        if cand.empty:
            cand = it[~it.tile_id.isin(used_tiles)]
        if cand.empty:
            break
        fresh = cand[~cand.region.isin(used_reg)]
        r = (fresh if len(fresh) else cand).sort_values("score").iloc[-1]
        picks.append((r.tile_id, int(r["sample"]), int(r.month), str(r.settlement), str(r.region)))
        used_tiles.add(r.tile_id); used_reg.add(r.region)
    return picks


class Snapshots:
    """Saves the predictions for a few fixed validation scenes after every epoch (snapshots/ep###.npz), so a
    notebook can show how the output maps change while the network learns."""

    def __init__(self, run: Path, picks, prep_dir: Path, label_key: str, mean, std, lee: bool):
        self.dir = run / "snapshots"
        self.dir.mkdir(exist_ok=True)
        mean = np.asarray(mean, np.float32)[:, None, None]
        std = np.asarray(std, np.float32)[:, None, None]
        xs, ys, raw = [], [], []
        for tid, k, m, *_ in picks:
            z = np.load(prep_dir / f"{tid}.npz")
            sc = lee_filter_db(z["s1"][k]) if lee else z["s1"][k]
            x, y = to_tensors(sc, z[label_key][m], mean, std)
            xs.append(x); ys.append(y); raw.append(z["s1"][k])
        self.x = torch.from_numpy(np.stack(xs))
        self.y = np.stack(ys)
        np.savez_compressed(self.dir / "examples.npz", s1=np.stack(raw), label=self.y.astype(np.int8),
                            tile_id=np.array([p[0] for p in picks]), sample=np.array([p[1] for p in picks]),
                            month=np.array([p[2] for p in picks]), settlement=np.array([p[3] for p in picks]),
                            region=np.array([p[4] for p in picks]))

    @torch.no_grad()
    def save(self, model, ep: int, device, amp_dtype):
        model.eval()
        with torch.autocast(device_type=device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
            p = model(self.x.to(device)).float().softmax(1).cpu().numpy()
        pred = p.argmax(1).astype(np.uint8)
        np.savez_compressed(self.dir / f"ep{ep:03d}.npz", pred=pred,
                            conf=(p.max(1) * 255).astype(np.uint8), p_built=(p[:, BUILT] * 255).astype(np.uint8))
        ok = self.y >= 0
        return float((pred[ok] == self.y[ok]).mean()) if ok.any() else float("nan")


def plot_curves(hist: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist.epoch, hist.train_loss, label="train")
    ax[0].plot(hist.epoch, hist.val_loss, label="val")
    ax[0].set_title("Loss (CE + Dice)"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(hist.epoch, hist.train_miou, label="train mIoU")
    ax[1].plot(hist.epoch, hist.val_miou, label="val mIoU")
    ax[1].plot(hist.epoch, hist.val_oa, label="val overall accuracy", linestyle="--")
    ax[1].set_title("mIoU and accuracy"); ax[1].set_xlabel("epoch"); ax[1].set_ylim(0, 1); ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prep-dir", required=True)
    ap.add_argument("--label", default="dw", choices=["dw", "hyb", "hyb_s", "hyb_y"])
    ap.add_argument("--name", required=True, help="experiment id, e.g. E0")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--encoder-weights", default="imagenet", help="'imagenet' (transfer learning) or 'none' (scratch)")
    ap.add_argument("--freeze-encoder", action="store_true", help="fixed feature extractor: train decoder only")
    ap.add_argument("--upsample", default="interp", choices=["interp", "transposed"])
    ap.add_argument("--patience", type=int, default=10, help="early stopping: epochs without val mIoU gain")
    ap.add_argument("--lee", action="store_true", help="apply a 5x5 Lee speckle filter to every scene")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--min-label-frac", type=float, default=0.2)
    ap.add_argument("--sampler", default="balanced", choices=["balanced", "uniform"],
                    help="balanced: draw tile-months by region x settlement and class rarity (see balance.py)")
    ap.add_argument("--geo-power", type=float, default=1.0)
    ap.add_argument("--class-power", type=float, default=0.5)
    ap.add_argument("--max-ratio", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dropout", type=float, default=0.0, help="Dropout2d before the 1x1 output layer (E6: 0.2)")
    ap.add_argument("--jitter-db", type=float, default=0.0, help="random VV / VH offset per scene, +- dB (E6: 1.5)")
    ap.add_argument("--noise-db", type=float, default=0.0, help="random per-pixel noise, std in dB (E6: 0.5)")
    ap.add_argument("--lovasz", type=float, default=0.0, help="weight of the Lovasz-softmax loss (E6: 0.5)")
    ap.add_argument("--train-frac", type=float, default=1.0,
                    help="use only this share of the training places (learning curve: does more data help?)")
    ap.add_argument("--snapshots", type=int, default=6,
                    help="save predictions of this many fixed validation scenes after every epoch (0 = off)")
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    prep = Path(args.prep_dir)
    run = Path(args.runs_dir) / args.name
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps({**vars(args), "n_classes": N_CLASSES, "class_names": CLASS_NAMES},
                                                indent=2))

    stats = json.loads((prep / "stats.json").read_text())
    (run / "stats.json").write_text(json.dumps(stats, indent=2))   # the prototype normalises with these
    index = pd.read_csv(prep / "index.csv")
    index = index[index["status"] == "ok"]
    split = {s: index[index["split"] == s] for s in ("train", "val", "test")}
    if args.train_frac < 1:
        # learning curve: keep a share of the TRAINING places (same share in every region x settlement type);
        # val and test stay unchanged so the runs are comparable
        tr = split["train"]
        rng = np.random.default_rng(args.seed)
        keep = []
        for _, g in tr.groupby(["region", "settlement"]):
            keep += list(rng.choice(g.index, max(1, round(len(g) * args.train_frac)), replace=False))
        split["train"] = tr.loc[sorted(keep)]   # all columns kept (groupby.apply drops the group columns in pandas 3)
        print(f"Learning curve: {len(split['train'])} of {len(tr)} training places ({args.train_frac:.0%})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = None if args.no_amp or device.type == "cpu" else torch.bfloat16
    print(f"Device: {device}{' (' + torch.cuda.get_device_name(0) + ')' if device.type == 'cuda' else ''}, "
          f"label: {args.label}")

    if args.lee and "mean_lee" not in stats:
        raise SystemExit("--lee needs mean_lee/std_lee in stats.json: rerun prepare_data.py with --scenes-dir")
    mean, std = (stats["mean_lee"], stats["std_lee"]) if args.lee else (stats["mean"], stats["std"])
    print(f"Input: {stats.get('input', 'monthly_median')}{' + Lee 5x5 filter' if args.lee else ', no speckle filter'}")
    items = {s: item_table(prep, split[s], args.label, args.min_label_frac) for s in ("train", "val")}
    ds_train = TileMonths(items["train"], prep, args.label, mean, std, True, args.lee, args.jitter_db, args.noise_db)
    ds_val = TileMonths(items["val"], prep, args.label, mean, std, False, args.lee)
    print(f"Samples (scenes): train {len(ds_train)}, val {len(ds_val)}; "
          f"tiles train/val/test {len(split['train'])}/{len(split['val'])}/{len(split['test'])}")

    if args.sampler == "balanced":
        sw = sample_weights(items["train"], args.geo_power, args.class_power, args.max_ratio)
        sampler = WeightedRandomSampler(torch.as_tensor(sw, dtype=torch.double), len(ds_train), replacement=True)
        print(f"Balanced sampler: effective sample size {effective_sample_size(sw):.0f} of {len(sw)}")
    else:
        sw, sampler = None, None
    raw_sh = effective_class_shares(items["train"])
    seen_sh = effective_class_shares(items["train"], sw)
    print("Class share of training pixels, uniform -> " + args.sampler + ": " + ", ".join(
        f"{n} {a * 100:.1f}->{b * 100:.1f}%" for n, a, b in zip(CLASS_NAMES, raw_sh, seen_sh)))
    pd.DataFrame({"class": CLASS_NAMES, "uniform_%": raw_sh * 100, "seen_%": seen_sh * 100}).to_csv(
        run / "train_class_shares.csv", index=False)

    # Docker containers often give /dev/shm only 64 MB; worker processes pass batches through it and then fail
    # with "unable to allocate shared memory". The scenes are already in RAM, so loading in the main process
    # (workers 0) costs little.
    try:
        import shutil
        shm_mb = shutil.disk_usage("/dev/shm").total / 2**20
    except OSError:
        shm_mb = float("inf")
    if args.workers > 0 and shm_mb < 2048:
        print(f"Shared memory /dev/shm is only {shm_mb:.0f} MB -> data loading in the main process (workers 0)")
        args.workers = 0
    dl_train = DataLoader(ds_train, args.batch_size, shuffle=sampler is None, sampler=sampler,
                          num_workers=args.workers, pin_memory=True, drop_last=True,
                          persistent_workers=args.workers > 0)
    dl_val = DataLoader(ds_val, args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=True, persistent_workers=args.workers > 0)

    model = build_model(args.encoder, args.encoder_weights, args.upsample, args.freeze_encoder,
                        classes=N_CLASSES, dropout=args.dropout).to(device)
    n_all = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: U-Net {args.encoder}, weights {args.encoder_weights}, upsample {args.upsample}, "
          f"{n_all / 1e6:.1f}M parameters ({n_train / 1e6:.1f}M trainable)")
    w = torch.tensor(loss_class_weights(seen_sh), dtype=torch.float32).to(device)
    print("Class weights: " + ", ".join(f"{n} {v:.2f}" for n, v in zip(CLASS_NAMES, w.tolist())))
    loss_fn = CEDice(w, args.lovasz).to(device)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                            weight_decay=args.weight_decay)
    steps = max(len(dl_train), 1) * args.epochs
    warm = max(len(dl_train), 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(steps - warm, 1))))

    snaps = None
    if args.snapshots > 0:
        picks = pick_snapshot_items(items["val"], args.snapshots)
        snaps = Snapshots(run, picks, prep, args.label, mean, std, args.lee)
        acc0 = snaps.save(model, 0, device, amp_dtype)          # epoch 0 = before any training
        print(f"Snapshots: {len(picks)} fixed validation scenes saved after every epoch "
              f"(epoch 0, untrained: accuracy {acc0:.3f})")

    hist, step_rows, best, best_ep = [], [], -1.0, 0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        step_log = []
        tr_loss, tr_cm = run_epoch(model, dl_train, loss_fn, device, opt, sched, amp_dtype, step_log)
        with torch.no_grad():
            va_loss, va_cm = run_epoch(model, dl_val, loss_fn, device, amp_dtype=amp_dtype)
        step_rows += [{"epoch": ep, "step": len(step_rows) + i + 1, "loss": l, "lr": lr} for i, (l, lr) in enumerate(step_log)]
        pd.DataFrame(step_rows).to_csv(run / "steps.csv", index=False)
        va_iou = iou_per_class(va_cm)
        row = {"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss,
               "train_miou": miou(tr_cm), "val_miou": miou(va_cm), "val_oa": float(np.diag(va_cm).sum() / max(va_cm.sum(), 1)),
               "lr": opt.param_groups[0]["lr"], "seconds": time.time() - t0,
               "train_oa": float(np.diag(tr_cm).sum() / max(tr_cm.sum(), 1)),
               **{f"val_iou_{c}": float(v) for c, v in zip(CLASS_NAMES, va_iou)}}
        if snaps is not None:
            row["snapshot_acc"] = snaps.save(model, ep, device, amp_dtype)
        hist.append(row)
        pd.DataFrame(hist).to_csv(run / "history.csv", index=False)
        torch.save(model.state_dict(), run / "last.pt")
        flag = ""
        if row["val_miou"] > best:
            best, best_ep = row["val_miou"], ep
            torch.save(model.state_dict(), run / "best.pt")
            flag = "  *best"
        print(f"ep {ep:3d}  loss {tr_loss:.3f}/{va_loss:.3f}  mIoU {row['train_miou']:.3f}/{row['val_miou']:.3f}"
              f"  OA {row['train_oa']:.3f}/{row['val_oa']:.3f}  {row['seconds']:.0f}s{flag}")
        print("        val IoU  " + "  ".join(f"{c} {v:.2f}" for c, v in zip(CLASS_NAMES, va_iou)))
        if ep - best_ep >= args.patience:
            print(f"Early stopping: no val mIoU gain for {args.patience} epochs (best epoch {best_ep}).")
            break

    plot_curves(pd.DataFrame(hist), run / "curves.png")

    print("\nTest set (best checkpoint):")
    model.load_state_dict(torch.load(run / "best.pt", map_location=device))
    res, cm = evaluate_tiles(model, split["test"], prep, args.label, mean, std, device, amp_dtype, args.lee)
    res["best_val_miou"] = best
    res["best_epoch"] = best_ep
    res["epochs_run"] = len(hist)
    (run / "test_metrics.json").write_text(json.dumps(res, indent=2))
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(run / "test_confusion.csv")
    f1, prec, rec = f1_per_class(cm)
    pd.DataFrame({"class": CLASS_NAMES, "iou": iou_per_class(cm), "f1": f1, "precision": prec, "recall": rec,
                  "label_pixels": cm.sum(1)}).to_csv(run / "test_per_class.csv", index=False)

    o = res["overall"]
    print(f"  OA {o['OA']:.3f} | mIoU {o['mIoU']:.3f} | macro F1 {o['macro_F1']:.3f}")
    for group in ("per_settlement", "per_region"):
        print(f"  {group.replace('_', ' ')}:")
        for k, v in res[group].items():
            print(f"    {k:15s} OA {v['OA']:.3f}  mIoU {v['mIoU']:.3f}  macro F1 {v['macro_F1']:.3f}")
    if "village_built" in res:
        vb = res["village_built"]
        print(f"  village built-up: recall {vb['recall']:.3f}, IoU {vb['iou']:.3f} ({vb['label_pixels']} label px)")
    if "flip_rate" in res:
        print(f"  month-to-month flip rate: prediction {res['flip_rate']['prediction']:.3f} "
              f"vs label {res['flip_rate']['label']:.3f}")
    b = res["built"]
    print(f"  built-up: precision {b['precision']:.3f}, recall {b['recall']:.3f}, false change rate "
          f"{b['false_change_rate']:.4f} (built status flips month to month where the label stays the same)")
    if "built_gain" in res:
        g = res["built_gain"]
        print(f"  new built-up over 12 months (prototype rule): predicted {g['pred_new_ha_per_km2']:.2f} ha/km2, "
              f"labels {g['label_new_ha_per_km2']:.2f} ha/km2, false {g['false_new_ha_per_km2']:.2f} ha/km2 "
              f"(precision {g['precision']:.2f}, recall {g['recall']:.2f})")
    print("  per-class IoU / F1: " + ", ".join(
        f"{k} {v:.2f}/{res['per_class_f1'][k]:.2f}" for k, v in res["per_class_iou"].items() if v == v))
    print(f"\nSaved to {run}/")


if __name__ == "__main__":
    main()
