"""Pack the dataset for publishing (Zenodo): prepared files per split, metadata, raw label GeoTIFFs, README.

    python3 pack_dataset.py --data tiles_v0 --out ~/vault/dataset_upload

Writes <out>/prepared_train.tar, prepared_val.tar, prepared_test.tar, metadata.zip, raw_labels.tar, README.md.
The .npz files are already compressed, so the tar files are not compressed again. Needs about 13 GB free.
"""
import argparse
import shutil
import tarfile
import zipfile
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="tiles_v0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out).expanduser()
    prep = data / "prepared"
    out.mkdir(parents=True, exist_ok=True)

    idx = pd.read_csv(prep / "index.csv")
    ok = idx[idx["status"] == "ok"]
    for split in ("train", "val", "test"):
        ids = ok[ok["split"] == split]["tile_id"]
        path = out / f"prepared_{split}.tar"
        with tarfile.open(path, "w") as tf:
            for tid in ids:
                tf.add(prep / f"{tid}.npz", arcname=f"prepared/{tid}.npz")
        print(f"{path.name}: {len(ids)} places, {path.stat().st_size / 1e9:.2f} GB")

    meta = out / "metadata.zip"
    with zipfile.ZipFile(meta, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in [data / "tiles.csv", data / "tiles.geojson", prep / "index.csv", prep / "stats.json",
                  *sorted(prep.glob("balance_report.*"))]:
            if f.exists():
                zf.write(f, arcname=f.name)
    print(f"{meta.name}: {meta.stat().st_size / 1e6:.1f} MB")

    raw = out / "raw_labels.tar"
    with tarfile.open(raw, "w") as tf:
        for f in sorted((data / "raw").glob("*.tif")):
            tf.add(f, arcname=f"raw/{f.name}")
    print(f"{raw.name}: {raw.stat().st_size / 1e9:.2f} GB")

    readme = Path(__file__).resolve().parent / "DATASET_README.md"
    shutil.copy(readme, out / "README.md")
    print(f"README.md copied. Total {sum(f.stat().st_size for f in out.iterdir()) / 1e9:.1f} GB in {out}")


if __name__ == "__main__":
    main()
