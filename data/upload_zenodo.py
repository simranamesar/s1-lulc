"""Upload the packed dataset to Zenodo as a DRAFT (never publishes: check it on zenodo.org and press Publish yourself).

1. zenodo.org -> log in (e.g. with GitHub) -> top right menu -> Applications -> Personal access tokens -> New token,
   scopes "deposit:write" and "deposit:actions". Copy the token.
2. On the server (the token is typed at a hidden prompt, it is not stored anywhere):
       python3 upload_zenodo.py --dir ~/vault/dataset_upload --author "Lastname, Firstname"
3. Open the printed draft link, check files and text, then press Publish (permanent: files cannot be changed after).

If the upload stops halfway, rerun with --deposition <id printed before>: files already there are skipped.
"""
import argparse
import getpass
import os
import sys
import time
from pathlib import Path

import requests

API = "https://zenodo.org/api"

DESCRIPTION = """<p>Training and evaluation data of the project <em>Sentinel-1 land cover monitor</em>: 600 places of
2.56 km &times; 2.56 km on every inhabited continent, stratified by world region, biome, settlement type and land
cover, with every Sentinel-1 IW scene of 2023 (34,981 scenes, VV, VH, VH&minus;VV in dB, terrain-flattened
&gamma;&#8304;, 10 m) and monthly land-cover labels in 6 classes (water, trees, grass &amp; shrub, crops, built-up,
bare &amp; snow) from Dynamic World and Open Buildings, aligned pixel by pixel. Train / validation / test =
409 / 61 / 130 places, split by 5&deg; geographic blocks.</p>
<p>See README.md for the file format. Code: <a href="https://github.com/simranamesar/s1-lulc">
https://github.com/simranamesar/s1-lulc</a>. Contains modified Copernicus Sentinel data 2023.</p>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="folder with the packed files (pack_dataset.py)")
    ap.add_argument("--author", required=True, help='"Lastname, Firstname"')
    ap.add_argument("--affiliation", default="SRH Hochschule Heidelberg")
    ap.add_argument("--deposition", default="", help="id of an existing draft to continue")
    args = ap.parse_args()

    token = os.environ.get("ZENODO_TOKEN") or getpass.getpass("Zenodo token (hidden): ").strip()
    auth = {"Authorization": f"Bearer {token}"}
    files = sorted(p for p in Path(args.dir).expanduser().iterdir() if p.is_file())
    if not files:
        sys.exit(f"no files in {args.dir}")

    if args.deposition:
        r = requests.get(f"{API}/deposit/depositions/{args.deposition}", headers=auth, timeout=60)
    else:
        r = requests.post(f"{API}/deposit/depositions", json={}, headers=auth, timeout=60)
    if r.status_code >= 300:
        sys.exit(f"Zenodo refused ({r.status_code}): {r.text[:300]}")
    dep = r.json()
    dep_id, bucket = dep["id"], dep["links"]["bucket"]
    print(f"draft id {dep_id} (use --deposition {dep_id} to continue if the upload stops)")

    have = {f["filename"]: f.get("filesize") for f in dep.get("files", [])}
    for f in files:
        size = f.stat().st_size
        if have.get(f.name) == size:
            print(f"  {f.name}: already uploaded, skipped")
            continue
        print(f"  uploading {f.name} ({size / 1e9:.2f} GB) ...", flush=True)
        t0 = time.time()
        with open(f, "rb") as fh:
            u = requests.put(f"{bucket}/{f.name}", data=fh, headers=auth, timeout=None)
        if u.status_code >= 300:
            sys.exit(f"upload of {f.name} failed ({u.status_code}): {u.text[:300]}")
        print(f"    done in {(time.time() - t0) / 60:.1f} min")

    meta = {"metadata": {
        "title": "S1-LULC-600: Sentinel-1 single scenes with monthly land-cover labels, 600 places worldwide (2023)",
        "upload_type": "dataset",
        "description": DESCRIPTION,
        "creators": [{"name": args.author, "affiliation": args.affiliation}],
        "keywords": ["Sentinel-1", "SAR", "land cover", "semantic segmentation", "Dynamic World", "deep learning",
                     "built-up change"],
        "license": "cc-by-4.0",
        "access_right": "open",
        "related_identifiers": [{"identifier": "https://github.com/simranamesar/s1-lulc",
                                 "relation": "isSupplementTo", "resource_type": "software"}],
    }}
    m = requests.put(f"{API}/deposit/depositions/{dep_id}", json=meta, headers=auth, timeout=60)
    if m.status_code >= 300:
        sys.exit(f"metadata refused ({m.status_code}): {m.text[:300]}")
    doi = m.json().get("metadata", {}).get("prereserve_doi", {}).get("doi", "")
    print(f"\nDraft ready (NOT published): https://zenodo.org/deposit/{dep_id}")
    if doi:
        print(f"Reserved DOI: https://doi.org/{doi}  (works once you press Publish)")


if __name__ == "__main__":
    main()
