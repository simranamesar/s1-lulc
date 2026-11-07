"""How many Sentinel-1 scenes exist over a few candidate places (no download, metadata only, ~1 min).

Used to replace the Grand Erg Occidental gallery place, which had no Sentinel-1 scenes in 2022-23.

    python3 check_coverage.py --project scm-ksns
"""
import argparse

import pipeline as P

CANDIDATES = [
    ("erg_chebbi", "Erg Chebbi dunes (Morocco)", 31.150, -3.980),
    ("douz", "Grand Erg Oriental near Douz (Tunisia)", 33.300, 9.000),
    ("liwa", "Liwa dunes, Rub' al Khali (UAE)", 23.050, 53.750),
    ("thar", "Sam sand dunes, Thar desert (India)", 26.830, 70.500),
    ("namib", "Namib Sand Sea, Sossusvlei (Namibia)", -24.750, 15.300),
    ("sahara", "Grand Erg Occidental dunes (Algeria)", 30.500, 0.500),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="scm-ksns")
    ap.add_argument("--start", default="2022-07")
    args = ap.parse_args()
    import ee
    ee.Initialize(project=args.project)
    from s1_composite import s1_scene_table
    months = P.month_list(args.start, 12 + P.EXTRA_MONTHS)
    print(f"Sentinel-1 scenes per month, {months[0]} to {months[-1]}:")
    for cid, title, lat, lon in CANDIDATES:
        region = P._region(P.area_grid(lat, lon, 512))
        counts = []
        for ym in months:
            y, m = map(int, ym.split("-"))
            end = f"{y + (m == 12):04d}-{m % 12 + 1:02d}-01"
            counts.append(len(s1_scene_table(region, f"{ym}-01", end)))
        ok = sum(c > 0 for c in counts)
        print(f"  {cid:11s} {title:42s} {sum(counts):4d} scenes, {ok:2d} of {len(months)} months  {counts}")


if __name__ == "__main__":
    main()
