"""The land-cover classes: ONE definition for data preparation, training, evaluation and the prototype.

6 classes (decided 2026-11-03), merged from the 9 Dynamic World classes where C-band radar cannot tell them apart
or the labels are least reliable:

    id  class            from Dynamic World                         why
    0   water            water
    1   trees            trees
    2   grass & shrub    grass + shrub & scrub + flooded vegetation  radar cannot separate grass and shrub;
                                                                    flooded vegetation has the worst labels
    3   crops            crops                                      kept: harvests show in the monthly change
    4   built-up         built
    5   bare & snow      bare + snow & ice                          snow is rare (< 3 %) and mostly seasonal

ESRI 10 m land cover (the independent check) maps onto the same 6 classes: its "rangeland" = grass + shrub.
"""

from __future__ import annotations

import numpy as np

CLASS_NAMES = ["water", "trees", "grass_shrub", "crops", "built", "bare_snow"]
DISPLAY_NAMES = ["water", "trees", "grass & shrub", "crops", "built-up", "bare & snow"]
CLASS_COLOURS = ["#419bdf", "#397d49", "#88b053", "#e49635", "#c4281b", "#a59b8f"]
N_CLASSES = len(CLASS_NAMES)
WATER, TREES, GRASS_SHRUB, CROPS, BUILT, BARE_SNOW = range(N_CLASSES)

# Dynamic World id (0 water, 1 trees, 2 grass, 3 flooded vegetation, 4 crops, 5 shrub & scrub, 6 built, 7 bare,
# 8 snow & ice) -> class id
DW_NAMES = ["water", "trees", "grass", "flooded_vegetation", "crops", "shrub_and_scrub", "built", "bare", "snow_and_ice"]
DW_TO_CLASS = np.array([WATER, TREES, GRASS_SHRUB, GRASS_SHRUB, CROPS, GRASS_SHRUB, BUILT, BARE_SNOW, BARE_SNOW],
                       dtype=np.int8)
DW_BUILT, DW_SNOW = 6, 8

# ESRI 10 m annual land cover codes -> class id (-1 = clouds / unknown). Rangeland = former grass + scrub.
ESRI_ORIGINAL = {1: WATER, 2: TREES, 4: GRASS_SHRUB, 5: CROPS, 7: BUILT, 8: BARE_SNOW, 9: BARE_SNOW, 10: -1,
                 11: GRASS_SHRUB}
ESRI_REMAPPED = {1: WATER, 2: TREES, 3: GRASS_SHRUB, 4: CROPS, 5: BUILT, 6: BARE_SNOW, 7: BARE_SNOW, 8: -1,
                 9: GRASS_SHRUB}


def from_dw(lbl: np.ndarray) -> np.ndarray:
    """Dynamic World ids (-1 = no label) -> class ids (-1 kept)."""
    out = DW_TO_CLASS[np.clip(lbl, 0, 8)]
    return np.where(lbl >= 0, out, -1).astype(np.int8)
