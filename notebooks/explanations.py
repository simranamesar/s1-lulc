"""Long explanations for the report notebooks: what each step does, why we did it that way, how it works,
how to read the output and what we learned. build_notebooks.py inserts them before / after the matching cells.

Each entry: (notebook, anchor, position, markdown)
    notebook  "NB1" ... "NB4"
    anchor    a piece of text that occurs in exactly the cell the explanation belongs to (first match is used)
    position  "before" or "after" that cell
"""

# ============================================================================================ 00 read me first
NB0 = [
"""md:# 0 · Read me first: the whole project in one page

**Goal.** Type in any coordinates on Earth and see **how much was built in one year**, from free satellite radar
alone. The model makes a land-cover map for every month (water, trees, grass & shrub, crops, built-up, bare & snow),
and the prototype measures how the built-up area changed.

**Why radar (Sentinel-1) and not normal photos (Sentinel-2)?** Optical satellites cannot see through clouds. In the
tropics (Lagos, Dhaka, the Amazon) some months have almost no cloud-free image, so monthly change cannot be measured.
Radar sends its own microwave signal and receives the echo: it works through clouds, at night, in every season. The
price: radar images do not look like photos, and some classes (grass vs crops vs bare soil) are hard to separate.

**The four notebooks**

| Notebook | Question it answers | Main files |
| --- | --- | --- |
| 01 Data | Where do the training examples come from, what do the input and the labels look like, how balanced and how noisy are they? | `data/*.py`, `data/tiles_v0/` |
| 02 Training & evaluation | Which network, which training choices, how good is it on places it never saw, which variant is best? | `train/*.py`, `train/runs/` |
| 03 Case studies | Does the prototype give sensible answers in real places, including hard ones? | `prototype/*.py`, `prototype/case_studies/` |
| 04 Training step by step | What happened during training, epoch by epoch and batch by batch, and how did the output maps change? | `train/runs/*/history.csv`, `steps.csv`, `snapshots/` |

**The pipeline in one picture**

```
 1 choose places            600 places of 2.56 km x 2.56 km, stratified over the world      data/sample_tiles.py
 2 download labels          monthly Dynamic World maps + Open Buildings roofs (2023)       data/export_tiles.py
 3 download radar           EVERY Sentinel-1 scene of 2023 over each place (~35,000)       data/export_scenes.py
 4 clean + align            6 classes, seasonal smoothing, everything on the same 10 m grid data/prepare_data.py
 5 balance                  each region / settlement type / place gets a fair share         train/balance.py
 6 train                    U-Net (ResNet34 encoder), one radar scene -> one map            train/train.py
 7 evaluate                 130 test places never seen in training, 11 variants compared   train/eval_*.py
 8 prototype                any point -> all scenes of 15 months -> 12 monthly maps         prototype/pipeline.py
                            -> class % and change -> built-up change -> checked against     prototype/app.py
                            two independent maps (ESRI land cover, Open Buildings Temporal)
```

**The most important decisions (and why)**

1. **Single raw scenes as training samples, not monthly averages.** Each radar scene is one example, so the
   model sees ~35,000 examples instead of 600 × 12 = 7,200, and it learns to cope with day-to-day radar noise. In the
   prototype, all scenes of a month are predicted and their probabilities averaged.
2. **Places stratified over the world.** If the training data were 20 German cities, the model would fail on African
   villages. Places were drawn by region × biome × settlement type × land cover, and the balanced sampler gives each
   region and settlement type an equal share of training.
3. **Spatial test split.** Test places lie in other 5° × 5° blocks than training places, so the test score is what a
   new, unseen place would get, not a neighbour of a training place.
4. **Hybrid, smoothed labels.** Dynamic World misses small houses and flickers between trees and shrub with the
   seasons; adding building footprints and smoothing over 3 months makes the labels cleaner (flicker 15.8 % → 9.8 %).
5. **6 classes, not 9.** Radar cannot tell grass from shrub; flooded vegetation and snow are rare and badly labelled.
6. **Ensemble + 3-month smoothing + a persistent-change rule + same-season comparison.** Each step removes false change:
   the month-to-month flicker of the maps fell from 25 % (one scene, one model) to 5.9 %.

**Headline results (130 unseen test places)**

| Setting | mIoU | Month-to-month flicker |
| --- | --- | --- |
| One model, one scene per month | 0.487 | 25 % |
| One model, mean of all scenes of the month | 0.522 | 15 % |
| Ensemble of 4 models, month mean | 0.544 | 12 % |
| **Ensemble + 3-month smoothing (prototype)** | **0.549** | **5.9 %** |

Built-up is the best class (F1 ≈ 0.84); grass & shrub vs crops vs bare are the hardest. Small village houses
are under-counted because a house of 8 m is smaller than one radar pixel.""",
"""md:## Glossary

**Radar and the input**

* **Sentinel-1**: two European radar satellites (C-band, wavelength 5.6 cm). In 2023 only Sentinel-1A was working
  (1B failed in Dec 2021), so a place is seen roughly every 6–12 days, from ascending and descending orbits. Where
  orbits overlap (much of Europe) there are many more scenes.
* **IW (Interferometric Wide swath)**: the standard mode over land, 250 km wide, 10 m pixels.
* **GRD**: Ground Range Detected, the processed image product (intensity only).
* **Backscatter (σ⁰, sigma-nought)**: how much of the radar signal comes back. Smooth surfaces (calm water, smooth
  sand, roads) reflect it away: dark. Rough surfaces and vegetation scatter some back: grey. Buildings act like corner
  reflectors (wall + ground, "double bounce"): very bright.
* **dB (decibel)**: backscatter is stored as 10·log10(σ⁰). Typical values: water −25 dB, forest −7 dB (VV),
  cities up to +5 dB.
* **VV, VH (polarisation)**: the radar sends a vertically polarised wave; VV = the vertical part of the echo, VH = the
  horizontal part. VH is created mainly by volume scattering (leaves, branches), so it is strong for vegetation.
  **VH − VV** (in dB = their ratio) is a third input channel that separates vegetation types.
* **Terrain flattening (γ⁰, gamma-nought)**: slopes facing the satellite look brighter than they are, slopes facing
  away look darker. Flattening divides by the illuminated area computed from a terrain model (Copernicus DEM, 30 m),
  so a forest on a slope looks like a forest on flat ground. Layover and shadow (steep terrain the radar cannot see)
  are masked.
* **Speckle**: grainy "salt and pepper" noise in every radar image, caused by interference between many small
  scatterers in one pixel. Reduced by averaging (several scenes, or a speckle filter such as the Lee filter).

**Labels and data**

* **Dynamic World**: Google / WRI land-cover maps from Sentinel-2 (optical), 9 classes, 10 m, for every cloud-free
  image. We take the most frequent class per pixel per month.
* **Open Buildings**: building footprints from high-resolution imagery (Africa, South & Southeast Asia, Latin
  America). Used to mark pixels with ≥ 15 % roof cover as built-up ("hybrid" label).
* **Tile / place**: a 256 × 256 pixel square (2.56 km × 2.56 km).
* **Scene / sample**: one radar acquisition over one place = one training example.
* **Train / validation / test split**: train = the network learns from it; validation = checked after every epoch
  to choose the best epoch; test = used once at the end, for the reported numbers.

**Network and training**

* **CNN (convolutional neural network)**: layers of small filters (3 × 3) that slide over the image and learn patterns:
  edges first, then textures, then objects.
* **Semantic segmentation**: a class for every pixel (not one label per image).
* **U-Net**: an encoder shrinks the image step by step (256 → 8 pixels) to understand context; a decoder enlarges it
  back to 256 to draw the map; skip connections pass fine details from the encoder to the decoder.
* **ResNet34**: the encoder; 34 layers with "residual" shortcuts that make deep networks trainable.
* **Transfer learning / ImageNet**: the encoder starts with weights learned on 1.2 million photos (ImageNet) instead
  of random numbers, then is fine-tuned on radar.
* **Logits → softmax → argmax**: the network outputs 6 scores per pixel (logits); softmax turns them into
  probabilities that sum to 1; argmax picks the class with the highest probability.
* **Loss**: the number training minimises. **Cross-entropy** punishes a low probability for the correct class;
  **Dice** measures the overlap of predicted and true areas per class (helps small classes).
* **Class weights**: rare classes count more in the loss so the network does not ignore them.
* **Epoch**: one pass through the training samples. **Batch**: 32 samples processed together; after each batch the
  weights are updated once (one **step**).
* **Optimizer (AdamW)** and **learning rate**: how big each weight update is. We start at 0.001 and lower it along a
  cosine curve.
* **Overfitting**: the network memorises the training places instead of learning general rules; training scores go
  up, validation scores stop improving. **Early stopping** ends training 10 epochs after the last improvement.
* **Augmentation**: random rotations and flips of training samples, so the network sees more variety.
* **Mixed precision (bf16)**: computing with 16-bit numbers where safe, about twice as fast on the GPU.

**Metrics**

* **TP / FP / FN**: true positive (correctly found), false positive (wrongly claimed), false negative (missed).
* **Overall accuracy (OA)**: share of pixels classified correctly. Can look good even if a rare class is never found.
* **Precision** = TP / (TP + FP): when the model says built-up, how often is it right?
* **Recall** = TP / (TP + FN): how much of the real built-up does it find?
* **F1** = 2 · precision · recall / (precision + recall).
* **IoU (intersection over union)** = TP / (TP + FP + FN): overlap of predicted and true area for one class.
  **mIoU** = the mean IoU over the 6 classes: the main score for segmentation (every class counts equally).
* **Confusion matrix**: rows = true class, columns = predicted class; the diagonal is correct, the rest shows which
  classes are mixed up.
* **Flicker / flip rate**: share of pixels whose class changes from one month to the next. Real land-use change is
  rare (< 1 % a year), so most flicker is error.

**Prototype**

* **Month mean**: the class probabilities of all scenes of a month averaged, then argmax.
* **Ensemble**: probabilities of 4 differently trained models averaged.
* **TTA (test-time augmentation)**: predict flipped copies too and average (tested, little gain).
* **smooth3**: each month's probabilities averaged with the previous and next month.
* **Same-season comparison**: built-up change = first 3 months vs the same 3 calendar months one year later.
* **Change rules**: *argmax* (built in most end months, in none of the start months), *moderate* and *confident*
  (built-up probability ≥ 0.5 at the end and below 0.35 / 0.2 in every start month).
* **Error level (noise floor)**: how much false new built-up the model reports on unseen test places per km² and year;
  a result is "clear growth" only at ≥ 3 × this level.""",
"""md:## How to read the notebooks

* Every section starts with a short title cell and is followed by **"How and why"** cells with this structure:
  **What** this step does · **Why** we did it this way · **How** it works (the code, in words) · **Reading the
  output** · **What we learned** (including what did not work).
* Code cells only read files that the scripts produced; notebooks 01, 02 and 04 run on the university server
  (where the data and the runs are), notebook 03 on any machine with the `prototype/case_studies/` files.
* Every attempt, what failed and why is also in the **Project Log** (Claude Docs).""",
]


# ============================================================================================ explanations
E = []


def add(nb, anchor, md, pos="after"):
    E.append((nb, anchor, pos, "md:" + md.strip("\n")))


# ------------------------------------------------------------------------------------------- NB1 data
add("NB1", "# 1 · Data: where, when, what", """
### How to read this notebook

The model can only be as good and as general as its training data, so this notebook is about the data, in the order
it was built: **which places** (1.1), **which radar images** (1.2), **which classes** (1.3), **how the data are
balanced** (1.4), **how good the labels are** (1.5), **one place in pictures** (1.6) and **every processing step on
one example** (1.7). Each step is explained in a "How and why" cell: what, why, how, how to read the output, what we
learned. Words in *italics* are in the glossary of notebook 00.""")

add("NB1", "tiles = pd.read_csv(DATA / \"tiles.csv\")", """
### How and why: choosing the places

**What.** The table lists how many of the 600 places lie in each world region and settlement type (city, town,
village, none), and how they are divided into train / validation / test.

**Why.** The project requirement: the model must work *anywhere*, not only where the training data came from. A model
trained on 20 German cities would learn "European roofs and European fields" and fail on mud-brick villages in Niger.
So the places must cover all continents, all climate zones (biomes), cities and empty land.

**How.** `data/sample_tiles.py` draws random points in Google Earth Engine, **stratified**: the world is cut into
groups (region × biome × settlement type × land cover) and points are drawn from each group, so rare groups (desert
towns, tropical cities) are not left out. Settlement type comes from the GHSL settlement model (EU JRC), biome from
RESOLVE Ecoregions, land cover from ESA WorldCover. Random points almost never fall into a city (cities cover < 1 %
of land), so 300 extra points were drawn directly inside GHSL towns and cities (cities 57 → 120).

**Splits.** Places are assigned to train / validation / test by **5° × 5° blocks** (about 550 km), not one by one.
Neighbouring places look alike (same crops, same building style); if the test place were next to a training place,
the test score would be too optimistic. With blocks, every test place is in a region the network has not seen.

**Reading the output.** Rows = regions, columns = settlement types, "All" = totals. The second table shows that each
settlement type appears in all three splits (409 train, 61 validation, 130 test places).

**What we learned.** The first version (300 places, random only) had too few cities; the learning curve (E1half,
notebook 02) showed that more places help (+2.4 mIoU points from doubling), so the dataset was doubled to 600.""")

add("NB1", "ax.set_title(\"The 600 places", """
**Reading the map.** Each dot is one place of 2.56 km × 2.56 km; colour = settlement type; squares / triangles mark
validation / test places. The places spread over every inhabited continent; deserts, rainforest, mountains and coasts
are included on purpose because they are the hard cases for radar. The biome table below counts places per biome (14
biomes from tundra to tropical forest).""")

add("NB1", "print(\"Recipe:\", s1_composite.PREPROCESSING)", """
### How and why: the radar input

**What.** For every place, every Sentinel-1 scene of 2023 was downloaded: about 58 per place on average, 34,981 in
total, each 256 × 256 pixels with 3 channels (VV, VH, VH − VV in dB).

**Why single scenes and not monthly averages?** (1) Many more training examples (35,000 instead of 7,200).
(2) Every scene has its own speckle and moisture, so the network learns to ignore day-to-day noise. (3) The prototype
can then use any number of scenes per month. An earlier version trained on monthly median composites (Project Log);
single scenes gave more stable results.

**How (the recipe).** In Earth Engine (`data/s1_composite.py`):
1. *S1_GRD* scenes in IW mode with both VV and VH, over the place, in 2023;
2. border noise masked (the scene edges are unreliable);
3. **terrain flattening** to γ⁰ with the Copernicus 30 m terrain model (Vollrath et al. 2020): removes the bright /
   dark effect of slopes; layover and shadow (steep terrain the radar cannot see) masked as *no data*;
4. conversion to dB, VH − VV computed, stored as whole numbers (dB × 100) to save space; −32768 = no data.

Slices of the same satellite pass (same date and orbit) are merged into one scene. Scenes covering < 50 % of the
place are dropped.

**Normalisation.** The network works best with inputs around 0 with spread 1, so each channel is standardised:
(value − mean) / std, with mean and std computed **from the training scenes only** (using test data here would leak
information). The printed numbers are these statistics.

**Reading the plots.** Left: scenes per place by region: Europe has up to ~180 (overlapping orbits), most other
places ~30. Right: scenes per split.

**What we learned.** Because Europe has 5× more scenes per place, a plain average over scenes would let Europe
dominate both training and test scores. So training draws and test scores are weighted **per place**, not per scene
(Project Log, iteration 17). Exporting dense European places first failed with Earth Engine memory errors; it was
fixed by reading the list of scenes first and building one scene at a time.""")

add("NB1", "from classes import DW_TO_CLASS", """
### How and why: 6 classes instead of 9

**What.** Dynamic World has 9 classes; we merge them into 6 (table above).

**Why.** A class should only exist if the input can distinguish it. C-band radar sees *structure and moisture*, not
colour:
* **grass + shrub & scrub + flooded vegetation → grass & shrub**: grass and low shrub give the same backscatter;
  flooded vegetation is rare and its labels are the least reliable;
* **bare + snow & ice → bare & snow**: snow is < 3 % of pixels and mostly seasonal; outside cold biomes, snow labels
  are removed altogether (they are usually cloud or bright sand misread by Dynamic World);
* **crops** stay separate: ploughing, growth and harvest change the backscatter month by month, which is useful;
* **built-up** stays: it is the class the prototype is about.

**Reading the output.** Class shares per split: grass & shrub, trees and crops dominate; built-up is a few percent,
water and bare less. The splits have similar mixes, so the test set represents the same kind of world as training.

**What we learned.** With 9 classes the network confused grass and shrub most of the time and the scores of those
classes were near zero; merging gave a fairer and more useful map.""")

add("NB1", "md = PREP / \"balance_report.md\"", """
### How and why: balancing what the network sees

**What.** The *balanced sampler* decides how often each training scene is drawn. The report shows the class shares
and the region / settlement shares before and after balancing.

**Why.** Without it the network would mostly see what is common in the data: European scenes (most scenes), rural
places (most places) and grass & trees (most pixels). It would get good at those and poor at African towns or
built-up areas, which are exactly what the prototype must find.

**How** (`train/balance.py`, three steps):
1. **class weight per scene**: scenes containing rare classes (built-up, water, bare) get a higher weight
   (inverse class share to the power 0.5, a mild boost);
2. **each place counts equally**: the weight is divided by the place's number of scenes, so 180 European scenes
   together count as much as 30 African ones;
3. **raking** (iterative proportional fitting): weights are scaled back and forth until every world region **and**
   every settlement type gets the same total share; extreme weights are capped at 10 × the median so the network
   does not see the same few scenes over and over.

The loss additionally gets mild **class weights** computed from what the sampler actually delivers.

**Reading the output.** "uniform" = every scene equally likely; "balanced" = what the network sees. Built-up and
water go up, grass and trees go down; every region gets 1/7 and every settlement type 1/4 of the draws. The
*effective sample size* (notebook 04) says how many independent scenes the weighted draw is worth: if it were very
small, the network would keep seeing the same scenes.

**What we learned.** The balance report flagged no warnings for the final 600 places.""")

add("NB1", "flip = ok[[\"flip_dw\", \"flip_hyb\", \"flip_hyb_s\"]]", """
### How and why: cleaning the monthly labels

**What.** The first table: the share of pixels whose label changes from one month to the next, for three label
versions. The second: how many built-up pixels the building footprints add, per settlement type.

**Why it matters.** The prototype measures *change*. If the labels themselves change class every month (clouds,
leaves falling, harvest), the network learns that flicker and the prototype reports change that did not happen.
Real land-use change is below 1 % of pixels per year; Dynamic World flips 15.8 % of pixels every month.

**How.** Three steps (`data/prepare_data.py`):
1. **hybrid**: pixels where Open Buildings shows ≥ 15 % roof cover become built-up. Dynamic World is made from
   10 m optical images and misses small, scattered houses, especially in African villages;
2. **seasonal pair fix**: a pixel that is only ever "trees" or "grass & shrub" during the year (deciduous or dry
   forest that loses its leaves) gets its majority class in every month;
3. **3-month weighted vote**: each month's label = the most frequent class over the previous, current and next month
   (current month counts 1.5×). Single-month outliers disappear; changes that last stay.

**Reading the output.** Flicker 15.8 % (raw Dynamic World) → 9.8 % (hybrid smoothed): better, but still far above
real change, which is one reason the prototype needs its own smoothing and strict change rules. The building
footprints add the most built-up in villages and towns, where houses are small.

**What we learned.** Training on the smoothed hybrid label (E1) instead of raw Dynamic World (E0) gave the same
mIoU against the labels but steadier maps. An annual label (E1y: the yearly majority for every month) was tested
too: no better, and it cannot learn seasonal classes (notebook 02).""")

add("NB1", "cand = ok[(ok.split == \"test\") & ok.settlement.isin", """
### How to read the example place

Three rows × 12 months: the radar VV image of the scene used for that month, the raw Dynamic World label and the
smoothed hybrid label (the training target). Look for:
* **radar**: water and smooth surfaces dark, settlements bright and grainy; the grain is *speckle*;
* **Dynamic World**: grey "no label" months (clouds) and classes jumping between months;
* **hybrid smoothed**: the same areas, but steadier through the year and with more built-up where there are houses.""")

add("NB1", "from s1_composite import lee_filter_db", """
### How and why: speckle and what each class looks like to radar

**What.** Left: one raw scene (what the network gets). Middle: the same scene after a 5 × 5 Lee speckle filter.
Right: the distribution of VV backscatter for each class in this scene.

**Why look at this.** It shows *why* some classes are easy and others hard. If two classes have the same backscatter
distribution, no network can separate them from one pixel; it has to use texture and context (neighbouring pixels),
which is what the convolution filters of a CNN do.

**How the Lee filter works.** For each pixel it looks at the 5 × 5 neighbourhood: in flat areas (low variance) it
replaces the pixel with the local mean (removes speckle); near edges (high variance) it keeps the pixel (keeps
detail). It is computed on linear power, not dB.

**What we learned.** Water (dark, left) and built-up (bright, right) are well separated. Trees, grass & shrub, crops
and bare overlap strongly, so they are the classes the model confuses (confusion matrix, notebook 02). The Lee
filter makes the image look cleaner but **did not improve the model** (variant E5: mIoU 0.493 vs 0.487, within
run-to-run noise, at 2.6× the processing time): a CNN already averages neighbouring pixels in its first layers, and
the filter also blurs small houses. Raw scenes were kept, and the speckle problem is solved over time instead (all
scenes of a month averaged in the prototype).""")

add("NB1", "print(\"STEP 0 · sample_tiles.py", """
### How and why: following one place through the whole pipeline

The next six cells show the **same place** after every script, so you can see exactly what each step changes:

| Step | Script | Output | What changes |
| --- | --- | --- | --- |
| 0 | `sample_tiles.py` | `tiles.csv` | a point + its region, biome, settlement type, split |
| 1 | `export_tiles.py` | `raw/<tile>.tif` | monthly labels downloaded on a 10 m UTM grid |
| 2 | `export_scenes.py` | `scenes/<tile>.npz` | every radar scene on exactly the same grid |
| 3 | `prepare_data.py` | (in memory) | labels cleaned: snow, 6 classes, buildings, smoothing |
| 4 | `prepare_data.py` | `prepared/<tile>.npz` | input + all label versions in one file |
| 5 | `balance.py`, `train.py` | tensors | normalised input + label + augmentation = a training sample |

**Why the same grid matters.** The label pixel (row 100, column 50) must describe exactly the ground under radar
pixel (100, 50). All downloads use the same UTM projection, the same 10 m pixel size and the same corner, computed
once per place, so nothing is resampled later.""")

add("NB1", "print(\"STEP 1 · export_tiles.py", """
**Reading step 1.** The GeoTIFF has one band per month and label version (e.g. `m07_DW` = July, Dynamic World;
`m07_HYB` = July, hybrid) plus static bands such as `BFRAC` (roof cover % from Open Buildings). The pictures show the
raw class ids (0–8, colours are only ids here), the hybrid label and the roof cover that turns pixels into built-up.""")

add("NB1", "print(\"STEP 2 · export_scenes.py", """
**Reading step 2.** One file per place with all its scenes: `s1` (scenes × 3 channels × 256 × 256), the date,
orbit direction (ascending / descending), relative orbit number and the share of the place covered (`valid`). The
printed 3 × 3 pixels show the storage trick: whole numbers = dB × 100, divided by 100 when read.""")

add("NB1", "print(\"STEP 3 · prepare_data.py: label cleaning", """
**Reading step 3.** Six versions of the same month: from raw Dynamic World ids to the smoothed hybrid label used
for training (E1) and the annual label (E1y). The table below follows the **most flickering pixel** of the place
through the 12 months: raw labels jump between classes; the smoothed version keeps one class or changes once.""")

add("NB1", "print(\"STEP 4 · prepared/<tile>.npz", """
**Reading step 4.** The prepared file contains everything training needs, aligned: `s1` (all scenes), `month` and
`day` of each scene, the label versions (`dw`, `hyb`, `hyb_s`, `hyb_y`; 12 months each), `n` = scenes per month,
`best` = the scene with the best coverage nearest mid-month (used for "one scene per month" evaluation).""")

add("NB1", "print(\"STEP 5 · balance.py + train.py", """
**Reading step 5.** `item_table` turns the place into training samples: one row per scene with enough labelled
pixels (≥ 20 %), with its month and class pixel counts (used for the sampling weights). `TileMonths` (the PyTorch
Dataset) turns one row into tensors: **x** = 3 × 256 × 256 floats (normalised: mean ≈ 0, std ≈ 1, as the table shows)
and **y** = 256 × 256 class ids, with −1 where there is no label or no radar data (these pixels are ignored by the
loss). The pictures show **augmentation**: during training each sample is randomly rotated by 0/90/180/270° and maybe
flipped; the label is rotated with it. A field looks the same from any direction, so this adds variety for free.
(Radiometric augmentation, random brightness shifts, was also tried as E6 and made results worse; notebook 02.)""")

# ------------------------------------------------------------------------------------------- NB2 training
add("NB2", "# 2 · Model, training and evaluation", """
### How to read this notebook

1. **The network** (2.1): its layers and one prediction traced from input to map.
2. **The variants** (2.2): 11 trainings, each changing one thing, to answer one question each.
3. **Training logs and curves** (2.3–2.4): how training went (notebook 04 shows this in much more detail).
4. **Test results** (2.5): scores on 130 places the network never saw, per class, region and settlement type.
5. **From scenes to monthly maps** (2.6), **learning curve** (2.7), **final setting** (2.8–2.9).
6. **Example predictions** (2.10).

An important rule throughout: **compare variants only on the same labels and the same test places**. Early in the
project two variants were compared against different labels, and a difference that was really in the labels looked
like a model improvement (Project Log).""")

add("NB2", "RUN_TRAINING = False", """
**About this cell.** Training is done with the script `train/train.py` on the server's GPU (each run 15–50 min), not
inside the notebook: a notebook would stop if the browser disconnects. The notebook reads the files the runs saved.
Setting `RUN_TRAINING = True` repeats E1 here to show that the script and the notebook give the same result.""")

add("NB2", "net = build_model(\"resnet34\", \"none\").eval()", """
### How and why: the U-Net

**What.** The table lists the size of the data after each block for one 256 × 256 input.

**How it works.**
* **Encoder (ResNet34)**: 4 stages of 3 × 3 convolutions; each stage halves the image size and increases the number
  of channels (feature maps): 256 → 128 → 64 → 32 → 16 → 8 pixels, 64 → 512 channels. Early layers detect edges and
  speckle texture; deep layers see a whole neighbourhood (an 8 × 8 map where each value "sees" ~250 m around it):
  "a bright, regular pattern surrounded by fields" = village.
* **Decoder**: 5 blocks, each doubles the size again (nearest-neighbour interpolation, or a learned transposed convolution in E3),
  concatenates the encoder features of the same size (**skip connections**: the fine details from early layers,
  which the deep layers have lost) and applies two 3 × 3 convolutions.
* **Head**: a 1 × 1 convolution turns the 16 decoder channels of each pixel into 6 class scores.

**Why a U-Net?** Segmentation needs both *context* (is this bright pixel a roof or a rock?) and *precision* (where
exactly is the edge of the town?). The encoder gives context, the skip connections give precision. It is the
standard architecture for satellite segmentation and works with small datasets.

**Why ResNet34 and ImageNet weights?** ResNet34 is deep enough to learn context but small enough (24 M parameters)
for 400 training places. ImageNet weights are a *transfer learning* start: the first layers already detect edges
and textures. Radar is not a photo, so the whole network is fine-tuned (all weights trained), not frozen. Our three
channels (VV, VH, VH − VV) replace the three colour channels (R, G, B), so the pretrained first layer fits.""")

add("NB2", "model, stats, cfg, device = load_model(RUNS / \"E1\")\nidx", """
### How and why: from input to map

**Reading the output.** Top row: the three normalised input channels, the label, the predicted map and the
confidence (highest probability; dark = the network is unsure). Bottom row: the probability of each class per pixel.
The small table shows one pixel: 6 **logits** (raw scores, any size), turned into **probabilities** by softmax
(exp(score) / sum of exp(scores): positive, sum to 1), and the class with the highest probability is the answer.

**What to notice.** The network is confident inside large uniform areas and unsure along borders and in mixed areas
(gardens between houses, fields with scattered trees). The prototype uses these probabilities, not only the final
class: for built-up change it requires a *high* built-up probability at the end and a *low* one at the start.""")

add("NB2", "ORDER = [\"E1\", \"E1s1\", \"E0\"", """
### How and why: the training setup common to all runs

| Setting | Value | Why |
| --- | --- | --- |
| Loss | weighted cross-entropy + Dice | cross-entropy trains every pixel; Dice optimises overlap per class, so small classes (built-up, water) are not drowned by large ones |
| Optimizer | AdamW, learning rate 0.001, weight decay 0.0001 | Adam adapts the step size per weight; weight decay keeps weights small (less overfitting) |
| Schedule | 1 warm-up epoch, then cosine decay to 0 | a gentle start protects the pretrained weights; small steps at the end fine-tune |
| Batch | 32 scenes | as large as GPU memory allows, smoother updates |
| Precision | bfloat16 mixed precision | about 2× faster, no accuracy loss |
| Augmentation | random 90° rotations + flips | fields and towns have no "up" |
| Epochs | up to 40, early stopping after 10 without improvement | stops before overfitting; the best epoch's weights are kept |
| Sampler | balanced (notebook 01, 1.4) | every region and settlement type equally |
| Labels ignored | pixels without label or radar data | the loss is computed only where the truth is known |

**How the variants were chosen.** Each variant changes **one** thing compared to E1, so a difference can be
attributed to that change (a controlled experiment). Second runs with another random seed (E1s1, E3s1) measure
how much results differ by chance alone; only differences larger than that count.""")

add("NB2", "log = ROOT / \"train\" / f\"runs_{r}.log\"", """
**Reading the logs.** Each run first prints its setup (scenes, sampler, model size, class weights), then one line per
epoch: `loss train/val`, `mIoU train/val`, accuracy, seconds, and `*best` when validation mIoU improved (that
checkpoint is saved), and at the end the test results. **Notebook 04 turns these logs into tables and charts.**""")

add("NB2", "ncol = 6\nnrow", """
**How to read training curves.** Top: loss (lower = better); bottom: mIoU (higher = better); blue = training,
orange = validation.
* Both falling / rising together = the network is learning general rules.
* Training still improving while validation is flat or worse = **overfitting** (memorising training places).
* The dotted line = best validation epoch; that checkpoint is used for testing (early stopping).
* Validation loss can rise while validation mIoU still improves: the loss also punishes *confident* mistakes,
  mIoU only counts right / wrong.

**What we learned.** All runs reach their best validation mIoU after 11–24 epochs; training mIoU keeps rising
afterwards, so without early stopping the models would overfit. The frozen encoder (E2a) stops earliest and scores lowest on the test places (0.462): radar
features differ from photo features, so the encoder must adapt.""")

add("NB2", "rows = []\nfor r in runs:\n    m = json.loads", """
### How and why: the test results

**What.** Each run's best checkpoint was evaluated once on the 130 test places, **every scene** against its month's
label (each place weighted equally). Per-class IoU below.

**Reading the columns.** OA = share of correct pixels; mIoU = mean IoU over 6 classes (the main score); built P / R =
built-up precision and recall; village built R = how much village built-up is found; flip rate = month-to-month
change of the predicted maps (one scene per month); best epoch.

**Careful.** Here each run is scored against **its own** training label (E0 against raw Dynamic World, E1 against
smoothed hybrid, ...), so the rows are not directly comparable. Section 2.6 compares all runs on the same labels.

**What we learned.** Single-scene mIoU is around 0.48–0.50 for all good variants: the differences are within the
run-to-run noise measured with second seeds (± 0.01 overall, ± 0.05 per region). The IoU per class shows the pattern
of notebook 01: water and built-up best, grass & shrub / crops / bare lowest.""")

add("NB2", "MAIN = \"E1\"\nm = json.loads", """
### How and why: confusion matrix, regions, settlement types

**Confusion matrix** (left): row = true class, column = predicted class, each row sums to 1. The diagonal is the
recall of each class. Off-diagonal cells show which classes are mixed up, e.g. grass & shrub ↔ crops ↔ trees: these
are radar-specific confusions (similar structure and moisture), not a bug.

**Per region and settlement type** (middle, right): this answers *"does it work on places unlike the training
data?"*. Every test place lies in a 5° block without training places, so these are all new places. Differences
between regions come from the landscapes (deserts and mountains are hard for radar) and from label quality (Dynamic
World is worse in the tropics, where clouds are frequent).

**Classification report** (table): precision, recall and F1 per class, the same format as
`sklearn.metrics.classification_report` in the lab exercises; *support* = number of labelled test pixels.""")

add("NB2", "RUN_MISSING_EVAL = True", """
### How and why: from single scenes to 12 monthly maps

**What.** The prototype needs one map per month, but each month has 2–15 scenes. Four ways to combine them were
compared (`train/eval_change.py`):
* **A best scene**: only the scene with the best coverage nearest mid-month;
* **D best scene, one orbit**: the same, but always from the same orbit direction (radar looks from a different side
  on ascending and descending passes);
* **B month mean**: class probabilities of *all* scenes of the month averaged, then argmax;
* **C month mean, one orbit**.

**Why averaging helps.** Each scene has its own speckle, soil moisture and look direction; their errors are partly
random, so averaging cancels part of them. It is the same reason an ensemble of models helps.

**Reading the tables.** The first table: E1 with the four methods. The second: **all runs with method B against the
same label (hyb_s)**: this is the **fair comparison** of the variants. Columns: flip_rate = monthly flicker;
built_false_change = how often built-up status flips where the label is stable; false_new_ha_km2 = new built-up
reported per km² per year that is not new in the labels.

**What we learned.** Month mean (B) is clearly best: mIoU 0.487 → 0.522 and flicker 25 % → 15 % for E1. Restricting to
one orbit did not help. On the same labels, E1 / E3 / E3s1 / E5 are within ~0.01 of each other: no single training
trick is a big win; how the scenes are combined matters much more.""")

add("NB2", "if \"E1half\" in ce and \"E1\" in ce", """
### How and why: is the dataset big enough?

**What.** E1half is E1 trained with only half of the training places (same share in every region × settlement type;
validation and test unchanged).

**Why.** If half the data gives the same result, more data would not help and effort should go elsewhere. If half
the data is clearly worse, more places would probably help further.

**What we learned.** Halving the places cost about 2.4 mIoU points (0.522 → 0.498 month mean). So the dataset was
doubled from 300 to 600 places; more places would likely help again. A fellow student's project found the opposite
for *more years of the same places*, consistent with this: **variety of places** matters more than more images of
the same places.""")

add("NB2", "f = RUNS / \"ensemble_eval.csv\"", """
### How and why: the final setting

**Ensemble.** The probabilities of 4 models (E3, E1, E5, E2b) are averaged. They were trained with different
settings, so they make partly different mistakes, which average out: +2 mIoU points, less flicker. Cost: 4× the
computing time, acceptable for a 5 km area.

**TTA (test-time augmentation).** Each scene is also predicted flipped (horizontally, vertically, both) and the
results averaged. Almost no gain at 4× the time: the month mean already averages many scenes. Not used.

**Change rules.** *argmax*: a pixel is new built-up if it is built-up in most of the last 3 months and in none of the
first 3. *confident*: the mean built-up probability at the end ≥ 0.5 **and** below 0.2 in every start month.
Patches smaller than 4 pixels (400 m²) are dropped in both. "false new" = new built-up not confirmed by the labels.

**What we learned.** The confident rule removes almost all false new built-up (≈ 0.005 ha/km² per year on test
places instead of ≈ 0.4 with argmax), but in the gallery (notebook 03) it also missed obvious growth (Lagos 0.5 ha).
A *moderate* rule (below 0.35 instead of 0.2 at the start) reported ~0 ha in the stable places and tens of hectares
where both references see growth, so it became the headline rule; the app shows confident and argmax as the lower and
upper bound.""")

add("NB2", "if all((RUNS / r / \"change_eval.csv\").exists() for r in [\"E1\", \"E1s1\"", """
### How and why: round 2 (run-to-run noise, E6, smoothing)

**Second seeds (E1s1, E3s1).** Training has randomness: the order of samples, augmentation, the random start of the
decoder. Training the same setting twice shows how much results vary by chance: about ± 0.01 mIoU overall and up to
± 0.05 per region. Earlier claims like "E3 has 25 % less false change" turned out to be partly this noise.

**E6 (regularisation).** E3 + Dropout 0.2 + random brightness shifts of ± 1.5 dB per scene + 0.5 dB pixel noise +
Lovász loss (a direct IoU optimiser). Idea: make the network robust to soil moisture and rain. Result: worse (mIoU
0.501 vs 0.522–0.527), especially bare & snow (IoU 0.36 → 0.26). Lesson: the absolute backscatter level carries class
information (dry sand is very dark, wet soil brighter); shifting brightness at random destroys that information.

**smooth3.** Each month's probabilities averaged with the previous and next month (not wrapped around the year).
Flicker 11.6 % → 5.9 % with slightly higher mIoU (0.544 → 0.549). Real land-use change lasts, so a 3-month window
removes mostly noise. This is the setting the prototype uses.""")

add("NB2", "from eval_change import scene_probs, monthly_maps", """
### How to read the example predictions

One test place per settlement type (city, town, village, none). Columns: radar VV, label, prediction from **one**
scene, prediction from the **mean of all scenes** of that month. The month-mean map is smoother and closer to the
label, especially at class borders; the single-scene map has scattered wrong pixels (speckle). Remaining errors are
mostly between grass & shrub, crops and bare, and small houses in villages that are missed (a single-storey house
of 8 × 8 m covers less than one 10 m pixel, and Sentinel-1's real resolution is about 20 m).""")

# ------------------------------------------------------------------------------------------- NB3 case studies
add("NB3", "# 3 · Prototype case studies", """
### How the prototype works, step by step

1. **Area.** The clicked point becomes the centre of a 512 × 512 pixel square (5.12 km) in the local UTM projection,
   the same 10 m grid recipe as the training data.
2. **Download.** Every Sentinel-1 scene of the 12 months + 3 more months is downloaded with exactly the training
   recipe (`data/s1_composite.py` is shared by training and prototype, so the input looks identical).
3. **Predict.** Every scene goes through the 4 models (ensemble); class probabilities are averaged over models and
   over the scenes of each month, then over neighbouring months (smooth3); argmax gives the monthly maps.
4. **Statistics.** % of area per class per month, month-to-month change and total change in percentage points.
5. **Built-up change.** First 3 months vs the same 3 months a year later, with the moderate rule (and the other two
   as bounds).
6. **Verdict.** New built-up compared with the error level (what the model reports as false new built-up on test
   places of the same size): ≥ 3× = clear growth, 1.5–3× = some growth, below = no change detectable.
7. **Independent checks.** ESRI 10 m land cover (two years) and Open Buildings Temporal (yearly building presence),
   neither used in training.

**Why the same-season comparison?** In a first version, built-up change compared the first 3 with the last 3 of the
12 months, i.e. different seasons. In Dhaka (monsoon floods) or Nairobi (rainy vs dry season) the radar signal changes
with the season, and the argmax rule reported hundreds of hectares of "new built-up" that were really wet vs dry
soil. Comparing July–September with July–September one year later removes the season from the comparison.

**Why a gallery?** The places were computed in advance (`case_studies.py`) so the app can show them instantly in
the screen recording; a live run of a new place takes 2–5 minutes.""")

add("NB3", "summary = []\nfor res in cases:", """
### How to read the summary table

* **scenes**: radar scenes used (more = steadier maps; Europe ~150, Africa ~30 per year);
* **compares**: same season (normal) or within year (fallback when the extra months are missing);
* **built-up start / end %**: share of built-up at the start and end period;
* **new ha (moderate / argmax / confident)**: new built-up area with each rule; moderate is the headline,
  argmax the upper bound, confident the strict lower bound;
* **error level ha**: false new built-up expected here from the test-set error rate;
* **verdict**: new built-up compared with the error level;
* **OBT / ESRI new ha**: new built-up according to the two independent products.

**Careful when comparing with the references.** They disagree with each other too (e.g. Dhaka: ESRI 286 ha vs Open
Buildings 25 ha; Kasoa: 0.1 vs 195 ha). ESRI is a 10 m land-cover map with its own yearly noise; Open Buildings
Temporal detects buildings from higher-resolution images but only in some regions. Agreement with *either* is
evidence, not proof.""")

add("NB3", "def month_strip(res):", """
### How to read each place

1. **Verdict** with the explanation (how many hectares, how many times the error level).
2. **12-month strip**: radar VV (the best scene of each month, with the number of scenes used), the monthly map and
   the built-up probability (black = 0, yellow = 1). Look for what changes through the year: crops and grass with
   the seasons (real seasonal change, not land use), water with floods, built-up should be stable.
3. **Animation** of the monthly maps.
4. **Year majority** (the most frequent class per pixel), the **built-up change map** (magenta new, cyan lost, dark
   grey built-up throughout) and the **class % per month**.
5. **Table**: our built-up change next to the two references.

The places include hard cases on purpose (Mekong rice fields, Swiss Alps, Sahara dunes, Niamey villages): a demo
that only shows easy places would hide where the method fails.""")

# ------------------------------------------------------------------------------------------- NB4 progress
add("NB4", "# 4 · Training, step by step", """
### Why this notebook

The scores in notebook 02 are the end result. This notebook shows **the process**: how fast the networks learn, when
they start to overfit, which classes come first, and how a predicted map develops from random to final. It also
connects every number printed during training to a chart, so the text logs and the pictures can be read side by
side.""")

add("NB4", "def grab(pat, t, f=float):", """
**Reading the table.** One row per run. *train / val scenes*: samples available; *effective sample size*: with
weighted drawing, some scenes are drawn often and some rarely; this is the number of equally weighted scenes the
draw is worth (higher = more variety); *parameters*: 24.4 M for the U-Net (*trainable* is less for E2a, frozen
encoder); *epochs run*, *best epoch*, *best val mIoU*, *early stopped*; *time*. The class weights show how much
each class counts in the loss; the bar chart compares the class mix of plain drawing with what the balanced sampler
delivers.""")

add("NB4", "RUN = \"E3\" if \"E3\" in runs else runs[0]", """
**Reading the epoch table.** One row = one printed `ep` line. train / val loss: lower is better; train / val mIoU
and OA: higher is better; lr: learning rate in that epoch; seconds per epoch. The ★ marks epochs where validation
mIoU reached a new best: those weights were saved as `best.pt`. After the last ★, 10 more epochs ran without
improvement, then early stopping ended the run.""")

add("NB4", "def heat(df, title, ax, vmin=0, vmax=0.8):", """
**Reading the heatmaps.** Look along a column to compare runs (which variant is best for built-up?) and along a row
to see a run's strengths and weaknesses. Regions with few test places (a few in Oceania, for example) vary more
between runs: a difference of 0.05 there can be chance.""")

add("NB4", "if WATCH is None:", """
**How the snapshots are made.** At the start of training, `train.py` picks 6 validation scenes (one per settlement
type, different regions, each with several classes) and saves their input and label (`snapshots/examples.npz`).
After every epoch it predicts these 6 scenes with the current weights and saves the map, the confidence and the
built-up probability (`snapshots/ep001.npz`, ...). Validation scenes are never trained on, so the snapshots show
how the network does on unseen data.""")
