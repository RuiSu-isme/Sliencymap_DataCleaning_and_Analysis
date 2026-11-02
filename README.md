# Saliency metrics — NSS and AUC-Judd

This folder scores per-frame saliency maps against recorded eye-tracking data.

**You do not need the model, the weights, a GPU, or PyTorch.** The metric
scripts only read images and CSV files.

The full pipeline is:

```
raw eye tracking  --(1) resample_gaze.py-->  one gaze row per video frame
stimulus video    --(2) a saliency model -->  one saliency map per video frame
gaze + maps       --(3) run_metrics.py   -->  NSS and AUC-J
```

This README teaches step 1 first, then step 3. Step 2 is already done for you:
the finished saliency maps are part of what you received, so no model has to be
run.

> **If you were already given a `Resample_Output` folder, skip section 3** and
> go straight to section 4. Section 3 only re-creates that folder from the raw
> tracker export.

---

## 1. What you received

| Path | What it is |
| --- | --- |
| `metrics/run_metrics.py` | driver for the two metric scripts |
| `metrics/nss_score_new.py` | NSS implementation |
| `metrics/auc_score_new.py` | AUC-Judd implementation |
| `metrics/gaze_kernel.py` | gaze model shared by both scripts |
| `metrics/test_auc.py`, `metrics/test_gaze_kernel.py` | standalone self-tests (optional) |
| `resample_gaze.py` | step 1: raw tracker samples → one row per video frame |
| `<saliency_root>/<task>/…/*.png` | the finished saliency maps, one PNG per video frame |
| `gazedata/*.csv` | the raw tracker export (only needed to redo step 1) |
| `Resample_Output/<task>/*.csv` | the resampled gaze tables (may be included) |

`run_metrics.py` finds the other three scripts **in its own folder**, so keep
`metrics/` together. Do not copy `run_metrics.py` on its own.

---

## 2. Install

```bash
pip install numpy pandas opencv-python
```

That is all. Tested with Python 3.7 (pandas 1.3) and Python 3.14 (pandas 3.0).

---

## 3. Step 1 — resample the gaze data

### What it does

The eye tracker exports about 120 rows per second. The metrics need **exactly
one row per video frame**, so that `videoFrame` *k* lines up with the saliency
map `frame_000k.png`. This script builds that grid and, for every video frame,
keeps the nearest valid gaze sample.

### Run it

```bash
python resample_gaze.py <input_folder> <output_folder> --by_task
```

```bat
py resample_gaze.py gazedata Resample_Output --by_task
```

`<input_folder>` holds the raw `*.csv` exports; `<output_folder>` is created if
needed. Use `--dry_run` first — it prints the frame rate and the destination of
every file and writes nothing:

```bash
python resample_gaze.py gazedata Resample_Output --by_task --dry_run
```

### ⚠️ Always pass `--by_task`

Without it every output CSV is written **flat into one folder**:

```
without --by_task                with --by_task
Resample_Output\                 Resample_Output\
├── 1268_..._srt19Aud.csv        ├── srt19Aud\
├── 1268_..._bananaQantH.csv     │   └── 1268_..._srt19Aud.csv
├── 1642_..._srt19Aud.csv        └── bananaQantH\
└── ... 700 more files               └── 1268_..._bananaQantH.csv
```

`run_metrics.py` expects **one sub-folder per task** (`<gaze_root>/<task>/*.csv`).
With a flat output folder it finds zero tasks — and nothing warns you while the
script runs, so check the result afterwards. If it is flat, delete it and run
again with `--by_task`.

### Frame rate

By default (`--fps auto`) the rate is read from the stimulus video
`<video_dir>/<task>.mp4`; a rate within 1 % of an integer is snapped to it, so
29.97 becomes 30.

```bash
python resample_gaze.py gazedata Resample_Output --by_task --video_dir "D:\videos"
```

If you do **not** have the videos, pass the rate explicitly. The metrics only
line up when the resampled frame numbers match the maps, so the rate matters:

| Tasks | Pass |
| --- | --- |
| `bananaQantH`, `blockTowerR`, `buildQueuH`, `colorMatchingR`, `look4BottleR`, `look4CupH`, `look4RunningH`, `memoryComplexH`, `memorySimpleR`, `motherElephantR` | `--fps 60` |
| `srt19Aud`, `srt7Aud` | `--fps 30` (recorded at 29.97) |
| `srt14Aud`, `srt30Aud` | `--fps 25` |

Split the batch with `--only` to use different rates, e.g.
`--only "srt19Aud,srt7Aud" --fps 30`.

### What you get

`<output_folder>/<task>/<same file name>.csv`, one row per video frame:

| Column | Meaning |
| --- | --- |
| `videoFrame` | 1-based video frame number |
| `videoTime` | that frame's time stamp in the video |
| `gaze_x`, `gaze_y` | gaze position, normalised to [0, 1], (0, 0) = top-left |
| `calib_winner` | which calibration was used (`raw` or `nudged`) |
| `gaze_missing` | `True` when no sample was close enough — left empty, **not interpolated** |

For each frame the script keeps the nearest valid sample **within 1/120 s
(8.33 ms)**; longer dropouts stay empty and are skipped by the metrics later.
The task name is the last `_`-separated part of the file name after removing the
`.mp4_gaze` / `_gaze` suffix:
`1268_bl_2025-04-28_bananaQantH.mp4_gaze.csv` → `bananaQantH`.

---

## 4. Step 2 — the metrics

### The three folders you have to point at

`run_metrics.py` never guesses where your data is. It needs three paths:

| Purpose | Flag | What must be inside |
| --- | --- | --- |
| Saliency maps | `--saliency_root` | one sub-folder per task, each holding that task's PNGs |
| Gaze tables | `--gaze_root` | one sub-folder per task, each holding one CSV per participant |
| Results | `--output_dir` | does not exist yet; it is created for you |

Use **absolute paths** and quote them — they usually contain spaces.

```
<saliency_root>/
├── srt19Aud/
│   └── maps/                      <- either this ...
│       ├── frame_000001.png
│       └── frame_000002.png
└── bananaQantH/
    ├── frame_0001.png             <- ... or the PNGs straight in the task folder
    └── frame_0002.png

<gaze_root>/
├── srt19Aud/
│   └── 1268_bl_2025-04-28_srt19Aud.mp4_gaze.csv
└── bananaQantH/
    └── 1268_bl_2025-04-28_bananaQantH.mp4_gaze.csv
```

Both map layouts are detected automatically, and the number of digits in the
file name (4 or 6) is read from the first file, so you never type a `--pattern`
yourself. The only requirement: the **task folder names must be identical in
both roots** — a task is scored only when both a map folder and a gaze folder
with the same name exist.

Windows:

```bat
python run_metrics.py ^
  --saliency_root "C:\work\unisal\saliency_output" ^
  --gaze_root     "C:\work\gaze\Resample_Output" ^
  --output_dir    "C:\work\results\unisal"
```

macOS / Linux:

```bash
python run_metrics.py \
  --saliency_root "$HOME/work/unisal/saliency_output" \
  --gaze_root     "$HOME/work/gaze/Resample_Output" \
  --output_dir    "$HOME/work/results/unisal"
```

Always pass `--output_dir`: it names the result folder after the model you are
scoring.

### Quick start

```bash
# 0) is the installation sane? synthetic data, temp folder
python run_metrics.py --self_test

# 1) what would be scored? writes nothing
python run_metrics.py --saliency_root "..." --gaze_root "..." --output_dir "..." --dry_run

# 2) two participants of one task — smoke test
python run_metrics.py --saliency_root "..." --gaze_root "..." --output_dir "..." \
    --only "srt19Aud" --limit 2

# 3) everything
python run_metrics.py --saliency_root "..." --gaze_root "..." --output_dir "..."
```

`--dry_run` prints one line per task (`<n> maps (<pattern>), <m> gaze files`).
If it reports 0 tasks, the folder names do not match — see section 8.

### What the inputs must look like

**Saliency maps**

* one PNG per video frame, grayscale (3-channel PNGs with equal channels are
  fine, they are read as grayscale)
* the frame number at the end of the file name: `frame_0001.png` or
  `frame_000001.png`
* **any intensity scale works.** Both metrics normalise every frame internally
  (NSS z-scores the map, AUC-J min-max normalises it), so `0..1`, `0..255` and
  `0..65535` give identical scores. No preprocessing or normalisation step is
  needed.
* the map must cover the whole screen (same aspect ratio as the video). Its
  resolution may differ from the video; the gaze kernel converts degrees to
  pixels using the map's own size.

**Gaze tables**

One CSV per participant, with at least a frame number and an x/y pair. These
layouts are recognised automatically:

| Columns | Produced by |
| --- | --- |
| `videoFrame`, `gaze_x`, `gaze_y` | `resample_gaze.py` |
| `frame`, `x`, `y` | generic exports |

Coordinates must be normalised to [0, 1], (0, 0) = top-left, (1, 1) =
bottom-right. For absolute pixels add `--coords pixel`. Frame numbers are
1-based and must line up with the map file names. Different column names can be
declared with `--frame_col`, `--x_col`, `--y_col`.

---

## 5. Output

At the top level of `--output_dir` you always get exactly four entries:

```
<output_dir>/
├── nss/                            per-task NSS results
├── auc/                            per-task AUC-J results
├── summary_all_participants.csv    one row per participant (NSS + AUC merged)
└── summary_all_tasks.csv           one row per task, plus a final ALL row
```

Inside the two result folders:

```
nss/<task>/
├── <task>_nss_summary.csv          one row per participant
└── <participant>.csv               per frame: frame, x, y, nss

auc/<task>/
├── <task>_auc_summary.csv          one row per participant
├── <task>_auc_pooled_by_frame.csv  AUC-J per frame, all participants pooled
├── <task>_auc_pooled_summary.csv   one-row summary of that pooled series
└── <participant>.csv               per frame: frame, x, y, auc_j
```

`summary_all_tasks.csv` columns:

| Column | Meaning |
| --- | --- |
| `task` | task name (`*` on the last row = average over tasks) |
| `participants` | number of participants scored |
| `gaze_frames` | number of scored gaze samples, all participants |
| `mean_nss`, `sd_nss` | mean and SD **across participants** of the per-participant mean NSS |
| `weighted_nss` | the same, weighted by the frame count of each participant |
| `mean_auc_j`, `sd_auc_j` | mean and SD across participants of the per-participant mean AUC-J |
| `weighted_auc_j` | frame-weighted version |
| `pooled_auc_j`, `pooled_auc_sd`, `pooled_frames` | AUC-J computed per frame with **all participants at once**, then averaged over frames — the "one number for the whole video" figure |

For a paper report `mean_nss ± sd_nss` and `mean_auc_j ± sd_auc_j` (statistics
over participants), and quote `pooled_auc_j` for the video as a whole.

`summary_all_participants.csv` is meant for SPSS / R / Excel:
`task, gaze_file, rows_in_file, rows_scored, mean_nss, std_nss, min_nss,
max_nss, mean_auc_j, std_auc_j, min_auc_j, max_auc_j, skipped_no_map,
skipped_out_of_bounds, skipped_missing_xy`.

`rows_scored / rows_in_file` is the coverage of that participant — a low value
means the tracker lost the eye for much of the recording. Those frames are not
scored, but the participant's mean is still computed from the remaining frames.

---

## 6. Verify the setup

**The first check does not depend on the model.** Score the task `srt19Aud`
(section 4, quick start step 3 with `--only "srt19Aud"`) and read
`summary_all_tasks.csv`:

| Task | participants | gaze_frames |
| --- | ---: | ---: |
| `srt19Aud` | **54** | **15,869** |

If you get 54 and 15,869, your paths, your gaze tables and the frame alignment
are correct — whatever maps you are scoring.

The NSS / AUC-J values themselves do depend on the maps. For the UNISAL maps of
this study:

| Task | mean NSS | sd NSS | mean AUC-J | sd AUC-J | pooled AUC-J |
| --- | ---: | ---: | ---: | ---: | ---: |
| `srt19Aud` (UNISAL maps) | 1.974 | 1.228 | 0.866 | 0.118 | 0.872 |

You only reproduce those five values with **the same map set**. A different
model gives different numbers — that is the comparison, not a bug.

A second model-independent check: `skipped_no_map` in
`summary_all_participants.csv` must be 0. Non-zero means a map file was missing
for some frame.

---

## 7. Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--saliency_root` | — | root of the saliency maps (required) |
| `--gaze_root` | `Resample_Output` next to the project | root of the gaze CSVs |
| `--output_dir` | `metrics_output_<model>` | where the results go (give it explicitly) |
| `--only` | all tasks | comma separated glob patterns of task names, e.g. `"srt*,look4*"` |
| `--limit N` | 0 (all) | score only the first N participants per task |
| `--force` | off | re-score tasks whose summaries already exist |
| `--dry_run` | off | print the plan, write nothing |
| `--python` | current interpreter | interpreter used for the two metric scripts (needs pandas) |
| `--timeout` | 3600 s | kill a metric run that takes longer |
| `--self_test` | off | synthetic end-to-end check, then exit |

**Keep the gaze-model options at their defaults** — they encode the measured
setup of this study, and changing them changes the scores:

| Option | Default |
| --- | --- |
| `--kernel` | `gaussian` |
| `--sigma_deg` | `1.0` (86.5 % of the mass inside 2 degrees) |
| `--coords` | `normalized` |
| screen geometry (in `gaze_kernel.py`) | 52.0 × 29.0 cm, 60.0 cm viewing distance |

---

## 8. Troubleshooting

**`No task has both maps ... and gaze ...`**
The task folder names under the two roots do not match, or `--by_task` was
forgotten in step 1 (section 3) so the gaze files are flat instead of grouped.
`--dry_run` shows what was found.

**`ModuleNotFoundError: No module named 'pandas'`**
`pip install pandas`. With several Python installations, pass
`--python "C:\path\to\python.exe"`.

**Every row is skipped (`skipped_no_map` == rows)**
Map file names do not look like `frame_0001.png` / `frame_000001.png`, or the
frame numbering does not match the maps. Compare one file name with the
`videoFrame` values in a gaze CSV; for unusual zero-padding pass
`--pattern "frame_{frame:06d}.png"`.

**`skipped_out_of_bounds` is large**
Gaze coordinates outside the frame — after calibration the tracker can point
off screen. Those rows are dropped; nothing to fix.

**`skipped_missing_xy` is large**
No tracker sample within 1/120 s of those frames (section 3). Those frames are
not scored.

**`--output_dir` cannot be created**
Point it at a folder you own, not at a read-only or network location.

---

## 9. How the scores are defined

**NSS (Normalized Scanpath Saliency)** — the map is standardised once per frame
to zero mean and unit standard deviation; the score of a gaze point is the mean
of that standardised map over a small Gaussian region around the point (sigma
1 degree of visual angle, because the tracker's accuracy is about 2 degrees).
0 = chance, negative = the gaze fell on a below-average region.

**AUC-J (AUC-Judd)** — the map is treated as a classifier of "is this pixel
looked at?", the threshold is swept downwards over the saliency levels, and the
ROC curve is integrated. 0.5 = chance, 1.0 = the gaze sits on the most salient
pixels. Ties in the quantised map are counted exactly, so the value is
reproducible to the last digit.

Both metrics normalise **per frame**, which is why the input intensity scale does
not matter: only the ranking of pixels inside one frame is used.

---

## 10. Reproducibility notes

* Do not edit `gaze_kernel.py`; its constants are the measured screen geometry
  and must stay identical across models for the numbers to be comparable.
* Score every model with **the same gaze tables, the same kernel and the same
  options**; only `--saliency_root` changes between models.
* Give each model its own `--output_dir`, otherwise the summary CSVs of one run
  overwrite those of another.
* Record the versions you used:
  `python -c "import pandas, numpy, cv2; print(pandas.__version__, numpy.__version__, cv2.__version__)"`
