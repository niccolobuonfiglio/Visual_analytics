# Continuous Error Scale for PCA vs MDS

A small toolkit that measures, **per point**, how faithfully a 2D projection
(PCA or MDS) preserves class structure, and visualizes the result on a
continuous **FN → True → FP** scale. **Wine is the default dataset, but it is
just a starting point** — any built-in sklearn classification dataset or a CSV
with a label column works.

---

## What it does

Each point gets a signed **error in [−1, +1]** on the scale **FN (−1) … True (0)
… FP (+1)**, and the same scale is applied to a **PCA** and an **MDS** projection
so the two can be compared point by point. Two metrics are available.

### Default: intrusion / extrusion (`--metric intrusion`)

This compares the **high-D** space to the **2D** projection (so it actually
measures what the projection did, not just whether classes look compact). For a
neighbourhood size `k`:

- **FP_i (intrusion):** points sitting in `i`'s 2D `k`-neighbourhood whose *true*
  high-D rank is > `k` — the projection asserted a closeness that is false.
- **FN_i (extrusion):** `i`'s true high-D `k`-neighbours whose 2D rank is > `k` —
  the projection pushed a real neighbour away.
- `error_i = FP_i − FN_i`, each side weighted by how wrong it is and normalised
  to [0, 1].

The two sides can be **reported separately** (see the untrustworthy-point table,
which prints `pca_FP`/`pca_FN`/`mds_FP`/`mds_FN` columns) so you can see whether
intrusion or extrusion drives each point's error — `error == FP − FN` always
holds.

**Weighting (`--weight`).** How each offending neighbour is scored:

- `rank` (default): weight by `rank_in_other_space − k` — the classic
  trustworthiness/continuity severity. With plain counts and a fixed `k`,
  intrusions and extrusions are forced equal and every point reads 0; the rank
  weighting is what makes the per-point score meaningful.
- `distance`: weight by the actual distance discrepancy — an intrusion counts
  more the *farther* the intruder truly is in high-D despite being pulled close
  in 2D, and symmetrically for extrusions. Severity of misplacement, not just
  its rank, then drives the score. (Distance-weighted values are smaller and
  benefit from `--normalize`.)

**Adaptive neighbourhood (`--adaptive`).** Instead of one global `k`, each point
gets a `k` from its local density: dense points get a smaller `k`, sparse /
outlying points a larger one, with the median point keeping the `--neighbors`
value (bounds `[3, 2k]`). Useful when density varies a lot across the data.

By default it is **class-aware** — only *cross-class* intrusions count as FP and
only *same-class* neighbours pushed away count as FN, so the score reflects
misrepresented **class** structure. Pass `--geometric` for the label-free version
(any rank-broken neighbour counts).

### Legacy: class compactness (`--metric classic`)

The 2D-only score
`error = -(alpha·d_same − beta·d_diff) / (alpha·d_same + beta·d_diff)` where
`d_same`/`d_diff` are mean distances to same/other-class points in the
projection. `alpha` up-weights within-class tightness, `beta` up-weights
between-class separation; `--alpha 1 --beta 1` (the default) is the original
scale. It ignores the high-D data, so it mostly reports class separability and
lands almost everything on the FP side (see Notes).

### Normalization (`--normalize`)

Optionally transform the final signed errors (sign always preserved) for display
contrast:

| method    | effect                                                            |
|-----------|-------------------------------------------------------------------|
| `none`    | unchanged (default)                                               |
| `tanh`    | `tanh(scale·err)` — smooth squash, keeps small values ~linear     |
| `sigmoid` | signed logistic `2·σ(scale·err) − 1` — sharper near 0             |
| `sumabs`  | divide by mean |error| so the typical magnitude is ~1            |

`--norm-scale` sets the steepness for `tanh`/`sigmoid` (default 3).

### Sign convention (all metrics)

| value        | label       | colour |
|--------------|-------------|--------|
| `error < 0`  | **FN** (−1) | red    |
| `error = 0`  | True (0)    | white  |
| `error > 0`  | **FP** (+1) | blue   |

Because intrusion-metric values are usually well inside [−1, 1], the colour scale
and plot axes auto-rescale to the data's peak (shown on the colour bar) for
contrast.

---

## Files

```
.
├── error_core.py         # matplotlib-free engine: loading, metrics, projections
├── utils.py              # helpers: describe, output paths, filename tags, normalize
├── plotting.py           # all figures: projections, comparison, histograms, parallel
├── wine_error_scale.py   # static analysis + figures (CLI)
├── brushing.py           # interactive linked-brushing + hover tool (CLI)
└── README.md             # this file
```

- **`error_core.py`** — the engine, no plotting code so both scripts import it
  without backend conflicts. Provides `load_dataset`, `maybe_subsample`,
  `build_projections`, `compute_error_scale` (classic, with `alpha`/`beta`),
  `compute_intrusion_extrusion_error` (with `weight`, `adaptive`,
  `return_parts`), `adaptive_k`, `compute_errors` (dispatch), and the
  `SKLEARN_LOADERS` registry (`wine`, `iris`, `breast_cancer`, `digits`).
- **`utils.py`** — `describe`, `resolve_outdir`, `build_fname_tag`, `nice_vmax`,
  and `normalize_errors`, kept free of plotting/sklearn imports.
- **`plotting.py`** — every figure, reusable by both front-ends:
  `plot_projections`, `plot_error_comparison`, `plot_error_histograms`,
  `plot_parallel_coordinates`.
- **`wine_error_scale.py`** — loads + describes a dataset, builds PCA and MDS,
  computes errors (with FP/FN parts), writes four figures, and prints the
  untrustworthy-point report.
- **`brushing.py`** — interactive window; drag a rectangle on either the PCA or
  MDS panel and the same points are ringed on **both** panels with their errors
  printed (works both directions). **Hover** any point for an index/class/error
  tooltip. This is the "fare brushing su uno e vedere i punti dall'altra parte"
  requirement.

### Outputs
Figures are written into per-dataset, per-preprocessing subfolders so runs never
overwrite each other:

```
outputs/<dataset>/<standardized|raw>/<tag>_{projections,error_comparison,histograms,parallel}.png
```

`<tag>` encodes the metric and settings, e.g. `wine_intrusion_k10_class`,
`wine_intrusion_kauto_class_tanh`, or `wine_classic`, so different
metrics/settings never overwrite each other.

---

## Requirements

- Python 3.9+
- `numpy`, `pandas`, `matplotlib`, `scikit-learn`

```bash
pip install numpy pandas matplotlib scikit-learn
```

Keep the `.py` files in the **same folder** (the scripts import `error_core`,
`utils`, `plotting`).

---

## How to run

### Static analysis + figures
```bash
python wine_error_scale.py                          # Wine (default)
python wine_error_scale.py --dataset iris
python wine_error_scale.py --dataset breast_cancer
python wine_error_scale.py --dataset digits --max-samples 400
python wine_error_scale.py --csv mydata.csv --label-col species

# new: distance weighting, adaptive k, display normalization
python wine_error_scale.py --weight distance --adaptive --normalize tanh
# new: weighted classic metric
python wine_error_scale.py --metric classic --alpha 1 --beta 2
```
Produces four figures (projections, PCA-vs-MDS comparison, **error histograms**,
**parallel coordinates**) plus the console report, which now breaks each
untrustworthy point into its FP/FN components.

### Interactive brushing
```bash
python brushing.py
python brushing.py --dataset iris
python brushing.py --dataset digits --max-samples 300
python brushing.py --weight distance --adaptive
python brushing.py --csv mydata.csv --label-col species
```
Drag a box on either panel to select points; **hover** a point for a tooltip.

**Needs a GUI backend** (TkAgg / QtAgg): run it from a real terminal, not a
headless server. It will *not* display in plain Jupyter — run `%matplotlib qt`
first, or just run it as a script. (The static script forces the `Agg` backend so
it can save PNGs anywhere; remove its `matplotlib.use("Agg")` line for pop-ups.)

### CLI options summary

| flag              | scripts        | meaning                                            |
|-------------------|----------------|----------------------------------------------------|
| `--dataset NAME`  | both           | `wine` (default), `iris`, `breast_cancer`, `digits`|
| `--csv PATH`      | both           | use a CSV instead of a built-in dataset            |
| `--label-col COL` | both           | label column name (required with `--csv`)          |
| `--metric NAME`   | both           | `intrusion` (default) or `classic`                 |
| `--neighbors K`   | both           | neighbourhood size for intrusion metric (10)       |
| `--geometric`     | both           | intrusion metric ignores labels                    |
| `--weight NAME`   | both           | `rank` (default) or `distance` offender weighting  |
| `--adaptive`      | both           | per-point density-driven k                         |
| `--normalize NAME`| both           | `none`/`tanh`/`sigmoid`/`sumabs` display transform  |
| `--norm-scale S`  | both           | steepness for tanh/sigmoid (3.0)                   |
| `--alpha A`       | both           | classic: weight on same-class distance (1.0)       |
| `--beta B`        | both           | classic: weight on different-class distance (1.0)  |
| `--no-standardize`| both           | skip `StandardScaler` (default: standardize)       |
| `--max-samples N` | both           | stratified subsample cap (speeds up MDS)           |
| `--threshold T`   | static only    | `\|error\|` cutoff for untrustworthy points (0.5)|
| `--outdir DIR`    | static only    | base folder; figures go to `<DIR>/<dataset>/<std\|raw>/` |

---

## Notes / gotchas

- **Standardization is on by default.** Many datasets (Wine, breast_cancer) have
  features on wildly different scales; without scaling, PCA is dominated by the
  largest-variance column. Use `--no-standardize` to compare.
- **MDS is O(n²).** For larger sets (e.g. Digits, 1797 points) use
  `--max-samples`. It subsamples *with stratification* and keeps the original row
  indices, so the untrustworthy report still prints true indices.
- **`classic` is almost always FP (blue).** Its `d_diff` pools *all* other-class
  points, which is large by construction, so `d_diff > d_same` (an FP error) is the
  default for any normal point. It scores global class compactness, not projection
  fidelity. Tuning `--beta` down (or `--alpha` up) shifts the balance. The
  `intrusion` metric is the real fix: it compares high-D to 2D and shows a genuine
  FP/FN mix with PCA vs MDS disagreement.
- **Distance vs rank weighting.** `rank` values fill the scale nicely; `distance`
  values are smaller in magnitude (they are distance fractions) — pair `--weight
  distance` with `--normalize tanh` or `sumabs` for readable colours.
- **Intrusion metric, picking `k`.** Small `k` (≈5–10) is local and sensitive;
  larger `k` smooths. `--adaptive` sidesteps the choice by scaling `k` with local
  density. `--geometric` (label-free) leans FP on these datasets; class-aware
  leans FN on cleanly separated ones (mild within-class reshuffling) — both are
  honest, they answer different questions.
- **"Untrustworthy" definition.** The default flags `|error| > threshold`, which on
  easy datasets selects the *best*-embedded points. If "untrustworthy" should mean
  *misrepresented*, filter on `error < -threshold` (the FN side) instead.

---

## Next steps (not yet implemented)

- **Weighted-Euclidean MDS**: custom metric → `pairwise_distances` →
  `MDS(dissimilarity='precomputed')`, then recompute and compare the error scale.
- Quantitative PCA-vs-MDS summary stats per dataset.