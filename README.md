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
- `error_i = FP_i − FN_i`, each side weighted by how wrong it is (its rank in the
  other space) and normalised to [0, 1].

The rank weighting matters: with plain counts and a fixed `k`, intrusions and
extrusions are forced to be equal and every point would read 0. (This weighted
form is the per-point trustworthiness/continuity decomposition.)

By default it is **class-aware** — only *cross-class* intrusions count as FP and
only *same-class* neighbours pushed away count as FN, so the score reflects
misrepresented **class** structure. Pass `--geometric` for the label-free version
(any rank-broken neighbour counts).

### Legacy: class compactness (`--metric classic`)

The original 2D-only score `error = -(d_same - d_diff) / (d_same + d_diff)` where
`d_same`/`d_diff` are mean distances to same/other-class points in the projection.
It ignores the high-D data, so it mostly reports class separability and lands
almost everything on the FP side (see Notes).

### Sign convention (both metrics)

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
├── error_core.py         # shared, matplotlib-free: data loading, metric, projections
├── wine_error_scale.py   # static analysis + figures (CLI)
├── brushing.py           # interactive linked-brushing tool (CLI)
└── README.md             # this file
```

- **`error_core.py`** — the engine, with no plotting code so both scripts can
  import it without backend conflicts. Provides `load_dataset`, `describe`,
  `maybe_subsample`, `compute_error_scale`, `build_projections`, and the
  `SKLEARN_LOADERS` registry (`wine`, `iris`, `breast_cancer`, `digits`).
- **`wine_error_scale.py`** — loads + describes a dataset, builds the PCA and MDS
  projections, computes errors, and writes two figures plus a console report of
  the untrustworthy points.
- **`brushing.py`** — opens an interactive window; drag a rectangle on either the
  PCA or MDS panel and the same points are ringed on **both** panels with their
  errors printed (works in both directions). This is the
  "fare brushing su uno e vedere i punti dall'altra parte" requirement.

### Outputs
Figures are written into per-dataset, per-preprocessing subfolders so runs never
overwrite each other:

```
outputs/
├── wine/
│   ├── standardized/   wine_projections.png, wine_error_comparison.png
│   └── raw/            (only if you pass --no-standardize)
├── iris/standardized/  ...
└── digits/standardized/ ...
```

The path is `outputs/<dataset>/<standardized|raw>/<tag>_*.png`, where `<tag>`
encodes the metric, e.g. `wine_intrusion_k10_class_*` or `wine_classic_*`, so
different metrics/settings never overwrite each other.

---

## Requirements

- Python 3.9+
- `numpy`, `pandas`, `matplotlib`, `scikit-learn`

```bash
pip install numpy pandas matplotlib scikit-learn
```

Keep the three `.py` files in the **same folder** (the scripts import `error_core`).

---

## How to run

### Static analysis + figures
```bash
python wine_error_scale.py                          # Wine (default)
python wine_error_scale.py --dataset iris
python wine_error_scale.py --dataset breast_cancer
python wine_error_scale.py --dataset digits --max-samples 400
python wine_error_scale.py --csv mydata.csv --label-col species
```
Common flags: `--no-standardize`, `--threshold 0.6` (untrustworthy cutoff),
`--outdir path` (where PNGs go). Prints dataset info, the first 5 PCA/MDS errors,
and the untrustworthy-point report.

### Interactive brushing
```bash
python brushing.py
python brushing.py --dataset iris
python brushing.py --dataset digits --max-samples 300
python brushing.py --csv mydata.csv --label-col species
```
Drag a box on either panel to select points.

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
  fidelity. The `intrusion` metric is the fix: it compares high-D to 2D and shows
  a genuine FP/FN mix with PCA vs MDS disagreement.
- **Intrusion metric, picking `k`.** Small `k` (≈5–10) is local and sensitive;
  larger `k` smooths. `--geometric` (label-free) leans FP on these datasets;
  class-aware leans FN on cleanly separated ones (mild within-class reshuffling) —
  both are honest, they answer different questions.
- **"Untrustworthy" definition.** The default flags `|error| > threshold`, which on
  easy datasets selects the *best*-embedded points. If "untrustworthy" should mean
  *misrepresented*, filter on `error < -threshold` (the FN side) instead.

---

## Next steps (not yet implemented)

- **Weighted-Euclidean MDS**: custom metric → `pairwise_distances` →
  `MDS(dissimilarity='precomputed')`, then recompute and compare the error scale.
- Quantitative PCA-vs-MDS summary stats per dataset.
