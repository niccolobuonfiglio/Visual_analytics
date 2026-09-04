# Projection Fidelity Toolkit — PCA vs MDS

A toolkit that measures, **per point**, how faithfully a 2-D projection (PCA or
MDS) preserves the structure of high-dimensional data, and — crucially —
**separates two independent questions** that are usually confused:

- **Data quality:** is the class structure genuinely in the data, and do the
  labels describe it? (These depend only on the data, not on any projection.)
- **Projection quality:** given the data, how faithfully does each 2-D
  projection preserve it, and which method is better?

A poor projection of poorly-structured data still looks poor — but that is a
*data* problem, not a *projection* problem, and the toolkit is built to tell
the two apart.

The result is rendered on a continuous **false-negative → true → false-positive**
scale, applied to a PCA and an MDS projection at once so they can be compared
point by point.

There are two ways to use it:

- a **web application** (an interactive instrument panel) — see
  `README_WEBAPP.md`;
- **command-line scripts** for static figures and interactive matplotlib
  brushing — documented here.

---

## What it measures

### Per-point fidelity (intrusion / extrusion)

Each point gets a signed **error in [−1, +1]** on the scale **FN (−1) … True (0)
… FP (+1)**. For a neighbourhood size `k`, each point's neighbours are ranked by
distance in both the high-D space and the 2-D projection:

- **FP (intrusion, +):** a point sitting inside `i`'s 2-D `k`-neighbourhood whose
  true high-D rank is worse than `k` — the projection asserted a closeness that
  is false.
- **FN (extrusion, −):** a true high-D neighbour of `i` that the projection
  pushed out of the 2-D `k`-neighbourhood — a real neighbour was lost.

`error = FP − FN`, each side weighted by how far the offender is displaced
(rank-severity by default) and normalised to [0, 1]. FP/FN can be **reported
separately**, since `error == FP − FN` always holds.

By default the metric is **geometric (label-free)** — it measures whether the
projection preserved the *data's neighbourhoods*, with no assumption that the
labels are correct. Turning on **class-aware** restricts FP to cross-class
intrusions and FN to same-class extrusions, so it measures *class-structure*
fidelity instead. The two answer different questions; geometric is the honest
default when you don't fully trust the labels.

### The analysis layer (data-side checks)

On top of the per-point scale, the toolkit adds checks that describe the **data**,
independent of any projection:

- **Cluster-structure metrics** — supervised precision, recall and F1 on a k-NN
  similarity graph over the high-D points. Answers *is the class structure
  genuinely in the data?* A low score is a property of the data, not the picture.
- **k-means ground-truth check** — clusters the data with **no labels** and asks
  whether those clusters line up with the classes. If not, an apparent
  "projection error" may really be a labels-vs-geometry mismatch. Supports
  **k-means**, **GMM** (elliptical), or **spectral** (non-convex, e.g. rings)
  clusterers.
- **Simulated annealing** — searches for the neighbourhood size `k` that
  maximises F1, since F1(k) is non-convex.
- **PCA-vs-MDS verdict** — a composite of four criteria (fidelity, trustworthiness,
  continuity, cluster preservation) with a paired-bootstrap 95% confidence
  interval; it declines to declare a winner when the evidence is thin.

---

## Datasets

Seven built-in datasets, selectable by name (or bring your own CSV):

| name | shape | classes | what it demonstrates |
|---|---|---|---|
| `wine` | 178 × 13 | 3 | the default; features on very different scales |
| `iris` | 150 × 4 | 3 | two overlapping classes (labels ≈ 2 clusters, not 3) |
| `breast_cancer` | 569 × 30 | 2 | many correlated features (favours PCA) |
| `digits` | 1797 × 64 | 10 | high-dimensional, many classes |
| `synthetic_noisy` | 300 × 50 | 3 | only 5 of 50 features informative — irrelevant features |
| `synthetic_circles` | 300 × 2 | 2 | concentric rings — non-convex (k-means fails, spectral works) |
| `synthetic_mislabeled` | 300 × 10 | 3 | real blobs with random labels — labels carry no structure |

The three synthetic sets each break a different assumption (relevant features /
convex clusters / meaningful labels) and use a fixed seed for reproducibility.

---

## Files

```
.
├── error_core.py         # matplotlib-free engine: loading, projections, the fidelity metric
├── analysis.py           # analysis layer: precision/recall/F1, k-means, annealing, verdict
├── utils.py              # pure helpers: describe, output paths, filename tags, normalize
├── plotting.py           # matplotlib figures (used by the CLI, not the web app)
├── wine_error_scale.py   # static analysis + figures (CLI)
├── brushing.py           # interactive linked-brushing + hover tool (CLI)
├── app.py                # Flask server for the web app (see README_WEBAPP.md)
├── static/               # web app front-end (index.html, app.js, style.css)
└── README.md             # this file
```

- **`error_core.py`** — the engine, with no plotting code so every front-end can
  import it without backend conflicts. Provides `load_dataset`,
  `maybe_subsample`, `build_projections` (PCA always unweighted; optional
  weighted MDS), `compute_intrusion_extrusion_error`, `adaptive_k`,
  `compute_feature_weights` (mutual-information weighting for weighted MDS),
  `compute_errors` (dispatch), and the `SKLEARN_LOADERS` / `SYNTHETIC_LOADERS` /
  `DATASET_NAMES` registries.
- **`analysis.py`** — `build_knn_graph`, `precision_recall_fscore` (with an α
  recall-strictness knob, and both macro and micro global scores),
  `kmeans_analysis` (k-means / GMM / spectral; ARI, matched accuracy, silhouette
  scan, contingency matrix), `simulated_annealing_k` /
  `optimise_relationship_k`, and `compare_methods` (the verdict + bootstrap CI).
- **`utils.py`** — `describe`, `resolve_outdir`, `build_fname_tag`, `nice_vmax`,
  `normalize_errors`; free of plotting/sklearn imports.
- **`plotting.py`** — `plot_projections`, `plot_error_comparison`,
  `plot_error_histograms`, `plot_parallel_coordinates`.

---

## Requirements

- Python 3.9+
- `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `scipy`

```bash
pip install numpy pandas matplotlib scikit-learn scipy
```

Keep the `.py` files in the **same folder** (the scripts import `error_core`,
`utils`, `plotting`, `analysis`). The web app additionally needs `flask` — see
`README_WEBAPP.md`.

---

## Command-line usage

### Static analysis + figures

```bash
python wine_error_scale.py                          # wine (default)
python wine_error_scale.py --dataset iris
python wine_error_scale.py --dataset breast_cancer
python wine_error_scale.py --dataset digits --max-samples 400
python wine_error_scale.py --dataset synthetic_circles
python wine_error_scale.py --csv mydata.csv --label-col species

# geometric (label-free) is the metric's spirit; class-aware is opt-in via the
# web app. The CLI exposes --geometric to force the label-free counting:
python wine_error_scale.py --geometric

# distance weighting, adaptive k, display normalization
python wine_error_scale.py --weight distance --adaptive --normalize tanh
```

Produces four figures (projections, PCA-vs-MDS comparison, error histograms,
parallel coordinates) plus a console report that breaks each flagged point into
its FP/FN components.

### Interactive brushing

```bash
python brushing.py
python brushing.py --dataset iris
python brushing.py --dataset digits --max-samples 300
python brushing.py --weight distance --adaptive
python brushing.py --csv mydata.csv --label-col species
```

Drag a box on either panel to select points; the same points ring on **both**
panels with their errors printed, and hovering a point shows an
index/class/error tooltip. **Needs a GUI backend** (TkAgg / QtAgg) — run it from
a real terminal, not a headless server or plain Jupyter.

### CLI options summary

| flag | scripts | meaning |
|---|---|---|
| `--dataset NAME` | both | one of the seven built-ins (default `wine`) |
| `--csv PATH` | both | use a CSV instead of a built-in dataset |
| `--label-col COL` | both | label column name (required with `--csv`) |
| `--metric NAME` | both | `intrusion` (default) or `classic` (2-D compactness, instructive contrast) |
| `--neighbors K` | both | neighbourhood size `k` (default 10) |
| `--geometric` | both | count all broken neighbours, not just cross-/same-class |
| `--weight NAME` | both | `rank` (default) or `distance` offender weighting |
| `--adaptive` | both | per-point density-driven `k` |
| `--normalize NAME` | both | `none`/`tanh`/`sigmoid`/`sumabs` display transform |
| `--norm-scale S` | both | steepness for tanh/sigmoid (default 3) |
| `--no-standardize` | both | skip `StandardScaler` (default: standardize) |
| `--max-samples N` | both | stratified subsample cap (speeds up MDS) |
| `--threshold T` | static only | `error` cutoff for flagged points (default 0.5) |
| `--outdir DIR` | static only | base output folder |

Figures are written to `outputs/<dataset>/<standardized|raw>/<tag>_*.png`, where
`<tag>` encodes the settings so runs never overwrite each other.

---

## Notes / gotchas

- **Standardization is on by default and matters a lot.** Many datasets have
  features on wildly different scales; without scaling, distance-based methods
  are dominated by the largest-variance column. On such data, disabling it can
  collapse the error range so everything reads near-zero / white — that is a
  preprocessing artifact, not a perfect projection.
- **MDS is O(n²) and stochastic.** For large sets (e.g. digits) use
  `--max-samples`; different seeds give slightly different layouts.
- **`error = FP − FN` is lossy.** A point with large FP *and* large FN reads
  ≈ 0 (they cancel) yet is badly distorted. In the web app, marker **size** encodes
  total distortion |FP| + |FN| precisely to disambiguate this; in the CLI, read
  the FP/FN columns, not just the net error.
- **k vs n.** If `k` is a large fraction of a small sample, the intrusion/extrusion
  balance degenerates toward all-FP. Keep `k` well under ~15% of `n`.
- **The `classic` metric** (`--metric classic`) scores 2-D class compactness,
  ignores the high-D data, and lands almost everything on the FP side. It is kept
  only as an instructive contrast; the `intrusion` metric is the real measure.

---

## The core idea

Structure absent from the data cannot be recovered by any projection (a data
property, diagnosed by the cluster-structure metrics and k-means); structure
present in the data but missing from the picture is a projection artifact (caught
by the fidelity metric). Keeping these two separate — and being honest, via
confidence intervals and self-checks, about the reliability of each answer — is
the toolkit's purpose.