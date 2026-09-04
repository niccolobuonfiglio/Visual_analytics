# Projection Fidelity Console — Web Interface

A browser front-end for the PCA/MDS fidelity toolkit. It wraps the `error_core`
and `analysis` pipeline in a Flask server and renders every result as an
interactive Plotly figure. The interface is organised around one central idea:
**data quality and projection quality are independent questions**, and the layout
separates them so they are never confused.

---

## The three tabs

Results are split into three tabs, each answering a distinct question. A short
intro line under the tab bar restates the question when you switch tabs.

### 1. Data Quality — *is the structure in the data, and do the labels describe it?*
These answers depend only on the data, **not** on any projection.

- **Cluster-structure metrics** — supervised precision, recall and F1 on a k-NN
  similarity graph over the high-D points, with **per-class radar charts** (one
  small radar per class), and both **macro and micro** global F1 (their gap
  reveals class imbalance).
- **k-means ground-truth check** — clusters the data with no labels and compares
  to the classes: ARI vs labels, a Hungarian-matched accuracy, and a
  **cluster-vs-class confusion heatmap** with computed **cluster purity** and
  **class coverage**. Choose the clusterer (**k-means** / **GMM** / **spectral**)
  in the controls — spectral recovers non-convex clusters (e.g. rings) that
  k-means cannot.
- **Silhouette-by-k plot** — how many groups the geometry naturally supports; the
  best-k is marked green and your class count red.
- **Neighbourhood search (simulated annealing)** — finds the `k` that maximises
  F1, with a three-panel trace (best-F1, proposed-k, temperature), an
  acceptance-rate stat, and an automatic **search-health diagnosis**.
- **Weighted MDS comparison** (when enabled) — plain vs weighted MDS side by
  side, coloured by class, with the class-separation change and a
  **feature-weight bar chart**.

### 2. Projection Quality — *how faithful is each projection, and which is better?*
These answers are about the *picture*, given the data.

- **PCA vs MDS verdict** — the headline result, a **component-score bar chart**
  (four criteria × two methods), and a **confidence-interval plot** showing
  whether the margin is statistically real. A collapsible section reveals the
  detailed breakdown and **live weight sliders** that recompute the verdict as
  you adjust them (and flag if the winner flips).
- **Error Space** — the main workspace: the PCA and MDS projections on top and
  the error space (PCA error vs MDS error) below, all sharing one selection.
  Every point is **double-encoded** — *colour* is the severity-weighted signed
  error, *size* is the total distortion |FP| + |FN| — so a balanced-but-distorted
  point (FP ≈ FN, error ≈ 0) reads pale but **large**, while a genuinely faithful
  point reads pale and **small**. Drag a box/lasso on any of the three plots and
  the same points ring on all three; four **quadrant buttons** (both-FP, both-FN,
  and the two mixed cases) plus a **near-zero** selector each carry a live count.
  A **"see classes"** toggle recolours the plots by true class.
- **Centroid trajectories** — each class centroid traced across high-D → PCA →
  MDS, showing coarse (configuration) distortion distinct from the per-point kind.
- **Error histograms** — the signed-error distribution per method.

### 3. Point Explorer — *which individual points are wrong, and how?*
- **Untrustworthy points table** — the points a projection asserts a *false
  closeness* for (intrusions), each decomposed into FP/FN for both methods.
- **Selection detail table** — whatever you brushed in the Error Space.

---

## The control panel

All parameters live in one panel at the top, grouped into **Source**, **Metric**,
**Display** and **Analysis**. The toggles are kept together with clear labels and
explanatory notes:

- **Adaptive k** (on by default) — a per-point neighbourhood sized to local
  density. When on, a small indicator by the `k` slider shows the resulting
  **min / median / max** k.
- **Weighted MDS** — mutual-information-weighted Euclidean MDS (only the MDS panel
  changes; PCA and the fidelity reference stay unweighted).
- **Class-aware** (off by default) — switch the fidelity metric from geometric
  (label-free) to class-structure fidelity.
- **Search best k** — run the simulated-annealing search.
- **Recall strictness α** (default 0.5) and **Ground-truth clusterer**
  (k-means / GMM / spectral).

Press **⏎** anywhere (outside a text field) to run the current parameters.

---

## Info popups for every metric

Every metric name is clickable, and each section header carries a small **(i)**
button. Clicking either opens a modal with a thorough description —
**what it is**, **what it's useful for**, and **how it works** — for the Error
Space, fidelity metric, verdict, confidence interval, component scores,
histograms, centroids, weighted MDS, k-means, confusion heatmap, silhouette,
cluster-structure metrics, per-class radar, annealing, untrustworthy points, and
adaptive k.

---

## Files

```
.
├── app.py                # Flask server: wraps error_core + analysis, persists runs
├── error_core.py         # matplotlib-free metric engine
├── analysis.py           # precision/recall/F1, k-means, annealing, verdict
├── utils.py              # helpers
├── static/
│   ├── index.html        # layout (control panel, three tabs, modals)
│   ├── style.css         # instrument-panel design system
│   └── app.js            # controls, Plotly figures, linked brushing, info modals
├── runs_store.json       # created at runtime — persisted runs (safe to delete)
└── user_csvs/            # created at runtime — uploaded CSVs
```

The web app imports `error_core`, `analysis` and `utils` directly and builds its
own Plotly figures in the browser; it does **not** use `plotting.py` (that is the
matplotlib path for the CLI scripts).

---

## Requirements

- Python 3.9+
- `flask`, plus the toolkit's deps: `numpy`, `pandas`, `scikit-learn`, `scipy`
  (matplotlib is only needed for the CLI scripts, not the server)

```bash
pip install flask numpy pandas scikit-learn scipy
```

Plotly is loaded in the browser from its CDN, so the machine viewing the page
needs internet access for the charts. If the CDN is unreachable the interface
degrades gracefully: the stat bar, tables and console still work.

---

## Running

```bash
python app.py
```

Then open <http://localhost:5000>. Set `PORT` to use a different port:

```bash
PORT=8000 python app.py
```

### Using a CSV

Drop a `.csv` onto the upload zone in the **Source** group, pick the label
column, then run. Every remaining numeric column is treated as a feature.

---

## Notes

- **Standardization is on by default** — many datasets have features on very
  different scales, and without scaling distance-based methods are dominated by
  the largest-variance column. Disabling it on such data collapses the error
  range so everything reads near-zero / white; that is a preprocessing artifact,
  not a perfect projection.
- **A k/n warning** appears under the stat bar when the neighbourhood `k` is a
  large fraction of the sample size, since the intrusion/extrusion balance
  degenerates there.
- **MDS is O(n²).** For `digits` or a large CSV set **Max samples** (e.g.
  300–400). Subsampling is stratified and preserves original row indices, so the
  point table still reports true indices.
- **The verdict scopes itself to MDS's randomness** — it re-runs MDS a few times
  and reports the spread, so a "decisive" call accounts for layout variability.
- **Weighted MDS** rebuilds only the MDS panel from a mutual-information-weighted
  dissimilarity; PCA and the high-D reference stay unweighted, so the error
  colours still measure true-geometry distortion.
- **Persistence.** The last run (parameters, results, console) is written to
  `runs_store.json` and restored on reload; up to 20 recent runs are kept. Delete
  the file (or use **Clear stored runs**) to reset. If a stored run was made with
  unusual settings (e.g. standardization off), it will display on reload — run a
  fresh analysis to replace it.

---

## The design goal

The interface is built to make one thing unmistakable: **data properties**
(cluster structure, label validity) are **independent** of **projection
properties** (fidelity, error distribution). A projection can only be blamed for
what a projection does; whether the structure was ever there is a separate,
data-side question — and the two tabs keep them apart.