# Projection Fidelity Console — Web Interface

A browser front-end for the PCA/MDS continuous error-scale toolkit. It wraps the
existing `error_core` pipeline in a Flask server and renders every result as an
interactive Plotly figure, with a linked-brushing workspace, a decomposed
point table, and a live console — all styled as a single-purpose instrument
panel around the false-negative → true → false-positive scale.

---

## What it does

- **Parameter controls** for every pipeline option: dataset (built-in or an
  uploaded CSV), metric (`intrusion` / `classic`), neighbourhood `k`, `rank` vs
  `distance` weighting, adaptive-`k`, class-aware vs geometric, normalization
  (`none` / `tanh` / `sigmoid` / `sum-abs`) with steepness, classic `α`/`β`,
  `max_samples`, standardization, the untrustworthy threshold, and **Weighted
  MDS** (weighted-Euclidean dissimilarity with `inverse variance` / `correlation`
  / `mutual info` / `custom` feature weights).
- **Run analysis** executes the static pipeline server-side and shows:
  - a **stat bar** (dataset shape, PCA/MDS mean error and FN share, flagged
    count, runtime);
  - **Projections** — a unified panel with a **View / Brush** toggle: *View*
    shows the PCA and MDS scatter coloured on the signed scale; *Brush* enables
    box/lasso selection on either plot, rings the same points on both, and lists
    them with their FP/FN breakdown below. The selection persists across the
    View↔Brush toggle;
  - **Error Space** — an integrated three-plot workspace showing the PCA
    projection, the MDS projection, and the **error space** (PCA error vs MDS
    error) side by side, all sharing one selection. Drag a box/lasso on *any* of
    the three and the same points ring on all three, with a shared detail table.
    Four **quadrant buttons** — *Both FP*, *Both FN*, *PCA-FP / MDS-FN*,
    *PCA-FN / MDS-FP* — each carry a live count and, when clicked, select every
    point whose PCA/MDS error signs match that quadrant (click again to clear).
    A manual brush overrides the active quadrant; **Clear selection** resets all
    three;
  - **Comparison** — per-point PCA error vs MDS error with sign quadrants;
  - **FP vs FN** — per-point intrusion (FP) against extrusion (FN) for each
    method, with a net-zero diagonal, so it is clear which distortion dominates
    each point;
  - **Centroids** — each class centroid traced across high-D → PCA → MDS on a
    normalised frame, plus the mean between-centroid distance per stage, showing
    which classes a method pulls apart or compresses and whether class
    separation is preserved;
  - **Histograms** — signed-error distribution per method, tinted by the scale,
    with a horizontal **mean ± 2σ** band on the bin counts;
  - **Point table** — every flagged point decomposed into FP (intrusion) and FN
    (extrusion) for both methods, sortable by any column.
- **Console panel** — a collapsible drawer that shows the exact stdout the
  pipeline produced (dataset summary, first errors, untrustworthy report, sign
  disagreements, timing). Errors surface here too.
- **Persistence** — the last run (parameters, results, console) is written to
  disk and restored automatically on reload. Up to 20 recent runs are kept.

The signature header element is a **signed-error balance bar** that fills red
from the centre for accumulated FN mass and blue for FP mass, so the overall
character of a run is legible at a glance.

---

## Files

```
.
├── app.py                # Flask server: wraps error_core, captures stdout, persists runs
├── error_core.py         # (unchanged) matplotlib-free metric engine
├── utils.py              # (unchanged) helpers: describe, tags, nice_vmax, normalize
├── plotting.py           # (unchanged) matplotlib figures — used by the CLI, not the web app
├── wine_error_scale.py   # (unchanged) static CLI
├── brushing.py           # (unchanged) matplotlib brushing CLI
├── static/
│   ├── index.html        # layout
│   ├── style.css         # instrument-panel design system
│   └── app.js            # controls, Plotly figures, linked brushing, table, console
├── runs_store.json       # created at runtime — persisted runs (safe to delete)
└── user_csvs/            # created at runtime — uploaded CSVs
```

The web app imports `error_core` and `utils` directly and builds its own Plotly
figures in the browser; it does **not** use `plotting.py` (that remains the
matplotlib path for the command-line scripts). Nothing in the original toolkit
was modified.

---

## Requirements

- Python 3.9+
- `flask`, plus the toolkit's own deps: `numpy`, `pandas`, `scikit-learn`
  (matplotlib is only needed for the CLI scripts, not the server)

```bash
pip install flask numpy pandas scikit-learn
```

Plotly is loaded in the browser from its CDN, so the machine viewing the page
needs internet access for the charts. If the CDN is unreachable the interface
degrades gracefully: the stat bar, point table and console still work and each
chart shows a short "plotting library unavailable" note.

---

## Running

```bash
python app.py
```

Then open <http://localhost:5000>. Set `PORT` to use a different port:

```bash
PORT=8000 python app.py
```

Press **⏎** anywhere (outside a text field) to run the current parameters.

### Using a CSV

Drop a `.csv` onto the upload zone (or click it) in the **Source** group. Pick
the label column from the dropdown that appears, then run. Every remaining
numeric column is treated as a feature.

---

## Notes

- **MDS is O(n²).** For `digits` or any large CSV, set **Max samples** (e.g.
  300–400). Subsampling is stratified and preserves original row indices, so the
  point table still reports true indices.
- **`classic` almost always reads FP.** That is expected — it measures 2D class
  compactness, not high-D fidelity. Lower `β` (or raise `α`) to shift the
  balance, or use `intrusion` for a genuine FP/FN mix.
- **`distance` weighting** produces smaller magnitudes than `rank`; pair it with
  a `tanh` or `sum-abs` normalization for contrast.
- **Weighted MDS** rebuilds only the MDS panel from a precomputed
  weighted-Euclidean dissimilarity (each feature scaled by `sqrt(weight)`); PCA
  and the high-D reference stay unweighted. With standardization on,
  `inverse variance` weights are ~uniform (every column already has unit
  variance) — `mutual info` or `correlation` show a clearer effect there. The
  computed weight vector (min/max) is printed to the console.
- Results persist in `runs_store.json`. Delete it (or use **Clear stored runs**)
  to reset. Incognito/again-from-scratch: just remove that file.