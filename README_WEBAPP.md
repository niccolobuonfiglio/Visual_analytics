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
  `max_samples`, standardization, and the untrustworthy threshold.
- **Run analysis** executes the static pipeline server-side and shows:
  - a **stat bar** (dataset shape, PCA/MDS mean error and FN share, flagged
    count, runtime);
  - **Projections** — PCA and MDS scatter, points coloured on the signed scale;
  - **Comparison** — per-point PCA error vs MDS error with sign quadrants;
  - **Histograms** — signed-error distribution per method, tinted by the scale;
  - **Parallel** — mean error per class across methods with a ±1σ band;
  - **Point table** — every flagged point decomposed into FP (intrusion) and FN
    (extrusion) for both methods, sortable by any column.
- **Brushing tab** — drag a rectangle (box- or lasso-select) on either the PCA
  or MDS plot; the same points ring on **both** and are listed with their full
  FP/FN breakdown. Selection works in either direction.
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
- Results persist in `runs_store.json`. Delete it (or use **Clear stored runs**)
  to reset. Incognito/again-from-scratch: just remove that file.
