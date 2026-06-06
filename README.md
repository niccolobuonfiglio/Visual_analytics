# Continuous Error Scale for PCA vs MDS

A small toolkit that measures, **per point**, how faithfully a 2D projection
(PCA or MDS) preserves class structure, and visualizes the result on a
continuous **FN → True → FP** scale. **Wine is the default dataset, but it is
just a starting point** — any built-in sklearn classification dataset or a CSV
with a label column works.

---

## What it does

For every projected point `i` we compute two average Euclidean distances **in the
2D projection**:

- `d_same` — mean distance to points of the **same** class (excluding self)
- `d_diff` — mean distance to points of **different** classes

and combine them into one signed error:

```
error_i = -(d_same - d_diff) / (d_same + d_diff)     # range [-1, +1]
```

### Sign convention (canonical "professor" scale)

| value        | meaning                                   | label       | colour |
|--------------|-------------------------------------------|-------------|--------|
| `error < 0`  | point is pulled toward **other** classes  | **FN** (−1) | red    |
| `error = 0`  | equidistant from own / other classes      | True (0)    | white  |
| `error > 0`  | point sits **tight inside its own** class | **FP** (+1) | blue   |

> The bare ratio `(d_same - d_diff)/(...)` gives the *opposite* sign. We negate it
> so the result matches the `−1 = FN … +1 = FP` scale. To get the raw version,
> remove the leading minus in `compute_error_scale` (in `error_core.py`).

The same metric is applied to a **PCA** and an **MDS** projection so the two
methods can be compared point by point.

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
Figures are written with a **dataset prefix** so multiple runs never overwrite:
`wine_projections.png`, `wine_error_comparison.png`, `digits_projections.png`, …

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
| `--no-standardize`| both           | skip `StandardScaler` (default: standardize)       |
| `--max-samples N` | both           | stratified subsample cap (speeds up MDS)           |
| `--threshold T`   | static only    | `\|error\|` cutoff for untrustworthy points (0.5)|
| `--outdir DIR`    | static only    | output folder for PNGs (default `.`)               |

---

## Notes / gotchas

- **Standardization is on by default.** Many datasets (Wine, breast_cancer) have
  features on wildly different scales; without scaling, PCA is dominated by the
  largest-variance column. Use `--no-standardize` to compare.
- **MDS is O(n²).** For larger sets (e.g. Digits, 1797 points) use
  `--max-samples`. It subsamples *with stratification* and keeps the original row
  indices, so the untrustworthy report still prints true indices.
- **Easy datasets sit on the FP side.** When classes separate cleanly (Wine,
  Iris), almost every point gets a positive error (tight in its own class), so the
  comparison plot clusters in the upper-right "both FP" quadrant and genuine FN
  points are rare. Harder, overlapping datasets (Digits) produce a real mix of FN
  and FP and clearer PCA/MDS disagreements — a better stress test.
- **"Untrustworthy" definition.** The default flags `|error| > threshold`, which on
  easy datasets selects the *best*-embedded points. If "untrustworthy" should mean
  *misrepresented*, filter on `error < -threshold` (the FN side) instead.

---

## Next steps (not yet implemented)

- **Weighted-Euclidean MDS**: custom metric → `pairwise_distances` →
  `MDS(dissimilarity='precomputed')`, then recompute and compare the error scale.
- Quantitative PCA-vs-MDS summary stats per dataset.
