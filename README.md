# Projection Fidelity Toolkit

A 2-D scatter plot of high-dimensional data gets read literally: points that look
close are assumed to be similar. Often they aren't. This toolkit measures, **for
every individual point**, how badly a PCA or MDS embedding misrepresents that
point's real neighbourhood, and puts the answer on the plot next to the point.

It then tells you which of the two projections to trust for your data, and says
so with a confidence interval rather than a bare number.

---

## Install

Python 3.10+.

```bash
pip install -r requirements.txt
```

## Run

```bash
python app.py            # then open http://localhost:5000
PORT=8080 python app.py  # if 5000 is taken
```

No build step and no database. Results are cached to `runs_store.json`, so
closing the tab or restarting the server doesn't lose your last analysis.

---

## The workflow

### 1. Pick your data

In the left rail, choose a built-in dataset or upload a CSV and name its label
column.

Four real datasets: `wine`, `iris`, `breast_cancer`, `digits`.

Five synthetic ones, each built to expose a specific failure mode — useful for
checking the tool behaves before you trust it on your own data:

| | what it isolates |
|---|---|
| `synthetic_circles` | separable but not by any convex boundary |
| `synthetic_noisy` | 5 informative features buried in 50 |
| `synthetic_mislabeled` | real clusters, labels assigned at random |
| `hierarchical` | two nested scales of structure at once |
| `imbalanced_overlap` | four classes of very unequal size, two overlapping |

MDS is O(n²) and gets slow past a couple of thousand points. Set **max samples**
to subsample (stratified, so class proportions survive).

### 2. Set the neighbourhood size

**k** is how many neighbours define "this point's neighbourhood". 10 is a sensible
default.

Leave **adaptive k** on. It gives each point a k scaled to its local density, so
points in dense and sparse regions are judged on comparable terms; the
median-density point keeps exactly the k you set. The readout beside the slider
shows the resulting spread.

If k is too large for your sample size a warning appears under the summary bar.
Take it seriously — above ~15% of n the scores lose contrast.

Press **Run**.

### 3. Read the verdict

A banner at the top names the more faithful projection. It only says
**decisive** when a paired bootstrap interval on the per-point difference
excludes zero *and* the margin clears 0.02. Anything else means the two
projections are comparable for your data — not that one won narrowly.

**Examine verdict** opens the full breakdown: the four weighted criteria
(fidelity 0.30, neighbourhood 0.25, Shepard 0.20, cluster agreement 0.25) scored
separately for both methods, plus Cohen's d and the confidence interval. If you
disagree with the weighting, read the four components directly.

### 4. Explore the projections

Two linked scatters, PCA and MDS, showing the same points. Colour is the
selected per-point metric, red = least trustworthy.

- **Drag** a box or lasso on either plot. The same points light up on the other
  one — that's the core move. A group that's tight in PCA and scattered in MDS
  is visible in a single gesture.
- **Hover** any point to ring it in both plots and see its scores side by side.
- **Colour by classes** instead of score answers "is this red patch a real
  problem, or just where two classes meet?" Flip between the two views.

Four metrics are available. They deliberately disagree:

| metric | what it asks |
|---|---|
| neighbour F1 | did the right points stay in the top-k? |
| JS fidelity | did their *relative* closeness survive? |
| local stress | did the distances themselves survive? |
| norm. stress | same, without rank weighting — widest spread, best for ranking points |

### 5. Narrow down

Filters compose with your brush selection; the status bar shows what survives.

- **by class** — isolate one group
- **by score** — separate windows for PCA and MDS, so "good in PCA, bad in MDS"
  is one query
- **by disagreement** — jump straight to the points where the choice of
  projection actually changes the picture
- **by attribute** — a range on any original feature. This is what connects a
  distortion to the data: if the bad points also share an extreme feature value,
  that's a finding.

Box plots and the metric-vs-metric scatters recompute over whatever is
currently active. Points far off the diagonal in the comparison view are the
ambiguous ones — worth looking at by hand.

### 6. Inspect the bad points

**Level & Evenness** shows both projections double-encoded:

- **fill** = Level, the mean of the four metrics — how faithful the point is on average
- **ring** = Evenness, 1 − (max − min) — whether the metrics agree about it

A point is **Untrustworthy** if either reading is red, **Medium** if either is
amber, **Trustworthy** only when both are green. A high Level with poor Evenness
is not a clean point; it's a point whose verdict depends on which metric you ask.

Use the tier toggle to dim everything else. The table below lists whatever is
visible, sortable, and clicking an index pins that point in every plot at once.

The **Untrustworthy Points** table underneath is the conservative summary: points
flagged by low consensus fidelity in the *worse* of the two projections, with the
metrics that independently agree. Read the count as "these are worth checking",
not as an exact number of bad points — the threshold is a readability heuristic,
capped at 20 rows.

---

## Without the browser

Same engine, no server.

```bash
# batch figures to disk
python wine_error_scale.py --dataset wine --adaptive
python wine_error_scale.py --csv mydata.csv --label-col species --outdir figs

# interactive linked brushing in a matplotlib window (needs a GUI backend)
python brushing.py --dataset iris --adaptive
```

Or drive the pipeline directly:

```python
from pipeline import run_pipeline
r = run_pipeline({"dataset": "wine", "neighbors": 10, "adaptive": True})
print(r["verdict"]["verdict"])
```

---

## Layout

```
datasets.py  projections.py  metrics.py  verdict.py   the engine
pipeline.py  store.py        app.py                   orchestration + HTTP
static/                                               the browser interface
legacy/  static/legacy/                               retired, not imported
```

`error_core.py` and `analysis.py` remain as compatibility facades. See
`REFACTOR.md` for the module map, the dependency rules, and the list of code
paths that were retired.

## Worth knowing

- Scores are **geometric**. Labels colour and filter, they never score. A low
  score means the projection distorted the point, not that it failed to separate
  your classes.
- The reference is standardised Euclidean distance in the feature space. That's
  the fair yardstick for PCA and MDS because it's the geometry they assume — it
  is not a claim that Euclidean distance is the right notion of similarity for
  your data.
- Tier and threshold cut-offs are conventions, not measurements. They're
  constants at the top of `static/untrust.js` and in `_flag_untrustworthy` in
  `pipeline.py`.