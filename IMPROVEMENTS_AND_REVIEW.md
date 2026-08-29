# Code Review: Improvements, Dubious Choices & Outdated Pieces

A thorough, honest walkthrough of the toolkit's current state — what is
questionable, what is outdated, what is a latent bug, and what is missing —
with the reasoning behind each point and a concrete suggested fix. Items are
grouped by area and tagged by severity:

- 🔴 **bug / correctness** — produces wrong results or crashes under some input.
- 🟠 **dubious** — works, but the design choice is questionable or misleading.
- 🟡 **outdated / dead** — legacy code, disabled features, or superseded ideas.
- 🔵 **enhancement** — not wrong, but a clear opportunity to do better.

---

## 1. Metric semantics

### 1.1 🔴 "Untrustworthy" flag selects the *best* points on clean data
**Where:** `wine_error_scale.py::report_untrustworthy`, and the "flagged" stat.
**Issue:** a point is flagged when `|error| > threshold`. But `error = FP − FN`,
and on cleanly separated datasets the class-aware metric is dominated by mild
within-class reshuffling (FN), so the *largest* `|error|` values are often the
points the projection handles in the most ordinary way, not the misleading ones.
The label "untrustworthy" is therefore backwards for the FP case: a large
positive error (a genuine intrusion — a point pulled among strangers) is the
truly untrustworthy one, while a large negative error is often benign.
**Fix:** separate the two directions. Report "misleading closeness" as
`error > +threshold` (intrusions) and "lost neighbours" as `error < −threshold`
(extrusions), and let the user choose which to flag. Never collapse both into one
symmetric `|error|` cutoff, because the two mean opposite things.

### 1.2 🟠 The net error `FP − FN` is lossy, and the UI leans on it
**Where:** everywhere `err` is used as the primary signal (colour, histograms,
quadrants, verdict fidelity).
**Issue:** a point with `FP = 0.5, FN = 0.5` and a point with `FP = 0, FN = 0`
both have `err = 0`. The first is maximally distorted in both directions; the
second is faithful. Any view keyed on `err` alone cannot tell them apart. This is
already mitigated in two places (the severity colour metric and the size channel)
but the mitigation is partial and easy to miss.
**Fix (partly done):** the new **TP-aware colour mode** addresses this directly —
whiteness is now earned only by low total distortion, so a balanced-but-distorted
point greys out instead of reading as a true positive. Consider making it the
default mode. Longer term, expose `max(FP, FN)` (worst-case distortion) as a
first-class channel alongside the signed `err`.

### 1.3 🟠 Class-aware vs geometric changes *what question is asked*, silently
**Where:** `error_core.compute_intrusion_extrusion_error(class_aware=…)`.
**Issue:** with `class_aware=True` only *cross-class* intrusions and *same-class*
extrusions count. That is a legitimate and useful mode, but it means the metric is
no longer a pure projection-fidelity measure — it is a *class-structure* fidelity
measure, and it will read very differently from the geometric mode on the same
data. A user toggling "Geometric" may not realise the number's meaning changed.
**Fix:** label the two modes in the UI with their semantics ("class-structure
fidelity" vs "pure neighbourhood fidelity"), and surface both means in the stat
bar so the difference is visible rather than implicit.

### 1.4 🟡 The "classic" metric is kept but is essentially a trap
**Where:** `error_core` classic branch; `--metric classic`.
**Issue:** it scores 2-D class compactness only, ignoring the high-D data, and by
construction lands ~99% of points on the FP side (its `d_diff` pools all
other-class points, so it is almost always larger than `d_same`). It measures
separability, not fidelity, and its "FP/FN parts" are just the two signs of one
number, not independent components.
**Fix:** keep it only as an explicit teaching contrast, clearly labelled as "not a
fidelity metric," which the app now does (the FP-vs-FN panel is hidden for it). If
the goal is only pedagogical, consider moving it behind an "advanced/compare" flag
so it is not on equal footing with the intrusion metric.

---

## 2. Adaptive neighbourhood size

### 2.1 🔵 Adaptive-k was disabled in the UI (now re-enabled)
**Where:** `collectParams` had `adaptive: false // commented out`.
**Issue:** the feature is implemented in `error_core.adaptive_k` and works, but the
checkbox was hard-wired off, so it was dead from the user's perspective.
**Fix (done):** the checkbox is restored and wired through. Worth a short UI note
that adaptive-k makes the single `k` slider a *median* target rather than a global
value.

### 2.2 🟠 The density→k mapping is linear and only loosely justified
**Where:** `adaptive_k`.
**Issue:** it maps each point's local radius through the population median to a
per-point `k`, clamped to `[3, 2·k_base]`. The linear scaling and the specific
`2×` bound are reasonable defaults but arbitrary; there is no principled reason a
point twice as sparse should get exactly twice the neighbours, and the hard clamp
can flatten genuinely extreme densities.
**Fix:** consider a rank/quantile-based mapping (assign `k` by the point's density
*percentile*, not its raw radius ratio), which is scale-free and avoids the clamp
doing the real work. Document the bound as a tunable, and expose `k_min`/`k_max`.

### 2.3 🔵 Adaptive-k is not reflected in the relationship graph or the verdict
**Where:** the precision/recall graph and the verdict both use a single global
`k`, even when the fidelity metric is run adaptively.
**Issue:** it is inconsistent to score the projection with per-point neighbourhoods
but judge cluster structure with one global `k`. Not wrong, but the two layers are
answering at different scales.
**Fix:** if adaptive is on, feed the per-point `k` distribution's median into the
graph, or note in the UI that the analysis layer uses a fixed `k` by design.

---

## 3. Precision / recall / F1

### 3.1 🔵 F1 was folded into a generic `f_score`; now surfaced explicitly (done)
**Where:** `precision_recall_fscore`.
**Issue:** the function computed `F_β` and returned it under `f_score`, so with the
default `β=1` you got F1 but it was never labelled as such, and changing `β` would
silently change the "f-score" everywhere downstream.
**Fix (done):** the function now always returns an explicit `f1` alongside the
tunable `f_score` (`F_β`), at per-point, per-label and global levels, and the UI
shows **F1** as the headline with precision/recall beside it.

### 3.2 🟠 Global averaging is macro (per-class), which can flatter tiny classes
**Where:** the global block averages within each class, then across classes.
**Issue:** macro-averaging is the right choice to stop a majority class dominating,
but on very imbalanced data a two-point class counts as much as a 300-point class,
which can make the global F1 look better (or worse) than the population reality.
**Fix:** report both macro (current) and micro (pool all points) global scores, and
let the reader see the gap — a large gap is itself a signal of imbalance.

### 3.3 🟠 Recall's α is powerful but effectively hidden
**Where:** `precision_recall_fscore(alpha=…)`, fixed at `0.0` in the API.
**Issue:** α interpolates between "class must be connected" (α=0) and "class must
be a clique" (α=1). This materially changes recall, yet the app always sends
`rel_alpha = 0.0` with no control. Users never see that recall is being computed in
its most lenient form.
**Fix:** expose α as a slider in the Analysis controls, or at minimum state in the
metrics card that recall uses the lenient (connectivity) definition.

### 3.4 🔵 O(n²) construction and Python-loop precision/recall
**Where:** `build_knn_graph` builds a dense `W`, and `precision_recall_fscore`
loops per vertex summing weights in Python.
**Issue:** fine for a few hundred points, but it will not scale. The dense `n×n`
weight matrix and the per-vertex Python loops are the bottleneck.
**Fix:** use a sparse adjacency (scipy.sparse) and vectorise the per-vertex sums;
the connected-components step already wants a sparse graph anyway.

---

## 4. k-means application

### 4.1 🔵 k-means now reports a matched accuracy and per-class capture (done)
**Where:** `kmeans_analysis`.
**Issue:** previously only ARI/NMI were reported. ARI is label-invariant and
correct but unintuitive ("is 0.62 good?").
**Fix (done):** clusters are now optimally matched to labels (Hungarian
assignment), yielding an interpretable **matched accuracy** and **per-class
capture**, shown in the k-means card next to the ARI.

### 4.2 🟠 Fixed `n_clusters = n_classes` for the preservation comparison
**Where:** `_fit` uses `n_clusters` (defaulting to the class count) in all three
spaces.
**Issue:** the 2-D-vs-high-D preservation ARI assumes the "right" number of
clusters is the number of classes. But the silhouette scan often disagrees (e.g.
Iris prefers 2, not 3). Forcing `k = n_classes` for the preservation test can
understate how well a projection preserves the *natural* clusters.
**Fix:** run the preservation comparison at both `k = n_classes` and
`k = best_silhouette_k`, and report whether the conclusion is robust to the choice.

### 4.3 🟠 k-means restart count and determinism
**Where:** `KMeans(n_init=10, random_state=42)`.
**Issue:** `n_init=10` is reasonable but modest; on harder geometry the reported
partition can hinge on initialisation. The fixed seed makes it reproducible but
also means a single unlucky init is baked into every report.
**Fix:** raise `n_init` (or use `k-means++` with more restarts) for the high-D fit
that feeds the verdict, and/or report the inertia spread across seeds so the
stability is visible.

### 4.4 🔵 k-means assumes convex, isotropic clusters
**Where:** the whole k-means layer.
**Issue:** k-means is the natural Euclidean counterpart to PCA/MDS, so this is a
defensible choice, but it cannot represent elongated or non-convex classes, which
means "classes are not clusters" can fire on data where the classes *are*
clusters, just not spherical ones.
**Fix:** offer an alternative (e.g. a Gaussian-mixture or spectral option) for the
ground-truth check, clearly noting the trade-off, so a non-spherical class is not
mislabelled as "not a cluster."

---

## 5. Projections (PCA / MDS)

### 5.1 🟠 MDS is a single stochastic embedding driving a "decisive" verdict
**Where:** `build_projections` (`MDS(n_init=4)`), consumed by the verdict.
**Issue:** MDS is non-convex and seed-dependent. The verdict can call MDS the
loser (or winner) "decisively" on the basis of one layout, and a fixed seed hides
the variance rather than accounting for it.
**Fix:** run MDS several times and report the distribution (best/median stress and
the resulting verdict spread), or at least surface the final stress so the reader
can judge layout quality. The confidence interval currently captures per-point
paired variation but *not* MDS's own run-to-run variation.

### 5.2 🟡 `MDS(normalized_stress=…)` / sklearn API drift
**Where:** `MDS(...)` construction.
**Issue:** recent scikit-learn versions changed MDS defaults and deprecated some
argument behaviours (e.g. the meaning of `normalized_stress`, and `n_init`
interactions with `dissimilarity='precomputed'`). Pinning is implicit.
**Fix:** pin the scikit-learn version in requirements, and pass `normalized_stress`
explicitly so behaviour is stable across environments.

### 5.3 🔵 Only two, deterministic-vs-not, techniques
**Where:** the whole toolkit compares exactly PCA and MDS.
**Issue:** the framework is general, but limiting to two methods leaves the linked
error-space comparison narrower than it could be, and the supervised metrics would
apply *natively* to graph-based methods.
**Fix:** add t-SNE/UMAP panels; the precision/recall graph then comes for free from
the method itself rather than being reconstructed via k-NN.

---

## 6. Weighted MDS & feature weighting

### 6.1 🔴 `compute_feature_weights` default `method="inverse_variance"` is unhandled
**Where:** `error_core.compute_feature_weights(method="inverse_variance")`.
**Issue:** the signature defaults to `"inverse_variance"`, but the body only
handles `custom`, `correlation`, and `mutual_info` — the default value falls
through to the `else` branch and raises `ValueError`. It works in the app only
because the server overrides the default with `weight_method="mutual_info"`. Any
direct caller using the default crashes.
**Fix:** either implement `inverse_variance` (weight `1/var`, meaningful only when
standardization is *off*) or change the default to `"mutual_info"` to match the
server. Also note that with standardization on, every column has unit variance, so
`inverse_variance` would produce uniform weights and do nothing — which is why the
option was effectively pointless and is best removed.

### 6.2 🟠 Weighted MDS changes MDS but not the fidelity reference
**Where:** `build_projections(weighted_mds=True)`.
**Issue:** the weighted-Euclidean dissimilarity reshapes the MDS layout, but the
high-D reference used by the error metric stays unweighted (correctly, so the
metric still measures the true geometry). The subtlety is that the *verdict* then
compares an unweighted PCA against a *weighted* MDS, which is arguably not
apples-to-apples.
**Fix:** when weighted MDS is on, either weight PCA's input consistently or clearly
flag in the verdict that the two methods saw different feature scalings.

### 6.3 🟡 Weight vector only reported as min/max in the console
**Where:** weight info printed to stdout.
**Issue:** the actual per-feature weights (which features were up/down-weighted)
are the interesting part and are not surfaced in the UI.
**Fix:** show the weight vector as a small bar chart in the Source panel when
weighted MDS is on.

---

## 7. Simulated annealing / parameter search

### 7.1 🔵 Now explained and visualised in the UI (done)
**Where:** the new annealing card.
**Issue:** previously the search ran only server-side and printed one line to the
terminal; users had no idea why it existed or whether it worked.
**Fix (done):** the Verdict & Analysis tab now explains *why* a metaheuristic is
needed (non-convex F1(k)), shows best-k, evaluation count vs exhaustive, uphill
moves, and start temperature, and plots the full search trace.

### 7.2 🟠 The temperature calibration is good but the schedule is fixed
**Where:** `simulated_annealing_k`.
**Issue:** T₀ is cleverly calibrated to the objective's own spread (scale-invariant,
which fixed a real bug), but the cooling rate, iteration count, and proposal width
are fixed constants. On a very multi-modal objective the fixed budget may stop
short.
**Fix:** make `n_iter` adaptive (stop when best-so-far stalls for N steps), and
expose the schedule as advanced options.

### 7.3 🟠 The search optimises F1 at a *fixed* α, β
**Where:** `optimise_relationship_k` calls the objective with the passed α/β.
**Issue:** the "best k" is only best *for that α and β*. Since α is hidden and
fixed at 0, the reported optimum is implicitly "best k for lenient recall,
balanced F1" — a caveat users can't see.
**Fix:** state the α/β used on the annealing card (β is shown implicitly via "F1";
α should be labelled), and re-search if the user changes them.

### 7.4 🔵 "vs exhaustive 15/59" is hard-coded to a 59-wide grid
**Where:** the annealing card computes the exhaustive baseline as a constant.
**Issue:** the "59" assumes the k-range `[2, 60]`; if the range changes with dataset
size (it does — `k_hi = min(60, n/3)`), the "vs exhaustive" denominator is wrong.
**Fix:** compute the exhaustive count from the actual `[k_lo, k_hi]` used and return
it in the payload rather than assuming 59.

---

## 8. Verdict

### 8.1 🟠 The four component weights (0.40/0.20/0.20/0.20) are editorial
**Where:** `compare_methods::WEIGHTS`.
**Issue:** a different weighting could flip the winner. The weights are stated (not
tuned), which is the honest choice, but they are still a value judgement baked into
a "mechanical" verdict.
**Fix (partly done):** the UI now always shows the four components separately so a
reader can re-weigh. Consider adding a "what if" control that lets the user drag the
weights and see whether the verdict changes — turning the hidden assumption into an
explicit, explorable one.

### 8.2 🟠 "Decisive" hinges on a paired bootstrap that ignores MDS run variance
**Where:** the bootstrap resamples per-point distortion differences.
**Issue:** the CI captures variation *across points* for one fixed pair of layouts.
It does **not** capture MDS's stochasticity, so "decisive" can be over-confident:
re-running MDS might move the margin more than the CI suggests.
**Fix:** nest the bootstrap inside a few MDS re-runs, or explicitly scope the claim
("decisive for these layouts") in the UI.

### 8.3 🔵 Fidelity component uses mean |FP|+|FN|, double-counting balanced points
**Where:** the fidelity component `1 − mean(|FP|+|FN|)`.
**Issue:** summing both sides penalises a balanced-but-distorted point twice, while
the signed error would cancel it to zero. This is arguably the *right* behaviour
(such a point really is badly distorted), but it means "fidelity" and "mean signed
error" tell different stories, which can confuse.
**Fix:** document precisely what the fidelity component measures (total distortion,
not net), and keep it distinct from the signed-error stat in the bar.

---

## 9. Frontend / architecture

### 9.1 🟡 Dead code: CSV upload wired in HTML but commented out in JS
**Where:** `handleCsvUpload` and its bindings are commented; `collectParams` has a
commented `csv_path` line.
**Issue:** the Source panel implies CSV upload works, but the handler is disabled,
so a user dropping a CSV gets nothing.
**Fix:** either finish wiring the CSV path (the backend `load_dataset` already
supports it) or remove the upload affordance until it is ready, to avoid a
dead-end.

### 9.2 🟠 `scattergl` marker colour arrays and selection opacity
**Where:** `esBaseTrace`.
**Issue:** with `scattergl`, per-point RGB string arrays (the new TP-aware mode)
and the `selected/unselected` opacity interaction can behave differently from SVG
`scatter`, occasionally not dimming unselected points on some GPUs.
**Fix:** verify the TP-aware colour + brushing combination on the WebGL path; fall
back to `scatter` for small n where crispness matters more than speed.

### 9.3 🔵 The three error-space plots recompute distortion three times
**Where:** `renderErrorSpace` calls `esBaseTrace` per plot, each recomputing
totals/severity.
**Issue:** minor inefficiency and a place for the three views to drift if one code
path changes.
**Fix:** compute the per-point colour/size arrays once per run and pass them in.

### 9.4 🟡 Persisted runs store full assignment arrays
**Where:** `runs_store.json` keeps up to 20 runs including k-means `assignments` and
per-point arrays.
**Issue:** the store can grow large, and old runs may contain fields newer code does
not expect (schema drift on reload).
**Fix:** version the stored payload and prune fields not needed for restore; guard
the restore path against missing keys.

---

## 10. Cross-cutting / smaller notes

- 🟡 **Distance-weighted values are tiny and need `--normalize`** to be legible;
  the UI could auto-suggest a normalization when distance weighting is chosen.
- 🔵 **No caching across runs:** changing only a display option (e.g. colour mode)
  re-runs nothing (good), but changing `k` recomputes the full pipeline including
  MDS; the expensive MDS could be cached when only the metric `k` changes.
- 🔵 **Euclidean everywhere:** appropriate for PCA/MDS, but a note that non-Euclidean
  or mixed-type data needs a different base dissimilarity would set expectations.
- 🟠 **Colour-blind safety:** the red–white–blue diverging scale is not guaranteed
  safe; the size channel helps, but a colour-blind-safe palette option would be
  a genuine accessibility win.
- 🟡 **`nice_vmax` floor of 0.1** can over-saturate a run whose true peak error is
  tiny, making a faithful projection look more colourful than it is; consider a
  data-driven floor.

---

## Priority summary

If addressing these in order of impact-for-effort:

1. **🔴 Fix `compute_feature_weights` default (6.1)** — a latent crash.
2. **🔴 Re-frame the "untrustworthy" flag (1.1)** — it currently labels the wrong
   points.
3. **🟠 Scope the "decisive" verdict to account for MDS variance (8.2 / 5.1)** —
   the strongest claim rests on the weakest assumption.
4. **🟠 Expose recall's α and the verdict weights (3.3, 8.1)** — turn hidden
   assumptions into visible controls.
5. **🔵 Sparse/vectorised precision/recall and cached MDS (3.4, 9.3, §10)** — the
   scalability ceiling.
6. **🟡 Remove or finish the dead CSV upload and the `inverse_variance` option
   (9.1, 6.1)** — cut the dead ends.

Everything tagged **(done)** above was addressed in the current round of changes:
the verdict display and popup, the annealing explanation and trace, the "see
classes" toggle, the TP-aware colour mode, the re-enabled adaptive-k, the explicit
F1, and the richer k-means output.