# Report Completion & Research-Grounding Guide

A section-by-section plan for finishing the "Path of Excellence VA Report",
grounding it as a research-quality document, and a companion list of code/report
improvements worth doing. Written against the current draft (the version with the
`[TO-DO]` and `[to work on]` markers) and the current state of the toolkit.

The guide has four parts:

1. **What to add, and where** — every section, including the empty ones, with the
   content to write and the verified numbers to quote.
2. **Possible conflicts** — internal inconsistencies to resolve so the report does
   not contradict itself or the code.
3. **References & research grounding** — what to cite, which sources are the
   *basis* you build on, and which parts are *your* contribution.
4. **Code / report improvements worth doing** — a prioritised backlog.

All numbers below were produced by the current toolkit (seed 42, intrusion
metric, k = 10, standardization on) and are safe to quote:

| Dataset | n × p | classes | verdict | ARI(labels) | silhouette k | F1 | anneal k |
|---|---|---|---|---|---|---|---|
| Wine | 178 × 13 | 3 | MDS, not decisive (+0.006) | 0.897 | 3 | 0.943 | 10 |
| Iris | 150 × 4 | 3 | PCA, not decisive (+0.002) | 0.620 | **2** | 0.937 | 13 |
| Breast-cancer | 569 × 30 | 2 | **PCA, decisive (+0.021)** | 0.654 | 2 | 0.947 | 7 |
| Digits | 1797 × 64 | 10 | **MDS, decisive** | 0.534 | 10 | 0.884 | 16 |

---

## PART 1 — What to add, and where

### 1 Introduction (exists — minor tightening)
The current intro is strong. Two fixes:

- **Typo:** "an error value in the interval [1, +1]" should read **[−1, +1]**. The
  minus sign was lost.
- **Resolve the dataset TO-DO:** you write "dataset-agnostic [TO-DO: … at the
  moment there are 4 datasets]". These are not in conflict — say it in one
  sentence: *"The toolkit is dataset-agnostic: it accepts any labelled dataset
  (built-in or uploaded as CSV); this report uses four built-in datasets — Iris,
  Wine, Breast-cancer and Digits — chosen to span a range of dimensionalities (4
  to 64 features) and difficulties."* That converts a TO-DO into a strength.

### 2.1 Projections (exists — add one concrete consequence)
Good as is. Add a single sentence naming the practical stakes so the problem
feels real: *"Because the analyst cannot see which is which, a projection artifact
can silently drive a real decision — a cluster that is an artifact of PCA may be
read as a finding."* Keep your FP/FN bullet definitions; they are correct.

### 2.2 Patterns and requirements (exists — align numbering with the build)
The six requirements are good and map cleanly onto what you built. Two small
additions:

- Add a seventh requirement that you actually satisfy: *"reach its verdict with a
  measure of confidence, and decline to decide when the evidence is thin."* You
  implemented exactly this (the bootstrap CI + tie threshold), so claim it.
- Cross-reference forward: after the list, one line — *"Sections 3–4 show how each
  requirement is met; Section 6 reports the evidence."*

### 3.1 Ground truth (exists — good; make the two-part structure explicit)
This paragraph is doing the most important conceptual work and is currently a
little dense. Split it into two short ideas, keeping your own wording:

1. **Geometric ground truth** — the high-D pairwise distances define each point's
   true neighbours; a faithful projection preserves them. (No labels needed.)
2. **Label ground truth** — labels are a *second, assumed* reference, not trusted
   blindly; k-means (3.4) tests whether they actually correspond to clusters.

Then keep your closing sentence about multiple assessments beating a single
metric — it is the right thesis. **Do not** reintroduce any "relationship
phase / mapping phase" wording here (see Part 2, conflict C4).

### 3.2 Intrusion and extrusion (exists — resolve the "work in progress")
The mechanics you wrote are correct. To close the `[TO-DO: Work in progress]`
after the weighting bullets, add:

- **The identity and the normaliser.** State that `error_i = FP_i − FN_i ∈
  [−1,+1]`, each side normalised by the maximum possible rank penalty
  `W = k(2n − 3k − 1)/2`. This is the one formula the section is missing.
- **Why weighting is essential (one line):** with plain counts at fixed k the
  number of intrusions and extrusions is forced equal, so every point would read
  exactly 0 — the severity weighting is what makes the score informative. (You
  imply this; state it outright, it is a strong point.)
- **Class-aware vs geometric** deserves one sentence: by default only cross-class
  intrusions and same-class extrusions count (class-structure fidelity); a
  geometric mode counts any broken neighbour (pure fidelity).

### 3.2.1 Classic metric (exists — add the empirical punchline)
You describe it correctly. Add the number that makes the point land: *"Because
`d_diff` pools all other-class points it is large by construction, so on Wine the
classic metric places ≈ 99% of points on the FP side — it measures separability,
not fidelity, which is exactly why the intrusion metric is the primary measure."*

### 3.3 Precision, recall and F1 score (EMPTY — the biggest gap)
This is the section most in need of writing, and you now have a full standalone
note for it (`Precision_Recall_and_Classic_Metric.md`). Distil it to ~4 short
paragraphs:

1. **What it asks and why it is separate.** Fidelity (3.2) asks "did the
   projection preserve the data's structure?"; precision/recall asks "is the
   expected class structure present in the data at all?" It is computed on a
   similarity graph over the high-D points, never on a projection.
2. **Precision** — is a point surrounded by its own class? (weighted by edge
   strength). **Recall** — is each class held together? (by cardinality, with the
   α knob between "connected" and "clique").
3. **F1** — the harmonic mean of the two, high only when *both* are high. Give the
   formula `F1 = 2PR/(P+R)` and one line on why harmonic (it is dominated by the
   smaller value, so a class that is pure but fragmented, or connected but impure,
   scores low).
4. **What it shows here.** Quote the table: recall = 1.00 on every dataset (no
   class fragments at k = 15), so precision (0.90–0.92) is the binding constraint;
   per-class F1 locates the hard classes (e.g. Iris 0.996 / 0.914 / 0.902).
   Explicitly note the link to method choice you hint at: the F1-optimal k is what
   the annealing (4.2) searches for.

**Important framing (see Part 3):** write this section in your own terms and cite
the precision/recall-for-DR idea as *inspiration only*. Do not quote or reproduce
the reserved paper.

### 3.4 k-means (partial — close the TO-DO)
You have the right justification (shared Euclidean geometry). Add the three
concrete outputs the toolkit produces:

1. **ARI(high-D k-means vs labels)** — *are the classes clusters?* Quote: Wine
   0.897 (yes), Iris 0.620 and Breast-cancer 0.654 (partly), Digits 0.534.
2. **Matched accuracy & per-class capture** — the interpretable version of ARI
   (Hungarian-matched); Wine reaches 0.966.
3. **ARI(2-D vs high-D), per method** — *does the projection preserve the
   partition?* — this is label-free and becomes a verdict component.
4. **The silhouette scan** — quote the Iris result: silhouette prefers **2**
   clusters against **3** classes, which the tool flags automatically. This is the
   single most persuasive concrete example in the whole report — it shows the tool
   catching a labels-≠-clusters case on its own.

### 3.5 Projection methods (exists — fine)
Correct and sufficient. Optionally add one line on cost, since it matters for
Setup and for the MDS caveat: PCA is closed-form and deterministic; MDS is
iterative, stochastic (seed-dependent), and O(n²), which is why large datasets are
subsampled.

### 3.6 Visualizations and interactions (exists — close the TO-DO, add the new UI)
Your description predates the latest build. Update it to include what now exists:

- The **verdict banner + "Verdict & Analysis" tab + popup** (the headline decision
  is now on the page, not only in the console).
- The **"see classes" toggle** on the error space (recolours the three plots by
  class so distortion can be compared against class membership).
- The **three colour modes** — severity-weighted, **TP-aware**, and raw signed
  error — and the **size = |FP|+|FN|** channel (tie this to 4.1).
- Keep your histogram / centroid / decomposition / table descriptions; they are
  accurate.

### 3.7 Weighted MDS (exists — good; add one caveat)
Your explanation is accurate. Add the honest caveat that also appears in the
review: weighted MDS reshapes only the MDS layout while PCA and the high-D
fidelity reference stay unweighted — so if the *verdict* is run with weighted MDS
on, PCA and MDS saw different feature scalings, which the report should flag as a
deliberate, disclosed choice.

### 4.1 Severity-weighted metric (exists — but now OUT OF DATE; see conflict C1)
This is the most important **conflict** to fix. Your draft says the naive repair
"is to multiply the error by the magnitude of distortion behind it" and lists it as
a future fix. But the toolkit **already implements** the proper fix and the naive
product was tried and rejected. Rewrite this section to match reality:

- The naive product `err × (FP+FN) = FP² − FN²` is quadratic in small values and
  **crushes ≈ 92% of Wine points** into the near-white band (vs 32% for raw
  error) — so it was rejected.
- The implemented metric is the **signed geometric mean**
  `S = sign(err)·√(|err|·(FP+FN))`, which keeps ≈ 34% in the neutral band
  (essentially matching raw error) while encoding severity, and preserves the sign.
- The newer **TP-aware colour mode** goes further: whiteness is earned only by low
  total distortion, so a balanced-but-distorted point greys out instead of
  masquerading as a true positive. Present size + TP-aware as the full resolution.

### 4.2 Simulated annealing (partial — close the TO-DO)
You have the intuition. Add the specifics that make it credible:

- **Why a metaheuristic at all:** F1(k) is neither concave nor convex (recall rises
  with k, precision falls non-monotonically), so gradient/ternary search is unsafe.
- **The temperature calibration** (a genuine engineering point worth claiming): the
  initial temperature is set from the standard deviation of a few probe evaluations,
  making acceptance scale-invariant — without it the search degenerates to a random
  walk or a hill-climber depending on the objective's scale.
- **Evidence it works:** on Wine it finds k = 10 (F1 = 0.9502) in 15 evaluations vs
  59 exhaustive, with ~10 accepted uphill moves; the objective is flat near the
  optimum so the exact k is not critical.

### 4.3 PCA-versus-MDS verdict (partial — close the TO-DO)
Describe the actual mechanism you built:

- **Four independent, same-direction criteria**, each normalised so 1 is better:
  fidelity (0.40), trustworthiness (0.20), continuity (0.20), cluster-preservation
  ARI (0.20). Say the weights are **stated, not tuned** (avoids circularity) and
  that all four are always shown.
- **Confidence:** a paired bootstrap 95% CI on the per-point distortion difference
  plus Cohen's d; "decisive" only when the CI excludes zero *and* the margin ≥
  0.02. Quote the discriminating results: Breast-cancer decisive for PCA, Digits
  decisive for MDS, Wine and Iris not decisive.
- **The honest limitation** (also a strength): the CI captures per-point variation
  but not MDS's run-to-run stochasticity, so "decisive" is scoped to the given
  layouts.

### 5.1 Datasets (EMPTY — quick to write)
A short table is enough. Use the four you have with their shapes and one-line
character notes:

- **Iris** 150×4, 3 classes — low-dimensional; two classes overlap (the
  cautionary labels-≠-clusters case).
- **Wine** 178×13, 3 classes — features on very different numeric scales (motivates
  standardization); the default.
- **Breast-cancer** 569×30, 2 classes — many correlated features (favours PCA).
- **Digits** 1797×64, 10 classes — high-dimensional and large (exercises
  subsampling and is where MDS wins decisively).

### 5.2 Preprocessing (EMPTY — quick to write)
Two paragraphs: (1) **Standardization on by default** — why it matters for
distance-based methods (without it a large-scale feature like Wine's proline
dominates and PCA just recovers that axis); a toggle lets you show the effect. (2)
**Stratified subsampling** for large data (Digits), preserving original indices so
any reported point traces back, because MDS is O(n²).

### 5.3 Libraries (EMPTY — quick to write)
List the stack and the role of each: NumPy (array maths, rank computations),
scikit-learn (PCA, MDS, k-means, mutual information, silhouette, standardization),
pandas (CSV + point table), Flask (thin server, captures console, persists runs),
Plotly.js (interactive linked-brushing charts), SciPy (Hungarian matching for
cluster↔label). Note matplotlib is used only by the CLI, not the web app.

### 5.4 Architecture (EMPTY — quick to write)
One decision explains the whole design: **the compute core carries no
UI/plotting code**, so the same engine serves a CLI, a brushing window, and the web
app. Then the three layers: compute core (metrics/projections/analysis, pure
Python) → Flask server (runs the pipeline, returns raw arrays as JSON, persists
runs) → browser (builds every Plotly figure and owns all linked-selection logic).
The philosophy: the server sends *data, not pictures*, so selection and re-styling
happen instantly client-side.

### 6 Verdicts and results (EMPTY — the second-biggest gap)
This is where the four-dataset table earns its place. Structure it as:

1. **Headline verdicts table** (the one at the top of this guide) with the CI and
   decisive/not-decisive column.
2. **Component breakdown** — that Breast-cancer wins for PCA on cluster
   preservation (PCA's regime: 30 correlated features) while Wine's components
   *disagree* (MDS leads fidelity, PCA leads cluster) which is *why* Wine is not
   decisive.
3. **The labels-≠-clusters finding** — Iris (silhouette 2 vs 3 classes; ARI 0.620)
   as the worked example of the ground-truth check firing.
4. **Precision/recall/F1** — recall = 1.0 everywhere, precision the binding
   constraint; per-class F1 locating hard classes.
5. **Parameter search** — the annealing efficiency result.
6. **Method agreement** — the quadrant census (both-FN dominates; small mixed
   populations mean the methods largely agree, another reason two verdicts are not
   decisive).

### 7 Conclusion (EMPTY — write last)
Three short paragraphs: (1) **Contribution** — restate the instrument framing
(per-point, signed, comparative, decision-ready, self-checking). (2) **Honest
limitations** — the k-NN scale is a chosen parameter PCA/MDS lack; verdict weights
are editorial; single MDS run; clean datasets. (3) **Future work** — t-SNE/UMAP
panels (the supervised metrics would apply natively), multi-seed MDS + weight
sensitivity, real-world data, scale beyond a few thousand points.

---

## PART 2 — Possible conflicts (resolve these)

**C1 — Severity metric: draft vs implementation (most important).**
§4.1 presents "multiply the error by the magnitude" as a *future* fix, but the
code already implements the signed geometric mean and the TP-aware mode, and the
naive product was tried and rejected for crushing 92% of points. If left as-is the
report contradicts the artifact. **Fix:** rewrite §4.1 as "done", per Part 1.

**C2 — "there are 4 datasets" TO-DO vs "dataset-agnostic".**
Not a real contradiction, but it reads like one. State that the *engine* is
dataset-agnostic and the *report* evaluates four built-ins. (Intro + §5.1.)

**C3 — Adaptive-k described as present, but was UI-disabled.**
§3.2 describes adaptive-k as a feature. It exists in the engine but was commented
out in the UI; it has since been re-enabled. Make sure the version you submit has
it enabled, or the demo won't match the report. Verify before the exam.

**C4 — Reserved-paper vocabulary.**
Earlier drafts used "relationship phase / mapping phase". Those are the reserved
paper's terms and must not appear. The current draft mostly avoids them — keep it
that way, and describe the two-questions idea (data vs projection) in your own
words. Also ensure the **code** contains no such references (the analysis module
was sanitised; confirm before sharing source).

**C5 — F1 vs "f-score".**
If §3.3 and the code use "f-score" loosely while meaning F1, state once that F1 is
F_β with β = 1 and that the toolkit reports an explicit F1. Otherwise a reader may
think two different metrics are in play.

**C6 — "digits" and MDS cost.**
If you quote Digits (1797 points) results, you must mention subsampling for MDS in
§5.2, otherwise a reader will object that MDS at n = 1797 is O(n²). The two must be
consistent.

**C7 — Verdict "decisive" strength vs MDS stochasticity.**
Don't over-claim. If §4.3/§6 call a result "decisive", pair it with the scope
caveat (one MDS layout), or a careful reader will challenge it.

---

## PART 3 — References & research grounding

The report currently reads as an engineering write-up. To make it defensible *as a
research paper*, add a short **Related Work** paragraph (or fold references into
§2–§3) and a reference list. Below: what to cite, split into the **basis** you
build on vs **your contribution**, and where each citation attaches.

### 3A. The basis you build on (cite these)
These are established ideas the project *uses*; citing them is what separates
"reused" from "claimed", and protects you from the charge of over-claiming.

- **Trustworthiness & continuity / rank-based neighbourhood preservation** — the
  direct ancestor of your intrusion/extrusion metric. Cite the original
  formulation and a survey. Attach to §3.2.
  - Venna & Kaski, *Neighborhood Preservation in Nonlinear Projection Methods*
    (2001) and *Information Retrieval Perspective to Nonlinear Dimensionality
    Reduction* (JMLR 2010).
  - Lee & Verleysen, *Quality assessment of dimensionality reduction:
    Rank-based criteria* (Neurocomputing 2009) — the co-ranking matrix; the
    natural formal home for "intrusion/extrusion".
- **The co-ranking matrix** as a unifying view of exactly your FP/FN split —
  Lee & Verleysen (2009). Attach to §3.2; it is the strongest scholarly anchor for
  your metric and worth reproducing as the theoretical frame.
- **PCA** — Jolliffe, *Principal Component Analysis* (2002). Attach to §3.5.
- **MDS / stress** — Borg & Groenen, *Modern Multidimensional Scaling* (2005);
  Kruskal (1964) for stress. Attach to §3.5 and the weighted-MDS §3.7.
- **k-means & cluster validity** — MacQueen (1967) for k-means; Rousseeuw (1987)
  for silhouette; Hubert & Arabie (1985) for the Adjusted Rand Index. Attach to
  §3.4.
- **Precision / recall & the F-measure** — van Rijsbergen, *Information Retrieval*
  (1979) for F_β; this justifies your F1 as a standard IR construct, independent of
  any DR paper. Attach to §3.3.
- **Simulated annealing** — Kirkpatrick, Gelatt & Vecchi (1983); Metropolis et al.
  (1953). Attach to §4.2.
- **Mutual-information feature relevance** — Kraskov et al. (2004) or the
  scikit-learn `mutual_info_classif` reference. Attach to §3.7.
- **Bootstrap confidence intervals & Cohen's d** — Efron & Tibshirani, *An
  Introduction to the Bootstrap* (1993); Cohen (1988). Attach to §4.3 — this is
  what makes the verdict statistically literate.
- **Projection-quality VA systems** (to position your tool) — e.g. t-SNE (van der
  Maaten & Hinton 2008) and UMAP (McInnes et al. 2018) as the techniques you would
  extend to; and a DR-quality VA survey (Nonato & Aupetit, *Multidimensional
  Projection for Visual Analytics*, IEEE TVCG 2019) to site the field. Attach to
  §2 / Related Work.

### 3B. The precision/recall-for-DR idea (inspiration only — do NOT cite or quote)
The specific precision/recall-on-a-graph framing came from the reserved paper your
professor shared. Treat it as **background inspiration** that motivated §3.3, and
express it in your own terms grounded in the *generic* IR precision/recall (3A,
van Rijsbergen) and the co-ranking literature. Do **not** quote it, reproduce its
formulas, name its authors, or cite it. If asked, describe the idea as your own
adaptation motivated by standard IR metrics.

### 3C. Your contribution (claim these clearly)
State explicitly, ideally in the Introduction and Conclusion, what is *new* here —
this is what makes it a project worth grading, and keeping it separate from 3A is
what makes the claims credible:

1. **A per-point, *signed*, *comparative* fidelity instrument** — turning
   trustworthiness/continuity from an aggregate scalar into a signed
   FN→True→FP per-point value applied to *two* methods on one scale, with linked
   brushing and a quadrant census.
2. **The double encoding done correctly** — recognising that colour = FP−FN is
   ambiguous, rejecting the naive product (measured 92% crush), and using the
   signed geometric mean + size + the TP-aware mode.
3. **A mechanical verdict with confidence** — a stated-weight composite plus a
   paired bootstrap CI that *declines* to decide when evidence is thin.
4. **Bringing supervised precision/recall/F1 + k-means ground-truth checking to the
   PCA/MDS comparison**, with simulated-annealing selection of the neighbourhood
   scale (and the scale-invariant temperature calibration).
5. **The synthesis as an instrument** — three questions (where is the error? is the
   structure real? which method, by how much?) answered in one linked interface.

### 3D. How to phrase the boundary (a sentence you can reuse)
> "The neighbourhood-preservation metric builds on rank-based quality criteria
> [Lee & Verleysen 2009; Venna & Kaski 2010]; the supervised precision/recall
> analysis adapts standard information-retrieval measures [van Rijsbergen 1979] to
> the cluster-structure question; the per-point signed comparative encoding, the
> severity-aware colour design, and the confidence-scored PCA-vs-MDS verdict are
> the contributions of this work."

---

## PART 4 — Code / report improvements worth doing

Prioritised; the first few are the highest impact for credibility.

### Do before submitting / the exam
1. **Fix §4.1 to match the implemented severity metric** (conflict C1). The report
   currently describes the *rejected* approach as future work.
2. **Confirm adaptive-k is enabled in the UI** and that the submitted build matches
   the report (conflict C3).
3. **Verify the source has no reserved-paper references** (conflict C4) before
   sharing code.
4. **Fix the `[1, +1]` → `[−1, +1]` typo** and the "an layer" / "blindingly"
   wordings in the intro/ground-truth.
5. **Write §5 and §6** — they are quick given the numbers table, and an empty
   Results section is the most damaging gap in a VA report.

### High-value substance
6. **Scope every "decisive" claim** to the given MDS layout, or nest the bootstrap
   inside a few MDS re-runs so the claim is honest (conflict C7). This is the
   single most likely question from a rigorous examiner.
7. **Expose recall's α and the verdict weights as controls** (even read-only in the
   report as "stated weights"), turning hidden assumptions into visible ones.
8. **Add the classes-≠-clusters Iris example prominently** (§3.4 and §6) — it is
   your best concrete evidence that the ground-truth check does real work.
9. **Report both macro and micro F1** so imbalance is visible (Digits, 10 classes,
   is where this matters).

### Code robustness (from the review)
10. **`compute_feature_weights` default** is `inverse_variance`, which is unhandled
    and raises — change the default to `mutual_info` or implement it (and note it
    is a no-op under standardization). Latent crash for any direct caller.
11. **Re-frame the "untrustworthy" flag** — `|error| > threshold` selects the
    *best* points on clean data; the misleading points are `error > +threshold`
    (intrusions). Split the two directions.
12. **Sparse/vectorise precision/recall and cache MDS** when only the metric k
    changes — the current O(n²) dense graph and Python per-vertex loops cap
    scalability, and re-running MDS on every k change is wasteful.
13. **Compute the "vs exhaustive" annealing baseline from the actual k-range**, not
    a hard-coded 59 (it is wrong when k_hi scales with n).
14. **Finish or remove the CSV upload path** — the UI implies it works but the
    handler was disabled; a dead affordance hurts a demo.

### Presentation / figures for the report
15. Add **three figures** that carry a lot of explanatory weight: (a) the
    FN→True→FP scale with a worked intruder/extruded example; (b) the colour
    comparison (raw vs naive-product vs severity, with the 32/92/34% crush numbers);
    (c) the annealing trace. You already generate versions of these.
16. Add a **method-agreement quadrant figure** for one dataset — it makes "the
    methods largely agree, hence not decisive" visual rather than asserted.
17. Consider a **colour-blind-safe palette** option and mention accessibility in
    §3.6 — a cheap credibility win for a visualization project.

---

### One-paragraph summary of the plan
Finish §3.3, §5 and §6 first (they are empty and quick), rewrite §4.1 to match the
implemented severity fix (the main conflict), add a Related Work paragraph that
cites the rank-based-quality and IR-metric literature while keeping the
precision/recall-for-DR idea as your own adaptation, and state your five
contributions explicitly in the Introduction and Conclusion. Then harden the two
claims a careful examiner will probe: the "decisive" verdict (scope it to the MDS
layout) and the neighbourhood-scale caveat (the k-NN parameter PCA/MDS do not
themselves possess).
