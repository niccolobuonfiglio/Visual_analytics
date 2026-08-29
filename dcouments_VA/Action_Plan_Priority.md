# Prioritized Action Plan

A single ordered to-do list for finishing the report and hardening the project,
sequenced by importance (impact × urgency). Each task states **why it ranks here**,
**how to do it** in concrete steps, the **effort**, and the **done-when** test.

Work top to bottom. Tasks 1–6 are the ones that most affect your grade and are
prerequisites for a coherent submission; 7–12 add rigor and research validity;
13–18 are polish and robustness.

All numbers quoted are from the current toolkit (seed 42, intrusion metric,
k = 10, standardization on) and are safe to paste:

| Dataset | n×p | classes | verdict | ARI(labels) | silhouette k | F1 | anneal k |
|---|---|---|---|---|---|---|---|
| Wine | 178×13 | 3 | MDS, not decisive (+0.006) | 0.897 | 3 | 0.943 | 10 |
| Iris | 150×4 | 3 | PCA, not decisive (+0.002) | 0.620 | **2** | 0.937 | 13 |
| Breast-cancer | 569×30 | 2 | **PCA, decisive (+0.021)** | 0.654 | 2 | 0.947 | 7 |
| Digits | 1797×64 | 10 | **MDS, decisive** | 0.534 | 10 | 0.884 | 16 |

---

## TIER A — Do first (correctness & the empty core of the report)

### 1. Rewrite §4.1 to match what the code actually does
**Why #1:** it is a direct contradiction between report and artifact — the single
most damaging thing an examiner can catch. Your draft lists "multiply the error by
the magnitude of distortion" as a *future fix*, but the toolkit already implements
the real fix and that naive product was tried and rejected.
**How:**
1. Delete the `[TO-DO: future fix]` line.
2. State the problem: colour = `err = FP − FN` is ~0 for both a faithful point
   (FP≈FN≈0) and a balanced-severe point (FP≈FN large), so both render white.
3. State the rejected fix and *why*: the naive product `err×(FP+FN) = FP²−FN²` is
   quadratic in small values and crushes **≈92%** of Wine points into the
   near-white band (vs 32% for raw error).
4. State the implemented fix: signed geometric mean
   `S = sign(err)·√(|err|·(FP+FN))`, which keeps ≈34% neutral (≈ raw error) while
   encoding severity, and preserves the sign.
5. Add the two channels that finish the job: **size = |FP|+|FN|** and the
   **TP-aware colour mode** (whiteness earned only by low total distortion, so a
   balanced-severe point greys out instead of masquerading as a true positive).
**Effort:** ~1 hour. **Done when:** §4.1 describes the geometric-mean + size +
TP-aware solution as *implemented*, with the 32/92/34% numbers, and no "future
fix" wording remains.

### 2. Write §6 (Verdicts and results) — the empty core
**Why #2:** an empty Results section is the biggest structural hole in a VA report;
everything else exists to produce these numbers.
**How — six short subsections:**
1. **Headline table** (the one above) with a decisive/not-decisive column and the
   95% CI.
2. **Component breakdown:** Breast-cancer wins for PCA on cluster-preservation
   (its regime: 30 correlated features); Wine's four components *disagree* (MDS
   leads fidelity/trust, PCA leads continuity/cluster) which is *why* it is not
   decisive.
3. **Labels-≠-clusters finding:** Iris — silhouette prefers 2 clusters vs 3
   classes, ARI 0.620 — the ground-truth check firing on its own.
4. **Precision/recall/F1:** recall = 1.00 everywhere (no fragmented classes at
   k=15), precision (0.90–0.92) is the binding constraint; per-class F1 locates
   hard classes (Iris 0.996/0.914/0.902).
5. **Parameter search:** annealing finds k=10 on Wine in 15 evals vs 59 exhaustive.
6. **Method agreement:** the quadrant census — both-FN dominates, small mixed
   populations, so the methods largely agree (another reason two verdicts are not
   decisive).
**Effort:** ~3 hours. **Done when:** all six subsections written, every number
traceable to a run.

### 3. Write §3.3 (Precision, recall and F1) — the biggest methodology gap
**Why #3:** it is empty, it is core methodology, and it is the part most tied to
your novelty claim. You already have the full standalone note
(`Precision_Recall_and_Classic_Metric.md`) — distil it.
**How — four short paragraphs:**
1. *What it asks and why separate:* fidelity (§3.2) asks "did the projection
   preserve the data's structure?"; precision/recall asks "is the class structure
   present in the data at all?" — computed on a similarity graph over the high-D
   points, never on a projection.
2. *Precision* (is a point among its own class? weighted by edge strength) and
   *recall* (is each class held together? by cardinality, with the α knob between
   "connected" and "clique").
3. *F1* = `2PR/(P+R)`, high only when both are high; one line on why harmonic (it
   is dominated by the smaller value).
4. *What it shows here:* the recall=1.0 / precision-bound result and per-class F1.
**Critical:** write it in your own terms; ground it in standard IR precision/recall
(van Rijsbergen 1979), and treat the reserved paper as inspiration only — do not
cite, quote, or name it.
**Effort:** ~2 hours. **Done when:** the four paragraphs exist with the F1 formula
and the numbers, and contain no reserved-paper vocabulary.

### 4. Verify the submitted build matches the report (adaptive-k, sanitized code)
**Why #4:** cheap to check, catastrophic if wrong — if the demo doesn't match the
report, or the source still names the reserved paper, that overshadows the content.
**How:**
1. Confirm **adaptive-k is enabled** in the UI (it was previously commented out in
   `collectParams`; it has been re-enabled — verify the version you submit has it).
2. Grep the source for reserved-paper references before sharing:
   `grep -rniE "van der hoorn|arleo|paulovich|relationship phase|mapping phase" *.py`
   — must return nothing (the analysis module was sanitized; confirm).
3. Run all four datasets once end-to-end to confirm no crashes.
**Effort:** ~30 min. **Done when:** adaptive-k toggles in the UI, grep is clean,
all four datasets run.

### 5. Write §5 (Setup and platform) — four quick empty sections
**Why #5:** four empty subsections, but each is short and factual; leaving them
blank looks unfinished.
**How:**
- **§5.1 Datasets:** the four-row table with one-line character notes (Iris = the
  overlap case; Wine = different feature scales; Breast-cancer = correlated
  features favour PCA; Digits = large, high-dim, exercises subsampling).
- **§5.2 Preprocessing:** standardization on by default (why: without it a
  large-scale feature like Wine's proline dominates and PCA just recovers that
  axis); stratified subsampling for Digits, preserving indices, because MDS is
  O(n²).
- **§5.3 Libraries:** NumPy, scikit-learn (PCA/MDS/k-means/MI/silhouette), pandas,
  Flask, Plotly.js, SciPy (Hungarian matching); matplotlib only in the CLI.
- **§5.4 Architecture:** compute core (no UI/plotting) → Flask (returns raw arrays
  as JSON, persists runs) → browser (builds all Plotly, owns linked selection);
  "server sends data, not pictures."
**Effort:** ~1.5 hours. **Done when:** all four subsections have content.

### 6. Fix the small errors and stale descriptions in existing sections
**Why #6:** trivial effort, and typos/contradictions undercut credibility.
**How:**
1. Intro: `[1, +1]` → **`[−1, +1]`**; fix "an layer" → "a layer"; "blindingly" →
   "blindly".
2. §2.2: turn the "4 datasets" TO-DO into a strength — "the engine is
   dataset-agnostic; this report evaluates four built-ins."
3. §3.6: update to include the **new UI** — the verdict banner + "Verdict &
   Analysis" tab + popup, the **"see classes"** toggle, and the **three colour
   modes**. The current text predates these.
4. §3.2: close the `[TO-DO]` by adding the `error = FP−FN` normaliser
   `W = k(2n−3k−1)/2` and the one-line "why weighting is essential" (plain counts
   force intrusions=extrusions, so every point reads 0).
**Effort:** ~1 hour. **Done when:** no typos, §3.6 matches the current UI, §3.2
has the formula.

---

## TIER B — Do next (rigor & research validity)

### 7. Add a Related Work paragraph + reference list
**Why:** this is what elevates the document from an engineering write-up to a
research report, and it is what protects your novelty claims.
**How:** add a short Related Work paragraph (or fold citations into §2–§3) citing
the *basis* you build on, attaching each to its section:
- Rank-based quality / co-ranking matrix (Lee & Verleysen 2009) and
  trustworthiness/continuity (Venna & Kaski 2001/2010) → §3.2 (your metric's
  ancestor).
- PCA (Jolliffe 2002), MDS/stress (Borg & Groenen 2005; Kruskal 1964) → §3.5, §3.7.
- k-means (MacQueen 1967), silhouette (Rousseeuw 1987), ARI (Hubert & Arabie 1985)
  → §3.4.
- F-measure (van Rijsbergen 1979) → §3.3 (grounds F1 independently of any DR paper).
- Simulated annealing (Kirkpatrick 1983; Metropolis 1953) → §4.2.
- Bootstrap (Efron & Tibshirani 1993), Cohen's d (Cohen 1988) → §4.3.
- Field positioning: t-SNE (van der Maaten & Hinton 2008), UMAP (McInnes 2018), a
  DR-quality VA survey (Nonato & Aupetit, TVCG 2019) → §2.
**Effort:** ~2 hours. **Done when:** a reference list exists and each citation is
attached to a claim. *Verify exact years/venues when formatting the bibliography.*

### 8. State your five contributions explicitly (Intro + Conclusion)
**Why:** separating "reused" from "new" is what makes the novelty credible; graders
look for this directly.
**How:** in the Introduction and Conclusion, claim exactly these and no more:
(1) per-point *signed comparative* fidelity instrument for two methods on one
scale; (2) the double encoding done right (rejecting the naive product, using
geometric-mean + size + TP-aware); (3) a mechanical verdict *with confidence* that
declines when evidence is thin; (4) bringing supervised precision/recall/F1 +
k-means ground-truth checking to the PCA/MDS comparison with annealed k-selection;
(5) the synthesis as an instrument answering three questions in one interface.
Use the boundary sentence from the completion guide (Part 3D) verbatim if useful.
**Effort:** ~1 hour. **Done when:** both sections list the five, phrased as
adaptation/synthesis, not invention.

### 9. Scope every "decisive" claim to the MDS layout
**Why:** the most likely question from a rigorous examiner. The bootstrap CI
captures per-point variation but **not** MDS's run-to-run stochasticity, so
"decisive" is currently over-stated.
**How (report):** wherever §4.3/§6 say "decisive", add "for these layouts" and one
sentence noting MDS is a single stochastic run.
**How (code, optional but stronger):** run MDS ~5 times and report the verdict
spread, or nest the bootstrap inside the re-runs.
**Effort:** 20 min (report) / ~2 hours (code). **Done when:** no "decisive" claim
stands without the scope caveat.

### 10. Write §7 (Conclusion)
**Why:** the report needs a landing; write it after §6 so it reflects the results.
**How — three short paragraphs:** (1) contribution (the instrument framing);
(2) honest limitations (k-NN scale is a chosen parameter PCA/MDS lack; editorial
verdict weights; single MDS run; clean datasets); (3) future work (t-SNE/UMAP
panels where the supervised metrics apply natively; multi-seed MDS + weight
sensitivity; real-world data; scale beyond a few thousand points).
**Effort:** ~1 hour. **Done when:** the three paragraphs exist and the limitations
are stated plainly (a maturity signal, not an apology).

### 11. Add the three highest-value figures
**Why:** a VA report is judged partly on its visuals; these three carry the most
explanatory weight and you already generate versions of them.
**How:** (a) the FN→True→FP scale with a worked intruder/extruded example;
(b) the colour comparison raw vs naive-product vs severity with the 32/92/34%
crush numbers; (c) the annealing trace. Optionally (d) a quadrant-census figure to
make "the methods agree, hence not decisive" visual.
**Effort:** ~2 hours. **Done when:** figures are captioned and referenced in text.

### 12. Add the two honest caveats that double as strengths
**Why:** naming a limitation before the examiner does is a credibility win.
**How:** (a) §3.7 — weighted MDS reshapes only the MDS layout while PCA and the
high-D reference stay unweighted, so a weighted-MDS verdict compares different
feature scalings (a disclosed choice). (b) §3.4/§6 — foreground the Iris
labels-≠-clusters example as evidence the ground-truth check does real work.
**Effort:** ~45 min. **Done when:** both caveats appear.

---

## TIER C — Polish & code robustness (if time permits)

### 13. Fix the `compute_feature_weights` latent crash
**Why:** a real bug — the default `method="inverse_variance"` is unhandled and
raises; it only works because the server overrides the default.
**How:** change the default to `"mutual_info"`, or implement `inverse_variance`
(weight `1/var`, and note it is a no-op under standardization since every column
has unit variance). **Effort:** 30 min. **Done when:** calling with defaults does
not raise.

### 14. Re-frame the "untrustworthy" flag
**Why:** `|error| > threshold` selects the *best* points on clean data; the
misleading ones are `error > +threshold` (intrusions).
**How:** split the two directions in the report and (ideally) the UI — "misleading
closeness" (`error > +t`) vs "lost neighbours" (`error < −t`). **Effort:** ~1 hour.
**Done when:** the flag no longer conflates opposite meanings.

### 15. Report both macro and micro F1
**Why:** macro-averaging (current) can flatter tiny classes; on Digits (10 classes)
the gap is informative. **How:** compute a micro (pooled) global F1 alongside the
macro one and show both. **Effort:** ~45 min. **Done when:** both appear; a gap
signals imbalance.

### 16. Fix the "vs exhaustive" annealing baseline
**Why:** it is hard-coded to 59, which is wrong when the k-range scales with n.
**How:** compute the exhaustive count from the actual `[k_lo, k_hi]` used and
return it in the payload. **Effort:** 30 min. **Done when:** the "N/M" denominator
matches the real range.

### 17. Expose recall's α and the verdict weights
**Why:** turns hidden assumptions into visible controls (recall currently uses the
lenient α=0 silently; weights are fixed at 0.40/0.20/0.20/0.20).
**How:** add an α slider to the Analysis controls and, at minimum, state the
weights in the verdict card (they are already shown per-component). **Effort:**
~1.5 hours. **Done when:** α is adjustable and the weights are visible.

### 18. Finish or remove the CSV upload path
**Why:** the UI implies CSV upload works but the handler was disabled — a dead
affordance hurts a live demo. **How:** either wire it through (`load_dataset`
already supports CSV) or hide the upload zone until ready. **Effort:** ~1 hour
(wire) / 10 min (hide). **Done when:** dropping a CSV either works or the
affordance is gone.

---

## Suggested schedule

- **Session 1 (half day):** Tasks 1, 4, 6 — remove the contradictions and errors
  first, so nothing you write next fights the code.
- **Session 2 (full day):** Tasks 2, 3, 5 — fill the empty core (§3.3, §5, §6).
- **Session 3 (half day):** Tasks 7, 8, 10 — Related Work, contributions,
  Conclusion — the research framing.
- **Session 4 (half day):** Tasks 9, 11, 12 — harden the verdict claim, add figures
  and caveats.
- **Remaining time:** Tier C in listed order.

**The single most important sequencing rule:** do Task 1 before writing anything
else in §4/§6, because the severity-metric contradiction propagates — several later
sections reference the colour metric, and they must all describe the *implemented*
version.
