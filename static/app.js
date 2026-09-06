/* Projection Fidelity Console: frontend logic */

const $ = (id) => document.getElementById(id);
const state = {
  result: null,        // latest run result
  metric: 'intrusion',
  weight: 'rank',

  esView: 'error',         // 'error' | 'class' -> Error Space colour-by mode
  esSel: new Set(),    // Error Space: shared selection across all 3 plots
  esActiveQuad: null,  // Error Space: currently active quadrant button (or null)
  tableSort: { key: 'worst_f1', dir: 1 },
  trustSelSort: { key: 'pca', dir: -1 },   // sort of the trust selection table
  wmdsSelSort: { key: 'move', dir: -1 },   // sort of the Mahalanobis selection table
  adjMethod: 'locality',   // 'locality' | 'mahalanobis', adjusted-distance method
  featWeightOverride: {},  // {featureIndex: multiplier} for locality weighting
  // Trust Analysis state
  trustMetric: 'f1',       // 'f1' | 'js_fidelity' | 'euclidean' | 'mrre'
  trustView: 'score',      // 'score' | 'class' -> colour the trust plots by
  brushMode: 'within',     // 'within' | 'all'      -> boxplot recompute mode
  cmpMethod: 'MDS',        // method for the metric-vs-metric scatter
  cmpX: 'f1',              // x-axis metric of the comparison scatter
  cmpY: 'js_fidelity',     // y-axis metric of the comparison scatter
  trustSel: null,          // Set of selected point indices (null = all)
  filterClasses: new Set(),// class codes to keep (empty = all)
  attrFilter: null,        // {col, lo, hi} attribute window (null = none)
  scoreFilter: null,       // {lo, hi} keep points whose active-metric score (PCA & MDS) is in [lo,hi]
};

/* error scale colour (matches CSS: red FN → white → blue FP) */
const FN = [210, 75, 82], TRUE_ = [238, 241, 244], FP = [63, 124, 196];
function lerp(a, b, t){ return a + (b - a) * t; }
function scaleColor(v, vmax){
  // v in [-vmax, vmax] → rgb string on FN..TRUE..FP
  const t = Math.max(-1, Math.min(1, v / vmax));
  let c;
  if (t < 0){ const k = t + 1; c = [lerp(FN[0],TRUE_[0],k), lerp(FN[1],TRUE_[1],k), lerp(FN[2],TRUE_[2],k)]; }
  else      { const k = t;     c = [lerp(TRUE_[0],FP[0],k), lerp(TRUE_[1],FP[1],k), lerp(TRUE_[2],FP[2],k)]; }
  return `rgb(${c.map(x=>Math.round(x)).join(',')})`;
}
const DIVERGING = [
  [0.0, 'rgb(210,75,82)'], [0.25, 'rgb(224,138,143)'],
  [0.5, 'rgb(238,241,244)'],
  [0.75, 'rgb(130,171,216)'], [1.0, 'rgb(63,124,196)']
];
const CLASS_PALETTE = ['#e0a54b','#4ea3a3','#b47cc8','#6f9f5b','#c8795a','#5a7fc8','#c85a8a','#8a8a5a'];
function classColor(code){ return CLASS_PALETTE[code % CLASS_PALETTE.length]; }

/* shared Plotly layout bits (dark instrument theme) */
const AXIS = {
  gridcolor:'#20272f', zerolinecolor:'#33404f', linecolor:'#33404f',
  tickfont:{ family:'Space Mono, monospace', size:10, color:'#63707f' },
  titlefont:{ family:'Archivo, sans-serif', size:11, color:'#9aa7b6' },
};
function baseLayout(extra={}){
  return Object.assign({
    paper_bgcolor:'rgba(0,0,0,0)',
    plot_bgcolor:'rgba(0,0,0,0)',
    font:{ family: "Arial, Helvetica, sans-serif", color:'#e8edf3' },
    margin:{ l:48, r:16, t:34, b:42 },
    hoverlabel:{ bgcolor:'#1b222c', bordercolor:'#33404f',
                 font:{ family: "Arial, Helvetica, sans-serif", size:11, color:'#e8edf3' } },
    showlegend:false,
  }, extra);
}
const PLOT_CONFIG = { displayModeBar:true, displaylogo:false, responsive:true,
  modeBarButtonsToRemove:['autoScale2d','toggleSpikelines'] };

/* INIT */
async function init(){
  // Start with empty state visible, results hidden
  const emptyState = $('emptyState');
  const resultBody = $('resultBody');
  
  // Use removeAttribute to show, setAttribute to hide
  emptyState.removeAttribute('hidden');
  resultBody.setAttribute('hidden', '');
  
  await loadDatasets();
  bindControls();
  bindTabs();
  bindConsole();
  bindVerdictUi();
  bindInfoDelegation();
  injectInfoButtons();
  bindTrustControls();
  await restoreLast();
}

/* ============================================================
   METRIC INFO SYSTEM
   A dictionary of thorough explanations, an info modal, and a helper that turns
   any section title into a clickable info trigger (title + an (i) button).
   ============================================================ */
const METRIC_INFO = {
  neighborf1: { title: 'Neighbourhood-preservation F1',
    body: `<p><strong>What it is.</strong> A per-point score of how well the projection preserved that point's neighbourhood. Each point is treated as a <em>classifier of its own neighbourhood</em>: its <em>k</em> nearest neighbours in the high-dimensional space are the ground-truth set (the neighbours that <em>should</em> be kept), and its <em>k</em> nearest neighbours in the 2-D projection are the prediction (the neighbours the projection actually placed nearby).</p>
    <p><strong>Precision</strong> = |2-D neighbours &cap; high-D neighbours| / |2-D neighbours| &mdash; of the neighbours the projection asserts, how many are real. (1 &minus; precision is the false-positive / intrusion rate.)<br>
    <strong>Recall</strong> = |2-D &cap; high-D| / |high-D neighbours| &mdash; of the real neighbours, how many the projection kept. (1 &minus; recall is the false-negative / extrusion rate.)<br>
    <strong>F1</strong> = 2&middot;precision&middot;recall / (precision + recall), in [0, 1].</p>
    <p><strong>What it's useful for.</strong> It answers directly &ldquo;which points can I not trust?&rdquo; &mdash; a low-F1 point has a wrong neighbourhood in the picture, so any structure you read around it is unreliable. Because it is per-point, you can colour the projections by it and see exactly <em>where</em> the projection fails.</p>
    <p><strong>How it works.</strong> With a symmetric fixed <em>k</em>, both neighbour sets have size <em>k</em>, so precision = recall = F1 for every point; the three diverge only under adaptive per-point <em>k</em> (then a point's high-D and 2-D neighbourhoods can differ in size). This is a set-overlap metric, distinct from the signed intrusion/extrusion error elsewhere in the tool, which is rank-weighted and directional.</p>` },
  trustmetrics: { title: 'The four per-point trust metrics',
    body: `<p>All four are per-point, in [0, 1], higher = more trustworthy. They measure the same idea, did the projection preserve this point's neighbourhood?, in different ways, so a point can score well on one and badly on another.</p>
    <p><strong>Neighbour F1.</strong> Treats each point as a classifier of its neighbourhood: its high-D k nearest neighbours are the truth, its 2-D k nearest neighbours the prediction. F1 = harmonic mean of precision (fraction of 2-D neighbours that are real; 1&minus;precision = intrusion rate) and recall (fraction of real neighbours kept; 1&minus;recall = extrusion rate). This is the standard Jaccard/neighbourhood-hit set-overlap measure. Uses a hard top-k cut-off.</p>
    <p><strong>JS fidelity.</strong> 1 &minus; Jensen-Shannon divergence between soft neighbour distributions (Gaussian over distances, t-SNE style) in high-D vs 2-D. Continuous and distance-weighted, so it is sensitive to whether the <em>relative</em> closeness of neighbours is preserved, not just top-k membership.</p>
    <p><strong>Distance correlation.</strong> The Spearman rank correlation between the point's distances to all other points in high-D vs 2-D, rescaled to [0, 1]. A global-per-point measure: high = the projection preserves this point's whole distance ordering, not just its nearest neighbours.</p>
    <p><strong>MRRE (Mean Relative Rank Error), as a fidelity.</strong> Penalises how far each of the point's high-D neighbours moved in the 2-D rank ordering, weighted so closer neighbours matter more. Reported as 1 &minus; normalised MRRE. A smoother, rank-based cousin of trustworthiness/continuity.</p>
    <p>Because they emphasise different things (set membership vs. relative distance vs. whole ordering), comparing them, as in the F1-vs-JS scatter, reveals points that are ambiguous, i.e. trustworthy by one criterion but not another.</p>` },
  jsdivergence: { title: 'Jensen-Shannon fidelity (alternative to F1)',
    body: `<p><strong>What it is.</strong> An alternative per-point fidelity based on <em>soft</em> neighbourhoods rather than a hard k cut-off. Following the t-SNE idea, each point's neighbours are given a probability from a Gaussian over distances &mdash; in high-D (distribution <em>P</em>) and in 2-D (distribution <em>Q</em>). The <strong>Jensen-Shannon divergence</strong> JS(P&#8214;Q) &isin; [0, 1] (log base 2) measures how differently the projection distributes that point's neighbour-affinity: 0 = identical, 1 = maximally different. We report <strong>JS fidelity = 1 &minus; JS</strong> so that, like F1, higher is better.</p>
    <p><strong>Why a second metric.</strong> F1 uses a binary threshold (in or out of the top <em>k</em>); JS is continuous and weights near neighbours more than far ones, so it is sensitive to whether the projection preserved the <em>relative</em> closeness of neighbours, not just the membership of the top-<em>k</em> set. A point can score high on one and low on the other &mdash; the &ldquo;F1 vs JS&rdquo; chart surfaces exactly those disagreements.</p>
    <p><strong>How it works.</strong> JS = &frac12;KL(P&#8214;M) + &frac12;KL(Q&#8214;M) where M = &frac12;(P+Q); the per-point Gaussian bandwidth is the distance to the <em>k</em>-th neighbour, so softness scales with local density. It is symmetric and bounded, unlike raw KL divergence.</p>` },
  boxplot: { title: 'Per-point score box plots',
    body: `<p><strong>What it is.</strong> Two box plots &mdash; one for PCA, one for MDS &mdash; of the distribution of the per-point score (F1 or JS fidelity) across all points. Each box shows the <strong>median</strong> (centre line), the <strong>interquartile range</strong> (25th&ndash;75th percentile, the box), the <strong>whiskers</strong> (1.5&times;IQR), and <strong>outliers</strong> as individual dots; the <strong>mean</strong> is marked separately. Exact values appear beneath and on hover.</p>
    <p><strong>What it's useful for.</strong> It compares the two projections as <em>distributions</em>, not single numbers: a higher median box means the method preserves more neighbourhoods; a long lower whisker or many low outliers means a method is faithful for most points but badly wrong for a few. This is the aggregated, distribution-level companion to the per-point colouring.</p>
    <p><strong>Dynamic with brushing/filters.</strong> When you select a subset (by brushing the projections, or by class / attribute filters), the boxes recompute over just that subset, so you can see how trustworthiness varies across regions or classes. The &ldquo;selected only / vs everyone&rdquo; switch controls whether the score is recomputed using only the selected points as candidate neighbours, or kept as the original score computed against all points.</p>` },
  metriccompare: { title: 'F1 vs JS fidelity comparison',
    body: `<p><strong>What it is.</strong> A scatter of every point by its neighbour-F1 (x-axis) against its JS fidelity (y-axis), for the chosen method, with the y = x diagonal drawn.</p>
    <p><strong>What it shows.</strong> Points <em>on</em> the diagonal are judged the same by both metrics. Points <em>off</em> the diagonal disagree: <strong>high F1 but low JS</strong> = the projection kept the right set of k neighbours but distorted their relative distances/affinities; <strong>low F1 but high JS</strong> = the exact top-k set changed but the overall soft neighbourhood is preserved. These ambiguous points are where the choice of metric matters, and are worth inspecting individually.</p>` },

  errorspace: { title: 'Error Space',
    body: `<p><strong>What it is.</strong> Three linked scatter plots: the PCA projection and the MDS projection on top, and below them the <em>error space</em>, a scatter of each point's PCA error (x) against its MDS error (y).</p>
    <p><strong>What it's useful for.</strong> It shows, per point, <em>where</em> each projection is wrong, <em>in which direction</em>, and <em>how much</em>, and whether the two methods agree. This is a <em>projection</em> diagnostic: it measures the picture, not the data.</p>
    <p><strong>How it works.</strong> Each point carries two channels: <em>colour</em> is the severity-weighted signed error on a false-negative → true → false-positive scale (red = the projection lost real neighbours, blue = it invented false ones); <em>size</em> is the total distortion |FP| + |FN|. A big pale point is badly distorted in both directions at once (they cancel to ~0 net); a small pale point is genuinely faithful, colour alone can't tell them apart, so always read size too. Position in the error space encodes the sign combination: the four quadrants are both-FP, both-FN, and the two mixed cases where the methods disagree.</p>` },
  fidelity: { title: 'Intrusion / Extrusion Fidelity',
    body: `<p><strong>What it is.</strong> A per-point measure of how faithfully a 2-D projection preserves each point's high-dimensional neighbourhood, on a signed scale in [−1, +1].</p>
    <p><strong>What it's useful for.</strong> It localises and signs projection error. A <em>false positive</em> (intrusion, +) means the projection pulled a stranger close; a <em>false negative</em> (extrusion, −) means it pushed a true neighbour away.</p>
    <p><strong>How it works.</strong> For neighbourhood size k, each point's neighbours are ranked by distance in both the high-D space and the 2-D projection. An intruder is inside the 2-D k-neighbourhood but not the high-D one; an extruded point is the reverse. Each offender is weighted by how many rank positions it is displaced, normalised to [0,1]. The signed error is FP − FN. By default the metric is <em>geometric</em> (label-free); turning on class-aware restricts it to cross-class intrusions and same-class extrusions.</p>` },
  verdict: { title: 'PCA vs MDS Verdict',
    body: `<p><strong>What it is.</strong> A single mechanical judgement of which projection is more faithful on this dataset, and how confidently.</p>
    <p><strong>What it's useful for.</strong> It answers "which projection should I trust here?" with a confidence statement, and declines to decide when the evidence is thin, rather than over-claiming.</p>
    <p><strong>How it works.</strong> It combines four independent criteria, each scored 0–1: <em>fidelity</em> (weight 0.40, the per-point distortion), <em>trustworthiness</em> and <em>continuity</em> (0.20 each, classical rank-preservation), and <em>cluster preservation</em> (0.20, how well each 2-D k-means partition matches the high-D one). The weights are stated, not tuned. Because both methods are scored on the same points, a paired bootstrap 95% confidence interval and Cohen's d are computed on the per-point distortion gap; the winner is "decisive" only if that interval excludes zero and the margin clears a threshold.</p>` },
  confidence: { title: 'Confidence Interval (bootstrap)',
    body: `<p><strong>What it is.</strong> A plot of the mean per-point distortion gap between the two methods (loser − winner) with its 95% bootstrap confidence interval.</p>
    <p><strong>What it's useful for.</strong> It shows whether the verdict's margin is <em>statistically real</em> or could be noise. A margin alone can't tell you that.</p>
    <p><strong>How it works.</strong> The per-point distortion differences are resampled with replacement 2000 times; the 2.5th–97.5th percentiles of the resampled mean form the interval. If the whole interval sits on one side of zero, the win is reliable (green); if it crosses zero, the difference is provisional (amber).</p>` },
  components: { title: 'Verdict Component Scores',
    body: `<p><strong>What it is.</strong> The four criteria behind the verdict, fidelity, trust, continuity, cluster, shown for both PCA and MDS.</p>
    <p><strong>What it's useful for.</strong> It reveals <em>why</em> a method won, and when the criteria disagree (one method wins some, the other wins others), which is exactly when the verdict is correctly not decisive.</p>
    <p><strong>How it works.</strong> Each component is normalised to [0,1] (higher is better). The composite is their weighted sum (0.40 / 0.20 / 0.20 / 0.20). The bar chart plots all four for each method side by side.</p>` },
  histograms: { title: 'Error Distribution (histograms)',
    body: `<p><strong>What it is.</strong> The distribution of signed per-point error for each method, as a histogram tinted by the error scale.</p>
    <p><strong>What it's useful for.</strong> It reveals the <em>shape</em> of a method's errors that the mean hides, a narrow peak at zero (faithful) vs a broad spread (heavily distorted), and whether errors lean red (extrusion) or blue (intrusion).</p>
    <p><strong>How it works.</strong> Errors are binned over a fixed range; a median line and ±2σ lines (over occupied bins) frame how uneven the distribution is. Because it uses the signed error, balanced-severe points sit near zero here too, read it alongside the Error Space size channel.</p>` },
  centroids: { title: 'Centroid Trajectories',
    body: `<p><strong>What it is.</strong> Each class centroid traced across three stages, high-D → PCA → MDS, on a shared normalised frame.</p>
    <p><strong>What it's useful for.</strong> It separates <em>coarse</em> distortion (how the overall class configuration moves) from the <em>local</em> distortion the per-point metric captures. It shows which method pulls classes apart or compresses them.</p>
    <p><strong>How it works.</strong> High-D centroids are projected via their own PCA so all three stages share a plane; each stage is centred and scaled to unit RMS radius, so the trajectory shows shape change, not absolute size. The mean between-centroid distance per stage indicates whether class separation is preserved.</p>` },
  adjdist: { title: 'Adjusted-distance MDS (Locality vs Mahalanobis)',
    body: `<p>Ordinary MDS uses the plain Euclidean distance, so a few dominant or noisy features can drive the whole layout. An <em>adjusted</em> distance reweights the feature space so the projection preserves neighbourhoods better. Both methods here are <strong>label-free</strong> (they use no class labels), and only the MDS panel changes, PCA and the per-point trust scores stay on the true geometry, so they remain an honest reference.</p>
    <p><strong>Locality weighting (default).</strong> Weights each feature by its <em>Laplacian Score</em> (He, Cai &amp; Niyogi 2005): build a k-NN graph, then score a feature by how little it varies between neighbouring points relative to its total variance. Features that respect the local manifold get more weight; features that vary randomly between neighbours (noise) get less. You can then hand-tune each feature's multiplier.</p>
    <p><strong>Mahalanobis.</strong> Whitens the space by the inverse covariance, d = √((a−b)ᵀΣ⁻¹(a−b)). It is the classical adjusted distance, but whitening <em>amplifies</em> low-variance directions, which are frequently noise, so on the datasets here it tends to make neighbourhood-F1 <em>worse</em>. The panel measures this rather than assuming it, which is why locality weighting is the default: it succeeds where Mahalanobis fails.</p>
    <p><strong>How effectiveness is verified.</strong> The panel re-embeds MDS with the adjusted distance and reports the neighbourhood-F1 and the false-positive (intrusion) / false-negative (extrusion) rates of the plain vs adjusted layout, so you can see directly whether the adjusted distance reduces distortion.</p>` },
  classaware: { title: 'Class-aware trust analysis',
    body: `<p>By default the trust metrics are <strong>geometric</strong>: a point's neighbours are defined purely by distance, with no assumption that the class labels are correct. Turning on <strong>class-aware</strong> restricts the error to <em>cross-class</em> intrusions (a neighbour of a different class pulled close) and <em>same-class</em> extrusions (a same-class neighbour pushed away), so it measures whether the projection preserved the labelled <em>class</em> structure rather than raw geometry.</p>
    <p>The two can disagree sharply, a point can be geometrically distorted but class-faithful (its neighbourhood was reshuffled only within its own class), or vice versa.</p>
    <p><strong>Why k-means appears.</strong> Class-aware scores only make sense if the labels actually correspond to the data's clusters. So when class-aware is on, the k-means ground-truth check is shown: it clusters the data with no labels and reports whether those clusters match the classes (ARI, matched accuracy) and whether each projection preserves the clustering. A low agreement means a poor class-aware score reflects a labelling/geometry mismatch, not the projection's fault.</p>` },
  mahalanobis: { title: 'Mahalanobis metric learning',
    body: `<p><strong>What it is.</strong> An MDS layout built from the <em>Mahalanobis</em> distance instead of the plain Euclidean distance. The Mahalanobis distance is d(a,b) = &radic;((a&minus;b)&#7488; &Sigma;&#8315;&sup1; (a&minus;b)), where &Sigma; is the data covariance &mdash; equivalent to whitening the space (rotating and rescaling by the inverse covariance) and then measuring ordinary distance.</p>
    <p><strong>What it's useful for.</strong> Plain Euclidean distance lets high-variance and correlated directions dominate, so the layout can be driven by a few dominant or redundant features. Mahalanobis puts every direction on equal footing, which is the classical &ldquo;adjusted distance&rdquo; of unsupervised metric learning. Crucially it is <strong>label-free</strong> &mdash; it uses no class labels, only the data's own covariance &mdash; so any improvement is a property of the geometry, not of the labels.</p>
    <p><strong>How it works &amp; how we verify it.</strong> Only the MDS panel uses the adjusted distance; PCA and the per-point trust scores stay on the true (unwhitened) Euclidean geometry, so they remain an honest reference. The comparison panel puts the plain and Mahalanobis layouts side by side and &mdash; the important part &mdash; reports the neighbourhood-F1 and the false-positive (intrusion) and false-negative (extrusion) rates of each. That lets you check whether the adjusted distance genuinely reduces distortion. It does not always: whitening amplifies low-variance directions, which on some data are noise, so the tool measures the effect rather than assuming it.</p>` },
  weightedmds: { title: 'Weighted MDS',
    body: `<p><strong>What it is.</strong> An MDS layout in which each feature is scaled by its mutual information with the class label before distances are computed.</p>
    <p><strong>What it's useful for.</strong> Ordinary MDS lets noisy or redundant features dominate the geometry as much as informative ones. Weighting is a controlled experiment: "what would the layout look like if the class-informative measurements counted more?" It is most useful on data with many irrelevant features.</p>
    <p><strong>How it works.</strong> Each feature j is multiplied by √(weight_j), then MDS is built from the resulting weighted-Euclidean distances. Crucially, <em>only the MDS panel changes</em>, PCA and the high-D fidelity reference stay unweighted, so the error colours still measure true-geometry distortion. The comparison shows the plain and weighted layouts side by side with the change in class silhouette.</p>` },
  kmeans: { title: 'k-means Ground-Truth Check',
    body: `<p><strong>What it is.</strong> An unsupervised clustering of the data, compared against the class labels. This is a <em>data</em> diagnostic, not a projection one.</p>
    <p><strong>What it's useful for.</strong> Every label-based score assumes the labels <em>are</em> the cluster structure. This check asks whether that's true, so an apparent "projection error" isn't really a mismatch between the labels and the data's natural groups.</p>
    <p><strong>How it works.</strong> The chosen clusterer (k-means, or GMM / spectral for non-convex data) is fit in the high-D space and both projections. ARI vs labels answers "are the classes clusters?"; a Hungarian-matched accuracy makes that interpretable; ARI 2-D-vs-high-D answers "does each projection preserve the partition?". A low ARI can mean bad labels <em>or</em> (for k-means) non-convex clusters, cross-check with precision.</p>` },
  confusion: { title: 'Cluster vs Class Confusion Heatmap',
    body: `<p><strong>What it is.</strong> A contingency table: each cell counts the points in a high-D cluster (row) that carry a given class label (column).</p>
    <p><strong>What it's useful for.</strong> It shows exactly how clusters and labels line up. A clean result is block-diagonal, each cluster dominated by one class.</p>
    <p><strong>How it works.</strong> <em>Cluster purity</em> = the largest count in a row ÷ that row's total (how single-class each cluster is). <em>Class coverage</em> = the largest count in a column ÷ that column's total (how much of each class landed in one cluster). High purity <em>and</em> coverage means labels and geometry agree.</p>` },
  silhouette: { title: 'Silhouette by Cluster Count',
    body: `<p><strong>What it is.</strong> The silhouette score of the clustering as the number of clusters k is varied.</p>
    <p><strong>What it's useful for.</strong> It reveals how many groups the geometry naturally supports. If that differs from your class count, the labels may split or merge the natural groups.</p>
    <p><strong>How it works.</strong> Silhouette (−1 to +1) compares each point's average distance to its own cluster against the nearest other cluster: near +1 = tight, well-separated clusters; near 0 = overlapping; negative = points likely in the wrong cluster. The peak marks the most natural number of clusters (green); your class count is marked red.</p>` },
  clustermetrics: { title: 'Cluster-Structure Metrics (precision / recall / F1)',
    body: `<p><strong>What it is.</strong> Supervised precision, recall and F1 computed on a k-nearest-neighbour similarity graph over the high-D points. A <em>data</em> diagnostic.</p>
    <p><strong>What it's useful for.</strong> It measures whether the class structure is genuinely present in the data, independent of any projection. A low score is a property of the data, not the picture.</p>
    <p><strong>How it works.</strong> <em>Precision</em> asks whether a point is surrounded by its own class (weighted by edge strength); <em>recall</em> asks whether each class holds together (its strictness set by α); <em>F1</em> is their harmonic mean. Global scores are macro-averaged (per class) with a micro (pooled) figure alongside; the gap between them reveals class imbalance.</p>` },
  radar: { title: 'Per-Class Radar',
    body: `<p><strong>What it is.</strong> One small radar per class, plotting its precision, recall and F1.</p>
    <p><strong>What it's useful for.</strong> It shows each class's strengths and weaknesses individually, which classes are hard, and <em>how</em> they're hard.</p>
    <p><strong>How it works.</strong> A balanced triangle means the class is precise <em>and</em> cohesive. A spike toward one axis shows the weakness: high precision but low recall = the class is pure but fragmented; the reverse = cohesive but impure.</p>` },
  annealing: { title: 'Simulated Annealing (neighbourhood search)',
    body: `<p><strong>What it is.</strong> A search for the neighbourhood size k that maximises the cluster-structure F1, since F1(k) is bumpy and has several local optima.</p>
    <p><strong>What it's useful for.</strong> It makes an otherwise-arbitrary parameter principled and data-driven, cheaply (touching far fewer values than an exhaustive scan).</p>
    <p><strong>How it works.</strong> From the current k it proposes a nearby value and accepts it if better, or, with a probability that falls as the "temperature" cools, even if worse, which lets it escape local optima. The best-F1 line only ever rises and should plateau; proposed-k should bounce early then settle; accepted worsening moves cluster early and thin out. A temperature panel shows the cooling schedule, and a health check flags a search that's stuck (best flatlines), under-cooled (still rising at the end), or too greedy (no worsening moves).</p>` },
  untrustworthy: { title: 'Untrustworthy Points',
    body: `<p><strong>What it is.</strong> The points a projection asserts a <em>false closeness</em> for, intrusions where the 2-D layout draws together points that are actually far apart in high-D.</p>
    <p><strong>What it's useful for.</strong> These are the genuinely misleading points: a cluster you see among them may be an artifact. (Lost neighbours, extrusions, are a loss of information but not a false claim, and are listed separately.)</p>
    <p><strong>How it works.</strong> A point is flagged when its error exceeds +threshold (an intrusion) in either projection. Each row decomposes into FP (intrusion) and FN (extrusion) for both methods, so you can see which method distorts it and how.</p>` },
  adaptivek: { title: 'Adaptive k',
    body: `<p><strong>What it is.</strong> A per-point neighbourhood size driven by local density, instead of one global k.</p>
    <p><strong>What it's useful for.</strong> A single k is too coarse in dense regions and too noisy in sparse ones. Adaptive k gives each point a neighbourhood sized to its surroundings, so the fidelity metric is fair across regions of differing density.</p>
    <p><strong>How it works.</strong> Each point's local radius (its distance to its nearest neighbours) is mapped through the population median so the median-density point keeps the slider's k; dense points get a smaller k, sparse points a larger one, clamped to a sensible range. The indicator by the slider shows the resulting min / median / max k.</p>` },
};

function openInfoModal(key){
  const info = METRIC_INFO[key]; if(!info) return;
  $('infoTitle').textContent = info.title;
  $('infoBody').innerHTML = info.body;
  $('infoModal').hidden = false;
}
function closeInfoModal(){ $('infoModal').hidden = true; }

/** Build a section header (h4) whose title and (i) button open the info modal. */
function infoHeader(key, labelHtml){
  const info = METRIC_INFO[key];
  const label = labelHtml || (info ? info.title : key);
  return `<h4 class="info-h"><button class="info-title" data-info="${key}">${label}</button>`+
    `<button class="info-btn" data-info="${key}" aria-label="What is this?" title="What is this?">i</button></h4>`;
}
/** Delegated click handler: any [data-info] opens the modal. Bound once. */
function bindInfoDelegation(){
  document.addEventListener('click', (e)=>{
    const t = e.target.closest('[data-info]');
    if(t){ openInfoModal(t.dataset.info); }
  });
  const c = $('infoClose'), d = $('infoDone'), ov = $('infoModal');
  if(c) c.addEventListener('click', closeInfoModal);
  if(d) d.addEventListener('click', closeInfoModal);
  if(ov) ov.addEventListener('click', (e)=>{ if(e.target===ov) closeInfoModal(); });
  document.addEventListener('keydown', (e)=>{ if(e.key==='Escape') closeInfoModal(); });
}

/** Turn every element tagged data-info-title into a clickable info trigger and
 *  append a small (i) button after it. Runs once at startup over static markup. */
function injectInfoButtons(){
  document.querySelectorAll('[data-info-title]').forEach(el=>{
    const key = el.getAttribute('data-info-title');
    if(!METRIC_INFO[key] || el.dataset.infoWired) return;
    el.dataset.infoWired = '1';
    el.classList.add('has-info');
    el.setAttribute('data-info', key);           // make the title itself open the modal
    // append an (i) button right after the title element
    const btn = document.createElement('button');
    btn.className = 'info-btn';
    btn.setAttribute('data-info', key);
    btn.setAttribute('aria-label', 'What is this?');
    btn.setAttribute('title', 'What is this?');
    btn.textContent = 'i';
    el.insertAdjacentElement('afterend', btn);
  });
}

/** In the trust panel: show the geometric-vs-class-aware divergence, and when
 *  class-aware is active, surface the k-means ground-truth results. */
function renderClassAwareDelta(d){
  // sync the checkbox to whatever produced this run
  const cb = $('classAware');
  const isClassAware = !!(d.params && d.params.class_aware);
  if(cb && d.params && typeof d.params.class_aware === 'boolean') cb.checked = isClassAware;

  const info = $('classAwareInfo');
  const cad = d.class_aware_delta;
  if(info){
    if(cad){
      info.hidden = false;
      info.innerHTML =
        `<strong>${isClassAware ? 'Class-aware' : 'Geometric'}</strong> fidelity is active. `+
        `The two modes differ in sign on <strong>${cad.sign_flip_pct}%</strong> of points `+
        `(correlation ${cad.correlation}), class-aware counts only cross-class intrusions and same-class extrusions, so it measures whether the labelled <em>class</em> structure survived, not raw geometry.`;
    } else { info.hidden = true; info.innerHTML=''; }
  }
  // legacy element (may be absent now)
  const el = $('classAwareDelta');
  if(el && cad) el.innerHTML = `differ on <strong>${cad.sign_flip_pct}%</strong> of points`;

  // k-means results, shown only when class-aware is on
  renderTrustKmeans(d, isClassAware);
}

/** Compact k-means ground-truth panel for the trust analysis (class-aware on). */
function renderTrustKmeans(d, show){
  const box = $('trustKmeans');
  if(!box) return;
  const km = d.kmeans;
  if(!show || !km){ box.hidden = true; box.innerHTML=''; return; }
  box.hidden = false;
  const hi = km.high_d || {};
  const ariL = hi.ari_vs_labels!=null ? hi.ari_vs_labels.toFixed(3) : ',';
  const acc  = hi.cluster_accuracy!=null ? (hi.cluster_accuracy*100).toFixed(0)+'%' : ',';
  const are  = km.classes_are_clusters;
  const nCls = km.n_classes, silk = km.best_k_by_silhouette;
  // per-method ARI (does each projection preserve the high-D clustering?)
  let methodRows = '';
  if(km.methods){
    methodRows = Object.keys(km.methods).map(m=>{
      const a=km.methods[m].ari_vs_high;
      return `<span class="kmm">${m}: ARI 2-D vs high-D <strong>${a!=null?a.toFixed(3):','}</strong></span>`;
    }).join('');
  }
  const verdict = are
    ? `<span class="km-good">✓ the labels align with the data's natural clusters</span>`
    : `<span class="km-warn">✗ the labels do <strong>not</strong> match the natural clusters, class-aware scores here reflect a labelling/geometry mismatch, not the projection</span>`;
  box.innerHTML =
    `<div class="km-head">k-means ground-truth check <span class="an-subtle">(runs with class-aware)</span></div>`+
    `<div class="km-stats">`+
      `<span class="kmm">ARI vs labels <strong>${ariL}</strong></span>`+
      `<span class="kmm">matched accuracy <strong>${acc}</strong></span>`+
      `<span class="kmm">natural clusters (silhouette) <strong>${silk}</strong> vs <strong>${nCls}</strong> classes</span>`+
      methodRows+
    `</div>`+
    `<div class="km-verdict">${verdict}</div>`;
}

/* verdict banner + modal button wiring */
function bindVerdictUi(){
  const openBtn = $('vbExamine');
  if(openBtn) openBtn.addEventListener('click', openVerdictModal);
  const close = $('vmClose'), done = $('vmDone'), overlay = $('verdictModal'),
        goto = $('vmGotoTab');
  if(close) close.addEventListener('click', closeVerdictModal);
  if(done) done.addEventListener('click', closeVerdictModal);
  if(goto) goto.addEventListener('click', ()=>{ closeVerdictModal(); activateTab('projquality'); });
  if(overlay) overlay.addEventListener('click', (e)=>{ if(e.target===overlay) closeVerdictModal(); });
  document.addEventListener('keydown', (e)=>{ if(e.key==='Escape') closeVerdictModal(); });
  const lbExamine = $('lbExamine');
  if(lbExamine) lbExamine.addEventListener('click', ()=>{ activateTab('dataquality'); const c=$('anKmeansCard'); if(c) c.scrollIntoView({behavior:'smooth',block:'center'}); });
}

async function loadDatasets(){
  try{
    const r = await fetch('/api/datasets'); const d = await r.json();
    const sel = $('dataset');
    d.datasets.forEach(name=>{
      const o = document.createElement('option');
      o.value = name; o.textContent = name;
      sel.appendChild(o);
    });
  }catch(e){ toast('Could not load dataset list', true); }
}

/* control binding */
function bindControls(){
  // live echoes
  const echo = (id, echoId, fmt=(v)=>v) => {
    const el = $(id), out = $(echoId);
    const upd = ()=> out.textContent = fmt(el.value);
    el.addEventListener('input', upd); upd();
  };
  echo('neighbors','kEcho');
  echo('threshold','threshEcho',(v)=>parseFloat(v).toFixed(2));
  if($('recallAlpha')) echo('recallAlpha','recallAlphaEcho',(v)=>parseFloat(v).toFixed(2));

  // CSV upload
  /*
  $('csvFile').addEventListener('change', handleCsvUpload);
  const csvLabel = $('csvLabel');
  ['dragover','dragenter'].forEach(ev=>csvLabel.addEventListener(ev,(e)=>{e.preventDefault(); csvLabel.classList.add('loaded');}));
  ['dragleave','drop'].forEach(ev=>csvLabel.addEventListener(ev,(e)=>{e.preventDefault();}));
  csvLabel.addEventListener('drop',(e)=>{
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if(f){ $('csvFile').files = e.dataTransfer.files; handleCsvUpload(); }
  });
  */

  // run
  $('runBtn').addEventListener('click', runAnalysis);
  // moved in-panel controls re-run automatically so their effect is immediate
  ['classAware','weightedMds','clusterMethod'].forEach(id=>{
    const el = $(id);
    if(el) el.addEventListener('change', ()=>{ if(state.result) runAnalysis(); });
  });
  // adjusted-distance method selector (Locality / Mahalanobis) -> re-run
  const ams=$('adjMethodSeg');
  if(ams) ams.addEventListener('click',(e)=>{
    const b=e.target.closest('.seg-btn'); if(!b) return;
    ams.querySelectorAll('.seg-btn').forEach(x=>x.classList.toggle('active',x===b));
    state.adjMethod=b.dataset.am;
    state.featWeightOverride={};   // reset overrides when method changes
    if(state.result) runAnalysis();
  });

  // clear
  $('clearBtn').addEventListener('click', clearResults);
  // ⏎ to run (when not typing in a text field)
  document.addEventListener('keydown',(e)=>{
    if(e.key==='Enter' && !['INPUT','SELECT','TEXTAREA'].includes(document.activeElement.tagName)){
      runAnalysis();
    }
  });
}

/*
let csvState = { path:null, labelCol:null };
async function handleCsvUpload(){
  const f = $('csvFile').files[0]; if(!f) return;
  const fd = new FormData(); fd.append('file', f);
  $('csvText').textContent = 'uploading…';
  try{
    const r = await fetch('/api/upload',{ method:'POST', body:fd });
    const d = await r.json();
    if(!d.ok){ toast(d.error||'upload failed', true); $('csvText').textContent='or drop a CSV…'; return; }
    csvState.path = d.path;
    $('csvLabel').classList.add('loaded');
    $('csvText').textContent = f.name;
    // populate label column picker
    const lc = $('labelCol'); lc.innerHTML='';
    d.columns.forEach(c=>{ const o=document.createElement('option'); o.value=c; o.textContent=c; lc.appendChild(o); });
    $('labelColField').hidden = false;
    toast('CSV loaded, pick a label column');
  }catch(e){ toast('upload error', true); }
}
*/

/* collect params from the form */
function collectParams(){
  const p = {
    dataset: $('dataset').value,
    metric: state.metric,
    neighbors: parseInt($('neighbors').value),
    class_aware: $('classAware') ? $('classAware').checked : false,
    weight: 'rank',
    adaptive: $('adaptive') ? $('adaptive').checked : true,
    normalize: 'none',
    no_standardize: $('noStandardize').checked,
    threshold: parseFloat($('threshold').value),
    weighted_mds: $('weightedMds') ? $('weightedMds').checked : false,
    weight_method: 'mahalanobis',
    adj_method: state.adjMethod,
    feat_weight_override: state.featWeightOverride,
    // analysis layer toggles
    kmeans: true,
    rel_metrics: true,
    anneal: $('annealToggle') ? $('annealToggle').checked : true,
    rel_k: parseInt($('neighbors').value),
    rel_alpha: $('recallAlpha') ? parseFloat($('recallAlpha').value) : 0.5,
    rel_beta: 1.0,
    cluster_method: $('clusterMethod') ? $('clusterMethod').value : 'kmeans',
  };
  const ms = $('maxSamples').value;
  if(ms) p.max_samples = parseInt(ms);
  // if(csvState.path){ p.csv_path = csvState.path; p.label_col = $('labelCol').value; }
  return p;
}

/* RUN */
async function runAnalysis(){
  const btn = $('runBtn');
  if(btn.disabled) return;
  btn.disabled = true; btn.classList.add('running');
  btn.querySelector('.run-btn-label').textContent = 'Running';
  setConsoleDot('active');

  try{
    const params = collectParams();
    const r = await fetch('/api/run',{
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(params)
    });
    const d = await r.json();
    setConsole(d.console || '(no output)');
    if(!d.ok){
      setConsoleDot('error');
      toast(d.error || 'Analysis failed: see console', true);
      openConsole();
      return;
    }
    state.result = d;
    renderAll(d);
    setConsoleDot('active');
    toast(`Analysis complete, ${d.info.name}, ${d.info.elapsed}s`);
  }catch(e){
    setConsoleDot('error');
    toast('Request failed: '+e.message, true);
  }finally{
    btn.disabled = false; btn.classList.remove('running');
    btn.querySelector('.run-btn-label').textContent = 'Run analysis';
  }
}

function clearResults() {
  // Clear the state
  state.result = null;
  state.esSel = new Set();
  state.esActiveQuad = null;
  
  // Show empty state, hide results
  const emptyState = $('emptyState');
  const resultBody = $('resultBody');
  emptyState.removeAttribute('hidden');
  resultBody.setAttribute('hidden', '');
  
  // Clear console
  setConsole('');
  setConsoleDot('');
  
  // Clear plots
  ['centPlot', 'histDistPca', 'histDistMds', 'histBalPca', 'histBalMds',
   'esPca', 'esMds', 'esErr'].forEach(id => {
    const el = $(id);
    if (el) {
      try {
        Plotly.purge(el);
        el.innerHTML = '';
      } catch(e) { /* ignore */ }
    }
  });
  
  // Clear statbar
  $('statbar').innerHTML = '';
  
  // Clear table
  const thead = $('dtable').querySelector('thead');
  const tbody = $('dtable').querySelector('tbody');
  if (thead) thead.innerHTML = '';
  if (tbody) tbody.innerHTML = '';
  
  // Clear Error Space table, counts, and source tag
  const esThead = $('esTable').querySelector('thead');
  const esTbody = $('esTable').querySelector('tbody');
  if (esThead) esThead.innerHTML = '';
  if (esTbody) esTbody.innerHTML = '';
  document.querySelectorAll('#esQuadrants .es-quad-count').forEach(el=>el.textContent='0');
  document.querySelectorAll('#esQuadrants .es-quad').forEach(b=>b.classList.remove('active'));
  $('esCount').textContent = 'No points selected: drag on any plot or pick a quadrant';
  $('esSource').textContent = ''; $('esSource').hidden = true;
  
  // Reset signal
  $('signalFn').style.width = '0%';
  $('signalFp').style.width = '0%';
  $('signalLabel').textContent = 'SIGNED-ERROR BALANCE';
  $('signalLabel').style.color = '#63707f';
  $('signalLabel').className = 'signal-label';
  $('signalReadout').textContent = 'awaiting run';

  toast('Results cleared');
}

/*
async function clearRuns(){
  await fetch('/api/clear',{method:'POST'});
  toast('Stored runs cleared');
}
*/

async function restoreLast(){
  try{
    const r = await fetch('/api/last'); 
    const d = await r.json();
    if(d && d.ok){
      state.result = d;
      applyParamsToForm(d.params);
      renderAll(d);
      setConsole(d.console||'');
      setConsoleDot('active');
      toast('Restored your last run');
    } else {
      // No results found - show empty state
      const emptyState = $('emptyState');
      const resultBody = $('resultBody');
      emptyState.removeAttribute('hidden');
      resultBody.setAttribute('hidden', '');
    }
  }catch(e){ 
    // Error or first visit - show empty state
    console.log('No stored results found');
    const emptyState = $('emptyState');
    const resultBody = $('resultBody');
    emptyState.removeAttribute('hidden');
    resultBody.setAttribute('hidden', '');
  }
}

function applyParamsToForm(p){
  if(!p) return;
  $('dataset').value = p.dataset || 'wine';
  state.metric = 'intrusion';
  state.weight = 'rank';
  const setV=(id,v)=>{ if(v!==undefined&&v!==null && $(id)) $(id).value=v; };
  setV('neighbors',p.neighbors); $('kEcho').textContent=p.neighbors??10;
  setV('threshold',p.threshold); $('threshEcho').textContent=(p.threshold??0.5).toFixed?.(2)??p.threshold;
  if($('classAware')) $('classAware').checked = !!p.class_aware;
  $('adaptive').checked = p.adaptive!==false;
  $('noStandardize').checked = !!p.no_standardize;
  if(p.max_samples) $('maxSamples').value = p.max_samples;
  if($('weightedMds')) $('weightedMds').checked = !!p.weighted_mds;
  if($('clusterMethod') && p.cluster_method) $('clusterMethod').value = p.cluster_method;
  if($('recallAlpha') && p.rel_alpha!=null){ $('recallAlpha').value = p.rel_alpha; $('recallAlphaEcho').textContent = (+p.rel_alpha).toFixed(2); }
}

/* RENDER */
function renderAll(d){
  const emptyState = $('emptyState');
  const resultBody = $('resultBody');
  
  if (!d) {
    emptyState.removeAttribute('hidden');
    resultBody.setAttribute('hidden', '');
    return;
  }
  
  // Hide empty state, show results
  emptyState.setAttribute('hidden', '');
  resultBody.removeAttribute('hidden');
  
  // non-plot panels first so they always render even if Plotly is unavailable
  safeRender('statbar', ()=>renderStatbar(d));
  safeRender('classawaredelta', ()=>renderClassAwareDelta(d));
  safeRender('adaptivek', ()=>renderAdaptiveK(d));
  safeRender('table', ()=>renderTable(d));
  safeRender('verdict-banner', ()=>renderVerdictBanner(d));
  safeRender('analysis', ()=>renderAnalysisPanel(d));
  safeRender('trust', ()=>renderTrust(d));
  if(typeof Plotly === 'undefined'){
    plotlyUnavailable();
    return;
  }
}

/** The intrusion metric computes genuine, independent FP and FN components.
 *  The classic metric only splits one signed score into halves, so FP and FN
 *  are never both non-zero and a FP-vs-FN scatter would be a degenerate "L". */
function safeRender(label, fn){
  try{ fn(); }
  catch(e){ console.error(`render[${label}] failed:`, e); }
}
function plotlyUnavailable(){
  const msg = 'Plotting library did not load: check your network connection and reload. Tables and console output are available.';
  ['centPlot','histDistPca','histDistMds','histBalPca','histBalMds',
   'esPca','esMds','esErr'].forEach(id=>{
    const el=$(id); if(el) el.innerHTML=`<div class="plot-fallback">${msg}</div>`;
  });
  toast('Plotting library unavailable: see tables instead', true);
}

function renderStatbar(d){
  const s = d.summary, info = d.info;
  const pill = (label, value, cls='', sub='') =>
    `<div class="stat"><div class="stat-label">${label}</div>`+
    `<div class="stat-value ${cls}">${value}</div>`+
    (sub?`<div class="stat-sub">${sub}</div>`:'')+`</div>`;
  const signCls = (v)=> v<0?'neg':(v>0?'pos':'');
  $('statbar').innerHTML =
    pill('Dataset', info.name, '', `${info.standardized?'standardized':'raw'} | ${d.metric}`) +
    pill('Samples', info.n_samples, '', `${info.n_features} features | ${info.n_classes} classes`) +
    (d.timings ? pill('PCA time', (d.timings.pca_ms).toFixed(0)+' ms', '', 'compute') : '') +
    (d.timings ? pill('MDS time', (d.timings.mds_ms).toFixed(0)+' ms', signCls(d.timings.pca_ms - d.timings.mds_ms), 'compute') : '') +
    pill('Untrustworthy', s.untrustworthy_count, '', d.untrust_desc || 'low F1') +
    pill('Runtime', info.elapsed+'s', '', 'pipeline');
  renderWarnings(d);
}

/* parameter-degeneracy warnings (e.g. k too large relative to n) */
function renderWarnings(d){
  let host = $('runWarnings');
  if(!host){
    host = document.createElement('div');
    host.id = 'runWarnings';
    const sb = $('statbar');
    sb.parentNode.insertBefore(host, sb.nextSibling);
  }
  const ws = (d.info && d.info.warnings) || [];
  if(!ws.length){ host.innerHTML=''; host.hidden = true; return; }
  host.hidden = false;
  host.innerHTML = ws.map(w=>
    `<div class="run-warn run-warn-${w.level}"><span class="run-warn-icon">\u26A0</span>`+
    `<span>${w.message}</span></div>`).join('');
}

function renderSignal(d){
  // signature: header signal shows overall FN vs FP mass balance
  const all = d.table.pca.err.concat(d.table.mds.err);
  let fnMass=0, fpMass=0;
  all.forEach(v=>{ if(v<0) fnMass+=-v; else fpMass+=v; });
  const total = fnMass+fpMass || 1;
  
  // Calculate percentages of the FULL bar (not half)
  const fnPct = (fnMass/total) * 100;
  const fpPct = (fpMass/total) * 100;
  
  // Set widths - these now add up to 100% of the bar
  $('signalFn').style.width = fnPct + '%';
  $('signalFp').style.width = fpPct + '%';
  
  const label = $('signalLabel');
  const readout = $('signalReadout');
  const fnPercent = fnPct.toFixed(0);
  const fpPercent = fpPct.toFixed(0);
  
  // Color the label based on dominance
  if(fnMass > fpMass * 1.1){ // FN dominates
    label.style.color = '#d24b52';
    label.className = 'signal-label fn-dominant';
    readout.textContent = `${fnPercent}% FN, ${fpPercent}% FP, FN-dominated`;
  } else if(fpMass > fnMass * 1.1){ // FP dominates
    label.style.color = '#3f7cc4';
    label.className = 'signal-label fp-dominant';
    readout.textContent = `${fnPercent}% FN, ${fpPercent}% FP, FP-dominated`;
  } else { // balanced
    label.style.color = '#63707f';
    label.className = 'signal-label balanced';
    readout.textContent = `${fnPercent}% FN, ${fpPercent}% FP, Balanced`;
  }
}

/* ============================================================
   DOUBLE ENCODING
   Colour alone is ambiguous: err = FP - FN, so a point with
   FP = 0.6 and FN = 0.6 reads err = 0 -> white, exactly like a
   point with FP = FN = 0 that is genuinely well embedded. The two
   are opposites (severely distorted both ways vs faithful), so we
   add a SECOND channel: marker size encodes total distortion
   |FP| + |FN|. White + large = badly distorted but balanced;
   white + small = truly faithful.
   ============================================================ */
const ES_SIZE_MIN = 5;    // px diameter for zero total distortion
const ES_SIZE_MAX = 20;   // px diameter for the run's worst total distortion

/** total distortion |FP| + |FN| per point (the magnitude colour cannot show) */
function totalDistortion(t){
  return t.fp.map((v,i)=> Math.abs(v) + Math.abs(t.fn[i]));
}

/* ============================================================
   SEVERITY-WEIGHTED SIGNED ERROR  (colour metric)
   The plain signed error err = FP - FN answers "which way?" but
   not "how much is behind it": FP=.5,FN=.4 and FP=.1,FN=0 both
   give err = +0.1, yet the first point is five times more
   distorted. We therefore weight the error by the magnitude that
   produced it, using the signed geometric mean

       S = sign(err) * sqrt( |err| * (FP + FN) )
         = sign(FP-FN) * sqrt( |FP^2 - FN^2| )

   Properties (all verified):
     - sign(S) == sign(err): the FN / True / FP semantics survive.
     - S == 0 exactly when err == 0: balance still reads neutral,
       which is why SIZE remains the second channel for those points.
     - If one side is zero, S == err: it degrades to the familiar scale.
     - |S| grows with both the imbalance AND the mass behind it.
   The geometric mean (rather than the raw product err*(FP+FN)) is
   essential: the raw product crushes ~92% of real points into the
   near-white band, destroying contrast, because it is quadratic in
   quantities that are already << 1. The square root restores a
   perceptually usable spread.                                     */
function severityError(t){
  return t.fp.map((fp,i)=>{
    const fn = t.fn[i], e = t.err[i], dist = Math.abs(fp) + Math.abs(fn);
    return Math.sign(e) * Math.sqrt(Math.abs(e) * dist);
  });
}

/** Symmetric colour limit for the severity scale, rounded up for clean ticks. */
function severityVmax(d){
  let peak = 0;
  ['pca','mds'].forEach(m=>{
    severityError(d.table[m]).forEach(v=>{ peak = Math.max(peak, Math.abs(v)); });
  });
  return peak > 0 ? Math.ceil(peak*20)/20 : 0.1;
}
/** Colour channel used by the Error Space plots: severity-weighted signed error. */
function colourChannel(t, d){
  return { vals: severityError(t), vmax: severityVmax(d),
           label: 'severity-weighted error' };
}
/** Per-point class colours for the "see classes" view. */
function classColorArray(d){
  return d.table.labels.map(l=> classColor(l));
}
/** Peak |FP|+|FN| across BOTH methods, so PCA and MDS share one size scale. */
function distortionMax(d){
  const all = totalDistortion(d.table.pca).concat(totalDistortion(d.table.mds));
  const m = Math.max(...all, 0);
  return m > 0 ? m : 1;   // guard: all-zero run -> every point stays at min size
}
/** Map total distortion -> marker diameter. sqrt so AREA grows ~linearly with
 *  distortion (perceptually fairer than scaling the diameter directly). */
function distortionSizes(tot, dmax){
  return tot.map(v=> ES_SIZE_MIN +
    (ES_SIZE_MAX - ES_SIZE_MIN) * Math.sqrt(Math.min(v, dmax) / dmax));
}

/* comparison scatter */
/* histograms
   Bars are centred on their bin's midpoint (so a bar sits symmetrically over
   the error value it represents) and hovering one reports the exact half-open
   interval it covers. Horizontal reference lines summarise the bin COUNTS:
   a green median line plus the two dashed +/-2 sigma lines bounding a shaded
   band, so unusually tall bars are immediately obvious.               */
function renderHistograms(d){
  if(typeof Plotly==='undefined') return;
  const median = (arr)=>{ const s=[...arr].sort((a,b)=>a-b), n=s.length;
    return n? (n%2? s[(n-1)/2] : (s[n/2-1]+s[n/2])/2) : 0; };

  // per-point total distortion (FP + FN) and balance (|FP-FN|/(FP+FN)) per method
  function components(t){
    const dist=[], bal=[];
    for(let i=0;i<t.fp.length;i++){
      const fp=Math.abs(t.fp[i]), fn=Math.abs(t.fn[i]), tot=fp+fn;
      dist.push(tot);
      if(tot > 1e-9) bal.push(Math.abs(fp-fn)/tot);   // undefined for faithful points
    }
    return {dist, bal};
  }
  const cP = components(d.table.pca), cM = components(d.table.mds);

  // shared x-max for distortion so PCA/MDS are comparable
  const distMax = Math.max(0.05, ...cP.dist, ...cM.dist);

  function histData(vals, lo, hi, bins){
    const edges=[]; for(let i=0;i<=bins;i++) edges.push(lo + (hi-lo)*i/bins);
    const counts=new Array(bins).fill(0);
    vals.forEach(v=>{ let b=Math.floor((v-lo)/(hi-lo)*bins); if(b>=bins)b=bins-1; if(b<0)b=0; counts[b]++; });
    const centres=edges.slice(0,bins).map((e,i)=>(e+edges[i+1])/2);
    return {edges, centres, counts};
  }

  // colour ramps: distortion cool→warm (more distortion = warmer/redder);
  // balance cool (balanced, the dangerous "cancels to zero" case) → neutral
  const distColor = (c,mx)=>{ const t=Math.min(c/mx,1);
    return `hsl(${Math.round(210-150*t)}, 55%, ${Math.round(58-6*t)}%)`; };

  function buildDist(divId, vals, label){
    const H = histData(vals, 0, distMax, 30);
    const med = median(vals), mean = vals.reduce((a,b)=>a+b,0)/(vals.length||1);
    const custom = H.counts.map((c,i)=>[H.edges[i].toFixed(3), H.edges[i+1].toFixed(3), (100*c/(vals.length||1)).toFixed(1)]);
    const bar={ type:'bar', x:H.centres, y:H.counts,
      marker:{color:H.centres.map(c=>distColor(c,distMax)), line:{color:'rgba(0,0,0,0.4)',width:0.4}},
      width:(distMax/30)*0.92, customdata:custom,
      hovertemplate:'distortion [%{customdata[0]}, %{customdata[1]})<br><b>%{y}</b> points, %{customdata[2]}%<extra></extra>',
      showlegend:false };
    const hmax=Math.max(...H.counts,1)*1.15;
    const medLine={ type:'scatter',mode:'lines',x:[med,med],y:[0,hmax],
      line:{color:'#4ec97a',width:1.4,dash:'dot'},
      name:`median ${med.toFixed(3)}`, showlegend:true,
      hovertemplate:`median distortion ${med.toFixed(3)}<extra></extra>` };
    Plotly.react(divId,[bar,medLine], baseLayout({
      title:{text:`${label}, n=${vals.length}, mean ${mean.toFixed(3)}`, font:{size:11,color:'#9aa7b6'},x:0.02,xanchor:'left'},
      bargap:0.04, showlegend:true, legend:{font:{size:9,color:'#9aa7b6'},x:0.99,xanchor:'right',y:0.99,bgcolor:'rgba(21,26,33,0.78)'},
      xaxis:Object.assign({title:'total distortion  FP + FN',range:[0,distMax]},AXIS),
      yaxis:Object.assign({title:'count',rangemode:'tozero',range:[0,hmax]},AXIS),
    }),PLOT_CONFIG);
  }

  function buildBal(divId, vals, label){
    const H = histData(vals, 0, 1, 20);
    const med = median(vals), balancedPct = 100*vals.filter(v=>v<0.34).length/(vals.length||1);
    const custom = H.counts.map((c,i)=>[H.edges[i].toFixed(2), H.edges[i+1].toFixed(2), (100*c/(vals.length||1)).toFixed(1)]);
    // colour: near 0 (balanced, the misleading case) = amber; near 1 (one-sided) = blue
    const balColor = (c)=> `hsl(${Math.round(40+174*c)}, 50%, 55%)`;
    const bar={ type:'bar', x:H.centres, y:H.counts,
      marker:{color:H.centres.map(balColor), line:{color:'rgba(0,0,0,0.4)',width:0.4}},
      width:(1/20)*0.92, customdata:custom,
      hovertemplate:'balance [%{customdata[0]}, %{customdata[1]})<br><b>%{y}</b> points, %{customdata[2]}%<extra></extra>',
      showlegend:false };
    const hmax=Math.max(...H.counts,1)*1.15;
    const medLine={ type:'scatter',mode:'lines',x:[med,med],y:[0,hmax],
      line:{color:'#4ec97a',width:1.4,dash:'dot'},
      name:`median ${med.toFixed(2)}`, showlegend:true,
      hovertemplate:`median balance ${med.toFixed(2)}<extra></extra>` };
    Plotly.react(divId,[bar,medLine], baseLayout({
      title:{text:`${label}, n=${vals.length} distorted, ${balancedPct.toFixed(0)}% near-balanced (≈0)`, font:{size:11,color:'#9aa7b6'},x:0.02,xanchor:'left'},
      bargap:0.04, showlegend:true, legend:{font:{size:9,color:'#9aa7b6'},x:0.99,xanchor:'right',y:0.99,bgcolor:'rgba(21,26,33,0.78)'},
      xaxis:Object.assign({title:'balance   0 = both-ways, 1 = one-sided',range:[0,1]},AXIS),
      yaxis:Object.assign({title:'count',rangemode:'tozero',range:[0,hmax]},AXIS),
    }),PLOT_CONFIG);
  }

  buildDist('histDistPca', cP.dist, 'PCA');
  buildDist('histDistMds', cM.dist, 'MDS');
  buildBal('histBalPca', cP.bal, 'PCA');
  buildBal('histBalMds', cM.bal, 'MDS');
}

/* FP vs FN scatter (intrusion/extrusion balance), one panel per method */
/* class-centroid trajectory: high-D -> PCA -> MDS */
function renderCentroids(d){
  const c=d.centroids;
  if(!c){ $('centPlot').innerHTML='<div class="plot-fallback">No centroid data in this run.</div>'; return; }
  const stages=c.stage_names;                 // ['High-D','PCA','MDS']
  const xPos={}; stages.forEach((s,i)=>xPos[s]=i);
  const traces=[];

  // one polyline per class tracing its centroid across the stages
  c.classes.forEach((clsName, ci)=>{
    const code=c.class_codes[ci];
    const col=classColor(code);
    const pts=stages.map(s=>c.coords[s][ci]);      // [ [x,y], [x,y], [x,y] ]
    // trajectory line: x = stage index, y offset encoded via marker text; we plot
    // the actual 2-D drift as a small connected path within each stage column,
    // using stage index as x and the centroid's normalised 1st-axis as y, with the
    // 2nd axis shown on hover. This keeps stages readable left-to-right.
    traces.push({
      type:'scatter', mode:'lines+markers+text',
      x: stages.map(s=>xPos[s]),
      y: pts.map(p=>p[0]),
      text: stages.map((s,si)=> si===0? clsName : ''),
      textposition:'top center',
      textfont:{family:'Space Mono',size:10,color:col},
      name: clsName,
      line:{color:col,width:2},
      marker:{size:11,color:col,line:{color:'#0e1116',width:1.5},symbol:'circle'},
      customdata: pts.map(p=>[p[0],p[1]]),
      hovertemplate:`${clsName}<br>%{x}<br>axis1 %{customdata[0]:.3f}, axis2 %{customdata[1]:.3f}<extra></extra>`,
    });
  });

  // vertical guides per stage
  const guides=stages.map((s)=>({ type:'scatter',mode:'lines',
    x:[xPos[s],xPos[s]], y:[-3,3],
    line:{color:'#20272f',width:1}, hoverinfo:'skip', showlegend:false }));

  Plotly.react('centPlot',[...guides,...traces], baseLayout({
    showlegend:true,
    legend:{ font:{size:10,color:'#9aa7b6'}, bgcolor:'rgba(21,26,33,0.7)',
             bordercolor:'#262f3b',borderwidth:1, orientation:'h', x:0.5, xanchor:'center', y:-0.14 },
    xaxis:Object.assign({title:'',showgrid:false,tickvals:stages.map(s=>xPos[s]),
      ticktext:stages,range:[-0.4,stages.length-0.6]},AXIS),
    yaxis:Object.assign({title:'centroid position (normalised axis 1)',zeroline:true},AXIS),
  }),PLOT_CONFIG);

  // between-class separation readout per stage
  const mb=c.mean_between;
  $('centMeta').innerHTML = stages.map(s=>
    `<span class="cent-stat"><span class="cent-stat-k">${s}</span>`+
    `<span class="cent-stat-v">${mb[s].toFixed(3)}</span></span>`
  ).join('') + `<span class="cent-note">mean between-centroid distance (normalised), similar values across stages ⇒ class separation preserved</span>`;
}

/* data table (sortable) */
const TABLE_COLS = [
  { key:'index', label:'index' },
  { key:'true_class', label:'class', cls:'col-class' },
  { key:'worst_f1', label:'worst F1', score:true },
  { key:'pca_f1', label:'PCA F1', score:true },
  { key:'mds_f1', label:'MDS F1', score:true },
  { key:'pca_js', label:'PCA JS', score:true },
  { key:'mds_js', label:'MDS JS', score:true },
];
function renderTable(d){
  const rows = d.untrustworthy.slice();
  $('tableMeta').innerHTML = rows.length
    ? `<span class="mono">${rows.length}</span> points flagged.`
    : 'No points exceed the current threshold: try lowering it.';
  const thead = $('dtable').querySelector('thead');
  const tbody = $('dtable').querySelector('tbody');
  thead.innerHTML = '<tr>' + TABLE_COLS.map(c=>{
    const sortCls = state.tableSort.key===c.key ? (state.tableSort.dir>0?'sorted-asc':'sorted-desc') : '';
    return `<th class="${c.cls||''} ${sortCls}" data-key="${c.key}">${c.label}</th>`;
  }).join('') + '</tr>';
  // sort
  const { key, dir } = state.tableSort;
  rows.sort((a,b)=>{
    const av=a[key], bv=b[key];
    if(typeof av==='string') return dir*av.localeCompare(bv);
    return dir*(av-bv);
  });
  tbody.innerHTML = rows.map(r=> '<tr>' + TABLE_COLS.map(c=>{
    let v=r[c.key];
    if(c.cls==='col-class'){
      const code=d.target_names.indexOf(v);
      return `<td class="col-class"><span class="class-chip" style="background:${classColor(code<0?0:code)}"></span>${v}</td>`;
    }
    if(c.score){
      const t = Math.max(0, Math.min(1, v));
      const cls = t<0.34?'cell-fn':(t<0.67?'cell-zero':'cell-fp');
      return `<td class="${cls}">${v.toFixed(3)}</td>`;
    }
    if(c.err){
      const cls = v<0?'cell-fn':(v>0?'cell-fp':'cell-zero');
      return `<td class="${cls}">${v>=0?'+':''}${v.toFixed(3)}</td>`;
    }
    if(c.part){
      const cls = v>0 ? (c.part==='fp'?'cell-fp':'cell-fn') : 'cell-zero';
      return `<td class="${cls}">${v.toFixed(3)}</td>`;
    }
    return `<td>${v}</td>`;
  }).join('') + '</tr>').join('');

  thead.querySelectorAll('th').forEach(th=>{
    th.addEventListener('click', ()=>{
      const k=th.dataset.key;
      if(state.tableSort.key===k) state.tableSort.dir*=-1;
      else state.tableSort={key:k,dir:-1};
      renderTable(d);
    });
  });
}

/* ERROR SPACE: integrated PCA + MDS + error-space exploration
   Three linked plots share one selection. Cross-brushing on any
   plot, quadrant buttons, and a shared detail table all drive
   the same state.esSel.
*/

// which quadrant each point falls in, by PCA/MDS error sign
// Near-zero threshold: a point whose combined error magnitude is within this
// fraction of vmax is treated as "close enough to zero" that its quadrant is
// noise. CAVEAT: near-zero net error does NOT mean faithful -- a balanced-severe
// point (large FP and large FN that cancel) also lands here; marker SIZE (total
// distortion) is what separates the genuinely faithful from the balanced-severe.
const ES_NEARZERO_FRAC = 0.15;
function esNearZeroTau(d){ return ES_NEARZERO_FRAC * (d ? d.vmax : 1); }
function esIsNearZero(pe, me, tau){ return Math.hypot(pe, me) < tau; }

const ES_QUADS = {
  bothFP:     (pe,me)=> pe>0 && me>0,
  bothFN:     (pe,me)=> pe<0 && me<0,
  pcaFPmdsFN: (pe,me)=> pe>0 && me<0,
  pcaFNmdsFP: (pe,me)=> pe<0 && me>0,
};

function esBaseTrace(t, vmax, d, kind, dmax){
  // kind: 'proj' (x/y are the method's 2-D coords) or 'err' (x/y are PCA/MDS error)
  const isErr = kind==='err';
  // DOUBLE ENCODING -----------------------------------------------------
  // colour = signed error (t.err); size = total distortion |FP| + |FN|.
  // For the error-space plot a point's two axes already carry both methods'
  // errors, so its "total distortion" is summed over BOTH methods.
  const tot = isErr
    ? d.table.pca.fp.map((v,i)=> Math.abs(v) + Math.abs(d.table.pca.fn[i]) +
                                 Math.abs(d.table.mds.fp[i]) + Math.abs(d.table.mds.fn[i]))
    : totalDistortion(t);
  const scaleMax = isErr ? 2*dmax : dmax;   // err plot sums two methods
  const sizes = distortionSizes(tot, scaleMax);

  // ----- colour: either the class-view, or one of the error colour modes -----
  const classView = state.esView === 'class';
  let marker;
  if(classView){
    // colour by class; size stays as distortion so structure is still visible
    marker = { size:sizes, color: classColorArray(d),
               line:{color:'rgba(0,0,0,0.45)',width:0.5} };
  } else {
    const ch = colourChannel(t, d);           // severity-weighted diverging scale
    marker = { size:sizes, color: ch.vals, colorscale:DIVERGING,
               cmin:-ch.vmax, cmax:ch.vmax,
               line:{color:'rgba(0,0,0,0.5)',width:0.5} };
  }
  const sevVals = severityError(isErr ? d.table.pca : t);  // for hover only
  return {
    type:'scattergl', mode:'markers',
    x: isErr ? d.table.pca.err : t.x,
    y: isErr ? d.table.mds.err : t.y,
    marker,
    customdata: d.table.ids.map((id,i)=>[
      d.table.class_names[i], id, d.table.pca.err[i], d.table.mds.err[i],
      t.fp[i], t.fn[i], tot[i], sevVals[i]]),
    hovertemplate: isErr
      ? 'idx %{customdata[1]}, %{customdata[0]}<br>' +
        'PCA err %{customdata[2]:.3f}, MDS err %{customdata[3]:.3f}<br>' +
        'total distortion %{customdata[6]:.3f}<extra></extra>'
      : 'idx %{customdata[1]}, %{customdata[0]}<br>' +
        'error %{customdata[2]:.3f} &nbsp; (FP %{customdata[4]:.3f}, FN %{customdata[5]:.3f})<br>' +
        'severity %{customdata[7]:+.3f}, total distortion %{customdata[6]:.3f}<extra></extra>',
    selected:{ marker:{ opacity:1 } },
    unselected:{ marker:{ opacity:0.15 } },
  };
}
function esRingTrace(){
  return { type:'scattergl', mode:'markers', x:[], y:[],
    marker:{ size:15, color:'rgba(0,0,0,0)', line:{color:'#7CFC4D',width:2.2} },
    hoverinfo:'skip', showlegend:false };
}

/** Build a Plotly annotation labelling a corner of the error-space scatter. */
function quadLabel(x, y, text, xanchor, yanchor){
  return { x, y, text, xref:'x', yref:'y', xanchor, yanchor,
    showarrow:false, font:{size:9, color:'#63707f'} };
}

function renderErrorSpace(d){
  const vmax=d.vmax;
  const dmax=distortionMax(d);      // shared size scale across PCA and MDS
  const projLayout = (title)=> baseLayout({
    title:{text:title,font:{size:12,color:'#e8edf3'},x:0.03,xanchor:'left'},
    dragmode:'select',
    xaxis:Object.assign({title:'dim 1'},AXIS), yaxis:Object.assign({title:'dim 2'},AXIS),
  });
  const errLayout = baseLayout({
    title:{text:'Error space',font:{size:12,color:'#e8edf3'},x:0.03,xanchor:'left'},
    dragmode:'select',
    xaxis:Object.assign({title:'PCA error',range:[-vmax,vmax],zeroline:true},AXIS),
    yaxis:Object.assign({title:'MDS error',range:[-vmax,vmax],zeroline:true},AXIS),
    shapes:[
      { type:'line', x0:0, x1:0, y0:-vmax, y1:vmax, line:{color:'#33404f',width:1} },
      { type:'line', x0:-vmax, x1:vmax, y0:0, y1:0, line:{color:'#33404f',width:1} },
      { type:'line', x0:-vmax, x1:vmax, y0:-vmax, y1:vmax,
        line:{color:'#2a3542',width:1,dash:'dot'} },
      // near-zero "faithful / balanced" zone: points inside have ~0 net error
      { type:'circle', xref:'x', yref:'y',
        x0:-esNearZeroTau(d), x1:esNearZeroTau(d),
        y0:-esNearZeroTau(d), y1:esNearZeroTau(d),
        line:{color:'#4a5a6a',width:1,dash:'dot'},
        fillcolor:'rgba(120,140,160,0.06)' },
    ],
    annotations:[
      quadLabel(-vmax*0.92,-vmax*0.92,'both FN','left','bottom'),
      quadLabel(vmax*0.92,vmax*0.92,'both FP','right','top'),
      quadLabel(-vmax*0.92,vmax*0.92,'PCA-FN / MDS-FP','left','top'),
      quadLabel(vmax*0.92,-vmax*0.92,'PCA-FP / MDS-FN','right','bottom'),
    ],
  });
  const cfg = Object.assign({}, PLOT_CONFIG, {modeBarButtonsToAdd:['select2d','lasso2d']});

  Plotly.react('esPca', [esBaseTrace(d.table.pca,vmax,d,'proj',dmax), esRingTrace()],
    projLayout('PCA'), cfg);
  Plotly.react('esMds', [esBaseTrace(d.table.mds,vmax,d,'proj',dmax), esRingTrace()],
    projLayout('MDS'), cfg);
  Plotly.react('esErr', [esBaseTrace(d.table.pca,vmax,d,'err',dmax), esRingTrace()],
    errLayout, cfg);

  bindEsPlots();
  bindEsQuadrants();
  updateEsQuadCounts(d);
  renderWmdsCompare(d);
  applyEsSelection(d, null);   // reflect any carried-over selection
}

/* Weighted-MDS comparison: plain vs weighted MDS layouts side by side, coloured
   by class, with the silhouette separation change quantified. Only shown when
   the run used weighted MDS. */
/** Feature-weight editor for the locality-weighted MDS. Shows each feature's
 *  weight with a multiplier slider; "Apply" re-runs with the overrides. */
function renderFeatWeights(d){
  const box=$('featWeights'); if(!box) return;
  const mc=d.mds_compare;
  const method = (mc && mc.adj_method) || (d.params && d.params.adj_method) || state.adjMethod;
  const isLocality = method==='locality';
  if(!isLocality || !mc || !mc.feature_weights){ box.hidden=true; box.innerHTML=''; return; }
  box.hidden=false;
  const w=mc.feature_weights, names=(d.table && d.table.feature_names)||w.map((_,j)=>'f'+j);
  const ov=state.featWeightOverride||{};
  let rows = w.map((val,j)=>{
    const mult = ov[j]!=null ? ov[j] : 1;
    return `<div class="fw-row"><span class="fw-name">${names[j]}</span>`+
      `<input type="range" class="fw-slider" data-fw="${j}" min="0" max="3" step="0.1" value="${mult}">`+
      `<span class="fw-mult" id="fwmult-${j}">×${(+mult).toFixed(1)}</span>`+
      `<span class="fw-eff">w=${(+val).toFixed(2)}</span></div>`;
  }).join('');
  box.innerHTML =
    `<div class="fw-head">Feature weights <span class="an-subtle">drag to boost/suppress a feature, then apply</span></div>`+
    `<div class="fw-list">${rows}</div>`+
    `<div class="fw-actions"><button class="ghost-btn ghost-btn-sm" id="fwReset">Reset weights</button>`+
    `<button class="run-btn run-btn-sm" id="fwApply">Apply &amp; re-run</button></div>`;
  box.querySelectorAll('.fw-slider').forEach(s=>{
    s.addEventListener('input', ()=>{ const el=$(`fwmult-${s.dataset.fw}`); if(el) el.textContent='×'+parseFloat(s.value).toFixed(1); });
  });
  const apply=$('fwApply');
  if(apply) apply.addEventListener('click', ()=>{
    const o={}; box.querySelectorAll('.fw-slider').forEach(s=>{ const m=parseFloat(s.value); if(Math.abs(m-1)>1e-6) o[s.dataset.fw]=m; });
    state.featWeightOverride=o; if(state.result) runAnalysis();
  });
  const rst=$('fwReset');
  if(rst) rst.addEventListener('click', ()=>{ state.featWeightOverride={}; if(state.result) runAnalysis(); });
}

function renderWmdsCompare(d){
  const box = $('wmdsCompare');
  const mc = d.mds_compare;
  // sync the method selector to the run that produced this
  const runMethod = (mc && mc.adj_method) || (d.params && d.params.adj_method);
  if(runMethod){
    const ams=$('adjMethodSeg');
    if(ams) ams.querySelectorAll('.seg-btn').forEach(x=>x.classList.toggle('active', x.dataset.am===runMethod));
  }
  renderFeatWeights(d);
  if(!mc){ if(box) box.hidden = true; if(typeof Plotly!=='undefined'){ ['wmdsPlain','wmdsWeighted'].forEach(id=>{ const el=$(id); if(el&&el.data) Plotly.purge(el); }); } return; }
  box.hidden = false;
  const dlt = mc.delta;
  const method = mc.method || 'Adjusted';
  const deltaEl = $('wmdsDelta');
  const f1d = mc.f1_delta!=null ? mc.f1_delta : 0;
  const better = f1d > 0.003, worse = f1d < -0.003;
  deltaEl.textContent = `neighbour F1 ${mc.f1_plain!=null?mc.f1_plain.toFixed(3):'?'} → ${mc.f1_weighted!=null?mc.f1_weighted.toFixed(3):'?'}  (${f1d>=0?'+':''}${f1d.toFixed(3)})`;
  deltaEl.className = 'wmds-delta ' + (better?'wmds-up':(worse?'wmds-down':'wmds-flat'));
  $('wmdsNote').innerHTML = better
    ? `The ${method} distance <strong>improved</strong> neighbourhood preservation here. Left = plain Euclidean MDS, right = ${method} MDS; brush either to see where the same points land in both. PCA and the trust scores stay on the true geometry.`
    : (worse
      ? `The ${method} distance <strong>reduced</strong> neighbourhood preservation here &mdash; whitening amplified directions that on this data carry noise. Brush either layout to compare; the evidence table below quantifies it.`
      : `The ${method} distance left neighbourhood preservation roughly unchanged. Brush either layout to compare where points land.`);

  // "adjusted distance improves projection quality" evidence: does weighting
  // raise neighbourhood-F1 and lower the FP (intrusion) and FN (extrusion) rates?
  if(mc.f1_plain != null){
    const f1up = mc.f1_delta > 0;
    const fpDown = mc.fp_weighted < mc.fp_plain, fnDown = mc.fn_weighted < mc.fn_plain;
    const arrow=(a,b)=> b<a ? '<span style="color:#7ee0a0">↓</span>' : (b>a?'<span style="color:#e8969b">↑</span>':'=');
    let extra = `<div class="wmds-fidelity">`+
      `<div class="wmds-fid-head">Does the adjusted distance improve <em>projection fidelity</em>?</div>`+
      `<table class="wmds-fid-table"><thead><tr><th></th><th>plain MDS</th><th>${method} MDS</th><th></th></tr></thead><tbody>`+
      `<tr><td>neighbour F1</td><td>${mc.f1_plain.toFixed(3)}</td><td class="${f1up?'an-best':''}">${mc.f1_weighted.toFixed(3)}</td><td>${arrow(mc.f1_plain,mc.f1_weighted)}</td></tr>`+
      `<tr><td>false-positive rate (intrusions)</td><td>${mc.fp_plain.toFixed(3)}</td><td class="${fpDown?'an-best':''}">${mc.fp_weighted.toFixed(3)}</td><td>${arrow(mc.fp_plain,mc.fp_weighted)}</td></tr>`+
      `<tr><td>false-negative rate (extrusions)</td><td>${mc.fn_plain.toFixed(3)}</td><td class="${fnDown?'an-best':''}">${mc.fn_weighted.toFixed(3)}</td><td>${arrow(mc.fn_plain,mc.fn_weighted)}</td></tr>`+
      (mc.localstress_plain!=null ? `<tr><td>local stress fidelity</td><td>${mc.localstress_plain.toFixed(3)}</td><td class="${mc.localstress_weighted>mc.localstress_plain?'an-best':''}">${mc.localstress_weighted.toFixed(3)}</td><td>${arrow(mc.localstress_plain,mc.localstress_weighted)==='='?'=':(mc.localstress_weighted>mc.localstress_plain?'<span style="color:#7ee0a0">↑</span>':'<span style="color:#e8969b">↓</span>')}</td></tr>` : '')+
      (mc.shepard_plain!=null ? `<tr><td>global distance corr. (Shepard)</td><td>${mc.shepard_plain.toFixed(3)}</td><td class="${mc.shepard_weighted>mc.shepard_plain?'an-best':''}">${mc.shepard_weighted.toFixed(3)}</td><td>${mc.shepard_weighted>mc.shepard_plain?'<span style="color:#7ee0a0">↑</span>':(mc.shepard_weighted<mc.shepard_plain?'<span style="color:#e8969b">↓</span>':'=')}</td></tr>` : '')+
      `</tbody></table>`+
      `<div class="wmds-fid-verdict">`+
        (f1up && (fpDown||fnDown)
          ? `The adjusted (weighted) distance <strong>improves projection quality</strong>: neighbour-F1 rises and the intrusion/extrusion rates fall.`
          : `Here the adjusted distance does not clearly improve neighbourhood F1, the up-weighted features may not align with the neighbourhood structure this data needs.`)+
      `</div></div>`;
    $('wmdsNote').innerHTML += extra;
  }

  if(typeof Plotly==='undefined') return;
  const classColorArr = (d.table.labels||[]).map(l=>classColor(l));
  const ids = d.table.ids;
  const mk = (xs,ys)=>({ type:'scattergl', mode:'markers', x:xs, y:ys,
    marker:{ size:7, color:classColorArr, line:{color:'rgba(0,0,0,0.5)',width:0.4} },
    customdata:ids.map((id,i)=>[d.table.class_names[i],id]),
    selected:{marker:{opacity:1}}, unselected:{marker:{opacity:0.12}},
    hovertemplate:'idx %{customdata[1]}, %{customdata[0]}<extra></extra>' });
  // a highlight-ring overlay trace (empty until a selection is made)
  const ringTrace = ()=>({ type:'scattergl', mode:'markers', x:[], y:[],
    marker:{size:13, color:'rgba(0,0,0,0)', line:{color:'#e8edf3', width:2}},
    hoverinfo:'skip', showlegend:false });
  const lay = (title)=> baseLayout({
    title:{text:title,font:{size:12,color:'#e8edf3'},x:0.03,xanchor:'left'},
    xaxis:Object.assign({title:'dim 1'},AXIS), yaxis:Object.assign({title:'dim 2'},AXIS),
    showlegend:false, height:300, dragmode:'lasso',
  });
  const plainXY = {x: mc.plain.x, y: mc.plain.y};
  const weightXY = mc.weighted_xy
    ? {x: mc.weighted_xy.x, y: mc.weighted_xy.y}
    : {x: d.table.mds.x, y: d.table.mds.y};
  const cfg = Object.assign({}, PLOT_CONFIG, {modeBarButtonsToAdd:['select2d','lasso2d']});
  Plotly.react('wmdsPlain', [mk(plainXY.x, plainXY.y), ringTrace()], lay('Plain MDS'), cfg);
  Plotly.react('wmdsWeighted', [mk(weightXY.x, weightXY.y), ringTrace()], lay((mc.method||'Adjusted')+' MDS'), cfg);

  // --- linked brushing: a selection on either plot rings the same points on
  // BOTH (they share point indices) and fills a selection table so you can see
  // exactly which points moved between the plain and adjusted layouts ---
  const applyWmdsSel = (idxSet)=>{
    state._wmdsSel = idxSet;
    const pin = Array.from(idxSet);
    Plotly.restyle('wmdsPlain', { x:[pin.map(i=>plainXY.x[i])], y:[pin.map(i=>plainXY.y[i])], 'marker.line.color':'#e8edf3' }, [1]);
    Plotly.restyle('wmdsWeighted', { x:[pin.map(i=>weightXY.x[i])], y:[pin.map(i=>weightXY.y[i])], 'marker.line.color':'#e8edf3' }, [1]);
    renderWmdsSelTable(d, idxSet, plainXY, weightXY);
  };
  const wmdsClear = ()=>{
    applyWmdsSel(new Set());
    if(typeof Plotly!=='undefined'){
      ['wmdsPlain','wmdsWeighted'].forEach(id=>{ const el=$(id); if(el&&el.data){
        Plotly.restyle(id, {selectedpoints:[null]}, [0]);
        try{ Plotly.relayout(id, {selections:[]}); }catch(e){}
      }});
    }
  };
  const onSel = (ev)=>{
    if(!ev || !ev.points){ return; }
    const idxSet = new Set(ev.points.filter(p=>p.curveNumber===0).map(p=>p.pointIndex));
    applyWmdsSel(idxSet);
  };
  const plainEl = $('wmdsPlain'), weightEl = $('wmdsWeighted');
  // hover: ring the same point in BOTH layouts and show its details
  const wmdsHoverRing = (i, on)=>{
    if(typeof Plotly==='undefined') return;
    const pin = on ? [i] : [];
    Plotly.restyle('wmdsPlain', { x:[pin.map(k=>plainXY.x[k])], y:[pin.map(k=>plainXY.y[k])],
      'marker.line.color': on?'#e0a54b':'#e8edf3' }, [1]);
    Plotly.restyle('wmdsWeighted', { x:[pin.map(k=>weightXY.x[k])], y:[pin.map(k=>weightXY.y[k])],
      'marker.line.color': on?'#e0a54b':'#e8edf3' }, [1]);
  };
  const onHover = (ev)=>{ if(!ev||!ev.points||!ev.points.length) return;
    const p=ev.points.find(pt=>pt.curveNumber===0); if(!p) return;
    const i=p.pointIndex; wmdsHoverRing(i, true);
    const ro=$('wmdsHover'), tt=d.table, nmr=d.neighbor_metrics;
    if(ro){
      const dx=weightXY.x[i]-plainXY.x[i], dy=weightXY.y[i]-plainXY.y[i];
      const move=Math.hypot(dx,dy);
      const metricCells=(mth)=>METRIC_KEYS.map(k=>`${METRIC_LABELS[k]||k} <strong>${nmr[mth][k][i].toFixed(3)}</strong>`).join(', ');
      ro.innerHTML =
        `<span class="hr-idx">idx ${tt.ids[i]}</span> <span class="class-chip" style="background:${classColor(tt.labels[i])}"></span>${tt.class_names[i]}, moved <strong>${move.toFixed(2)}</strong> plain→adjusted`+
        `<span class="hr-row"><span class="hr-m">PCA</span> ${metricCells('PCA')}</span>`+
        `<span class="hr-row"><span class="hr-m">MDS</span> ${metricCells('MDS')}</span>`;
    }
  };
  const onUnhover = ()=>{ wmdsHoverRing(0, false); if(state.result){ /* restore selection rings */ applyWmdsSel(state._wmdsSel||new Set()); }
    const ro=$('wmdsHover'); if(ro) ro.innerHTML='Hover a point to see its details; the same point is highlighted in the other layout.'; };
  if(plainEl && plainEl.on){
    plainEl.on('plotly_selected', onSel);
    plainEl.on('plotly_deselect', wmdsClear);
    plainEl.on('plotly_hover', onHover);
    plainEl.on('plotly_unhover', onUnhover);
  }
  if(weightEl && weightEl.on){
    weightEl.on('plotly_selected', onSel);
    weightEl.on('plotly_deselect', wmdsClear);
    weightEl.on('plotly_hover', onHover);
    weightEl.on('plotly_unhover', onUnhover);
  }
  const clr=$('wmdsSelClear');
  if(clr && !clr.dataset.wired){ clr.dataset.wired='1';
    clr.addEventListener('click', wmdsClear); }
}

// selection table for the Mahalanobis comparison: how far each selected point
// moved between the plain and adjusted layout, plus its per-point trust F1
function renderWmdsSelTable(d, idxSet, plainXY, weightXY){
  const head=$('wmdsSelCount'), clr=$('wmdsSelClear'), wrap=$('wmdsSelWrap');
  if(!head) return;
  if(!idxSet || !idxSet.size){
    head.textContent='No points selected: drag on either layout to compare where the same points land.';
    if(clr) clr.hidden=true; if(wrap) wrap.hidden=true; return;
  }
  // remember args so header-click sorting can re-render
  _wmdsSelCtx = {d, idxSet, plainXY, weightXY};
  const t=d.table, nm=d.neighbor_metrics;
  head.innerHTML=`<span class="mono">${idxSet.size}</span> points selected, &Delta;pos = how far the point moved from the plain to the adjusted layout.`;
  if(clr) clr.hidden=false; if(wrap) wrap.hidden=false;
  let rows=[...idxSet].map(i=>{
    const dx=weightXY.x[i]-plainXY.x[i], dy=weightXY.y[i]-plainXY.y[i];
    return { i, index:t.ids[i], cls:t.class_names[i], move: Math.hypot(dx,dy), f1: nm?nm.MDS.f1[i]:0 };
  });
  const st=state.wmdsSelSort;
  rows.sort((a,b)=>{
    let va=a[st.key], vb=b[st.key];
    if(typeof va==='string') return st.dir*va.localeCompare(vb);
    return st.dir*(va-vb);
  });
  const cols=[{k:'index',label:'index'},{k:'cls',label:'class'},{k:'move',label:'Δpos'},{k:'f1',label:'MDS F1'}];
  const tbl=$('wmdsSelTable');
  tbl.querySelector('thead').innerHTML='<tr>'+cols.map(c=>{
    const cl = st.key===c.k ? (st.dir>0?'sorted-asc':'sorted-desc') : '';
    return `<th class="th-sort ${cl}" data-wmdssort="${c.k}">${c.label}</th>`;
  }).join('')+'</tr>';
  tbl.querySelector('tbody').innerHTML=rows.slice(0,200).map(r=>{
    const code=t.labels[r.i];
    const f1cell=(()=>{const c=r.f1<0.34?'cell-fn':(r.f1<0.67?'cell-zero':'cell-fp');return `<td class="${c}">${r.f1.toFixed(3)}</td>`;})();
    return `<tr><td>${r.index}</td>`+
      `<td class="col-class"><span class="class-chip" style="background:${classColor(code)}"></span>${r.cls}</td>`+
      `<td>${r.move.toFixed(2)}</td>`+f1cell+'</tr>';
  }).join('');
  tbl.querySelectorAll('th[data-wmdssort]').forEach(th=>{
    th.addEventListener('click', ()=>{
      const k=th.dataset.wmdssort;
      if(state.wmdsSelSort.key===k) state.wmdsSelSort.dir*=-1;
      else state.wmdsSelSort={key:k, dir:1};
      if(_wmdsSelCtx) renderWmdsSelTable(_wmdsSelCtx.d, _wmdsSelCtx.idxSet, _wmdsSelCtx.plainXY, _wmdsSelCtx.weightXY);
    });
  });
}
let _wmdsSelCtx = null;

/* Adaptive-k distribution: histogram of per-point k values, shown only when
   adaptive k is enabled. Bins 3-5 / 6-8 / 9-12 / 13+, as percentages. */
function renderAdaptiveK(d){
  const mini = $('adaptkMini');
  const akd = d.adaptive_k_dist;
  if(!mini) return;
  if(!akd){ mini.hidden = true; return; }
  mini.hidden = false;
  const vals = $('adaptkMiniVals');
  if(vals){
    vals.innerHTML =
      `min <strong>${akd.k_min}</strong>, median <strong>${akd.k_median}</strong>, max <strong>${akd.k_max}</strong>`;
  }
}

let _esQuadBound=false;
function bindEsQuadrants(){
  if(_esQuadBound) return;      // buttons are static; bind once
  _esQuadBound=true;
  document.querySelectorAll('#esQuadrants .es-quad').forEach(btn=>{
    btn.addEventListener('click', ()=>{
      const d=state.result; if(!d) return;
      const quad=btn.dataset.quad;
      if(state.esActiveQuad===quad){
        // clicking the active quadrant again clears it
        state.esActiveQuad=null; state.esSel=new Set();
      }else{
        state.esActiveQuad=quad;
        const tau=esNearZeroTau(d);
        const test = quad==='nearZero'
          ? (pe,me)=> esIsNearZero(pe,me,tau)
          : ES_QUADS[quad];
        const sel=new Set();
        d.table.pca.err.forEach((pe,i)=>{ if(test(pe, d.table.mds.err[i])) sel.add(i); });
        state.esSel=sel;
      }
      // clear Plotly's own drag-selection outlines when a quadrant drives selection
      ['esPca','esMds','esErr'].forEach(id=>Plotly.relayout(id,{selections:[]}));
      applyEsSelection(d, 'quadrant');
    });
  });
  $('esClear').addEventListener('click', ()=>clearEsSelection());
}

function bindEsPlots(){
  ['esPca','esMds','esErr'].forEach(id=>{
    const div=$(id);
    div.removeAllListeners?.('plotly_selected');
    div.removeAllListeners?.('plotly_deselect');
    div.on('plotly_selected', (ev)=>{
      if(!ev || !ev.points) return;
      state.esSel = new Set(ev.points.map(p=>p.pointIndex));
      state.esActiveQuad = null;               // manual brush overrides quadrant
      applyEsSelection(state.result, 'brush');
    });
    div.on('plotly_deselect', ()=>{
      state.esSel=new Set(); state.esActiveQuad=null;
      applyEsSelection(state.result, 'brush');
    });
  });
}

// paint the shared selection onto all three plots + table + counts
function applyEsSelection(d, source){
  if(!d) return;
  const idx=[...state.esSel];
  const ringXY = (xs,ys)=>({ x:idx.map(i=>xs[i]), y:idx.map(i=>ys[i]) });
  const rP=ringXY(d.table.pca.x, d.table.pca.y);
  const rM=ringXY(d.table.mds.x, d.table.mds.y);
  const rE=ringXY(d.table.pca.err, d.table.mds.err);
  // rings must track each point's double-encoded size, or a big point's ring
  // would sit inside it and a small point's ring would float far outside
  const dmax=distortionMax(d);
  const ringSizes = (sizes)=> idx.map(i=> sizes[i] + 7);
  const szP = ringSizes(distortionSizes(totalDistortion(d.table.pca), dmax));
  const szM = ringSizes(distortionSizes(totalDistortion(d.table.mds), dmax));
  const totE = d.table.pca.fp.map((v,i)=> Math.abs(v) + Math.abs(d.table.pca.fn[i]) +
                    Math.abs(d.table.mds.fp[i]) + Math.abs(d.table.mds.fn[i]));
  const szE = ringSizes(distortionSizes(totE, 2*dmax));
  Plotly.restyle('esPca', {x:[rP.x], y:[rP.y], 'marker.size':[szP]}, [1]);
  Plotly.restyle('esMds', {x:[rM.x], y:[rM.y], 'marker.size':[szM]}, [1]);
  Plotly.restyle('esErr', {x:[rE.x], y:[rE.y], 'marker.size':[szE]}, [1]);
  const sp=[idx.length?idx:null];
  Plotly.restyle('esPca', {selectedpoints:sp}, [0]);
  Plotly.restyle('esMds', {selectedpoints:sp}, [0]);
  Plotly.restyle('esErr', {selectedpoints:sp}, [0]);
  // reflect active-quadrant button styling
  document.querySelectorAll('#esQuadrants .es-quad').forEach(b=>
    b.classList.toggle('active', b.dataset.quad===state.esActiveQuad));
  updateEsDetail(d, source);
}

function clearEsSelection(){
  state.esSel=new Set(); state.esActiveQuad=null;
  if(state.result){
    ['esPca','esMds','esErr'].forEach(id=>Plotly.relayout(id,{selections:[]}));
    applyEsSelection(state.result, null);
  }
}

// per-quadrant point counts (independent of current selection)
function updateEsQuadCounts(d){
  const counts={ bothFP:0, bothFN:0, pcaFPmdsFN:0, pcaFNmdsFP:0, nearZero:0 };
  const tau=esNearZeroTau(d);
  // "large" near-zero points = balanced-severe (cancelling FP & FN); flag them
  const dmax=distortionMax(d);
  let nearZeroLarge=0;
  d.table.pca.err.forEach((pe,i)=>{
    const me=d.table.mds.err[i];
    if(esIsNearZero(pe,me,tau)){
      counts.nearZero++;
      const tot=Math.abs(d.table.pca.fp[i])+Math.abs(d.table.pca.fn[i])+
                Math.abs(d.table.mds.fp[i])+Math.abs(d.table.mds.fn[i]);
      if(tot > 0.35*dmax) nearZeroLarge++;   // sizeable distortion despite ~0 error
      return;                                  // near-zero points don't count as a quadrant
    }
    for(const q in ES_QUADS){ if(ES_QUADS[q](pe,me)){ counts[q]++; break; } }
  });
  Object.keys(counts).forEach(q=>{
    const el=document.querySelector(`#esQuadrants .es-quad-count[data-count="${q}"]`);
    if(el) el.textContent=counts[q];
  });
  // annotate the near-zero button with how many are balanced-severe (a warning)
  const nzBtn=document.querySelector('#esQuadrants .es-quad-nearzero .es-quad-desc');
  const nzBtnEl=document.querySelector('#esQuadrants .es-quad-nearzero');
  if(nzBtn){
    if(nearZeroLarge>0){
      nzBtn.innerHTML = `<strong style="color:#e0a54b">${nearZeroLarge} are large</strong> &mdash; distorted, not faithful`;
      if(nzBtnEl) nzBtnEl.classList.add('es-quad-warn');
    } else {
      nzBtn.innerHTML = `all small &mdash; genuinely faithful`;
      if(nzBtnEl) nzBtnEl.classList.remove('es-quad-warn');
    }
  }
}

const ES_TABLE_COLS=[
  {k:'idx',l:'index'},{k:'cls',l:'class',cls:'col-class'},
  {k:'pe',l:'PCA err',err:true},{k:'me',l:'MDS err',err:true},
  {k:'pfp',l:'PCA FP',part:'fp'},{k:'pfn',l:'PCA FN',part:'fn'},
  {k:'mfp',l:'MDS FP',part:'fp'},{k:'mfn',l:'MDS FN',part:'fn'},
];
function updateEsDetail(d, source){
  const idx=[...state.esSel];
  const quadName={ bothFP:'Both FP', bothFN:'Both FN',
    pcaFPmdsFN:'PCA-FP / MDS-FN', pcaFNmdsFP:'PCA-FN / MDS-FP',
    nearZero:'Near-zero (check size for balanced-severe)' };
  $('esCount').textContent = idx.length
    ? `${idx.length} point${idx.length>1?'s':''} selected`
    : 'No points selected: drag on any plot or pick a quadrant';
  const tag=$('esSource');
  if(idx.length && state.esActiveQuad){ tag.textContent='quadrant, '+quadName[state.esActiveQuad]; tag.hidden=false; }
  else if(idx.length && source==='brush'){ tag.textContent='brushed'; tag.hidden=false; }
  else { tag.textContent=''; tag.hidden=true; }

  const thead=$('esTable').querySelector('thead');
  const tbody=$('esTable').querySelector('tbody');
  thead.innerHTML='<tr>'+ES_TABLE_COLS.map(c=>`<th class="${c.cls||''}">${c.l}</th>`).join('')+'</tr>';
  const rows=idx.map(i=>({
    idx:d.table.ids[i], cls:d.table.class_names[i],
    pe:d.table.pca.err[i], me:d.table.mds.err[i],
    pfp:d.table.pca.fp[i], pfn:d.table.pca.fn[i],
    mfp:d.table.mds.fp[i], mfn:d.table.mds.fn[i],
  })).sort((a,b)=>a.pe-b.pe);
  tbody.innerHTML = rows.map(r=>'<tr>'+ES_TABLE_COLS.map(c=>{
    let v=r[c.k];
    if(c.cls==='col-class'){
      const code=d.target_names.indexOf(v);
      return `<td class="col-class"><span class="class-chip" style="background:${classColor(code<0?0:code)}"></span>${v}</td>`;
    }
    if(c.err){ const cl=v<0?'cell-fn':(v>0?'cell-fp':'cell-zero'); return `<td class="${cl}">${v>=0?'+':''}${v.toFixed(3)}</td>`; }
    if(c.part){ const cl=v>0?(c.part==='fp'?'cell-fp':'cell-fn'):'cell-zero'; return `<td class="${cl}">${v.toFixed(3)}</td>`; }
    return `<td>${v}</td>`;
  }).join('')+'</tr>').join('');
}

/* ============================================================
   VERDICT BANNER + MODAL + ANALYSIS PANEL
   ============================================================ */
function _verdictParts(v){
  // returns {winner, loser, decisive, tie, margin, relpct, ci, d, sWin, sLose}
  const pm = v.per_method || {};
  const names = v.ranking || Object.keys(pm);
  const winner = v.winner, loser = names[names.length-1];
  return {
    winner, loser,
    decisive: !!v.decisive,
    tie: v.margin_absolute < 0.005,
    margin: v.margin_absolute,
    relpct: v.margin_relative_pct,
    ci: v.bootstrap_ci_mean_distortion_diff || [0,0],
    d: v.effect_size_cohen_d,
    sWin: pm[winner] ? pm[winner].score : null,
    sLose: pm[loser] ? pm[loser].score : null,
  };
}

function renderVerdictBanner(d){
  const banner = $('verdictBanner');
  const v = d.verdict;
  if(!v){ banner.hidden = true; return; }
  banner.hidden = false;
  const p = _verdictParts(v);
  const badge = $('vbBadge'), headline = $('vbHeadline'), sub = $('vbSub');
  banner.classList.remove('vb-decisive','vb-tie','vb-soft');
  if(p.tie){
    banner.classList.add('vb-tie');
    badge.textContent = 'TIE';
    headline.textContent = 'PCA and MDS are effectively tied';
    sub.textContent = `composite ${fmt3(p.sWin)} vs ${fmt3(p.sLose)}, no meaningful difference on this dataset`;
  } else if(p.decisive){
    banner.classList.add('vb-decisive');
    badge.textContent = p.winner;
    headline.innerHTML = `<strong>${p.winner}</strong> is the more faithful projection`;
    sub.textContent = `decisive, composite ${fmt3(p.sWin)} vs ${fmt3(p.sLose)} (+${fmt3(p.margin)}, ${p.relpct.toFixed(1)}% relative), 95% CI excludes zero`;
  } else {
    banner.classList.add('vb-soft');
    badge.textContent = p.winner;
    headline.innerHTML = `<strong>${p.winner}</strong> leads, but not decisively`;
    sub.textContent = `composite ${fmt3(p.sWin)} vs ${fmt3(p.sLose)} (+${fmt3(p.margin)}), the confidence interval straddles zero, so treat as provisional`;
  }
}

const fmt3 = (x)=> (x==null||isNaN(x)) ? ',' : (x>=0? x.toFixed(3) : x.toFixed(3));
const sgn3 = (x)=> (x>=0?'+':'') + x.toFixed(3);

/* DATA & LABELS verdict: do the class labels describe the data's structure?
   Driven by k-means ARI vs labels, with the silhouette cluster count as
   supporting evidence. Three tiers so it never over-claims. */
function renderLabelsBanner(d){
  const banner = $('labelsBanner');
  const km = d.kmeans;
  if(!km){ banner.hidden = true; return; }
  banner.hidden = false;
  const hi = km.high_d || {};
  const ari = hi.ari_vs_labels;
  const nCls = km.n_classes;
  const silK = km.best_k_by_silhouette;
  const badge = $('lbBadge'), headline = $('lbHeadline'), sub = $('lbSub');
  banner.classList.remove('vb-decisive','vb-tie','vb-soft');

  // tier by ARI: >=0.7 good, 0.4-0.7 partial, <0.4 poor
  let tier;
  if(ari == null) tier = 'partial';
  else if(ari >= 0.7) tier = 'good';
  else if(ari >= 0.4) tier = 'partial';
  else tier = 'poor';

  const ariStr = ari!=null ? ari.toFixed(2) : ',';
  const silNote = (silK!=null && nCls!=null && silK!==nCls)
    ? ` The data looks like it has ${silK} natural group${silK>1?'s':''}, not ${nCls}.`
    : (silK!=null ? ` The data's natural group count (${silK}) matches your ${nCls} labels.` : '');

  if(tier === 'good'){
    banner.classList.add('vb-decisive');       // green styling
    badge.textContent = 'LABELS ✓';
    headline.innerHTML = `The labels <strong>describe the data well</strong>`;
    sub.textContent = `k-means clusters match your labels (ARI ${ariStr}).${silNote} Class-based scores are trustworthy here.`;
  } else if(tier === 'poor'){
    banner.classList.add('vb-soft');            // amber styling
    badge.textContent = 'LABELS ✗';
    headline.innerHTML = `The labels <strong>do NOT match</strong> the data's structure`;
    sub.textContent = `k-means finds a different grouping (ARI ${ariStr}).${silNote} Prefer the geometric (label-free) metric; a "bad" projection may be the labels' fault.`;
  } else {
    banner.classList.add('vb-tie');             // neutral styling
    badge.textContent = 'LABELS ~';
    headline.innerHTML = `The labels <strong>partly match</strong> the data`;
    sub.textContent = `k-means only partially agrees with your labels (ARI ${ariStr}).${silNote} Read class-based scores with some caution.`;
  }
}

function componentBars(pm){
  // pm: per_method dict. Build a small aligned bar table for the 4 components.
  const comps = ['fidelity','trust','continuity','cluster'];
  const methods = Object.keys(pm);
  let html = '<table class="an-comp-table"><thead><tr><th>criterion</th>';
  methods.forEach(m=> html += `<th>${m}</th>`);
  html += '<th></th></tr></thead><tbody>';
  comps.forEach(c=>{
    const vals = methods.map(m=> (pm[m].components && pm[m].components[c]!=null) ? pm[m].components[c] : null);
    if(vals.every(x=>x==null)) return;
    const best = Math.max(...vals.filter(x=>x!=null));
    html += `<tr><td class="an-comp-name">${c}</td>`;
    methods.forEach((m,i)=>{
      const val = vals[i];
      if(val==null){ html += '<td>,</td>'; return; }
      const isBest = Math.abs(val-best)<1e-9;
      html += `<td class="${isBest?'an-best':''}">${val.toFixed(3)}</td>`;
    });
    // mini bar showing the gap
    const lead = methods[vals.indexOf(best)] || '';
    html += `<td class="an-comp-lead">${lead}</td></tr>`;
  });
  // weights row
  const anyM = methods[0];
  const w = pm[anyM] && pm[anyM].weights_used ? pm[anyM].weights_used : null;
  html += '</tbody></table>';
  return html;
}

/* ============================================================
   TRUST ANALYSIS: per-point neighbourhood-preservation F1 / JS
   ============================================================ */

// which point indices pass the current class + attribute filters (not brushing)
function trustFilteredIndices(d){
  const t = d.table, n = t.ids.length;
  const out = [];
  const fc = state.filterClasses, af = state.attrFilter, sf = state.scoreFilter;
  const nm = d.neighbor_metrics, key = state.trustMetric;
  for(let i=0;i<n;i++){
    if(fc.size && !fc.has(t.labels[i])) continue;
    if(af){
      const v = t.feature_values[i][af.col];
      if(v < af.lo || v > af.hi) continue;
    }
    if(sf && nm){
      // keep only points whose score is within [lo,hi] in BOTH projections
      const sp = nm.PCA[key][i], sm = nm.MDS[key][i];
      if(sp < sf.lo || sp > sf.hi || sm < sf.lo || sm > sf.hi) continue;
    }
    out.push(i);
  }
  return out;
}

// the active selection = intersection of brushing and filters (or filters alone)
function trustActiveSet(d){
  const filt = new Set(trustFilteredIndices(d));
  if(state.trustSel && state.trustSel.size){
    return new Set([...state.trustSel].filter(i=>filt.has(i)));
  }
  return filt;
}

function renderTrust(d){
  const nm = d.neighbor_metrics;
  const panel = document.querySelector('[data-panel="trust"]');
  if(!nm){ if(panel) panel.style.display='none'; return; }
  if(panel) panel.style.display='';
  buildTrustFilters(d);
  renderTrustProjections(d);
  renderTrustBox(d);
  renderTrustCompare(d);
  renderTrustSelTable(d);
}

const F1_SCALE = [[0,'#d24b52'],[0.5,'#e0a54b'],[1,'#4ec97a']];  // red→amber→green

// the two projections coloured by the chosen per-point score
const F1_SCALE_LOCAL = [[0,'#d24b52'],[0.5,'#e0a54b'],[1,'#4ec97a']];
const METRIC_KEYS = ['f1','js_fidelity','localstress','mrre','lindist'];

function renderTrustProjections(d){
  if(typeof Plotly==='undefined') return;
  const nm = d.neighbor_metrics, key = state.trustMetric;
  const t = d.table, ids = t.ids;
  const active = trustActiveSet(d);
  const filtering = active.size !== ids.length;
  const classView = state.trustView === 'class';

  const baseMarker = (scores)=> classView
    ? { size: ids.map((_,i)=>filtering && !active.has(i) ? 4 : 8),
        color: (t.labels||[]).map(l=>classColor(l)),
        line:{color:'rgba(0,0,0,0.45)',width:0.4} }
    : { size: ids.map((_,i)=>filtering && !active.has(i) ? 4 : 8),
        color: scores, colorscale:F1_SCALE_LOCAL, cmin:0, cmax:1,
        colorbar:{title:{text:METRIC_LABELS[key]||key,font:{size:9,color:'#9aa7b6'}},thickness:10,len:0.9},
        line:{color:'rgba(0,0,0,0.4)',width:0.4} };

  const mk = (proj, scores)=>({
    type:'scattergl', mode:'markers', x:proj.x, y:proj.y,
    marker: baseMarker(scores),
    customdata: ids.map((id,i)=>[t.class_names[i], id, scores[i]]),
    hovertemplate:'idx %{customdata[1]}, %{customdata[0]}<br>'+(METRIC_LABELS[key]||key)+' %{customdata[2]:.3f}<extra></extra>',
    selected:{marker:{opacity:1}}, unselected:{marker:{opacity: filtering?0.12:1}},
  });
  // ring-overlay trace for cross-plot hover + brush highlight (empty initially)
  const ring = ()=>({ type:'scattergl', mode:'markers', x:[], y:[],
    marker:{size:13, color:'rgba(0,0,0,0)', line:{color:'#e8edf3',width:2}},
    hoverinfo:'skip', showlegend:false });
  const lay=(title)=>baseLayout({
    title:{text:title,font:{size:12,color:'#e8edf3'},x:0.03,xanchor:'left'},
    xaxis:Object.assign({title:'dim 1'},AXIS), yaxis:Object.assign({title:'dim 2'},AXIS),
    showlegend:false, height:320, dragmode:'lasso',
  });
  const cfg = Object.assign({}, PLOT_CONFIG, {modeBarButtonsToAdd:['select2d','lasso2d']});
  Plotly.react('trustPca', [mk(t.pca, nm.PCA[key]), ring()], lay('PCA'), cfg);
  Plotly.react('trustMds', [mk(t.mds, nm.MDS[key]), ring()], lay('MDS'), cfg);
  // reflect current selection as rings
  if(state.trustSel && state.trustSel.size) applyTrustRings(d, state.trustSel);

  // (Re)bind interaction handlers AFTER every react. scattergl can drop its
  // selection listeners when the traces are replaced (e.g. on a metric or
  // colour-by switch), so we clear and re-attach every time rather than once,
  // this is what makes brushing work under JS fidelity / classes view too.
  const pcaEl=$('trustPca'), mdsEl=$('trustMds');
  const onSel = (ev)=>{
    if(!ev || !ev.points){ return; }
    const idx = new Set(ev.points.filter(p=>p.curveNumber===0).map(p=>p.pointIndex));
    state.trustSel = idx.size ? idx : null;
    applyTrustRings(state.result, state.trustSel || new Set());
    renderTrustBox(state.result); renderTrustCompare(state.result);
    renderTrustSelTable(state.result); updateTrustFilterNote(state.result);
  };
  const onDesel = ()=>{ clearTrustSelection(); };
  const onHover = (ev)=>{ if(!ev||!ev.points||!ev.points.length) return;
    const p=ev.points.find(pt=>pt.curveNumber===0); if(!p) return;
    const i=p.pointIndex;
    applyTrustRings(state.result, new Set([i]), /*hover*/true);
    // show the point's stats (all metrics, both projections) in the readout
    const ro=$('trustHover'); const r=state.result;
    if(ro && r){ const nmr=r.neighbor_metrics, tt=r.table;
      const fmt=(v)=>v.toFixed(3);
      const metricCells=(mth)=>METRIC_KEYS.map(k=>`${METRIC_LABELS[k]||k} <strong>${fmt(nmr[mth][k][i])}</strong>`).join(', ');
      ro.innerHTML =
        `<span class="hr-idx">idx ${tt.ids[i]}</span> <span class="class-chip" style="background:${classColor(tt.labels[i])}"></span>${tt.class_names[i]}`+
        `<span class="hr-row"><span class="hr-m">PCA</span> ${metricCells('PCA')}</span>`+
        `<span class="hr-row"><span class="hr-m">MDS</span> ${metricCells('MDS')}</span>`;
    }
  };
  const onUnhover = ()=>{ applyTrustRings(state.result, state.trustSel || new Set());
    const ro=$('trustHover'); if(ro) ro.innerHTML='Hover a point to see its scores in both projections; it is highlighted in the other plot too.'; };
  [pcaEl, mdsEl].forEach(el=>{
    if(el && el.on){
      if(el.removeAllListeners){
        el.removeAllListeners('plotly_selected');
        el.removeAllListeners('plotly_deselect');
        el.removeAllListeners('plotly_hover');
        el.removeAllListeners('plotly_unhover');
      }
      el.on('plotly_selected', onSel);
      el.on('plotly_deselect', onDesel);
      el.on('plotly_hover', onHover);
      el.on('plotly_unhover', onUnhover);
    }
  });
}

// draw highlight rings for a set of point indices on BOTH trust projections
function applyTrustRings(d, idxSet, hover){
  if(typeof Plotly==='undefined' || !d) return;
  const t=d.table, pin=[...idxSet];
  const col = hover ? '#e0a54b' : '#e8edf3';
  ['trustPca','trustMds'].forEach(id=>{
    const el=$(id); if(!el||!el.data) return;
    const xy = id==='trustPca' ? t.pca : t.mds;
    Plotly.restyle(id, { x:[pin.map(i=>xy.x[i])], y:[pin.map(i=>xy.y[i])],
      'marker.line.color': col }, [1]);
  });
}

// selection detail table (same style as the error-space one)
function renderTrustSelTable(d){
  const sel = state.trustSel;
  const head = $('trustSelCount'), clr = $('trustSelClear'), wrap = $('trustSelWrap');
  if(!head) return;
  if(!sel || !sel.size){
    head.textContent = 'No points selected: drag a box or lasso on either plot.';
    if(clr) clr.hidden = true; if(wrap) wrap.hidden = true; return;
  }
  const nm=d.neighbor_metrics, key=state.trustMetric, t=d.table;
  head.innerHTML = `<span class="mono">${sel.size}</span> points selected, scores shown for the active metric (${METRIC_LABELS[key]||key}).`;
  if(clr) clr.hidden = false; if(wrap) wrap.hidden = false;
  const tbl = $('trustSelTable');
  // sortable columns
  const cols=[
    {k:'index', label:'index', get:i=>t.ids[i]},
    {k:'class', label:'class', get:i=>t.class_names[i]},
    {k:'pca',   label:'PCA',   get:i=>nm.PCA[key][i]},
    {k:'mds',   label:'MDS',   get:i=>nm.MDS[key][i]},
  ];
  const st = state.trustSelSort;
  const col = cols.find(c=>c.k===st.key) || cols[2];
  const idx=[...sel].sort((a,b)=>{
    let va=col.get(a), vb=col.get(b);
    if(typeof va==='string'){ return st.dir*va.localeCompare(vb); }
    return st.dir*(va-vb);
  });
  tbl.querySelector('thead').innerHTML = '<tr>'+cols.map(c=>{
    const cls = st.key===c.k ? (st.dir>0?'sorted-asc':'sorted-desc') : '';
    return `<th class="th-sort ${cls}" data-selsort="${c.k}">${c.label}</th>`;
  }).join('')+'</tr>';
  tbl.querySelector('tbody').innerHTML = idx.slice(0,200).map(i=>{
    const code=t.labels[i];
    const cell=(v)=>{ const c=v<0.34?'cell-fn':(v<0.67?'cell-zero':'cell-fp'); return `<td class="${c}">${v.toFixed(3)}</td>`; };
    return `<tr><td>${t.ids[i]}</td>`+
      `<td class="col-class"><span class="class-chip" style="background:${classColor(code)}"></span>${t.class_names[i]}</td>`+
      cell(nm.PCA[key][i])+cell(nm.MDS[key][i])+'</tr>';
  }).join('');
  // header click -> sort
  tbl.querySelectorAll('th[data-selsort]').forEach(th=>{
    th.addEventListener('click', ()=>{
      const k=th.dataset.selsort;
      if(state.trustSelSort.key===k) state.trustSelSort.dir*=-1;
      else state.trustSelSort={key:k, dir:1};
      renderTrustSelTable(d);
    });
  });
}

const METRIC_LABELS = { f1:'neighbour F1', js_fidelity:'JS fidelity',
  localstress:'local stress', mrre:'MRRE', lindist:'linear dist' };

// quartile helper
function quantile(sorted, q){
  if(!sorted.length) return 0;
  const pos=(sorted.length-1)*q, base=Math.floor(pos), rest=pos-base;
  return sorted[base+1]!==undefined ? sorted[base]+rest*(sorted[base+1]-sorted[base]) : sorted[base];
}
function boxStats(vals){
  const s=[...vals].sort((a,b)=>a-b), n=s.length;
  if(!n) return null;
  const q1=quantile(s,0.25), q2=quantile(s,0.5), q3=quantile(s,0.75), iqr=q3-q1;
  const loW=Math.max(s[0], q1-1.5*iqr), hiW=Math.min(s[n-1], q3+1.5*iqr);
  const mean=s.reduce((a,b)=>a+b,0)/n;
  const outliers=s.filter(v=>v<loW||v>hiW);
  return {n, min:s[0], max:s[n-1], q1, median:q2, q3, iqr, loW, hiW, mean, outliers};
}

// boxplots of the score distribution, PCA vs MDS, over the active subset
function renderTrustBox(d){
  if(typeof Plotly==='undefined') return;
  const nm=d.neighbor_metrics, key=state.trustMetric, t=d.table;
  const active=trustActiveSet(d);
  const idxArr=[...active];
  // "within" mode recomputes the score against ONLY the selected neighbours;
  // "all" mode keeps the score computed against every point. Recompute is a
  // frontend approximation, so when 'within' is chosen we note it and still use
  // the precomputed per-point score restricted to the subset (a full recompute
  // would require the distance matrix on the client). We surface the choice.
  const pick=(method)=> idxArr.map(i=> nm[method][key][i]);
  const pcaV=pick('PCA'), mdsV=pick('MDS');
  const mlabel = METRIC_LABELS[key] || key;
  // HORIZONTAL boxes: value on x, method on y. pointpos:0 keeps outlier dots on
  // the same line as the box centre. hoveron:'points' + no box hovertemplate
  // means ONLY the outlier markers are hoverable (the box itself is not).
  const traces=[
    {type:'box', x:pcaV, name:'PCA', orientation:'h', marker:{color:'#3f7cc4', size:5},
     boxmean:true, boxpoints:'outliers', pointpos:0, jitter:0, hoveron:'points',
     hovertemplate:'PCA outlier, '+mlabel+' %{x:.3f}<extra></extra>'},
    {type:'box', x:mdsV, name:'MDS', orientation:'h', marker:{color:'#e0a54b', size:5},
     boxmean:true, boxpoints:'outliers', pointpos:0, jitter:0, hoveron:'points',
     hovertemplate:'MDS outlier, '+mlabel+' %{x:.3f}<extra></extra>'},
  ];
  Plotly.react('trustBox', traces, baseLayout({
    xaxis:Object.assign({title:mlabel, range:[0,1]},AXIS),
    yaxis:Object.assign({automargin:true},AXIS), showlegend:false,
    height:190, margin:{l:46,r:14,t:8,b:34}, boxgap:0.35,
  }), PLOT_CONFIG);
  // explicit numbers beneath, now including the lower/upper WHISKER FENCES
  const sp=boxStats(pcaV), sm=boxStats(mdsV);
  const fmt=(s)=> s? `min ${s.min.toFixed(2)}, lower fence ${s.loW.toFixed(2)}, Q1 ${s.q1.toFixed(2)}, med ${s.median.toFixed(2)}, Q3 ${s.q3.toFixed(2)}, upper fence ${s.hiW.toFixed(2)}, max ${s.max.toFixed(2)}, mean ${s.mean.toFixed(3)}, ${s.outliers.length} outlier${s.outliers.length!==1?'s':''}` : ',';
  const st=$('trustBoxStats');
  if(st){
    st.innerHTML =
      `<div class="tbs-row"><span class="tbs-tag tbs-pca">PCA, n=${sp?sp.n:0}</span><span class="tbs-vals">${fmt(sp)}</span></div>`+
      `<div class="tbs-row"><span class="tbs-tag tbs-mds">MDS, n=${sm?sm.n:0}</span><span class="tbs-vals">${fmt(sm)}</span></div>`+
      (active.size!==t.ids.length ? `<div class="tbs-note">Showing ${active.size} of ${t.ids.length} points (filtered/brushed).${state.brushMode==='within'?' Scores are the original per-point values restricted to this subset.':''}</div>` : '');
  }
}

// F1 vs JS-fidelity scatter for the chosen method, over the active subset
function renderTrustCompare(d){
  if(typeof Plotly==='undefined') return;
  const nm=d.neighbor_metrics, m=state.cmpMethod, t=d.table;
  const active=trustActiveSet(d);
  const xKey=state.cmpX, yKey=state.cmpY;
  const xArr=nm[m][xKey], yArr=nm[m][yKey];
  const xs=[], ys=[], cd=[];
  active.forEach(i=>{ xs.push(xArr[i]); ys.push(yArr[i]); cd.push([t.class_names[i], t.ids[i]]); });
  const xLab=METRIC_LABELS[xKey]||xKey, yLab=METRIC_LABELS[yKey]||yKey;
  // colour by |disagreement| = distance from the y=x diagonal
  const disagree = xs.map((x,i)=>Math.abs(x-ys[i]));
  const same = xKey===yKey;
  Plotly.react('trustCompare', [
    {type:'scattergl', mode:'markers', x:xs, y:ys, customdata:cd,
     marker:{size:7, color: same?'#46566a':disagree, colorscale:[[0,'#46566a'],[0.5,'#e0a54b'],[1,'#d24b52']],
       cmin:0, cmax:0.6, showscale: !same,
       colorbar: same?undefined:{title:{text:'|disagree|',font:{size:9,color:'#9aa7b6'}},thickness:10,len:0.9},
       line:{color:'rgba(0,0,0,0.4)',width:0.4}},
     hovertemplate:'idx %{customdata[1]}, %{customdata[0]}<br>'+xLab+' %{x:.3f}, '+yLab+' %{y:.3f}<extra></extra>'},
    {type:'scatter', mode:'lines', x:[0,1], y:[0,1], line:{color:'#63707f',width:1,dash:'dash'}, hoverinfo:'skip', showlegend:false},
  ], baseLayout({
    xaxis:Object.assign({title:xLab,range:[-0.02,1.02]},AXIS),
    yaxis:Object.assign({title:yLab,range:[-0.02,1.02]},AXIS),
    showlegend:false, height:340, margin:{l:52,r:14,t:12,b:40},
  }), PLOT_CONFIG);
}

// Build a clean dual-range control (two handles, min never exceeds max, they
// may be equal). Returns the wrapper element; calls onChange(lo, hi) live.
function makeDualRange(min, max, onChange){
  const step = (max - min) / 100 || 0.01;
  const wrap = document.createElement('span');
  wrap.className = 'dual-range';
  wrap.innerHTML =
    `<input type="range" class="dr-lo" min="${min}" max="${max}" step="${step}" value="${min}">`+
    `<input type="range" class="dr-hi" min="${min}" max="${max}" step="${step}" value="${max}">`+
    `<span class="dr-vals"></span>`;
  const lo = wrap.querySelector('.dr-lo'), hi = wrap.querySelector('.dr-hi');
  const vals = wrap.querySelector('.dr-vals');
  const fmt = (max - min) >= 100 ? 0 : 2;
  const upd = ()=>{
    let a = parseFloat(lo.value), b = parseFloat(hi.value);
    // clamp: lo may not exceed hi and hi may not go below lo (equal is allowed)
    if(a > b){ if(document.activeElement === lo){ b = a; hi.value = b; } else { a = b; lo.value = a; } }
    vals.textContent = `${a.toFixed(fmt)} – ${b.toFixed(fmt)}`;
    onChange(a, b);
  };
  lo.addEventListener('input', upd);
  hi.addEventListener('input', upd);
  upd();
  return wrap;
}

// build the class + score + attribute filter controls (once per run)
function buildTrustFilters(d){
  const t=d.table;
  // classes
  const cls=$('tfClasses');
  if(cls && cls.dataset.built!==String(t.ids.length)+t.class_names[0]){
    cls.dataset.built=String(t.ids.length)+t.class_names[0];
    const uniq=[...new Set(t.labels)].sort((a,b)=>a-b);
    cls.innerHTML=uniq.map(code=>{
      const nm=t.class_names[t.labels.indexOf(code)];
      return `<button class="tf-chip" data-class="${code}">${nm}</button>`;
    }).join('');
    cls.querySelectorAll('.tf-chip').forEach(b=>b.addEventListener('click',()=>{
      const code=parseInt(b.dataset.class);
      if(state.filterClasses.has(code)){ state.filterClasses.delete(code); b.classList.remove('active'); }
      else { state.filterClasses.add(code); b.classList.add('active'); }
      refreshTrust(d);
    }));
  }

  // SCORE filter: a 0–1 dual range on the currently-selected metric, keeping
  // only points whose PCA and MDS score both fall in the window. Rebuilt when
  // the metric changes (so the label matches).
  const sc=$('tfScore');
  if(sc && sc.dataset.metric!==state.trustMetric){
    sc.dataset.metric=state.trustMetric;
    state.scoreFilter=null;   // reset when metric changes
    sc.innerHTML='';
    sc.appendChild(makeDualRange(0, 1, (a,b)=>{
      state.scoreFilter = (a<=0 && b>=1) ? null : {lo:a, hi:b};
      refreshTrust(d);
    }));
    const lbl=$('tfScoreMetric'); if(lbl) lbl.textContent='('+(METRIC_LABELS[state.trustMetric]||state.trustMetric)+')';
  }

  // ATTRIBUTE window: feature dropdown + a clean dual range on that feature
  const attr=$('tfAttr');
  if(attr && attr.dataset.built!==t.feature_names.join(',')){
    attr.dataset.built=t.feature_names.join(',');
    const opts=t.feature_names.map((nm,j)=>`<option value="${j}">${nm}</option>`).join('');
    attr.innerHTML=`<select id="tfAttrSel" class="inp inp-sm"><option value="">, feature,</option>${opts}</select>`+
      `<span id="tfAttrRange"></span>`;
    const sel=$('tfAttrSel');
    sel.addEventListener('change',()=>{
      const j=sel.value;
      const holder=$('tfAttrRange'); holder.innerHTML='';
      if(j===''){ state.attrFilter=null; refreshTrust(d); return; }
      const col=parseInt(j);
      const vals=t.feature_values.map(r=>r[col]);
      const mn=Math.min(...vals), mx=Math.max(...vals);
      state.attrFilter={col, lo:mn, hi:mx};
      holder.appendChild(makeDualRange(mn, mx, (a,b)=>{
        state.attrFilter={col, lo:a, hi:b}; refreshTrust(d);
      }));
    });
  }

  const clr=$('tfClear');
  if(clr && !clr.dataset.wired){ clr.dataset.wired='1';
    clr.addEventListener('click',()=>{
      state.filterClasses.clear(); state.attrFilter=null; state.scoreFilter=null; state.trustSel=null;
      document.querySelectorAll('#tfClasses .tf-chip').forEach(b=>b.classList.remove('active'));
      const sel=$('tfAttrSel'); if(sel){ sel.value=''; } const ar=$('tfAttrRange'); if(ar) ar.innerHTML='';
      // rebuild the score range to reset its handles
      const sc=$('tfScore'); if(sc){ sc.dataset.metric=''; }
      // clear any lasso/dimming on the plots
      if(typeof Plotly!=='undefined'){
        ['trustPca','trustMds'].forEach(id=>{ const el=$(id); if(el&&el.data){
          Plotly.restyle(id, {selectedpoints:[null]}, [0]);
          try{ Plotly.relayout(id, {selections:[]}); }catch(e){}
        }});
      }
      buildTrustFilters(d);
      refreshTrust(d);
    });
  }
  updateTrustFilterNote(d);
}
function updateTrustFilterNote(d){
  const note=$('tfNote'); if(!note) return;
  const active=trustActiveSet(d), n=d.table.ids.length;
  const parts=[];
  if(state.filterClasses.size) parts.push(`${state.filterClasses.size} class${state.filterClasses.size>1?'es':''}`);
  if(state.scoreFilter) parts.push(`${METRIC_LABELS[state.trustMetric]||state.trustMetric} ∈ [${state.scoreFilter.lo.toFixed(2)}, ${state.scoreFilter.hi.toFixed(2)}]`);
  if(state.attrFilter) parts.push(`${d.table.feature_names[state.attrFilter.col]} ∈ [${state.attrFilter.lo.toFixed(2)}, ${state.attrFilter.hi.toFixed(2)}]`);
  if(state.trustSel && state.trustSel.size) parts.push('a brushed region');
  note.textContent = parts.length
    ? `Active filters: ${parts.join(', ')} → ${active.size} of ${n} points.`
    : `No filters, all ${n} points. Brush a projection, pick classes, or set a score / attribute window to focus.`;
}
function refreshTrust(d){
  renderTrustProjections(d); renderTrustBox(d); renderTrustCompare(d); updateTrustFilterNote(d);
}

// wire the trust-panel segmented controls once
function bindTrustControls(){
  const seg=(id, fn)=>{ const el=$(id); if(!el) return;
    el.addEventListener('click',(e)=>{ const b=e.target.closest('.seg-btn'); if(!b) return;
      el.querySelectorAll('.seg-btn').forEach(x=>x.classList.toggle('active',x===b)); fn(b); }); };
  seg('trustMetricSeg', b=>{ state.trustMetric=b.dataset.tm; if(state.result){ buildTrustFilters(state.result); renderTrustProjections(state.result); renderTrustBox(state.result); renderTrustSelTable(state.result); } });
  seg('trustViewSeg', b=>{ state.trustView=b.dataset.tv; if(state.result) renderTrustProjections(state.result); });
  seg('brushModeSeg', b=>{ state.brushMode=b.dataset.bm; if(state.result) renderTrustBox(state.result); });
  seg('cmpMethodSeg', b=>{ state.cmpMethod=b.dataset.cm; if(state.result) renderTrustCompare(state.result); });
  const cx=$('cmpXSel'), cy=$('cmpYSel');
  if(cx) cx.addEventListener('change', ()=>{ state.cmpX=cx.value; if(state.result) renderTrustCompare(state.result); });
  if(cy) cy.addEventListener('change', ()=>{ state.cmpY=cy.value; if(state.result) renderTrustCompare(state.result); });
  const clr=$('trustSelClear');
  if(clr) clr.addEventListener('click', ()=> clearTrustSelection());
}

// Fully reset the trust plots to their normal state: no selection, no lasso
// box, uniform marker size, no dimming.
function clearTrustSelection(){
  state.trustSel = null;
  if(typeof Plotly!=='undefined'){
    ['trustPca','trustMds'].forEach(id=>{
      const el=$(id); if(!el||!el.data) return;
      // clear Plotly's selection (removes dimming) and the lasso/box outline
      Plotly.restyle(id, {selectedpoints:[null]}, [0]);
      try{ Plotly.relayout(id, {selections:[], dragmode:'lasso'}); }catch(e){}
    });
  }
  // re-render so marker sizes/opacity return to normal, then clear rings
  if(state.result){
    renderTrustProjections(state.result);
    applyTrustRings(state.result, new Set());
    renderTrustBox(state.result); renderTrustCompare(state.result);
    renderTrustSelTable(state.result); updateTrustFilterNote(state.result);
  }
}

function renderAnalysisPanel(d){
  renderVerdictCard(d);
  renderAnnealCard(d);
  renderWmdsCompare(d);
}

function renderVerdictCard(d){
  const v = d.verdict;
  const chip = $('anVerdictChip'), body = $('anVerdictBody'),
        comp = $('anVerdictComponents'), note = $('anVerdictNote');
  if(!v){ chip.textContent='n/a'; body.textContent='Verdict unavailable.'; comp.innerHTML=''; note.textContent=''; return; }
  const p = _verdictParts(v);
  chip.textContent = p.tie ? 'tie' : (p.decisive ? 'decisive' : 'provisional');
  chip.className = 'an-chip ' + (p.decisive ? 'an-chip-good' : (p.tie ? 'an-chip-quiet' : 'an-chip-warn'));
  body.innerHTML = `<p class="an-verdict-sentence">${v.verdict}</p>`;
  // SIMPLIFIED per request: no CI plot, no component bar chart. Just the four-
  // criteria table (each name clickable for an explanation) and the adjustable
  // weights (default 25% each). The composite recomputes live as weights change.
  comp.innerHTML =
    '<div class="verdict-2col">' +
      '<div class="verdict-col-table">' +
        '<h4>The four criteria <span class="an-subtle">click a name for detail</span></h4>' +
        componentBars(v.per_method) +
      '</div>' +
      '<div class="verdict-col-weights">' +
        '<div class="wtsliders wtsliders-sm" id="wtSliders">'+
          '<h4>Adjust the weights <span class="an-subtle">must sum to 100%</span></h4>'+
          wtSliderRow('fidelity', 0.25) + wtSliderRow('trust', 0.25) +
          wtSliderRow('continuity', 0.25) + wtSliderRow('cluster', 0.25) +
          '<div class="wt-sum-row"><span id="wtSum">sum: 100%</span>'+
            '<button class="ghost-btn ghost-btn-sm" id="wtReset">Reset</button>'+
            '<button class="run-btn run-btn-sm" id="wtRerun">Re-run</button></div>'+
          '<div class="wt-live" id="wtLive"></div>'+
        '</div>' +
      '</div>' +
    '</div>';
  note.innerHTML = '';
  renderVerdictCharts(d);
  bindWeightSliders(d);
  bindCriterionInfo();
}

/* clicking a criterion name in the component table opens its explanation */
const CRITERION_INFO = {
  fidelity: { title: 'Fidelity criterion',
    body: `<p><strong>What it is.</strong> The per-point neighbourhood-preservation error averaged over all points, turned into a 0&ndash;1 score (higher = better). It is the toolkit's own intrusion/extrusion metric: for each point it counts how many of its true high-D neighbours the projection keeps and how many strangers it wrongly pulls in.</p>
    <p><strong>How it's calculated.</strong> Per point, FP = rank-weighted intrusions (points close in 2-D but far in high-D) and FN = rank-weighted extrusions (true neighbours pushed away); the score is 1 &minus; mean(|FP| + |FN|). It is the only <em>per-point</em> component, which is why it usually carries the most weight by default.</p>
    <p><strong>Why it is indicative.</strong> It directly measures the thing the projection is supposed to do &mdash; preserve local neighbourhoods &mdash; so a method that scores high here distorts fewer points.</p>` },
  trust: { title: 'Trustworthiness criterion',
    body: `<p><strong>What it is.</strong> The classical trustworthiness score (Venna &amp; Kaski): it penalises <em>false neighbours</em> &mdash; points that appear close in the 2-D projection but are actually far apart in high-D. In [0, 1], higher = fewer false neighbours.</p>
    <p><strong>How it's calculated.</strong> For each point, it sums how many rank positions each intruding 2-D neighbour is displaced from its true high-D rank, normalised so 1 is perfect. It is an established, aggregate metric, included as an independent cross-check from the literature.</p>
    <p><strong>Why it is indicative.</strong> It answers &ldquo;can I trust that points shown close together really are close?&rdquo; &mdash; the credibility of the projection's neighbourhoods.</p>` },
  continuity: { title: 'Continuity criterion',
    body: `<p><strong>What it is.</strong> The dual of trustworthiness: it penalises <em>missing neighbours</em> &mdash; points that are genuinely close in high-D but get pushed apart in the projection. In [0, 1], higher = fewer missing neighbours.</p>
    <p><strong>How it's calculated.</strong> For each point, it sums how many rank positions each true high-D neighbour is displaced in the 2-D ordering, normalised so 1 is perfect.</p>
    <p><strong>Why it is indicative.</strong> It answers &ldquo;did the projection tear apart groups that were together?&rdquo; &mdash; the completeness of the projection's neighbourhoods. Together, trust and continuity capture the two directions of neighbourhood error.</p>` },
  cluster: { title: 'Cluster-preservation criterion',
    body: `<p><strong>What it is.</strong> How well the projection preserves the data's cluster structure, measured label-free. In [0, 1], higher = the 2-D grouping matches the high-D grouping.</p>
    <p><strong>How it's calculated.</strong> k-means is run in the high-D space and in the 2-D projection, and the two partitions are compared with the Adjusted Rand Index (ARI), which is corrected for chance. No class labels are used, so it is a purely structural comparison.</p>
    <p><strong>Why it is indicative.</strong> The other three criteria are neighbourhood-based (local); this one checks global cluster structure &mdash; whether the clumps you see in the projection correspond to real clumps in the data.</p>` },
};
function bindCriterionInfo(){
  document.querySelectorAll('#anVerdictComponents .an-comp-name').forEach(cell=>{
    const key = cell.textContent.trim().toLowerCase();
    if(!CRITERION_INFO[key]) return;
    cell.classList.add('has-info');
    cell.style.cursor='pointer';
    cell.onclick = ()=>{ const info=CRITERION_INFO[key];
      $('infoTitle').textContent=info.title; $('infoBody').innerHTML=info.body;
      $('infoModal').hidden=false; };
  });
}

/* one weight slider row */
function wtSliderRow(name, def){
  return `<label class="wt-row">`+
    `<span class="wt-name">${name}</span>`+
    `<input type="range" class="wt-slider" data-wt="${name}" min="0" max="1" step="0.05" value="${def}">`+
    `<span class="wt-val" id="wtval-${name}">${def.toFixed(2)}</span></label>`;
}

/* recompute the composite verdict on the frontend as weights change */
function bindWeightSliders(d){
  const box = $('wtSliders'); if(!box) return;
  const v = d.verdict; if(!v || !v.per_method) return;
  const readWeights = ()=>{
    const w = {};
    box.querySelectorAll('.wt-slider').forEach(s=>{ w[s.dataset.wt] = parseFloat(s.value); });
    return w;
  };
  // recompute the verdict under the current weights and show it
  const runVerdict = ()=>{
    const w = readWeights();
    const total = Object.values(w).reduce((a,b)=>a+b,0) || 1;
    const pm = v.per_method, methods = Object.keys(pm);
    const scores = {};
    methods.forEach(m=>{
      let s=0; Object.keys(w).forEach(c=>{
        const cv = pm[m].components && pm[m].components[c]!=null ? pm[m].components[c] : 0;
        s += cv * w[c];
      });
      scores[m] = s/total;
    });
    const ranked = methods.slice().sort((a,b)=>scores[b]-scores[a]);
    const win = ranked[0], lose = ranked[ranked.length-1];
    const margin = scores[win]-scores[lose];
    const live = $('wtLive');
    if(live){
      const flipped = win !== v.winner;
      live.innerHTML =
        `Under these weights: <strong>${win}</strong> ${methods.length>1?`(${scores[win].toFixed(3)}) vs ${lose} (${scores[lose].toFixed(3)})`:''}, `+
        `margin ${margin>=0?'+':''}${margin.toFixed(3)}. `+
        (flipped ? `<span class="wt-flip">⚠ the winner flipped from ${v.winner}</span>`
                 : `<span class="wt-same">same winner as the stated weights</span>`);
    }
  };
  // update the live value labels + the sum + enable/disable the re-run button
  const refresh = ()=>{
    box.querySelectorAll('.wt-slider').forEach(s=>{
      const el=$(`wtval-${s.dataset.wt}`); if(el) el.textContent=(parseFloat(s.value)*100).toFixed(0)+'%';
    });
    const w = readWeights();
    const sumPct = Math.round(Object.values(w).reduce((a,b)=>a+b,0)*100);
    const sumEl=$('wtSum'), btn=$('wtRerun');
    const ok = sumPct === 100;
    if(sumEl){ sumEl.textContent=`sum: ${sumPct}%`; sumEl.className = ok ? 'wt-sum-ok' : 'wt-sum-bad'; }
    if(btn){ btn.disabled = !ok; btn.classList.toggle('is-disabled', !ok); }
  };
  box.querySelectorAll('.wt-slider').forEach(s=> s.addEventListener('input', refresh));
  const rr=$('wtRerun'); if(rr) rr.addEventListener('click', ()=>{ if(!rr.disabled) runVerdict(); });
  const rst=$('wtReset'); if(rst) rst.addEventListener('click', ()=>{
    box.querySelectorAll('.wt-slider').forEach(s=>{ s.value='0.25'; });
    refresh(); runVerdict();
  });
  refresh(); runVerdict();   // initial state (25% each sums to 100)
}

/* Grouped component bar chart (PCA vs MDS) + confidence-interval plot. */
function renderVerdictCharts(d){
  // Verdict CI plot and component bar chart removed per request, the verdict
  // card now shows only the four-criteria table and the adjustable weights.
  return;
  /* eslint-disable no-unreachable */
  if(typeof Plotly==='undefined') return;
  const v = d.verdict; if(!v) return;
  const pm = v.per_method;
  const methods = Object.keys(pm);
  const comps = ['fidelity','trust','continuity','cluster'];
  const present = comps.filter(c => methods.some(m => pm[m].components && pm[m].components[c]!=null));

  // --- grouped bar chart: four components x two methods ---
  const bars = $('verdictBars');
  if(bars){
    const methodColor = {PCA:'#3f7cc4', MDS:'#e0a54b'};
    const traces = methods.map(m=>({
      type:'bar', name:m, x:present,
      y: present.map(c=> pm[m].components && pm[m].components[c]!=null ? pm[m].components[c] : 0),
      marker:{color: methodColor[m] || '#7f9ab5'},
      hovertemplate:`${m} %{x}: %{y:.3f}<extra></extra>`,
    }));
    Plotly.react(bars, traces, baseLayout({
      barmode:'group',
      xaxis:Object.assign({title:''},AXIS),
      yaxis:Object.assign({title:'score (0–1)', range:[0,1]},AXIS),
      showlegend:true, legend:{font:{size:10,color:'#9aa7b6'},orientation:'h',x:0,y:1.12},
      height:220, margin:{l:48,r:14,t:24,b:30},
    }), PLOT_CONFIG);
  }

  // --- confidence-interval plot: mean distortion gap + 95% CI ---
  const ciDiv = $('verdictCI');
  const ci = v.bootstrap_ci_mean_distortion_diff;
  if(ciDiv && ci){
    const lo = ci[0], hi = ci[1], mean = (lo+hi)/2;
    const decisive = lo > 0 || hi < 0;               // CI excludes zero
    const col = decisive ? '#4ec97a' : '#e0a54b';
    // pad x-range to include zero and the interval
    const pad = Math.max(0.01, (hi-lo)*0.4);
    const xlo = Math.min(0, lo) - pad, xhi = Math.max(0, hi) + pad;
    Plotly.react(ciDiv, [{
      type:'scatter', mode:'markers', x:[mean], y:[0],
      error_x:{type:'data', symmetric:false, array:[hi-mean], arrayminus:[mean-lo],
        color:col, thickness:2, width:8},
      marker:{color:col, size:14, line:{color:'#0e1116',width:1}},
      hovertemplate:`gap ${mean.toFixed(4)}, 95% CI [${lo.toFixed(4)}, ${hi.toFixed(4)}]<extra></extra>`,
    }], baseLayout({
      xaxis:Object.assign({title:'mean distortion gap (loser − winner)', range:[xlo,xhi], zeroline:false},AXIS),
      yaxis:Object.assign({showticklabels:false, range:[-1,1]},AXIS),
      height:130, margin:{l:14,r:14,t:22,b:36},
      shapes:[{type:'line', x0:0, x1:0, y0:-1, y1:1, line:{color:'#d24b52',width:1.5,dash:'dash'}}],
      annotations:[
        {x:0, y:-0.85, text:'zero (no difference)', showarrow:false, font:{size:9,color:'#d24b52'}, xanchor: (mean>=0?'left':'right'), xshift:(mean>=0?3:-3)},
        {x:mean, y:0.82, text: decisive ? '✓ CI excludes zero, reliable' : '✗ CI crosses zero, provisional',
         showarrow:false, font:{size:10, color:col}, xanchor:'center'},
      ],
    }), PLOT_CONFIG);
  }
}

function renderAnnealCard(d){
  const a = d.anneal;
  const chip = $('anAnnealChip'), stats = $('anAnnealStats'),
        note = $('anAnnealNote'), plotDiv = $('anAnnealPlot');
  if(!a){
    chip.textContent = 'off';
    stats.innerHTML = '<p class="an-muted">Simulated-annealing search did not run for this dataset.</p>';
    note.textContent=''; if(typeof Plotly!=='undefined' && plotDiv) Plotly.purge(plotDiv);
    return;
  }
  chip.textContent = `k* = ${a.best_k}`;
  const grid = a.exhaustive_count != null ? a.exhaustive_count
             : (a.k_hi != null && a.k_lo != null ? (a.k_hi - a.k_lo + 1) : ',');
  // compute acceptance rate of WORSENING proposals: of all proposals that were
  // worse than the current point, what fraction were accepted anyway?
  let worseProposed = 0, worseAccepted = 0, startScore = null, endBest = null;
  if(a.trace && a.trace.length){
    startScore = a.trace[0].best_score;
    endBest = a.trace[a.trace.length-1].best_score;
    let curBest = a.trace[0].score;
    for(let i=0;i<a.trace.length;i++){
      const t = a.trace[i];
      // a proposal is "worse" if its score is below the running current;
      // approximate using best_score progression + uphill flags
      if(t.uphill){ worseProposed++; worseAccepted++; }
    }
    // count rejected worsening moves: iterations not accepted (score dipped, not counted as uphill)
    for(let i=0;i<a.trace.length;i++){ if(a.trace[i].accepted===false) worseProposed++; }
  }
  const accRate = worseProposed>0 ? Math.round(100*worseAccepted/worseProposed) : null;
  stats.innerHTML =
    statCell('best k', a.best_k) +
    statCell('F1 at k*', (a.best_score!=null? a.best_score.toFixed(4):',')) +
    statCell('evaluations', `${a.n_evaluations}`) +
    statCell('vs exhaustive', `${a.n_evaluations}/${grid}`) +
    statCell('uphill accepted', a.uphill_moves_accepted!=null?a.uphill_moves_accepted:',') +
    statCell('worsening accept rate', accRate!=null?`${accRate}%`:',') +
    statCell('start temp', a.t0!=null?a.t0.toFixed(3):',') +
    statCell('F1 gain', (startScore!=null&&endBest!=null)?`+${(endBest-startScore).toFixed(3)}`:',');

  // self-diagnosis: healthy vs problematic search
  let diag = '';
  if(a.trace && a.trace.length){
    const n = a.trace.length;
    const improvedLate = (a.trace[n-1].best_score - a.trace[Math.floor(n*0.8)].best_score) > 1e-6;
    const flatFromStart = Math.abs(endBest - startScore) < 1e-6;
    const noUphill = (a.uphill_moves_accepted||0) === 0;
    const gained = (endBest - startScore) > 1e-6;
    const items = [];
    // good signs
    if(gained) items.push(`<li class="diag-good">✓ final best F1 (+${(endBest-startScore).toFixed(3)}) exceeds the start, the search improved</li>`);
    if((a.uphill_moves_accepted||0) > 0 && !improvedLate) items.push(`<li class="diag-good">✓ it explored early (worsening moves accepted) then settled, healthy cooling</li>`);
    // problem signs
    if(flatFromStart) items.push(`<li class="diag-bad">✗ best F1 never improved, likely stuck in a local optimum (temperature too low)</li>`);
    if(improvedLate) items.push(`<li class="diag-warn">⚠ best F1 was still rising at the end, may be under-cooled; consider more iterations</li>`);
    if(noUphill && !flatFromStart) items.push(`<li class="diag-warn">⚠ no worsening moves were accepted, the search was greedy and may have missed better regions</li>`);
    if(!items.length) items.push(`<li class="diag-good">✓ search behaved normally</li>`);
    diag = `<div class="diag-box"><div class="diag-title">Search health</div><ul class="diag-list">${items.join('')}</ul></div>`;
  }

  const reheatTxt = a.reheats ? ` It re-heated ${a.reheats} time${a.reheats>1?'s':''} after stalling, to escape local optima.` : '';
  note.innerHTML =
    `<strong>What a healthy search looks like.</strong> The best F1 should exceed the starting F1 (a real improvement), a few worsening moves should be accepted early (exploration), and the search should touch far fewer than the exhaustive count of k values.${reheatTxt}` +
    diag;

  // Plot intentionally omitted, only the numeric summary is shown.
  if(typeof Plotly!=='undefined' && plotDiv) Plotly.purge(plotDiv);
}

function statCell(label, value){
  return `<div class="an-stat"><span class="an-stat-v">${value}</span><span class="an-stat-l">${label}</span></div>`;
}

function renderMetricsCard(d){
  const r = d.rel_metrics;
  const chip = $('anMetricsChip'), grid = $('anMetricsGrid'), pc = $('anMetricsPerClass');
  const stale = document.getElementById('anMacroMicro'); if(stale) stale.remove();
  if(!r){ chip.textContent='n/a'; grid.innerHTML=''; pc.innerHTML=''; return; }
  const g = r.global;
  const gm = r.global_micro;
  chip.textContent = `k = ${r.k}`;
  grid.innerHTML =
    bigStat('purity', g.precision, 'are points among their own class?') +
    bigStat('cohesion', g.recall, 'is each class held together?') +
    bigStat('F1', (g.f1!=null?g.f1:g.f_score), 'harmonic mean of the two', true);
  // macro vs micro line: the gap reveals class imbalance
  if(gm){
    const macroF1 = g.f1!=null?g.f1:g.f_score, microF1 = gm.f1!=null?gm.f1:gm.f_score;
    const gap = Math.abs(macroF1 - microF1);
    const gapNote = gap>0.05
      ? `Because they differ by ${gap.toFixed(3)}, the classes are <strong>imbalanced or uneven</strong> &mdash; some classes score much better than others, and the small classes are pulling the macro score ${macroF1<microF1?'down':'up'}.`
      : `They are close (gap ${gap.toFixed(3)}), so every class scores about equally well &mdash; the overall F1 is representative of each class.`;
    grid.insertAdjacentHTML('afterend',
      `<div class="an-note macro-micro" id="anMacroMicro">`+
      `<div class="mm-row"><span class="mm-tag mm-macro">MACRO F1 ${macroF1.toFixed(3)}</span>`+
      `<span class="mm-desc">treats every <em>class</em> equally: average the F1 within each class, then average those &mdash; so a tiny class counts as much as a huge one.</span></div>`+
      `<div class="mm-row"><span class="mm-tag mm-micro">MICRO F1 ${microF1.toFixed(3)}</span>`+
      `<span class="mm-desc">treats every <em>point</em> equally: pool all points together &mdash; so large classes dominate.</span></div>`+
      `<div class="mm-gap">${gapNote}</div>`+
      `</div>`);
  }
  // per-class table
  const names = r.target_names || [];
  const pl = r.per_label || {};
  const keys = Object.keys(pl);
  if(keys.length){
    // one small radar per class, laid out in a responsive grid, so each class's
    // precision/recall/F1 shape can be read individually
    pc.innerHTML = '<h4>Per class (purity, cohesion, F1)</h4>'+
      '<p class="an-note" style="margin-bottom:8px">Each class has its own radar. A '+
      'balanced triangle means the class is pure <em>and</em> cohesive; a spike '+
      'toward one axis shows where it is weak (e.g. high purity but low cohesion = '+
      'pure but fragmented).</p>'+
      '<div class="radar-grid" id="radarGrid"></div>';
    const gridEl = $('radarGrid');
    if(typeof Plotly!=='undefined' && gridEl){
      const axes = ['purity','cohesion','F1'];
      const palette = ['#3f7cc4','#e0a54b','#4ec97a','#d24b52','#9b7cc4','#5ac8c8','#c88a4a','#8a94a0','#e88a8f','#82abd8'];
      keys.forEach((k,idx)=>{
        const row = pl[k];
        const f1 = row.f1!=null ? row.f1 : row.f_score;
        const nm = names[k] !== undefined ? String(names[k]) : `class ${k}`;
        const rvals = [row.precision, row.recall, f1];
        const col = palette[idx % palette.length];
        // container per class
        const cell = window.document.createElement('div');
        cell.className = 'radar-cell';
        const title = window.document.createElement('div');
        title.className = 'radar-cell-title';
        title.textContent = `${nm} (n=${row.size})`;
        const plotDiv = window.document.createElement('div');
        plotDiv.className = 'radar-cell-plot';
        plotDiv.id = `radar-${idx}`;
        cell.appendChild(title); cell.appendChild(plotDiv);
        gridEl.appendChild(cell);
        Plotly.react(plotDiv, [{
          type:'scatterpolar', mode:'lines+markers',
          r: rvals.concat([rvals[0]]), theta: axes.concat([axes[0]]),
          line:{color:col,width:2}, marker:{size:5,color:col},
          fill:'toself', fillcolor:hexToRgba(col,0.25),
          hovertemplate:`${nm} %{theta}: %{r:.3f}<extra></extra>`,
        }], baseLayout({
          polar:{ bgcolor:'#0b0e13',
            radialaxis:{range:[0,1], tickvals:[0.5,1], tickfont:{size:7,color:'#63707f'}, gridcolor:'#262f3b', linecolor:'#262f3b'},
            angularaxis:{tickfont:{size:9,color:'#9aa7b6'}, gridcolor:'#262f3b', linecolor:'#262f3b'} },
          showlegend:false, height:180, margin:{l:28,r:28,t:16,b:16},
        }), PLOT_CONFIG);
      });
    }
  } else pc.innerHTML='';
}

/** Convert a #rrggbb hex to an rgba() string with the given alpha. */
function hexToRgba(hex, a){
  const h = hex.replace('#','');
  const r = parseInt(h.slice(0,2),16), g = parseInt(h.slice(2,4),16), b = parseInt(h.slice(4,6),16);
  return `rgba(${r},${g},${b},${a})`;
}

function bigStat(label, val, sub, hero){
  const v = (val==null||isNaN(val)) ? ',' : val.toFixed(3);
  return `<div class="an-bigstat ${hero?'an-bigstat-hero':''}">`+
    `<div class="an-bigstat-v">${v}</div>`+
    `<div class="an-bigstat-l">${label}</div>`+
    `<div class="an-bigstat-s">${sub}</div></div>`;
}

function renderKmeansCard(d){
  const km = d.kmeans;
  const chip = $('anKmeansChip'), body = $('anKmeansBody');
  if(!km){ chip.textContent='n/a'; body.innerHTML=''; return; }
  const hi = km.high_d || {};
  const are = km.classes_are_clusters;
  const method = km.cluster_method || 'kmeans';
  const methodName = {kmeans:'k-means', gmm:'GMM', spectral:'spectral'}[method] || method;
  chip.textContent = (are ? 'labels ≈ clusters' : 'labels ≠ clusters') + `, ${methodName}`;
  chip.className = 'an-chip ' + (are ? 'an-chip-good' : 'an-chip-warn');
  const acc = hi.cluster_accuracy!=null ? (100*hi.cluster_accuracy).toFixed(1)+'%' : ',';
  const ariL = hi.ari_vs_labels!=null ? hi.ari_vs_labels.toFixed(3) : ',';
  let html =
    '<div class="an-stat-row">' +
    statCell('ARI vs labels', ariL) +
    statCell('matched accuracy', acc) +
    statCell('silhouette best k', km.best_k_by_silhouette) +
    statCell('# classes', km.n_classes) +
    '</div>';
  // verdict sentence
  if(are){
    html += `<p class="an-kmeans-say an-say-good">The high-dimensional clusters line up with the class labels (ARI ${ariL}), so the labels are a trustworthy reference here.</p>`;
  } else {
    html += `<p class="an-kmeans-say an-say-warn">The natural clusters do <strong>not</strong> match the labels well (ARI ${ariL})`;
    if(km.best_k_by_silhouette !== km.n_classes){
      html += `, silhouette even prefers <strong>${km.best_k_by_silhouette}</strong> clusters against <strong>${km.n_classes}</strong> classes`;
    }
    html += `. Part of any apparent "projection error" may therefore be a label/geometry mismatch, not the projection's fault.</p>`;
    if(method === 'kmeans'){
      html += `<p class="an-note">If the classes might be <em>non-convex</em> (e.g. rings or crescents), k-means cannot see them &mdash; try the <strong>spectral</strong> clusterer in the Analysis controls before concluding the labels are bad.</p>`;
    }
  }
  // per-method preservation
  if(km.methods){
    html += '<h4>Does each projection preserve the clustering?</h4><div class="an-stat-row">';
    Object.keys(km.methods).forEach(m=>{
      const ari = km.methods[m].ari_vs_high;
      html += statCell(`${m}: ARI 2-D vs high-D`, ari!=null?ari.toFixed(3):',');
    });
    html += '</div>';
  }
  // containers for the two charts
  html += '<h4>Cluster vs class confusion</h4>'+
          '<p class="an-note" style="margin-bottom:8px">Each cell is the number of points in a high-D cluster (row) that carry a given class label (column). A clean result is <strong>block-diagonal</strong>: each cluster is dominated by one class. <strong>Cluster purity</strong> = the biggest count in a row &divide; that row&rsquo;s total (how single-class each cluster is). <strong>Class coverage</strong> = the biggest count in a column &divide; that column&rsquo;s total (how much of each class landed in one cluster).</p>'+
          '<div class="plot" id="kmHeatmap" style="height:230px"></div>'+
          '<div id="kmPurity"></div>';
  body.innerHTML = html;
  renderKmeansCharts(d);
}

/* Confusion heatmap (high-D clusters x class labels) + silhouette-vs-k plot. */
function renderKmeansCharts(d){
  if(typeof Plotly==='undefined') return;
  const km = d.kmeans; if(!km) return;
  const names = (d.rel_metrics && d.rel_metrics.target_names) || d.target_names || [];

  // --- confusion heatmap ---
  const hm = $('kmHeatmap');
  if(hm && km.contingency && km.contingency.length){
    const M = km.contingency;                       // rows=clusters, cols=classes
    const nClusters = M.length, nClasses = M[0].length;
    const flatMax = Math.max(...M.flat(), 1);
    const classNames = Array.from({length:nClasses}, (_,j)=> names[j]!==undefined?String(names[j]):`class ${j}`);
    const clusterNames = Array.from({length:nClusters}, (_,i)=>{
      const ml = km.matched_labels && km.matched_labels[i]!=null ? km.matched_labels[i] : null;
      const mlName = (ml!=null && names[ml]!==undefined) ? String(names[ml]) : (ml!=null?`class ${ml}`:'');
      return mlName ? `cl ${i} → ${mlName}` : `cluster ${i}`;
    });
    const ann = [];
    for(let i=0;i<nClusters;i++) for(let j=0;j<nClasses;j++){
      const v = M[i][j];
      ann.push({x:classNames[j], y:clusterNames[i], text:String(v), showarrow:false,
        font:{size:11, color: v > 0.5*flatMax ? '#0e1116' : '#e8edf3'}});
    }
    Plotly.react(hm, [{
      type:'heatmap', z:M, x:classNames, y:clusterNames,
      colorscale:[[0,'#0e1116'],[0.5,'#2a5a8a'],[1,'#4ec97a']],
      showscale:true, colorbar:{title:{text:'points',font:{size:9,color:'#9aa7b6'}}, thickness:10, len:0.9},
      hovertemplate:'cluster %{y}, %{x}: %{z} points<extra></extra>',
    }], baseLayout({
      xaxis:Object.assign({title:'class label', side:'bottom'},AXIS),
      yaxis:Object.assign({title:'high-D cluster', automargin:true},AXIS),
      annotations:ann, height:230, margin:{l:100,r:20,t:10,b:44},
    }), PLOT_CONFIG);

    // --- cluster purity (per row) and class coverage (per column) ---
    const purityDiv = $('kmPurity');
    if(purityDiv){
      const rowTotals = M.map(r=>r.reduce((a,b)=>a+b,0));
      const colTotals = classNames.map((_,j)=> M.reduce((a,r)=>a+r[j],0));
      const rowPurity = M.map((r,i)=> rowTotals[i]>0 ? Math.max(...r)/rowTotals[i] : 0);
      const colCoverage = classNames.map((_,j)=>{
        const col = M.map(r=>r[j]); return colTotals[j]>0 ? Math.max(...col)/colTotals[j] : 0;
      });
      const overallPurity = rowTotals.reduce((a,b)=>a+b,0)>0
        ? M.reduce((a,r)=>a+Math.max(...r),0) / rowTotals.reduce((a,b)=>a+b,0) : 0;
      const pill = (label,val)=>{
        const pct = (val*100).toFixed(0);
        const cls = val>=0.8?'pcell-good':(val>=0.5?'pcell-mid':'pcell-bad');
        return `<span class="pcell ${cls}"><span class="pcell-l">${label}</span><span class="pcell-v">${pct}%</span></span>`;
      };
      let ph = `<div class="purity-head">Overall cluster purity: <strong>${(overallPurity*100).toFixed(0)}%</strong> of points fall in a cluster dominated by their own class</div>`;
      ph += '<div class="purity-grid"><div class="purity-col"><div class="purity-col-t">Cluster purity (each cluster is how single-class?)</div>';
      rowPurity.forEach((p,i)=> ph += pill(clusterNames[i], p));
      ph += '</div><div class="purity-col"><div class="purity-col-t">Class coverage (each class is how concentrated in one cluster?)</div>';
      colCoverage.forEach((cvg,j)=> ph += pill(classNames[j], cvg));
      ph += '</div></div>';
      purityDiv.innerHTML = ph;
    }
  } else {
    const purityDiv = $('kmPurity'); if(purityDiv) purityDiv.innerHTML='';
  }
}

/* verdict modal */
function openVerdictModal(){
  const d = state.result; if(!d || !d.verdict) return;
  const v = d.verdict, p = _verdictParts(v);
  const body = $('vmBody');
  let html = `<div class="vm-headline ${p.decisive?'vm-good':(p.tie?'vm-tie':'vm-soft')}">${v.verdict}</div>`;
  html += '<h4>The four criteria</h4>' + componentBars(v.per_method);
  html += `<div class="an-note" style="margin-top:12px">`+
    `<strong>Confidence.</strong> Cohen's d = ${fmt3(p.d)}; 95% bootstrap CI on the per-point distortion gap = [${sgn3(p.ci[0])}, ${sgn3(p.ci[1])}]. `+
    (p.decisive ? 'The interval excludes zero, the win is reliable.'
                : (p.tie ? 'Within the tie threshold.' : 'The interval crosses zero, provisional.'))+
    `</div>`;
  html += `<div class="an-why" style="margin-top:12px"><strong>How to read this.</strong> `+
    `Fidelity (weight 0.40) is the per-point distortion; trust and continuity (0.20 each) are the classical neighbour-preservation scalars; cluster (0.20) is how well each projection preserves the k-means partition. Weights are fixed, not tuned, and every component is shown so you can weigh them yourself.</div>`;
  body.innerHTML = html;
  $('verdictModal').hidden = false;
}
function closeVerdictModal(){ $('verdictModal').hidden = true; }


/* TABS / CONSOLE / TOAST */
const TAB_INTROS = {
  dataquality: `<strong>Data Quality</strong>, is the class structure genuinely in the data, and do the labels describe it? These answers depend only on the data, <em>not</em> on any projection. If the data has no structure, or the labels don't match it, no projection can fix that.`,
  projquality: `<strong>Projection Quality</strong>, given the data, how faithfully does each 2-D projection preserve it, and which is better? These answers are about the <em>picture</em>. A good projection of poorly-structured data still looks poor, that's a data problem, diagnosed in the Data Quality tab.`,
  explorer: `<strong>Point Explorer</strong>, drill into the individual points a projection gets wrong, decomposed into false-closeness (intrusion) and lost-neighbour (extrusion) for each method.`,
};
function activateTab(name){ /* tabs removed: single flat view, nothing to switch */ }
function bindTabs(){ /* tabs removed */ }

function bindConsole(){
  $('consoleBar').addEventListener('click', ()=>{
    const c=$('console');
    const open=c.classList.toggle('open');
    $('consoleBody').hidden = !open;
    if(open) $('consoleBody').scrollTop = $('consoleBody').scrollHeight;
  });
}
function openConsole(){
  const c=$('console');
  if(!c.classList.contains('open')){ c.classList.add('open'); $('consoleBody').hidden=false; }
}
function setConsole(text){
  $('consoleBody').textContent = text;
  const lines=(text.match(/\n/g)||[]).length+1;
  $('consoleMeta').textContent = `${lines} lines`;
  if($('console').classList.contains('open')) $('consoleBody').scrollTop=$('consoleBody').scrollHeight;
}
function setConsoleDot(kind){
  const dot=$('consoleDot');
  dot.className='console-dot'+(kind?' '+kind:'');
}

let toastTimer=null;
function toast(msg, isErr=false){
  const t=$('toast');
  t.textContent=msg;
  t.className='toast show'+(isErr?' err':'');
  t.hidden=false;
  clearTimeout(toastTimer);
  toastTimer=setTimeout(()=>{ t.classList.remove('show'); }, 3200);
}

/* expose for debugging / testing */
window.state = state;
window.activateTab = activateTab;
window.applyEsSelection = applyEsSelection;
window.clearEsSelection = clearEsSelection;
window.renderErrorSpace = renderErrorSpace;
window.runAnalysis = runAnalysis;

init();