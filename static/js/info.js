/* ======================================================================
   METRIC INFO
   Explanation text for every metric, the info modal, and the delegated
   click handler that turns any [data-info] element into a trigger.
   ====================================================================== */

/* ============================================================
   METRIC INFO SYSTEM
   A dictionary of thorough explanations, an info modal, and a helper that turns
   any section title into a clickable info trigger (title + an (i) button).
   ============================================================ */
const METRIC_INFO = {
  neighborf1: { title: 'Neighbourhood-preservation F1',
    body: `<p><strong>What it is.</strong> A per-point score of how well the projection preserved that point's neighbourhood. Each point is treated as a <em>classifier of its own neighbourhood</em>: its <em>k</em> nearest neighbours in the high-dimensional space are the ground-truth set (the neighbours that <em>should</em> be kept), and its <em>k</em> nearest neighbours in the 2-D projection are the prediction (the neighbours the projection actually placed nearby).</p>
    <p><strong>Precision</strong> = |2-D neighbours &cap; high-D neighbours| / |2-D neighbours|, of the neighbours the projection asserts, how many are real. (1 &minus; precision is the false-positive / intrusion rate.)<br>
    <strong>Recall</strong> = |2-D &cap; high-D| / |high-D neighbours|, of the real neighbours, how many the projection kept. (1 &minus; recall is the false-negative / extrusion rate.)<br>
    <strong>F1</strong> = 2&middot;precision&middot;recall / (precision + recall), in [0, 1].</p>
    <p><strong>What it's useful for.</strong> It answers directly &ldquo;which points can I not trust?&rdquo;, a low-F1 point has a wrong neighbourhood in the picture, so any structure you read around it is unreliable. Because it is per-point, you can colour the projections by it and see exactly <em>where</em> the projection fails.</p>
    <p><strong>How it works.</strong> With a symmetric fixed <em>k</em>, both neighbour sets have size <em>k</em>, so precision = recall = F1 for every point; the three diverge only under adaptive per-point <em>k</em> (then a point's high-D and 2-D neighbourhoods can differ in size). This is a set-overlap metric, distinct from the signed intrusion/extrusion error elsewhere in the tool, which is rank-weighted and directional.</p>` },
  trustmetrics: { title: 'The four per-point trust metrics',
    body: `<p>All four are per-point, in [0, 1], higher = more trustworthy. They measure the same idea, did the projection preserve this point's neighbourhood?, in different ways, so a point can score well on one and badly on another.</p>
    <p><strong>Neighbour F1.</strong> Treats each point as a classifier of its neighbourhood: its high-D k nearest neighbours are the truth, its 2-D k nearest neighbours the prediction. F1 = harmonic mean of precision (fraction of 2-D neighbours that are real; 1&minus;precision = intrusion rate) and recall (fraction of real neighbours kept; 1&minus;recall = extrusion rate). This is the standard Jaccard/neighbourhood-hit set-overlap measure. Uses a hard top-k cut-off.</p>
    <p><strong>JS fidelity.</strong> 1 &minus; Jensen-Shannon divergence between soft neighbour distributions (Gaussian over distances, t-SNE style) in high-D vs 2-D. Continuous and distance-weighted, so it is sensitive to whether the <em>relative</em> closeness of neighbours is preserved, not just top-k membership.</p>
    <p><strong>Distance correlation.</strong> The Spearman rank correlation between the point's distances to all other points in high-D vs 2-D, rescaled to [0, 1]. A global-per-point measure: high = the projection preserves this point's whole distance ordering, not just its nearest neighbours.</p>
    <p><strong>MRRE (Mean Relative Rank Error), as a fidelity.</strong> Penalises how far each of the point's high-D neighbours moved in the 2-D rank ordering, weighted so closer neighbours matter more. Reported as 1 &minus; normalised MRRE. A smoother, rank-based cousin of trustworthiness/continuity.</p>
    <p>Because they emphasise different things (set membership vs. relative distance vs. whole ordering), comparing them, as in the F1-vs-JS scatter, reveals points that are ambiguous, i.e. trustworthy by one criterion but not another.</p>` },
  jsdivergence: { title: 'Jensen-Shannon fidelity (alternative to F1)',
    body: `<p><strong>What it is.</strong> An alternative per-point fidelity based on <em>soft</em> neighbourhoods rather than a hard k cut-off. Following the t-SNE idea, each point's neighbours are given a probability from a Gaussian over distances, in high-D (distribution <em>P</em>) and in 2-D (distribution <em>Q</em>). The <strong>Jensen-Shannon divergence</strong> JS(P&#8214;Q) &isin; [0, 1] (log base 2) measures how differently the projection distributes that point's neighbour-affinity: 0 = identical, 1 = maximally different. We report <strong>JS fidelity = 1 &minus; JS</strong> so that, like F1, higher is better.</p>
    <p><strong>Why a second metric.</strong> F1 uses a binary threshold (in or out of the top <em>k</em>); JS is continuous and weights near neighbours more than far ones, so it is sensitive to whether the projection preserved the <em>relative</em> closeness of neighbours, not just the membership of the top-<em>k</em> set. A point can score high on one and low on the other, the &ldquo;F1 vs JS&rdquo; chart surfaces exactly those disagreements.</p>
    <p><strong>How it works.</strong> JS = &frac12;KL(P&#8214;M) + &frac12;KL(Q&#8214;M) where M = &frac12;(P+Q); the per-point Gaussian bandwidth is the distance to the <em>k</em>-th neighbour, so softness scales with local density. It is symmetric and bounded, unlike raw KL divergence.</p>` },
  boxplot: { title: 'Per-point score box plots',
    body: `<p><strong>What it is.</strong> Two box plots, one for PCA, one for MDS, of the distribution of the per-point score (F1 or JS fidelity) across all points. Each box shows the <strong>median</strong> (centre line), the <strong>interquartile range</strong> (25th&ndash;75th percentile, the box), the <strong>whiskers</strong> (1.5&times;IQR), and <strong>outliers</strong> as individual dots; the <strong>mean</strong> is marked separately. Exact values appear beneath and on hover.</p>
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
    <p><strong>Locality weighting (default).</strong> Weights each feature by its <em>Laplacian Score</em> (He, Cai and; Niyogi 2005): build a k-NN graph, then score a feature by how little it varies between neighbouring points relative to its total variance. Features that respect the local manifold get more weight; features that vary randomly between neighbours (noise) get less. You can then hand-tune each feature's multiplier.</p>
    <p><strong>Mahalanobis.</strong> Whitens the space by the inverse covariance, d = √((a−b)ᵀΣ⁻¹(a−b)). It is the classical adjusted distance, but whitening <em>amplifies</em> low-variance directions, which are frequently noise, so on the datasets here it tends to make neighbourhood-F1 <em>worse</em>. The panel measures this rather than assuming it, which is why locality weighting is the default: it succeeds where Mahalanobis fails.</p>
    <p><strong>How effectiveness is verified.</strong> The panel re-embeds MDS with the adjusted distance and reports the neighbourhood-F1 and the false-positive (intrusion) / false-negative (extrusion) rates of the plain vs adjusted layout, so you can see directly whether the adjusted distance reduces distortion.</p>` },
  classaware: { title: 'Class-aware trust analysis',
    body: `<p>By default the trust metrics are <strong>geometric</strong>: a point's neighbours are defined purely by distance, with no assumption that the class labels are correct. Turning on <strong>class-aware</strong> restricts the error to <em>cross-class</em> intrusions (a neighbour of a different class pulled close) and <em>same-class</em> extrusions (a same-class neighbour pushed away), so it measures whether the projection preserved the labelled <em>class</em> structure rather than raw geometry.</p>
    <p>The two can disagree sharply, a point can be geometrically distorted but class-faithful (its neighbourhood was reshuffled only within its own class), or vice versa.</p>
    <p><strong>Why k-means appears.</strong> Class-aware scores only make sense if the labels actually correspond to the data's clusters. So when class-aware is on, the k-means ground-truth check is shown: it clusters the data with no labels and reports whether those clusters match the classes (ARI, matched accuracy) and whether each projection preserves the clustering. A low agreement means a poor class-aware score reflects a labelling/geometry mismatch, not the projection's fault.</p>` },
  mahalanobis: { title: 'Mahalanobis metric learning',
    body: `<p><strong>What it is.</strong> An MDS layout built from the <em>Mahalanobis</em> distance instead of the plain Euclidean distance. The Mahalanobis distance is d(a,b) = &radic;((a&minus;b)&#7488; &Sigma;&#8315;&sup1; (a&minus;b)), where &Sigma; is the data covariance, equivalent to whitening the space (rotating and rescaling by the inverse covariance) and then measuring ordinary distance.</p>
    <p><strong>What it's useful for.</strong> Plain Euclidean distance lets high-variance and correlated directions dominate, so the layout can be driven by a few dominant or redundant features. Mahalanobis puts every direction on equal footing, which is the classical &ldquo;adjusted distance&rdquo; of unsupervised metric learning. Crucially it is <strong>label-free</strong>, it uses no class labels, only the data's own covariance, so any improvement is a property of the geometry, not of the labels.</p>
    <p><strong>How it works and; how we verify it.</strong> Only the MDS panel uses the adjusted distance; PCA and the per-point trust scores stay on the true (unwhitened) Euclidean geometry, so they remain an honest reference. The comparison panel puts the plain and Mahalanobis layouts side by side and, the important part, reports the neighbourhood-F1 and the false-positive (intrusion) and false-negative (extrusion) rates of each. That lets you check whether the adjusted distance genuinely reduces distortion. It does not always: whitening amplifies low-variance directions, which on some data are noise, so the tool measures the effect rather than assuming it.</p>` },
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
