/* Projection Fidelity Console: frontend logic */

const $ = (id) => document.getElementById(id);
const state = {
  result: null,        // latest run result
  metric: 'intrusion',
  weight: 'rank',

  esView: 'error',         // 'error' | 'class' -> Error Space colour-by mode
  esSel: new Set(),    // Error Space: shared selection across all 3 plots
  esActiveQuad: null,  // Error Space: currently active quadrant button (or null)
  tableSort: { key: 'pca_error', dir: -1 },
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
    font:{ family:'Archivo, sans-serif', color:'#e8edf3' },
    margin:{ l:48, r:16, t:34, b:42 },
    hoverlabel:{ bgcolor:'#1b222c', bordercolor:'#33404f',
                 font:{ family:'Space Mono, monospace', size:11, color:'#e8edf3' } },
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
  await restoreLast();
}

/* ============================================================
   METRIC INFO SYSTEM
   A dictionary of thorough explanations, an info modal, and a helper that turns
   any section title into a clickable info trigger (title + an (i) button).
   ============================================================ */
const METRIC_INFO = {
  errorspace: { title: 'Error Space',
    body: `<p><strong>What it is.</strong> Three linked scatter plots: the PCA projection and the MDS projection on top, and below them the <em>error space</em> — a scatter of each point's PCA error (x) against its MDS error (y).</p>
    <p><strong>What it's useful for.</strong> It shows, per point, <em>where</em> each projection is wrong, <em>in which direction</em>, and <em>how much</em> — and whether the two methods agree. This is a <em>projection</em> diagnostic: it measures the picture, not the data.</p>
    <p><strong>How it works.</strong> Each point carries two channels: <em>colour</em> is the severity-weighted signed error on a false-negative → true → false-positive scale (red = the projection lost real neighbours, blue = it invented false ones); <em>size</em> is the total distortion |FP| + |FN|. A big pale point is badly distorted in both directions at once (they cancel to ~0 net); a small pale point is genuinely faithful — colour alone can't tell them apart, so always read size too. Position in the error space encodes the sign combination: the four quadrants are both-FP, both-FN, and the two mixed cases where the methods disagree.</p>` },
  fidelity: { title: 'Intrusion / Extrusion Fidelity',
    body: `<p><strong>What it is.</strong> A per-point measure of how faithfully a 2-D projection preserves each point's high-dimensional neighbourhood, on a signed scale in [−1, +1].</p>
    <p><strong>What it's useful for.</strong> It localises and signs projection error. A <em>false positive</em> (intrusion, +) means the projection pulled a stranger close; a <em>false negative</em> (extrusion, −) means it pushed a true neighbour away.</p>
    <p><strong>How it works.</strong> For neighbourhood size k, each point's neighbours are ranked by distance in both the high-D space and the 2-D projection. An intruder is inside the 2-D k-neighbourhood but not the high-D one; an extruded point is the reverse. Each offender is weighted by how many rank positions it is displaced, normalised to [0,1]. The signed error is FP − FN. By default the metric is <em>geometric</em> (label-free); turning on class-aware restricts it to cross-class intrusions and same-class extrusions.</p>` },
  verdict: { title: 'PCA vs MDS Verdict',
    body: `<p><strong>What it is.</strong> A single mechanical judgement of which projection is more faithful on this dataset, and how confidently.</p>
    <p><strong>What it's useful for.</strong> It answers "which projection should I trust here?" with a confidence statement — and declines to decide when the evidence is thin, rather than over-claiming.</p>
    <p><strong>How it works.</strong> It combines four independent criteria, each scored 0–1: <em>fidelity</em> (weight 0.40, the per-point distortion), <em>trustworthiness</em> and <em>continuity</em> (0.20 each, classical rank-preservation), and <em>cluster preservation</em> (0.20, how well each 2-D k-means partition matches the high-D one). The weights are stated, not tuned. Because both methods are scored on the same points, a paired bootstrap 95% confidence interval and Cohen's d are computed on the per-point distortion gap; the winner is "decisive" only if that interval excludes zero and the margin clears a threshold.</p>` },
  confidence: { title: 'Confidence Interval (bootstrap)',
    body: `<p><strong>What it is.</strong> A plot of the mean per-point distortion gap between the two methods (loser − winner) with its 95% bootstrap confidence interval.</p>
    <p><strong>What it's useful for.</strong> It shows whether the verdict's margin is <em>statistically real</em> or could be noise. A margin alone can't tell you that.</p>
    <p><strong>How it works.</strong> The per-point distortion differences are resampled with replacement 2000 times; the 2.5th–97.5th percentiles of the resampled mean form the interval. If the whole interval sits on one side of zero, the win is reliable (green); if it crosses zero, the difference is provisional (amber).</p>` },
  components: { title: 'Verdict Component Scores',
    body: `<p><strong>What it is.</strong> The four criteria behind the verdict — fidelity, trust, continuity, cluster — shown for both PCA and MDS.</p>
    <p><strong>What it's useful for.</strong> It reveals <em>why</em> a method won, and when the criteria disagree (one method wins some, the other wins others) — which is exactly when the verdict is correctly not decisive.</p>
    <p><strong>How it works.</strong> Each component is normalised to [0,1] (higher is better). The composite is their weighted sum (0.40 / 0.20 / 0.20 / 0.20). The bar chart plots all four for each method side by side.</p>` },
  histograms: { title: 'Error Distribution (histograms)',
    body: `<p><strong>What it is.</strong> The distribution of signed per-point error for each method, as a histogram tinted by the error scale.</p>
    <p><strong>What it's useful for.</strong> It reveals the <em>shape</em> of a method's errors that the mean hides — a narrow peak at zero (faithful) vs a broad spread (heavily distorted), and whether errors lean red (extrusion) or blue (intrusion).</p>
    <p><strong>How it works.</strong> Errors are binned over a fixed range; a median line and ±2σ lines (over occupied bins) frame how uneven the distribution is. Because it uses the signed error, balanced-severe points sit near zero here too — read it alongside the Error Space size channel.</p>` },
  centroids: { title: 'Centroid Trajectories',
    body: `<p><strong>What it is.</strong> Each class centroid traced across three stages — high-D → PCA → MDS — on a shared normalised frame.</p>
    <p><strong>What it's useful for.</strong> It separates <em>coarse</em> distortion (how the overall class configuration moves) from the <em>local</em> distortion the per-point metric captures. It shows which method pulls classes apart or compresses them.</p>
    <p><strong>How it works.</strong> High-D centroids are projected via their own PCA so all three stages share a plane; each stage is centred and scaled to unit RMS radius, so the trajectory shows shape change, not absolute size. The mean between-centroid distance per stage indicates whether class separation is preserved.</p>` },
  weightedmds: { title: 'Weighted MDS',
    body: `<p><strong>What it is.</strong> An MDS layout in which each feature is scaled by its mutual information with the class label before distances are computed.</p>
    <p><strong>What it's useful for.</strong> Ordinary MDS lets noisy or redundant features dominate the geometry as much as informative ones. Weighting is a controlled experiment: "what would the layout look like if the class-informative measurements counted more?" It is most useful on data with many irrelevant features.</p>
    <p><strong>How it works.</strong> Each feature j is multiplied by √(weight_j), then MDS is built from the resulting weighted-Euclidean distances. Crucially, <em>only the MDS panel changes</em> — PCA and the high-D fidelity reference stay unweighted, so the error colours still measure true-geometry distortion. The comparison shows the plain and weighted layouts side by side with the change in class silhouette.</p>` },
  kmeans: { title: 'k-means Ground-Truth Check',
    body: `<p><strong>What it is.</strong> An unsupervised clustering of the data, compared against the class labels. This is a <em>data</em> diagnostic, not a projection one.</p>
    <p><strong>What it's useful for.</strong> Every label-based score assumes the labels <em>are</em> the cluster structure. This check asks whether that's true — so an apparent "projection error" isn't really a mismatch between the labels and the data's natural groups.</p>
    <p><strong>How it works.</strong> The chosen clusterer (k-means, or GMM / spectral for non-convex data) is fit in the high-D space and both projections. ARI vs labels answers "are the classes clusters?"; a Hungarian-matched accuracy makes that interpretable; ARI 2-D-vs-high-D answers "does each projection preserve the partition?". A low ARI can mean bad labels <em>or</em> (for k-means) non-convex clusters — cross-check with precision.</p>` },
  confusion: { title: 'Cluster vs Class Confusion Heatmap',
    body: `<p><strong>What it is.</strong> A contingency table: each cell counts the points in a high-D cluster (row) that carry a given class label (column).</p>
    <p><strong>What it's useful for.</strong> It shows exactly how clusters and labels line up. A clean result is block-diagonal — each cluster dominated by one class.</p>
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
    <p><strong>What it's useful for.</strong> It shows each class's strengths and weaknesses individually — which classes are hard, and <em>how</em> they're hard.</p>
    <p><strong>How it works.</strong> A balanced triangle means the class is precise <em>and</em> cohesive. A spike toward one axis shows the weakness: high precision but low recall = the class is pure but fragmented; the reverse = cohesive but impure.</p>` },
  annealing: { title: 'Simulated Annealing (neighbourhood search)',
    body: `<p><strong>What it is.</strong> A search for the neighbourhood size k that maximises the cluster-structure F1, since F1(k) is bumpy and has several local optima.</p>
    <p><strong>What it's useful for.</strong> It makes an otherwise-arbitrary parameter principled and data-driven, cheaply (touching far fewer values than an exhaustive scan).</p>
    <p><strong>How it works.</strong> From the current k it proposes a nearby value and accepts it if better, or — with a probability that falls as the "temperature" cools — even if worse, which lets it escape local optima. The best-F1 line only ever rises and should plateau; proposed-k should bounce early then settle; accepted worsening moves cluster early and thin out. A temperature panel shows the cooling schedule, and a health check flags a search that's stuck (best flatlines), under-cooled (still rising at the end), or too greedy (no worsening moves).</p>` },
  untrustworthy: { title: 'Untrustworthy Points',
    body: `<p><strong>What it is.</strong> The points a projection asserts a <em>false closeness</em> for — intrusions where the 2-D layout draws together points that are actually far apart in high-D.</p>
    <p><strong>What it's useful for.</strong> These are the genuinely misleading points: a cluster you see among them may be an artifact. (Lost neighbours — extrusions — are a loss of information but not a false claim, and are listed separately.)</p>
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
  echo('normScale','normScaleEcho',(v)=>parseFloat(v).toFixed(1));
  echo('threshold','threshEcho',(v)=>parseFloat(v).toFixed(2));
  if($('recallAlpha')) echo('recallAlpha','recallAlphaEcho',(v)=>parseFloat(v).toFixed(2));

  // metric segmented control
  // weight segmented control
  $('weightSeg').addEventListener('click', (e)=>{
    const b = e.target.closest('.seg-btn'); if(!b) return;
    state.weight = b.dataset.weight;
    document.querySelectorAll('#weightSeg .seg-btn').forEach(x=>x.classList.toggle('active', x===b));
  });

  // normalize → show steepness only for tanh/sigmoid
  $('normalize').addEventListener('change', (e)=>{
    const v = e.target.value;
    $('normScaleField').hidden = !(v==='tanh'||v==='sigmoid');
  });

  // Error Space view: colour by error/fidelity, or by class ("see classes")
  $('esViewSeg').addEventListener('click', (e)=>{
    const b = e.target.closest('.seg-btn'); if(!b) return;
    state.esView = b.dataset.view;
    document.querySelectorAll('#esViewSeg .seg-btn').forEach(x=>
      x.classList.toggle('active', x===b));
    const cls = state.esView==='class';
    // dim the error-colour legend when showing classes; show a class hint
    $('encLegend').style.opacity = cls ? 0.4 : 1;
    $('esViewHint').textContent = cls
      ? 'each point tinted by its true class; compare against the error view to see which classes the projection distorts'
      : 'points tinted by their severity-weighted error; drag to select, quadrants below split by error sign';
    if(state.result) renderErrorSpace(state.result);
  });

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
    weight: state.weight,
    adaptive: $('adaptive') ? $('adaptive').checked : false,
    normalize: $('normalize').value,
    norm_scale: parseFloat($('normScale').value),
    no_standardize: $('noStandardize').checked,
    threshold: parseFloat($('threshold').value),
    weighted_mds: $('weightedMds').checked,
    weight_method: 'mutual_info',
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
    toast(`Analysis complete · ${d.info.name} · ${d.info.elapsed}s`);
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
  ['centPlot', 'histPca', 'histMds',
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
  state.weight = p.weight || 'rank';
  document.querySelectorAll('#weightSeg .seg-btn').forEach(x=>x.classList.toggle('active', x.dataset.weight===state.weight));
  const setV=(id,v)=>{ if(v!==undefined&&v!==null) $(id).value=v; };
  setV('neighbors',p.neighbors); $('kEcho').textContent=p.neighbors??10;
  setV('normalize',p.normalize);
  setV('normScale',p.norm_scale); $('normScaleEcho').textContent=(p.norm_scale??3).toFixed?.(1)??p.norm_scale;
  $('normScaleField').hidden = !(p.normalize==='tanh'||p.normalize==='sigmoid');
  setV('threshold',p.threshold); $('threshEcho').textContent=(p.threshold??0.5).toFixed?.(2)??p.threshold;
  if($('classAware')) $('classAware').checked = !!p.class_aware;
  $('adaptive').checked = !!p.adaptive;
  $('noStandardize').checked = !!p.no_standardize;
  if(p.max_samples) $('maxSamples').value = p.max_samples;
  $('weightedMds').checked = !!p.weighted_mds;
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
  safeRender('adaptivek', ()=>renderAdaptiveK(d));
  safeRender('signal', ()=>renderSignal(d));
  safeRender('table', ()=>renderTable(d));
  safeRender('verdict-banner', ()=>renderVerdictBanner(d));
  safeRender('labels-banner', ()=>renderLabelsBanner(d));
  safeRender('analysis', ()=>renderAnalysisPanel(d));
  if(typeof Plotly === 'undefined'){
    plotlyUnavailable();
    return;
  }
  // clear any stale selection from the previous run
  state.esSel = new Set();
  state.esActiveQuad = null;
  // the classic metric has no separable FP/FN (its "parts" are just the
  // positive/negative halves of one score), so the FP-vs-FN panel is
  // meaningless there and is hidden entirely
  safeRender('errorspace', ()=>renderErrorSpace(d));
  safeRender('centroids', ()=>renderCentroids(d));
  safeRender('histograms', ()=>renderHistograms(d));
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
  ['centPlot','histPca','histMds',
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
    pill('PCA mean', d.summary.pca_mean.toFixed(3), signCls(s.pca_mean), `${(s.pca_fn_share*100).toFixed(0)}% FN`) +
    pill('MDS mean', d.summary.mds_mean.toFixed(3), signCls(s.mds_mean), `${(s.mds_fn_share*100).toFixed(0)}% FN`) +
    pill('Flagged', s.untrustworthy_count, '', `false closeness > +${d.params.threshold}`) +
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
    readout.textContent = `${fnPercent}% FN · ${fpPercent}% FP · FN-dominated`;
  } else if(fpMass > fnMass * 1.1){ // FP dominates
    label.style.color = '#3f7cc4';
    label.className = 'signal-label fp-dominant';
    readout.textContent = `${fnPercent}% FN · ${fpPercent}% FP · FP-dominated`;
  } else { // balanced
    label.style.color = '#63707f';
    label.className = 'signal-label balanced';
    readout.textContent = `${fnPercent}% FN · ${fpPercent}% FP · Balanced`;
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
  const vmax = d.vmax, bins=30;
  const edges=[]; for(let i=0;i<=bins;i++) edges.push(-vmax + (2*vmax)*i/bins);
  const binW = edges[1]-edges[0];

  function histTrace(err){
    const centres=[], counts=new Array(bins).fill(0), colors=[];
    for(let i=0;i<bins;i++){
      // bar is CENTRED on the bin midpoint -> it straddles [edge_i, edge_i+1]
      centres.push((edges[i]+edges[i+1])/2);
      colors.push(scaleColor(centres[i],vmax));
    }
    err.forEach(v=>{
      const cv=Math.max(-vmax,Math.min(vmax,v));
      let b=Math.floor((cv+vmax)/(2*vmax)*bins); if(b>=bins)b=bins-1; if(b<0)b=0;
      counts[b]++;
    });
    return { centres, counts, colors };
  }
  const median = (arr)=>{
    const s=[...arr].sort((a,b)=>a-b), n=s.length;
    if(!n) return 0;
    return n%2 ? s[(n-1)/2] : (s[n/2-1]+s[n/2])/2;
  };

  // both panels share one y-range so PCA and MDS bar heights are comparable
  const hP = histTrace(d.table.pca.err), hM = histTrace(d.table.mds.err);
  const globalMax = Math.max(...hP.counts, ...hM.counts, 1);

  function build(divId, err, label, h){
    // ---- per-bar hover: exact interval + count + share of all points ----
    const custom = h.counts.map((c,i)=>[
      edges[i].toFixed(3), edges[i+1].toFixed(3), (100*c/err.length).toFixed(1),
    ]);
    const bar={
      type:'bar', x:h.centres, y:h.counts,
      marker:{color:h.colors, line:{color:'rgba(0,0,0,0.45)',width:0.5}},
      width: binW*0.92,
      customdata: custom,
      // count printed on top of every non-empty bar
      text: h.counts.map(c=> c>0 ? String(c) : ''),
      textposition:'outside', textfont:{family:'Space Mono',size:9,color:'#9aa7b6'},
      cliponaxis:false,
      hovertemplate:'error interval [%{customdata[0]}, %{customdata[1]})<br>'+
                    '<b>%{y}</b> points · %{customdata[2]}% of set<extra></extra>',
      name:'count', showlegend:false,
    };

    // ---- horizontal statistics on the bin COUNTS ----
    // Computed over OCCUPIED bins only. The signed-error distribution is
    // typically concentrated in a handful of bins, so >half of them are empty
    // and a median taken over all bins would simply be 0 -- a flat green line
    // on the axis, with the -2 sigma bound below zero. Restricting to non-empty
    // bins answers the useful question ("how tall is a typical populated bin?")
    // and keeps every reference line inside the data.
    const occupied = h.counts.filter(c=>c>0);
    const medCount = median(occupied);
    const meanCount = occupied.reduce((a,b)=>a+b,0)/(occupied.length||1);
    const sdCount = Math.sqrt(
      occupied.reduce((a,b)=>a+(b-meanCount)**2,0)/(occupied.length||1));
    const loBand = medCount-2*sdCount, hiBand = medCount+2*sdCount;
    const xSpan=[-vmax,vmax];

    // shaded +/-2 sigma band (drawn first so the lines sit on top of it)
    const band={ type:'scatter',mode:'lines',
      x:[xSpan[0],xSpan[1],xSpan[1],xSpan[0]],
      y:[Math.max(0,loBand),Math.max(0,loBand),hiBand,hiBand],
      fill:'toself', fillcolor:'rgba(232,237,243,0.07)',
      line:{width:0}, hoverinfo:'skip', showlegend:false };
    // the two sigma BOUNDARY lines, both explicitly drawn so they are visible
    const sigHi={ type:'scatter',mode:'lines',x:xSpan,y:[hiBand,hiBand],
      line:{color:'#8f9dad',width:1.2,dash:'dash'},
      name:'median ± 2σ (occupied bins)', showlegend:true,
      hovertemplate:`+2σ = ${hiBand.toFixed(1)} counts<extra></extra>` };
    const sigLo={ type:'scatter',mode:'lines',x:xSpan,y:[loBand,loBand],
      line:{color:'#8f9dad',width:1.2,dash:'dash'},
      showlegend:false,
      hovertemplate:`−2σ = ${loBand.toFixed(1)} counts<extra></extra>` };
    // GREEN median-count line
    const medLine={ type:'scatter',mode:'lines',x:xSpan,y:[medCount,medCount],
      line:{color:'#4ec97a',width:1.8},
      name:`median count = ${medCount} (${occupied.length}/${bins} bins used)`,
      showlegend:true,
      hovertemplate:`median count over occupied bins = ${medCount}<extra></extra>` };

    // ---- vertical reference: zero, and the median ERROR of the points ----
    const headroom = Math.max(globalMax, hiBand)*1.16;
    const zeroLine={ type:'scatter',mode:'lines',x:[0,0],y:[0,headroom],
      line:{color:'#63707f',width:1,dash:'dash'},hoverinfo:'skip',showlegend:false };
    const medErr = median(err);
    const medErrLine={ type:'scatter',mode:'lines',x:[medErr,medErr],y:[0,headroom],
      line:{color:'#4ec97a',width:1.2,dash:'dot'},
      name:`median error = ${medErr>=0?'+':''}${medErr.toFixed(3)}`, showlegend:true,
      hovertemplate:`median error ${medErr.toFixed(3)}<extra></extra>` };

    // ---- distribution stats for the subtitle ----
    const meanErr = err.reduce((a,b)=>a+b,0)/err.length;
    const sdErr = Math.sqrt(err.reduce((a,b)=>a+(b-meanErr)**2,0)/err.length);
    const fnShare = 100*err.filter(v=>v<0).length/err.length;
    const skew = meanErr - medErr;   // >0 => a long FP tail pulls the mean right

    Plotly.react(divId,[band,sigLo,sigHi,medLine,zeroLine,bar,medErrLine], baseLayout({
      title:{ text:`${label}  ·  n=${err.length}  ·  mean ${meanErr>=0?'+':''}${meanErr.toFixed(3)}`+
                   `  ·  σ ${sdErr.toFixed(3)}  ·  ${fnShare.toFixed(0)}% FN`+
                   `  ·  tail ${skew>=0?'→FP':'→FN'}`,
              font:{size:11,color:'#9aa7b6'},x:0.02,xanchor:'left' },
      bargap:0.04,
      showlegend:true,
      legend:{ font:{size:9,color:'#9aa7b6'}, bgcolor:'rgba(21,26,33,0.78)',
               bordercolor:'#262f3b', borderwidth:1, x:0.99, xanchor:'right', y:0.99 },
      xaxis:Object.assign({title:'signed error (FN < 0 < FP)',range:[-vmax,vmax]},AXIS),
      yaxis:Object.assign({title:'count',rangemode:'tozero',range:[0,headroom]},AXIS),
    }),PLOT_CONFIG);
  }
  build('histPca', d.table.pca.err, 'PCA', hP);
  build('histMds', d.table.mds.err, 'MDS', hM);
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
      hovertemplate:`${clsName}<br>%{x}<br>axis1 %{customdata[0]:.3f} · axis2 %{customdata[1]:.3f}<extra></extra>`,
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
  { key:'pca_error', label:'PCA err', err:true },
  { key:'mds_error', label:'MDS err', err:true },
  { key:'pca_FP', label:'PCA FP', part:'fp' },
  { key:'pca_FN', label:'PCA FN', part:'fn' },
  { key:'mds_FP', label:'MDS FP', part:'fp' },
  { key:'mds_FN', label:'MDS FN', part:'fn' },
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
      ? 'idx %{customdata[1]} · %{customdata[0]}<br>' +
        'PCA err %{customdata[2]:.3f} · MDS err %{customdata[3]:.3f}<br>' +
        'total distortion %{customdata[6]:.3f}<extra></extra>'
      : 'idx %{customdata[1]} · %{customdata[0]}<br>' +
        'error %{customdata[2]:.3f} &nbsp; (FP %{customdata[4]:.3f} · FN %{customdata[5]:.3f})<br>' +
        'severity %{customdata[7]:+.3f} · total distortion %{customdata[6]:.3f}<extra></extra>',
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
function renderWmdsCompare(d){
  const box = $('wmdsCompare');
  const mc = d.mds_compare;
  if(!mc){ if(box) box.hidden = true; if(typeof Plotly!=='undefined'){ ['wmdsPlain','wmdsWeighted'].forEach(id=>{ const el=$(id); if(el&&el.data) Plotly.purge(el); }); } return; }
  box.hidden = false;
  const dlt = mc.delta;
  const deltaEl = $('wmdsDelta');
  const better = dlt > 0.005, worse = dlt < -0.005;
  deltaEl.textContent = `silhouette ${mc.silhouette_plain.toFixed(3)} → ${mc.silhouette_weighted.toFixed(3)}  (${dlt>=0?'+':''}${dlt.toFixed(3)})`;
  deltaEl.className = 'wmds-delta ' + (better?'wmds-up':(worse?'wmds-down':'wmds-flat'));
  $('wmdsNote').innerHTML = better
    ? `Weighting the label-informative features <strong>improved</strong> class separation. Both panels are coloured by class; PCA and the fidelity reference stay unweighted, so the error colours still measure true-geometry distortion.`
    : (worse
      ? `Weighting <strong>reduced</strong> class separation here &mdash; the emphasised features may not align with the true neighbourhood structure. Both panels coloured by class.`
      : `Weighting left class separation roughly unchanged. Both panels coloured by class.`);

  if(typeof Plotly==='undefined') return;
  const classColorArr = (d.table.labels||[]).map(l=>classColor(l));
  const mk = (xs,ys)=>({ type:'scattergl', mode:'markers', x:xs, y:ys,
    marker:{ size:7, color:classColorArr, line:{color:'rgba(0,0,0,0.5)',width:0.4} },
    customdata:d.table.ids.map((id,i)=>[d.table.class_names[i],id]),
    hovertemplate:'idx %{customdata[1]} · %{customdata[0]}<extra></extra>' });
  const lay = (title)=> baseLayout({
    title:{text:title,font:{size:12,color:'#e8edf3'},x:0.03,xanchor:'left'},
    xaxis:Object.assign({title:'dim 1'},AXIS), yaxis:Object.assign({title:'dim 2'},AXIS),
    showlegend:false, height:300,
  });
  Plotly.react('wmdsPlain', [mk(mc.plain.x, mc.plain.y)], lay('Plain MDS'), PLOT_CONFIG);
  Plotly.react('wmdsWeighted', [mk(d.table.mds.x, d.table.mds.y)], lay('Weighted MDS'), PLOT_CONFIG);

  // feature-weight bar chart: which features were up-weighted (drove the layout)
  const wi = d.weight_info;
  const wDiv = $('wmdsWeights');
  if(wi && wi.weights && wDiv){
    const w = wi.weights;
    const idx = w.map((v,i)=>i);
    // colour bars by magnitude (above-average weight = accent)
    const mean = w.reduce((a,b)=>a+b,0)/w.length;
    const colors = w.map(v=> v>=mean ? '#4ec97a' : '#46566a');
    const labels = idx.map(i=>`f${i}`);
    Plotly.react(wDiv, [{
      type:'bar', x:labels, y:w, marker:{color:colors},
      hovertemplate:'feature %{x}: weight %{y:.3f}<extra></extra>',
    }], baseLayout({
      xaxis:Object.assign({title: w.length>25?'':'feature', showticklabels: w.length<=25},AXIS),
      yaxis:Object.assign({title:'weight'},AXIS),
      showlegend:false, height:180, margin:{l:44,r:12,t:8,b:w.length>25?18:34},
      shapes:[{type:'line', x0:-0.5, x1:w.length-0.5, y0:mean, y1:mean,
        line:{color:'#7f9ab5',width:1,dash:'dot'}}],
    }), PLOT_CONFIG);
  } else if(wDiv && typeof Plotly!=='undefined' && wDiv.data){
    Plotly.purge(wDiv);
  }
}

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
      `min <strong>${akd.k_min}</strong> · median <strong>${akd.k_median}</strong> · max <strong>${akd.k_max}</strong>`;
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
  if(idx.length && state.esActiveQuad){ tag.textContent='quadrant · '+quadName[state.esActiveQuad]; tag.hidden=false; }
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
    sub.textContent = `composite ${fmt3(p.sWin)} vs ${fmt3(p.sLose)} — no meaningful difference on this dataset`;
  } else if(p.decisive){
    banner.classList.add('vb-decisive');
    badge.textContent = p.winner;
    headline.innerHTML = `<strong>${p.winner}</strong> is the more faithful projection`;
    sub.textContent = `decisive — composite ${fmt3(p.sWin)} vs ${fmt3(p.sLose)} (+${fmt3(p.margin)}, ${p.relpct.toFixed(1)}% relative), 95% CI excludes zero`;
  } else {
    banner.classList.add('vb-soft');
    badge.textContent = p.winner;
    headline.innerHTML = `<strong>${p.winner}</strong> leads, but not decisively`;
    sub.textContent = `composite ${fmt3(p.sWin)} vs ${fmt3(p.sLose)} (+${fmt3(p.margin)}) — the confidence interval straddles zero, so treat as provisional`;
  }
}

const fmt3 = (x)=> (x==null||isNaN(x)) ? '—' : (x>=0? x.toFixed(3) : x.toFixed(3));
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

  const ariStr = ari!=null ? ari.toFixed(2) : '—';
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
      if(val==null){ html += '<td>—</td>'; return; }
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
  if(w){
    html += '<div class="an-weights">weights: ' +
      comps.filter(c=>w[c]!=null).map(c=>`${c} ${w[c]}`).join(' · ') + '</div>';
  }
  return html;
}

function renderAnalysisPanel(d){
  renderVerdictCard(d);
  renderAnnealCard(d);
  renderMetricsCard(d);
  renderKmeansCard(d);
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
  // SIMPLIFIED: headline (above) + component bar chart + confidence-interval plot
  // are always visible; the detailed table, weight sliders and long-form
  // confidence/MDS notes live in a collapsible <details> section below.
  comp.innerHTML =
    infoHeader('components', 'Component scores') +
    '<div class="plot" id="verdictBars" style="height:200px"></div>' +
    infoHeader('confidence', 'Is the difference statistically real?') +
    '<p class="an-note" style="margin-bottom:6px">The dot is the mean per-point distortion gap between the two methods (loser &minus; winner); the bar is the 95% bootstrap confidence interval. <strong>If the whole bar sits to one side of the dashed zero line, the win is reliable</strong> (green); if it crosses zero, the difference is provisional (amber).</p>' +
    '<div class="plot" id="verdictCI" style="height:130px"></div>' +
    '<details class="verdict-details"><summary>Show detailed breakdown &amp; weight controls</summary>' +
      '<div class="verdict-details-body">' +
        '<h4>How the four criteria voted</h4>' + componentBars(v.per_method) +
        '<div class="wtsliders" id="wtSliders">'+
          '<h4>Adjust the weights <span class="an-subtle">recomputes live</span></h4>'+
          wtSliderRow('fidelity', 0.40) + wtSliderRow('trust', 0.20) +
          wtSliderRow('continuity', 0.20) + wtSliderRow('cluster', 0.20) +
          '<div class="wt-live" id="wtLive"></div>'+
        '</div>'+
        '<div class="an-note" id="anVerdictDetailNote"></div>'+
      '</div>' +
    '</details>';
  let noteHtml =
    `<strong>Confidence.</strong> Effect size (Cohen's d) = ${fmt3(p.d)}; ` +
    `95% bootstrap CI on the per-point distortion gap = [${sgn3(p.ci[0])}, ${sgn3(p.ci[1])}]. ` +
    (p.decisive
      ? 'The interval lies entirely on one side of zero, so the win is reliable.'
      : (p.tie ? 'The scores are within the tie threshold.'
               : 'The interval crosses zero, so the lead is not statistically reliable.'));
  // MDS variability: does the verdict survive re-running the stochastic MDS?
  const ms = v.mds_stability;
  if(ms){
    const robust = ms.margin!=null && (ms.fidelity_max - ms.fidelity_min) < Math.abs(ms.margin);
    noteHtml += `<br><br><strong>MDS variability.</strong> Across ${ms.n_layouts} MDS `+
      `re-runs (different seeds) the MDS fidelity component was ${ms.fidelity_mean} `+
      `&plusmn; ${ms.fidelity_std} (range ${ms.fidelity_min}&ndash;${ms.fidelity_max}). ` +
      (robust
        ? `This spread is smaller than the verdict margin, so the result is robust to MDS's randomness.`
        : `This spread is comparable to the verdict margin, so treat a close verdict with caution &mdash; a different MDS layout could shift it.`);
  }
  // the long-form note now lives inside the collapsible details section
  note.innerHTML = '';
  const detailNote = $('anVerdictDetailNote');
  if(detailNote) detailNote.innerHTML = noteHtml;
  renderVerdictCharts(d);
  bindWeightSliders(d);
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
  const recompute = ()=>{
    const w = {};
    box.querySelectorAll('.wt-slider').forEach(s=>{ w[s.dataset.wt] = parseFloat(s.value); });
    box.querySelectorAll('.wt-slider').forEach(s=>{
      const el = $(`wtval-${s.dataset.wt}`); if(el) el.textContent = parseFloat(s.value).toFixed(2);
    });
    const total = Object.values(w).reduce((a,b)=>a+b,0) || 1;
    const pm = v.per_method, methods = Object.keys(pm);
    const scores = {};
    methods.forEach(m=>{
      let s = 0;
      Object.keys(w).forEach(c=>{
        const cv = pm[m].components && pm[m].components[c]!=null ? pm[m].components[c] : 0;
        s += cv * w[c];
      });
      scores[m] = s / total;
    });
    // determine winner and margin under the new weights
    const ranked = methods.slice().sort((a,b)=>scores[b]-scores[a]);
    const win = ranked[0], lose = ranked[ranked.length-1];
    const margin = scores[win] - scores[lose];
    const live = $('wtLive');
    if(live){
      const origWin = v.winner;
      const flipped = win !== origWin;
      live.innerHTML =
        `Under these weights: <strong>${win}</strong> ${methods.length>1?`(${scores[win].toFixed(3)}) vs ${lose} (${scores[lose].toFixed(3)})`:''}, `+
        `margin ${margin>=0?'+':''}${margin.toFixed(3)}. `+
        (flipped ? `<span class="wt-flip">⚠ the winner flipped from ${origWin}</span>`
                 : `<span class="wt-same">same winner as the stated weights</span>`)+
        `<button class="ghost-btn ghost-btn-sm" id="wtReset">Reset to stated weights</button>`;
      const rb = $('wtReset');
      if(rb) rb.addEventListener('click', ()=>{
        const defs = {fidelity:0.40, trust:0.20, continuity:0.20, cluster:0.20};
        box.querySelectorAll('.wt-slider').forEach(s=>{ s.value = defs[s.dataset.wt]; });
        recompute();
      });
    }
    // update the grouped bar chart to reflect... (bars show component scores,
    // which don't change with weights, so we leave them; the live line is the
    // weight-dependent readout)
  };
  box.querySelectorAll('.wt-slider').forEach(s=> s.addEventListener('input', recompute));
  recompute();
}

/* Grouped component bar chart (PCA vs MDS) + confidence-interval plot. */
function renderVerdictCharts(d){
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
      hovertemplate:`gap ${mean.toFixed(4)} · 95% CI [${lo.toFixed(4)}, ${hi.toFixed(4)}]<extra></extra>`,
    }], baseLayout({
      xaxis:Object.assign({title:'mean distortion gap (loser − winner)', range:[xlo,xhi], zeroline:false},AXIS),
      yaxis:Object.assign({showticklabels:false, range:[-1,1]},AXIS),
      height:130, margin:{l:14,r:14,t:22,b:36},
      shapes:[{type:'line', x0:0, x1:0, y0:-1, y1:1, line:{color:'#d24b52',width:1.5,dash:'dash'}}],
      annotations:[
        {x:0, y:-0.85, text:'zero (no difference)', showarrow:false, font:{size:9,color:'#d24b52'}, xanchor: (mean>=0?'left':'right'), xshift:(mean>=0?3:-3)},
        {x:mean, y:0.82, text: decisive ? '✓ CI excludes zero — reliable' : '✗ CI crosses zero — provisional',
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
    stats.innerHTML = '<p class="an-muted">Enable “Search best k” in the Analysis controls and re-run to see the search.</p>';
    note.textContent=''; if(typeof Plotly!=='undefined') Plotly.purge(plotDiv);
    return;
  }
  chip.textContent = `k* = ${a.best_k}`;
  const grid = a.exhaustive_count != null ? a.exhaustive_count
             : (a.k_hi != null && a.k_lo != null ? (a.k_hi - a.k_lo + 1) : '—');
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
    statCell('F1 at k*', (a.best_score!=null? a.best_score.toFixed(4):'—')) +
    statCell('evaluations', `${a.n_evaluations}`) +
    statCell('vs exhaustive', `${a.n_evaluations}/${grid}`) +
    statCell('uphill accepted', a.uphill_moves_accepted!=null?a.uphill_moves_accepted:'—') +
    statCell('worsening accept rate', accRate!=null?`${accRate}%`:'—') +
    statCell('start temp', a.t0!=null?a.t0.toFixed(3):'—') +
    statCell('F1 gain', (startScore!=null&&endBest!=null)?`+${(endBest-startScore).toFixed(3)}`:'—');

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
    if(gained) items.push(`<li class="diag-good">✓ final best F1 (+${(endBest-startScore).toFixed(3)}) exceeds the start — the search improved</li>`);
    if((a.uphill_moves_accepted||0) > 0 && !improvedLate) items.push(`<li class="diag-good">✓ it explored early (worsening moves accepted) then settled — healthy cooling</li>`);
    // problem signs
    if(flatFromStart) items.push(`<li class="diag-bad">✗ best F1 never improved — likely stuck in a local optimum (temperature too low)</li>`);
    if(improvedLate) items.push(`<li class="diag-warn">⚠ best F1 was still rising at the end — may be under-cooled; consider more iterations</li>`);
    if(noUphill && !flatFromStart) items.push(`<li class="diag-warn">⚠ no worsening moves were accepted — the search was greedy and may have missed better regions</li>`);
    if(!items.length) items.push(`<li class="diag-good">✓ search behaved normally</li>`);
    diag = `<div class="diag-box"><div class="diag-title">Search health</div><ul class="diag-list">${items.join('')}</ul></div>`;
  }

  const reheatTxt = a.reheats ? ` It re-heated ${a.reheats} time${a.reheats>1?'s':''} after stalling, to escape local optima.` : '';
  note.innerHTML =
    `<strong>What a working search looks like.</strong> The <span style="color:#4ec97a">green best-F1 line</span> is a cumulative maximum, so it <em>never decreases</em>; it should rise then plateau. The <span style="color:#7f9ab5">grey proposed-k line</span> should bounce widely early (exploration) then settle late (refinement). Red rings (accepted worsening moves) should cluster early and thin out as it cools. The final best should beat the start.${reheatTxt}` +
    diag;

  // THREE stacked panels: best-F1 (top), k-trace (middle), temperature (bottom).
  if(typeof Plotly!=='undefined' && a.trace && a.trace.length){
    const it = a.trace.map(t=>t.iter), kk = a.trace.map(t=>t.k),
          bestF = a.trace.map(t=>t.best_score), temp = a.trace.map(t=>t.T);
    const up = a.trace.filter(t=>t.uphill);
    const bestScoreLine = { type:'scatter', mode:'lines', x:it, y:bestF,
      name:'best F1 so far', xaxis:'x', yaxis:'y', line:{color:'#4ec97a',width:2.5} };
    const proposed = { type:'scatter', mode:'lines+markers', x:it, y:kk,
      name:'proposed k', xaxis:'x', yaxis:'y2',
      line:{color:'#7f9ab5',width:1}, marker:{size:5,color:'#7f9ab5'} };
    const uphill = { type:'scatter', mode:'markers', x:up.map(t=>t.iter), y:up.map(t=>t.k),
      name:'accepted worsening', xaxis:'x', yaxis:'y2',
      marker:{size:11,color:'rgba(0,0,0,0)', line:{color:'#d24b52',width:1.6}} };
    const tempLine = { type:'scatter', mode:'lines', x:it, y:temp,
      name:'temperature', xaxis:'x', yaxis:'y3', line:{color:'#c88a4a',width:1.5} };
    Plotly.react(plotDiv, [bestScoreLine, proposed, uphill, tempLine], baseLayout({
      grid:{rows:3, columns:1, pattern:'independent', roworder:'top to bottom'},
      xaxis:Object.assign({title:'iteration', anchor:'y3'},AXIS),
      yaxis:Object.assign({title:'best F1', domain:[0.71,1.0]},AXIS),
      yaxis2:Object.assign({title:'k tried', domain:[0.38,0.63]},AXIS),
      yaxis3:Object.assign({title:'temp (cooling)', domain:[0.0,0.28]},AXIS),
      showlegend:true, legend:{font:{size:9,color:'#9aa7b6'},orientation:'h',x:0,y:1.1,xanchor:'left',yanchor:'bottom'},
      height:440, margin:{l:52,r:16,t:24,b:38},
    }), PLOT_CONFIG);
  }
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
    bigStat('precision', g.precision, 'are points among their own class?') +
    bigStat('recall', g.recall, 'is each class held together?') +
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
    pc.innerHTML = '<h4>Per class (precision · recall · F1)</h4>'+
      '<p class="an-note" style="margin-bottom:8px">Each class has its own radar. A '+
      'balanced triangle means the class is precise <em>and</em> cohesive; a spike '+
      'toward one axis shows where it is weak (e.g. high precision but low recall = '+
      'pure but fragmented).</p>'+
      '<div class="radar-grid" id="radarGrid"></div>';
    const gridEl = $('radarGrid');
    if(typeof Plotly!=='undefined' && gridEl){
      const axes = ['precision','recall','F1'];
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
  const v = (val==null||isNaN(val)) ? '—' : val.toFixed(3);
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
  chip.textContent = (are ? 'labels ≈ clusters' : 'labels ≠ clusters') + ` · ${methodName}`;
  chip.className = 'an-chip ' + (are ? 'an-chip-good' : 'an-chip-warn');
  const acc = hi.cluster_accuracy!=null ? (100*hi.cluster_accuracy).toFixed(1)+'%' : '—';
  const ariL = hi.ari_vs_labels!=null ? hi.ari_vs_labels.toFixed(3) : '—';
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
      html += ` — silhouette even prefers <strong>${km.best_k_by_silhouette}</strong> clusters against <strong>${km.n_classes}</strong> classes`;
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
      html += statCell(`${m}: ARI 2-D vs high-D`, ari!=null?ari.toFixed(3):'—');
    });
    html += '</div>';
  }
  // containers for the two charts
  html += '<h4>Cluster vs class confusion</h4>'+
          '<p class="an-note" style="margin-bottom:8px">Each cell is the number of points in a high-D cluster (row) that carry a given class label (column). A clean result is <strong>block-diagonal</strong>: each cluster is dominated by one class. <strong>Cluster purity</strong> = the biggest count in a row &divide; that row&rsquo;s total (how single-class each cluster is). <strong>Class coverage</strong> = the biggest count in a column &divide; that column&rsquo;s total (how much of each class landed in one cluster).</p>'+
          '<div class="plot" id="kmHeatmap" style="height:230px"></div>'+
          '<div id="kmPurity"></div>';
  if(km.silhouette_scan && km.silhouette_scan.length){
    html += '<h4>Silhouette by cluster count</h4>'+
            '<p class="an-note" style="margin-bottom:6px">The <strong>silhouette</strong> score (&minus;1 to +1) measures how well-separated the clusters are: each point compares its average distance to its own cluster against the nearest other cluster. Near <strong>+1</strong> = tight, well-separated clusters; near <strong>0</strong> = overlapping clusters; <strong>negative</strong> = points likely in the wrong cluster. The peak marks the number of clusters the geometry most naturally supports &mdash; if that differs from your class count, the labels may split or merge the natural groups.</p>'+
            '<div class="plot" id="kmSilhouette" style="height:200px"></div>';
  }
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
      hovertemplate:'cluster %{y} · %{x}: %{z} points<extra></extra>',
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

  // --- silhouette vs k ---
  const sil = $('kmSilhouette');
  if(sil && km.silhouette_scan && km.silhouette_scan.length){
    const ks = km.silhouette_scan.map(s=>s.k);
    const vals = km.silhouette_scan.map(s=>s.silhouette);
    const bestK = km.best_k_by_silhouette, nClasses = km.n_classes;
    const colors = ks.map(k=> k===bestK ? '#4ec97a' : (k===nClasses ? '#d24b52' : '#7f9ab5'));
    const sizes = ks.map(k=> (k===bestK||k===nClasses) ? 12 : 7);
    Plotly.react(sil, [{
      type:'scatter', mode:'lines+markers', x:ks, y:vals,
      line:{color:'#46566a',width:1.5},
      marker:{color:colors, size:sizes, line:{color:'#0e1116',width:1}},
      hovertemplate:'k=%{x}: silhouette %{y:.3f}<extra></extra>',
    }], baseLayout({
      xaxis:Object.assign({title:'number of clusters k', dtick:1},AXIS),
      yaxis:Object.assign({title:'silhouette'},AXIS),
      height:200, margin:{l:48,r:14,t:10,b:36},
      annotations:[
        {x:bestK, y:vals[ks.indexOf(bestK)], text:'best', showarrow:true, arrowhead:0,
         ax:0, ay:-20, font:{size:9,color:'#4ec97a'}, arrowcolor:'#4ec97a'},
        (ks.includes(nClasses) && nClasses!==bestK ? {x:nClasses, y:vals[ks.indexOf(nClasses)],
         text:`#classes (${nClasses})`, showarrow:true, arrowhead:0, ax:0, ay:22,
         font:{size:9,color:'#d24b52'}, arrowcolor:'#d24b52'} : null),
      ].filter(Boolean),
    }), PLOT_CONFIG);
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
    (p.decisive ? 'The interval excludes zero — the win is reliable.'
                : (p.tie ? 'Within the tie threshold.' : 'The interval crosses zero — provisional.'))+
    `</div>`;
  html += `<div class="an-why" style="margin-top:12px"><strong>How to read this.</strong> `+
    `Fidelity (weight 0.40) is the per-point distortion; trust and continuity (0.20 each) are the classical neighbour-preservation scalars; cluster (0.20) is how well each projection preserves the k-means partition. Weights are fixed, not tuned, and every component is shown so you can weigh them yourself.</div>`;
  body.innerHTML = html;
  $('verdictModal').hidden = false;
}
function closeVerdictModal(){ $('verdictModal').hidden = true; }


/* TABS / CONSOLE / TOAST */
const TAB_INTROS = {
  dataquality: `<strong>Data Quality</strong> — is the class structure genuinely in the data, and do the labels describe it? These answers depend only on the data, <em>not</em> on any projection. If the data has no structure, or the labels don't match it, no projection can fix that.`,
  projquality: `<strong>Projection Quality</strong> — given the data, how faithfully does each 2-D projection preserve it, and which is better? These answers are about the <em>picture</em>. A good projection of poorly-structured data still looks poor — that's a data problem, diagnosed in the Data Quality tab.`,
  explorer: `<strong>Point Explorer</strong> — drill into the individual points a projection gets wrong, decomposed into false-closeness (intrusion) and lost-neighbour (extrusion) for each method.`,
};
function activateTab(name){
  document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('active',x.dataset.tab===name));
  // show every panel whose data-group matches the active tab
  document.querySelectorAll('.panel').forEach(p=>p.classList.toggle('active', p.dataset.group===name));
  const intro = $('tabIntro');
  if(intro) intro.innerHTML = TAB_INTROS[name] || '';
  // Plotly needs a resize nudge when hidden plots become visible
  setTimeout(()=>{
    document.querySelectorAll(`.panel[data-group="${name}"] .plot`).forEach(pl=>{
      if(pl.data && typeof Plotly!=='undefined') Plotly.Plots.resize(pl);
    });
    if(name==='projquality' && state.result){
      applyEsSelection(state.result, null);
      safeRender('anneal', ()=>renderAnnealCard(state.result));
    }
    if(name==='dataquality' && state.result){
      safeRender('anneal', ()=>renderAnnealCard(state.result));
      safeRender('kmeans-charts', ()=>renderKmeansCharts(state.result));
      safeRender('metrics', ()=>renderMetricsCard(state.result));
    }
  }, 40);
}
function bindTabs(){
  $('tabs').addEventListener('click',(e)=>{
    const t=e.target.closest('.tab'); if(!t) return;
    activateTab(t.dataset.tab);
  });
}

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