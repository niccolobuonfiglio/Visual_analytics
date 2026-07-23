/* Projection Fidelity Console: frontend logic */

const $ = (id) => document.getElementById(id);
const state = {
  result: null,        // latest run result
  metric: 'intrusion',
  weight: 'rank',
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
  await restoreLast();
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
  echo('alpha','alphaEcho',(v)=>parseFloat(v).toFixed(1));
  echo('beta','betaEcho',(v)=>parseFloat(v).toFixed(1));
  echo('normScale','normScaleEcho',(v)=>parseFloat(v).toFixed(1));
  echo('threshold','threshEcho',(v)=>parseFloat(v).toFixed(2));

  // metric segmented control
  $('metricSeg').addEventListener('click', (e)=>{
    const b = e.target.closest('.seg-btn'); if(!b) return;
    state.metric = b.dataset.metric;
    document.querySelectorAll('#metricSeg .seg-btn').forEach(x=>x.classList.toggle('active', x===b));
    $('intrusionPanel').hidden = state.metric!=='intrusion';
    $('classicPanel').hidden = state.metric!=='classic';
  });

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

  // weighted MDS → reveal method picker; method=custom → reveal weights box
  const syncWeighted = ()=>{
    const on = $('weightedMds').checked;
    $('weightMethodField').hidden = !on;
    $('customWeightsField').hidden = !(on && $('weightMethod').value==='custom');
  };
  $('weightedMds').addEventListener('change', syncWeighted);
  $('weightMethod').addEventListener('change', syncWeighted);

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
    geometric: $('geometric').checked,
    weight: state.weight,
    adaptive: $('adaptive').checked,
    normalize: $('normalize').value,
    norm_scale: parseFloat($('normScale').value),
    alpha: parseFloat($('alpha').value),
    beta: parseFloat($('beta').value),
    no_standardize: $('noStandardize').checked,
    threshold: parseFloat($('threshold').value),
    weighted_mds: $('weightedMds').checked,
    weight_method: $('weightMethod').value,
  };
  const ms = $('maxSamples').value;
  if(ms) p.max_samples = parseInt(ms);
  if(p.weighted_mds && p.weight_method==='custom'){
    const raw = $('customWeights').value.trim();
    if(raw){
      p.custom_weights = raw.split(',').map(s=>parseFloat(s.trim())).filter(v=>!isNaN(v));
    }
  }
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
  ['cmpPlot', 'balPca', 'balMds', 'centPlot', 'histPca', 'histMds',
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
  state.metric = p.metric || 'intrusion';
  document.querySelectorAll('#metricSeg .seg-btn').forEach(x=>x.classList.toggle('active', x.dataset.metric===state.metric));
  $('intrusionPanel').hidden = state.metric!=='intrusion';
  $('classicPanel').hidden = state.metric!=='classic';
  state.weight = p.weight || 'rank';
  document.querySelectorAll('#weightSeg .seg-btn').forEach(x=>x.classList.toggle('active', x.dataset.weight===state.weight));
  const setV=(id,v)=>{ if(v!==undefined&&v!==null) $(id).value=v; };
  setV('neighbors',p.neighbors); $('kEcho').textContent=p.neighbors??10;
  setV('alpha',p.alpha); $('alphaEcho').textContent=(p.alpha??1).toFixed?.(1)??p.alpha;
  setV('beta',p.beta); $('betaEcho').textContent=(p.beta??1).toFixed?.(1)??p.beta;
  setV('normalize',p.normalize);
  setV('normScale',p.norm_scale); $('normScaleEcho').textContent=(p.norm_scale??3).toFixed?.(1)??p.norm_scale;
  $('normScaleField').hidden = !(p.normalize==='tanh'||p.normalize==='sigmoid');
  setV('threshold',p.threshold); $('threshEcho').textContent=(p.threshold??0.5).toFixed?.(2)??p.threshold;
  $('geometric').checked = !!p.geometric;
  $('adaptive').checked = !!p.adaptive;
  $('noStandardize').checked = !!p.no_standardize;
  if(p.max_samples) $('maxSamples').value = p.max_samples;
  $('weightedMds').checked = !!p.weighted_mds;
  if(p.weight_method) $('weightMethod').value = p.weight_method;
  if(p.custom_weights) $('customWeights').value = p.custom_weights.join(', ');
  $('weightMethodField').hidden = !p.weighted_mds;
  $('customWeightsField').hidden = !(p.weighted_mds && p.weight_method==='custom');
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
  safeRender('signal', ()=>renderSignal(d));
  safeRender('table', ()=>renderTable(d));
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
  syncBalanceTab(d);
  safeRender('errorspace', ()=>renderErrorSpace(d));
  safeRender('comparison', ()=>renderComparison(d));
  if(hasSeparableParts(d)) safeRender('balance', ()=>renderBalance(d));
  safeRender('centroids', ()=>renderCentroids(d));
  safeRender('histograms', ()=>renderHistograms(d));
}

/** The intrusion metric computes genuine, independent FP and FN components.
 *  The classic metric only splits one signed score into halves, so FP and FN
 *  are never both non-zero and a FP-vs-FN scatter would be a degenerate "L". */
function hasSeparableParts(d){ return d.metric !== 'classic'; }

/** Show/hide the FP vs FN tab to match the metric, and move off it if it is
 *  the active tab when it disappears. */
function syncBalanceTab(d){
  const show = hasSeparableParts(d);
  const tab = document.querySelector('.tab[data-tab="balance"]');
  const panel = document.querySelector('.panel[data-panel="balance"]');
  if(!tab || !panel) return;
  tab.hidden = !show;
  if(!show && tab.classList.contains('active')){
    // fall back to the Error Space tab
    activateTab('errorspace');
  }
}
function safeRender(label, fn){
  try{ fn(); }
  catch(e){ console.error(`render[${label}] failed:`, e); }
}
function plotlyUnavailable(){
  const msg = 'Plotting library did not load: check your network connection and reload. Tables and console output are available.';
  ['cmpPlot','balPca','balMds','centPlot','histPca','histMds',
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
    pill('Flagged', s.untrustworthy_count, '', `|error| ≥ ${d.params.threshold}`) +
    pill('Runtime', info.elapsed+'s', '', 'pipeline');
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
function renderComparison(d){
  const vmax = d.vmax;
  const labels = d.table.labels, names = d.table.class_names;
  const uniq = [...new Set(labels)];
  const traces = uniq.map(code=>{
    const idx = labels.map((l,i)=>l===code?i:-1).filter(i=>i>=0);
    return {
      type:'scattergl', mode:'markers',
      x: idx.map(i=>d.table.pca.err[i]),
      y: idx.map(i=>d.table.mds.err[i]),
      name: names[idx[0]],
      marker:{ size:8, color:classColor(code), line:{color:'rgba(0,0,0,0.5)',width:0.5} },
      customdata: idx.map(i=>[d.table.ids[i], names[i]]),
      hovertemplate:'idx %{customdata[0]} · %{customdata[1]}<br>PCA %{x:.3f} · MDS %{y:.3f}<extra></extra>',
    };
  });
  const guide = {
    type:'scatter', mode:'lines', x:[-vmax,vmax], y:[-vmax,vmax],
    line:{color:'#33404f',width:1,dash:'dot'}, hoverinfo:'skip', showlegend:false
  };
  const layout = baseLayout({
    showlegend:true,
    legend:{ font:{size:10,color:'#9aa7b6'}, bgcolor:'rgba(21,26,33,0.7)',
             bordercolor:'#262f3b', borderwidth:1, x:0.02, y:0.98 },
    xaxis:Object.assign({title:'PCA error',range:[-vmax,vmax],zeroline:true},AXIS),
    yaxis:Object.assign({title:'MDS error',range:[-vmax,vmax],zeroline:true},AXIS),
    annotations:[
      quadLabel(-vmax*0.9,-vmax*0.9,'both FN','left','bottom'),
      quadLabel(vmax*0.9,vmax*0.9,'both FP','right','top'),
      quadLabel(-vmax*0.9,vmax*0.9,'PCA-FN / MDS-FP','left','top'),
      quadLabel(vmax*0.9,-vmax*0.9,'PCA-FP / MDS-FN','right','bottom'),
    ]
  });
  Plotly.react('cmpPlot', [guide, ...traces], layout, PLOT_CONFIG);
}
function quadLabel(x,y,text,xa,ya){
  return { x, y, text, showarrow:false, xanchor:xa, yanchor:ya,
    font:{family:'Space Mono',size:9,color:'#63707f'} };
}

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
function renderBalance(d){
  const labels=d.table.labels, names=d.table.class_names;
  const uniq=[...new Set(labels)].sort((a,b)=>a-b);

  function build(divId, t, label){
    // symmetric upper bound so the FP≈FN diagonal sits at 45°
    const maxv=Math.max(0.001, ...t.fp, ...t.fn);
    const hi=Math.ceil(maxv*20)/20;   // round up to a clean-ish tick
    const traces=uniq.map(code=>{
      const idx=labels.map((l,i)=>l===code?i:-1).filter(i=>i>=0);
      return {
        type:'scattergl', mode:'markers',
        x: idx.map(i=>t.fp[i]),
        y: idx.map(i=>t.fn[i]),
        name: names[idx[0]],
        marker:{ size:7, color:classColor(code), line:{color:'rgba(0,0,0,0.5)',width:0.5} },
        customdata: idx.map(i=>[d.table.ids[i], names[i], t.err[i]]),
        hovertemplate:'idx %{customdata[0]} · %{customdata[1]}<br>FP %{x:.3f} · FN %{y:.3f}<br>net %{customdata[2]:.3f}<extra></extra>',
      };
    });
    const diag={ type:'scatter',mode:'lines',x:[0,hi],y:[0,hi],
      line:{color:'#33404f',width:1,dash:'dot'}, hoverinfo:'skip', showlegend:false,
      name:'FP = FN' };
    Plotly.react(divId,[diag,...traces], baseLayout({
      title:{text:`${label}  ·  FP vs FN`,font:{size:12,color:'#9aa7b6'},x:0.02,xanchor:'left'},
      showlegend:true,
      legend:{ font:{size:9,color:'#9aa7b6'}, bgcolor:'rgba(21,26,33,0.7)',
               bordercolor:'#262f3b',borderwidth:1, x:0.98, xanchor:'right', y:0.98 },
      annotations:[
        { x:hi*0.97, y:hi*0.97, text:'both high', showarrow:false, xanchor:'right', yanchor:'top',
          font:{family:'Space Mono',size:9,color:'#63707f'} },
        { x:hi*0.5, y:hi*0.5, text:'FP ≈ FN (net ≈ 0)', showarrow:false, xanchor:'left', yanchor:'bottom',
          textangle:-45, font:{family:'Space Mono',size:8,color:'#4a5563'} },
      ],
      xaxis:Object.assign({title:'FP  (intrusion)',range:[0,hi],zeroline:true},AXIS),
      yaxis:Object.assign({title:'FN  (extrusion)',range:[0,hi],zeroline:true,scaleanchor:'x',scaleratio:1},AXIS),
    }),PLOT_CONFIG);
  }
  build('balPca', d.table.pca, 'PCA');
  build('balMds', d.table.mds, 'MDS');
}

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
  return {
    type:'scattergl', mode:'markers',
    x: isErr ? d.table.pca.err : t.x,
    y: isErr ? d.table.mds.err : t.y,
    marker:{ size:sizes, color: t.err, colorscale:DIVERGING, cmin:-vmax, cmax:vmax,
      line:{color:'rgba(0,0,0,0.5)',width:0.5} },
    customdata: d.table.ids.map((id,i)=>[
      d.table.class_names[i], id, d.table.pca.err[i], d.table.mds.err[i],
      t.fp[i], t.fn[i], tot[i]]),
    hovertemplate: isErr
      ? 'idx %{customdata[1]} · %{customdata[0]}<br>' +
        'PCA err %{x:.3f} · MDS err %{y:.3f}<br>' +
        'total distortion %{customdata[6]:.3f}<extra></extra>'
      : 'idx %{customdata[1]} · %{customdata[0]}<br>' +
        'error %{marker.color:.3f} &nbsp; (FP %{customdata[4]:.3f} · FN %{customdata[5]:.3f})<br>' +
        'total distortion %{customdata[6]:.3f}<extra></extra>',
    selected:{ marker:{ opacity:1 } },
    unselected:{ marker:{ opacity:0.15 } },
  };
}
function esRingTrace(){
  return { type:'scattergl', mode:'markers', x:[], y:[],
    marker:{ size:15, color:'rgba(0,0,0,0)', line:{color:'#7CFC4D',width:2.2} },
    hoverinfo:'skip', showlegend:false };
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
  applyEsSelection(d, null);   // reflect any carried-over selection
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
        const test=ES_QUADS[quad];
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
  const counts={ bothFP:0, bothFN:0, pcaFPmdsFN:0, pcaFNmdsFP:0 };
  d.table.pca.err.forEach((pe,i)=>{
    const me=d.table.mds.err[i];
    for(const q in ES_QUADS){ if(ES_QUADS[q](pe,me)){ counts[q]++; break; } }
  });
  Object.keys(counts).forEach(q=>{
    const el=document.querySelector(`#esQuadrants .es-quad-count[data-count="${q}"]`);
    if(el) el.textContent=counts[q];
  });
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
    pcaFPmdsFN:'PCA-FP / MDS-FN', pcaFNmdsFP:'PCA-FN / MDS-FP' };
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

/* TABS / CONSOLE / TOAST */
function activateTab(name){
  document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('active',x.dataset.tab===name));
  document.querySelectorAll('.panel').forEach(p=>p.classList.toggle('active',p.dataset.panel===name));
  // Plotly needs a resize nudge when a hidden plot becomes visible
  setTimeout(()=>{
    document.querySelectorAll(`.panel[data-panel="${name}"] .plot`).forEach(pl=>{
      if(pl.data && typeof Plotly!=='undefined') Plotly.Plots.resize(pl);
    });
    // re-apply the Error Space rings after the resize so they stay aligned
    if(name==='errorspace' && state.result) applyEsSelection(state.result, null);
  }, 30);
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