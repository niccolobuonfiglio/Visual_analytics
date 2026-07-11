/* ============================================================
   Projection Fidelity Console — frontend logic
   ============================================================ */

const $ = (id) => document.getElementById(id);
const state = {
  result: null,        // latest run result
  metric: 'intrusion',
  weight: 'rank',
  projMode: 'view',    // 'view' | 'brush' — projections panel mode
  brushSel: new Set(), // selected point indices (persists across mode switches)
  esSel: new Set(),    // Error Space: shared selection across all 3 plots
  esActiveQuad: null,  // Error Space: currently active quadrant button (or null)
  tableSort: { key: 'pca_error', dir: -1 },
};

/* ---- error scale colour (matches CSS: red FN → white → blue FP) ---- */
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

/* ---- shared Plotly layout bits (dark instrument theme) ---- */
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

/* ============================================================
   INIT
   ============================================================ */
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

/* ---- control binding ---- */
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

  // projections View/Brush mode toggle
  $('projModeSeg').addEventListener('click', (e)=>{
    const b = e.target.closest('.seg-btn'); if(!b) return;
    setProjMode(b.dataset.mode);
  });
  $('brushClear').addEventListener('click', clearBrushSelection);

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
    toast('CSV loaded — pick a label column');
  }catch(e){ toast('upload error', true); }
}
*/

/* ---- collect params from the form ---- */
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

/* ============================================================
   RUN
   ============================================================ */
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
      toast(d.error || 'Analysis failed — see console', true);
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
  state.brushSel = new Set();
  state.projMode = 'view';
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
  ['projPca', 'projMds', 'cmpPlot', 'balPca', 'balMds', 'centPlot', 'histPca', 'histMds',
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
  
  // Clear brush table
  const brushThead = $('brushTable').querySelector('thead');
  const brushTbody = $('brushTable').querySelector('tbody');
  if (brushThead) brushThead.innerHTML = '';
  if (brushTbody) brushTbody.innerHTML = '';

  // Clear Error Space table, counts, and source tag
  const esThead = $('esTable').querySelector('thead');
  const esTbody = $('esTable').querySelector('tbody');
  if (esThead) esThead.innerHTML = '';
  if (esTbody) esTbody.innerHTML = '';
  document.querySelectorAll('#esQuadrants .es-quad-count').forEach(el=>el.textContent='0');
  document.querySelectorAll('#esQuadrants .es-quad').forEach(b=>b.classList.remove('active'));
  $('esCount').textContent = 'No points selected — drag on any plot or pick a quadrant';
  $('esSource').textContent = ''; $('esSource').hidden = true;
  
  // Reset signal
  $('signalFn').style.width = '0%';
  $('signalFp').style.width = '0%';
  $('signalLabel').textContent = 'SIGNED-ERROR BALANCE';
  $('signalLabel').style.color = '#63707f';
  $('signalLabel').className = 'signal-label';
  $('signalReadout').textContent = 'awaiting run';
  
  // Reset brush count
  $('brushCount').textContent = 'No points selected — drag a box on either plot';
  
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

/* ============================================================
   RENDER
   ============================================================ */
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
  // reset projections to View mode on each fresh render, clear stale selection
  state.brushSel = new Set();
  state.esSel = new Set();
  state.esActiveQuad = null;
  safeRender('projections', ()=>renderProjections(d));
  safeRender('errorspace', ()=>renderErrorSpace(d));
  safeRender('comparison', ()=>renderComparison(d));
  safeRender('balance', ()=>renderBalance(d));
  safeRender('centroids', ()=>renderCentroids(d));
  safeRender('histograms', ()=>renderHistograms(d));
}
function safeRender(label, fn){
  try{ fn(); }
  catch(e){ console.error(`render[${label}] failed:`, e); }
}
function plotlyUnavailable(){
  const msg = 'Plotting library did not load — check your network connection and reload. Tables and console output are available.';
  ['projPca','projMds','cmpPlot','balPca','balMds','centPlot','histPca','histMds',
   'esPca','esMds','esErr'].forEach(id=>{
    const el=$(id); if(el) el.innerHTML=`<div class="plot-fallback">${msg}</div>`;
  });
  toast('Plotting library unavailable — see tables instead', true);
}

function renderStatbar(d){
  const s = d.summary, info = d.info;
  const pill = (label, value, cls='', sub='') =>
    `<div class="stat"><div class="stat-label">${label}</div>`+
    `<div class="stat-value ${cls}">${value}</div>`+
    (sub?`<div class="stat-sub">${sub}</div>`:'')+`</div>`;
  const signCls = (v)=> v<0?'neg':(v>0?'pos':'');
  $('statbar').innerHTML =
    pill('Dataset', info.name, '', `${info.standardized?'standardized':'raw'} · ${d.metric}`) +
    pill('Samples', info.n_samples, '', `${info.n_features} features · ${info.n_classes} classes`) +
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

/* ---- projection scatter (one method) ---- */
function projTrace(t, vmax){
  return {
    type:'scattergl', mode:'markers',
    x:t.x, y:t.y,
    marker:{
      size:9, color:t.err, colorscale:DIVERGING, cmin:-vmax, cmax:vmax,
      line:{ color:'rgba(0,0,0,0.55)', width:0.6 },
      colorbar:{
        thickness:10, len:0.82, x:1.0, tickfont:{family:'Space Mono',size:9,color:'#63707f'},
        tickvals:[-vmax,0,vmax], ticktext:[`FN -${vmax}`,'0',`FP +${vmax}`],
        outlinewidth:0, bgcolor:'rgba(0,0,0,0)'
      }
    },
    customdata: t.err.map((e,i)=>[t.cls_names[i], t.ids[i], t.fp[i], t.fn[i]]),
    hovertemplate:'idx %{customdata[1]} · %{customdata[0]}<br>error %{marker.color:.3f}<br>FP %{customdata[2]:.3f} · FN %{customdata[3]:.3f}<extra></extra>',
  };
}
function withMeta(m, d){
  return Object.assign({}, m, {
    cls_names: d.table.class_names, ids: d.table.ids
  });
}
function renderProjections(d){
  // reflect current mode in the UI, then draw the plots for that mode
  document.querySelectorAll('#projModeSeg .seg-btn').forEach(x=>
    x.classList.toggle('active', x.dataset.mode===state.projMode));
  $('brushClear').hidden = state.projMode!=='brush';
  $('brushDetail').hidden = state.projMode!=='brush';
  if(state.projMode==='brush') drawBrushMode(d);
  else drawViewMode(d);
}

/* ---- View mode: static coloured scatter (the classic projection plots) ---- */
function drawViewMode(d){
  const vmax = d.vmax;
  const pca = withMeta(d.table.pca, d), mds = withMeta(d.table.mds, d);
  Plotly.react('projPca', [projTrace(pca, vmax)],
    baseLayout({ title:{text:'PCA',font:{size:13,color:'#e8edf3'},x:0.02,xanchor:'left'},
      xaxis:Object.assign({title:'dim 1'},AXIS), yaxis:Object.assign({title:'dim 2'},AXIS) }),
    PLOT_CONFIG);
  Plotly.react('projMds', [projTrace(mds, vmax)],
    baseLayout({ title:{text:'MDS',font:{size:13,color:'#e8edf3'},x:0.02,xanchor:'left'},
      xaxis:Object.assign({title:'dim 1'},AXIS), yaxis:Object.assign({title:'dim 2'},AXIS) }),
    PLOT_CONFIG);
}

/* ---- switch between View and Brush, preserving the current selection ---- */
function setProjMode(mode){
  if(mode===state.projMode) return;
  state.projMode = mode;
  if(state.result) renderProjections(state.result);
}

/* ---- comparison scatter ---- */
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

/* ---- histograms ---- */
function renderHistograms(d){
  const vmax = d.vmax, bins=30;
  const edges=[]; for(let i=0;i<=bins;i++) edges.push(-vmax + (2*vmax)*i/bins);
  function histTrace(err){
    const centres=[], counts=new Array(bins).fill(0), colors=[];
    for(let i=0;i<bins;i++){ centres.push((edges[i]+edges[i+1])/2); colors.push(scaleColor(centres[i],vmax)); }
    err.forEach(v=>{
      const cv=Math.max(-vmax,Math.min(vmax,v));
      let b=Math.floor((cv+vmax)/(2*vmax)*bins); if(b>=bins)b=bins-1; if(b<0)b=0;
      counts[b]++;
    });
    return { centres, counts, colors };
  }
  function build(divId, err, label){
    const h=histTrace(err);
    const bar={ type:'bar', x:h.centres, y:h.counts, marker:{color:h.colors,line:{color:'rgba(0,0,0,0.4)',width:0.4}},
      width:(edges[1]-edges[0])*0.92, hovertemplate:'error %{x:.3f}<br>%{y} pts<extra></extra>' };
    const maxc=Math.max(...h.counts,1);

    // horizontal statistics on the bin COUNTS (vertical axis): mean ± 2σ
    const meanCount = h.counts.reduce((a,b)=>a+b,0)/h.counts.length;
    const sdCount = Math.sqrt(h.counts.reduce((a,b)=>a+(b-meanCount)**2,0)/h.counts.length);
    const loBand = Math.max(0, meanCount-2*sdCount), hiBand = meanCount+2*sdCount;
    const xSpan=[-vmax,vmax];

    // shaded ±2σ band around the mean count, drawn as a filled horizontal ribbon
    const band={ type:'scatter',mode:'lines',
      x:[xSpan[0],xSpan[1],xSpan[1],xSpan[0]],
      y:[loBand,loBand,hiBand,hiBand],
      fill:'toself', fillcolor:'rgba(232,237,243,0.09)',
      line:{width:0}, hoverinfo:'skip', showlegend:false };
    // the mean-count line itself, carrying the legend label
    const meanLine={ type:'scatter',mode:'lines',x:xSpan,y:[meanCount,meanCount],
      line:{color:'#e8edf3',width:1.4,dash:'dot'},
      name:`mean ± 2σ`, showlegend:true,
      hovertemplate:`mean count ${meanCount.toFixed(1)} ± ${(2*sdCount).toFixed(1)}<extra></extra>` };
    const zeroLine={ type:'scatter',mode:'lines',x:[0,0],y:[0,Math.max(maxc,hiBand)*1.05],
      line:{color:'#63707f',width:1,dash:'dash'},hoverinfo:'skip',showlegend:false };

    Plotly.react(divId,[bar,zeroLine,band,meanLine], baseLayout({
      title:{text:`${label}  ·  n=${err.length}`,font:{size:12,color:'#9aa7b6'},x:0.02,xanchor:'left'},
      bargap:0.04,
      showlegend:true,
      legend:{ font:{size:10,color:'#9aa7b6'}, bgcolor:'rgba(21,26,33,0.7)',
               bordercolor:'#262f3b', borderwidth:1, x:0.98, xanchor:'right', y:0.98 },
      xaxis:Object.assign({title:'signed error (FN < 0 < FP)',range:[-vmax,vmax]},AXIS),
      yaxis:Object.assign({title:'count',rangemode:'tozero'},AXIS),
    }),PLOT_CONFIG);
  }
  build('histPca', d.table.pca.err, 'PCA');
  build('histMds', d.table.mds.err, 'MDS');
}

/* ---- FP vs FN scatter (intrusion/extrusion balance), one panel per method ---- */
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

/* ---- class-centroid trajectory: high-D -> PCA -> MDS ---- */
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
  ).join('') + `<span class="cent-note">mean between-centroid distance (normalised) — similar values across stages ⇒ class separation preserved</span>`;
}

/* ---- data table (sortable) ---- */
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
    : 'No points exceed the current threshold — try lowering it.';
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

/* ============================================================
   BRUSHING — linked rectangle selection (Brush mode of Projections)
   ============================================================ */
function brushBaseTrace(t, vmax, d){
  return {
    type:'scattergl', mode:'markers',
    x:t.x, y:t.y,
    marker:{ size:8, color:t.err, colorscale:DIVERGING, cmin:-vmax, cmax:vmax,
      line:{color:'rgba(0,0,0,0.5)',width:0.5} },
    customdata: t.err.map((e,i)=>[d.table.class_names[i], d.table.ids[i]]),
    hovertemplate:'idx %{customdata[1]} · %{customdata[0]}<br>error %{marker.color:.3f}<extra></extra>',
    selected:{ marker:{ opacity:1 } },
    unselected:{ marker:{ opacity:0.18 } },
  };
}
function brushRingTrace(){
  return { type:'scattergl', mode:'markers', x:[], y:[],
    marker:{ size:16, color:'rgba(0,0,0,0)', line:{color:'#7CFC4D',width:2.2} },
    hoverinfo:'skip', showlegend:false };
}
/* ---- Brush mode: selection-enabled scatter on the shared projection divs ---- */
function drawBrushMode(d){
  const vmax=d.vmax;
  const layoutFor = (title)=> baseLayout({
    title:{text:title,font:{size:13,color:'#e8edf3'},x:0.02,xanchor:'left'},
    dragmode:'select',
    xaxis:Object.assign({title:'dim 1'},AXIS), yaxis:Object.assign({title:'dim 2'},AXIS),
  });
  Plotly.react('projPca', [brushBaseTrace(d.table.pca,vmax,d), brushRingTrace()],
    layoutFor('PCA'), Object.assign({}, PLOT_CONFIG, {modeBarButtonsToAdd:['select2d','lasso2d']}));
  Plotly.react('projMds', [brushBaseTrace(d.table.mds,vmax,d), brushRingTrace()],
    layoutFor('MDS'), Object.assign({}, PLOT_CONFIG, {modeBarButtonsToAdd:['select2d','lasso2d']}));

  // bind selection events (once per div; Plotly re-emits on re-render)
  bindBrush('projPca'); bindBrush('projMds');
  applyBrush();   // re-apply any selection carried over from a previous mode switch
}
function bindBrush(divId){
  const div=$(divId);
  div.removeAllListeners?.('plotly_selected');
  div.on('plotly_selected', (ev)=>{
    if(!ev || !ev.points){ return; }
    const sel=new Set(ev.points.map(p=>p.pointIndex));
    state.brushSel = sel;
    applyBrush();
  });
  div.on('plotly_deselect', ()=>{ state.brushSel=new Set(); applyBrush(); });
}
function applyBrush(){
  const d=state.result; if(!d || state.projMode!=='brush') return;
  const idx=[...state.brushSel];
  // ring the same points on BOTH plots (trace index 1 = ring layer)
  const ringPca={ x:idx.map(i=>d.table.pca.x[i]), y:idx.map(i=>d.table.pca.y[i]) };
  const ringMds={ x:idx.map(i=>d.table.mds.x[i]), y:idx.map(i=>d.table.mds.y[i]) };
  Plotly.restyle('projPca', {x:[ringPca.x], y:[ringPca.y]}, [1]);
  Plotly.restyle('projMds', {x:[ringMds.x], y:[ringMds.y]}, [1]);
  // reflect selection styling on both base traces
  Plotly.restyle('projPca', {selectedpoints:[idx.length?idx:null]}, [0]);
  Plotly.restyle('projMds', {selectedpoints:[idx.length?idx:null]}, [0]);
  updateBrushDetail(d);
}
function updateBrushDetail(d){
  const idx=[...state.brushSel];
  $('brushCount').textContent = idx.length
    ? `${idx.length} point${idx.length>1?'s':''} selected`
    : 'No points selected — drag a box on either plot';
  const cols=[
    {k:'idx',l:'index'},{k:'cls',l:'class',cls:'col-class'},
    {k:'pe',l:'PCA err',err:true},{k:'me',l:'MDS err',err:true},
    {k:'pfp',l:'PCA FP',part:'fp'},{k:'pfn',l:'PCA FN',part:'fn'},
    {k:'mfp',l:'MDS FP',part:'fp'},{k:'mfn',l:'MDS FN',part:'fn'},
  ];
  const thead=$('brushTable').querySelector('thead');
  const tbody=$('brushTable').querySelector('tbody');
  thead.innerHTML='<tr>'+cols.map(c=>`<th class="${c.cls||''}">${c.l}</th>`).join('')+'</tr>';
  const rows=idx.map(i=>({
    idx:d.table.ids[i], cls:d.table.class_names[i],
    pe:d.table.pca.err[i], me:d.table.mds.err[i],
    pfp:d.table.pca.fp[i], pfn:d.table.pca.fn[i],
    mfp:d.table.mds.fp[i], mfn:d.table.mds.fn[i],
  })).sort((a,b)=>a.pe-b.pe);
  tbody.innerHTML = rows.map(r=>'<tr>'+cols.map(c=>{
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
function clearBrushSelection(){
  state.brushSel=new Set();
  if(state.result && state.projMode==='brush'){
    Plotly.restyle('projPca',{x:[[]],y:[[]]},[1]);
    Plotly.restyle('projMds',{x:[[]],y:[[]]},[1]);
    Plotly.restyle('projPca',{selectedpoints:[null]},[0]);
    Plotly.restyle('projMds',{selectedpoints:[null]},[0]);
    // also drop Plotly's own selection outline
    Plotly.relayout('projPca',{selections:[]});
    Plotly.relayout('projMds',{selections:[]});
    updateBrushDetail(state.result);
  }
}

/* ============================================================
   ERROR SPACE — integrated PCA + MDS + error-space exploration
   Three linked plots share one selection. Cross-brushing on any
   plot, quadrant buttons, and a shared detail table all drive
   the same state.esSel.
   ============================================================ */

// which quadrant each point falls in, by PCA/MDS error sign
const ES_QUADS = {
  bothFP:     (pe,me)=> pe>0 && me>0,
  bothFN:     (pe,me)=> pe<0 && me<0,
  pcaFPmdsFN: (pe,me)=> pe>0 && me<0,
  pcaFNmdsFP: (pe,me)=> pe<0 && me>0,
};

function esBaseTrace(t, vmax, d, kind){
  // kind: 'proj' (x/y are coords) or 'err' (x/y are PCA/MDS error)
  const isErr = kind==='err';
  return {
    type:'scattergl', mode:'markers',
    x: isErr ? d.table.pca.err : t.x,
    y: isErr ? d.table.mds.err : t.y,
    marker:{ size:8, color: t.err, colorscale:DIVERGING, cmin:-vmax, cmax:vmax,
      line:{color:'rgba(0,0,0,0.5)',width:0.5} },
    customdata: d.table.ids.map((id,i)=>[
      d.table.class_names[i], id, d.table.pca.err[i], d.table.mds.err[i]]),
    hovertemplate: isErr
      ? 'idx %{customdata[1]} · %{customdata[0]}<br>PCA %{x:.3f} · MDS %{y:.3f}<extra></extra>'
      : 'idx %{customdata[1]} · %{customdata[0]}<br>PCA err %{customdata[2]:.3f} · MDS err %{customdata[3]:.3f}<extra></extra>',
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

  Plotly.react('esPca', [esBaseTrace(d.table.pca,vmax,d,'proj'), esRingTrace()],
    projLayout('PCA'), cfg);
  Plotly.react('esMds', [esBaseTrace(d.table.mds,vmax,d,'proj'), esRingTrace()],
    projLayout('MDS'), cfg);
  Plotly.react('esErr', [esBaseTrace(d.table.pca,vmax,d,'err'), esRingTrace()],
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
  Plotly.restyle('esPca', {x:[rP.x], y:[rP.y]}, [1]);
  Plotly.restyle('esMds', {x:[rM.x], y:[rM.y]}, [1]);
  Plotly.restyle('esErr', {x:[rE.x], y:[rE.y]}, [1]);
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
    : 'No points selected — drag on any plot or pick a quadrant';
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
   TABS / CONSOLE / TOAST
   ============================================================ */
function bindTabs(){
  $('tabs').addEventListener('click',(e)=>{
    const t=e.target.closest('.tab'); if(!t) return;
    const name=t.dataset.tab;
    document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('active',x===t));
    document.querySelectorAll('.panel').forEach(p=>p.classList.toggle('active',p.dataset.panel===name));
    // Plotly needs a resize nudge when a hidden plot becomes visible
    setTimeout(()=>{
      document.querySelectorAll(`.panel[data-panel="${name}"] .plot`).forEach(pl=>{
        if(pl.data) Plotly.Plots.resize(pl);
      });
      // re-apply the Error Space rings after the resize so they stay aligned
      if(name==='errorspace' && state.result) applyEsSelection(state.result, null);
    }, 30);
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

/* ---- expose for debugging / testing ---- */
window.state = state;
window.applyBrush = applyBrush;
window.setProjMode = setProjMode;
window.applyEsSelection = applyEsSelection;
window.clearEsSelection = clearEsSelection;
window.renderErrorSpace = renderErrorSpace;
window.runAnalysis = runAnalysis;

/* ---- go ---- */
init();