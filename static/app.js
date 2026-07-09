/* ============================================================
   Projection Fidelity Console — frontend logic
   ============================================================ */

const $ = (id) => document.getElementById(id);
const state = {
  result: null,        // latest run result
  metric: 'intrusion',
  weight: 'rank',
  brushSel: new Set(), // selected point indices (brushing tab)
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
  await loadDatasets();
  bindControls();
  bindTabs();
  bindConsole();
  bindBrushClear();
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

  // CSV upload
  $('csvFile').addEventListener('change', handleCsvUpload);
  const csvLabel = $('csvLabel');
  ['dragover','dragenter'].forEach(ev=>csvLabel.addEventListener(ev,(e)=>{e.preventDefault(); csvLabel.classList.add('loaded');}));
  ['dragleave','drop'].forEach(ev=>csvLabel.addEventListener(ev,(e)=>{e.preventDefault();}));
  csvLabel.addEventListener('drop',(e)=>{
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if(f){ $('csvFile').files = e.dataTransfer.files; handleCsvUpload(); }
  });

  // run
  $('runBtn').addEventListener('click', runAnalysis);
  $('clearBtn').addEventListener('click', clearRuns);
  // ⏎ to run (when not typing in a text field)
  document.addEventListener('keydown',(e)=>{
    if(e.key==='Enter' && !['INPUT','SELECT','TEXTAREA'].includes(document.activeElement.tagName)){
      runAnalysis();
    }
  });
}

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
  };
  const ms = $('maxSamples').value;
  if(ms) p.max_samples = parseInt(ms);
  if(csvState.path){ p.csv_path = csvState.path; p.label_col = $('labelCol').value; }
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

async function clearRuns(){
  await fetch('/api/clear',{method:'POST'});
  toast('Stored runs cleared');
}

async function restoreLast(){
  try{
    const r = await fetch('/api/last'); const d = await r.json();
    if(d && d.ok){
      state.result = d;
      applyParamsToForm(d.params);
      renderAll(d);
      setConsole(d.console||'');
      setConsoleDot('active');
      toast('Restored your last run');
    }
  }catch(e){ /* first visit, nothing stored */ }
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
}

/* ============================================================
   RENDER
   ============================================================ */
function renderAll(d){
  $('emptyState').hidden = true;
  $('resultBody').hidden = false;
  // non-plot panels first so they always render even if Plotly is unavailable
  safeRender('statbar', ()=>renderStatbar(d));
  safeRender('signal', ()=>renderSignal(d));
  safeRender('table', ()=>renderTable(d));
  if(typeof Plotly === 'undefined'){
    plotlyUnavailable();
    return;
  }
  safeRender('projections', ()=>renderProjections(d));
  safeRender('comparison', ()=>renderComparison(d));
  safeRender('histograms', ()=>renderHistograms(d));
  safeRender('parallel', ()=>renderParallel(d));
  safeRender('brushing', ()=>renderBrushing(d));
}
function safeRender(label, fn){
  try{ fn(); }
  catch(e){ console.error(`render[${label}] failed:`, e); }
}
function plotlyUnavailable(){
  const msg = 'Plotting library did not load — check your network connection and reload. Tables and console output are available.';
  ['projPca','projMds','cmpPlot','histPca','histMds','parPlot','brushPca','brushMds'].forEach(id=>{
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
    const mean=err.reduce((a,b)=>a+b,0)/err.length;
    const bar={ type:'bar', x:h.centres, y:h.counts, marker:{color:h.colors,line:{color:'rgba(0,0,0,0.4)',width:0.4}},
      width:(edges[1]-edges[0])*0.92, hovertemplate:'error %{x:.3f}<br>%{y} pts<extra></extra>' };
    const maxc=Math.max(...h.counts,1);
    const meanLine={ type:'scatter',mode:'lines',x:[mean,mean],y:[0,maxc*1.05],
      line:{color:'#e8edf3',width:1.2,dash:'dot'},hoverinfo:'skip' };
    const zeroLine={ type:'scatter',mode:'lines',x:[0,0],y:[0,maxc*1.05],
      line:{color:'#63707f',width:1,dash:'dash'},hoverinfo:'skip' };
    Plotly.react(divId,[bar,zeroLine,meanLine], baseLayout({
      title:{text:`${label}  ·  mean ${mean>=0?'+':''}${mean.toFixed(3)}`,font:{size:12,color:'#9aa7b6'},x:0.02,xanchor:'left'},
      bargap:0.04,
      xaxis:Object.assign({title:'signed error (FN < 0 < FP)',range:[-vmax,vmax]},AXIS),
      yaxis:Object.assign({title:'count'},AXIS),
    }),PLOT_CONFIG);
  }
  build('histPca', d.table.pca.err, 'PCA');
  build('histMds', d.table.mds.err, 'MDS');
}

/* ---- parallel coordinates (mean error per class across methods) ---- */
function renderParallel(d){
  const methods=['PCA','MDS'];
  const labels=d.table.labels, names=d.table.class_names;
  const uniq=[...new Set(labels)].sort((a,b)=>a-b);
  const err={PCA:d.table.pca.err, MDS:d.table.mds.err};
  const traces=[];
  uniq.forEach(code=>{
    const idx=labels.map((l,i)=>l===code?i:-1).filter(i=>i>=0);
    const means=[], upper=[], lower=[];
    methods.forEach(m=>{
      const vals=idx.map(i=>err[m][i]);
      const mu=vals.reduce((a,b)=>a+b,0)/vals.length;
      const sd=Math.sqrt(vals.reduce((a,b)=>a+(b-mu)**2,0)/vals.length);
      means.push(mu); upper.push(mu+sd); lower.push(mu-sd);
    });
    const col=classColor(code);
    // sigma band (upper then reversed lower)
    traces.push({ type:'scatter',mode:'lines',x:[...methods,...methods.slice().reverse()],
      y:[...upper,...lower.slice().reverse()], fill:'toself',
      fillcolor:hexA(col,0.12), line:{width:0}, hoverinfo:'skip', showlegend:false });
    traces.push({ type:'scatter',mode:'lines+markers',x:methods,y:means,name:names[idx[0]],
      line:{color:col,width:2}, marker:{size:8,color:col,line:{color:'#0e1116',width:1}},
      hovertemplate:`%{x}<br>${names[idx[0]]} mean %{y:.3f}<extra></extra>` });
  });
  const zeroLine={ type:'scatter',mode:'lines',x:methods,y:[0,0],
    line:{color:'#33404f',width:1,dash:'dash'},hoverinfo:'skip',showlegend:false };
  Plotly.react('parPlot',[zeroLine,...traces], baseLayout({
    showlegend:true,
    legend:{ font:{size:10,color:'#9aa7b6'}, bgcolor:'rgba(21,26,33,0.7)',
             bordercolor:'#262f3b',borderwidth:1, orientation:'h', x:0.5, xanchor:'center', y:-0.12 },
    xaxis:Object.assign({title:'',showgrid:false},AXIS),
    yaxis:Object.assign({title:'mean signed error (FN < 0 < FP)',range:[-d.vmax,d.vmax],zeroline:true},AXIS),
  }),PLOT_CONFIG);
}
function hexA(hex,a){
  const n=parseInt(hex.slice(1),16);
  return `rgba(${(n>>16)&255},${(n>>8)&255},${n&255},${a})`;
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
   BRUSHING — linked rectangle selection
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
function renderBrushing(d){
  const vmax=d.vmax;
  state.brushSel = new Set();
  const layoutFor = (title)=> baseLayout({
    title:{text:title,font:{size:13,color:'#e8edf3'},x:0.02,xanchor:'left'},
    dragmode:'select',
    xaxis:Object.assign({title:'dim 1'},AXIS), yaxis:Object.assign({title:'dim 2'},AXIS),
  });
  Plotly.react('brushPca', [brushBaseTrace(d.table.pca,vmax,d), brushRingTrace()],
    layoutFor('PCA'), Object.assign({}, PLOT_CONFIG, {modeBarButtonsToAdd:['select2d','lasso2d']}));
  Plotly.react('brushMds', [brushBaseTrace(d.table.mds,vmax,d), brushRingTrace()],
    layoutFor('MDS'), Object.assign({}, PLOT_CONFIG, {modeBarButtonsToAdd:['select2d','lasso2d']}));

  // bind selection events (once per div; Plotly re-emits on re-render)
  bindBrush('brushPca'); bindBrush('brushMds');
  updateBrushDetail(d);
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
  const d=state.result; if(!d) return;
  const idx=[...state.brushSel];
  // ring the same points on BOTH plots (trace index 1 = ring layer)
  const ringPca={ x:idx.map(i=>d.table.pca.x[i]), y:idx.map(i=>d.table.pca.y[i]) };
  const ringMds={ x:idx.map(i=>d.table.mds.x[i]), y:idx.map(i=>d.table.mds.y[i]) };
  Plotly.restyle('brushPca', {x:[ringPca.x], y:[ringPca.y]}, [1]);
  Plotly.restyle('brushMds', {x:[ringMds.x], y:[ringMds.y]}, [1]);
  // reflect selection styling on both base traces
  Plotly.restyle('brushPca', {selectedpoints:[idx.length?idx:null]}, [0]);
  Plotly.restyle('brushMds', {selectedpoints:[idx.length?idx:null]}, [0]);
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
function bindBrushClear(){
  $('brushClear').addEventListener('click', ()=>{
    state.brushSel=new Set();
    if(state.result){
      Plotly.restyle('brushPca',{x:[[]],y:[[]]},[1]);
      Plotly.restyle('brushMds',{x:[[]],y:[[]]},[1]);
      Plotly.restyle('brushPca',{selectedpoints:[null]},[0]);
      Plotly.restyle('brushMds',{selectedpoints:[null]},[0]);
      updateBrushDetail(state.result);
    }
  });
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
window.runAnalysis = runAnalysis;

/* ---- go ---- */
init();
