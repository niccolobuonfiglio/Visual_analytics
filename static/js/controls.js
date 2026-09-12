/* ======================================================================
   CONTROLS
   Dataset list, control-rail wiring, CSV upload, parameter collection,
   the run request, and restoring the last run on load.
   ====================================================================== */

/* verdict banner + modal button wiring */
function bindVerdictUi(){
  const openBtn = $('vbExamine');
  if(openBtn) openBtn.addEventListener('click', openVerdictModal);
  const close = $('vmClose'), done = $('vmDone'), overlay = $('verdictModal');
  if(close) close.addEventListener('click', closeVerdictModal);
  if(done) done.addEventListener('click', closeVerdictModal);
  if(overlay) overlay.addEventListener('click', (e)=>{ if(e.target===overlay) closeVerdictModal(); });
  document.addEventListener('keydown', (e)=>{ if(e.key==='Escape') closeVerdictModal(); });
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
  echo('untrustFloor','floorEcho',(v)=>parseFloat(v).toFixed(2));

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
    weight: 'rank',
    adaptive: $('adaptive') ? $('adaptive').checked : true,
    normalize: 'none',
    no_standardize: $('noStandardize').checked,
    threshold: parseFloat($('threshold').value),
    untrust_floor: parseFloat($('untrustFloor').value),
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
  ['trustPca','trustMds','trustBox','trustComparePca','trustCompareMds'].forEach(id => {
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
  
  // REMOVED: the Error Space table/quadrant reset and the signed-error balance
  // readout. Both targeted elements that no longer exist in index.html, and the
  // lookups were unguarded -- pressing "Clear results" threw here.

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
  $('adaptive').checked = p.adaptive!==false;
  $('noStandardize').checked = !!p.no_standardize;
  if(p.max_samples) $('maxSamples').value = p.max_samples;
}
