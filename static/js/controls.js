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

  // CSV upload: click the zone or drop a file on it
  const csvInput = $('csvFile'), csvLabel = $('csvLabel');
  if(csvInput && csvLabel){
    csvInput.addEventListener('change', handleCsvUpload);
    ['dragover','dragenter'].forEach(ev=>csvLabel.addEventListener(ev,(e)=>{
      e.preventDefault(); csvLabel.classList.add('drop-target'); }));
    ['dragleave','dragend'].forEach(ev=>csvLabel.addEventListener(ev,(e)=>{
      e.preventDefault(); csvLabel.classList.remove('drop-target'); }));
    csvLabel.addEventListener('drop',(e)=>{
      e.preventDefault();
      csvLabel.classList.remove('drop-target');
      const f = e.dataTransfer.files[0];
      if(f){ csvInput.files = e.dataTransfer.files; handleCsvUpload(); }
    });
  }
  const csvClear = $('csvClear');
  if(csvClear) csvClear.addEventListener('click', clearCsv);

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

/* An uploaded CSV takes priority over the Dataset dropdown until it is
   cleared. The backend keeps the file and returns its column names so the
   label column can be picked here; every other column that is numeric becomes
   a feature. */
const CSV_PROMPT = 'or drop a CSV here';
let csvState = { path:null, name:null };

async function handleCsvUpload(){
  const input = $('csvFile');
  const f = input && input.files[0];
  if(!f) return;
  const fd = new FormData(); fd.append('file', f);
  $('csvText').textContent = 'uploading\u2026';
  try{
    const r = await fetch('/api/upload', { method:'POST', body:fd });
    const d = await r.json();
    if(!d.ok){
      toast(d.error || 'upload failed', true);
      $('csvText').textContent = CSV_PROMPT;
      return;
    }
    csvState.path = d.path;
    csvState.name = f.name;
    $('csvLabel').classList.add('loaded');
    $('csvText').textContent = f.name;

    const lc = $('labelCol');
    lc.innerHTML = '';
    d.columns.forEach(c=>{
      const o = document.createElement('option');
      o.value = c; o.textContent = c; lc.appendChild(o);
    });
    // Best guess: a column actually called "label"/"class"/"target"/"species".
    const guess = d.columns.find(c=>/^(label|class|target|species|y)$/i.test(c));
    if(guess) lc.value = guess;

    $('labelColField').hidden = false;
    $('csvClear').hidden = false;
    $('dataset').disabled = true;
    toast(guess ? `CSV loaded, label column "${guess}" \u2014 change it if wrong`
                : 'CSV loaded, pick a label column');
  }catch(e){
    toast('upload error', true);
    $('csvText').textContent = CSV_PROMPT;
  }
}

/* Drop the uploaded file and go back to the built-in datasets. */
function clearCsv(){
  csvState = { path:null, name:null };
  const input = $('csvFile'); if(input) input.value = '';
  $('csvLabel').classList.remove('loaded');
  $('csvText').textContent = CSV_PROMPT;
  $('labelColField').hidden = true;
  $('csvClear').hidden = true;
  $('dataset').disabled = false;
}

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
  if(csvState.path){ p.csv_path = csvState.path; p.label_col = $('labelCol').value; }
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
  // A stored run may have come from an uploaded CSV; re-establish that state so
  // pressing Run again reproduces it instead of silently reverting to Wine.
  if(p && p.csv_path){
    csvState = { path:p.csv_path, name:p.csv_path.split('/').pop() };
    $('csvLabel').classList.add('loaded');
    $('csvText').textContent = csvState.name;
    const lc = $('labelCol');
    if(p.label_col && !lc.querySelector(`option[value="${p.label_col}"]`)){
      const o = document.createElement('option');
      o.value = p.label_col; o.textContent = p.label_col; lc.appendChild(o);
    }
    if(p.label_col) lc.value = p.label_col;
    $('labelColField').hidden = false;
    $('csvClear').hidden = false;
    $('dataset').disabled = true;
  }
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