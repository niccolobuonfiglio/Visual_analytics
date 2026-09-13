/* ======================================================================
   RENDER
   Top-level render fan-out, summary bar, degeneracy warnings, the
   untrustworthy-points table, and the adaptive-k indicator.
   ====================================================================== */

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
  ['trustPca','trustMds','trustBox','trustComparePca','trustCompareMds'].forEach(id=>{
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
    pill('Dataset', info.name, '', `${info.standardized?'standardized':'raw'}`) +
    pill('Samples', info.n_samples, '', `${info.n_features} features | ${info.n_classes} classes`) +
    (d.timings ? pill('Left projection time', (d.timings.pca_ms).toFixed(0)+' ms', '', 'computed') : '') +
    (d.timings ? pill('Right projection time', (d.timings.mds_ms).toFixed(0)+' ms', signCls(d.timings.pca_ms - d.timings.mds_ms), 'computed') : '') +
    pill('Untrustworthy', s.untrustworthy_count, '', d.untrust_desc || 'consensus fidelity') +
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


/* data table (sortable) */
const TABLE_COLS = [
  { key:'index', label:'index' },
  { key:'true_class', label:'class', cls:'col-class' },
  { key:'consensus', label:'consensus', score:true },
  { key:'pca_consensus', label:'PCA', score:true },
  { key:'mds_consensus', label:'MDS', score:true },
  { key:'which', label:'untrustworthy in', badge:true },
  { key:'agree', label:'metrics agree', plain:true },
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
  
  // Build the table with the correct array index for pinning
  // We need to find the actual array index for each row
  const allIds = d.table.ids;  // Array of all dataset IDs in original order
  
  tbody.innerHTML = rows.map(r=> {
    // Find the array index for this dataset ID
    const arrayIndex = allIds.indexOf(r.index);
    // If not found (shouldn't happen), fallback to using r.index
    const pinIndex = arrayIndex !== -1 ? arrayIndex : r.index;
    
    return '<tr>' + TABLE_COLS.map(c=>{
      let v = r[c.key];
      if(c.cls==='col-class'){
        const code = d.target_names.indexOf(v);
        return `<td class="col-class"><span class="class-chip" style="background:${classColor(code<0?0:code)}"></span>${v}</td>`;
      }
      if(c.score){
        const t = Math.max(0, Math.min(1, v));
        const cls = t<0.34?'cell-fn':(t<0.67?'cell-zero':'cell-fp');
        return `<td class="${cls}">${v.toFixed(3)}</td>`;
      }
      if(c.badge){
        const b = v==='both' ? '<span class="out-badge out-both">both</span>'
          : v==='PCA' ? '<span class="out-badge out-pca">PCA</span>'
          : '<span class="out-badge out-mds">MDS</span>';
        return `<td>${b}</td>`;
      }
      if(c.plain){ return `<td class="mono">${v}</td>`; }
      if(c.err){
        const cls = v<0?'cell-fn':(v>0?'cell-fp':'cell-zero');
        return `<td class="${cls}">${v>=0?'+':''}${v.toFixed(3)}</td>`;
      }
      if(c.part){
        const cls = v>0 ? (c.part==='fp'?'cell-fp':'cell-fn') : 'cell-zero';
        return `<td class="${cls}">${v.toFixed(3)}</td>`;
      }
      // Make index column clickable with the correct array index
      if(c.key === 'index'){
        return `<td class="idx-click" data-pt="${pinIndex}">${v}</td>`;
      }
      return `<td>${v}</td>`;
    }).join('') + '</tr>';
  }).join('');

  thead.querySelectorAll('th').forEach(th=>{
    th.addEventListener('click', ()=>{
      const k=th.dataset.key;
      if(state.tableSort.key===k) state.tableSort.dir*=-1;
      else state.tableSort={key:k,dir:-1};
      renderTable(d);
    });
  });
  
  // Click handlers for index cells
  tbody.querySelectorAll('.idx-click').forEach(td=>{
    td.addEventListener('click', ()=> {
      const idx = parseInt(td.dataset.pt);
      pinPoint(idx);
    });
  });
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
      `min <strong>${akd.k_min}</strong>, median <strong>${akd.k_median}</strong>, max <strong>${akd.k_max}</strong>`;
  }
}
