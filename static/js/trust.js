/* ======================================================================
   TRUST ANALYSIS
   The main panel: linked projection scatters, brushing and hover,
   filters, distribution box plots, metric comparison and the
   selection tables.
   ====================================================================== */

/* ============================================================
   TRUST ANALYSIS: per-point neighbourhood-preservation F1 / JS
   ============================================================ */

// which point indices pass the current class + attribute filters (not brushing)
function trustFilteredIndices(d){
  const t = d.table, n = t.ids.length;
  const out = [];
  const fc = state.filterClasses, af = state.attrFilter;
  const sfPca = state.scoreFilterPca, sfMds = state.scoreFilterMds;
  const nm = d.neighbor_metrics, key = state.trustMetric;
  const fd = state.filterDisagreement;
  
  for(let i = 0; i < n; i++){
    // Class filter
    if(fc.size && !fc.has(t.labels[i])) continue;
    
    // Attribute filter
    if(af){
      const v = t.feature_values[i][af.col];
      if(v < af.lo || v > af.hi) continue;
    }
    
    // Score filter - separate PCA and MDS
    if((sfPca || sfMds) && nm){
      const sp = nm.PCA[key][i];
      const sm = nm.MDS[key][i];
      
      // Check PCA filter
      if(sfPca) {
        if(sp < sfPca.lo || sp > sfPca.hi) continue;
      }
      
      // Check MDS filter
      if(sfMds) {
        if(sm < sfMds.lo || sm > sfMds.hi) continue;
      }
    }
    
    // Disagreement filter
    if(fd && nm){
      const sp = nm.PCA[key][i];
      const sm = nm.MDS[key][i];
      const diff = Math.abs(sp - sm);
      
      switch(fd){
        case 'large':
          if(diff <= 0.2) continue;
          break;
        case 'medium':
          if(diff <= 0.1 || diff >= 0.2) continue;
          break;
        case 'small':
          if(diff >= 0.05) continue;
          break;
        case 'pca_better':
          if(sp <= sm) continue;
          break;
        case 'mds_better':
          if(sm <= sp) continue;
          break;
        default:
          break;
      }
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
  renderUntrust(d);
  updateTrustFilterNote(d); 
  updateFilterStatus(d);
}

const F1_SCALE = [[0,'#d24b52'],[0.5,'#e0a54b'],[1,'#4ec97a']];  // red→amber→green

// the two projections coloured by the chosen per-point score
const F1_SCALE_LOCAL = [[0,'#d24b52'],[0.5,'#e0a54b'],[1,'#4ec97a']];
const METRIC_KEYS = ['f1','js_fidelity','localstress','normstress'];

function renderTrustProjections(d){
  if(typeof Plotly==='undefined') return;
  const nm = d.neighbor_metrics, key = state.trustMetric;
  const t = d.table, ids = t.ids;
  const active = trustActiveSet(d);
  const filtering = active.size !== ids.length;
  const classView = state.trustView === 'class';

  // Main marker - size stays the same (no size change)
  const baseMarker = (projScores) => {
    const colors = classView 
      ? (t.labels||[]).map(l => classColor(l))
      : projScores;
    
    return {
      size: 7,  // always the same size
      color: ids.map((_, i) => {
        if (filtering && !active.has(i)) {
          if (classView) {
            const c = colors[i];
            return c + '33';
          }
          return '#2a3340';
        }
        return colors[i];
      }),
      opacity: ids.map((_, i) => {
        if (filtering && !active.has(i)) return 0.25;
        return 1;
      }),
      line: {
        color: 'rgba(0,0,0,0.4)',
        width: 0.4
      },
      colorscale: classView ? undefined : F1_SCALE_LOCAL,
      cmin: classView ? undefined : 0,
      cmax: classView ? undefined : 1,
      colorbar: classView ? undefined : {
        title: { text: METRIC_LABELS[key] || key, font: { size: 9, color: '#9aa7b6' } },
        thickness: 10,
        len: 0.9
      }
    };
  };

  const mk = (proj, scores) => ({
    type:'scattergl', mode:'markers', x:proj.x, y:proj.y,
    marker: baseMarker(scores),
    customdata: ids.map((id,i)=>[t.class_names[i], id, scores[i]]),
    hovertemplate:'idx %{customdata[1]}, %{customdata[0]}<br>'+(METRIC_LABELS[key]||key)+' %{customdata[2]:.3f}<extra></extra>',
    selected:{marker:{opacity:1}}, 
    unselected:{marker:{opacity: filtering?0.12:1}},
  });

  // Ring trace - large ring, transparent fill
  const ringTrace = (proj) => ({
    type: 'scattergl',
    mode: 'markers',
    x: filtering ? [...active].map(i => proj.x[i]) : [],
    y: filtering ? [...active].map(i => proj.y[i]) : [],
    marker: {
      size: 16,  // large ring
      color: 'rgba(0,0,0,0)',  // transparent fill
      line: {
        color: '#e8edf3',  // white ring
        width: 2.5
      }
    },
    hoverinfo: 'skip',
    showlegend: false
  });

  const lay=(title)=>baseLayout({
    title:{text:title,font:{size:12,color:'#e8edf3'},x:0.03,xanchor:'left'},
    xaxis:Object.assign({title:'dim 1'},AXIS), yaxis:Object.assign({title:'dim 2'},AXIS),
    showlegend:false, height:320, dragmode:'lasso',
  });
  const cfg = Object.assign({}, PLOT_CONFIG, {modeBarButtonsToAdd:['select2d','lasso2d']});
  
  // Render the plots
  Plotly.react('trustPca', [mk(t.pca, nm.PCA[key]), ringTrace(t.pca)], lay('PCA'), cfg);
  Plotly.react('trustMds', [mk(t.mds, nm.MDS[key]), ringTrace(t.mds)], lay('MDS'), cfg);
  
  // reflect current selection as rings
  if((state.trustSel && state.trustSel.size) || state.pinned!=null) applyTrustRings(d, state.trustSel || new Set());

  // --- DEFINE AND ATTACH EVENT HANDLERS ---
  const pcaEl=$('trustPca'), mdsEl=$('trustMds');
  
  const onSel = (ev)=>{
    if(!ev || !ev.points){ return; }
    const idx = new Set(ev.points.filter(p=>p.curveNumber===0).map(p=>p.pointIndex));
    state.trustSel = idx.size ? idx : null;
    applyTrustRings(state.result, state.trustSel || new Set());
    renderTrustBox(state.result); 
    renderTrustCompare(state.result);
    renderTrustSelTable(state.result); 
    updateTrustFilterNote(state.result);
    updateTrustPlotClearBtn();
    updateFilterStatus(state.result);
  };
  
  const onDesel = ()=>{ 
    clearTrustSelection(); 
  };
  
  const onHover = (ev)=>{ 
    if(!ev||!ev.points||!ev.points.length) return;
    const p=ev.points.find(pt=>pt.curveNumber===0); if(!p) return;
    const i=p.pointIndex;
    // Save current selection and add hover point
    const currentSel = state.trustSel || new Set();
    const ringSet = new Set(currentSel);
    ringSet.add(i);
    applyTrustRings(state.result, ringSet, /*hover*/true);
    
    // Show the point's stats
    const ro=$('trustHover'); const r=state.result;
    if(ro && r){ const nmr=r.neighbor_metrics, tt=r.table;
      const fmt=(v)=>v.toFixed(3);
      const metricCells=(mth)=>METRIC_KEYS.map(k=>`${METRIC_LABELS[k]||k} <strong>${fmt(nmr[mth][k][i])}</strong>`).join(', ');
      const kUsed = (nmr.MDS.k_used && nmr.MDS.k_used[i]!=null) ? nmr.MDS.k_used[i] : null;
      ro.innerHTML =
        `<span class="hr-idx">idx ${tt.ids[i]}</span> <span class="class-chip" style="background:${classColor(tt.labels[i])}"></span>${tt.class_names[i]}`+
        (kUsed!=null ? ` <span class="hr-k">k = ${kUsed}</span>` : '')+
        `<span class="hr-row"><span class="hr-m">PCA</span> ${metricCells('PCA')}</span>`+
        `<span class="hr-row"><span class="hr-m">MDS</span> ${metricCells('MDS')}</span>`;
    }
  };
  
  const onUnhover = ()=>{ 
    // Restore the filter selection
    const sel = state.trustSel || new Set();
    applyTrustRings(state.result, sel);
    const ro=$('trustHover'); 
    if(ro) ro.innerHTML='Hover a point to see its scores in both projections; it is highlighted in the other plot too.'; 
  };

  // Attach event handlers to both plots
  [pcaEl, mdsEl].forEach(el=>{
    if(el && el.on){
      // Remove old listeners to avoid duplicates
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
  const t=d.table;
  const sel=[...idxSet];
  const pin = state.pinned;
  const baseCol = hover ? '#e0a54b' : '#e8edf3';
  const idxList = sel.slice();
  const colors = sel.map(()=>baseCol);
  if(pin!=null && !idxList.includes(pin)){ idxList.push(pin); colors.push('#4ec97a'); }
  else if(pin!=null){ colors[idxList.indexOf(pin)] = '#4ec97a'; }
  
  ['trustPca','trustMds','untrustPca','untrustMds'].forEach(id=>{
    const el=$(id); if(!el||!el.data) return;
    const xy = id.endsWith('Pca') ? t.pca : t.mds;
    // Find which trace index is the ring trace (should be index 1)
    const ringIdx = 1; // The ring trace is the second trace
    Plotly.restyle(id, { 
      x: [idxList.map(i=>xy.x[i])], 
      y: [idxList.map(i=>xy.y[i])],
      'marker.line.color': [colors] 
    }, ringIdx);
  });
}

// Pin a single point: highlight it with a green ring in ALL its plots (the two
// trust scatters, and the plain/adjusted MDS plots if visible). Clicking the
// same index again un-pins it.
function pinPoint(i){
  state.pinned = (state.pinned === i) ? null : i;
  // trust scatters
  applyTrustRings(state.result, state.trustSel || new Set());
  // reflect pinned row styling in any visible table
  document.querySelectorAll('.idx-click').forEach(td=>{
    td.classList.toggle('idx-pinned', parseInt(td.dataset.pt)===state.pinned);
  });
  updateTrustPlotClearBtn();
}

function updateFilterStatus(d) {
  const status = document.getElementById('filterStatus');
  const countEl = document.getElementById('filterCountNum');
  const pcaAvgEl = document.getElementById('filterAvgPca');
  const mdsAvgEl = document.getElementById('filterAvgMds');
  const metricLabel = document.getElementById('filterMetricLabel');
  const clearBtn = document.getElementById('filterStatusClear');
  
  if (!status || !countEl || !pcaAvgEl || !mdsAvgEl) return;
  
  const nm = d.neighbor_metrics;
  const key = state.trustMetric;
  const active = trustActiveSet(d);
  const total = d.table.ids.length;
  const filtering = active.size !== total;
  
  // Update metric label
  if (metricLabel) {
    metricLabel.textContent = `(${METRIC_LABELS[key] || key})`;
  }
  
  // Calculate averages for PCA and MDS
  const pcaScores = nm.PCA[key];
  const mdsScores = nm.MDS[key];
  
  let pcaAvg, mdsAvg;
  if (filtering && active.size > 0) {
    const activeArr = [...active];
    let pcaSum = 0, mdsSum = 0;
    activeArr.forEach(i => {
      pcaSum += pcaScores[i];
      mdsSum += mdsScores[i];
    });
    pcaAvg = pcaSum / activeArr.length;
    mdsAvg = mdsSum / activeArr.length;
    countEl.textContent = active.size;
  } else if (filtering && active.size === 0) {
    pcaAvg = 0;
    mdsAvg = 0;
    countEl.textContent = 'No';
  } else {
    pcaAvg = pcaScores.reduce((a,b) => a + b, 0) / pcaScores.length;
    mdsAvg = mdsScores.reduce((a,b) => a + b, 0) / mdsScores.length;
    countEl.textContent = 'All';
  }
  
  // Update the display
  pcaAvgEl.textContent = pcaAvg.toFixed(3);
  mdsAvgEl.textContent = mdsAvg.toFixed(3);
  
  // Show/hide clear button and add class
  if (filtering) {
    status.classList.add('has-filters');
    if (clearBtn) clearBtn.style.display = 'inline-block';
  } else {
    status.classList.remove('has-filters');
    if (clearBtn) clearBtn.style.display = 'none';
  }
  
  // --- SHOW THE SELECTION TABLE WHEN FILTERS ARE ACTIVE ---
  // If filters are active but no brush selection, show the filtered points in the table
  const sel = state.trustSel;
  if (filtering && (!sel || sel.size === 0)) {
    // Show the table with filtered points
    const head = document.getElementById('trustSelCount');
    const clr = document.getElementById('trustSelClear');
    const wrap = document.getElementById('trustSelWrap');
    
    if (head) {
      head.innerHTML = `<span class="mono">${active.size}</span> points match the current filters (click to pin)`;
    }
    if (clr) clr.hidden = false;
    if (wrap) wrap.hidden = false;
    
    // Render the filtered points table
    renderFilteredPointsTable(d, active);
  } else if (!filtering && (!sel || sel.size === 0)) {
    // No filters and no selection - hide the table
    const head = document.getElementById('trustSelCount');
    const clr = document.getElementById('trustSelClear');
    const wrap = document.getElementById('trustSelWrap');
    if (head) head.textContent = 'No points selected: drag a box or lasso on either plot.';
    if (clr) clr.hidden = true;
    if (wrap) wrap.hidden = true;
  }
  
  // Clear button handler
  if (clearBtn && !clearBtn.dataset.wired) {
    clearBtn.dataset.wired = '1';
    clearBtn.onclick = function() {
      state.filterClasses.clear();
      state.attrFilter = null;
      state.scoreFilterPca = null;
      state.scoreFilterMds = null;
      state.filterDisagreement = null;
      state.trustSel = null;
      
      // Reset UI elements
      document.querySelectorAll('#tfClasses .tf-chip').forEach(b => b.classList.remove('active'));
      const sel = document.getElementById('tfAttrSel');
      if (sel) sel.value = '';
      const ar = document.getElementById('tfAttrRange');
      if (ar) ar.innerHTML = '';
      const dis = document.getElementById('tfDisagreement');
      if (dis) dis.value = '';
      
      // Rebuild and refresh
      buildTrustFilters(d);
      refreshTrust(d);
    };
  }
}

// New function to render the filtered points table