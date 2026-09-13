/* ======================================================================
   TRUST ANALYSIS: TABLES, DISTRIBUTIONS, FILTERS
   The lower half of the trust panel: the selection and outlier tables, the
   per-point score box plots, the metric-vs-metric comparison scatters, the
   filter column and the controls that bind them together.
   Loaded after trust.js, which owns the projection scatters and selection
   state these read from.
   ====================================================================== */

function renderFilteredPointsTable(d, activeSet) {
  const tbl = document.getElementById('trustSelTable');
  if (!tbl) return;
  
  const nm = d.neighbor_metrics;
  const key = state.trustMetric;
  const t = d.table;
  
  // Sortable columns
  const cols = [
    {k:'index', label:'index', get:i=>t.ids[i]},
    {k:'class', label:'class', get:i=>t.class_names[i]},
    {k:'pca',   label:'PCA',   get:i=>nm.PCA[key][i]},
    {k:'mds',   label:'MDS',   get:i=>nm.MDS[key][i]},
  ];
  
  const st = state.trustSelSort;
  const col = cols.find(c=>c.k===st.key) || cols[2];
  const idx = [...activeSet].sort((a,b)=>{
    let va = col.get(a), vb = col.get(b);
    if (typeof va === 'string') { return st.dir * va.localeCompare(vb); }
    return st.dir * (va - vb);
  });
  
  tbl.querySelector('thead').innerHTML = '<tr>' + cols.map(c=>{
    const cls = st.key===c.k ? (st.dir>0?'sorted-asc':'sorted-desc') : '';
    return `<th class="th-sort ${cls}" data-selsort="${c.k}">${c.label}</th>`;
  }).join('') + '</tr>';
  
  tbl.querySelector('tbody').innerHTML = idx.slice(0, 200).map(i=>{
    const code = t.labels[i];
    const cell = (v) => {
      const c = v < 0.34 ? 'cell-fn' : (v < 0.67 ? 'cell-zero' : 'cell-fp');
      return `<td class="${c}">${v.toFixed(3)}</td>`;
    };
    return `<tr>
      <td class="idx-click" data-pt="${i}">${t.ids[i]}</td>
      <td class="col-class"><span class="class-chip" style="background:${classColor(code)}"></span>${t.class_names[i]}</td>
      ${cell(nm.PCA[key][i])}
      ${cell(nm.MDS[key][i])}
    </tr>`;
  }).join('');
  
  // Header click -> sort
  tbl.querySelectorAll('th[data-selsort]').forEach(th=>{
    th.addEventListener('click', ()=>{
      const k = th.dataset.selsort;
      if (state.trustSelSort.key === k) state.trustSelSort.dir *= -1;
      else state.trustSelSort = {key:k, dir:1};
      // Re-render the filtered table
      renderFilteredPointsTable(d, activeSet);
    });
  });
  
  // Index click -> pin point
  tbl.querySelectorAll('.idx-click').forEach(td=>{
    td.addEventListener('click', ()=> pinPoint(parseInt(td.dataset.pt)));
  });
}

// selection detail table (same style as the error-space one)
function renderTrustSelTable(d){
  const sel = state.trustSel;
  const head = $('trustSelCount'), clr = $('trustSelClear'), wrap = $('trustSelWrap');
  if(!head) return;
  
  // Check if filters are active
  const active = trustActiveSet(d);
  const filtering = active.size !== d.table.ids.length;
  
  // If filters are active but no brush selection, show filtered points
  if (filtering && (!sel || sel.size === 0)) {
    // This is handled by updateFilterStatus
    return;
  }
  
  if(!sel || !sel.size){
    head.textContent = 'No points selected: drag a box or lasso on either plot.';
    if(clr) clr.hidden = true; 
    if(wrap) wrap.hidden = true; 
    return;
  }
  
  const nm=d.neighbor_metrics, key=state.trustMetric, t=d.table;
  head.innerHTML = `<span class="mono">${sel.size}</span> points selected, scores shown for the active metric (${METRIC_LABELS[key]||key}).`;
  if(clr) clr.hidden = false; 
  if(wrap) wrap.hidden = false;
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
  tbl.querySelectorAll('.idx-click').forEach(td=>{
    td.addEventListener('click', ()=> pinPoint(parseInt(td.dataset.pt)));
  });
}

const METRIC_LABELS = { f1:'neighbour F1', js_fidelity:'JS fidelity',
  localstress:'local stress', normstress:'norm. stress' };

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
  renderOutlierTable(d, idxArr, sp, sm, key, mlabel);
}

// Table of the box-plot outliers: each point that falls beyond a whisker fence
// in PCA and/or MDS, with its index, class, which method(s) it is an outlier
// for, and its value in that method. Clicking an index highlights the point in
// all its plots (same as the selection table).
function renderOutlierTable(d, idxArr, sp, sm, key, mlabel){
  const wrap=$('trustOutlierWrap'); if(!wrap) return;
  const nm=d.neighbor_metrics, t=d.table;
  // classify each active point against the two fences
  const rows=[];
  idxArr.forEach(i=>{
    const vp=nm.PCA[key][i], vm=nm.MDS[key][i];
    const pOut = sp && (vp < sp.loW || vp > sp.hiW);
    const mOut = sm && (vm < sm.loW || vm > sm.hiW);
    if(!pOut && !mOut) return;
    const where = pOut && mOut ? 'both' : (pOut ? 'PCA' : 'MDS');
    // the value shown is for the method(s) it's an outlier in (PCA if 'both')
    const val = pOut ? vp : vm;
    rows.push({ i, index:t.ids[i], cls:t.class_names[i], code:t.labels[i], where, val });
  });
  if(!rows.length){ wrap.hidden=true; return; }
  wrap.hidden=false;
  const cnt=$('trustOutlierCount'); if(cnt) cnt.textContent=`(${rows.length}) metric: ${mlabel}`;
  // sortable
  const stt=state.outlierSort;
  rows.sort((a,b)=>{
    let va=a[stt.key], vb=b[stt.key];
    if(typeof va==='string') return stt.dir*va.localeCompare(vb);
    return stt.dir*(va-vb);
  });
  const cols=[{k:'index',label:'index'},{k:'cls',label:'class'},{k:'where',label:'outlier in'},{k:'val',label:'value'}];
  const tbl=$('trustOutlierTable');
  tbl.querySelector('thead').innerHTML='<tr>'+cols.map(c=>{
    const cl=stt.key===c.k ? (stt.dir>0?'sorted-asc':'sorted-desc') : '';
    return `<th class="th-sort ${cl}" data-outsort="${c.k}">${c.label}</th>`;
  }).join('')+'</tr>';
  tbl.querySelector('tbody').innerHTML=rows.slice(0,200).map(r=>{
    const badge = r.where==='both' ? '<span class="out-badge out-both">both</span>'
      : r.where==='PCA' ? '<span class="out-badge out-pca">PCA</span>'
      : '<span class="out-badge out-mds">MDS</span>';
    return `<tr><td class="idx-click" data-pt="${r.i}">${r.index}</td>`+
      `<td class="col-class"><span class="class-chip" style="background:${classColor(r.code)}"></span>${r.cls}</td>`+
      `<td>${badge}</td><td class="mono">${r.val.toFixed(3)}</td></tr>`;
  }).join('');
  // header sort
  tbl.querySelectorAll('th[data-outsort]').forEach(th=>{
    th.addEventListener('click', ()=>{
      const k=th.dataset.outsort;
      if(state.outlierSort.key===k) state.outlierSort.dir*=-1;
      else state.outlierSort={key:k, dir:1};
      renderOutlierTable(d, idxArr, sp, sm, key, mlabel);
    });
  });
  // index click -> pin-highlight the point in all plots
  tbl.querySelectorAll('.idx-click').forEach(td=>{
    td.addEventListener('click', ()=> pinPoint(parseInt(td.dataset.pt)));
  });
}

function renderTrustCompare(d){
  if(typeof Plotly==='undefined') {
    console.warn('Plotly not available');
    return;
  }
  
  const nm = d.neighbor_metrics, t = d.table;
  if (!nm) {
    console.warn('No neighbor metrics available');
    return;
  }
  
  const active = trustActiveSet(d);
  if (!active || active.size === 0) {
    console.warn('No active points');
    return;
  }
  
  const xKey = state.cmpX || 'f1';
  const yKey = state.cmpY || 'js_fidelity';
  const xLab = METRIC_LABELS[xKey] || xKey;
  const yLab = METRIC_LABELS[yKey] || yKey;
  
  // ---- Calculate displacement for each point ----
  // Displacement = Euclidean distance between PCA and MDS positions
  const pcaX = nm.PCA[xKey];
  const pcaY = nm.PCA[yKey];
  const mdsX = nm.MDS[xKey];
  const mdsY = nm.MDS[yKey];
  
  // Check if data exists
  if (!pcaX || !pcaY || !mdsX || !mdsY) {
    console.warn('Missing metric data for displacement calculation');
    return;
  }
  
  // Calculate displacement for active points
  const rawDisplacement = [];
  const activeArr = [...active];
  const activeIndices = [];
  
  activeArr.forEach(i => {
    if (pcaX[i] !== undefined && pcaY[i] !== undefined && mdsX[i] !== undefined && mdsY[i] !== undefined) {
      const dx = mdsX[i] - pcaX[i];
      const dy = mdsY[i] - pcaY[i];
      const dist = Math.sqrt(dx*dx + dy*dy);
      rawDisplacement.push(dist);
      activeIndices.push(i);
    }
  });
  
  if (activeIndices.length === 0) {
    console.warn('No valid points for displacement calculation');
    return;
  }
  
  // Normalize displacement to [0, 1] based on the maximum
  const maxDisp = Math.max(0.001, ...rawDisplacement);
  const displacement = rawDisplacement.map(d => d / maxDisp);
  
  // Helper to create a trace for a given method
  function createTrace(method, xArr, yArr) {
    const xs = [], ys = [], cd = [];
    
    activeIndices.forEach((i, idx) => {
      xs.push(xArr[i]);
      ys.push(yArr[i]);
      cd.push([t.class_names[i], t.ids[i], i, displacement[idx], rawDisplacement[idx]]);
    });
    
    return {
      type: 'scattergl',
      mode: 'markers',
      x: xs,
      y: ys,
      customdata: cd,
      marker: {
        size: 7,
        color: displacement,
        // same five bands as everywhere else, reversed: here a LARGE value is bad
        colorscale: SCORE_SCALE.map(([pos, col], i, a) => [pos, a[a.length-1-i][1]]),
        cmin: 0,
        cmax: 1,
        showscale: true,
        colorbar: {
          title: { text: 'Metric Displacement', font: { size: 12, color: '#ffffff' } },
          thickness: 10,
          len: 0.9,
          tickvals: [0, 0.5, 1],
          ticktext: ['stable', 'moderate', 'unstable']
        },
        line: { color: 'rgba(0,0,0,0.4)', width: 0.4 }
      },
      hovertemplate: 
        'idx %{customdata[1]}, %{customdata[0]}<br>' +
        xLab + ' %{x:.3f}, ' + yLab + ' %{y:.3f}<br>' +
        'Displacement: %{customdata[3]:.3f}<extra></extra>'
    };
  }
  
  // Create traces for PCA and MDS
  const pcaTrace = createTrace('PCA', pcaX, pcaY);
  const mdsTrace = createTrace('MDS', mdsX, mdsY);
  
  // Diagonal line trace (shared)
  const diagTrace = {
    type: 'scatter',
    mode: 'lines',
    x: [0, 1],
    y: [0, 1],
    line: { color: '#63707f', width: 1, dash: 'dash' },
    hoverinfo: 'skip',
    showlegend: false
  };
  
  // Ring trace for highlighting (empty initially)
  const ringTrace = {
    type: 'scattergl',
    mode: 'markers',
    x: [],
    y: [],
    marker: {
      size: 15,
      color: 'rgba(0,0,0,0)',
      line: {
        color: '#e0a54b',
        width: 2.5
      }
    },
    hoverinfo: 'skip',
    showlegend: false
  };
  
  // Layout for the plots - square aspect ratio with equal ranges
  const compareLayout = (title) => baseLayout({
    title: { text: title, font: { size: 12, color: '#9aa7b6' }, x: 0.02, xanchor: 'left' },
    xaxis: Object.assign({ 
      title: xLab, 
      range: [-0.02, 1.02],
      scaleanchor: 'y',
      scaleratio: 1
    }, AXIS),
    yaxis: Object.assign({ 
      title: yLab, 
      range: [-0.02, 1.02] 
    }, AXIS),
    showlegend: false,
    margin: { l: 58, r: 18, t: 32, b: 46 },
    autosize: true,  
  });
  
  try {
    // Clear any existing plots
    const pcaEl = document.getElementById('trustComparePca');
    const mdsEl = document.getElementById('trustCompareMds');
    
    if (pcaEl) {
      Plotly.purge(pcaEl);
    }
    if (mdsEl) {
      Plotly.purge(mdsEl);
    }
    
    // Render PCA plot with ring trace
    Plotly.react('trustComparePca', [pcaTrace, ringTrace, diagTrace], compareLayout('PCA'), PLOT_CONFIG);
    
    // Render MDS plot with ring trace
    Plotly.react('trustCompareMds', [mdsTrace, ringTrace, diagTrace], compareLayout('MDS'), PLOT_CONFIG);
    
    console.log('Trust compare plots rendered successfully');
  } catch(e) {
    console.error('Error rendering trust compare plots:', e);
    // Show error message in the plot containers
    const pcaEl = document.getElementById('trustComparePca');
    const mdsEl = document.getElementById('trustCompareMds');
    if (pcaEl) pcaEl.innerHTML = '<div class="plot-fallback">Error loading PCA comparison plot. See console for details.</div>';
    if (mdsEl) mdsEl.innerHTML = '<div class="plot-fallback">Error loading MDS comparison plot. See console for details.</div>';
  }
  
  // --- LINKED HOVERING ---
  // Store the data for each method to look up points
  const methodData = {
    PCA: { x: pcaX, y: pcaY },
    MDS: { x: mdsX, y: mdsY }
  };
  const displacementMap = {};
  activeIndices.forEach((i, idx) => {
    displacementMap[i] = displacement[idx];
  });
  const rawDisplacementMap = {};
  activeIndices.forEach((i, idx) => {
    rawDisplacementMap[i] = rawDisplacement[idx];
  });
  
  // Function to highlight a point in both plots
  function highlightPoint(idx, sourcePlot) {
    const ro = document.getElementById('trustCompareHover');
    
    if (idx === null || idx === undefined) {
      // Clear highlighting
      ['trustComparePca', 'trustCompareMds'].forEach(id => {
        const el = document.getElementById(id);
        if (el && el.data) {
          try {
            Plotly.restyle(id, {
              x: [[]],
              y: [[]],
              'marker.line.color': [[]]
            }, [1]);
          } catch(e) {}
        }
      });
      if (ro) ro.innerHTML = 'Hover a point to see it highlighted in both plots with its values.';
      return;
    }
    
    // Check if the index is in the active set
    if (!activeIndices.includes(idx)) {
      return;
    }
    
    // Get the point data for both methods
    const pcaXval = methodData.PCA.x[idx];
    const pcaYval = methodData.PCA.y[idx];
    const mdsXval = methodData.MDS.x[idx];
    const mdsYval = methodData.MDS.y[idx];
    
    // Update the ring traces
    try {
      Plotly.restyle('trustComparePca', {
        x: [[pcaXval]],
        y: [[pcaYval]],
        'marker.line.color': [['#e0a54b']]
      }, [1]);
      
      Plotly.restyle('trustCompareMds', {
        x: [[mdsXval]],
        y: [[mdsYval]],
        'marker.line.color': [['#e0a54b']]
      }, [1]);
    } catch(e) {}
    
    // Update hover readout
    if (ro) {
      const cls = t.class_names[idx];
      const id = t.ids[idx];
      
      const pcaVals = {
        f1: nm.PCA.f1[idx],
        js: nm.PCA.js_fidelity[idx],
        ls: nm.PCA.localstress[idx],
        ns: nm.PCA.normstress[idx]
      };
      const mdsVals = {
        f1: nm.MDS.f1[idx],
        js: nm.MDS.js_fidelity[idx],
        ls: nm.MDS.localstress[idx],
        ns: nm.MDS.normstress[idx]
      };
      
      // Calculate displacement
      const dx = mdsXval - pcaXval;
      const dy = mdsYval - pcaYval;
      const dist = Math.sqrt(dx*dx + dy*dy);
      const normDisp = displacementMap[idx] || 0;
      
      // Determine stability level
      let stability = 'stable';
      let stabilityColor = '#4ec97a';
      if (normDisp > 0.7) {
        stability = 'unstable';
        stabilityColor = '#d24b52';
      } else if (normDisp > 0.3) {
        stability = 'moderate';
        stabilityColor = '#e0a54b';
      }
      
      ro.innerHTML = `
        <span class="hr-idx">idx ${id}</span> 
        <span class="class-chip" style="background:${classColor(t.labels[idx])}"></span>
        <span style="font-weight:600; color:var(--ink);">${cls}</span>
        <span style="color:var(--ink); margin-left:12px;">
          PCA (${pcaXval.toFixed(3)}, ${pcaYval.toFixed(3)}) → 
          MDS (${mdsXval.toFixed(3)}, ${mdsYval.toFixed(3)})
        </span>
        <span style="display:block; margin-top:4px; font-size:16px;">
          <span style="color:${stabilityColor};">${stability.toUpperCase()}</span>
          <span style="color:var(--ink);"> displacement: ${dist.toFixed(3)} (${(normDisp * 100).toFixed(0)}% of max)</span>
        </span>
      `;
    }
  }
  
  // Attach hover events to both plots
  ['trustComparePca', 'trustCompareMds'].forEach(plotId => {
    const el = document.getElementById(plotId);
    if (el && el.on) {
      // Remove existing listeners to avoid duplicates
      if (el.removeAllListeners) {
        el.removeAllListeners('plotly_hover');
        el.removeAllListeners('plotly_unhover');
      }
      
      el.on('plotly_hover', (ev) => {
        if (!ev || !ev.points || !ev.points.length) return;
        const p = ev.points.find(pt => pt.curveNumber === 0);
        if (!p) return;
        const idx = p.customdata ? p.customdata[2] : p.pointIndex;
        highlightPoint(idx, plotId);
      });
      
      el.on('plotly_unhover', () => {
        highlightPoint(null);
      });
    }
  });
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

  // SCORE filter: two separate dual ranges - one for PCA, one for MDS
  const sc = $('tfScore');
  if(sc && sc.dataset.metric!==state.trustMetric){
    sc.dataset.metric=state.trustMetric;
    state.scoreFilterPca=null;
    state.scoreFilterMds=null;
    
    // Clear and rebuild
    sc.innerHTML = `
      <div class="tf-score-method">
        <span style="font-size: 12px; color: #82abd8; font-weight: 600;">PCA</span>
        <span id="tfScorePca"></span>
      </div>
      <div class="tf-score-method">
        <span style="font-size: 12px; color: #e0a54b; font-weight: 600;">MDS</span>
        <span id="tfScoreMds"></span>
      </div>
    `;
    
    // Create PCA slider
    const pcaContainer = document.getElementById('tfScorePca');
    if (pcaContainer) {
      pcaContainer.appendChild(makeDualRange(0, 1, (a,b) => {
        state.scoreFilterPca = (a<=0 && b>=1) ? null : {lo:a, hi:b};
        refreshTrust(d);
      }));
    }
    
    // Create MDS slider
    const mdsContainer = document.getElementById('tfScoreMds');
    if (mdsContainer) {
      mdsContainer.appendChild(makeDualRange(0, 1, (a,b) => {
        state.scoreFilterMds = (a<=0 && b>=1) ? null : {lo:a, hi:b};
        refreshTrust(d);
      }));
    }
    
    const lbl=$('tfScoreMetric'); 
    if(lbl) lbl.textContent='('+(METRIC_LABELS[state.trustMetric]||state.trustMetric)+')';
  }

  // Disagreement filter
  const dis = $('tfDisagreement');
  if(dis && !dis.dataset.built){
    dis.dataset.built = '1';
    dis.addEventListener('change', function() {
      state.filterDisagreement = this.value || null;
      refreshTrust(d);
    });
  }

  // ATTRIBUTE window: feature dropdown + a clean dual range on that feature
  const attr=$('tfAttr');
  if(attr && attr.dataset.built!==t.feature_names.join(',')){
    attr.dataset.built=t.feature_names.join(',');
    const opts=t.feature_names.map((nm,j)=>`<option value="${j}">${nm}</option>`).join('');
    attr.innerHTML=`<select id="tfAttrSel" class="inp inp-sm"><option value="">-- feature --</option>${opts}</select>`+
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
  if(clr && !clr.dataset.wired){ 
    clr.dataset.wired='1';
    clr.addEventListener('click',()=>{
      state.filterClasses.clear(); 
      state.attrFilter=null; 
      state.scoreFilterPca=null;
      state.scoreFilterMds=null;
      state.trustSel=null; 
      state.filterDisagreement = null;
      document.querySelectorAll('#tfClasses .tf-chip').forEach(b=>b.classList.remove('active'));
      const sel=$('tfAttrSel'); if(sel){ sel.value=''; } 
      const ar=$('tfAttrRange'); if(ar) ar.innerHTML='';
      const dis = $('tfDisagreement');
      if(dis) dis.value = '';
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
  const note = $('tfNote'); if(!note) return;
  const active = trustActiveSet(d), n = d.table.ids.length;
  const parts = [];
  
  if(state.filterClasses.size) parts.push(`${state.filterClasses.size} class${state.filterClasses.size>1?'es':''}`);
  
  if(state.scoreFilterPca) {
    parts.push(`PCA ${METRIC_LABELS[state.trustMetric]||state.trustMetric} ∈ [${state.scoreFilterPca.lo.toFixed(2)}, ${state.scoreFilterPca.hi.toFixed(2)}]`);
  }
  if(state.scoreFilterMds) {
    parts.push(`MDS ${METRIC_LABELS[state.trustMetric]||state.trustMetric} ∈ [${state.scoreFilterMds.lo.toFixed(2)}, ${state.scoreFilterMds.hi.toFixed(2)}]`);
  }
  
  if(state.attrFilter) parts.push(`${d.table.feature_names[state.attrFilter.col]} ∈ [${state.attrFilter.lo.toFixed(2)}, ${state.attrFilter.hi.toFixed(2)}]`);
  if(state.trustSel && state.trustSel.size) parts.push('a brushed region');
  
  // Disagreement filter note
  if(state.filterDisagreement) {
    const labels = {
      'large': 'large disagreement (>0.2)',
      'medium': 'medium disagreement (0.1-0.2)',
      'small': 'small disagreement (<0.05)',
      'pca_better': 'PCA better than MDS',
      'mds_better': 'MDS better than PCA'
    };
    parts.push(labels[state.filterDisagreement] || state.filterDisagreement);
  }
  
  note.textContent = parts.length
    ? `Active filters: ${parts.join(', ')} → ${active.size} of ${n} points.`
    : `No filters, all ${n} points. Brush a projection, pick classes, or set a score / attribute window to focus.`;
}

function refreshTrust(d){
  renderTrustProjections(d); renderTrustBox(d); renderTrustCompare(d); renderUntrust(d); updateTrustFilterNote(d); updateFilterStatus(d);
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
  const pclr=$('trustPlotClear');
  if(pclr) pclr.addEventListener('click', ()=> clearTrustSelection());
}

// Fully reset the trust plots to their normal state: no selection, no lasso
// box, uniform marker size, no dimming.
function clearTrustSelection(){
  state.trustSel = null;
  state.pinned = null;
  if(typeof Plotly!=='undefined'){
    ['trustPca','trustMds','untrustPca','untrustMds'].forEach(id=>{
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
    renderTrustBox(state.result); 
    renderTrustCompare(state.result);
    renderTrustSelTable(state.result); 
    updateTrustFilterNote(state.result);
    updateFilterStatus(state.result);  // This will now show filtered points if filters are active
  }
  document.querySelectorAll('.idx-click').forEach(td=>td.classList.remove('idx-pinned'));
  updateTrustPlotClearBtn();
}

// show the scatter-plot clear button only when there's a selection or a pin
function updateTrustPlotClearBtn(){
  const btn=$('trustPlotClear'); if(!btn) return;
  const has = (state.trustSel && state.trustSel.size) || state.pinned!=null;
  btn.hidden = !has;
}