/* ====================================================================
   QUARANTINED -- NOT LOADED BY index.html
   ====================================================================
   Plain-vs-adjusted MDS comparison and the per-feature weight sliders.
   The backend no longer computes mds_compare.

   This file is kept for reference only. It is not referenced by any <script>
   tag, so it costs nothing at runtime. Re-adding it requires restoring both
   the markup it targets and the backend fields it reads.
   ==================================================================== */

function renderFeatWeights(d){
  const box=$('featWeights'); if(!box) return;
  const mc=d.mds_compare;
  const method = (mc && mc.adj_method) || (d.params && d.params.adj_method) || state.adjMethod;
  const isLocality = method==='locality';
  if(!isLocality || !mc || !mc.feature_weights){ box.hidden=true; box.innerHTML=''; return; }
  box.hidden=false;
  const w=mc.feature_weights, names=(d.table && d.table.feature_names)||w.map((_,j)=>'f'+j);
  const ov=state.featWeightOverride||{};
  let rows = w.map((val,j)=>{
    const mult = ov[j]!=null ? ov[j] : 1;
    return `<div class="fw-row"><span class="fw-name">${names[j]}</span>`+
      `<input type="range" class="fw-slider" data-fw="${j}" min="0" max="3" step="0.1" value="${mult}">`+
      `<span class="fw-mult" id="fwmult-${j}">×${(+mult).toFixed(1)}</span>`+
      `<span class="fw-eff">w=${(+val).toFixed(2)}</span></div>`;
  }).join('');
  box.innerHTML =
    `<div class="fw-head">Feature weights <span class="an-subtle">drag to boost/suppress a feature, then apply</span></div>`+
    `<div class="fw-list">${rows}</div>`+
    `<div class="fw-actions"><button class="ghost-btn ghost-btn-sm" id="fwReset">Reset weights</button>`+
    `<button class="run-btn run-btn-sm" id="fwApply">Apply and; re-run</button></div>`;
  box.querySelectorAll('.fw-slider').forEach(s=>{
    s.addEventListener('input', ()=>{ const el=$(`fwmult-${s.dataset.fw}`); if(el) el.textContent='×'+parseFloat(s.value).toFixed(1); });
  });
  const apply=$('fwApply');
  if(apply) apply.addEventListener('click', ()=>{
    const o={}; box.querySelectorAll('.fw-slider').forEach(s=>{ const m=parseFloat(s.value); if(Math.abs(m-1)>1e-6) o[s.dataset.fw]=m; });
    state.featWeightOverride=o; if(state.result) runAnalysis();
  });
  const rst=$('fwReset');
  if(rst) rst.addEventListener('click', ()=>{ state.featWeightOverride={}; if(state.result) runAnalysis(); });
}

function renderWmdsCompare(d){
  const box = $('wmdsCompare');
  const mc = d.mds_compare;
  // sync the method selector to the run that produced this
  const runMethod = (mc && mc.adj_method) || (d.params && d.params.adj_method);
  if(runMethod){
    const ams=$('adjMethodSeg');
    if(ams) ams.querySelectorAll('.seg-btn').forEach(x=>x.classList.toggle('active', x.dataset.am===runMethod));
  }
  renderFeatWeights(d);
  if(!mc){ if(box) box.hidden = true; if(typeof Plotly!=='undefined'){ ['wmdsPlain','wmdsWeighted'].forEach(id=>{ const el=$(id); if(el&&el.data) Plotly.purge(el); }); } return; }
  box.hidden = false;
  const dlt = mc.delta;
  const method = mc.method || 'Adjusted';
  const deltaEl = $('wmdsDelta');
  const f1d = mc.f1_delta!=null ? mc.f1_delta : 0;
  const better = f1d > 0.003, worse = f1d < -0.003;
  deltaEl.textContent = `neighbour F1 ${mc.f1_plain!=null?mc.f1_plain.toFixed(3):'?'} → ${mc.f1_weighted!=null?mc.f1_weighted.toFixed(3):'?'}  (${f1d>=0?'+':''}${f1d.toFixed(3)})`;
  deltaEl.className = 'wmds-delta ' + (better?'wmds-up':(worse?'wmds-down':'wmds-flat'));
  $('wmdsNote').innerHTML = better
    ? `The ${method} distance <strong>improved</strong> neighbourhood preservation here. Left = plain Euclidean MDS, right = ${method} MDS; brush either to see where the same points land in both. PCA and the trust scores stay on the true geometry.`
    : (worse
      ? `The ${method} distance <strong>reduced</strong> neighbourhood preservation here, whitening amplified directions that on this data carry noise. Brush either layout to compare; the evidence table below quantifies it.`
      : `The ${method} distance left neighbourhood preservation roughly unchanged. Brush either layout to compare where points land.`);

  // "adjusted distance improves projection quality" evidence: does weighting
  // raise neighbourhood-F1 and lower the FP (intrusion) and FN (extrusion) rates?
  if(mc.f1_plain != null){
    const f1up = mc.f1_delta > 0;
    const fpDown = mc.fp_weighted < mc.fp_plain, fnDown = mc.fn_weighted < mc.fn_plain;
    const arrow=(a,b)=> b<a ? '<span style="color:#7ee0a0">↓</span>' : (b>a?'<span style="color:#e8969b">↑</span>':'=');
    let extra = `<div class="wmds-fidelity">`+
      `<div class="wmds-fid-head">Does the adjusted distance improve <em>projection fidelity</em>?</div>`+
      `<table class="wmds-fid-table"><thead><tr><th></th><th>plain MDS</th><th>${method} MDS</th><th></th></tr></thead><tbody>`+
      `<tr><td>neighbour F1</td><td>${mc.f1_plain.toFixed(3)}</td><td class="${f1up?'an-best':''}">${mc.f1_weighted.toFixed(3)}</td><td>${arrow(mc.f1_plain,mc.f1_weighted)}</td></tr>`+
      `<tr><td>false-positive rate (intrusions)</td><td>${mc.fp_plain.toFixed(3)}</td><td class="${fpDown?'an-best':''}">${mc.fp_weighted.toFixed(3)}</td><td>${arrow(mc.fp_plain,mc.fp_weighted)}</td></tr>`+
      `<tr><td>false-negative rate (extrusions)</td><td>${mc.fn_plain.toFixed(3)}</td><td class="${fnDown?'an-best':''}">${mc.fn_weighted.toFixed(3)}</td><td>${arrow(mc.fn_plain,mc.fn_weighted)}</td></tr>`+
      (mc.localstress_plain!=null ? `<tr><td>local stress fidelity</td><td>${mc.localstress_plain.toFixed(3)}</td><td class="${mc.localstress_weighted>mc.localstress_plain?'an-best':''}">${mc.localstress_weighted.toFixed(3)}</td><td>${arrow(mc.localstress_plain,mc.localstress_weighted)==='='?'=':(mc.localstress_weighted>mc.localstress_plain?'<span style="color:#7ee0a0">↑</span>':'<span style="color:#e8969b">↓</span>')}</td></tr>` : '')+
      (mc.shepard_plain!=null ? `<tr><td>global distance corr. (Shepard)</td><td>${mc.shepard_plain.toFixed(3)}</td><td class="${mc.shepard_weighted>mc.shepard_plain?'an-best':''}">${mc.shepard_weighted.toFixed(3)}</td><td>${mc.shepard_weighted>mc.shepard_plain?'<span style="color:#7ee0a0">↑</span>':(mc.shepard_weighted<mc.shepard_plain?'<span style="color:#e8969b">↓</span>':'=')}</td></tr>` : '')+
      `</tbody></table>`+
      `<div class="wmds-fid-verdict">`+
        (f1up && (fpDown||fnDown)
          ? `The adjusted (weighted) distance <strong>improves projection quality</strong>: neighbour-F1 rises and the intrusion/extrusion rates fall.`
          : `Here the adjusted distance does not clearly improve neighbourhood F1, the up-weighted features may not align with the neighbourhood structure this data needs.`)+
      `</div></div>`;
    $('wmdsNote').innerHTML += extra;
  }

  if(typeof Plotly==='undefined') return;
  const classColorArr = (d.table.labels||[]).map(l=>classColor(l));
  const ids = d.table.ids;
  const mk = (xs,ys)=>({ type:'scattergl', mode:'markers', x:xs, y:ys,
    marker:{ size:7, color:classColorArr, line:{color:'rgba(0,0,0,0.5)',width:0.4} },
    customdata:ids.map((id,i)=>[d.table.class_names[i],id]),
    selected:{marker:{opacity:1}}, unselected:{marker:{opacity:0.12}},
    hovertemplate:'idx %{customdata[1]}, %{customdata[0]}<extra></extra>' });
  // a highlight-ring overlay trace (empty until a selection is made)
  const ringTrace = ()=>({ type:'scattergl', mode:'markers', x:[], y:[],
    marker:{size:13, color:'rgba(0,0,0,0)', line:{color:'#e8edf3', width:2}},
    hoverinfo:'skip', showlegend:false });
  const lay = (title)=> baseLayout({
    title:{text:title,font:{size:12,color:'#e8edf3'},x:0.03,xanchor:'left'},
    xaxis:Object.assign({title:'dim 1'},AXIS), yaxis:Object.assign({title:'dim 2'},AXIS),
    showlegend:false, height:300, dragmode:'lasso',
  });
  const plainXY = {x: mc.plain.x, y: mc.plain.y};
  const weightXY = mc.weighted_xy
    ? {x: mc.weighted_xy.x, y: mc.weighted_xy.y}
    : {x: d.table.mds.x, y: d.table.mds.y};
  const cfg = Object.assign({}, PLOT_CONFIG, {modeBarButtonsToAdd:['select2d','lasso2d']});
  Plotly.react('wmdsPlain', [mk(plainXY.x, plainXY.y), ringTrace()], lay('Plain MDS'), cfg);
  Plotly.react('wmdsWeighted', [mk(weightXY.x, weightXY.y), ringTrace()], lay((mc.method||'Adjusted')+' MDS'), cfg);

  // --- linked brushing: a selection on either plot rings the same points on
  // BOTH (they share point indices) and fills a selection table so you can see
  // exactly which points moved between the plain and adjusted layouts ---
  const applyWmdsSel = (idxSet)=>{
    state._wmdsSel = idxSet;
    const pin = Array.from(idxSet);
    Plotly.restyle('wmdsPlain', { x:[pin.map(i=>plainXY.x[i])], y:[pin.map(i=>plainXY.y[i])], 'marker.line.color':'#e8edf3' }, [1]);
    Plotly.restyle('wmdsWeighted', { x:[pin.map(i=>weightXY.x[i])], y:[pin.map(i=>weightXY.y[i])], 'marker.line.color':'#e8edf3' }, [1]);
    renderWmdsSelTable(d, idxSet, plainXY, weightXY);
  };
  const wmdsClear = ()=>{
    applyWmdsSel(new Set());
    if(typeof Plotly!=='undefined'){
      ['wmdsPlain','wmdsWeighted'].forEach(id=>{ const el=$(id); if(el&&el.data){
        Plotly.restyle(id, {selectedpoints:[null]}, [0]);
        try{ Plotly.relayout(id, {selections:[]}); }catch(e){}
      }});
    }
  };
  const onSel = (ev)=>{
    if(!ev || !ev.points){ return; }
    const idxSet = new Set(ev.points.filter(p=>p.curveNumber===0).map(p=>p.pointIndex));
    applyWmdsSel(idxSet);
  };
  const plainEl = $('wmdsPlain'), weightEl = $('wmdsWeighted');
  // hover: ring the same point in BOTH layouts and show its details
  const wmdsHoverRing = (i, on)=>{
    if(typeof Plotly==='undefined') return;
    const pin = on ? [i] : [];
    Plotly.restyle('wmdsPlain', { x:[pin.map(k=>plainXY.x[k])], y:[pin.map(k=>plainXY.y[k])],
      'marker.line.color': on?'#e0a54b':'#e8edf3' }, [1]);
    Plotly.restyle('wmdsWeighted', { x:[pin.map(k=>weightXY.x[k])], y:[pin.map(k=>weightXY.y[k])],
      'marker.line.color': on?'#e0a54b':'#e8edf3' }, [1]);
  };
  const onHover = (ev)=>{ if(!ev||!ev.points||!ev.points.length) return;
    const p=ev.points.find(pt=>pt.curveNumber===0); if(!p) return;
    const i=p.pointIndex; wmdsHoverRing(i, true);
    const ro=$('wmdsHover'), tt=d.table, nmr=d.neighbor_metrics;
    if(ro){
      const dx=weightXY.x[i]-plainXY.x[i], dy=weightXY.y[i]-plainXY.y[i];
      const move=Math.hypot(dx,dy);
      const metricCells=(mth)=>METRIC_KEYS.map(k=>`${METRIC_LABELS[k]||k} <strong style="font-size:15px">${nmr[mth][k][i].toFixed(3)}</strong>`).join(', ');
      const kUsed = (nmr.MDS.k_used && nmr.MDS.k_used[i]!=null) ? nmr.MDS.k_used[i] : null;
      ro.innerHTML =
        `<span class="hr-idx" style="font-size:15px">idx ${tt.ids[i]}</span> <span class="class-chip" style="background:${classColor(tt.labels[i])}"></span>${tt.class_names[i]}, moved <strong>${move.toFixed(2)}</strong> plain→adjusted`+
        (kUsed!=null ? ` <span class="hr-k" style="font-size:15px">k = ${kUsed}</span>` : '')+
        `<span class="hr-row"><span class="hr-m" style="font-size:15px">PCA</span> ${metricCells('PCA')}</span>`+
        `<span class="hr-row"><span class="hr-m" style="font-size:15px">MDS</span> ${metricCells('MDS')}</span>`;
    }
  };
  const onUnhover = ()=>{ wmdsHoverRing(0, false); if(state.result){ /* restore selection rings */ applyWmdsSel(state._wmdsSel||new Set()); }
    const ro=$('wmdsHover'); if(ro) ro.innerHTML='Hover a point to see its details; the same point is highlighted in the other layout.'; };
  if(plainEl && plainEl.on){
    plainEl.on('plotly_selected', onSel);
    plainEl.on('plotly_deselect', wmdsClear);
    plainEl.on('plotly_hover', onHover);
    plainEl.on('plotly_unhover', onUnhover);
  }
  if(weightEl && weightEl.on){
    weightEl.on('plotly_selected', onSel);
    weightEl.on('plotly_deselect', wmdsClear);
    weightEl.on('plotly_hover', onHover);
    weightEl.on('plotly_unhover', onUnhover);
  }
  const clr=$('wmdsSelClear');
  if(clr && !clr.dataset.wired){ clr.dataset.wired='1';
    clr.addEventListener('click', wmdsClear); }
}

// selection table for the Mahalanobis comparison: how far each selected point
// moved between the plain and adjusted layout, plus its per-point trust F1
function renderWmdsSelTable(d, idxSet, plainXY, weightXY){
  const head=$('wmdsSelCount'), clr=$('wmdsSelClear'), wrap=$('wmdsSelWrap');
  if(!head) return;
  if(!idxSet || !idxSet.size){
    head.textContent='No points selected: drag on either layout to compare where the same points land.';
    if(clr) clr.hidden=true; if(wrap) wrap.hidden=true; return;
  }
  // remember args so header-click sorting can re-render
  _wmdsSelCtx = {d, idxSet, plainXY, weightXY};
  const t=d.table, nm=d.neighbor_metrics;
  head.innerHTML=`<span class="mono">${idxSet.size}</span> points selected, &Delta;pos = how far the point moved from the plain to the adjusted layout.`;
  if(clr) clr.hidden=false; if(wrap) wrap.hidden=false;
  let rows=[...idxSet].map(i=>{
    const dx=weightXY.x[i]-plainXY.x[i], dy=weightXY.y[i]-plainXY.y[i];
    return { i, index:t.ids[i], cls:t.class_names[i], move: Math.hypot(dx,dy), f1: nm?nm.MDS.f1[i]:0 };
  });
  const st=state.wmdsSelSort;
  rows.sort((a,b)=>{
    let va=a[st.key], vb=b[st.key];
    if(typeof va==='string') return st.dir*va.localeCompare(vb);
    return st.dir*(va-vb);
  });
  const cols=[{k:'index',label:'index'},{k:'cls',label:'class'},{k:'move',label:'Δpos'},{k:'f1',label:'MDS F1'}];
  const tbl=$('wmdsSelTable');
  tbl.querySelector('thead').innerHTML='<tr>'+cols.map(c=>{
    const cl = st.key===c.k ? (st.dir>0?'sorted-asc':'sorted-desc') : '';
    return `<th class="th-sort ${cl}" data-wmdssort="${c.k}">${c.label}</th>`;
  }).join('')+'</tr>';
  tbl.querySelector('tbody').innerHTML=rows.slice(0,200).map(r=>{
    const code=t.labels[r.i];
    const f1cell=(()=>{const c=r.f1<0.34?'cell-fn':(r.f1<0.67?'cell-zero':'cell-fp');return `<td class="${c}">${r.f1.toFixed(3)}</td>`;})();
    return `<tr><td>${r.index}</td>`+
      `<td class="col-class"><span class="class-chip" style="background:${classColor(code)}"></span>${r.cls}</td>`+
      `<td>${r.move.toFixed(2)}</td>`+f1cell+'</tr>';
  }).join('');
  tbl.querySelectorAll('th[data-wmdssort]').forEach(th=>{
    th.addEventListener('click', ()=>{
      const k=th.dataset.wmdssort;
      if(state.wmdsSelSort.key===k) state.wmdsSelSort.dir*=-1;
      else state.wmdsSelSort={key:k, dir:1};
      if(_wmdsSelCtx) renderWmdsSelTable(_wmdsSelCtx.d, _wmdsSelCtx.idxSet, _wmdsSelCtx.plainXY, _wmdsSelCtx.weightXY);
    });
  });
  tbl.querySelectorAll('.idx-click').forEach(td=>{
    td.addEventListener('click', ()=> pinPoint(parseInt(td.dataset.pt)));
  });
}
let _wmdsSelCtx = null;


// draw the pinned point as a green ring on the plain/adjusted MDS plots
function applyWmdsPin(){
  if(typeof Plotly==='undefined' || !_wmdsSelCtx) return;
  const {plainXY, weightXY}=_wmdsSelCtx;
  const cur = state._wmdsSel || new Set();
  const sel=[...cur];
  const colors=sel.map(()=>'#e8edf3');
  const pin=state.pinned;
  const idxList=sel.slice();
  if(pin!=null && plainXY.x[pin]!=null){
    if(!idxList.includes(pin)){ idxList.push(pin); colors.push('#4ec97a'); }
    else colors[idxList.indexOf(pin)]='#4ec97a';
  }
  const el1=$('wmdsPlain'), el2=$('wmdsWeighted');
  if(el1&&el1.data) Plotly.restyle('wmdsPlain',{x:[idxList.map(i=>plainXY.x[i])],y:[idxList.map(i=>plainXY.y[i])],'marker.line.color':[colors]},[1]);
  if(el2&&el2.data) Plotly.restyle('wmdsWeighted',{x:[idxList.map(i=>weightXY.x[i])],y:[idxList.map(i=>weightXY.y[i])],'marker.line.color':[colors]},[1]);
}
