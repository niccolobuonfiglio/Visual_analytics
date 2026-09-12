/* ======================================================================
   UNTRUSTWORTHY POINTS: LEVEL / EVENNESS SCATTERS
   Two linked projection scatters built the same way as the Trust Analysis
   pair, but double-encoded: marker FILL is the point's Level (how faithful
   it is on average) and marker RING is its Evenness (how much the four
   metrics agree about it). A tier toggle dims non-matching points, and the
   table below lists whatever is currently visible.
   ====================================================================== */

/* ------------------------------------------------------------------
   Band thresholds.

   These are the only tunable numbers in this file; both sets are taken
   from cut-offs the codebase already uses, so the new view agrees with
   the existing ones rather than inventing a second opinion.

   LEVEL bands reuse the score-cell cut-offs in render.js (cell-fn /
   cell-zero / cell-fp at 0.34 and 0.67).

   EVENNESS bands reuse the method-disagreement filter in trust.js, which
   calls a spread above 0.2 "large" and 0.1-0.2 "medium". Evenness is
   1 - spread, so those become 0.80 and 0.90.
   ------------------------------------------------------------------ */
const UT_LEVEL_CUTS = { red: 0.34, amber: 0.67 };

/* Evenness needs per-run cut-offs, not fixed ones. The four metrics sit on
   systematically different scales -- on Wine, neighbour F1 has a median of
   0.38 while local stress has 0.86 -- so (max - min) is large for EVERY
   point, and any fixed cut classifies almost the whole dataset the same way.
   Measured: fixed cuts of 0.80/0.90 put 94% of Wine and 98% of Breast Cancer
   in a single tier, which tells you nothing.

   So the cut-offs come from the run's own evenness distribution, pooled
   across PCA and MDS so the two plots stay comparable: the bottom quartile
   is red, up to the median is amber, above it is green. Evenness then means
   "do the metrics agree about this point, relative to how much they usually
   agree on this data" -- which is the question the tier rule needs answered.

   Set UT_EVEN_ADAPTIVE to false to fall back to the fixed cuts below. */
const UT_EVEN_ADAPTIVE = true;
const UT_EVEN_CUTS_FIXED = { red: 0.80, amber: 0.90 };

const UT_BAND_COLOR  = { red:'#d24b52', amber:'#e0a54b', green:'#4ec97a' };
const UT_DIM_FILL    = '#2a3340';
const UT_TIERS       = ['Untrustworthy','Medium','Trustworthy'];
const UT_TIER_BAND   = { Untrustworthy:'red', Medium:'amber', Trustworthy:'green' };

let _utBound = false;

/* ------------------------------------------------------------------
   Per point, per method: Level, Evenness, Tier.
   Cached on the result object -- the inputs never change for a given run,
   so re-filtering or re-sorting must not recompute this.
   ------------------------------------------------------------------ */
function utBand(v, cuts){
  if(v < cuts.red) return 'red';
  if(v < cuts.amber) return 'amber';
  return 'green';
}

function utStats(d){
  if(d._untrust) return d._untrust;
  const nm = d.neighbor_metrics;
  const n = d.table.ids.length;
  const raw = {};

  // pass 1: Level and Evenness per point per method
  ['PCA','MDS'].forEach(m=>{
    const level=new Array(n), even=new Array(n);
    for(let i=0;i<n;i++){
      let sum=0, lo=Infinity, hi=-Infinity;
      for(const k of METRIC_KEYS){
        const v = nm[m][k][i];
        sum += v;
        if(v<lo) lo=v;
        if(v>hi) hi=v;
      }
      level[i] = sum / METRIC_KEYS.length;
      even[i]  = 1 - (hi - lo);
    }
    raw[m] = { level, even };
  });

  // Evenness cut-offs from the pooled distribution (see note above)
  let evenCuts = UT_EVEN_CUTS_FIXED;
  if(UT_EVEN_ADAPTIVE){
    const pooled = raw.PCA.even.concat(raw.MDS.even).slice().sort((a,b)=>a-b);
    const pct = (q)=> pooled[Math.min(pooled.length-1, Math.floor(q*(pooled.length-1)))];
    evenCuts = { red: pct(0.25), amber: pct(0.50) };
  }

  // pass 2: bands and tiers
  const out = {};
  ['PCA','MDS'].forEach(m=>{
    const { level, even } = raw[m];
    const tier=new Array(n), lb=new Array(n), eb=new Array(n);
    for(let i=0;i<n;i++){
      lb[i] = utBand(level[i], UT_LEVEL_CUTS);
      eb[i] = utBand(even[i],  evenCuts);
      // Either band red -> Untrustworthy; otherwise either amber -> Medium.
      tier[i] = (lb[i]==='red' || eb[i]==='red') ? 'Untrustworthy'
              : (lb[i]==='amber' || eb[i]==='amber') ? 'Medium'
              : 'Trustworthy';
    }
    out[m] = { level, even, tier, levelBand:lb, evenBand:eb };
  });
  out._evenCuts = evenCuts;
  d._untrust = out;
  return out;
}

/* Points visible in a given method's plot: they must pass the panel's
   existing filters and brushing, and match the tier toggle. */
function utVisible(d, method){
  const st = utStats(d)[method];
  const active = trustActiveSet(d);
  const want = state.untrustTier;
  const out = [];
  active.forEach(i=>{
    if(want==='all' || st.tier[i]===want) out.push(i);
  });
  return out.sort((a,b)=>a-b);
}

/* ------------------------------------------------------------------
   The two scatters
   ------------------------------------------------------------------ */
function renderUntrustPlots(d){
  if(typeof Plotly==='undefined') return;
  const st = utStats(d), t = d.table, ids = t.ids;
  const n = ids.length;

  const mk = (method, proj) => {
    const s = st[method];
    const vis = new Set(utVisible(d, method));
    const dimming = vis.size !== n;

    const fill = [], ring = [], op = [];
    for(let i=0;i<n;i++){
      const on = vis.has(i);
      fill.push(on ? UT_BAND_COLOR[s.levelBand[i]] : UT_DIM_FILL);
      ring.push(on ? UT_BAND_COLOR[s.evenBand[i]]  : UT_DIM_FILL);
      op.push(on ? 1 : 0.22);
    }
    return {
      type:'scattergl', mode:'markers', x:proj.x, y:proj.y,
      marker:{ size:9, color:fill, opacity:op,
               line:{ color:ring, width:2 } },
      customdata: ids.map((id,i)=>[id, t.class_names[i], s.tier[i]]),
      hovertemplate:'idx %{customdata[0]}, %{customdata[1]}<br>'+
                    '%{customdata[2]}<extra></extra>',
      selected:{marker:{opacity:1}},
      unselected:{marker:{opacity: dimming?0.12:1}},
    };
  };

  // ring trace (index 1) so applyTrustRings can drive pin/selection here too
  const ringTrace = () => ({
    type:'scattergl', mode:'markers', x:[], y:[],
    marker:{ size:18, color:'rgba(0,0,0,0)', line:{ color:'#e8edf3', width:2.5 } },
    hoverinfo:'skip', showlegend:false,
  });

  const lay = (title) => baseLayout({
    title:{text:title, font:{size:12,color:'#e8edf3'}, x:0.03, xanchor:'left'},
    xaxis:Object.assign({title:'dim 1'},AXIS), yaxis:Object.assign({title:'dim 2'},AXIS),
    showlegend:false, height:320, dragmode:'lasso',
  });
  const cfg = Object.assign({}, PLOT_CONFIG, {modeBarButtonsToAdd:['select2d','lasso2d']});

  Plotly.react('untrustPca', [mk('PCA', t.pca), ringTrace()], lay('PCA'), cfg);
  Plotly.react('untrustMds', [mk('MDS', t.mds), ringTrace()], lay('MDS'), cfg);

  if((state.trustSel && state.trustSel.size) || state.pinned!=null){
    applyTrustRings(d, state.trustSel || new Set());
  }

  // --- hover: full readout, and cross-highlight in the sibling plot ---
  const onHover = (ev)=>{
    if(!ev||!ev.points||!ev.points.length) return;
    const p = ev.points.find(pt=>pt.curveNumber===0); if(!p) return;
    const i = p.pointIndex;
    const ringSet = new Set(state.trustSel || new Set());
    ringSet.add(i);
    applyTrustRings(state.result, ringSet, /*hover*/true);

    const ro = $('untrustHover'), r = state.result;
    if(!ro || !r) return;
    const s = utStats(r), tt = r.table, nmr = r.neighbor_metrics;
    const f = (v)=>v.toFixed(3);
    const row = (m)=>{
      const mm = s[m];
      const metrics = METRIC_KEYS.map(k=>
        `${METRIC_LABELS[k]||k} <strong>${f(nmr[m][k][i])}</strong>`).join(', ');
      return `<span class="hr-row"><span class="hr-m">${m}</span> ${metrics}`+
             `<span class="ut-hr-sep">|</span>`+
             `Level <strong class="ut-${mm.levelBand[i]}">${f(mm.level[i])}</strong>, `+
             `Evenness <strong class="ut-${mm.evenBand[i]}">${f(mm.even[i])}</strong> `+
             `<span class="ut-badge ut-${UT_TIER_BAND[mm.tier[i]]}">${mm.tier[i]}</span></span>`;
    };
    ro.innerHTML =
      `<span class="hr-idx">idx ${tt.ids[i]}</span> `+
      `<span class="class-chip" style="background:${classColor(tt.labels[i])}"></span>`+
      `${tt.class_names[i]}` + row('PCA') + row('MDS');
  };

  const onUnhover = ()=>{
    applyTrustRings(state.result, state.trustSel || new Set());
    const ro=$('untrustHover');
    if(ro) ro.innerHTML = UT_HOVER_HINT;
  };

  const onSel = (ev)=>{
    if(!ev || !ev.points) return;
    const idx = new Set(ev.points.filter(p=>p.curveNumber===0).map(p=>p.pointIndex));
    state.trustSel = idx.size ? idx : null;
    applyTrustRings(state.result, state.trustSel || new Set());
    refreshTrust(state.result);
    renderTrustSelTable(state.result);
    renderUntrust(state.result);
    updateTrustPlotClearBtn();
  };

  [$('untrustPca'), $('untrustMds')].forEach(el=>{
    if(!el || !el.on) return;
    if(el.removeAllListeners){
      el.removeAllListeners('plotly_selected');
      el.removeAllListeners('plotly_deselect');
      el.removeAllListeners('plotly_hover');
      el.removeAllListeners('plotly_unhover');
    }
    el.on('plotly_selected', onSel);
    el.on('plotly_deselect', ()=>clearTrustSelection());
    el.on('plotly_hover', onHover);
    el.on('plotly_unhover', onUnhover);
  });
}

const UT_HOVER_HINT = 'Hover a point for its four metrics, Level, Evenness and '+
  'tier in both projections; it is ringed in the other plot too.';

/* ------------------------------------------------------------------
   The table: whatever is currently visible, for the selected method
   ------------------------------------------------------------------ */
const UT_TABLE_COLS = [
  { key:'index',  label:'index' },
  { key:'cls',    label:'class', cls:'col-class' },
  { key:'level',  label:'Level', score:true },
  { key:'even',   label:'Evenness', score:true },
  { key:'tier',   label:'Tier', tier:true },
  { key:'f1',            label:'neighbour F1', score:true },
  { key:'js_fidelity',   label:'JS fidelity', score:true },
  { key:'localstress',   label:'local stress', score:true },
  { key:'normstress',    label:'norm. stress', score:true },
];

function renderUntrustTable(d){
  const table = $('utTable'); if(!table) return;
  const m = state.untrustMethod;
  const s = utStats(d)[m], t = d.table, nm = d.neighbor_metrics;
  const vis = utVisible(d, m);

  const meta = $('utTableMeta');
  if(meta){
    const tierTxt = state.untrustTier==='all' ? 'all tiers' : state.untrustTier.toLowerCase();
    meta.innerHTML = vis.length
      ? `<span class="mono">${vis.length}</span> visible ${m} point(s), ${tierTxt}.`
      : `No ${m} points match the current tier and filters.`;
  }

  const rows = vis.map(i=>{
    const r = { i, index:t.ids[i], cls:t.class_names[i], code:t.labels[i],
                level:s.level[i], even:s.even[i], tier:s.tier[i],
                levelBand:s.levelBand[i], evenBand:s.evenBand[i] };
    METRIC_KEYS.forEach(k=> r[k] = nm[m][k][i]);
    return r;
  });

  const { key, dir } = state.untrustSort;
  const tierRank = { Untrustworthy:0, Medium:1, Trustworthy:2 };
  rows.sort((a,b)=>{
    let av=a[key], bv=b[key];
    if(key==='tier'){ av=tierRank[av]; bv=tierRank[bv]; }
    if(typeof av==='string') return dir*av.localeCompare(bv);
    return dir*(av-bv);
  });

  const thead = table.querySelector('thead'), tbody = table.querySelector('tbody');
  thead.innerHTML = '<tr>' + UT_TABLE_COLS.map(c=>{
    const sc = state.untrustSort.key===c.key
      ? (state.untrustSort.dir>0?'sorted-asc':'sorted-desc') : '';
    return `<th class="${c.cls||''} ${sc}" data-key="${c.key}">${c.label}</th>`;
  }).join('') + '</tr>';

  tbody.innerHTML = rows.map(r=> '<tr>' + UT_TABLE_COLS.map(c=>{
    const v = r[c.key];
    if(c.key==='index')
      return `<td class="idx-click${state.pinned===r.i?' idx-pinned':''}" data-pt="${r.i}">${v}</td>`;
    if(c.cls==='col-class')
      return `<td class="col-class"><span class="class-chip" style="background:${classColor(r.code)}"></span>${v}</td>`;
    if(c.tier)
      return `<td><span class="ut-badge ut-${UT_TIER_BAND[v]}">${v}</span></td>`;
    if(c.score){
      // Level and Evenness carry their own band; the raw metrics reuse the
      // existing score-cell shading so they read the same as every other table.
      const band = c.key==='level' ? r.levelBand
                 : c.key==='even'  ? r.evenBand
                 : utBand(v, UT_LEVEL_CUTS);
      return `<td class="ut-cell ut-${band}">${v.toFixed(3)}</td>`;
    }
    return `<td>${v}</td>`;
  }).join('') + '</tr>').join('');

  thead.querySelectorAll('th').forEach(th=>{
    th.addEventListener('click', ()=>{
      const k = th.dataset.key;
      if(state.untrustSort.key===k) state.untrustSort.dir *= -1;
      else state.untrustSort = { key:k, dir:1 };
      renderUntrustTable(d);
    });
  });
  tbody.querySelectorAll('.idx-click').forEach(td=>{
    td.addEventListener('click', ()=> pinPoint(parseInt(td.dataset.pt)));
  });
}

/* ------------------------------------------------------------------
   Controls + entry point
   ------------------------------------------------------------------ */
function bindUntrustControls(){
  if(_utBound) return;
  _utBound = true;
  const seg = (id, fn)=>{
    const el = $(id); if(!el) return;
    el.addEventListener('click',(e)=>{
      const b = e.target.closest('.seg-btn'); if(!b) return;
      el.querySelectorAll('.seg-btn').forEach(x=>x.classList.toggle('active', x===b));
      fn(b);
      if(state.result) renderUntrust(state.result);
    });
  };
  seg('utTierSeg',   (b)=>{ state.untrustTier   = b.dataset.ut; });
  seg('utMethodSeg', (b)=>{ state.untrustMethod = b.dataset.um; });
}

function renderUntrust(d){
  const host = document.querySelector('[data-panel="untrust-plots"]');
  if(!d || !d.neighbor_metrics){ if(host) host.style.display='none'; return; }
  if(host) host.style.display='';
  bindUntrustControls();
  renderUntrustPlots(d);
  renderUntrustTable(d);
}