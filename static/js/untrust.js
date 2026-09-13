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

/* Minimum unevenness penalty; the effective value is derived per run. */
const UT_PENALTY_FLOOR = 0.5;
const UT_EVEN_CUTS_FIXED = { red: 0.80, amber: 0.90 };

const UT_BAND_COLOR  = { red:'#d24b52', amber:'#e0a54b', green:'#4ec97a' };
const UT_DIM_FILL    = '#2a3340';
const UT_RING_SIZE   = 20;   // Evenness ring diameter, px
const UT_RING_WIDTH  = 2;    // ... at unchanged stroke weight
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

  // pass 1: raw mean, spread and Evenness per point per method
  ['PCA','MDS'].forEach(m=>{
    const mean=new Array(n), spread=new Array(n), even=new Array(n);
    for(let i=0;i<n;i++){
      let sum=0, lo=Infinity, hi=-Infinity;
      for(const k of METRIC_KEYS){
        const v = nm[m][k][i];
        sum += v;
        if(v<lo) lo=v;
        if(v>hi) hi=v;
      }
      mean[i]   = sum / METRIC_KEYS.length;
      spread[i] = hi - lo;
      even[i]   = 1 - spread[i];
    }
    raw[m] = { mean, spread, even };
  });

  // Evenness cut-offs from the pooled distribution (see note above)
  let evenCuts = UT_EVEN_CUTS_FIXED;
  if(UT_EVEN_ADAPTIVE){
    const pooled = raw.PCA.even.concat(raw.MDS.even).slice().sort((a,b)=>a-b);
    const pct = (q)=> pooled[Math.min(pooled.length-1, Math.floor(q*(pooled.length-1)))];
    evenCuts = { red: pct(0.25), amber: pct(0.50) };
  }

  /* ----------------------------------------------------------------
     Level is the mean of the four metrics MINUS a penalty for how much
     they disagree:

         Level = clip( mean - lambda * spread , 0, 1 )

     Without it the plain mean produced a contradiction the view could not
     defend: a point whose metrics ranged from 0.20 to 0.95 averaged to a
     comfortable green Level while its Evenness was red. Measured on the
     bundled datasets, 48 of 89 red-Evenness points on Wine and all 75 on
     Iris showed a green Level. A point the metrics cannot agree on is not
     a point you can call faithful, so the disagreement now costs it.

     lambda is set per run rather than fixed, so the invariant holds by
     construction instead of by luck. Red Evenness means spread > S, where
     S = 1 - evenCuts.red. Since mean <= 1, choosing

         lambda >= (1 - amberCut) / S

     forces every red-Evenness point below the green threshold. The floor of
     0.5 keeps a mild penalty on runs where the metrics already agree
     closely and the derived value would be near zero.
     ---------------------------------------------------------------- */
  const S = Math.max(1e-6, 1 - evenCuts.red);
  const lambda = Math.max(UT_PENALTY_FLOOR, (1 - UT_LEVEL_CUTS.amber) / S);

  // pass 2: penalised Level, bands, tiers
  const out = {};
  ['PCA','MDS'].forEach(m=>{
    const { mean, spread, even } = raw[m];
    const level=new Array(n), tier=new Array(n), lb=new Array(n), eb=new Array(n);
    for(let i=0;i<n;i++){
      level[i] = Math.max(0, Math.min(1, mean[i] - lambda * spread[i]));
      lb[i] = utBand(level[i], UT_LEVEL_CUTS);
      eb[i] = utBand(even[i],  evenCuts);
      // Either band red -> Untrustworthy; otherwise either amber -> Medium.
      tier[i] = (lb[i]==='red' || eb[i]==='red') ? 'Untrustworthy'
              : (lb[i]==='amber' || eb[i]==='amber') ? 'Medium'
              : 'Trustworthy';
    }
    out[m] = { level, even, tier, levelBand:lb, evenBand:eb,
               mean: mean, spread: spread };
  });
  out._evenCuts = evenCuts;
  out._lambda = lambda;
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

/* Rows for the single table: a point appears if it is visible in EITHER plot,
   so the table is the union of what the two projections are showing. */
function utVisibleAny(d){
  const st = utStats(d);
  const want = state.untrustTier;
  const out = [];
  trustActiveSet(d).forEach(i=>{
    if(want==='all' || st.PCA.tier[i]===want || st.MDS.tier[i]===want) out.push(i);
  });
  return out.sort((a,b)=>a-b);
}

/* Which projections call this point untrustworthy: 'PCA', 'MDS', 'both', or '--'. */
function utFlaggedIn(st, i){
  const p = st.PCA.tier[i]==='Untrustworthy', m = st.MDS.tier[i]==='Untrustworthy';
  return p && m ? 'both' : p ? 'PCA' : m ? 'MDS' : '--';
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

    const fill = [], op = [];
    for(let i=0;i<n;i++){
      const on = vis.has(i);
      fill.push(on ? scoreColor(s.level[i]) : UT_DIM_FILL);
      op.push(on ? 1 : 0.22);
    }
    return {
      type:'scattergl', mode:'markers', x:proj.x, y:proj.y,
      marker:{ size:9, color:fill, opacity:op,
               line:{ color:'rgba(0,0,0,0.45)', width:0.5 } },
      customdata: ids.map((id,i)=>[id, t.class_names[i], s.tier[i]]),
      hovertemplate:'idx %{customdata[0]}, %{customdata[1]}<br>'+
                    '%{customdata[2]}<extra></extra>',
      selected:{marker:{opacity:1}},
      unselected:{marker:{opacity: dimming?0.12:1}},
    };
  };

  /* Evenness ring: its own trace so the radius can grow without the stroke
     getting heavier. A ring drawn as marker.line on the fill trace would have
     to thicken to become more visible; this one just sits further out, the way
     the pin ring does. */
  const evenRing = (method, proj) => {
    const s = st[method];
    const vis = new Set(utVisible(d, method));
    const col = [], op = [];
    for(let i=0;i<n;i++){
      const on = vis.has(i);
      col.push(on ? scoreColor(s.even[i]) : UT_DIM_FILL);
      op.push(on ? 1 : 0.18);
    }
    return {
      type:'scattergl', mode:'markers', x:proj.x, y:proj.y,
      marker:{ size:UT_RING_SIZE, color:'rgba(0,0,0,0)', opacity:op,
               line:{ color:col, width:UT_RING_WIDTH } },
      hoverinfo:'skip', showlegend:false,
    };
  };

  /* Selection / pin ring (trace index 2), drawn outside the Evenness ring so
     the two never overlap. applyTrustRings in trust.js drives this trace. */
  const ringTrace = () => ({
    type:'scattergl', mode:'markers', x:[], y:[],
    marker:{ size:UT_RING_SIZE+10, color:'rgba(0,0,0,0)',
             line:{ color:'#e8edf3', width:2.5 } },
    hoverinfo:'skip', showlegend:false,
  });

  const lay = (title) => baseLayout({
    title:{text:title, font:{size:12,color:'#e8edf3'}, x:0.03, xanchor:'left'},
    xaxis:Object.assign({title:'dim 1'},AXIS), yaxis:Object.assign({title:'dim 2'},AXIS),
    showlegend:false, height:320, dragmode:'lasso',
  });
  const cfg = Object.assign({}, PLOT_CONFIG, {modeBarButtonsToAdd:['select2d','lasso2d']});

  Plotly.react('untrustPca',
    [mk('PCA', t.pca), evenRing('PCA', t.pca), ringTrace()], lay(mName('PCA')), cfg);
  Plotly.react('untrustMds',
    [mk('MDS', t.mds), evenRing('MDS', t.mds), ringTrace()], lay(mName('MDS')), cfg);

  if((state.trustSel && state.trustSel.size) || state.pinned!=null){
    applyTrustRings(d, state.trustSel || new Set());
  }

  // --- hover: full readout, and cross-highlight in the sibling plot ---
  const onHover = (ev)=>{
    if(!ev||!ev.points||!ev.points.length) return;
    const p = ev.points.find(pt=>pt.curveNumber===0); if(!p) return;
    const ringSet = new Set(state.trustSel || new Set());
    ringSet.add(p.pointIndex);
    applyTrustRings(state.result, ringSet, /*hover*/true);
    utShowPoint(p.pointIndex);
  };

  const onUnhover = ()=>{
    applyTrustRings(state.result, state.trustSel || new Set());
    // Fall back to the pinned point rather than the hint, so a point opened by
    // clicking its index in the table stays on screen.
    if(state.pinned != null) utShowPoint(state.pinned);
    else { const ro=$('untrustHover'); if(ro) ro.innerHTML = UT_HOVER_HINT; }
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

/* Write one point's full breakdown into the readout above the plots: its four
   raw metrics, Level, Evenness and tier, for BOTH projections. This is the only
   place per-metric numbers are shown -- the table no longer carries them. */
function utShowPoint(i){
  const ro = $('untrustHover'), r = state.result;
  if(!ro || !r || i == null) return;
  const s = utStats(r), tt = r.table, nmr = r.neighbor_metrics;
  const f = (v)=>v.toFixed(3);
  const row = (m)=>{
    const mm = s[m];
    const metrics = METRIC_KEYS.map(k=>
      `${METRIC_LABELS[k]||k} <strong>${f(nmr[m][k][i])}</strong>`).join(', ');
    return `<span class="hr-row"><span class="hr-m">${mName(m)}</span> ${metrics}`+
           `<span class="ut-hr-sep">|</span>`+
           `Level <strong class="ut-${mm.levelBand[i]}">${f(mm.level[i])}</strong>, `+
           `Evenness <strong class="ut-${mm.evenBand[i]}">${f(mm.even[i])}</strong> `+
           `<span class="ut-badge ut-${UT_TIER_BAND[mm.tier[i]]}">${mm.tier[i]}</span></span>`;
  };
  ro.innerHTML =
    `<span class="hr-idx">idx ${tt.ids[i]}</span> `+
    `<span class="class-chip" style="background:${classColor(tt.labels[i])}"></span>`+
    `${tt.class_names[i]}`+
    (state.pinned===i ? ' <span class="ut-pin-tag">pinned</span>' : '')+
    row('PCA') + row('MDS');
}

const UT_HOVER_HINT = 'Hover a point, or click an index in the table below, to '+
  'see its four metrics, Level, Evenness and tier in both projections.';

/* ------------------------------------------------------------------
   The table: whatever is currently visible, for the selected method
   ------------------------------------------------------------------ */
const UT_TABLE_COLS = [
  { key:'index',     label:'index' },
  { key:'cls',       label:'class', cls:'col-class' },
  { key:'pcaLevel',  label:'Level', slot:'PCA', band:'pcaLevelBand' },
  { key:'pcaEven',   label:'Evenness', slot:'PCA', band:'pcaEvenBand' },
  { key:'mdsLevel',  label:'Level', slot:'MDS', band:'mdsLevelBand' },
  { key:'mdsEven',   label:'Evenness', slot:'MDS', band:'mdsEvenBand' },
  { key:'tier',      label:'Tier', tier:true },
  { key:'flagged',   label:'untrustworthy in', flag:true },
];

const UT_TIER_RANK = { Untrustworthy:0, Medium:1, Trustworthy:2 };

function renderUntrustTable(d){
  const table = $('utTable'); if(!table) return;
  const st = utStats(d), t = d.table;
  const vis = utVisibleAny(d);

  const meta = $('utTableMeta');
  if(meta){
    const tierTxt = state.untrustTier==='all'
      ? 'all tiers' : `tier ${state.untrustTier.toLowerCase()}`;
    meta.innerHTML = vis.length
      ? `<span class="mono">${vis.length}</span> visible point(s), ${tierTxt}.`
      : 'No points match the current tier and filters.';
  }

  const rows = vis.map(i=>({
    i, index:t.ids[i], cls:t.class_names[i], code:t.labels[i],
    pcaLevel:st.PCA.level[i], pcaEven:st.PCA.even[i],
    mdsLevel:st.MDS.level[i], mdsEven:st.MDS.even[i],
    pcaLevelBand:st.PCA.levelBand[i], pcaEvenBand:st.PCA.evenBand[i],
    mdsLevelBand:st.MDS.levelBand[i], mdsEvenBand:st.MDS.evenBand[i],
    // the row's overall verdict is the worse of the two projections
    tier: UT_TIER_RANK[st.PCA.tier[i]] <= UT_TIER_RANK[st.MDS.tier[i]]
            ? st.PCA.tier[i] : st.MDS.tier[i],
    flagged: utFlaggedIn(st, i),
  }));

  const { key, dir } = state.untrustSort;
  const flagRank = { both:0, PCA:1, MDS:2, '--':3 };
  rows.sort((a,b)=>{
    let av=a[key], bv=b[key];
    if(key==='tier'){ av=UT_TIER_RANK[av]; bv=UT_TIER_RANK[bv]; }
    else if(key==='flagged'){ av=flagRank[av]; bv=flagRank[bv]; }
    if(typeof av==='string') return dir*av.localeCompare(bv);
    return dir*(av-bv);
  });

  const thead = table.querySelector('thead'), tbody = table.querySelector('tbody');
  thead.innerHTML = '<tr>' + UT_TABLE_COLS.map(c=>{
    const sc = state.untrustSort.key===c.key
      ? (state.untrustSort.dir>0?'sorted-asc':'sorted-desc') : '';
    const lbl = c.slot ? `${mName(c.slot)} ${c.label}` : c.label;
    return `<th class="${c.cls||''} ${sc}" data-key="${c.key}">${lbl}</th>`;
  }).join('') + '</tr>';

  tbody.innerHTML = rows.map(r=> '<tr>' + UT_TABLE_COLS.map(c=>{
    const v = r[c.key];
    if(c.key==='index')
      return `<td class="idx-click${state.pinned===r.i?' idx-pinned':''}" data-pt="${r.i}">${v}</td>`;
    if(c.cls==='col-class')
      return `<td class="col-class"><span class="class-chip" style="background:${classColor(r.code)}"></span>${v}</td>`;
    if(c.tier)
      return `<td><span class="ut-badge ut-${UT_TIER_BAND[v]}">${v}</span></td>`;
    if(c.flag){
      if(v==='--') return '<td class="ut-flag-none">--</td>';
      const cls = v==='both' ? 'out-both' : v==='PCA' ? 'out-pca' : 'out-mds';
      const txt = v==='both' ? 'both' : mName(v);
      return `<td><span class="out-badge ${cls}">${txt}</span></td>`;
    }
    if(c.band)
      return `<td class="ut-cell ut-${r[c.band]}">${v.toFixed(3)}</td>`;
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

  // Clicking an index pins the point everywhere AND opens its per-metric
  // breakdown in the readout above the plots. That detail is deliberately not
  // carried as table columns.
  tbody.querySelectorAll('.idx-click').forEach(td=>{
    td.addEventListener('click', ()=>{
      const i = parseInt(td.dataset.pt);
      pinPoint(i);
      if(state.pinned === i) utShowPoint(i);
      else { const ro=$('untrustHover'); if(ro) ro.innerHTML = UT_HOVER_HINT; }
      renderUntrustTable(state.result);
    });
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
  seg('utTierSeg', (b)=>{ state.untrustTier = b.dataset.ut; });
}

function renderUntrust(d){
  const host = document.querySelector('[data-panel="untrust-plots"]');
  if(!d || !d.neighbor_metrics){ if(host) host.style.display='none'; return; }
  if(host) host.style.display='';
  bindUntrustControls();
  renderUntrustPlots(d);
  renderUntrustTable(d);
}