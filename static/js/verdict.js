/* ======================================================================
   VERDICT
   The PCA-vs-MDS decision: headline banner, analysis card, component
   bars, weight sliders, confidence chart and the full-detail modal.
   ====================================================================== */


/* ============================================================
   VERDICT BANNER + MODAL + ANALYSIS PANEL
   ============================================================ */
function _verdictParts(v){
  // returns {winner, loser, decisive, tie, margin, relpct, ci, d, sWin, sLose}
  const pm = v.per_method || {};
  const names = v.ranking || Object.keys(pm);
  const winner = v.winner, loser = names[names.length-1];
  return {
    winner, loser,
    decisive: !!v.decisive,
    tie: v.margin_absolute < 0.005,
    margin: v.margin_absolute,
    relpct: v.margin_relative_pct,
    ci: v.bootstrap_ci_mean_distortion_diff || [0,0],
    d: v.effect_size_cohen_d,
    sWin: pm[winner] ? pm[winner].score : null,
    sLose: pm[loser] ? pm[loser].score : null,
  };
}

function renderVerdictBanner(d){
  const banner = $('verdictBanner');
  const v = d.verdict;
  if(!v){ banner.hidden = true; return; }
  banner.hidden = false;
  const p = _verdictParts(v);
  const badge = $('vbBadge'), headline = $('vbHeadline'), sub = $('vbSub');
  banner.classList.remove('vb-decisive','vb-tie','vb-soft');
  if(p.tie){
    banner.classList.add('vb-tie');
    badge.textContent = 'TIE';
    headline.textContent = 'PCA and MDS are effectively tied';
    sub.textContent = `composite ${fmt3(p.sWin)} vs ${fmt3(p.sLose)}, no meaningful difference on this dataset`;
  } else if(p.decisive){
    banner.classList.add('vb-decisive');
    badge.textContent = mName(p.winner);
    headline.innerHTML = `<strong>${mName(p.winner)}</strong> is the more faithful projection`;
    sub.textContent = `decisive, composite ${fmt3(p.sWin)} vs ${fmt3(p.sLose)} (+${fmt3(p.margin)}, ${p.relpct.toFixed(1)}% relative), 95% CI excludes zero`;
  } else {
    banner.classList.add('vb-soft');
    badge.textContent = mName(p.winner);
    headline.innerHTML = `<strong>${mName(p.winner)}</strong> leads, but not decisively`;
    sub.textContent = `composite ${fmt3(p.sWin)} vs ${fmt3(p.sLose)} (+${fmt3(p.margin)}), the confidence interval is small, not extremely indicative`;
  }
}

const fmt3 = (x)=> (x==null||isNaN(x)) ? ',' : (x>=0? x.toFixed(3) : x.toFixed(3));
const sgn3 = (x)=> (x>=0?'+':'') + x.toFixed(3);


function componentBars(pm){
  // pm: per_method dict. Build a small aligned bar table for the 4 components.
  const comps = ['fidelity','neighbourhood','shepard','cluster'];
  const methods = Object.keys(pm);
  let html = '<table class="an-comp-table"><thead><tr><th>criterion</th>';
  methods.forEach(m=> html += `<th>${mName(m)}</th>`);
  html += '<th></th></tr></thead><tbody>';
  comps.forEach(c=>{
    const vals = methods.map(m=> (pm[m].components && pm[m].components[c]!=null) ? pm[m].components[c] : null);
    if(vals.every(x=>x==null)) return;
    const best = Math.max(...vals.filter(x=>x!=null));
    html += `<tr><td class="an-comp-name">${c}</td>`;
    methods.forEach((m,i)=>{
      const val = vals[i];
      if(val==null){ html += '<td>,</td>'; return; }
      const isBest = Math.abs(val-best)<1e-9;
      html += `<td class="${isBest?'an-best':''}">${val.toFixed(3)}</td>`;
    });
    // mini bar showing the gap
    const lead = methods[vals.indexOf(best)] || '';
    html += `<td class="an-comp-lead">${lead}</td></tr>`;
  });
  // weights row
  const anyM = methods[0];
  const w = pm[anyM] && pm[anyM].weights_used ? pm[anyM].weights_used : null;
  html += '</tbody></table>';
  return html;
}


function renderAnalysisPanel(d){
  renderVerdictCard(d);
}

function renderVerdictCard(d){
  const v = d.verdict;
  const chip = $('anVerdictChip'), body = $('anVerdictBody'),
        comp = $('anVerdictComponents'), note = $('anVerdictNote');
  if(!v){ chip.textContent='n/a'; body.textContent='Verdict unavailable.'; comp.innerHTML=''; note.textContent=''; return; }
  const p = _verdictParts(v);
  chip.textContent = p.tie ? 'tie' : (p.decisive ? 'decisive' : 'provisional');
  chip.className = 'an-chip ' + (p.decisive ? 'an-chip-good' : (p.tie ? 'an-chip-quiet' : 'an-chip-warn'));
  body.innerHTML = `<p class="an-verdict-sentence">${v.verdict}</p>`;
  // SIMPLIFIED per request: no CI plot, no component bar chart. Just the four-
  // criteria table (each name clickable for an explanation) and the adjustable
  // weights (default 25% each). The composite recomputes live as weights change.
  comp.innerHTML =
    '<div class="verdict-2col">' +
      '<div class="verdict-col-table">' +
        '<h4>The four criteria <span class="an-subtle">click a name for detail</span></h4>' +
        componentBars(v.per_method) +
      '</div>' +
      '<div class="verdict-col-weights">' +
        '<div class="wtsliders wtsliders-sm" id="wtSliders">'+
          '<h4>Adjust the weights <span class="an-subtle">must sum to 100%</span></h4>'+
          wtSliderRow('fidelity', 0.30) + wtSliderRow('neighbourhood', 0.25) +
          wtSliderRow('shepard', 0.20) + wtSliderRow('cluster', 0.25) +
          '<div class="wt-sum-row"><span id="wtSum">sum: 100%</span>'+
            '<button class="ghost-btn ghost-btn-sm" id="wtReset">Reset</button>'+
            '<button class="run-btn run-btn-sm" id="wtRerun">Re-run</button></div>'+
          '<div class="wt-live" id="wtLive"></div>'+
        '</div>' +
      '</div>' +
    '</div>';
  note.innerHTML = '';
  renderVerdictCharts(d);
  bindWeightSliders(d);
  bindCriterionInfo();
}

/* clicking a criterion name in the component table opens its explanation */
const CRITERION_INFO = {
  fidelity: { title: 'Fidelity criterion',
    body: `<p><strong>What it is.</strong> The per-point neighbourhood-preservation error averaged over all points, turned into a 0&ndash;1 score (higher = better). It is the toolkit's own intrusion/extrusion metric: for each point it counts how many of its true high-D neighbours the projection keeps and how many strangers it wrongly pulls in.</p>
    <p><strong>How it's calculated.</strong> Per point, FP = rank-weighted intrusions (points close in 2-D but far in high-D) and FN = rank-weighted extrusions (true neighbours pushed away); the score is 1 &minus; mean(|FP| + |FN|). It is the only <em>per-point</em> component, which is why it usually carries the most weight by default.</p>
    <p><strong>Why it is indicative.</strong> It directly measures the thing the projection is supposed to do, preserve local neighbourhoods, so a method that scores high here distorts fewer points.</p>` },
  neighbourhood: { title: 'Neighbourhood criterion',
    body: `<p><strong>What it is.</strong> How well the projection preserves each point's <em>local neighbourhood</em>, in both directions at once. It consolidates the two classical measures, trustworthiness and continuity (Venna and; Kaski), into a single score by averaging them. In [0, 1], higher = better.</p>
    <p><strong>How it's calculated.</strong> <em>Trustworthiness</em> penalises <em>false neighbours</em> (points close in 2-D but far in high-D); <em>continuity</em> penalises <em>missing neighbours</em> (points close in high-D but pushed apart in 2-D). Each sums the rank displacements of the offending neighbours, normalised so 1 is perfect; the component is their mean.</p>
    <p><strong>Why it is indicative.</strong> It answers &ldquo;can I trust that points shown close together really are close, and that close things stayed together?&rdquo;, the credibility and completeness of the projection's neighbourhoods in one number.</p>` },
  shepard: { title: 'Shepard criterion (global distance preservation)',
    body: `<p><strong>What it is.</strong> How well the projection preserves <em>all</em> pairwise distances, not just neighbourhoods. In [0, 1], higher = the 2-D distances track the high-D distances across the whole dataset.</p>
    <p><strong>How it's calculated.</strong> The Pearson correlation between every pair's high-D distance and its 2-D distance, the numerical version of the classical Shepard diagram (a scatter of high-D vs 2-D distances). Clamped to [0, 1].</p>
    <p><strong>Why it is indicative.</strong> The neighbourhood criterion only checks local structure; this one checks the <em>global</em> geometry, whether far-apart clusters are shown far apart and the overall shape is preserved. A projection can have good neighbourhoods but a distorted global layout, and this catches that.</p>` },
  cluster: { title: 'Cluster-preservation criterion',
    body: `<p><strong>What it is.</strong> How well the projection preserves the data's cluster structure, measured label-free. In [0, 1], higher = the 2-D grouping matches the high-D grouping.</p>
    <p><strong>How it's calculated.</strong> k-means is run in the high-D space and in the 2-D projection, and the two partitions are compared with the Adjusted Rand Index (ARI), which is corrected for chance. No class labels are used, so it is a purely structural comparison.</p>
    <p><strong>Why it is indicative.</strong> It checks whether the clumps you see in the projection correspond to real clumps in the data.</p>` },
};
function bindCriterionInfo(){
  document.querySelectorAll('#anVerdictComponents .an-comp-name').forEach(cell=>{
    const key = cell.textContent.trim().toLowerCase();
    if(!CRITERION_INFO[key]) return;
    cell.classList.add('has-info');
    cell.style.cursor='pointer';
    cell.onclick = ()=>{ const info=CRITERION_INFO[key];
      $('infoTitle').textContent=info.title; $('infoBody').innerHTML=info.body;
      $('infoModal').hidden=false; };
  });
}

/* one weight slider row */
function wtSliderRow(name, def){
  return `<label class="wt-row">`+
    `<span class="wt-name">${name}</span>`+
    `<input type="range" class="wt-slider" data-wt="${name}" min="0" max="1" step="0.05" value="${def}">`+
    `<span class="wt-val" id="wtval-${name}">${def.toFixed(2)}</span></label>`;
}

/* recompute the composite verdict on the frontend as weights change */
function bindWeightSliders(d){
  const box = $('wtSliders'); if(!box) return;
  const v = d.verdict; if(!v || !v.per_method) return;
  const readWeights = ()=>{
    const w = {};
    box.querySelectorAll('.wt-slider').forEach(s=>{ w[s.dataset.wt] = parseFloat(s.value); });
    return w;
  };
  // recompute the verdict under the current weights and show it
  const runVerdict = ()=>{
    const w = readWeights();
    const total = Object.values(w).reduce((a,b)=>a+b,0) || 1;
    const pm = v.per_method, methods = Object.keys(pm);
    const scores = {};
    methods.forEach(m=>{
      let s=0; Object.keys(w).forEach(c=>{
        const cv = pm[m].components && pm[m].components[c]!=null ? pm[m].components[c] : 0;
        s += cv * w[c];
      });
      scores[m] = s/total;
    });
    const ranked = methods.slice().sort((a,b)=>scores[b]-scores[a]);
    const win = ranked[0], lose = ranked[ranked.length-1];
    const margin = scores[win]-scores[lose];
    const live = $('wtLive');
    if(live){
      const flipped = win !== v.winner;
      live.innerHTML =
        `Under these weights: <strong>${win}</strong> ${methods.length>1?`(${scores[win].toFixed(3)}) vs ${lose} (${scores[lose].toFixed(3)})`:''}, `+
        `margin ${margin>=0?'+':''}${margin.toFixed(3)}. `+
        (flipped ? `<span class="wt-flip">The winner flipped from ${v.winner}</span>`
                 : `<span class="wt-same">same winner as the stated weights</span>`);
    }
  };
  // update the live value labels + the sum + enable/disable the re-run button
  const refresh = ()=>{
    box.querySelectorAll('.wt-slider').forEach(s=>{
      const el=$(`wtval-${s.dataset.wt}`); if(el) el.textContent=(parseFloat(s.value)*100).toFixed(0)+'%';
    });
    const w = readWeights();
    const sumPct = Math.round(Object.values(w).reduce((a,b)=>a+b,0)*100);
    const sumEl=$('wtSum'), btn=$('wtRerun');
    const ok = sumPct === 100;
    if(sumEl){ sumEl.textContent=`sum: ${sumPct}%`; sumEl.className = ok ? 'wt-sum-ok' : 'wt-sum-bad'; }
    if(btn){ btn.disabled = !ok; btn.classList.toggle('is-disabled', !ok); }
  };
  box.querySelectorAll('.wt-slider').forEach(s=> s.addEventListener('input', refresh));
  const rr=$('wtRerun'); if(rr) rr.addEventListener('click', ()=>{ if(!rr.disabled) runVerdict(); });
  const rst=$('wtReset'); if(rst) rst.addEventListener('click', ()=>{
    box.querySelectorAll('.wt-slider').forEach(s=>{ s.value='0.25'; });
    refresh(); runVerdict();
  });
  refresh(); runVerdict();   // initial state (25% each sums to 100)
}

/* Grouped component bar chart (PCA vs MDS) + confidence-interval plot. */
function renderVerdictCharts(d){
  // Verdict CI plot and component bar chart removed per request, the verdict
  // card now shows only the four-criteria table and the adjustable weights.
  return;
  /* eslint-disable no-unreachable */
  if(typeof Plotly==='undefined') return;
  const v = d.verdict; if(!v) return;
  const pm = v.per_method;
  const methods = Object.keys(pm);
  const comps = ['fidelity','neighbourhood','shepard','cluster'];
  const present = comps.filter(c => methods.some(m => pm[m].components && pm[m].components[c]!=null));

  // --- grouped bar chart: four components x two methods ---
  const bars = $('verdictBars');
  if(bars){
    const methodColor = {PCA:'#3f7cc4', MDS:'#e0a54b'};
    const traces = methods.map(m=>({
      type:'bar', name:m, x:present,
      y: present.map(c=> pm[m].components && pm[m].components[c]!=null ? pm[m].components[c] : 0),
      marker:{color: methodColor[m] || '#7f9ab5'},
      hovertemplate:`${mName(m)} %{x}: %{y:.3f}<extra></extra>`,
    }));
    Plotly.react(bars, traces, baseLayout({
      barmode:'group',
      xaxis:Object.assign({title:''},AXIS),
      yaxis:Object.assign({title:'score (0–1)', range:[0,1]},AXIS),
      showlegend:true, legend:{font:{size:10,color:'#9aa7b6'},orientation:'h',x:0,y:1.12},
      height:220, margin:{l:48,r:14,t:24,b:30},
    }), PLOT_CONFIG);
  }

  // --- confidence-interval plot: mean distortion gap + 95% CI ---
  const ciDiv = $('verdictCI');
  const ci = v.bootstrap_ci_mean_distortion_diff;
  if(ciDiv && ci){
    const lo = ci[0], hi = ci[1], mean = (lo+hi)/2;
    const decisive = lo > 0 || hi < 0;               // CI excludes zero
    const col = decisive ? '#4ec97a' : '#e0a54b';
    // pad x-range to include zero and the interval
    const pad = Math.max(0.01, (hi-lo)*0.4);
    const xlo = Math.min(0, lo) - pad, xhi = Math.max(0, hi) + pad;
    Plotly.react(ciDiv, [{
      type:'scatter', mode:'markers', x:[mean], y:[0],
      error_x:{type:'data', symmetric:false, array:[hi-mean], arrayminus:[mean-lo],
        color:col, thickness:2, width:8},
      marker:{color:col, size:14, line:{color:'#0e1116',width:1}},
      hovertemplate:`gap ${mean.toFixed(4)}, 95% CI [${lo.toFixed(4)}, ${hi.toFixed(4)}]<extra></extra>`,
    }], baseLayout({
      xaxis:Object.assign({title:'mean distortion gap (loser − winner)', range:[xlo,xhi], zeroline:false},AXIS),
      yaxis:Object.assign({showticklabels:false, range:[-1,1]},AXIS),
      height:130, margin:{l:14,r:14,t:22,b:36},
      shapes:[{type:'line', x0:0, x1:0, y0:-1, y1:1, line:{color:'#d24b52',width:1.5,dash:'dash'}}],
      annotations:[
        {x:0, y:-0.85, text:'zero (no difference)', showarrow:false, font:{size:9,color:'#d24b52'}, xanchor: (mean>=0?'left':'right'), xshift:(mean>=0?3:-3)},
        {x:mean, y:0.82, text: decisive ? '✓ CI excludes zero, reliable' : '✗ CI crosses zero, provisional',
         showarrow:false, font:{size:10, color:col}, xanchor:'center'},
      ],
    }), PLOT_CONFIG);
  }
}


/* verdict modal */
function openVerdictModal(){
  const d = state.result; if(!d || !d.verdict) return;
  const v = d.verdict, p = _verdictParts(v);
  const body = $('vmBody');
  let html = `<div class="vm-headline ${p.decisive?'vm-good':(p.tie?'vm-tie':'vm-soft')}">${v.verdict}</div>`;
  html += '<h4>The four criteria</h4>' + componentBars(v.per_method);
  html += `<div class="an-note" style="margin-top:12px">`+
    `<strong>Confidence.</strong> Cohen's d = ${fmt3(p.d)}; 95% bootstrap CI on the per-point distortion gap = [${sgn3(p.ci[0])}, ${sgn3(p.ci[1])}]. `+
    (p.decisive ? 'The interval excludes zero, the win is reliable.'
                : (p.tie ? 'Within the tie threshold.' : 'The interval crosses zero, provisional.'))+
    `</div>`;
  body.innerHTML = html;
  $('verdictModal').hidden = false;
}
function closeVerdictModal(){ $('verdictModal').hidden = true; }