/* ====================================================================
   QUARANTINED -- NOT LOADED BY index.html
   ====================================================================
   Simulated-annealing card, cluster-structure metrics card, k-means
   ground-truth cards and the class-aware toggle readout.
   The backend no longer computes anneal / rel_metrics / kmeans.

   This file is kept for reference only. It is not referenced by any <script>
   tag, so it costs nothing at runtime. Re-adding it requires restoring both
   the markup it targets and the backend fields it reads.
   ==================================================================== */

/** In the trust panel: show the geometric-vs-class-aware divergence, and when
 *  class-aware is active, surface the k-means ground-truth results. */
function renderClassAwareDelta(d){
  // sync the checkbox to whatever produced this run
  const cb = $('classAware');
  const isClassAware = !!(d.params && d.params.class_aware);
  if(cb && d.params && typeof d.params.class_aware === 'boolean') cb.checked = isClassAware;

  const info = $('classAwareInfo');
  const cad = d.class_aware_delta;
  /*
  if(info){
    if(cad){
      info.hidden = false;
      info.innerHTML =
        `<strong>${isClassAware ? 'Class-aware' : 'Geometric'}</strong> fidelity is active. `+
        `The two modes differ in sign on <strong>${cad.sign_flip_pct}%</strong> of points `+
        `(correlation ${cad.correlation}), class-aware counts only cross-class intrusions and same-class extrusions, so it measures whether the labelled <em>class</em> structure survived, not raw geometry.`;
    } else { info.hidden = true; info.innerHTML=''; }
  }
    */
  // legacy element (may be absent now)
  const el = $('classAwareDelta');
  if(el && cad) el.innerHTML = `differ on <strong>${cad.sign_flip_pct}%</strong> of points`;

  // k-means results, shown only when class-aware is on
  renderTrustKmeans(d, isClassAware);
}

/** Compact k-means ground-truth panel for the trust analysis (class-aware on). */
function renderTrustKmeans(d, show){
  const box = $('trustKmeans');
  if(!box) return;
  const km = d.kmeans;
  if(!show || !km){ box.hidden = true; box.innerHTML=''; return; }
  box.hidden = false;
  const hi = km.high_d || {};
  const ariL = hi.ari_vs_labels!=null ? hi.ari_vs_labels.toFixed(3) : ',';
  const acc  = hi.cluster_accuracy!=null ? (hi.cluster_accuracy*100).toFixed(0)+'%' : ',';
  const are  = km.classes_are_clusters;
  const nCls = km.n_classes, silk = km.best_k_by_silhouette;
  // per-method ARI (does each projection preserve the high-D clustering?)
  let methodRows = '';
  if(km.methods){
    methodRows = Object.keys(km.methods).map(m=>{
      const a=km.methods[m].ari_vs_high;
      return `<span class="kmm">${m}: ARI 2-D vs high-D <strong>${a!=null?a.toFixed(3):','}</strong></span>`;
    }).join('');
  }
  const verdict = are
    ? `<span class="km-good">✓ the labels align with the data's natural clusters</span>`
    : `<span class="km-warn">✗ the labels do <strong>not</strong> match the natural clusters, class-aware scores here reflect a labelling/geometry mismatch, not the projection</span>`;
  box.innerHTML =
    `<div class="km-head">k-means ground-truth check <span class="an-subtle">(runs with class-aware)</span></div>`+
    `<div class="km-stats">`+
      `<span class="kmm">ARI vs labels <strong>${ariL}</strong></span>`+
      `<span class="kmm">matched accuracy <strong>${acc}</strong></span>`+
      `<span class="kmm">natural clusters (silhouette) <strong>${silk}</strong> vs <strong>${nCls}</strong> classes</span>`+
      methodRows+
    `</div>`+
    `<div class="km-verdict">${verdict}</div>`;
}


/* DATA & LABELS verdict: do the class labels describe the data's structure?
   Driven by k-means ARI vs labels, with the silhouette cluster count as
   supporting evidence. Three tiers so it never over-claims. */
function renderLabelsBanner(d){
  const banner = $('labelsBanner');
  const km = d.kmeans;
  if(!km){ banner.hidden = true; return; }
  banner.hidden = false;
  const hi = km.high_d || {};
  const ari = hi.ari_vs_labels;
  const nCls = km.n_classes;
  const silK = km.best_k_by_silhouette;
  const badge = $('lbBadge'), headline = $('lbHeadline'), sub = $('lbSub');
  banner.classList.remove('vb-decisive','vb-tie','vb-soft');

  // tier by ARI: >=0.7 good, 0.4-0.7 partial, <0.4 poor
  let tier;
  if(ari == null) tier = 'partial';
  else if(ari >= 0.7) tier = 'good';
  else if(ari >= 0.4) tier = 'partial';
  else tier = 'poor';

  const ariStr = ari!=null ? ari.toFixed(2) : ',';
  const silNote = (silK!=null && nCls!=null && silK!==nCls)
    ? ` The data looks like it has ${silK} natural group${silK>1?'s':''}, not ${nCls}.`
    : (silK!=null ? ` The data's natural group count (${silK}) matches your ${nCls} labels.` : '');

  if(tier === 'good'){
    banner.classList.add('vb-decisive');       // green styling
    badge.textContent = 'LABELS ✓';
    headline.innerHTML = `The labels <strong>describe the data well</strong>`;
    sub.textContent = `k-means clusters match your labels (ARI ${ariStr}).${silNote} Class-based scores are trustworthy here.`;
  } else if(tier === 'poor'){
    banner.classList.add('vb-soft');            // amber styling
    badge.textContent = 'LABELS ✗';
    headline.innerHTML = `The labels <strong>do NOT match</strong> the data's structure`;
    sub.textContent = `k-means finds a different grouping (ARI ${ariStr}).${silNote} Prefer the geometric (label-free) metric; a "bad" projection may be the labels' fault.`;
  } else {
    banner.classList.add('vb-tie');             // neutral styling
    badge.textContent = 'LABELS ~';
    headline.innerHTML = `The labels <strong>partly match</strong> the data`;
    sub.textContent = `k-means only partially agrees with your labels (ARI ${ariStr}).${silNote} Read class-based scores with some caution.`;
  }
}


function renderAnnealCard(d){
  const a = d.anneal;
  const chip = $('anAnnealChip'), stats = $('anAnnealStats'),
        note = $('anAnnealNote'), plotDiv = $('anAnnealPlot');
  if(!a){
    chip.textContent = 'off';
    stats.innerHTML = '<p class="an-muted">Simulated-annealing search did not run for this dataset.</p>';
    note.textContent=''; if(typeof Plotly!=='undefined' && plotDiv) Plotly.purge(plotDiv);
    return;
  }
  chip.textContent = `k* = ${a.best_k}`;
  const grid = a.exhaustive_count != null ? a.exhaustive_count
             : (a.k_hi != null && a.k_lo != null ? (a.k_hi - a.k_lo + 1) : ',');
  // compute acceptance rate of WORSENING proposals: of all proposals that were
  // worse than the current point, what fraction were accepted anyway?
  let worseProposed = 0, worseAccepted = 0, startScore = null, endBest = null;
  if(a.trace && a.trace.length){
    startScore = a.trace[0].best_score;
    endBest = a.trace[a.trace.length-1].best_score;
    let curBest = a.trace[0].score;
    for(let i=0;i<a.trace.length;i++){
      const t = a.trace[i];
      // a proposal is "worse" if its score is below the running current;
      // approximate using best_score progression + uphill flags
      if(t.uphill){ worseProposed++; worseAccepted++; }
    }
    // count rejected worsening moves: iterations not accepted (score dipped, not counted as uphill)
    for(let i=0;i<a.trace.length;i++){ if(a.trace[i].accepted===false) worseProposed++; }
  }
  const accRate = worseProposed>0 ? Math.round(100*worseAccepted/worseProposed) : null;
  stats.innerHTML =
    statCell('best k', a.best_k) +
    statCell('F1 at k*', (a.best_score!=null? a.best_score.toFixed(4):',')) +
    statCell('evaluations', `${a.n_evaluations}`) +
    statCell('vs exhaustive', `${a.n_evaluations}/${grid}`) +
    statCell('uphill accepted', a.uphill_moves_accepted!=null?a.uphill_moves_accepted:',') +
    statCell('worsening accept rate', accRate!=null?`${accRate}%`:',') +
    statCell('start temp', a.t0!=null?a.t0.toFixed(3):',') +
    statCell('F1 gain', (startScore!=null&&endBest!=null)?`+${(endBest-startScore).toFixed(3)}`:',');

  // self-diagnosis: healthy vs problematic search
  let diag = '';
  if(a.trace && a.trace.length){
    const n = a.trace.length;
    const improvedLate = (a.trace[n-1].best_score - a.trace[Math.floor(n*0.8)].best_score) > 1e-6;
    const flatFromStart = Math.abs(endBest - startScore) < 1e-6;
    const noUphill = (a.uphill_moves_accepted||0) === 0;
    const gained = (endBest - startScore) > 1e-6;
    const items = [];
    // good signs
    if(gained) items.push(`<li class="diag-good">✓ final best F1 (+${(endBest-startScore).toFixed(3)}) exceeds the start, the search improved</li>`);
    if((a.uphill_moves_accepted||0) > 0 && !improvedLate) items.push(`<li class="diag-good">✓ it explored early (worsening moves accepted) then settled, healthy cooling</li>`);
    // problem signs
    if(flatFromStart) items.push(`<li class="diag-bad">✗ best F1 never improved, likely stuck in a local optimum (temperature too low)</li>`);
    if(improvedLate) items.push(`<li class="diag-warn">best F1 was still rising at the end, may be under-cooled; consider more iterations</li>`);
    if(noUphill && !flatFromStart) items.push(`<li class="diag-warn">no worsening moves were accepted, the search was greedy and may have missed better regions</li>`);
    if(!items.length) items.push(`<li class="diag-good">✓ search behaved normally</li>`);
    diag = `<div class="diag-box"><div class="diag-title">Search health</div><ul class="diag-list">${items.join('')}</ul></div>`;
  }

  const reheatTxt = a.reheats ? ` It re-heated ${a.reheats} time${a.reheats>1?'s':''} after stalling, to escape local optima.` : '';
  note.innerHTML =
    `<strong>What a healthy search looks like.</strong> The best F1 should exceed the starting F1 (a real improvement), a few worsening moves should be accepted early (exploration), and the search should touch far fewer than the exhaustive count of k values.${reheatTxt}` +
    diag;

  // Plot intentionally omitted, only the numeric summary is shown.
  if(typeof Plotly!=='undefined' && plotDiv) Plotly.purge(plotDiv);
}

function statCell(label, value){
  return `<div class="an-stat"><span class="an-stat-v">${value}</span><span class="an-stat-l">${label}</span></div>`;
}

function renderMetricsCard(d){
  const r = d.rel_metrics;
  const chip = $('anMetricsChip'), grid = $('anMetricsGrid'), pc = $('anMetricsPerClass');
  const stale = document.getElementById('anMacroMicro'); if(stale) stale.remove();
  if(!r){ chip.textContent='n/a'; grid.innerHTML=''; pc.innerHTML=''; return; }
  const g = r.global;
  const gm = r.global_micro;
  chip.textContent = `k = ${r.k}`;
  grid.innerHTML =
    bigStat('purity', g.precision, 'are points among their own class?') +
    bigStat('cohesion', g.recall, 'is each class held together?') +
    bigStat('F1', (g.f1!=null?g.f1:g.f_score), 'harmonic mean of the two', true);
  // macro vs micro line: the gap reveals class imbalance
  if(gm){
    const macroF1 = g.f1!=null?g.f1:g.f_score, microF1 = gm.f1!=null?gm.f1:gm.f_score;
    const gap = Math.abs(macroF1 - microF1);
    const gapNote = gap>0.05
      ? `Because they differ by ${gap.toFixed(3)}, the classes are <strong>imbalanced or uneven</strong>, some classes score much better than others, and the small classes are pulling the macro score ${macroF1<microF1?'down':'up'}.`
      : `They are close (gap ${gap.toFixed(3)}), so every class scores about equally well, the overall F1 is representative of each class.`;
    grid.insertAdjacentHTML('afterend',
      `<div class="an-note macro-micro" id="anMacroMicro">`+
      `<div class="mm-row"><span class="mm-tag mm-macro">MACRO F1 ${macroF1.toFixed(3)}</span>`+
      `<span class="mm-desc">treats every <em>class</em> equally: average the F1 within each class, then average those, so a tiny class counts as much as a huge one.</span></div>`+
      `<div class="mm-row"><span class="mm-tag mm-micro">MICRO F1 ${microF1.toFixed(3)}</span>`+
      `<span class="mm-desc">treats every <em>point</em> equally: pool all points together, so large classes dominate.</span></div>`+
      `<div class="mm-gap">${gapNote}</div>`+
      `</div>`);
  }
  // per-class table
  const names = r.target_names || [];
  const pl = r.per_label || {};
  const keys = Object.keys(pl);
  if(keys.length){
    // one small radar per class, laid out in a responsive grid, so each class's
    // precision/recall/F1 shape can be read individually
    pc.innerHTML = '<h4>Per class (purity, cohesion, F1)</h4>'+
      '<p class="an-note" style="margin-bottom:8px">Each class has its own radar. A '+
      'balanced triangle means the class is pure <em>and</em> cohesive; a spike '+
      'toward one axis shows where it is weak (e.g. high purity but low cohesion = '+
      'pure but fragmented).</p>'+
      '<div class="radar-grid" id="radarGrid"></div>';
    const gridEl = $('radarGrid');
    if(typeof Plotly!=='undefined' && gridEl){
      const axes = ['purity','cohesion','F1'];
      const palette = ['#3f7cc4','#e0a54b','#4ec97a','#d24b52','#9b7cc4','#5ac8c8','#c88a4a','#8a94a0','#e88a8f','#82abd8'];
      keys.forEach((k,idx)=>{
        const row = pl[k];
        const f1 = row.f1!=null ? row.f1 : row.f_score;
        const nm = names[k] !== undefined ? String(names[k]) : `class ${k}`;
        const rvals = [row.precision, row.recall, f1];
        const col = palette[idx % palette.length];
        // container per class
        const cell = window.document.createElement('div');
        cell.className = 'radar-cell';
        const title = window.document.createElement('div');
        title.className = 'radar-cell-title';
        title.textContent = `${nm} (n=${row.size})`;
        const plotDiv = window.document.createElement('div');
        plotDiv.className = 'radar-cell-plot';
        plotDiv.id = `radar-${idx}`;
        cell.appendChild(title); cell.appendChild(plotDiv);
        gridEl.appendChild(cell);
        Plotly.react(plotDiv, [{
          type:'scatterpolar', mode:'lines+markers',
          r: rvals.concat([rvals[0]]), theta: axes.concat([axes[0]]),
          line:{color:col,width:2}, marker:{size:5,color:col},
          fill:'toself', fillcolor:hexToRgba(col,0.25),
          hovertemplate:`${nm} %{theta}: %{r:.3f}<extra></extra>`,
        }], baseLayout({
          polar:{ bgcolor:'#0b0e13',
            radialaxis:{range:[0,1], tickvals:[0.5,1], tickfont:{size:7,color:'#63707f'}, gridcolor:'#262f3b', linecolor:'#262f3b'},
            angularaxis:{tickfont:{size:9,color:'#9aa7b6'}, gridcolor:'#262f3b', linecolor:'#262f3b'} },
          showlegend:false, height:180, margin:{l:28,r:28,t:16,b:16},
        }), PLOT_CONFIG);
      });
    }
  } else pc.innerHTML='';
}

/** Convert a #rrggbb hex to an rgba() string with the given alpha. */
function hexToRgba(hex, a){
  const h = hex.replace('#','');
  const r = parseInt(h.slice(0,2),16), g = parseInt(h.slice(2,4),16), b = parseInt(h.slice(4,6),16);
  return `rgba(${r},${g},${b},${a})`;
}

function bigStat(label, val, sub, hero){
  const v = (val==null||isNaN(val)) ? ',' : val.toFixed(3);
  return `<div class="an-bigstat ${hero?'an-bigstat-hero':''}">`+
    `<div class="an-bigstat-v">${v}</div>`+
    `<div class="an-bigstat-l">${label}</div>`+
    `<div class="an-bigstat-s">${sub}</div></div>`;
}

function renderKmeansCard(d){
  const km = d.kmeans;
  const chip = $('anKmeansChip'), body = $('anKmeansBody');
  if(!km){ chip.textContent='n/a'; body.innerHTML=''; return; }
  const hi = km.high_d || {};
  const are = km.classes_are_clusters;
  const method = km.cluster_method || 'kmeans';
  const methodName = {kmeans:'k-means', gmm:'GMM', spectral:'spectral'}[method] || method;
  chip.textContent = (are ? 'labels ≈ clusters' : 'labels ≠ clusters') + `, ${methodName}`;
  chip.className = 'an-chip ' + (are ? 'an-chip-good' : 'an-chip-warn');
  const acc = hi.cluster_accuracy!=null ? (100*hi.cluster_accuracy).toFixed(1)+'%' : ',';
  const ariL = hi.ari_vs_labels!=null ? hi.ari_vs_labels.toFixed(3) : ',';
  let html =
    '<div class="an-stat-row">' +
    statCell('ARI vs labels', ariL) +
    statCell('matched accuracy', acc) +
    statCell('silhouette best k', km.best_k_by_silhouette) +
    statCell('# classes', km.n_classes) +
    '</div>';
  // verdict sentence
  if(are){
    html += `<p class="an-kmeans-say an-say-good">The high-dimensional clusters line up with the class labels (ARI ${ariL}), so the labels are a trustworthy reference here.</p>`;
  } else {
    html += `<p class="an-kmeans-say an-say-warn">The natural clusters do <strong>not</strong> match the labels well (ARI ${ariL})`;
    if(km.best_k_by_silhouette !== km.n_classes){
      html += `, silhouette even prefers <strong>${km.best_k_by_silhouette}</strong> clusters against <strong>${km.n_classes}</strong> classes`;
    }
    html += `. Part of any apparent "projection error" may therefore be a label/geometry mismatch, not the projection's fault.</p>`;
    if(method === 'kmeans'){
      html += `<p class="an-note">If the classes might be <em>non-convex</em> (e.g. rings or crescents), k-means cannot see them, try the <strong>spectral</strong> clusterer in the Analysis controls before concluding the labels are bad.</p>`;
    }
  }
  // per-method preservation
  if(km.methods){
    html += '<h4>Does each projection preserve the clustering?</h4><div class="an-stat-row">';
    Object.keys(km.methods).forEach(m=>{
      const ari = km.methods[m].ari_vs_high;
      html += statCell(`${m}: ARI 2-D vs high-D`, ari!=null?ari.toFixed(3):',');
    });
    html += '</div>';
  }
  // containers for the two charts
  html += '<h4>Cluster vs class confusion</h4>'+
          '<p class="an-note" style="margin-bottom:8px">Each cell is the number of points in a high-D cluster (row) that carry a given class label (column). A clean result is <strong>block-diagonal</strong>: each cluster is dominated by one class. <strong>Cluster purity</strong> = the biggest count in a row &divide; that row&rsquo;s total (how single-class each cluster is). <strong>Class coverage</strong> = the biggest count in a column &divide; that column&rsquo;s total (how much of each class landed in one cluster).</p>'+
          '<div class="plot" id="kmHeatmap" style="height:230px"></div>'+
          '<div id="kmPurity"></div>';
  body.innerHTML = html;
  renderKmeansCharts(d);
}

/* Confusion heatmap (high-D clusters x class labels) + silhouette-vs-k plot. */
function renderKmeansCharts(d){
  if(typeof Plotly==='undefined') return;
  const km = d.kmeans; if(!km) return;
  const names = (d.rel_metrics && d.rel_metrics.target_names) || d.target_names || [];

  // --- confusion heatmap ---
  const hm = $('kmHeatmap');
  if(hm && km.contingency && km.contingency.length){
    const M = km.contingency;                       // rows=clusters, cols=classes
    const nClusters = M.length, nClasses = M[0].length;
    const flatMax = Math.max(...M.flat(), 1);
    const classNames = Array.from({length:nClasses}, (_,j)=> names[j]!==undefined?String(names[j]):`class ${j}`);
    const clusterNames = Array.from({length:nClusters}, (_,i)=>{
      const ml = km.matched_labels && km.matched_labels[i]!=null ? km.matched_labels[i] : null;
      const mlName = (ml!=null && names[ml]!==undefined) ? String(names[ml]) : (ml!=null?`class ${ml}`:'');
      return mlName ? `cl ${i} → ${mlName}` : `cluster ${i}`;
    });
    const ann = [];
    for(let i=0;i<nClusters;i++) for(let j=0;j<nClasses;j++){
      const v = M[i][j];
      ann.push({x:classNames[j], y:clusterNames[i], text:String(v), showarrow:false,
        font:{size:11, color: v > 0.5*flatMax ? '#0e1116' : '#e8edf3'}});
    }
    Plotly.react(hm, [{
      type:'heatmap', z:M, x:classNames, y:clusterNames,
      colorscale:[[0,'#0e1116'],[0.5,'#2a5a8a'],[1,'#4ec97a']],
      showscale:true, colorbar:{title:{text:'points',font:{size:9,color:'#9aa7b6'}}, thickness:10, len:0.9},
      hovertemplate:'cluster %{y}, %{x}: %{z} points<extra></extra>',
    }], baseLayout({
      xaxis:Object.assign({title:'class label', side:'bottom'},AXIS),
      yaxis:Object.assign({title:'high-D cluster', automargin:true},AXIS),
      annotations:ann, height:230, margin:{l:100,r:20,t:10,b:44},
    }), PLOT_CONFIG);

    // --- cluster purity (per row) and class coverage (per column) ---
    const purityDiv = $('kmPurity');
    if(purityDiv){
      const rowTotals = M.map(r=>r.reduce((a,b)=>a+b,0));
      const colTotals = classNames.map((_,j)=> M.reduce((a,r)=>a+r[j],0));
      const rowPurity = M.map((r,i)=> rowTotals[i]>0 ? Math.max(...r)/rowTotals[i] : 0);
      const colCoverage = classNames.map((_,j)=>{
        const col = M.map(r=>r[j]); return colTotals[j]>0 ? Math.max(...col)/colTotals[j] : 0;
      });
      const overallPurity = rowTotals.reduce((a,b)=>a+b,0)>0
        ? M.reduce((a,r)=>a+Math.max(...r),0) / rowTotals.reduce((a,b)=>a+b,0) : 0;
      const pill = (label,val)=>{
        const pct = (val*100).toFixed(0);
        const cls = val>=0.8?'pcell-good':(val>=0.5?'pcell-mid':'pcell-bad');
        return `<span class="pcell ${cls}"><span class="pcell-l">${label}</span><span class="pcell-v">${pct}%</span></span>`;
      };
      let ph = `<div class="purity-head">Overall cluster purity: <strong>${(overallPurity*100).toFixed(0)}%</strong> of points fall in a cluster dominated by their own class</div>`;
      ph += '<div class="purity-grid"><div class="purity-col"><div class="purity-col-t">Cluster purity (each cluster is how single-class?)</div>';
      rowPurity.forEach((p,i)=> ph += pill(clusterNames[i], p));
      ph += '</div><div class="purity-col"><div class="purity-col-t">Class coverage (each class is how concentrated in one cluster?)</div>';
      colCoverage.forEach((cvg,j)=> ph += pill(classNames[j], cvg));
      ph += '</div></div>';
      purityDiv.innerHTML = ph;
    }
  } else {
    const purityDiv = $('kmPurity'); if(purityDiv) purityDiv.innerHTML='';
  }
}
