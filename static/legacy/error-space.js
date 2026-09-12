/* ====================================================================
   QUARANTINED -- NOT LOADED BY index.html
   ====================================================================
   Error Space panel and the double-encoding colour/size channels.
   The panel was removed from index.html; nothing here is loaded.

   This file is kept for reference only. It is not referenced by any <script>
   tag, so it costs nothing at runtime. Re-adding it requires restoring both
   the markup it targets and the backend fields it reads.
   ==================================================================== */

/* error scale colour (matches CSS: red FN → white → blue FP) */
const FN = [210, 75, 82], TRUE_ = [238, 241, 244], FP = [63, 124, 196];
function lerp(a, b, t){ return a + (b - a) * t; }
function scaleColor(v, vmax){
  // v in [-vmax, vmax] → rgb string on FN..TRUE..FP
  const t = Math.max(-1, Math.min(1, v / vmax));
  let c;
  if (t < 0){ const k = t + 1; c = [lerp(FN[0],TRUE_[0],k), lerp(FN[1],TRUE_[1],k), lerp(FN[2],TRUE_[2],k)]; }
  else      { const k = t;     c = [lerp(TRUE_[0],FP[0],k), lerp(TRUE_[1],FP[1],k), lerp(TRUE_[2],FP[2],k)]; }
  return `rgb(${c.map(x=>Math.round(x)).join(',')})`;
}
const DIVERGING = [
  [0.0, 'rgb(210,75,82)'], [0.25, 'rgb(224,138,143)'],
  [0.5, 'rgb(238,241,244)'],
  [0.75, 'rgb(130,171,216)'], [1.0, 'rgb(63,124,196)']
];

function renderSignal(d){
  // signature: header signal shows overall FN vs FP mass balance
  const all = d.table.pca.err.concat(d.table.mds.err);
  let fnMass=0, fpMass=0;
  all.forEach(v=>{ if(v<0) fnMass+=-v; else fpMass+=v; });
  const total = fnMass+fpMass || 1;
  
  // Calculate percentages of the FULL bar (not half)
  const fnPct = (fnMass/total) * 100;
  const fpPct = (fpMass/total) * 100;
  
  // Set widths - these now add up to 100% of the bar
  $('signalFn').style.width = fnPct + '%';
  $('signalFp').style.width = fpPct + '%';
  
  const label = $('signalLabel');
  const readout = $('signalReadout');
  const fnPercent = fnPct.toFixed(0);
  const fpPercent = fpPct.toFixed(0);
  
  // Color the label based on dominance
  if(fnMass > fpMass * 1.1){ // FN dominates
    label.style.color = '#d24b52';
    label.className = 'signal-label fn-dominant';
    readout.textContent = `${fnPercent}% FN, ${fpPercent}% FP, FN-dominated`;
  } else if(fpMass > fnMass * 1.1){ // FP dominates
    label.style.color = '#3f7cc4';
    label.className = 'signal-label fp-dominant';
    readout.textContent = `${fnPercent}% FN, ${fpPercent}% FP, FP-dominated`;
  } else { // balanced
    label.style.color = '#63707f';
    label.className = 'signal-label balanced';
    readout.textContent = `${fnPercent}% FN, ${fpPercent}% FP, Balanced`;
  }
}

/* ============================================================
   DOUBLE ENCODING
   Colour alone is ambiguous: err = FP - FN, so a point with
   FP = 0.6 and FN = 0.6 reads err = 0 -> white, exactly like a
   point with FP = FN = 0 that is genuinely well embedded. The two
   are opposites (severely distorted both ways vs faithful), so we
   add a SECOND channel: marker size encodes total distortion
   |FP| + |FN|. White + large = badly distorted but balanced;
   white + small = truly faithful.
   ============================================================ */
const ES_SIZE_MIN = 5;    // px diameter for zero total distortion
const ES_SIZE_MAX = 20;   // px diameter for the run's worst total distortion

/** total distortion |FP| + |FN| per point (the magnitude colour cannot show) */
function totalDistortion(t){
  return t.fp.map((v,i)=> Math.abs(v) + Math.abs(t.fn[i]));
}

/* ============================================================
   SEVERITY-WEIGHTED SIGNED ERROR  (colour metric)
   The plain signed error err = FP - FN answers "which way?" but
   not "how much is behind it": FP=.5,FN=.4 and FP=.1,FN=0 both
   give err = +0.1, yet the first point is five times more
   distorted. We therefore weight the error by the magnitude that
   produced it, using the signed geometric mean

       S = sign(err) * sqrt( |err| * (FP + FN) )
         = sign(FP-FN) * sqrt( |FP^2 - FN^2| )

   Properties (all verified):
     - sign(S) == sign(err): the FN / True / FP semantics survive.
     - S == 0 exactly when err == 0: balance still reads neutral,
       which is why SIZE remains the second channel for those points.
     - If one side is zero, S == err: it degrades to the familiar scale.
     - |S| grows with both the imbalance AND the mass behind it.
   The geometric mean (rather than the raw product err*(FP+FN)) is
   essential: the raw product crushes ~92% of real points into the
   near-white band, destroying contrast, because it is quadratic in
   quantities that are already << 1. The square root restores a
   perceptually usable spread.                                     */
function severityError(t){
  return t.fp.map((fp,i)=>{
    const fn = t.fn[i], e = t.err[i], dist = Math.abs(fp) + Math.abs(fn);
    return Math.sign(e) * Math.sqrt(Math.abs(e) * dist);
  });
}

/** Symmetric colour limit for the severity scale, rounded up for clean ticks. */
function severityVmax(d){
  let peak = 0;
  ['pca','mds'].forEach(m=>{
    severityError(d.table[m]).forEach(v=>{ peak = Math.max(peak, Math.abs(v)); });
  });
  return peak > 0 ? Math.ceil(peak*20)/20 : 0.1;
}
/** Colour channel used by the Error Space plots: severity-weighted signed error. */
function colourChannel(t, d){
  return { vals: severityError(t), vmax: severityVmax(d),
           label: 'severity-weighted error' };
}
/** Per-point class colours for the "see classes" view. */
function classColorArray(d){
  return d.table.labels.map(l=> classColor(l));
}
/** Peak |FP|+|FN| across BOTH methods, so PCA and MDS share one size scale. */
function distortionMax(d){
  const all = totalDistortion(d.table.pca).concat(totalDistortion(d.table.mds));
  const m = Math.max(...all, 0);
  return m > 0 ? m : 1;   // guard: all-zero run -> every point stays at min size
}
/** Map total distortion -> marker diameter. sqrt so AREA grows ~linearly with
 *  distortion (perceptually fairer than scaling the diameter directly). */
function distortionSizes(tot, dmax){
  return tot.map(v=> ES_SIZE_MIN +
    (ES_SIZE_MAX - ES_SIZE_MIN) * Math.sqrt(Math.min(v, dmax) / dmax));
}

/* comparison scatter */
/* histograms
   Bars are centred on their bin's midpoint (so a bar sits symmetrically over
   the error value it represents) and hovering one reports the exact half-open
   interval it covers. Horizontal reference lines summarise the bin COUNTS:
   a green median line plus the two dashed +/-2 sigma lines bounding a shaded
   band, so unusually tall bars are immediately obvious.               */
function renderHistograms(d){
  if(typeof Plotly==='undefined') return;
  const median = (arr)=>{ const s=[...arr].sort((a,b)=>a-b), n=s.length;
    return n? (n%2? s[(n-1)/2] : (s[n/2-1]+s[n/2])/2) : 0; };

  // per-point total distortion (FP + FN) and balance (|FP-FN|/(FP+FN)) per method
  function components(t){
    const dist=[], bal=[];
    for(let i=0;i<t.fp.length;i++){
      const fp=Math.abs(t.fp[i]), fn=Math.abs(t.fn[i]), tot=fp+fn;
      dist.push(tot);
      if(tot > 1e-9) bal.push(Math.abs(fp-fn)/tot);   // undefined for faithful points
    }
    return {dist, bal};
  }
  const cP = components(d.table.pca), cM = components(d.table.mds);

  // shared x-max for distortion so PCA/MDS are comparable
  const distMax = Math.max(0.05, ...cP.dist, ...cM.dist);

  function histData(vals, lo, hi, bins){
    const edges=[]; for(let i=0;i<=bins;i++) edges.push(lo + (hi-lo)*i/bins);
    const counts=new Array(bins).fill(0);
    vals.forEach(v=>{ let b=Math.floor((v-lo)/(hi-lo)*bins); if(b>=bins)b=bins-1; if(b<0)b=0; counts[b]++; });
    const centres=edges.slice(0,bins).map((e,i)=>(e+edges[i+1])/2);
    return {edges, centres, counts};
  }

  // colour ramps: distortion cool→warm (more distortion = warmer/redder);
  // balance cool (balanced, the dangerous "cancels to zero" case) → neutral
  const distColor = (c,mx)=>{ const t=Math.min(c/mx,1);
    return `hsl(${Math.round(210-150*t)}, 55%, ${Math.round(58-6*t)}%)`; };

  function buildDist(divId, vals, label){
    const H = histData(vals, 0, distMax, 30);
    const med = median(vals), mean = vals.reduce((a,b)=>a+b,0)/(vals.length||1);
    const custom = H.counts.map((c,i)=>[H.edges[i].toFixed(3), H.edges[i+1].toFixed(3), (100*c/(vals.length||1)).toFixed(1)]);
    const bar={ type:'bar', x:H.centres, y:H.counts,
      marker:{color:H.centres.map(c=>distColor(c,distMax)), line:{color:'rgba(0,0,0,0.4)',width:0.4}},
      width:(distMax/30)*0.92, customdata:custom,
      hovertemplate:'distortion [%{customdata[0]}, %{customdata[1]})<br><b>%{y}</b> points, %{customdata[2]}%<extra></extra>',
      showlegend:false };
    const hmax=Math.max(...H.counts,1)*1.15;
    const medLine={ type:'scatter',mode:'lines',x:[med,med],y:[0,hmax],
      line:{color:'#4ec97a',width:1.4,dash:'dot'},
      name:`median ${med.toFixed(3)}`, showlegend:true,
      hovertemplate:`median distortion ${med.toFixed(3)}<extra></extra>` };
    Plotly.react(divId,[bar,medLine], baseLayout({
      title:{text:`${label}, n=${vals.length}, mean ${mean.toFixed(3)}`, font:{size:11,color:'#9aa7b6'},x:0.02,xanchor:'left'},
      bargap:0.04, showlegend:true, legend:{font:{size:9,color:'#9aa7b6'},x:0.99,xanchor:'right',y:0.99,bgcolor:'rgba(21,26,33,0.78)'},
      xaxis:Object.assign({title:'total distortion  FP + FN',range:[0,distMax]},AXIS),
      yaxis:Object.assign({title:'count',rangemode:'tozero',range:[0,hmax]},AXIS),
    }),PLOT_CONFIG);
  }

  function buildBal(divId, vals, label){
    const H = histData(vals, 0, 1, 20);
    const med = median(vals), balancedPct = 100*vals.filter(v=>v<0.34).length/(vals.length||1);
    const custom = H.counts.map((c,i)=>[H.edges[i].toFixed(2), H.edges[i+1].toFixed(2), (100*c/(vals.length||1)).toFixed(1)]);
    // colour: near 0 (balanced, the misleading case) = amber; near 1 (one-sided) = blue
    const balColor = (c)=> `hsl(${Math.round(40+174*c)}, 50%, 55%)`;
    const bar={ type:'bar', x:H.centres, y:H.counts,
      marker:{color:H.centres.map(balColor), line:{color:'rgba(0,0,0,0.4)',width:0.4}},
      width:(1/20)*0.92, customdata:custom,
      hovertemplate:'balance [%{customdata[0]}, %{customdata[1]})<br><b>%{y}</b> points, %{customdata[2]}%<extra></extra>',
      showlegend:false };
    const hmax=Math.max(...H.counts,1)*1.15;
    const medLine={ type:'scatter',mode:'lines',x:[med,med],y:[0,hmax],
      line:{color:'#4ec97a',width:1.4,dash:'dot'},
      name:`median ${med.toFixed(2)}`, showlegend:true,
      hovertemplate:`median balance ${med.toFixed(2)}<extra></extra>` };
    Plotly.react(divId,[bar,medLine], baseLayout({
      title:{text:`${label}, n=${vals.length} distorted, ${balancedPct.toFixed(0)}% near-balanced (≈0)`, font:{size:11,color:'#9aa7b6'},x:0.02,xanchor:'left'},
      bargap:0.04, showlegend:true, legend:{font:{size:9,color:'#9aa7b6'},x:0.99,xanchor:'right',y:0.99,bgcolor:'rgba(21,26,33,0.78)'},
      xaxis:Object.assign({title:'balance   0 = both-ways, 1 = one-sided',range:[0,1]},AXIS),
      yaxis:Object.assign({title:'count',rangemode:'tozero',range:[0,hmax]},AXIS),
    }),PLOT_CONFIG);
  }

  buildDist('histDistPca', cP.dist, 'PCA');
  buildDist('histDistMds', cM.dist, 'MDS');
  buildBal('histBalPca', cP.bal, 'PCA');
  buildBal('histBalMds', cM.bal, 'MDS');
}

/* FP vs FN scatter (intrusion/extrusion balance), one panel per method */
/* class-centroid trajectory: high-D -> PCA -> MDS */
function renderCentroids(d){
  const c=d.centroids;
  if(!c){ $('centPlot').innerHTML='<div class="plot-fallback">No centroid data in this run.</div>'; return; }
  const stages=c.stage_names;                 // ['High-D','PCA','MDS']
  const xPos={}; stages.forEach((s,i)=>xPos[s]=i);
  const traces=[];

  // one polyline per class tracing its centroid across the stages
  c.classes.forEach((clsName, ci)=>{
    const code=c.class_codes[ci];
    const col=classColor(code);
    const pts=stages.map(s=>c.coords[s][ci]);      // [ [x,y], [x,y], [x,y] ]
    // trajectory line: x = stage index, y offset encoded via marker text; we plot
    // the actual 2-D drift as a small connected path within each stage column,
    // using stage index as x and the centroid's normalised 1st-axis as y, with the
    // 2nd axis shown on hover. This keeps stages readable left-to-right.
    traces.push({
      type:'scatter', mode:'lines+markers+text',
      x: stages.map(s=>xPos[s]),
      y: pts.map(p=>p[0]),
      text: stages.map((s,si)=> si===0? clsName : ''),
      textposition:'top center',
      textfont:{family:'Space Mono',size:10,color:col},
      name: clsName,
      line:{color:col,width:2},
      marker:{size:11,color:col,line:{color:'#0e1116',width:1.5},symbol:'circle'},
      customdata: pts.map(p=>[p[0],p[1]]),
      hovertemplate:`${clsName}<br>%{x}<br>axis1 %{customdata[0]:.3f}, axis2 %{customdata[1]:.3f}<extra></extra>`,
    });
  });

  // vertical guides per stage
  const guides=stages.map((s)=>({ type:'scatter',mode:'lines',
    x:[xPos[s],xPos[s]], y:[-3,3],
    line:{color:'#20272f',width:1}, hoverinfo:'skip', showlegend:false }));

  Plotly.react('centPlot',[...guides,...traces], baseLayout({
    showlegend:true,
    legend:{ font:{size:10,color:'#9aa7b6'}, bgcolor:'rgba(21,26,33,0.7)',
             bordercolor:'#262f3b',borderwidth:1, orientation:'h', x:0.5, xanchor:'center', y:-0.14 },
    xaxis:Object.assign({title:'',showgrid:false,tickvals:stages.map(s=>xPos[s]),
      ticktext:stages,range:[-0.4,stages.length-0.6]},AXIS),
    yaxis:Object.assign({title:'centroid position (normalised axis 1)',zeroline:true},AXIS),
  }),PLOT_CONFIG);

  // between-class separation readout per stage
  const mb=c.mean_between;
  $('centMeta').innerHTML = stages.map(s=>
    `<span class="cent-stat"><span class="cent-stat-k">${s}</span>`+
    `<span class="cent-stat-v">${mb[s].toFixed(3)}</span></span>`
  ).join('') + `<span class="cent-note">mean between-centroid distance (normalised), similar values across stages ⇒ class separation preserved</span>`;
}


/* ERROR SPACE: integrated PCA + MDS + error-space exploration
   Three linked plots share one selection. Cross-brushing on any
   plot, quadrant buttons, and a shared detail table all drive
   the same state.esSel.
*/

// which quadrant each point falls in, by PCA/MDS error sign
// Near-zero threshold: a point whose combined error magnitude is within this
// fraction of vmax is treated as "close enough to zero" that its quadrant is
// noise. CAVEAT: near-zero net error does NOT mean faithful -- a balanced-severe
// point (large FP and large FN that cancel) also lands here; marker SIZE (total
// distortion) is what separates the genuinely faithful from the balanced-severe.
const ES_NEARZERO_FRAC = 0.15;
function esNearZeroTau(d){ return ES_NEARZERO_FRAC * (d ? d.vmax : 1); }
function esIsNearZero(pe, me, tau){ return Math.hypot(pe, me) < tau; }

const ES_QUADS = {
  bothFP:     (pe,me)=> pe>0 && me>0,
  bothFN:     (pe,me)=> pe<0 && me<0,
  pcaFPmdsFN: (pe,me)=> pe>0 && me<0,
  pcaFNmdsFP: (pe,me)=> pe<0 && me>0,
};

function esBaseTrace(t, vmax, d, kind, dmax){
  // kind: 'proj' (x/y are the method's 2-D coords) or 'err' (x/y are PCA/MDS error)
  const isErr = kind==='err';
  // DOUBLE ENCODING -----------------------------------------------------
  // colour = signed error (t.err); size = total distortion |FP| + |FN|.
  // For the error-space plot a point's two axes already carry both methods'
  // errors, so its "total distortion" is summed over BOTH methods.
  const tot = isErr
    ? d.table.pca.fp.map((v,i)=> Math.abs(v) + Math.abs(d.table.pca.fn[i]) +
                                 Math.abs(d.table.mds.fp[i]) + Math.abs(d.table.mds.fn[i]))
    : totalDistortion(t);
  const scaleMax = isErr ? 2*dmax : dmax;   // err plot sums two methods
  const sizes = distortionSizes(tot, scaleMax);

  // ----- colour: either the class-view, or one of the error colour modes -----
  const classView = state.esView === 'class';
  let marker;
  if(classView){
    // colour by class; size stays as distortion so structure is still visible
    marker = { size:sizes, color: classColorArray(d),
               line:{color:'rgba(0,0,0,0.45)',width:0.5} };
  } else {
    const ch = colourChannel(t, d);           // severity-weighted diverging scale
    marker = { size:sizes, color: ch.vals, colorscale:DIVERGING,
               cmin:-ch.vmax, cmax:ch.vmax,
               line:{color:'rgba(0,0,0,0.5)',width:0.5} };
  }
  const sevVals = severityError(isErr ? d.table.pca : t);  // for hover only
  return {
    type:'scattergl', mode:'markers',
    x: isErr ? d.table.pca.err : t.x,
    y: isErr ? d.table.mds.err : t.y,
    marker,
    customdata: d.table.ids.map((id,i)=>[
      d.table.class_names[i], id, d.table.pca.err[i], d.table.mds.err[i],
      t.fp[i], t.fn[i], tot[i], sevVals[i]]),
    hovertemplate: isErr
      ? 'idx %{customdata[1]}, %{customdata[0]}<br>' +
        'PCA err %{customdata[2]:.3f}, MDS err %{customdata[3]:.3f}<br>' +
        'total distortion %{customdata[6]:.3f}<extra></extra>'
      : 'idx %{customdata[1]}, %{customdata[0]}<br>' +
        'error %{customdata[2]:.3f} &nbsp; (FP %{customdata[4]:.3f}, FN %{customdata[5]:.3f})<br>' +
        'severity %{customdata[7]:+.3f}, total distortion %{customdata[6]:.3f}<extra></extra>',
    selected:{ marker:{ opacity:1 } },
    unselected:{ marker:{ opacity:0.15 } },
  };
}
function esRingTrace(){
  return { type:'scattergl', mode:'markers', x:[], y:[],
    marker:{ size:15, color:'rgba(0,0,0,0)', line:{color:'#7CFC4D',width:2.2} },
    hoverinfo:'skip', showlegend:false };
}

/** Build a Plotly annotation labelling a corner of the error-space scatter. */
function quadLabel(x, y, text, xanchor, yanchor){
  return { x, y, text, xref:'x', yref:'y', xanchor, yanchor,
    showarrow:false, font:{size:9, color:'#63707f'} };
}

function renderErrorSpace(d){
  const vmax=d.vmax;
  const dmax=distortionMax(d);      // shared size scale across PCA and MDS
  const projLayout = (title)=> baseLayout({
    title:{text:title,font:{size:12,color:'#e8edf3'},x:0.03,xanchor:'left'},
    dragmode:'select',
    xaxis:Object.assign({title:'dim 1'},AXIS), yaxis:Object.assign({title:'dim 2'},AXIS),
  });
  const errLayout = baseLayout({
    title:{text:'Error space',font:{size:12,color:'#e8edf3'},x:0.03,xanchor:'left'},
    dragmode:'select',
    xaxis:Object.assign({title:'PCA error',range:[-vmax,vmax],zeroline:true},AXIS),
    yaxis:Object.assign({title:'MDS error',range:[-vmax,vmax],zeroline:true},AXIS),
    shapes:[
      { type:'line', x0:0, x1:0, y0:-vmax, y1:vmax, line:{color:'#33404f',width:1} },
      { type:'line', x0:-vmax, x1:vmax, y0:0, y1:0, line:{color:'#33404f',width:1} },
      { type:'line', x0:-vmax, x1:vmax, y0:-vmax, y1:vmax,
        line:{color:'#2a3542',width:1,dash:'dot'} },
      // near-zero "faithful / balanced" zone: points inside have ~0 net error
      { type:'circle', xref:'x', yref:'y',
        x0:-esNearZeroTau(d), x1:esNearZeroTau(d),
        y0:-esNearZeroTau(d), y1:esNearZeroTau(d),
        line:{color:'#4a5a6a',width:1,dash:'dot'},
        fillcolor:'rgba(120,140,160,0.06)' },
    ],
    annotations:[
      quadLabel(-vmax*0.92,-vmax*0.92,'both FN','left','bottom'),
      quadLabel(vmax*0.92,vmax*0.92,'both FP','right','top'),
      quadLabel(-vmax*0.92,vmax*0.92,'PCA-FN / MDS-FP','left','top'),
      quadLabel(vmax*0.92,-vmax*0.92,'PCA-FP / MDS-FN','right','bottom'),
    ],
  });
  const cfg = Object.assign({}, PLOT_CONFIG, {modeBarButtonsToAdd:['select2d','lasso2d']});

  Plotly.react('esPca', [esBaseTrace(d.table.pca,vmax,d,'proj',dmax), esRingTrace()],
    projLayout('PCA'), cfg);
  Plotly.react('esMds', [esBaseTrace(d.table.mds,vmax,d,'proj',dmax), esRingTrace()],
    projLayout('MDS'), cfg);
  Plotly.react('esErr', [esBaseTrace(d.table.pca,vmax,d,'err',dmax), esRingTrace()],
    errLayout, cfg);

  bindEsPlots();
  bindEsQuadrants();
  updateEsQuadCounts(d);
  renderWmdsCompare(d);
  applyEsSelection(d, null);   // reflect any carried-over selection
}

/* Weighted-MDS comparison: plain vs weighted MDS layouts side by side, coloured
   by class, with the silhouette separation change quantified. Only shown when
   the run used weighted MDS. */
/** Feature-weight editor for the locality-weighted MDS. Shows each feature's
 *  weight with a multiplier slider; "Apply" re-runs with the overrides. */

let _esQuadBound=false;
function bindEsQuadrants(){
  if(_esQuadBound) return;      // buttons are static; bind once
  _esQuadBound=true;
  document.querySelectorAll('#esQuadrants .es-quad').forEach(btn=>{
    btn.addEventListener('click', ()=>{
      const d=state.result; if(!d) return;
      const quad=btn.dataset.quad;
      if(state.esActiveQuad===quad){
        // clicking the active quadrant again clears it
        state.esActiveQuad=null; state.esSel=new Set();
      }else{
        state.esActiveQuad=quad;
        const tau=esNearZeroTau(d);
        const test = quad==='nearZero'
          ? (pe,me)=> esIsNearZero(pe,me,tau)
          : ES_QUADS[quad];
        const sel=new Set();
        d.table.pca.err.forEach((pe,i)=>{ if(test(pe, d.table.mds.err[i])) sel.add(i); });
        state.esSel=sel;
      }
      // clear Plotly's own drag-selection outlines when a quadrant drives selection
      ['esPca','esMds','esErr'].forEach(id=>Plotly.relayout(id,{selections:[]}));
      applyEsSelection(d, 'quadrant');
    });
  });
  $('esClear').addEventListener('click', ()=>clearEsSelection());
}

function bindEsPlots(){
  ['esPca','esMds','esErr'].forEach(id=>{
    const div=$(id);
    div.removeAllListeners?.('plotly_selected');
    div.removeAllListeners?.('plotly_deselect');
    div.on('plotly_selected', (ev)=>{
      if(!ev || !ev.points) return;
      state.esSel = new Set(ev.points.map(p=>p.pointIndex));
      state.esActiveQuad = null;               // manual brush overrides quadrant
      applyEsSelection(state.result, 'brush');
    });
    div.on('plotly_deselect', ()=>{
      state.esSel=new Set(); state.esActiveQuad=null;
      applyEsSelection(state.result, 'brush');
    });
  });
}

// paint the shared selection onto all three plots + table + counts
function applyEsSelection(d, source){
  if(!d) return;
  const idx=[...state.esSel];
  const ringXY = (xs,ys)=>({ x:idx.map(i=>xs[i]), y:idx.map(i=>ys[i]) });
  const rP=ringXY(d.table.pca.x, d.table.pca.y);
  const rM=ringXY(d.table.mds.x, d.table.mds.y);
  const rE=ringXY(d.table.pca.err, d.table.mds.err);
  // rings must track each point's double-encoded size, or a big point's ring
  // would sit inside it and a small point's ring would float far outside
  const dmax=distortionMax(d);
  const ringSizes = (sizes)=> idx.map(i=> sizes[i] + 7);
  const szP = ringSizes(distortionSizes(totalDistortion(d.table.pca), dmax));
  const szM = ringSizes(distortionSizes(totalDistortion(d.table.mds), dmax));
  const totE = d.table.pca.fp.map((v,i)=> Math.abs(v) + Math.abs(d.table.pca.fn[i]) +
                    Math.abs(d.table.mds.fp[i]) + Math.abs(d.table.mds.fn[i]));
  const szE = ringSizes(distortionSizes(totE, 2*dmax));
  Plotly.restyle('esPca', {x:[rP.x], y:[rP.y], 'marker.size':[szP]}, [1]);
  Plotly.restyle('esMds', {x:[rM.x], y:[rM.y], 'marker.size':[szM]}, [1]);
  Plotly.restyle('esErr', {x:[rE.x], y:[rE.y], 'marker.size':[szE]}, [1]);
  const sp=[idx.length?idx:null];
  Plotly.restyle('esPca', {selectedpoints:sp}, [0]);
  Plotly.restyle('esMds', {selectedpoints:sp}, [0]);
  Plotly.restyle('esErr', {selectedpoints:sp}, [0]);
  // reflect active-quadrant button styling
  document.querySelectorAll('#esQuadrants .es-quad').forEach(b=>
    b.classList.toggle('active', b.dataset.quad===state.esActiveQuad));
  updateEsDetail(d, source);
}

function clearEsSelection(){
  state.esSel=new Set(); state.esActiveQuad=null;
  if(state.result){
    ['esPca','esMds','esErr'].forEach(id=>Plotly.relayout(id,{selections:[]}));
    applyEsSelection(state.result, null);
  }
}

// per-quadrant point counts (independent of current selection)
function updateEsQuadCounts(d){
  const counts={ bothFP:0, bothFN:0, pcaFPmdsFN:0, pcaFNmdsFP:0, nearZero:0 };
  const tau=esNearZeroTau(d);
  // "large" near-zero points = balanced-severe (cancelling FP & FN); flag them
  const dmax=distortionMax(d);
  let nearZeroLarge=0;
  d.table.pca.err.forEach((pe,i)=>{
    const me=d.table.mds.err[i];
    if(esIsNearZero(pe,me,tau)){
      counts.nearZero++;
      const tot=Math.abs(d.table.pca.fp[i])+Math.abs(d.table.pca.fn[i])+
                Math.abs(d.table.mds.fp[i])+Math.abs(d.table.mds.fn[i]);
      if(tot > 0.35*dmax) nearZeroLarge++;   // sizeable distortion despite ~0 error
      return;                                  // near-zero points don't count as a quadrant
    }
    for(const q in ES_QUADS){ if(ES_QUADS[q](pe,me)){ counts[q]++; break; } }
  });
  Object.keys(counts).forEach(q=>{
    const el=document.querySelector(`#esQuadrants .es-quad-count[data-count="${q}"]`);
    if(el) el.textContent=counts[q];
  });
  // annotate the near-zero button with how many are balanced-severe (a warning)
  const nzBtn=document.querySelector('#esQuadrants .es-quad-nearzero .es-quad-desc');
  const nzBtnEl=document.querySelector('#esQuadrants .es-quad-nearzero');
  if(nzBtn){
    if(nearZeroLarge>0){
      nzBtn.innerHTML = `<strong style="color:#e0a54b">${nearZeroLarge} are large</strong>, distorted, not faithful`;
      if(nzBtnEl) nzBtnEl.classList.add('es-quad-warn');
    } else {
      nzBtn.innerHTML = `all small, genuinely faithful`;
      if(nzBtnEl) nzBtnEl.classList.remove('es-quad-warn');
    }
  }
}

const ES_TABLE_COLS=[
  {k:'idx',l:'index'},{k:'cls',l:'class',cls:'col-class'},
  {k:'pe',l:'PCA err',err:true},{k:'me',l:'MDS err',err:true},
  {k:'pfp',l:'PCA FP',part:'fp'},{k:'pfn',l:'PCA FN',part:'fn'},
  {k:'mfp',l:'MDS FP',part:'fp'},{k:'mfn',l:'MDS FN',part:'fn'},
];
function updateEsDetail(d, source){
  const idx=[...state.esSel];
  const quadName={ bothFP:'Both FP', bothFN:'Both FN',
    pcaFPmdsFN:'PCA-FP / MDS-FN', pcaFNmdsFP:'PCA-FN / MDS-FP',
    nearZero:'Near-zero (check size for balanced-severe)' };
  $('esCount').textContent = idx.length
    ? `${idx.length} point${idx.length>1?'s':''} selected`
    : 'No points selected: drag on any plot or pick a quadrant';
  const tag=$('esSource');
  if(idx.length && state.esActiveQuad){ tag.textContent='quadrant, '+quadName[state.esActiveQuad]; tag.hidden=false; }
  else if(idx.length && source==='brush'){ tag.textContent='brushed'; tag.hidden=false; }
  else { tag.textContent=''; tag.hidden=true; }

  const thead=$('esTable').querySelector('thead');
  const tbody=$('esTable').querySelector('tbody');
  thead.innerHTML='<tr>'+ES_TABLE_COLS.map(c=>`<th class="${c.cls||''}">${c.l}</th>`).join('')+'</tr>';
  const rows=idx.map(i=>({
    idx:d.table.ids[i], cls:d.table.class_names[i],
    pe:d.table.pca.err[i], me:d.table.mds.err[i],
    pfp:d.table.pca.fp[i], pfn:d.table.pca.fn[i],
    mfp:d.table.mds.fp[i], mfn:d.table.mds.fn[i],
  })).sort((a,b)=>a.pe-b.pe);
  tbody.innerHTML = rows.map(r=>'<tr>'+ES_TABLE_COLS.map(c=>{
    let v=r[c.k];
    if(c.cls==='col-class'){
      const code=d.target_names.indexOf(v);
      return `<td class="col-class"><span class="class-chip" style="background:${classColor(code<0?0:code)}"></span>${v}</td>`;
    }
    if(c.err){ const cl=v<0?'cell-fn':(v>0?'cell-fp':'cell-zero'); return `<td class="${cl}">${v>=0?'+':''}${v.toFixed(3)}</td>`; }
    if(c.part){ const cl=v>0?(c.part==='fp'?'cell-fp':'cell-fn'):'cell-zero'; return `<td class="${cl}">${v.toFixed(3)}</td>`; }
    return `<td>${v}</td>`;
  }).join('')+'</tr>').join('');
}
