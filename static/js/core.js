/* ======================================================================
   CORE
   DOM helper, shared client state, colour palette, Plotly defaults.
   ====================================================================== */

/* Projection Fidelity Console: frontend logic */

const $ = (id) => document.getElementById(id);
const state = {
  result: null,        // latest run result
  metric: 'intrusion',
  weight: 'rank',

  esView: 'error',         // 'error' | 'class' -> Error Space colour-by mode
  esSel: new Set(),    // Error Space: shared selection across all 3 plots
  esActiveQuad: null,  // Error Space: currently active quadrant button (or null)
  tableSort: { key: 'consensus', dir: 1 },
  trustSelSort: { key: 'pca', dir: -1 },   // sort of the trust selection table
  wmdsSelSort: { key: 'move', dir: -1 },   // sort of the Mahalanobis selection table
  outlierSort: { key: 'val', dir: 1 },     // sort of the outlier table
  pinned: null,                            // a single pinned point index (green ring)
  // Trust Analysis state
  trustMetric: 'f1',       // 'f1' | 'js_fidelity' | 'euclidean' | 'mrre'
  untrustTier: 'all',      // 'all' | 'Untrustworthy' | 'Medium' | 'Trustworthy'
  untrustSort: { key: 'tier', dir: 1 },   // worst tier first
  trustView: 'score',      // 'score' | 'class' -> colour the trust plots by
  brushMode: 'within',     // 'within' | 'all'      -> boxplot recompute mode
  cmpMethod: 'MDS',        // method for the metric-vs-metric scatter
  scoreFilterPca: null,    // {lo, hi} for PCA scores
  scoreFilterMds: null,    // {lo, hi} for MDS scores
  filterDisagreement: null,  // 'large' | 'medium' | 'small' | 'pca_better' | 'mds_better' | null
  cmpX: 'f1',              // x-axis metric of the comparison scatter
  cmpY: 'js_fidelity',     // y-axis metric of the comparison scatter
  trustSel: null,          // Set of selected point indices (null = all)
  filterClasses: new Set(),// class codes to keep (empty = all)
  attrFilter: null,        // {col, lo, hi} attribute window (null = none)
  scoreFilter: null,       // {lo, hi} keep points whose active-metric score (PCA & MDS) is in [lo,hi]
};


/* ------------------------------------------------------------------
   Five-band categorical scale for every [0,1] fidelity score.

   A continuous red-to-green ramp makes 0.55 and 0.62 indistinguishable, which
   is exactly the comparison these plots exist to support. Hard bands mean a
   point's colour is readable on its own, without eyeballing it against the
   colour bar.

     0.00 - 0.20   dark red
     0.20 - 0.40   dark orange
     0.40 - 0.60   yellow
     0.60 - 0.80   light green
     0.80 - 1.00   dark green
   ------------------------------------------------------------------ */
const SCORE_BANDS = [
  { lo:0.0, hi:0.2, color:'#8c1d1d', label:'0.0-0.2' },
  { lo:0.2, hi:0.4, color:'#c2600f', label:'0.2-0.4' },
  { lo:0.4, hi:0.6, color:'#d9bb36', label:'0.4-0.6' },
  { lo:0.6, hi:0.8, color:'#6cbf59', label:'0.6-0.8' },
  { lo:0.8, hi:1.0, color:'#17703a', label:'0.8-1.0' },
];

/* Plotly draws a discrete scale when each band is given both its start and end
   stop at the same colour, so there is no interpolation across a boundary. */
const SCORE_SCALE = SCORE_BANDS.flatMap(b => [[b.lo, b.color], [b.hi, b.color]]);

/* The band a score falls in. Upper edges are inclusive of the band below, so
   0.2 reads as dark red and 0.201 as dark orange. */
function scoreBand(v){
  const x = Math.max(0, Math.min(1, v));
  for(const b of SCORE_BANDS){ if(x <= b.hi) return b; }
  return SCORE_BANDS[SCORE_BANDS.length - 1];
}
function scoreColor(v){ return scoreBand(v).color; }

/* Ticks on the boundaries, so the colour bar reads as five blocks. */
const SCORE_TICKS = [0, 0.2, 0.4, 0.6, 0.8, 1];

/* The two panels are addressed internally as the slots 'PCA' and 'MDS', but
   either may hold any of PCA / MDS / t-SNE / UMAP. mName maps a slot to the
   method that actually filled it, for every user-facing label. */
function mName(slot){
  const m = state.result && state.result.info && state.result.info.methods;
  return (m && m[slot]) || slot;
}

const CLASS_PALETTE = ['#e0a54b','#4ea3a3','#b47cc8','#6f9f5b','#c8795a','#5a7fc8','#c85a8a','#8a8a5a'];
function classColor(code){ return CLASS_PALETTE[code % CLASS_PALETTE.length]; }

/* shared Plotly layout bits (dark instrument theme) */
const AXIS = {
  gridcolor:'#20272f', zerolinecolor:'#33404f', linecolor:'#33404f',
  tickfont:{ family:'Space Mono, monospace', size:10, color:'#63707f' },
  titlefont:{ family:'Archivo, sans-serif', size:11, color:'#9aa7b6' },
};
function baseLayout(extra={}){
  return Object.assign({
    paper_bgcolor:'rgba(0,0,0,0)',
    plot_bgcolor:'rgba(0,0,0,0)',
    font:{ family: "Arial, Helvetica, sans-serif", color:'#e8edf3' },
    margin:{ l:48, r:16, t:34, b:42 },
    hoverlabel:{ bgcolor:'#1b222c', bordercolor:'#33404f',
                 font:{ family: "Arial, Helvetica, sans-serif", size:11, color:'#e8edf3' } },
    showlegend:false,
  }, extra);
}
const PLOT_CONFIG = { displayModeBar:true, displaylogo:false, responsive:true,
  modeBarButtonsToRemove:['autoScale2d','toggleSpikelines'] };