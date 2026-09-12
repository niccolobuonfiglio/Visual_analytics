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
