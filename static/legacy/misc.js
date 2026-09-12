/* ====================================================================
   QUARANTINED -- NOT LOADED BY index.html
   ====================================================================
   Odds and ends with no remaining caller: the infoHeader builder and
   the tab intro copy from the removed tab bar.

   This file is kept for reference only. It is not referenced by any <script>
   tag, so it costs nothing at runtime. Re-adding it requires restoring both
   the markup it targets and the backend fields it reads.
   ==================================================================== */

/** Build a section header (h4) whose title and (i) button open the info modal. */
function infoHeader(key, labelHtml){
  const info = METRIC_INFO[key];
  const label = labelHtml || (info ? info.title : key);
  return `<h4 class="info-h"><button class="info-title" data-info="${key}">${label}</button>`+
    `<button class="info-btn" data-info="${key}" aria-label="What is this?" title="What is this?">i</button></h4>`;
}
/** Delegated click handler: any [data-info] opens the modal. Bound once. */

const TAB_INTROS = {
  dataquality: `<strong>Data Quality</strong>, is the class structure genuinely in the data, and do the labels describe it? These answers depend only on the data, <em>not</em> on any projection. If the data has no structure, or the labels don't match it, no projection can fix that.`,
  projquality: `<strong>Projection Quality</strong>, given the data, how faithfully does each 2-D projection preserve it, and which is better? These answers are about the <em>picture</em>. A good projection of poorly-structured data still looks poor, that's a data problem, diagnosed in the Data Quality tab.`,
  explorer: `<strong>Point Explorer</strong>, drill into the individual points a projection gets wrong, decomposed into false-closeness (intrusion) and lost-neighbour (extrusion) for each method.`,
};
