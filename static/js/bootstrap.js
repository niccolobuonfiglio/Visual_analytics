/* ======================================================================
   BOOTSTRAP
   Console panel, toast, debug exports, and the init() call that starts
   the application. Loaded last.
   ====================================================================== */

/* INIT */
async function init(){
  // Start with empty state visible, results hidden
  const emptyState = $('emptyState');
  const resultBody = $('resultBody');
  
  // Use removeAttribute to show, setAttribute to hide
  emptyState.removeAttribute('hidden');
  resultBody.setAttribute('hidden', '');
  
  await loadDatasets();
  bindControls();
  bindTabs();
  bindConsole();
  bindVerdictUi();
  bindInfoDelegation();
  injectInfoButtons();
  bindTrustControls();
  await restoreLast();
}


/* TABS / CONSOLE / TOAST */

function bindTabs(){ /* tabs removed */ }

function bindConsole(){
  $('consoleBar').addEventListener('click', ()=>{
    const c=$('console');
    const open=c.classList.toggle('open');
    $('consoleBody').hidden = !open;
    if(open) $('consoleBody').scrollTop = $('consoleBody').scrollHeight;
  });
}
function openConsole(){
  const c=$('console');
  if(!c.classList.contains('open')){ c.classList.add('open'); $('consoleBody').hidden=false; }
}
function setConsole(text){
  $('consoleBody').textContent = text;
  const lines=(text.match(/\n/g)||[]).length+1;
  $('consoleMeta').textContent = `${lines} lines`;
  if($('console').classList.contains('open')) $('consoleBody').scrollTop=$('consoleBody').scrollHeight;
}
function setConsoleDot(kind){
  const dot=$('consoleDot');
  dot.className='console-dot'+(kind?' '+kind:'');
}

let toastTimer=null;
function toast(msg, isErr=false){
  const t=$('toast');
  t.textContent=msg;
  t.className='toast show'+(isErr?' err':'');
  t.hidden=false;
  clearTimeout(toastTimer);
  toastTimer=setTimeout(()=>{ t.classList.remove('show'); }, 3200);
}

/* expose for debugging / testing */
window.state = state;
window.runAnalysis = runAnalysis;

init();
