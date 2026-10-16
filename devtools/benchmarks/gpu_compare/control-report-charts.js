/* Matched control: failed/missing pairs are never plotted as zero. */
function labelMethod(r) {
  const functional = r.functional === 'wb97x' ? 'ωB97X' : r.functional.toUpperCase();
  return `${functional}/${r.basis === 'aug-cc-pvtz' ? 'aug-cc-pVTZ' : 'cc-pVTZ'}`.replace(
    /[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function selectedRows() {
  return DATA.rows.filter(r => r.system === $('#systemSelect').value &&
    r.basis === $('#basisSelect').value && r.df_mode === $('#modeSelect').value);
}
function reportBars(id, rows, field, units) {
  const svg = $('#'+id), W = 1020, H = 100 + rows.length*72, left = 290, right = 84;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const values = rows.flatMap(r => ['psi4','pyscf'].map(e => r.eligible ? r[e][field] : null)).filter(v => v != null);
  const max = Math.max(1, ...values)*1.1, width = W-left-right;
  let out = '';
  for (let i=0;i<=4;i++) {
    const x = left+width*i/4;
    out += `<line class="gridline" x1="${x}" x2="${x}" y1="24" y2="${H-55}"/><text x="${x}" y="${H-25}" text-anchor="middle">${(max*i/4).toFixed(1)}</text>`;
  }
  rows.forEach((r,i) => {
    const y=38+i*72;
    out += `<text x="${left-16}" y="${y+20}" text-anchor="end">${labelMethod(r)}</text>`;
    ['psi4','pyscf'].forEach((engine,k) => {
      const value=r.eligible?r[engine][field]:null;
      if(value == null) {
        if(k===0) out+=`<text x="${left+12}" y="${y+20}" class="failed">withheld — audit failed</text>`;
        return;
      }
      out+=`<rect data-row="${i}" data-engine="${engine}" x="${left}" y="${y+k*24}" width="${width*value/max}" height="19" rx="2" fill="${css(k===0?'--s1':'--s2')}"/>`;
      out+=`<text x="${left+width*value/max+8}" y="${y+k*24+15}">${value.toFixed(2)}</text>`;
    });
  });
  out+=`<text x="${left+width/2}" y="${H-2}" text-anchor="middle">${units}</text>`;
  svg.innerHTML=out;
  svg.onmousemove=e=>{
    const mark=e.target.closest('rect'); if(!mark) return tip();
    const r=rows[mark.dataset.row],engine=mark.dataset.engine,value=r[engine][field];
    tip(e,`<b>${engine==='psi4'?'Psi4 cuEST':'GPU4PySCF'}</b><br>${labelMethod(r)}<br>${value.toFixed(4)} ${units}<br>${r[engine].repeats} timed repeats`);
  };
  svg.onmouseleave=()=>tip();
  svg.closest('.fig').querySelector('.legend').innerHTML=
    `<span><i class="sw" style="background:var(--s1)"></i>Psi4 cuEST + CUDA LibXC</span><span><i class="sw" style="background:var(--s2)"></i>GPU4PySCF</span>`;
}
function reportNum(value,digits=2) { return value == null ? '—' : Number(value).toFixed(digits); }
function reportCount(value) { return value == null ? '—' : Number(value).toLocaleString('en-US'); }
function renderControlReport() {
  $('#jobLabel').textContent=`${DATA.meta.gpu_model} · job ${DATA.meta.job_id}`;
  $('#compiledCommit').textContent=DATA.meta.compiled_commit;
  $('#harnessCommit').textContent=DATA.meta.harness_commit;
  for(const [phase,id] of [['peptide-preflight','peptideCoverage'],['protein83-panel','proteinCoverage']]) {
    const c=DATA.meta.phase_coverage[phase];
    $('#'+id).textContent=`${c.passed}/${c.attempted} strategy/basis comparisons complete`;
  }
  const rows=selectedRows(),title=`${$('#systemSelect').selectedOptions[0].text} · ${$('#basisSelect').selectedOptions[0].text} · ${$('#modeSelect').selectedOptions[0].text}`;
  $('#timeTitle').textContent=title;
  $('#timeCaption').textContent=DATA.timing;
  $('#memoryCaption').textContent=DATA.memory;
  reportBars('timeChart',rows,'full_process_s','seconds · complete fresh process');
  reportBars('rssChart',rows,'peak_rss_gib','GiB · peak worker RSS');
  reportBars('gpuChart',rows,'sampled_gpu_device_gib','GiB · sampled whole-device peak');
  for (const [id,field,unit] of [['timeTable','full_process_s','s'],['rssTable','peak_rss_gib','GiB'],['gpuTable','sampled_gpu_device_gib','GiB']])
    table($('#'+id),['Method/basis',`Psi4 · ${unit}`,`GPU4PySCF · ${unit}`],rows.map(r=>[
      labelMethod(r),reportNum(r.psi4[field],3),reportNum(r.pyscf[field],3)]),v=>v);
  table($('#gridTable'),['Method/basis','NBF · Psi4 / PySCF','Retained MOs · Psi4 / PySCF',
    'Stored grid · Psi4','Stored grid · PySCF','Shared nonzero points','Audit'],rows.map(r=>[
      labelMethod(r),`${reportCount(r.psi4.nbf)} / ${reportCount(r.pyscf.nbf)}`,
      `${reportCount(r.psi4.nmo)} / ${reportCount(r.pyscf.nmo)}`,
      reportCount(r.psi4.stored_grid_points),reportCount(r.pyscf.stored_grid_points),
      reportCount(r.nonzero_grid_points),r.eligible?'verified':'withheld']),v=>v);
  table($('#energyTable'),['Method/basis','PySCF − Psi4 · Eh','Repeats · Psi4 / PySCF','State'],rows.map(r=>[
    labelMethod(r),r.energy_delta_eh == null ? '—' : r.energy_delta_eh.toExponential(4),
    `${r.psi4.repeats} / ${r.pyscf.repeats}`,
    r.eligible?(Math.abs(r.energy_delta_eh)>1e-6?'complete · energy review':'complete'):r.status]),v=>v);
}
['systemSelect','basisSelect','modeSelect'].forEach(id=>$('#'+id).onchange=renderControlReport);
window.addEventListener('themechange',renderControlReport);
renderControlReport();
