/* Inline, dependency-free measured charts. DATA is the share-safe page snapshot. */
const controls = ['basis', 'system', 'functional'].map(id => document.getElementById(id));
const engines = [
  {key: 'psi4', name: 'Psi4 cuEST + CUDA LibXC', color: '--s1'},
  {key: 'pyscf', name: 'GPU4PySCF', color: '--s2'},
];
const chartMetrics = [
  {id: 'time-plot', key: 'calculation_s', unit: 's', digits: 3, spread: true},
  {id: 'rss-plot', key: 'peak_rss_gib', unit: 'GiB', digits: 3},
  {id: 'gpu-plot', key: 'sampled_gpu_device_peak_gib', unit: 'GiB', digits: 3},
];
const escaped = value => String(value).replace(/[&<>"']/g, c =>
  ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const valueText = (value, metric) => value === null ? 'N/A' : value.toFixed(metric.digits) + ' ' + metric.unit;
function selectedRows() {
  return DATA.filter(row => controls.every(c => c.value === 'all' || row[c.id] === c.value));
}
function barChart(metric, rows) {
  const figure = document.getElementById(metric.id), svg = figure.querySelector('svg');
  const W = Math.max(340, Math.round(svg.getBoundingClientRect().width));
  const H = Math.max(200, 86 * rows.length + 78);
  const m = {l: Math.min(238, W * 0.43), r: 72, t: 20, b: 52};
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const values = rows.flatMap(row => engines.map(engine =>
    metric.spread ? row[engine.key].time_max : row[engine.key][metric.key]))
    .filter(value => Number.isFinite(value));
  const maximum = values.length ? Math.max(...values) * 1.03 : 1;
  const X = value => m.l + value / (maximum || 1) * (W - m.l - m.r);
  const tickCount = W < 500 ? 2 : 5;
  let content = '';
  for (let i = 0; i <= tickCount; i++) {
    const value = maximum * i / tickCount, x = X(value);
    content += `<line class="gridline" x1="${x}" x2="${x}" y1="${m.t}" y2="${H-m.b}"/>`;
    content += `<text x="${x}" y="${H-m.b+25}" text-anchor="middle">${value.toFixed(maximum < 10 ? 1 : 0)}</text>`;
  }
  content += `<text x="${(m.l+W-m.r)/2}" y="${H-4}" text-anchor="middle">${metric.unit}</text>`;
  if (!rows.length) content += `<text x="${W/2}" y="${H/2}" text-anchor="middle">No archived comparisons selected</text>`;
  rows.forEach((row, i) => {
    const y = m.t + i * 86 + 17;
    const label = `${row.functional}/${row.basis}`;
    content += `<text x="${m.l-10}" y="${y+3}" text-anchor="end">${escaped(row.system)}</text>`;
    content += `<text x="${m.l-10}" y="${y+26}" text-anchor="end" style="font-size:14px">${escaped(label)}</text>`;
    engines.forEach((engine, k) => {
      const observation = row[engine.key], value = observation[metric.key];
      const by = y - 14 + k * 30;
      const failed = observation.passed !== observation.attempted;
      const tooltip = `<b>${escaped(engine.name)}</b><br>${escaped(row.system)} · ${escaped(label)}<br>` +
        `${valueText(value, metric)}<br>NBF: ${row.nbf ?? 'N/A'} · grid storage: ${observation.grid_storage_points ?? 'N/A'}<br>` +
        `${observation.passed}/${observation.attempted} unprofiled attempts passed` +
        (failed ? '<br><b>Failed/partial arm; not complete-SCF capacity.</b>' : '') +
        (metric.spread && value !== null ? `<br>Range: ${observation.time_min.toFixed(3)}–${observation.time_max.toFixed(3)} s` : '') +
        '<br><b>Historical unmatched grids.</b>';
      const index = i * 2 + k;
      svg._tips = svg._tips || {};
      svg._tips[index] = tooltip;
      if (value === null) {
        content += `<text data-mark="${index}" tabindex="0" aria-label="${escaped(engine.name+' '+label+' unavailable')}" x="${m.l+7}" y="${by+20}" fill="var(${engine.color})">N/A</text>`;
      } else {
        const width = X(value) - m.l;
        content += `<rect data-mark="${index}" tabindex="0" aria-label="${escaped(engine.name+' '+label+' '+valueText(value,metric))}" x="${m.l}" y="${by}" width="${Math.max(0.5,width)}" height="23" rx="2" fill="var(${engine.color})" ${failed?'fill-opacity=".38" stroke="var('+engine.color+')" stroke-dasharray="4 3"':''}/>`;
        if (metric.spread && observation.time_min !== null && observation.time_max !== null) {
          const lo = X(observation.time_min), hi = X(observation.time_max);
          content += `<path d="M${lo} ${by+12}H${hi}M${lo} ${by+7}V${by+17}M${hi} ${by+7}V${by+17}" stroke="var(--text-primary)" fill="none" pointer-events="none"/>`;
        }
        content += `<text x="${X(value)+6}" y="${by+18}" style="font-size:15px">${value.toFixed(metric.digits)}</text>`;
      }
    });
  });
  svg.innerHTML = content;
  svg.onmousemove = event => {
    const mark = event.target.closest('[data-mark]');
    if (mark) tip(event, svg._tips[mark.dataset.mark]); else tip();
  };
  svg.onmouseleave = () => tip();
  svg.onfocusin = event => {
    const mark = event.target.closest('[data-mark]');
    if (!mark) return;
    const box = mark.getBoundingClientRect();
    tip({clientX: box.left, clientY: box.bottom}, svg._tips[mark.dataset.mark]);
  };
  svg.onfocusout = () => tip();
  figure.querySelector('.legend').innerHTML = engines.map(engine =>
    `<span><span class="sw" style="background:var(${engine.color})"></span>${escaped(engine.name)}</span>`).join('');
  const dataRows = rows.map(row => `<tr><td>${escaped(row.system)}</td><td>${escaped(row.functional+'/'+row.basis)}</td>` +
    engines.map(engine => `<td>${valueText(row[engine.key][metric.key],metric)}</td><td>${row[engine.key].passed}/${row[engine.key].attempted}</td>`).join('') +
    (metric.key === 'peak_rss_gib' ? `<td>${row.psi4.rss_coverage}/${row.psi4.attempted}; ${row.pyscf.rss_coverage}/${row.pyscf.attempted}</td>` : '') + '</tr>');
  figure.querySelector('.tablewrap').innerHTML = '<table><thead><tr><th>System</th><th>Method/basis</th>' +
    '<th>Psi4 / '+metric.unit+'</th><th>Passed</th><th>GPU4PySCF / '+metric.unit+'</th><th>Passed</th>' +
    (metric.key === 'peak_rss_gib' ? '<th>RSS coverage P4; Py</th>' : '') +
    '</tr></thead><tbody>'+dataRows.join('')+'</tbody></table>';
}
function redraw() {
  const rows = selectedRows();
  document.querySelectorAll('#records tbody tr').forEach(row =>
    row.hidden = controls.some(c => c.value !== 'all' && row.dataset[c.id] !== c.value));
  chartMetrics.forEach(metric => barChart(metric, rows));
}
controls.forEach(control => control.addEventListener('change', redraw));
window.addEventListener('themechange', redraw);
let resizeFrame;
window.addEventListener('resize', () => {
  cancelAnimationFrame(resizeFrame);
  resizeFrame = requestAnimationFrame(redraw);
});
redraw();
glossary();
let modified = null;
async function checkSnapshot() {
  try {
    const response = await fetch(location.pathname+'?check='+Date.now(), {method:'HEAD',cache:'no-store'});
    if (!response.ok) return;
    const next = response.headers.get('last-modified');
    if (modified && next && next !== modified) {location.reload();return;}
    modified = next;
    document.getElementById('refresh').textContent = 'Snapshot update check: '+new Date().toLocaleTimeString()+' · not a SLURM poll';
  } catch (_) { /* A lost network connection is not new results. */ }
}
checkSnapshot();
setInterval(checkSnapshot,15000);
