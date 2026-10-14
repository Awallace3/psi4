"""Figure-first dashboard shell adapted from the user-level docs-html template."""
import html
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent


def figure(identifier, title, caption):
    return f"""<div class="fig" id="{identifier}">
<div class="fig-head"><span class="t">{title}</span><span class="tag measured">measured</span>
<button class="expand" type="button">⤢ Expand</button></div>
<div class="legend"></div><svg class="chart" role="img" aria-label="{title}"></svg>
<p class="cap">{caption}</p>
<details class="data"><summary>Plotted values and coverage</summary><div class="tablewrap"></div></details></div>"""


def render_page(rows, table, stamp):
    page = (HERE / "dashboard-template.html").read_text()
    # This dashboard has neither equations nor source-code panes. No CDN assets
    # are needed; keep the template's layout, figures, tooltip and glossary code.
    lines = page.splitlines()
    page = "\n".join(line for line in lines if "cdn.jsdelivr.net" not in line)
    page = page.replace('<html lang="en" data-theme="dark">',
                        '<html lang="en" class="dark" data-theme="dark">')
    page = page.replace("TOPIC", "GPU DFT timing and memory").replace("PROJECT", "Phoenix · H200")
    page = page.replace("YYYY-MM-DD", stamp)
    page = page.replace('id="themeBtn" type="button">Light / dark',
                        'id="themeBtn" type="button" aria-pressed="true">Light theme')
    demo_start = page.index("/* demo figure:")
    glossary_start = page.index("/* glossary:", demo_start)
    page = page[:demo_start] + page[glossary_start:]
    page = page.replace('{"demo": []}', json.dumps(rows, allow_nan=False).replace("<", "\\u003c"))
    # Template theme machinery remains authoritative; update button and class too.
    page = page.replace("document.documentElement.dataset.theme = t; localStorage.setItem('docs-theme', t);",
        """document.documentElement.dataset.theme = t; localStorage.setItem('docs-theme', t);
  document.documentElement.classList.toggle('dark', t === 'dark');
  $('#themeBtn').textContent = t === 'dark' ? 'Light theme' : 'Dark theme';
  $('#themeBtn').setAttribute('aria-pressed', t === 'dark');""")
    revision = subprocess.check_output(
        ["git", "-C", str(HERE), "rev-parse", "--short=10", "HEAD"], text=True).strip()
    if subprocess.check_output(
            ["git", "-C", str(HERE), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        revision += "+local-edits"
    style = """<style>
body{font:20px/1.6 system-ui,sans-serif}h1{font-size:32px}h2{font-size:28px}
.sub,.toc .meta,.cap,details.data>summary{font-size:16px}.toc nav a{font-size:17px}
.toc nav .sum{font-size:15px}button,select{font-size:17px;padding:7px 10px}
.fig-head .t{font-size:19px}.legend{font-size:17px}.tag{font-size:14px}
table{font-size:17px}th{font-size:15px}td,th{padding:10px 12px}
small{display:block;font-size:14px;color:var(--muted)}
.warn{color:var(--bad)}.status{border-left:4px solid var(--s4);background:var(--surface-1);padding:12px 18px}
.filters{display:flex;gap:14px 22px;flex-wrap:wrap;align-items:center;margin:22px 0}
#tip{font-size:17px}svg.chart text{font-size:17px}
.fig:focus-within{outline:1px solid var(--s1)}svg [data-mark]:focus{outline:none;stroke:var(--text-primary);stroke-width:3}
.grid-note{font-size:17px}.tablewrap{max-width:100%}
@media(max-width:900px){h2,h3{scroll-margin-top:68px}}
@media(max-width:600px){.filters label{display:flex;gap:8px;align-items:center}main{padding-left:12px;padding-right:12px}.fig{padding:12px 8px}}
</style>"""
    page = page.replace("</head>", style + "</head>")
    functional_options = "".join(
        f"<option value='{html.escape(name)}'{' selected' if name == 'pbe' else ''}>{html.escape(name)}</option>"
        for name in sorted({row["functional"] for row in rows}))
    content = """<main><button class="menu-btn" id="menuBtn" type="button" aria-controls="toc">☰ Contents</button>
<h1>GPU DFT timing and memory</h1><p class="sub">Phoenix · H200 · cc-pVDZ and aug-cc-pVTZ · """ + stamp + """</p>
<p class="status"><strong>Exploratory: actual grids were not matched.</strong>
This snapshot contains completed cc-pVDZ production and aug-cc-pVTZ preflights only.
Native-default exports and both programs on the smaller default remain pending.
The separate grid-export build was submitted on free embers; its submission is not grid validation.</p>
<div class="filters"><label>Basis <select id="basis"><option value="all">Both bases</option><option>cc-pvdz</option><option>aug-cc-pvtz</option></select></label>
<label>System <select id="system"><option value="all">All systems</option><option>peptide</option><option>nanotube</option><option>protein83</option><option>protein157</option></select></label>
<label>Functional <select id="functional"><option value="all">All functionals</option>""" + functional_options + """</select></label></div>
<p class="grid-note">The initial view selects PBE/cc-pVDZ and PBE/aug-cc-pVTZ; select “All functionals” for all rungs.
Psi4 is blue and GPU4PySCF orange throughout. PBE/aug-cc-pVTZ on Protein157 has a failed Psi4 arm.</p>
<h2 id="walltime" data-summary="Walltime is time to convergence on different quadratures, not a grid-controlled speedup.">1 · Walltime observations</h2>
""" + figure("time-plot", "Calculation walltime / seconds",
    "cc-pVDZ: median and min–max of three independent unprofiled runs; aug-cc-pVTZ: one preflight per arm. "
    "No timing bar is drawn for a missing or failed result. Profiles and grid-export diagnostics are excluded.") + """
<h2 id="rss" data-summary="Peak CPU RSS is measured separately from device memory, with missing failed-worker observations preserved.">2 · Peak CPU RSS</h2>
""" + figure("rss-plot", "Worker peak RSS / GiB",
    "Largest worker peak resident set across unprofiled attempts. CPU RSS excludes device memory. "
    "Missing measurements are N/A, not zero; see coverage in the data table.") + """
<h2 id="gpu" data-summary="GPU values are sampled whole-device observations, not exact per-process allocator peaks.">3 · Observed GPU memory</h2>
""" + figure("gpu-plot", "Sampled whole-device GPU memory / GiB",
    "Largest one-second whole-device observation during unprofiled worker intervals; a lower-bound sampled peak, "
    "not exact process allocation. Dashed bars identify failed attempts and do not establish complete-SCF capacity.") + """
<h2 id="results" data-summary="The record table preserves NBF, available grid storage counts, failures and provisional timing ratios.">4 · Records and grids</h2>
<p>Seconds; memory in GiB. Ratio = GPU4PySCF/Psi4 time (&gt;1 favors Psi4);
<strong>all ratios are provisional because actual grids were not matched.</strong></p>
<div class="tablewrap"><table id="records"><thead><tr><th>Basis</th><th>System</th><th>Functional</th><th>NBF</th>
<th>Psi4 grid points</th><th>PySCF stored grid points</th><th>Psi4 / s</th><th>GPU4PySCF / s</th>
<th>Py/P4 ratio</th><th>Psi4 RSS</th><th>PySCF RSS</th><th>Psi4 GPU*</th><th>PySCF GPU*</th><th>ΔE Py−P4 / Eh</th><th>Review status</th></tr></thead><tbody>""" + "".join(table) + """</tbody></table></div>
<p class="grid-note">PySCF counts are stored array lengths and may include zero-weight padding.
Psi4’s host-side “Total Points = 0” is an empty host-grid placeholder, <strong>not</strong> its cuEST device-grid size.
Unavailable values are N/A; nominal radial × angular × atoms is never substituted for an actual count.</p>
<details><summary>Grid-controlled comparison pending · failures and scope</summary><ul>
<li>Protein157, PBE/aug-cc-pVTZ: Psi4 requested 364.065 GiB during integral setup with 138.043 GiB free.
This failed request is <strong>not an allocated GPU peak</strong>; historical failed-worker peak RSS is unavailable.
GPU4PySCF converged in 161.3 s, with 5,382 basis functions and 7,265 auxiliary functions.</li>
<li>Protein83, ωB97X/aug-cc-pVTZ: GPU4PySCF failed in range-separated DF batching.
Protein83 total-energy review flags: PBE/aug-cc-pVTZ +1.137e−4 Eh,
M06-L/aug-cc-pVTZ +2.886e−4 Eh and B3LYP/aug-cc-pVTZ +1.003e−4 Eh.</li>
<li>One restricted supermolecular DFT SCF, not SAPT(DFT)-D4 with iterative GRAC and repeated SCF solves.
Both engines use spherical orbitals, def2-universal-jkfit, native FP64 and eight CPU threads.
The historical nominal unpruned 99×590 grids, solver norms and iteration counts are not identical quadratures.</li>
<li>Calculation clocks start after imports and include runtime, molecule and basis setup through converged energy.
Historical GPU peaks are whole-device samples; the newer external worker/PID-aware memory instrumentation
was not used in these archives. The six-hour augmented reservation expired before production repeats.</li>
</ul></details>
<h2 id="glossary" data-summary="Units, grids and memory accounting terms used throughout the figures and tables.">5 · Glossary</h2>
<dl class="gloss">
<dt data-g="rss" data-match="RSS">RSS</dt><dd>Resident set size: CPU-resident process memory; peak Linux ru_maxrss, excluding GPU memory.</dd>
<dt data-g="nbf" data-match="NBF">NBF</dt><dd>Number of primary orbital basis functions (spherical AOs).</dd>
<dt data-g="gib" data-match="GiB">GiB</dt><dd>Gibibytes, 2³⁰ bytes. One GiB = 1024 MiB.</dd>
<dt data-g="gpu" data-match="GPU">GPU</dt><dd>Graphics processing unit; these runs used a single H200 per campaign.</dd>
<dt data-g="cpu" data-match="CPU">CPU</dt><dd>Host processor; eight threads per calculation.</dd>
<dt data-g="dft" data-match="DFT">DFT</dt><dd>Density functional theory; numerical XC quadrature differs between these historical program runs.</dd>
<dt data-g="scf" data-match="SCF">SCF</dt><dd>Self-consistent field solve, including initialization and convergence iterations.</dd>
<dt data-g="df" data-match="DF">DF</dt><dd>Density fitting using def2-universal-jkfit for Coulomb and exchange terms.</dd>
<dt data-g="xc" data-match="XC">XC</dt><dd>Exchange-correlation; quadrature coordinates and final weights must agree for an identical-grid workload.</dd>
<dt data-g="eh" data-match="Eh">Eh</dt><dd>Hartree, the total-energy unit. Review threshold is 1e−4 Eh, not accuracy certification.</dd>
<dt data-g="na" data-match="N/A">N/A</dt><dd>Measurement unavailable; never treated as a zero value.</dd>
</dl>
<footer>Generated from saptdft-cuest-pyscf@""" + revision + """ on """ + stamp + """ · derived historical archives.
<span id="refresh">Checks for republished snapshots every 15 seconds; not a SLURM poll.</span></footer></main>"""
    main_start, main_end = page.index("<main>"), page.index("</main>") + len("</main>")
    page = page[:main_start] + content + page[main_end:]
    script = (HERE / "dashboard-charts.js").read_text()
    return page.replace("</body>", "<script>\n" + script + "\n</script></body>")
