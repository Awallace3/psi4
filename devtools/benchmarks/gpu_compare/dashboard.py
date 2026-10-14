#!/usr/bin/env python3
"""Build a share-safe, self-contained timing and memory dashboard."""
import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import statistics

from memory_report import campaign_rows


def data_for(campaigns):
    comparisons = []
    for base in campaigns:
        records = json.loads((base / "records.json").read_text())
        summaries = json.loads((base / "summary.json").read_text())
        memory, _ = campaign_rows(base)
        config = json.loads((base / "config.json").read_text())
        for summary in summaries:
            item = dict(basis=config["basis"], system=summary["system"],
                        functional=summary["functional"], rung=summary["rung"],
                        energy_delta_eh=summary.get("energy_delta_pyscf_minus_psi4_eh"),
                        energy_review=summary.get("energy_review", False))
            counts = sorted({r["nbf"] for r in records if r.get("nbf") is not None and
                             (r["system"], r["functional"]) == (item["system"], item["functional"])})
            item["nbf"] = counts[0] if len(counts) == 1 else None
            for engine in ("psi4", "pyscf"):
                group = [r for r in records if not r["profile"] and
                         (r["system"], r["functional"], r["engine"]) ==
                         (item["system"], item["functional"], engine)]
                good = [r for r in group if r["ok"] and r.get("process_returncode") == 0]
                observations = [r for r in memory if not r["profile"] and
                                (r["system"], r["functional"], r["engine"]) ==
                                (item["system"], item["functional"], engine)]
                def peak(key):
                    return max((r[key] for r in observations if r[key] is not None), default=None)
                times = [r["calculation_wall_s"] for r in good]
                grids = sorted({r["grid_points"] for r in group if r.get("grid_points") is not None})
                # No paths, geometry, environment manifests or private identifiers are published.
                item[engine] = dict(passed=len(good), attempted=len(group),
                    calculation_s=statistics.median(times) if times else None,
                    time_min=min(times) if times else None, time_max=max(times) if times else None,
                    peak_rss_gib=peak("peak_rss_gib"),
                    rss_coverage=sum(r["peak_rss_gib"] is not None for r in observations),
                    grid_storage_points=grids[0] if len(grids) == 1 else None,
                    sampled_gpu_device_peak_gib=(peak("sampled_gpu_device_peak_mib") / 1024
                                                if peak("sampled_gpu_device_peak_mib") is not None else None))
            comparisons.append(item)
    return comparisons


def render(rows):
    def num(value, digits=2):
        return "N/A" if value is None else f"{value:.{digits}f}"
    table = []
    for r in rows:
        cells = [r["basis"], r["system"], r["functional"], num(r.get("nbf"), 0)]
        p, g = r["psi4"], r["pyscf"]
        cells += [num(p.get("grid_storage_points"), 0), num(g.get("grid_storage_points"), 0)]
        status = "unmatched grids"
        if p["passed"] != p["attempted"] or g["passed"] != g["attempted"]:
            status = "failed arm"
        elif r["energy_review"]:
            status = "energy review"
        for e in (p, g):
            cells.append(f"{num(e['calculation_s'])}<small>{e['passed']}/{e['attempted']} passed</small>")
        advantage = None
        if status != "failed arm" and p["calculation_s"] and g["calculation_s"]:
            advantage = g["calculation_s"] / p["calculation_s"]
        cells.append(num(advantage) + ("×" if advantage else ""))
        cells += [num(p["peak_rss_gib"], 3), num(g["peak_rss_gib"], 3),
                  num(p["sampled_gpu_device_peak_gib"]), num(g["sampled_gpu_device_peak_gib"])]
        cells.append("N/A" if r["energy_delta_eh"] is None else f"{r['energy_delta_eh']:+.2e}")
        cells.append(f"<span class='warn'>{status}</span>")
        table.append(f"<tr data-basis='{html.escape(r['basis'])}' data-system='{html.escape(r['system'])}'>"
                     + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return """<!doctype html><html lang="en" class="dark"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GPU DFT comparison · timings and memory</title><style>
:root{color-scheme:light;--page:#f3f6f6;--panel:#fff;--ink:#142731;--muted:#49616e;--line:#c5d4da;--ok:#176346;--warn:#9b283b}
html.dark{color-scheme:dark;--page:#071014;--panel:#0d1a20;--ink:#eaf2f6;--muted:#a8bbc5;--line:#29434e;--ok:#72d7aa;--warn:#ff8fa3}
*{box-sizing:border-box}body{background:var(--page);color:var(--ink);margin:0;font:20px/1.6 Candara,"Trebuchet MS",sans-serif}
main{max-width:1800px;margin:auto;padding:36px 28px 70px}h1{font:clamp(44px,5vw,76px)/1.1 Georgia,serif;letter-spacing:-.035em;margin:24px 0}
h2{font:34px Georgia,serif}header{border-bottom:2px solid var(--ink);padding-bottom:24px}.toolbar,.filters{display:flex;gap:20px;align-items:center;flex-wrap:wrap}.toolbar{justify-content:space-between}
.eyebrow{font-size:14px;text-transform:uppercase;letter-spacing:.15em}small{display:block;font-size:14px;color:var(--muted)}
button,select{font:inherit;padding:7px 12px;background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:4px}
.status{border-left:4px solid var(--warn);background:var(--panel);padding:12px 18px}.metrics{display:flex;gap:32px;flex-wrap:wrap}.metrics b{display:block;font:46px Georgia,serif}
section{padding:22px 0;border-bottom:1px solid var(--line)}.scroll{overflow:auto;margin-top:20px}
table{border-collapse:collapse;width:100%;background:var(--panel);font-size:17px}th,td{padding:13px 12px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}th{color:var(--muted);font-size:14px}td:nth-child(n+4){font-family:Consolas,monospace}.ok{color:var(--ok)}.warn{color:var(--warn)}li{margin:8px 0}
@media(max-width:700px){main{padding:22px 16px}.metrics{gap:20px}}@media print{button,.filters{display:none}}
</style></head><body><main><header><div class="toolbar"><span class="eyebrow">Phoenix · H200 · GPU4PySCF vs Psi4 cuEST + CUDA LibXC</span>
<button id="theme" aria-pressed="true">Light theme</button></div><h1>Timing is only half<br>the comparison.</h1>
<p>Calculation wall time, resident CPU memory, and observed GPU memory across cc-pVDZ and aug-cc-pVTZ.</p>
<p class="status"><b>No allocation currently active.</b> cc-pVDZ production is complete.
aug-cc-pVTZ stopped after preflight: Protein157/Psi4 exceeded H200 memory;
Protein83/GPU4PySCF ωB97X failed in range-separated DF batching.
The six-hour reservation expired; augmented-basis production repeats were never launched.</p>
<p class="status"><b>Grid-controlled comparison pending.</b> Matching nominal 99×590 options did not
establish identical coordinates or weights. Existing timing ratios are exploratory, not
apples-to-apples grid-controlled performance results. Native-default grids and a shared smaller-default
grid still need explicit export and testing.</p>
<div class="metrics"><div><b>120 / 120</b><small>cc-pVDZ production workers passed</small></div>
<div><b>12 / 14</b><small>aug-cc-pVTZ preflight workers passed</small></div>
<div><b>133 / 134</b><small>archived workers with recorded peak RSS</small></div></div>
<p><small>Published """ + stamp + """ · <span id="refresh">Checks for republished page snapshots every 15 seconds; does not poll SLURM.</span></small></p></header>
<section><h2>Per-functional results</h2><p>Seconds; memory in GiB. cc-pVDZ: median of three independent timing runs.
aug-cc-pVTZ: <b>one preflight run per arm</b>. Profiles excluded.
Ratio = GPU4PySCF/Psi4 time (&gt;1 favors Psi4); <b>all ratios are provisional because actual grids were not matched.</b></p>
<div class="filters"><label>Basis <select id="basis"><option value="all">Both bases</option><option>cc-pvdz</option><option>aug-cc-pvtz</option></select></label>
<label>System <select id="system"><option value="all">All systems</option><option>peptide</option><option>nanotube</option><option>protein83</option><option>protein157</option></select></label></div>
<div class="scroll"><table><thead><tr><th>Basis</th><th>System</th><th>Functional</th><th>NBF</th>
<th>Psi4 grid points</th><th>PySCF stored grid points</th><th>Psi4 / s</th><th>GPU4PySCF / s</th>
<th>Py/P4 ratio</th><th>Psi4 RSS</th><th>PySCF RSS</th><th>Psi4 GPU*</th><th>PySCF GPU*</th><th>ΔE Py−P4 / Eh</th><th>Review status</th></tr></thead><tbody>""" + "".join(table) + """</tbody></table></div>
<p><small>* Largest 1-second whole-device observation during worker intervals—not exact process allocation peak.
RSS = largest worker peak resident set across unprofiled attempts. N/A is missing data, not zero.</small></p></section>
<section><h2>Grid point-count audit</h2><p>PySCF's recorded count is the stored coordinate/weight-array
length and can include zero-weight alignment padding; it is not yet a nonzero-weight count.
Psi4's actual cuEST device-grid count was not exported in these worker records.
The host-side printed “Total Points = 0” is <b>not</b> an empty integration grid and must not be used
as its grid count. Unavailable values are N/A. Nominal 99×590×atoms is not substituted for actual counts.</p></section>
<section><h2>Failures and numerical review</h2><ul>
<li><b>Protein157/Psi4, aug-cc-pVTZ/PBE:</b> device-buffer request <b>364.065 GiB</b>, free 138.043 GiB.
Failed during integral setup before RSS serialization. The request is not an allocated GPU peak; recorded RSS is unavailable.</li>
<li><b>Protein157/GPU4PySCF/PBE:</b> converged in 161.3 s, 31 iterations, 5,382 AOs / 7,265 auxiliary functions.
No paired timing ratio or energy comparison because the Psi4 arm failed.</li>
<li><b>Protein83/GPU4PySCF/ωB97X:</b> range-separated DF evaluator batching assertion; no converged energy or timing.</li>
<li><b>Protein83 energy-review flags:</b> PBE +1.137e−4 Eh; M06-L +2.886e−4 Eh; B3LYP +1.003e−4 Eh.
These exceed the 1e−4 Eh diagnostic threshold and need investigation before accuracy claims.</li></ul></section>
<section><h2>What is being compared?</h2><ul>
<li>One restricted supermolecular DFT SCF, not the older SAPT(DFT)-D4 workflow with iterative GRAC and normally nine SCF solves.</li>
<li>Spherical orbital basis; def2-universal-jkfit in both engines. Native FP64, eight CPU threads, one H200 per campaign.
Nominal unpruned 99×590 grids and convergence targets; actual quadratures, solver norms and iteration counts differ.</li>
<li>Common calculation clocks start after imports and include runtime/molecule/basis setup through converged energy.
They exclude validation and teardown. No claim of identical-iteration kernel throughput.</li>
<li>Historical peak RSS is worker Linux ru_maxrss. GPU observations are whole-device, UUID-restricted samples.
New instrumentation is implemented but was <b>not used in these historical runs</b>: it adds external failed/interrupted-worker RSS,
CPU-tree sampling and PID-filtered GPU memory with explicit coverage/errors.</li>
<li>The allocated GPU memory of pure-J paths need not scale like stored hybrid DF tensors.
Memory values are measured observations, not dense-tensor estimates or universal feasibility guarantees.</li></ul></section>
<footer><p>Share-safe derived results only; no geometries, software-environment manifests or credentials.
The page follows the existing cuEST report palette and typography.</p></footer></main><script>
const basis=document.getElementById('basis'), system=document.getElementById('system');
function filter(){document.querySelectorAll('tr[data-basis]').forEach(r=>r.hidden=(basis.value!=='all'&&r.dataset.basis!==basis.value)||(system.value!=='all'&&r.dataset.system!==system.value))}
basis.onchange=system.onchange=filter;
document.getElementById('theme').onclick=function(){const dark=document.documentElement.classList.toggle('dark');this.textContent=dark?'Light theme':'Dark theme';this.setAttribute('aria-pressed',dark)};
let modified=null;
async function check(){try{const r=await fetch(location.pathname+'?check='+Date.now(),{method:'HEAD',cache:'no-store'});
if(!r.ok)return;const next=r.headers.get('last-modified');if(modified&&next&&next!==modified){location.reload();return}modified=next;
document.getElementById('refresh').textContent='Snapshot update check: '+new Date().toLocaleTimeString()+' · not a SLURM poll';}catch(e){}}
check();setInterval(check,15000);
</script></body></html>"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    rows = data_for(a.campaign)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(render(rows) + "\n")
    print(f"Rendered {len(rows)} comparisons: {a.output}")


if __name__ == "__main__":
    main()
