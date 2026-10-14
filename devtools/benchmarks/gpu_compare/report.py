#!/usr/bin/env python3
"""Audit and render the completed three-system GPU comparison."""
import argparse
import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from compare import summarize


def timer(record, name):
    entries = [t for t in record["timer_records"].values() if t["timer_name"] == name]
    require(len(entries) == 1, f"Expected one {name} timer, found {len(entries)}")
    return dict(wall_s=entries[0]["wall_time"], calls=entries[0]["n_calls"])


def require(condition, message):
    if not condition:
        raise ValueError(message)


def require_summary_equal(regenerated, saved):
    # atomic_json sorts config keys; original functional insertion order is lost.
    # Compare complete rows by identity, not the incidental presentation order.
    def indexed(rows):
        result = {(r["system"], r["functional"]): r for r in rows}
        require(len(result) == len(rows), "Duplicate summary identities")
        return result
    require(indexed(regenerated) == indexed(saved), "Summary replay differs")


def audit(base):
    complete = json.loads((base / "COMPLETE.json").read_text())
    records = json.loads((base / "records.json").read_text())
    manifest = json.loads((base / "manifest.json").read_text())
    saved = json.loads((base / "summary.json").read_text())
    config = json.loads((base / "config.json").read_text())
    require(manifest["config"] == config, "Manifest config differs")
    require(complete["complete"] and complete["final_build_hashes_unchanged"],
            "Incomplete campaign or changed build")
    require(manifest["systems"] == ["peptide", "nanotube", "protein83"], "Unexpected systems")
    require(len(records) == 120 and all(r["ok"] and r["process_returncode"] == 0 for r in records),
            "Missing or failed workers")
    keys = {(r["system"], r["functional"], r["engine"], r["repeat"], r["profile"]) for r in records}
    require(len(keys) == 120, "Duplicate workers")
    expected = {(system, functional, engine, repeat, profile)
                for system in manifest["systems"]
                for functional in ("svwn", "pbe", "m06-l", "b3lyp", "wb97x")
                for engine in ("psi4", "pyscf")
                for repeat, profile in ((0, False), (1, False), (2, False), ("profile", True))}
    # Profile repeat labels are controller-specific; validate their occurrence
    # separately while enforcing every unprofiled repeat's exact identity.
    require({k for k in keys if not k[-1]} == {k for k in expected if not k[-1]} and
            {(s, f, e) for s, f, e, _, prof in keys if prof} ==
            {(s, f, e) for s, f, e, _, prof in expected if prof},
            "Unexpected worker matrix")
    uuids = {r["gpu"]["uuid"] for r in records}
    require(len(uuids) == 1, "GPU identity mismatch")
    manifest_gpus = list(csv.DictReader(io.StringIO(manifest["gpu"]), skipinitialspace=True))
    require(len(manifest_gpus) == 1 and manifest_gpus[0]["uuid"] in uuids,
            "Manifest GPU identity mismatch")
    require(all(r["config_sha256"] == hashlib.sha256((base / "config.json").read_bytes()).hexdigest()
                for r in records), "Config hash mismatch")
    regenerated = summarize(records, manifest["config"], manifest["systems"])
    require_summary_equal(regenerated, saved)
    require(len(saved) == 15 and all(r["complete"] and r["comparison_eligible"] for r in saved),
            "Incomplete or ineligible comparisons")
    # Exact filenames delimit worker intervals; NVML remains whole-device sampling.
    samples = []
    with (base / "gpu-samples.csv").open() as f:
        for x in csv.DictReader(f, skipinitialspace=True):
            require(x["uuid"] in uuids, "Sample GPU identity mismatch")
            timestamp = datetime.strptime(x["timestamp"], "%Y/%m/%d %H:%M:%S.%f").replace(
                tzinfo=ZoneInfo("America/New_York")).timestamp()
            samples.append((timestamp, float(x["memory.used [MiB]"].split()[0])))
    rows = []
    for row in saved:
        row = dict(row)
        row["psi4_speedup"] = 1 / row["psi4_over_pyscf_calculation_time"]
        row["process_psi4_speedup"] = (row["pyscf"]["process_wall_s"]["median"] /
                                     row["psi4"]["process_wall_s"]["median"])
        for engine in ("psi4", "pyscf"):
            group = [r for r in records if r["system"] == row["system"] and
                     r["functional"] == row["functional"] and r["engine"] == engine and not r["profile"]]
            row[engine]["maxrss_gib_range"] = [min(r["maxrss_kib"] for r in group) / 1024**2,
                                             max(r["maxrss_kib"] for r in group) / 1024**2]
            values = [mem for t, mem in samples if any(
                r["started_unix_s"] <= t <= r["ended_unix_s"] for r in group)]
            row[engine]["sampled_whole_device_max_mib"] = max(values) if values else None
        p, g = [next(r for r in records if r["system"] == row["system"]
                    and r["functional"] == row["functional"] and r["profile"]
                    and r["engine"] == e) for e in ("psi4", "pyscf")]
        row["profile"] = dict(psi4_jk=timer(p, "JK: JK"),
                             psi4_xc=timer(p, "RV: Form V"),
                             psi4_device_functional=timer(p, "cuEST XC: CUDA LibXC Functional"),
                             pyscf_jk=g["diagnostic_timers"]["jk"],
                             pyscf_xc=g["diagnostic_timers"]["xc"],
                             pyscf_df_setup=g["diagnostic_timers"]["df_setup"])
        for key, t in list(row["profile"].items()):
            row["profile"][key] = dict(t, per_call_s=t["wall_s"] / t["calls"])
        rows.append(row)
    return dict(rows=rows, complete=complete, gpu=manifest["gpu"], manifest=manifest,
                production_elapsed_minutes=(max(r["ended_unix_s"] for r in records) -
                                            min(r["started_unix_s"] for r in records)) / 60,
                source_sha256={name: hashlib.sha256((base / name).read_bytes()).hexdigest()
                               for name in ("COMPLETE.json", "records.json", "summary.json", "manifest.json",
                                            "config.json", "gpu-samples.csv")})


def markdown(data):
    out = [
        "# GPU4PySCF versus Psi4 cuEST + CUDA LibXC",
        "",
        "**Completed:** 120/120 production calculations; 15/15 comparisons eligible.",
        f"Production worker span: {data['production_elapsed_minutes']:.1f} minutes. "
        "One H200, 8 CPU threads, Phoenix inferno allocation 14000107, released after work finished.",
        "",
        "## Main result",
        "",
        "Psi4 is faster on all 15 common calculation-time comparisons: **1.34–2.85×**.",
        "This is time to convergence, not a kernel-only speedup or a matched-iteration test.",
        "Protein157 was explicitly excluded from the user's consolidated run.",
        "",
        "Median seconds from three independent fresh-process runs. Profiles excluded.",
        "",
        "| System | Functional | Psi4 / s | GPU4PySCF / s | Psi4 speedup | Iterations Psi4 / PySCF | ΔE PySCF−Psi4 / Eh |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in data["rows"]:
        out.append(f"| {r['system']} | {r['functional']} | "
                   f"{r['psi4']['calculation_wall_s']['median']:.3f} | "
                   f"{r['pyscf']['calculation_wall_s']['median']:.3f} | {r['psi4_speedup']:.2f}× | "
                   f"{r['psi4']['scf_iterations']['median']} / {r['pyscf']['scf_iterations']['median']} | "
                   f"{r['energy_delta_pyscf_minus_psi4_eh']:+.3e} |")
    out += ["", "## Protein83 profiling", "",
            "One separate diagnostic run per arm. Times are inclusive and nested: do not sum them.",
            "Psi4 XC is its full `RV: Form V` timer, not just device functional evaluation. "
            "PySCF XC is synchronized `nr_rks`; scope and synchronization overhead are not identical.",
            "",
            "| Functional | Psi4 JK / s | PySCF JK / s | Psi4 XC / s | PySCF XC / s | XC calls Psi4 / PySCF |",
            "|---|---:|---:|---:|---:|---:|"]
    for r in data["rows"]:
        if r["system"] != "protein83":
            continue
        p = r["profile"]
        out.append(f"| {r['functional']} | {p['psi4_jk']['wall_s']:.3f} | "
                   f"{p['pyscf_jk']['wall_s']:.3f} | {p['psi4_xc']['wall_s']:.3f} | "
                   f"{p['pyscf_xc']['wall_s']:.3f} | {p['psi4_xc']['calls']} / {p['pyscf_xc']['calls']} |")
    out += ["", "## Per-comparison CPU and GPU memory", "",
            "Peak worker RSS: maximum across three unprofiled repeats per engine. "
            "GPU: largest 1-second whole-device observation within those worker intervals, "
            "not exact process allocation peak. All values in GiB; profiles excluded.", "",
            "| System | Functional | Psi4 peak RSS | PySCF peak RSS | Psi4 sampled GPU device peak | PySCF sampled GPU device peak |",
            "|---|---|---:|---:|---:|---:|"]
    for r in data["rows"]:
        cpu = [f"{max(r[e]['maxrss_gib_range']):.3f}" for e in ("psi4", "pyscf")]
        gpu = [f"{r[e]['sampled_whole_device_max_mib'] / 1024:.3f}"
               if r[e]["sampled_whole_device_max_mib"] is not None else "N/A"
               for e in ("psi4", "pyscf")]
        out.append(f"| {r['system']} | {r['functional']} | " + " | ".join(cpu + gpu) + " |")
    out += ["", "GPU4PySCF spends most of its Protein83 diagnostic SCF clock in XC "
            "(approximately 68–92%). It also performs more XC calls and SCF iterations. "
            "Per-call XC is still slower in these diagnostic boundaries, but grids, "
            "instrumentation, solver criteria and call scope differ; this is not a controlled "
            "single-kernel attribution.",
            "",
            "## Measurement limits and numerical review", "",
            "- Spherical cc-pVDZ / def2-universal-jkfit; 210, 444, 766 AOs.",
            "- SVWN and B3LYP both explicitly use the same VWN-RPA components; "
            "PBE, M06-L and ωB97X identities/weights/exchange parameters also validated.",
            "- FP64 Psi4/cuEST (mixed precision off), GPU4PySCF 1.8.1, PySCF 2.14.0, "
            "CuPy 13.4.1 and cuTENSOR 2.2.0. Historical Psi4 CUDA-LibXC build; "
            "staged file hashes, not current harness HEAD, identify its compiled code.",
            "- 99×590 unpruned nominal grids are **not identical quadratures**. Radial/radii, "
            "orientation, weights and screening differ. Matching nominal grid counts does not isolate XC kernels.",
            "- E target 1e-9 Eh; gradient target 1e-7. Psi4 uses RMS commutator residual, "
            "PySCF orbital-gradient norm. Guess/DIIS and stopping metrics differ. "
            "Psi4 took 11–15 iterations versus PySCF 16–22.",
            "- Common calculation clock starts after imports, includes runtime/device setup, "
            "molecule/basis construction through converged energy, and ends before audit bookkeeping. "
            "The separate `psi4.energy` versus `mf.kernel` clocks have asymmetric setup scope.",
            "- DF thresholds share nominal values but fitting/screening algorithms differ. "
            "GPU4PySCF's pure-functional J path can compute DF J without retaining a full CDERI tensor; "
            "cuEST's persistent-plan strategy differs. No claim of identical DF work.",
            "- Maximum |Δ total energy| is **3.00e-5 Eh**, Protein83/ωB97X. No row exceeds "
            "the 1e-4 Eh review threshold. This is **not** a universal accuracy certification, "
            "and the old interaction-energy tolerance does not apply to these totals.",
            "- Host peak RSS is about 1.20–1.69 GiB. GPU memory is sampled at 1-second resolution "
            "on the validated UUID; it is whole-device memory, not exact per-worker allocation peak.",
            "- Process clocks include imports, audit serialization and interpreter teardown. "
            "Full medians/ranges, iteration counts, inclusive profiles, energy differences, "
            "memory observations and input/artifact hashes are preserved in the JSON report.",
            "",
            "## Provenance", "",
            "Campaign: `/storage/project/r-cs207-0/awallace43/runs/gpu-dft-compare/"
            "20261013T113848Z-interactive-inferno/production/campaign/`.",
            "Harness used: `9595a76753`; allocation GPU "
            "`GPU-951a11c2-40ee-3fcb-f71f-49849e744cb1`.",
            "The first setup and diagnostic failures were harness errors and remain in "
            "separate archived directories; none is silently merged into production medians.",
            "Audit independently regenerates `summary.json` from all 120 worker records and "
            "requires exact equality before rendering.", ""]
    return "\n".join(out)


def html_report(data):
    table_rows = []
    profile_rows = []
    memory_rows = []
    for r in data["rows"]:
        p, g = r["psi4"], r["pyscf"]
        fmt = lambda x: f"{x['median']:.3f} <small>[{x['min']:.3f}–{x['max']:.3f}]</small>"
        table_rows.append(f"<tr data-system='{r['system']}'><td>{r['system']}</td><td>{r['functional']}</td>"
            f"<td>{fmt(p['calculation_wall_s'])}</td><td>{fmt(g['calculation_wall_s'])}</td>"
            f"<td class='win'>{r['psi4_speedup']:.2f}×</td><td>{p['scf_iterations']['median']} / "
            f"{g['scf_iterations']['median']}</td><td>{r['energy_delta_pyscf_minus_psi4_eh']:+.2e}</td></tr>")
        values = [f"{max(r[e]['maxrss_gib_range']):.3f}" for e in ("psi4", "pyscf")]
        values += [f"{r[e]['sampled_whole_device_max_mib'] / 1024:.3f}"
                   if r[e]["sampled_whole_device_max_mib"] is not None else "N/A"
                   for e in ("psi4", "pyscf")]
        memory_rows.append(f"<tr><td>{r['system']}</td><td>{r['functional']}</td>" +
                           "".join(f"<td>{v}</td>" for v in values) + "</tr>")
        if r["system"] == "protein83":
            t = r["profile"]
            profile_rows.append(f"<tr><td>{r['functional']}</td>"
                + "".join(f"<td>{t[k]['wall_s']:.3f}</td>" for k in
                          ("psi4_jk", "pyscf_jk", "psi4_xc", "pyscf_xc"))
                + f"<td>{t['psi4_xc']['calls']} / {t['pyscf_xc']['calls']}</td></tr>")
    return """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GPU DFT • H200 comparison</title><style>
:root{color-scheme:light;--page:#f3f6f6;--panel:#fff;--ink:#142731;--muted:#49616e;--line:#c5d4da;--gpu:#176346;--risk:#9b283b}
html.dark{color-scheme:dark;--page:#071014;--panel:#0d1a20;--ink:#eaf2f6;--muted:#a8bbc5;--line:#29434e;--gpu:#72d7aa;--risk:#ff8fa3}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font:16px/1.6 Candara,"Trebuchet MS",sans-serif}
main{max-width:1240px;margin:auto;padding:40px 30px 70px}header{border-bottom:2px solid var(--ink);padding-bottom:25px}
h1{font:clamp(38px,6vw,72px)/1.05 Georgia,serif;letter-spacing:-.04em;font-weight:normal;margin:25px 0}
h2{font:30px Georgia,serif}.eyebrow{letter-spacing:.15em;font-size:12px;text-transform:uppercase}
.toolbar{display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap}button,select{font:inherit;padding:7px 12px;border:1px solid var(--line);border-radius:4px;background:var(--panel);color:var(--ink)}
.metrics{display:flex;gap:35px;flex-wrap:wrap}.metric{border-left:3px solid var(--gpu);padding-left:15px}.metric strong{display:block;font:40px Georgia,serif}
.metric span,small{color:var(--muted);font-size:12px}section{padding:22px 0;border-bottom:1px solid var(--line)}
.callout{border-left:4px solid var(--gpu);background:var(--panel);padding:15px 20px;max-width:90ch}
.table-wrap{overflow:auto;max-width:100%}table{border-collapse:collapse;width:100%;font-size:14px;background:var(--panel)}th,td{padding:12px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}th{color:var(--muted);font-size:12px}td:not(:first-child){font-family:Consolas,monospace;font-size:13px}.win{color:var(--gpu);font-weight:bold}small{display:block}li{margin:10px 0}code{overflow-wrap:anywhere}details{margin:20px 0}summary{cursor:pointer;font-weight:bold}
@media(max-width:700px){main{padding:24px 16px}.metrics{gap:20px}.metric strong{font-size:30px}th,td{padding:9px}}
@media print{button,select{display:none}main{padding:10px}}
</style></head><body><main><header><div class="toolbar">
<span class="eyebrow">Computational audit · Phoenix · 2026-10-13</span>
<button id="theme" aria-pressed="false">Dark theme</button></div>
<h1>Same H200.<br>Different SCF pipelines.</h1>
<p>Matched nominal cc-pVDZ / DF / 99×590 targets. Psi4 cuEST + CUDA LibXC is faster
on every measured time-to-convergence comparison through Protein83.</p>
<div class="metrics"><div class="metric"><strong>120 / 120</strong><span>successful production calculations</span></div>
<div class="metric"><strong>1.34–2.85×</strong><span>Psi4 calculation-time advantage</span></div>
<div class="metric"><strong>3.00e−5 Eh</strong><span>maximum absolute total-energy difference</span></div></div></header>
<section><h2>Three sizes, five functionals</h2><p>Median seconds of three fresh processes.
Brackets show min–max, not confidence intervals. Profiles are excluded.
Speedup = GPU4PySCF / Psi4 calculation time; greater than 1 favors Psi4.</p>
<label>System <select id="system"><option value="all">All systems</option><option>peptide</option>
<option>nanotube</option><option>protein83</option></select></label><div class="table-wrap">
<table><thead><tr><th>System</th><th>Functional</th><th>Psi4 / s</th><th>GPU4PySCF / s</th>
<th>Psi4 advantage</th><th>Iterations P4 / PySCF</th><th>ΔE PySCF−P4 / Eh</th></tr></thead>
<tbody>""" + "".join(table_rows) + """</tbody></table></div></section>
<section><h2>The Protein83 profiling signal</h2><p>Separate inclusive diagnostic clocks, seconds.
Do not sum nested timers. Psi4 XC = full RV: Form V, not just CUDA functional evaluation.</p>
<div class="table-wrap"><table><thead><tr><th>Functional</th><th>Psi4 JK</th><th>PySCF JK</th>
<th>Psi4 XC</th><th>PySCF XC</th><th>XC calls P4 / PySCF</th></tr></thead><tbody>""" + "".join(profile_rows) + """</tbody></table></div>
<p class="callout">GPU4PySCF's diagnostic clock is XC-dominated and takes more iterations.
Per-call XC is also slower in these measured boundaries, but different quadratures,
screening, solver norms and instrumentation prevent attributing this ratio to a single kernel.</p></section>
<section><h2>CPU and GPU memory</h2><p>GiB. Peak RSS is the maximum across three unprofiled workers.
GPU values are 1-second whole-device observed peaks within those worker intervals—not exact
per-process device allocation high-water marks. Profiles excluded.</p><div class="table-wrap">
<table><thead><tr><th>System</th><th>Functional</th><th>Psi4 peak RSS</th><th>PySCF peak RSS</th>
<th>Psi4 GPU device</th><th>PySCF GPU device</th></tr></thead><tbody>""" + "".join(memory_rows) + """</tbody>
</table></div></section>
<section><h2>What this does—and does not—establish</h2><ul>
<li>Post-import common calculation boundary includes runtime/device setup, molecule/basis
construction and converged energy; excludes validation/provenance bookkeeping.</li>
<li>Psi4/cuEST native FP64; GPU4PySCF 1.8.1 / PySCF 2.14.0 / CuPy 13.4.1 / cuTENSOR 2.2.0.
One H200, eight CPU threads. Identical actual GPU UUID; staged build hashes verified unchanged.</li>
<li>Explicit matching LibXC identities/weights, including VWN-RPA for SVWN/B3LYP.
Spherical cc-pVDZ / def2-universal-jkfit; 210/444/766 AOs.</li>
<li>Nominal unpruned 99×590 grids are <b>not identical quadratures</b>. DF plans,
screening, guesses, DIIS and convergence-norm definitions differ. The measured
time-to-convergence advantage is not a matched-iteration kernel benchmark.</li>
<li>All 15 energy differences are below the 1e−4 Eh review threshold—not a universal
accuracy certification. The old interaction-energy tolerance does not apply to these total energies.</li>
<li>Host RSS spans about 1.20–1.69 GiB. One-second GPU-memory samples are whole-device
observations, not exact allocation peaks. Full data and timing ranges are in the JSON.</li>
<li>Protein157 is excluded. Earlier failed diagnostic attempts are separately archived,
not silently merged into these successful production medians.</li></ul></section>
<section><details><summary>Audit and data files</summary><p>The report independently regenerates all
15 comparison rows from 120 worker records and requires exact equality with the recorded summary.
Completion, zero worker exit codes, unique identities, shared GPU UUID and config hashes are verified.</p>
<p>Allocation <code>14000107</code>; harness <code>9595a76753</code>. Paid allocation released
after completion with user approval. Production worker span: """ + f"{data['production_elapsed_minutes']:.1f}" + """ minutes.</p>
<p><a href="gpu-dft-comparison.json">Audited JSON</a> ·
<a href="gpu-dft-comparison.md">Markdown report</a></p></details></section>
<footer><p>Fixed benchmark geometries; timing evidence, not optimized molecular chemistry.</p></footer>
</main><script>
document.getElementById('theme').onclick=function(){const d=document.documentElement.classList.toggle('dark');this.textContent=d?'Light theme':'Dark theme';this.setAttribute('aria-pressed',d)};
document.getElementById('system').onchange=function(){document.querySelectorAll('tr[data-system]').forEach(r=>r.hidden=this.value!=='all'&&r.dataset.system!==this.value)};
</script></body></html>"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    data = audit(a.campaign)
    a.output.mkdir(parents=True, exist_ok=False)
    for ext, content in (("json", json.dumps(data, indent=2, allow_nan=False)),
                         ("md", markdown(data)), ("html", html_report(data))):
        (a.output / f"gpu-dft-comparison.{ext}").write_text(content + "\n")
    print(f"Audited {len(data['rows'])} comparisons; outputs: {a.output}")


if __name__ == "__main__":
    main()
