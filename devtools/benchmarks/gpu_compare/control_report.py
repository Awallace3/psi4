"""Publish only finalized matched-grid control evidence, never legacy clocks."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ENGINES = ("psi4", "pyscf")


def _single(values):
    values = {value for value in values if value is not None}
    return next(iter(values)) if len(values) == 1 else None


def _max_memory(summary, name, denominator):
    value = summary.get("memory", {}).get(name, {}).get("max")
    return value / denominator if isinstance(value, (int, float)) else None


def data_for_control(root):
    def load(path):
        return json.loads(path.read_text()) if path.exists() else {}

    build = load(root / "compiled-build-manifest.json")
    identity = load(root / "harness-build-identity.json")
    manifest = load(root / "protein83-panel/manifest.json") or load(root / "peptide-preflight/manifest.json")
    rows, gpu_models = [], set()
    for phase in ("peptide-preflight", "protein83-panel"):
        base = root / phase
        result_file = base / "control-results.json"
        if not result_file.exists():
            continue
        for case in json.loads(result_file.read_text()):
            for basis, summary in case.get("basis_runs", {}).items():
                campaign = base / case["system"] / case["functional"] / case["df_mode"] / basis / "campaign"
                records = [json.loads(p.read_text()) for p in sorted(campaign.glob("*.record.json"))]
                gpu_models.update(r["gpu_name"] for r in records if r.get("gpu_name"))
                eligible = bool(summary.get("complete") and summary.get("comparison_eligible")
                                and summary.get("actual_grid_audit_verified"))
                source = case.get("source_grid", {})
                item = dict(phase=phase, system=case["system"], functional=case["functional"],
                            basis=basis, df_mode=case["df_mode"], eligible=eligible,
                            status="complete" if eligible else "withheld: four-arm audit failed",
                            nonzero_grid_points=source.get("nonzero_weight_points"),
                            energy_delta_eh=summary.get("energy_delta_pyscf_minus_psi4_eh") if eligible else None)
                for engine in ENGINES:
                    group = [r for r in records if r.get("engine") == engine]
                    values = summary.get(engine, {})
                    clock = values.get("full_process_wall_s", {})
                    item[engine] = dict(
                        repeats=values.get("succeeded", 0),
                        nbf=_single(r.get("nbf") for r in group),
                        nmo=_single(r.get("nmo") for r in group),
                        stored_grid_points=_single(r.get("grid_points") for r in group),
                        full_process_s=clock.get("median") if eligible else None,
                        time_min_s=clock.get("min") if eligible else None,
                        time_max_s=clock.get("max") if eligible else None,
                        peak_rss_gib=_max_memory(values, "peak_rss_kib", 1024**2) if eligible else None,
                        sampled_gpu_device_gib=_max_memory(
                            values, "sampled_gpu_device_peak_mib", 1024) if eligible else None,
                        sampled_gpu_process_gib=_max_memory(
                            values, "sampled_gpu_process_peak_mib", 1024) if eligible else None)
                rows.append(item)
    return dict(
        rows=rows,
        meta=dict(job_id=manifest.get("job_id", "unknown"),
                  gpu_model=_single(gpu_models) or manifest.get("gpu", {}).get("name", "GPU not recorded"),
                  compiled_commit=identity.get("compiled_commit", build.get("commit", "unknown")),
                  harness_commit=identity.get("harness_commit", "unknown"),
                  phase_coverage={phase: dict(
                      passed=sum(r["eligible"] for r in rows if r["phase"] == phase),
                      attempted=sum(r["phase"] == phase for r in rows))
                      for phase in ("peptide-preflight", "protein83-panel")}),
        protocol="One exported actual cuEST quadrature per functional/DF strategy, "
                 "reused by both engines and both bases; overlap rank/spectrum audited. "
                 "This is NOT the smaller-native-default-grid experiment.",
        timing="Fresh worker launch-to-reap including interpreter startup, imports, DF/grid "
               "setup, SCF, validation, result writing and teardown; median over timed repeats.",
        memory="Peak RSS is Linux child ru_maxrss. GPU values are sampled lower bounds, "
               "not exact peaks; whole-device readings are not process-attributed. "
               "Device/process queries occur separately and can miss different transients.",
    )


def render(data, template, revision):
    body = (HERE / "control-report-body.html").read_text()
    body = body.replace("ANALYSIS_REVISION", revision)
    body = body.replace("REPORT_DATE", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    result = re.sub(r"<main>.*?</main>", lambda _: body, template, flags=re.S)
    result = result.replace("TOPIC", "Matched-grid GPU DFT").replace("PROJECT", "Psi4 / GPU4PySCF")
    result = result.replace("YYYY-MM-DD", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    result = re.sub(r'(<script type="application/json" id="data">).*?(</script>)',
                    lambda m: m[1] + payload + m[2], result, flags=re.S)
    result = re.sub(r"/\* demo figure:.*?(?=/\* code panes:)", "", result, flags=re.S)
    js = (HERE / "control-report-charts.js").read_text()
    result = result.replace("/* code panes:", js + "\n/* code panes:", 1)
    result = result.replace("</style>", """
body {font-size:19px} .sub,.cap {font-size:16px}
.grid2 {grid-template-columns:repeat(auto-fit,minmax(min(100%,800px),1fr))}
table {font-size:17px} th {font-size:16px} button,select,.legend {font-size:17px}
td,th {padding:8px 12px} .chartscroll {overflow-x:auto}
svg.chart {min-width:760px} svg.chart text {font-size:16px}
.notice {border-left:3px solid var(--s4);padding:8px 16px;background:var(--surface-1)}
.failed {color:var(--bad)} svg.chart text.failed {fill:var(--bad)} .ok {color:var(--s3)}
.filterbar {display:flex;gap:18px;flex-wrap:wrap;margin:20px 0}
.filterbar label {display:grid;gap:4px;font-size:16px;color:var(--text-secondary)}
pre {overflow:auto;font-size:15px}
</style>""", 1)
    # This report uses neither mathematics nor code panes; make it self-contained
    # rather than depend on external CDNs for unused renderer globals.
    result = re.sub(r'<link[^>]+href="https://cdn\.jsdelivr\.net/[^"]+"[^>]*>\n?', "", result)
    result = re.sub(r'<script defer src="https://cdn\.jsdelivr\.net/[^"]+"></script>\n?', "", result)
    result = result.replace("  renderMathInElement(document.body, {",
                            "  if (window.renderMathInElement) renderMathInElement(document.body, {")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    data = data_for_control(args.control)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(data, args.template.read_text(), args.revision) + "\n")
    print(f"Published {sum(r['eligible'] for r in data['rows'])}/{len(data['rows'])} eligible comparisons: {args.output}")


if __name__ == "__main__":
    main()
