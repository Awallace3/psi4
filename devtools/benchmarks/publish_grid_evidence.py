#!/usr/bin/env python3
"""Verify archived completed runs and publish compact, reproducible report data."""
import argparse
import copy
import contextlib
import hashlib
import importlib.util
import json
import io
from pathlib import Path
import re
import shutil
import subprocess
import sys

spec = importlib.util.spec_from_file_location("grid_evidence", Path(__file__).with_name("grid_evidence.py"))
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)
memory_spec = importlib.util.spec_from_file_location("grid_memory", Path(__file__).with_name("grid_memory.py"))
memory_algorithm = importlib.util.module_from_spec(memory_spec)
memory_spec.loader.exec_module(memory_algorithm)
SCIENCE = "c28950041bbe8bb10617264a7b6ddc56a436cda1"
HARNESS = "17b80ed87d7b7d6c07ab4523f4bca9daa28d446d"
PAIRED_JOB = "14057019"
PAIRED_HARNESS = "f59d78d3f7307550753dc17a31f38d4affa5d246"


def load(path):
    return json.loads(path.read_text())


def completed(root, job_id, harness=HARNESS, backends=("host", "cuda"), binary_directory=None):
    assert (root / "metadata/COMPLETED").is_file() and (root / "VERIFIED").is_file()
    runtime = dict(line.split("=", 1) for line in (root / "metadata/runtime.env").read_text().splitlines()
                   if "=" in line)
    assert runtime["job_id"] == job_id and runtime["exit_code"] == "0" and runtime["term_received"] == "0"
    assert runtime["scientific_commit"] == SCIENCE and runtime["harness_commit"] == harness
    assert "H200" in (root / "gpu.txt").read_text()
    scoped = load(root / "supported-workload.json")
    assert scoped["scoped_validation_passed"] and not scoped["full_sapt_validation_passed"]
    assert scoped["scientific_commit"] == SCIENCE
    for backend in backends:
        expected = scoped["backends"]["host" if backend == "cpu" else backend]["core_sha256"]
        assert load((binary_directory or root) / f"binary-{backend}.json")["core_sha256"] == expected
    return runtime


def operation_partition(row, report, operations, *, return_run=False):
    records = {k: dict(v, timer_path=k.split(";")) for k, v in row["timers"].items()}
    tree = report.tree_from_records(records)
    top = {n["name"]: n for n in tree["children"]}
    stages = [report.partition(top[t]) for _, _, t in report.STAGES]
    simple = {n["name"]: n["wall"] for n in tree["children"]}
    post = dict(
        hfTerms=sum(v for k, v in simple.items() if k.startswith("SAPT(HF)") or k.startswith("JK:")),
        d4=simple["SAPT(DFT):D4 Interaction Energy"],
        build=simple["SAPT(DFT):Build JK"],
        terms=sum(simple[k] for k in ("SAPT(DFT):elst", "SAPT(DFT):exch", "SAPT(DFT):ind")))
    bins = {k: sum(s["bins"][k] for s in stages) for k in report.CATS}
    bins["post"] = sum(post.values())
    bins["outer"] = row["wall_s"] - sum(s["wall"] for s in stages) - bins["post"]
    values = operations.operation_values(dict(bins=bins, groups=post, wall=row["wall_s"]))
    # Split, rather than add to, the inclusive XC segment.
    values[:1] = [bins["functional"], bins["xcRest"]]
    assert abs(sum(values) - row["wall_s"]) < 1e-7 and min(values) >= -1e-8
    assert abs(values[0] - row["functional_s"]) < 1e-7
    assert abs(values[0] + values[1] - row["xc_s"]) < 1e-7
    if return_run:
        # CPU has no separately timed cuEST functional child. Keep inclusive XC,
        # rather than interpreting the absent child as zero functional cost.
        return dict(wall=row["wall_s"], bins=bins, groups=post, path=row["result_path"])
    return values


def paired_campaigns(root, report, operations):
    """Recompute the matched CPU/GPU cohort with its frozen controller."""
    directory = root / "timing"
    completed(root, PAIRED_JOB, PAIRED_HARNESS, ("cpu", "host", "cuda"), directory)
    frozen = Path(__file__).resolve().parents[3] / "saptdft-cuest-cpu-gpu-grids-v2"
    assert subprocess.check_output(["git", "-C", str(frozen), "rev-parse", "HEAD"],
                                   text=True).strip() == PAIRED_HARNESS
    finished, manifest, evidence = (load(directory / name)
                                   for name in ("COMPLETE.json", "campaign.json", "evidence.json"))
    assert finished["ok"] and finished["count"] == 108
    assert manifest["job_id"] == PAIRED_JOB and manifest["scientific_commit"] == SCIENCE
    assert manifest["threads"] == 8 and manifest["repeats"] == 3
    assert not manifest["microprofiling"] and not manifest["memory_sampler"]
    for filename, key in (("cpu_gpu_grids.py", "harness_sha256"),
                          ("grid_evidence.py", "grid_worker_sha256")):
        source = subprocess.check_output(["git", "-C", str(frozen), "show",
            f"{PAIRED_HARNESS}:devtools/benchmarks/{filename}"])
        assert hashlib.sha256(source).hexdigest() == manifest[key]
        assert (frozen / "devtools/benchmarks" / filename).read_bytes() == source
    pinned_spec = importlib.util.spec_from_file_location("pinned_cpu_gpu_grids",
        frozen / "devtools/benchmarks/cpu_gpu_grids.py")
    pinned = importlib.util.module_from_spec(pinned_spec)
    pinned_spec.loader.exec_module(pinned)
    assert set(manifest["systems"]) == set(pinned.grid.SYSTEMS)
    assert tuple(manifest["arms"]) == pinned.ARMS and tuple(manifest["routes"]) == pinned.ROUTES
    assert len(manifest["records"]) == 108
    expected = {f"{s}-{a}-{b}-{r}" for s in manifest["systems"]
                for a in pinned.ARMS for b in pinned.ROUTES for r in (1, 2, 3)}
    assert {r["name"] for r in manifest["records"]} == expected
    rows = []
    for record in manifest["records"]:
        assert record["returncode"] == 0
        row = load(directory / record["name"] / "result.json")
        assert record["name"] == f'{row["system"]}-{row["arm"]}-{row["backend"]}-{row["repeat"]}'
        assert row["ok"] and row["job_id"] == PAIRED_JOB and row["scientific_commit"] == SCIENCE
        assert row["options"]["SAPT_DFT_INDUCTION_TYPE"] == "NONE"
        assert not any("XC micro:" in key for key in row["timers"])
        assert row["mapped_libxc_sha256"], "No actual loaded LibXC receipt"
        row["result_path"] = record["result"]
        operation_partition(row, report, operations)
        rows.append(row)
    recomputed = pinned.summarize(rows)
    assert json.loads(json.dumps(recomputed)) == evidence
    assert len(recomputed["groups"]) == 4 and not recomputed["failures"]
    assert finished["accuracy_pass"] == all(r["acceptable_speedup"]
        for g in evidence["groups"] for a in g["arms"]
        for b, r in a["routes"].items() if b != "cpu")
    result = []
    titles = dict(fixed99="Fixed 99×590", aggressive99="35×194 → 99×590",
                  coarse75="35×194 → 75×302")
    for backend in ("host", "cuda"):
        title = f'Matched CPU / {"Host" if backend == "host" else "CUDA"} LibXC GPU · job {PAIRED_JOB}'
        for arm in pinned.ARMS:
            cases = {}
            for group in evidence["groups"]:
                routes = next(a["routes"] for a in group["arms"] if a["arm"] == arm)
                cpu, gpu = routes["cpu"], routes[backend]
                cases[f'{group["system"]} / {group["basis"]}'] = dict(
                    cpu=operation_partition(cpu["representative"], report, operations, return_run=True),
                    gpu=operation_partition(gpu["representative"], report, operations, return_run=True),
                    accuracy_pass=gpu["acceptable_speedup"],
                    accuracy_note=(f'Component target 1e-5 Eh: CPU grid error {cpu["grid_error_eh"]:.3e}; '
                        f'GPU grid error {gpu["grid_error_eh"]:.3e}; same-grid CPU/GPU '
                        f'error {gpu["cpu_gpu_error_eh"]:.3e}.'),
                    speedup=gpu["speedup"])
            result.append(dict(id=f"cpu-gpu-{PAIRED_JOB}-{backend}-{arm}",
                comparison_cohort=f"cpu-gpu-{PAIRED_JOB}-{backend}", title=title,
                grid_arm=arm, grid_title=titles[arm], cases=cases, job_id=PAIRED_JOB,
                source=f"cpu-gpu-grid-results-{PAIRED_JOB}/timing/evidence.json"))
    return result


def timing_campaigns(root, report, operations, job_id="14046460", harness=HARNESS, arms=None):
    selected = list(experiment.ARMS) if arms is None else list(arms)
    count = len(experiment.SYSTEMS) * len(selected) * 3
    completed(root, job_id, harness)
    campaigns = []
    for backend in ("host", "cuda"):
        directory = root / backend
        finished, manifest, evidence = (load(directory / name)
                                       for name in ("COMPLETE.json", "campaign.json", "evidence.json"))
        assert finished["ok"] and finished["count"] == count
        assert manifest["job_id"] == job_id and manifest["scientific_commit"] == SCIENCE
        assert manifest["memory_sampler"] is False and manifest["microprofiling"] is False
        assert manifest["repeats"] == 3 and len(manifest["records"]) == count
        assert set(manifest["arms"]) == set(selected)
        rows, names = [], set()
        for record in manifest["records"]:
            name = record["name"]
            assert name not in names and Path(name).name == name and record["returncode"] == 0
            names.add(name)
            row = load(directory / name / "result.json")
            assert row["job_id"] == job_id and row["backend"] == backend and row["ok"]
            assert row["scientific_commit"] == SCIENCE
            assert row["options"]["SAPT_DFT_INDUCTION_TYPE"] == "NONE"
            assert row["options"]["CUEST_XC_TIMING"] is False
            assert not any("XC micro:" in key for key in row["timers"])
            row["result_path"] = record["result"]
            # Validate every run, not only the published median-nearest sample.
            operation_partition(row, report, operations)
            rows.append(row)
        recomputed = experiment.summarize(rows, arms=selected)
        assert json.loads(json.dumps(recomputed["groups"])) == evidence["groups"]
        assert not evidence["failures"] and len(evidence["groups"]) == 4
        assert finished["accuracy_pass"] == all(a["agreement_pass"] for g in evidence["groups"] for a in g["arms"])
        groups = copy.deepcopy(evidence["groups"])
        for group in groups:
            for arm in group["arms"]:
                r = arm["representative"]
                arm["representative"] = {k: r[k] for k in (
                    "xc_s", "functional_s", "switch_s", "scf_s", "initial_iterations",
                    "final_iterations", "all_iterations", "nbf", "effective_grid_options",
                    "geometry_sha256", "result_path")}
                assert r["wall_s"] == arm["median_s"]
                arm["representative"]["wall_s"] = r["wall_s"]
                arm["representative"]["operation_values"] = operation_partition(r, report, operations)
        campaigns.append(dict(backend="Host LibXC" if backend == "host" else "CUDA LibXC",
            backend_key=backend, job_id=job_id, scientific_commit=SCIENCE, harness_commit=harness,
            core_sha256=load(root / f"binary-{backend}.json")["core_sha256"],
            groups=groups, complete_cases=count, all_accuracy_pass=finished["accuracy_pass"],
            source=f"grid-results-{job_id}/{backend}/evidence.json"))
    return campaigns


def memory_measurements(root):
    completed(root, "14046461")
    report = load(root / "memory-evidence.json")
    assert report["complete"] and report["expected_cases"] == len(report["cases"]) == 30
    assert report["eligible_for_speedup_bars"] is False
    expected = {(b, f"{s}-{a}") for b in ("host", "cuda")
                for s in ("water", "nanotube", "protein83") for a in experiment.ARMS}
    assert {(r["backend"], r["case"]) for r in report["cases"]} == expected
    groups = []
    for backend in ("host", "cuda"):
        for system in ("water", "nanotube", "protein83"):
            arms, dimensions = [], set()
            for arm in experiment.ARMS:
                case = f"{system}-{arm}"
                row = next(r for r in report["cases"] if r["backend"] == backend and r["case"] == case)
                summary = load(root / backend / case / "memory/summary.json")
                assert all(row[k] == v for k, v in summary.items())
                samples_path = root / backend / case / "memory/samples.jsonl"
                host = {}
                if samples_path.is_file():
                    samples = [json.loads(line) for line in samples_path.read_text().splitlines()]
                    recomputed = memory_algorithm.summarize_samples(samples)
                    assert all(summary[k] == v for k, v in recomputed.items()), "Raw NVML summary mismatch"
                    for source, target in (("VmRSS_kib", "sampled_host_rss_peak_mib"),
                                           ("VmHWM_kib", "observed_host_hwm_mib")):
                        values = [r["host"][source] / 1024 for r in samples if source in r.get("host", {})]
                        host[target] = max(values, default=None)
                assert row["ok"] and row["returncode"] == 0 and row["gpu_observed"]
                assert row["failed_samples"] == row["unavailable_samples"] == 0
                assert row["job_id"] == "14046461" and not row["eligible_for_speedup_bars"]
                worker = load(root / backend / case / "case/result.json")
                assert worker["ok"] and worker["diagnostic_only"] and not worker["eligible_for_speedup_bars"]
                assert worker["options"]["CUEST_XC_TIMING"] is False
                boundaries = [json.loads(line) for line in
                              (root / backend / case / "memory/boundaries.jsonl").read_text().splitlines()]
                dims = {(r["nbf"], r["naux"]) for r in boundaries if r.get("nbf") and r.get("naux")}
                dimensions.add(max(dims))
                peak_phases = sorted(row["phases"].items(),
                    key=lambda kv: kv[1]["sampled_peak_mib"] or 0, reverse=True)[:3]
                arms.append(dict(arm=arm, peak_mib=row["sampled_peak_mib"],
                    samples=row["samples"], max_interval_s=row["max_observed_interval_s"],
                    peak_phases=[dict(phase=k, **v) for k, v in peak_phases], **host))
            assert len(dimensions) == 1, "A matched grid comparison changed NBF/Naux"
            nbf, naux = dimensions.pop()
            groups.append(dict(backend=backend, system=system, nbf=nbf, naux=naux, arms=arms))
    return groups, max(r["max_observed_interval_s"] for r in report["cases"])


def basis_measurements(root):
    harness = "c451268473619df4b0024fc1e34017b34e8da928"
    completed(root, "14049789", harness, ("host",))
    report = load(root / "basis-memory-evidence.json")
    assert report["complete"] and report["diagnostic_only"] and not report["eligible_for_speedup_bars"]
    cases = report["cases"]
    expected = {"primary634", "primary766", "control922", "primary1293", "primary1786", "aux4762"}
    assert {r["name"] for r in cases} == {f"{c}-{i}" for c in expected for i in (1, 2)}
    groups, gap = [], 0
    for case in sorted(expected, key=lambda c: next(r["worker"]["nbf"] for r in cases
                                                  if r["worker"]["case"] == c)):
        rows = [r for r in cases if r["worker"]["case"] == case]
        peaks, phases = [], {"plan_ready": [], "jk_initialize": [], "JK_compute": []}
        for row in rows:
            directory = root / row["name"]
            worker, summary = load(directory / "case/result.json"), load(directory / "memory/summary.json")
            assert worker == row["worker"] and summary == row["memory"]
            assert worker["ok"] and worker["synthetic_coefficients"] and not worker["xc_grid_constructed"]
            assert not worker["cpu_smoke"] and worker["nocc"] == 146
            assert worker["job_id"] == "14049789" and worker["scientific_commit"] == SCIENCE
            assert summary["ok"] and summary["returncode"] == 0 and summary["gpu_observed"]
            assert summary["failed_samples"] == summary["unavailable_samples"] == 0
            samples = [json.loads(s) for s in (directory / "memory/samples.jsonl").read_text().splitlines()]
            recomputed = memory_algorithm.summarize_samples(samples)
            assert all(summary[k] == v for k, v in recomputed.items())
            gap = max(gap, summary["max_observed_interval_s"])
            peaks.append(summary["sampled_peak_mib"] / 1024)
            for phase in phases:
                values = [v["sampled_peak_mib"] / 1024 for k, v in summary["phases"].items()
                          if k.startswith(f"basis_scan:{phase}:") or k.startswith(f"{phase}:")]
                assert values
                phases[phase].append(max(values))
        workers = [r["worker"] for r in rows]
        for key in ("nbf", "naux", "primary", "auxiliary", "queried_workspaces"):
            assert workers[0][key] == workers[1][key]
        w = workers[0]
        assert w["naux"] == (4762 if case == "aux4762" else 3735)
        groups.append(dict(case=case, **{k: w[k] for k in ("nbf", "naux", "primary", "auxiliary")},
            repeats=2, peak_gib_range=[min(peaks), max(peaks)],
            phase_gib_ranges={k: [min(v), max(v)] for k, v in phases.items()},
            queried_gib={r["allocation"]: r["queried_gib_rounded"] for r in w["queried_workspaces"]}))
    return dict(job_id="14049789", submission_state="COMPLETED 0:0; 12/12 probes",
        complete=True, scientific_commit=SCIENCE, harness_commit=harness, groups=groups,
        scope="Same protein83 geometry; NBF 634–1786 and Naux 3735/4762; 146 occupied columns.",
        max_observed_interval_s=gap,
        limitations="JK-only synthetic dense coefficients, no XC grid, SCF energy or SAPT calculation. "
        "Two fresh processes per configuration. Sampled peaks are lower bounds, not an allocation ledger. "
        "Queried workspaces are rounded to 0.001 GiB and are not additive peak measurements.")


def publish(archives, docs):
    speed, memory = archives / "gpu-speed-14046460", archives / "gpu-memory-14046461"
    sys.path.insert(0, str(docs.resolve()))
    import build_breakdown_2026_10_04 as report
    import operation_pairs as operations
    campaigns = timing_campaigns(speed, report, operations)
    measurements, gap = memory_measurements(memory)
    data = load(docs / "grid-evidence.json")
    paired_status = data.setdefault("cpu_gpu_grid_status", {})
    paired_status.update(job_id=PAIRED_JOB, scientific_commit=SCIENCE,
        harness_commit=PAIRED_HARNESS, expected_calculations=108, monitor_interval_s=1800,
        route="gts-cs207 / gpu-h200 / inferno",
        arms=["fixed99", "aggressive99", "coarse75"])
    paired_root = archives / f"gpu-cpu-grid-{PAIRED_JOB}"
    if (paired_root / "artifacts.tar.gz").is_file():
        data["cpu_gpu_grid_campaigns"] = paired_campaigns(paired_root, report, operations)
        paired_status.update(complete=True, status=(
            f"Job {PAIRED_JOB}: 108/108 timing calculations verified. Select its evidence cohort "
            "and a grid scheme below. Accuracy failures remain visible but are not acceptable speedups."))
    else:
        data.pop("cpu_gpu_grid_campaigns", None)
        paired_status.update(complete=False, status=(
            f"Job {PAIRED_JOB} submitted for 108 matched calculations: three grids × three routes "
            "× four systems × three repeats. PENDING / Priority at submission, not a live queue "
            "reading. Sparse CPU/GPU ratios are withheld until completed runs are verified; "
            "historical evidence below retains its fixed 99×590 grid."))
    followup = data.setdefault("aggressive_grid_followup", dict(
        job_id="14053288", scientific_commit=SCIENCE,
        harness_commit="b82584e988ac850a85161a8608fd352bc41dabcc",
        state="Submitted; PENDING / Priority at submission (not a live queue reading)",
        complete=False, monitor_interval_s=1800,
        route="gts-cs207 / gpu-h200 / inferno",
        arms=["fixed99", "quarter99", "aggressive99"], expected_calculations=72))
    aggressive = archives / f'gpu-aggressive99-{followup["job_id"]}'
    if (aggressive / "artifacts.tar.gz").exists():
        new_campaigns = timing_campaigns(aggressive, report, operations,
            followup["job_id"], followup["harness_commit"], followup["arms"])
        campaigns.extend(new_campaigns)
        followup.update(complete=True, state="COMPLETED 0:0; 72/72 verified timing calculations")
        followup["results"] = []
        for campaign in new_campaigns:
            for group in campaign["groups"]:
                arms = {a["arm"]: a for a in group["arms"]}
                base, quarter, small = (arms[k] for k in ("fixed99", "quarter99", "aggressive99"))
                followup["results"].append(dict(
                    backend=campaign["backend"], system=group["system"], basis=group["basis"],
                    fixed99_s=base["median_s"], quarter99_s=quarter["median_s"],
                    aggressive99_s=small["median_s"], speedup=small["speedup"],
                    error_eh=small["max_component_error_eh"], agreement_pass=small["agreement_pass"],
                    quarter_final_iterations=quarter["representative"]["final_iterations"],
                    aggressive_final_iterations=small["representative"]["final_iterations"]))
        followup["all_aggressive_accuracy_pass"] = all(r["agreement_pass"] for r in followup["results"])
        followup["worst_aggressive_error_eh"] = max(r["error_eh"] for r in followup["results"])
        followup["cohort_warning"] = (
            "Job 14053288 has its own matched fixed99 and quarter99 controls. Absolute baselines "
            "differ substantially from job 14046460 (Protein83 CUDA fixed99: 117.942 versus "
            "70.916 s); the cause was not isolated. Do not pool jobs or compare a new arm with an "
            "older denominator. The rainbow backend selector names each job explicitly.")
    data["campaigns"] = campaigns
    data["rainbow_operations"] = [
        ["XC functional region", "#59d4e8"], ["XC outside functional region", "#3697bf"],
        *[[label, color] for label, color, _ in operations.OPERATIONS[1:]]]
    data["partition_source_sha256"] = {
        name: hashlib.sha256((docs / name).read_bytes()).hexdigest()
        for name in ("build_breakdown_2026_10_04.py", "operation_pairs.py")}
    data["status"] = ("Completed 120 matched timing runs and 30 separate memory diagnostics on H200. "
        "Protein83 coarse75: 2.23x host LibXC and 1.43x CUDA LibXC speedup. "
        "75x302 fails nanotube accuracy (~1.38e-5 Eh); 99x590 switching passes. "
        "These historical cohorts use induction NONE/D4(I), not CPKS; separately validated "
        "LDA/GGA cuEST response on the unmerged sibling branch is described in #cpks.")
    if followup["complete"]:
        data["status"] += " A separate matched 72-run final99 aggressive-grid follow-up is also complete."
    data["speed_job"].update(submission_state="COMPLETED 0:0; 120/120 calculations",
                             accuracy_pass_all_arms=False)
    data["accuracy_warning"] = ("Do not choose 75x302 universally: both fixed75 and coarse75 fail "
        "the nanotube 1e-5 Eh component target in both backends. Those timings remain visible but "
        "are not acceptable speedups. Water, peptide and protein83 pass.")
    data["memory_diagnostics"].update(
        status=("Completed 30/30 independent diagnostics, no unavailable/failed samples. "
                "Protein83 sampled peak reduction is 3.1% host / 5.5% CUDA for coarse75 versus fixed99; "
                "coarse99 does not lower that system's sampled peak. These are single-run sampled "
                "lower bounds, not exact allocation high-water marks or demonstrated OOM cures."),
        measurements=measurements, max_observed_interval_s=gap,
        measured_findings=[
            "Protein83 has NBF=922, Naux=3735 in all grid arms. CUDA fixed99/quarter99/third99 peak: "
            "16682 MiB; fixed75/coarse75: 15758 MiB. The measured reduction comes from the smaller "
            "final grid, not an observed benefit from coarse-to-final switching alone.",
            "Host LibXC protein83: 15964 MiB (99 final) versus 15468 MiB (75 final). "
            "Grid reduction changes total residency far less than the nominal point-count ratio.",
            "The largest protein83 sample occurs between SCFs, with nearly the same residency during "
            "DFT form_G. Retained plans/buffers coexist; Python phase hooks cannot attribute every "
            "byte to internal C++ JK versus XC.",
            "Water peaks vary sharply across arms. Sampling alone cannot distinguish missed short "
            "spikes from discrete library workspace choices; do not fit a grid-memory scaling law "
            "to those peaks.",
            "The separate same-geometry JK-only basis scan below isolates primary/auxiliary changes. "
            "It does not measure full SAPT memory or reproduce an OOM.",
        ])
    queried_path = memory / "queried-jk-workspaces.json"
    if queried_path.exists():
        queried = load(queried_path)
        assert queried["job_id"] == "14046461" and len(queried["cases"]) == 30
        protein = [r for r in queried["cases"] if r["case"].startswith("protein83-")]
        requests = [{kind: sorted({v["queried_gib_rounded"] for v in r["workspace_records"]
                                  if v["allocation"] == kind})
                     for kind in ("AO pair list", "DF int plan", "J/K scratch")} for r in protein]
        assert len(requests) == 10 and all(r == requests[0] for r in requests)
        data["memory_diagnostics"]["queried_jk_protein83_gib"] = requests[0]
        data["memory_diagnostics"]["measured_findings"].insert(0,
            "The library's queried Protein83 DF-plan requests reach 6.622 GiB; J/K scratch "
            "requests reach 7.449 GiB, unchanged across all grid arms and both backends. "
            "These are separate queried workspace maxima, printed to 0.001 GiB, not an "
            "additive allocation peak. Construction temporaries are not included.")
    sources = [(speed, "grid-results-14046460", True), (memory, "grid-memory-14046461", False)]
    if followup["complete"]:
        sources.append((aggressive, f'grid-results-{followup["job_id"]}', True))
    for root, name, is_timing in sources:
        target = docs / name
        target.mkdir(exist_ok=True)
        shutil.copyfile(root / "artifacts.tar.gz", target / "artifacts.tar.gz")
        for relative in ("supported-workload.json", "binary-host.json", "binary-cuda.json", "gpu.txt"):
            shutil.copyfile(root / relative, target / relative)
        if is_timing:
            for backend in ("host", "cuda"):
                (target / backend).mkdir(exist_ok=True)
                for filename in ("evidence.json", "campaign.json", "COMPLETE.json"):
                    shutil.copyfile(root / backend / filename, target / backend / filename)
        else:
            shutil.copyfile(root / "memory-evidence.json", target / "memory-evidence.json")
            if queried_path.exists():
                shutil.copyfile(queried_path, target / "queried-jk-workspaces.json")
            if (root / "memory-samples.tar.gz").is_file():
                shutil.copyfile(root / "memory-samples.tar.gz", target / "memory-samples.tar.gz")
    data["artifact_sha256"] = {
        name: hashlib.sha256((docs / name / "artifacts.tar.gz").read_bytes()).hexdigest()
        for _, name, _ in sources}
    if (memory / "memory-samples.tar.gz").is_file():
        data["artifact_sha256"]["grid-memory-14046461/memory-samples.tar.gz"] = hashlib.sha256(
            (memory / "memory-samples.tar.gz").read_bytes()).hexdigest()
    basis = archives / "gpu-basis-memory-14049789"
    if basis.exists():
        data["memory_diagnostics"]["basis_scan"] = basis_measurements(basis)
        target = docs / "grid-basis-memory-14049789"
        target.mkdir(exist_ok=True)
        for filename in ("artifacts.tar.gz", "basis-memory-evidence.json", "supported-workload.json",
                         "binary-host.json", "gpu.txt", "VERIFIED"):
            shutil.copyfile(basis / filename, target / filename)
        data["artifact_sha256"]["grid-basis-memory-14049789"] = hashlib.sha256(
            (target / "artifacts.tar.gz").read_bytes()).hexdigest()
    if paired_status["complete"]:
        target = docs / f"cpu-gpu-grid-results-{PAIRED_JOB}"
        target.mkdir(exist_ok=True)
        for relative in ("artifacts.tar.gz", "supported-workload.json", "gpu.txt", "VERIFIED",
                         "metadata/runtime.env", "metadata/COMPLETED",
                         "timing/campaign.json", "timing/evidence.json", "timing/COMPLETE.json",
                         "timing/binary-cpu.json", "timing/binary-host.json", "timing/binary-cuda.json"):
            (target / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(paired_root / relative, target / relative)
        data["artifact_sha256"][target.name] = hashlib.sha256(
            (target / "artifacts.tar.gz").read_bytes()).hexdigest()
    (docs / "grid-evidence.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    print("Verified/recomputed timing, grid-memory and available basis-memory evidence; published.")


def render_preserving_history(docs):
    """Refresh the dated page without advancing any historical measurement."""
    sys.path.insert(0, str(docs.resolve()))
    import build_breakdown_2026_10_04 as report
    original = hashlib.sha256((docs / "breakdown.html").read_bytes()).hexdigest()
    page = docs / "breakdown-2026-10-04.html"

    def historical():
        payload = json.loads(re.search(r'<script id="(?:measurements|data)"[^>]*>(.*?)</script>',
                                       page.read_text(), re.S)[1])
        payload.pop("gridEvidence", None)
        # Editorial correction only: keep every historical number/receipt byte-for-byte
        # in the decoded payload, while qualifying the formerly page-wide backend claim.
        description = payload["categories"]["functional"][1]
        assert description in (
            "Host LibXC in every campaign on this page",
            "Host LibXC in the historical lifecycle cohorts; backend is cohort-specific elsewhere")
        payload["categories"]["functional"][1] = "Historical lifecycle: host LibXC"
        return payload

    before = historical()
    with contextlib.redirect_stdout(io.StringIO()):
        report.main(update_original=False)
    assert historical() == before
    assert hashlib.sha256((docs / "breakdown.html").read_bytes()).hexdigest() == original
    print("Dated report rendered; historical payload and original report preserved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    publish(args.archives, args.docs)
    if args.render:
        render_preserving_history(args.docs)
