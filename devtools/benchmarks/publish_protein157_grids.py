#!/usr/bin/env python3
"""Verify the dedicated Protein157 cohort without changing historical validators.

This is GPU-only grid evidence, never a CPU/GPU speedup denominator. The
capacity query is not an allocated peak; memory diagnostics are not timings.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import publish_grid_evidence as publication

JOB = "14074070"
HARNESS = "006bf18f75509befb103d35134418d156589aecc"
SCIENCE = publication.SCIENCE
CORE = "91d6018627e830cb7a088bf44b6e715c9a9694aaebb86907b78186f03735b8d7"
GEOMETRY = "a1e9c8949bb5950fdb17c07ab718cead826835f042a6b6e94c5cb7dcf75b6b60"
ARMS = ("fixed99", "fixed75", "coarse75")
SOURCE = f"protein157-grid-results-{JOB}"
load = publication.load


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_controller(frozen, manifest):
    assert subprocess.check_output(
        ["git", "-C", str(frozen), "rev-parse", "HEAD"], text=True).strip() == HARNESS
    for filename, key in (("protein157_grids.py", "harness_sha256"),
                          ("grid_evidence.py", "grid_worker_sha256")):
        source = subprocess.check_output(
            ["git", "-C", str(frozen), "show", f"{HARNESS}:devtools/benchmarks/{filename}"])
        assert hashlib.sha256(source).hexdigest() == manifest[key]
        assert (frozen / "devtools/benchmarks" / filename).read_bytes() == source
    # Load the frozen worker directly: it has no Psi4 import at module scope.
    spec = importlib.util.spec_from_file_location(
        "protein157_frozen_grid", frozen / "devtools/benchmarks/grid_evidence.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    worker.SYSTEMS["protein157"] = manifest["selected"]["basis"]
    return worker


def validate_row(row, manifest, arm, *, diagnostic):
    assert row["ok"] and row["job_id"] == JOB and row["scientific_commit"] == SCIENCE
    assert row["system"] == "protein157" and row["arm"] == arm and row["backend"] == "cuda"
    assert row["nbf"] == manifest["selected"]["nbf"] == 2171
    assert row["auxiliary"] == manifest["selected"]["auxiliary"] == "def2-universal-jkfit"
    assert row["geometry_sha256"] == GEOMETRY
    assert hashlib.sha256(row["geometry"].encode()).hexdigest() == GEOMETRY
    assert str(Path(row["package"]).parent) == manifest["expected_package"]
    assert row["threads"] == manifest["threads"] == 8 and row["memory"] == "112 GiB"
    assert row["diagnostic_only"] is diagnostic
    assert row["eligible_for_speedup_bars"] is (not diagnostic)
    opts = row["options"]
    expected = dict(BASIS="heavy-aug-cc-pvdz", SCF_TYPE="DF",
                    SAPT_DFT_FUNCTIONAL="PBE0", SAPT_DFT_INDUCTION_TYPE="NONE",
                    SAPT_DFT_GRAC_COMPUTE="ITERATIVE", SAPT_DFT_GRAC_SEED_CATION=False,
                    SAPT_DFT_DO_DHF=True, USE_CUEST=True, CUEST_XC=True, CUEST_SAD=True,
                    CUEST_MIXED_PRECISION=False, CUEST_XC_TIMING=False,
                    INTS_TOLERANCE=1.e-12, DFT_GRID_SWITCH_FACTOR=10.)
    assert all(opts[k] == v for k, v in expected.items())
    final = (99, 590) if arm == "fixed99" else (75, 302)
    initial = (35, 194) if arm == "coarse75" else (0, 0)
    for name, value in zip(("DFT_RADIAL_POINTS", "DFT_SPHERICAL_POINTS",
                            "DFT_RADIAL_POINTS_INITIAL", "DFT_SPHERICAL_POINTS_INITIAL"),
                           (*final, *initial)):
        assert opts[name] == value
    assert not any("XC micro:" in k for k in row["timers"])
    assert any("cuEST XC: CUDA LibXC Functional" in k for k in row["timers"])
    assert row["mapped_libxc_sha256"] and all(
        "/libxc-7.1.2-cuda/" in p for p in row["mapped_libxc_sha256"])


def verify(root, selection, frozen, docs):
    runtime = dict(line.split("=", 1) for line in
                   (root / "metadata/runtime.env").read_text().splitlines() if "=" in line)
    assert (root / "VERIFIED").is_file() and (root / "metadata/COMPLETED").is_file()
    assert runtime["job_id"] == JOB and runtime["capacity_job"] == "14073708"
    assert runtime["exit_code"] == "0" and runtime["term_received"] == "0"
    assert "H200" in (root / "gpu.txt").read_text()
    assert "gts-cs207" in (root / "allocation.txt").read_text()
    assert "inferno" in (root / "allocation.txt").read_text()
    scoped = load(root / "supported-workload.json")
    assert scoped["scoped_validation_passed"] and scoped["scientific_commit"] == SCIENCE
    assert load(root / "binary-cuda.json")["core_sha256"] == CORE
    assert scoped["backends"]["cuda"]["core_sha256"] == CORE
    manifest = load(root / "timing/campaign.json")
    finished = load(root / "timing/COMPLETE.json")
    assert manifest["job_id"] == JOB and manifest["scientific_commit"] == SCIENCE
    assert manifest["geometry_sha256"] == GEOMETRY and manifest["variant"] == "cuda"
    assert tuple(manifest["arms"]) == ARMS and manifest["repeats"] == 3
    assert manifest["memory_sampler_for_timing"] is False
    assert finished == dict(records=9, completed=True, accuracy_pass=True)
    capacity = load(selection)
    assert sha(selection) == manifest["source_selection_sha256"]
    assert capacity["complete"] and capacity["job_id"] == "14073708"
    assert capacity["geometry_sha256"] == GEOMETRY and capacity["scientific_commit"] == SCIENCE
    assert capacity["selected"] == manifest["selected"]
    assert capacity["selected"]["naux"] == 7265
    assert capacity["selected"]["query"]["df_plan_allocated"] is False
    worker = frozen_controller(frozen, manifest)
    sys.path.insert(0, str(docs.resolve()))
    import build_breakdown_2026_10_04 as report
    import operation_pairs as operations

    expected = {f"protein157-{a}-{r}" for a in ARMS for r in (1, 2, 3)}
    assert len(manifest["records"]) == 9
    assert {Path(p).parent.name for p in manifest["records"]} == expected
    rows, partitions, libxc = [], {}, set()
    for source in manifest["records"]:
        name = Path(source).parent.name
        row = load(root / "timing" / name / "result.json")
        assert name == f'protein157-{row["arm"]}-{row["repeat"]}'
        validate_row(row, manifest, row["arm"], diagnostic=False)
        partitions[name] = publication.operation_partition(row, report, operations)
        libxc.add(tuple(sorted(row["mapped_libxc_sha256"].items())))
        rows.append(row)
    assert len(libxc) == 1
    recomputed = worker.summarize(rows, repeats=3, arms=ARMS)
    evidence = load(root / "timing/evidence.json")
    assert json.loads(json.dumps(recomputed)) == evidence
    assert not evidence["failures"] and len(evidence["groups"]) == 1
    groups = copy.deepcopy(evidence["groups"])
    for group in groups:
        for arm in group["arms"]:
            assert arm["agreement_pass"] and arm["max_component_error_eh"] <= 1.e-5
            r = arm["representative"]
            name = f'protein157-{r["arm"]}-{r["repeat"]}'
            arm["representative"] = {k: r[k] for k in (
                "wall_s", "xc_s", "functional_s", "switch_s", "scf_s", "initial_iterations",
                "final_iterations", "all_iterations", "nbf", "effective_grid_options",
                "geometry_sha256")}
            arm["representative"].update(operation_values=partitions[name],
                                         result_path=f"{SOURCE}/timing/{name}/result.json")
    memory = []
    for arm in ARMS:
        directory = root / f"memory-{arm}"
        summary = load(directory / "summary.json")
        samples = [json.loads(line) for line in (directory / "samples.jsonl").read_text().splitlines()]
        calculated = publication.memory_algorithm.summarize_samples(samples)
        assert all(summary[k] == v for k, v in calculated.items())
        assert summary["returncode"] == 0 and summary["gpu_observed"]
        assert summary["failed_samples"] == summary["unavailable_samples"] == 0
        row = load(root / f"diagnostic-{arm}/result.json")
        validate_row(row, manifest, arm, diagnostic=True)
        assert tuple(sorted(row["mapped_libxc_sha256"].items())) in libxc
        boundaries = [json.loads(line) for line in (directory / "boundaries.jsonl").read_text().splitlines()]
        dims = {(r["nbf"], r["naux"]) for r in boundaries if r.get("nbf") and r.get("naux")}
        assert (2171, 7265) in dims
        rss = [s["host"]["VmRSS_kib"]/2**20 for s in samples
               if s.get("host", {}).get("VmRSS_kib") is not None]
        memory.append(dict(arm=arm, sampled_peak_gib=summary["sampled_peak_mib"]/1024,
                           samples=summary["samples"], host_rss_gib=max(rss, default=None),
                           max_interval_s=summary["max_observed_interval_s"],
                           source=f"{SOURCE}/memory-{arm}/summary.json"))
    campaign = dict(backend="CUDA LibXC · Protein157", backend_key="cuda", job_id=JOB,
                    scientific_commit=SCIENCE, harness_commit=HARNESS, core_sha256=CORE,
                    groups=groups, complete_cases=9, all_accuracy_pass=True,
                    source=f"{SOURCE}/timing/evidence.json")
    return campaign, dict(job_id=JOB, capacity_job_id="14073708", basis=groups[0]["basis"],
                          nbf=2171, naux=7265, auxiliary=manifest["selected"]["auxiliary"],
                          query=manifest["selected"]["query"], arms=memory,
                          source=f"{SOURCE}/artifacts.tar.gz")


def estimated_cpu_gpu(root, docs, campaign):
    """Use real old CPU work, but never call this a matched speedup."""
    sys.path.insert(0, str(docs.resolve()))
    import build_breakdown_2026_10_04 as report
    import operation_pairs as operations
    relative = ("phoenix-2026-10-04/protein157-inferno/results-13868775/"
                "protein157-6-31+g**-new-cpu-1/result.json")
    source = docs / relative
    assert sha(source) == "2e2a2921d043f2e275af8d2bf878b0b9169628c3d228c1d221b9bc1db70fc94c"
    old = load(source)
    assert old["ok"] and old["slurm_job_id"] == "13868775"
    assert old["basis"] == "6-31+g**" and old["nbf"] == 1786 and old["threads"] == 8
    assert not old["gpu_builder_seen"] and not old["disk_df_builder_seen"]
    assert not old["options"]["USE_CUEST"]
    assert hashlib.sha256(old["geometry"].encode()).hexdigest() == GEOMETRY
    cpu = report.parse_run(source)
    operations.operation_values(cpu)  # Verify complete elapsed-time closure.
    cpu = {k: cpu[k] for k in ("wall", "bins", "groups", "path")}
    cpu.update(label="CPU · 6-31+g** · eight threads · job 13868775",
               grid_title="Historical fixed 99×590", nbf=1786)
    labels = dict(fixed99="Fixed 99×590", fixed75="Fixed 75×302 · Psi4 default",
                  coarse75="35×194 → 75×302 · Psi4 default final")
    estimates = []
    for arm in campaign["groups"][0]["arms"]:
        rep = arm["representative"]
        name = Path(rep["result_path"]).parent.name
        row = load(root / "timing" / name / "result.json")
        row["result_path"] = rep["result_path"]
        gpu = publication.operation_partition(row, report, operations, return_run=True)
        gpu.update(label="GPU · heavy-aug-cc-pVDZ · eight host threads · H200 / CUDA LibXC",
                   grid_title=labels[arm["arm"]], nbf=2171)
        note = (
            "ESTIMATE ONLY: historical CPU 6-31+g** (NBF 1786), fixed 99×590, one run "
            "from job 13868775; GPU heavy-aug-cc-pVDZ (NBF 2171), selected grid, "
            "three-run median from job 14074070. Same geometry and eight threads, "
            "but basis, grid, build, allocation and date differ; this elapsed-time ratio "
            "is not a controlled GPU speedup or a CPU/GPU numerical-agreement test. "
            "CPU is reused unchanged for every GPU grid selection.")
        name = "protein157 / heavy-aug-cc-pVDZ (GPU); 6-31+g** (CPU)"
        estimates.append(dict(
            id=f"protein157-estimate-{arm['arm']}",
            comparison_cohort="protein157-historical-cpu-estimate",
            title="Protein157 · ESTIMATE · historical CPU / current GPU",
            estimated=True, grid_arm=arm["arm"], grid_title=labels[arm["arm"]],
            cases={name: dict(cpu=cpu, gpu=gpu, comparison_note=note)},
            source_cpu_sha256=sha(source), source_cpu_job="13868775",
            source_gpu_job=JOB))
    return estimates


def publish(root, selection, frozen, docs, render=False):
    campaign, memory = verify(root, selection, frozen, docs)
    path = docs / "grid-evidence.json"
    data = load(path)
    before = copy.deepcopy(data)
    # Idempotent append: never reconstruct or weaken older cohort validators.
    data["campaigns"] = [c for c in data["campaigns"] if c["job_id"] != JOB] + [campaign]
    data["protein157"] = memory
    data["protein157"]["scope"] = (
        "GPU-only SAPT(DFT)-D4(I)/heavy-aug-cc-pVDZ; PBE0, induction NONE, "
        "iterative unseeded GRAC, delta-HF and native FP64. Not a CPU/GPU or CPKS comparison.")
    data["cpu_gpu_estimate_campaigns"] = estimated_cpu_gpu(root, docs, campaign)
    defaults = subprocess.check_output(
        ["git", "-C", str(frozen), "show", f"{SCIENCE}:psi4/src/read_options.cc"], text=True)
    for declaration in ('options.add_int("DFT_RADIAL_POINTS", 75)',
                        'options.add_int("DFT_SPHERICAL_POINTS", 302)',
                        'options.add_str("DFT_PRUNING_SCHEME", "NONE"',
                        'options.add_str("DFT_RADIAL_SCHEME", "TREUTLER"'):
        assert declaration in defaults
    data["protein157"]["default_grid"] = dict(
        radial=75, angular=302, radial_scheme="TREUTLER", pruning="NONE", named_grid="",
        scientific_commit=SCIENCE, source="psi4/src/read_options.cc",
        note="The measured coarse75 arm already finishes on Psi4's default grid; "
             "99×590 was this benchmark's tighter reference, not Psi4's default.")
    target = docs / SOURCE
    target.mkdir(exist_ok=True)
    shutil.copyfile(root.parent / "artifacts.tar.gz", target / "artifacts.tar.gz")
    # Retain exact original receipts, results and raw samples alongside the archive.
    for item in root.iterdir():
        if item.is_dir():
            shutil.copytree(item, target / item.name, dirs_exist_ok=True)
        else:
            shutil.copyfile(item, target / item.name)
    shutil.copyfile(selection, target / "capacity-queries.json")
    data.setdefault("artifact_sha256", {})[SOURCE] = sha(target / "artifacts.tar.gz")
    for key in before:
        if key not in ("campaigns", "protein157", "artifact_sha256", "cpu_gpu_estimate_campaigns"):
            assert data[key] == before[key], f"Historical evidence changed: {key}"
    assert [c for c in data["campaigns"] if c["job_id"] != JOB] == [
        c for c in before["campaigns"] if c["job_id"] != JOB]
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    if render:
        publication.render_preserving_history(docs)
    print(json.dumps(dict(campaign=campaign, memory=memory), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--frozen", required=True, type=Path)
    parser.add_argument("--docs", required=True, type=Path)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.publish:
        publish(args.root, args.selection, args.frozen, args.docs, args.render)
    else:
        campaign, memory = verify(args.root, args.selection, args.frozen, args.docs)
        print(json.dumps(dict(campaign=campaign, memory=memory), indent=2))
