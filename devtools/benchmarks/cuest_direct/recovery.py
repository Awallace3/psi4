#!/usr/bin/env python3
"""One-allocation, bounded recovery/gate/benchmark sequence; no submissions."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gpu_compare"))
from compare import atomic_json, digest, engine_env
from containment import run_contained
from protocol import verify_compiled_sources
from cuest_sdk import verify_build_manifest


def probe_passed(path, mode):
    try:
        r = json.loads(Path(path).read_text())
        return (r.get("passed") is True and
                r["config"]["pyscf_eigensolver"] == mode and
                len(r["basis_cases"]) == 2 and
                {c["basis"] for c in r["basis_cases"]} == {"cc-pvtz", "aug-cc-pvtz"} and
                all(c.get("passed") is True for c in r["basis_cases"]))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def record_gate_completion(output, selected):
    """Gate-only acceptance is deliberately distinct from panel completion."""
    output = Path(output)
    preflight = json.loads((output / "preflight/COMPLETED.json").read_text())
    if preflight.get("complete") is not True or selected not in ("native", "cupy-canonical"):
        raise RuntimeError("Cannot accept incomplete GPU gates")
    atomic_json(output / "GATES_COMPLETE.json", dict(
        complete=True, eigensolver=selected, unmodified_gpu4pyscf=(selected == "native"),
        rks_grid_direct_preflight=True, benchmarks_run=False,
        bases=["cc-pvtz", "aug-cc-pvtz"]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("repo", "build", "output", "data", "gpu-env", "psi-env",
                "cuda12-lib", "psi-cuda-lib", "psi-xc-lib"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("commit", "build-job"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--gpu", choices=("H100", "H200"), default="H200")
    p.add_argument("--qos", choices=("embers", "inferno"), default="embers")
    p.add_argument("--gates-only", action="store_true",
                   help="Stop after eigensolver, RKS/grid and peptide direct/direct gates; no timing panel")
    a = p.parse_args()
    if not os.getenv("SLURM_JOB_ID"):
        p.error("Requires one allocated GPU job")
    for name, value in vars(a).items():
        if isinstance(value, Path):
            setattr(a, name, value.resolve())
    a.output.mkdir(exist_ok=False)
    deadline = time.monotonic() + 26000
    stages = []
    gpu_python = str(a.gpu_env / "bin/python")
    scripts = a.repo / "devtools/benchmarks/gpu_compare"
    a.threads = 8
    gpu_environment = engine_env("pyscf", a)
    # Fail closed BEFORE any CUDA device/canary subprocess. A self-consistent
    # old SDK manifest is not independent 0.2.50 admission.
    manifest = json.loads((a.build / "build-manifest.json").read_text())
    verify_build_manifest(manifest)
    device = json.loads(subprocess.check_output(
        [gpu_python, str(scripts / "compare.py"), "device"],
        env=gpu_environment, text=True, timeout=45))
    device["name"] = subprocess.check_output(
        ["nvidia-smi", "-i", device["uuid"], "--query-gpu=name",
         "--format=csv,noheader"], text=True, timeout=10).strip()
    if a.gpu not in device["name"]:
        raise RuntimeError(f"Wrong GPU hardware: {device}")
    atomic_json(a.output / "device.json", device)

    def idle(label):
        text = subprocess.check_output(
            ["nvidia-smi", "-i", device["uuid"],
             "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
             "--format=csv,noheader,nounits"], text=True, timeout=10)
        path = a.output / (label + ".gpu-processes.csv")
        with path.open("x") as stream:
            stream.write(text)
        if text.strip():
            raise RuntimeError("GPU cleanup/exclusivity unverified; further work withheld")

    idle("before")
    verify_compiled_sources(a.repo, manifest["commit"], a.commit)
    if manifest["job_id"] != a.build_job:
        raise RuntimeError("Wrong compiled build job")
    package = a.build / "build/stage/lib/psi4"
    core = next(package.glob("core*.so"))
    if (digest(core) != manifest["core_sha256"] or
            digest(manifest["cuest_library"]) != manifest["cuest_library_sha256"]):
        raise RuntimeError("Compiled core/SDK changed")
    atomic_json(a.output / "compiled-build-manifest.json", manifest)
    config = json.loads((a.repo / "devtools/benchmarks/gpu_compare/config-cc-pvtz.json").read_text())
    for name in ("auxbasis", "df_fitting_condition", "df_mode"):
        config.pop(name, None)
    config.update(jk_mode="conventional-direct", threads=8, memory_gib=64,
                  require_energy_agreement=True,
                  worker_timeout_s=1200)
    config_path = a.output / "config-probe.json"
    atomic_json(config_path, config)
    xyz = a.data / "geometries/peptide.xyz"
    atomic_json(a.output / "recovery-manifest.json", dict(
        commit=a.commit, compiled_commit=manifest["commit"], job_id=os.environ["SLURM_JOB_ID"],
        config_sha256=digest(config_path), peptide_sha256=digest(xyz),
        diagnostic_solver_selection=True, timings_include_imports=True, gpu_device=device,
        planned_systems=["peptide"] if a.gates_only else ["peptide", "protein83"],
        gates_only=a.gates_only, rks_only=True,
        account=os.getenv("SLURM_JOB_ACCOUNT"), qos=a.qos, gpu=a.gpu))

    def run(label, command, seconds):
        remaining = deadline - time.monotonic()
        if remaining < 120:
            raise RuntimeError("Insufficient remaining bounded job budget")
        cleanup = {}
        row = dict(label=label, command=command, bound_s=min(seconds, remaining-60))
        stages.append(row)
        atomic_json(a.output / "stages.json", stages)
        try:
            rc = run_contained(command, row["bound_s"], cleanup, grace=20, env=gpu_environment)
            if not cleanup.get("cleanup_verified"):
                rc = 125
            row["exit_code"] = rc
            return rc
        finally:
            row["cleanup"] = cleanup
            atomic_json(a.output / "stages.json", stages)
            idle("after-" + label)

    selected = None
    for mode in ("native", "cupy-canonical"):
        out = a.output / ("eigenprobe-" + mode)
        cmd = [gpu_python, str(scripts / "probe_eigensolver.py"), "--config", str(config_path),
               "--xyz", str(xyz), "--mode", mode, "--output", str(out)]
        rc = run("eigenprobe-" + mode, cmd, 180)
        if not stages[-1]["cleanup"].get("cleanup_verified"):
            raise RuntimeError("Unverified probe process cleanup; no further work")
        if rc == 0 and probe_passed(out / "result.json", mode):
            selected = mode
            break
    atomic_json(a.output / "solver-selection.json", dict(
        selected=selected, unmodified_gpu4pyscf=(selected == "native"),
        scientific_cutoff_unchanged=True,
        note="Alternative is explicit CuPy GPU-only canonical diagonalization, never a host fallback"))
    if selected is None:
        raise RuntimeError("Neither GPU eigensolver passed the actual-molecule CPU-reference gate")
    preflight = a.output / "preflight"
    cmd = ["bash", str(a.repo / "devtools/benchmarks/cuest_direct/compare_job.sh"),
           str(a.repo), a.commit, str(a.build), str(preflight), str(a.data),
           a.build_job, a.gpu, "preflight", a.qos, selected]
    if run("rks-grid-preflight", cmd, 3600):
        raise RuntimeError("RKS/grid/direct/direct preflight failed; production withheld")
    if not json.loads((preflight / "COMPLETED.json").read_text()).get("complete"):
        raise RuntimeError("Preflight completion artifact missing")
    if a.gates_only:
        record_gate_completion(a.output, selected)
        return 0
    config["pyscf_eigensolver"] = selected
    config_path = a.output / "config-production.json"
    atomic_json(config_path, config)
    cmd = [gpu_python, str(scripts / "tz_control.py"), "--config", str(config_path),
           "--jk-mode", "conventional-direct", "--psi-env", str(a.psi_env),
           "--psi-package", str(package), "--psi-cuda-lib", str(a.psi_cuda_lib),
           "--psi-xc-lib", str(a.psi_xc_lib), "--gpu-env", str(a.gpu_env),
           "--cuda12-lib", str(a.cuda12_lib), "--data", str(a.data),
           "--threads", "8", "--systems", "peptide", "protein83", "--repeats", "3",
           "--stop-on-worker-failure",
           "--output", str(a.output / "panel")]
    if run("five-rung-panel", cmd, 21600):
        raise RuntimeError("Production panel incomplete; inspect immutable per-case artifacts")
    panel = json.loads((a.output / "panel/COMPLETE.json").read_text())
    if panel.get("complete") is not True:
        raise RuntimeError("Panel failed acceptance gates")
    results = json.loads((a.output / "panel/control-results.json").read_text())
    if any(row.get("energy_review") is not False
           for case in results for row in case["basis_runs"].values()):
        raise RuntimeError("Panel needs numerical energy review; not accepted as complete")
    atomic_json(a.output / "COMPLETE.json", dict(
        complete=True, eigensolver=selected, unmodified_gpu4pyscf=(selected == "native"),
        systems=["peptide", "protein83"], functionals=list(config["functionals"]),
        bases=["cc-pvtz", "aug-cc-pvtz"], repeats=3))
    return 0


if __name__ == "__main__":
    def terminate(*_):
        raise KeyboardInterrupt("Scheduler termination")
    signal.signal(signal.SIGTERM, terminate)
    sys.exit(main())
