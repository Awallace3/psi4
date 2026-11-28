"""Unified CPU16 node-local NVMe ab initio SAPT0 & F-SAPT0 baseline suite.

Runs the matching CPU16 (16-thread, node-local NVMe) references against a
completed CPU-admitted Build A installation for fsapt1, peptide, nanotube,
and protein83 (canonical SAPT0+dHF and instrumented F-SAPT0), writing
incremental checkpoints to cpu16-result.json.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from phoenix_timing import clean, digest, save


def execute(args):
    source = args.science_source.resolve(strict=True)
    outer = Path(__file__).resolve().parents[2]
    build_root = args.build_root.resolve(strict=True)
    build_manifest = json.loads((build_root / "manifest.json").read_text())
    build_job = (build_root / "job-id.txt").read_text().strip()
    science_pin = build_manifest["source_pin"]
    helper_pin = subprocess.check_output(
        ["git", "-C", str(outer), "rev-parse", "HEAD"], text=True
    ).strip()
    clean(source, science_pin)
    clean(outer, helper_pin)
    subprocess.run(
        ["git", "-C", str(outer), "diff", "--exit-code", science_pin, helper_pin,
         "--", "psi4", "psi4_runtime_support"],
        check=True,
    )

    helper = source / "devtools/disp0/phoenix_validation"
    outer_helper = outer / "devtools/disp0/phoenix_validation"
    sys.path.insert(0, str(helper))
    from gpu_runtime import read_config, retirement_channel
    from run_gate import environment
    from run_full import execute_owned
    from validate_cpu_allocation import fields
    def _load_outer(name):
        spec = importlib.util.spec_from_file_location("_outer_" + name, outer_helper / (name + ".py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    require_cpu_runtime_admission = _load_outer("stage_scaling").require_cpu_runtime_admission
    step_scratch = _load_outer("sealed_h200").step_scratch

    if os.environ.get("SLURM_JOB_GPUS"):
        raise RuntimeError("CPU16 reference suite must not allocate a GPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.update(DISP0_SOURCE=str(source), DISP0_PIN=science_pin)

    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "STARTED",
        "scope": "CPU16_NODE_LOCAL_ABINITIO_BASELINE_SUITE",
        "threads": args.threads,
        "science_pin": science_pin,
        "helper_pin": helper_pin,
        "build_root": str(build_root),
        "build_job": build_job,
        "node": os.uname().nodename,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "stages": {},
        "processes": [],
    }
    checkpoint = lambda: save(root / "cpu16-result.json", report)
    if len(report["cpu_affinity"]) < args.threads:
        raise RuntimeError(f"Need {args.threads} CPUs, got {len(report['cpu_affinity'])}")

    groups, owner = set(), None
    checkpoint()
    try:
        build = build_root / ("job-" + build_job)
        base_path = build / "cpu-sealed-admission/runtime-config.json"
        base, proof = read_config(base_path, digest(base_path))
        if proof["pin"] != science_pin:
            raise RuntimeError("Build pin mismatch")
        report["cpu_runtime_admission"] = require_cpu_runtime_admission(base)

        aliases = root / "sealed-workers"
        aliases.mkdir(mode=0o700)
        config_path = root / "runtime-config.json"
        derived = {**base, "threads": args.threads, "sealed_alias_parent": str(aliases)}
        save(config_path, derived)
        config_sha = digest(config_path)
        config, _ = read_config(config_path, config_sha)
        require_cpu_runtime_admission(config)

        owner = step_scratch(root)
        scratch = Path(os.environ["TMPDIR"])
        report["scratch_mount"] = fields(subprocess.check_output(
            ["findmnt", "--first-only", "-T", str(scratch), "-n", "-P", "-o", "SOURCE,FSTYPE,TARGET"],
            text=True, timeout=30,
        ))
        disk = os.statvfs(scratch)
        report["scratch_free_bytes"] = disk.f_bavail * disk.f_frsize
        if report["scratch_free_bytes"] < 32 * 1024**3:
            raise RuntimeError("Node-local scratch headroom below 32 GiB")
        checkpoint()

        bootstrap = helper / "sealed_bootstrap.py"
        suite_worker = outer / "devtools/fsapt_integration/suite_fsapt_worker.py"
        timing_worker = outer / "devtools/fsapt_integration/timing_worker.py"
        inputs = args.inputs.resolve(strict=True)
        inputs_sha = args.inputs_sha256
        if digest(inputs) != inputs_sha:
            raise RuntimeError("Inputs SHA256 mismatch")
        sapt0_inputs = args.sapt0_inputs.resolve(strict=True)

        def psi4_wrapped(payload):
            return [
                "/bin/bash", "-c",
                'set -euo pipefail; export PYTHONPATH="${PYTHONPATH-}"; '
                'eval "$("$1" --psiapi)"; shift; exec "$@"',
                "cpu16-abinitio", config["stage"] + "/bin/psi4",
                config["dependency_environment"] + "/bin/python", "-B", *payload,
            ]

        def worker_env(label):
            work = scratch / label
            work.mkdir()
            alias = aliases / label
            alias.mkdir(mode=0o700)
            env = environment(config, gpu=False, sealed_alias_directory=alias)
            env.update(
                TMPDIR=str(work), SCRATCH=str(work), PSI_SCRATCH=str(work),
                OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
                OPENBLAS_NUM_THREADS=str(args.threads),
            )
            return env

        def run_child(label, payload, *, timeout=7200):
            row = {"label": label, "started_monotonic": time.monotonic()}
            report["processes"].append(row)
            checkpoint()
            env = worker_env(label)
            print("START", label, flush=True)
            with (root / (label + ".stdout")).open("x") as out, (
                root / (label + ".stderr")
            ).open("x") as err, retirement_channel(env) as channel:
                t0 = time.perf_counter()
                code = execute_owned(
                    psi4_wrapped(payload), env, out, err, timeout,
                    groups.add, reaped=groups.discard, pass_fds=channel.pass_fds,
                )
                row.update(
                    returncode=code,
                    wall_seconds=time.perf_counter() - t0,
                    retirement_reason=channel.reason(),
                )
            checkpoint()
            if code or row["retirement_reason"] is not None:
                raise RuntimeError(f"{label} failed (rc={code}, retirement={row['retirement_reason']})")
            print("DONE", label, f"({row['wall_seconds']:.1f}s)", flush=True)
            return row

        def run_stage(name, fn):
            box = {"status": "STARTED"}
            report["stages"][name] = box
            checkpoint()
            try:
                box.update(fn())
                box.setdefault("status", "PASS")
            except BaseException as err:
                box["status"] = "FAILED"
                box["error"] = f"{type(err).__name__}: {err}"
                print(f"STAGE_FAILED {name}: {box['error']}", flush=True)
            finally:
                checkpoint()

        def stage_fsapt1():
            out_dir = root / "fsapt1-cpu16"
            run_child("fsapt1-cpu16", [
                str(suite_worker), "--bootstrap", str(bootstrap),
                "--config", str(config_path), "--config-sha256", config_sha,
                "--mode", "fsapt1", "--arm", "cpu", "--output", str(out_dir),
                "--block-size", "24", "--factor-cache-bytes", "0",
                "--scf-convergence", "10", "--scf-density-convergence", "12",
            ], timeout=2400)
            doc = json.loads((out_dir / "result.json").read_text())
            return {
                "whole_seconds": doc["benchmark_whole_energy_seconds"],
                "step_timers": doc["step_timers"],
                "energy_Eh": doc["energy_Eh"],
            }

        def stage_sapt0(system):
            out_dir = root / f"sapt0-{system}-cpu16"
            run_child(f"sapt0-{system}-cpu16", [
                str(bootstrap), "--sapt0",
                "--config", str(config_path), "--config-sha256", config_sha,
                "--case", str(sapt0_inputs / f"{system}.json"),
                "--output", str(out_dir), "--arm", "cpu",
            ], timeout=3600)
            doc = json.loads((out_dir / "result.json").read_text())
            secs = doc.get("seconds") or {}
            return {
                "whole_seconds": secs.get("total"),
                "seconds": secs,
                "energies": doc.get("energies"),
            }

        def stage_fsapt_inst(system, dconv):
            out_dir = root / f"fsapt-{system}-cpu16-inst"
            receipt = root / f"fsapt-{system}-cpu16-timers.json"
            run_child(f"fsapt-{system}-cpu16-inst", [
                str(timing_worker), "--bootstrap", str(bootstrap),
                "--receipt", str(receipt),
                "--config", str(config_path), "--config-sha256", config_sha,
                "--arm", "cpu", "--scaled", "--benchmark", "--output", str(out_dir),
                "--inputs", str(inputs), "--inputs-sha256", inputs_sha,
                "--system", system, "--block-size", "24", "--factor-cache-bytes", "0",
                "--scf-convergence", "10", "--scf-density-convergence", str(dconv),
            ], timeout=7200)
            doc = json.loads((out_dir / "result.json").read_text())
            rec = json.loads(receipt.read_text())
            return {
                "whole_seconds": doc["benchmark_whole_energy_seconds"],
                "step_seconds": rec["seconds"],
                "energy_Eh": doc["energy_Eh"],
            }

        run_stage("fsapt1_fsapt", stage_fsapt1)
        run_stage("peptide_sapt0", lambda: stage_sapt0("peptide"))
        run_stage("peptide_fsapt", lambda: stage_fsapt_inst("peptide", 12))
        run_stage("nanotube_sapt0", lambda: stage_sapt0("nanotube"))
        run_stage("nanotube_fsapt", lambda: stage_fsapt_inst("nanotube", 10))
        run_stage("protein83_sapt0", lambda: stage_sapt0("protein83"))

        all_ok = all(v.get("status") == "PASS" for v in report["stages"].values())
        report["status"] = "PASS_CPU16_SUITE" if all_ok else "PARTIAL_CPU16_SUITE"
    except BaseException as err:
        report["status"] = "FAILED"
        report["error"] = f"{type(err).__name__}: {err}"
        raise
    finally:
        if owner is not None:
            report["scratch_cleanup"] = owner.cleanup(groups)
        checkpoint()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--science-source", type=Path, required=True)
    p.add_argument("--build-root", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--inputs-sha256", required=True)
    p.add_argument("--sapt0-inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, default=16)
    execute(p.parse_args())
