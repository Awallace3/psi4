"""Single-allocation H200 embers suite: water gate, fsapt1, timing_test_suite F-SAPT & SAPT0.

Runs all timing-suite benchmarks (peptide, nanotube, protein83, bounded protein157)
plus the F-SAPT-specific functional-group partition test (tests/fsapt1 phenol dimer)
inside ONE 8-hour H200 embers allocation with incremental checkpointing and per-stage
error isolation after the required small H200 gate.
"""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from phoenix_timing import clean, digest, ratios, save, summary


HARTR_TO_KCAL = 627.5094740631
RESERVE_SECONDS = 600


def compare_fsapt_tables(cpu_analysis, gpu_analysis, *, atol_eh=2e-7, rtol=2e-5):
    """Compare every functional-group interaction cell (in kcal/mol from fsapt.py)."""
    if set(cpu_analysis) != set(gpu_analysis):
        raise ValueError("Mismatched F-SAPT link modes")
    atol_kcal = atol_eh * HARTR_TO_KCAL
    max_abs_kcal = 0.0
    max_cell = None
    cells_checked = 0
    for mode in sorted(cpu_analysis):
        if set(cpu_analysis[mode]) != set(gpu_analysis[mode]):
            raise ValueError("Mismatched F-SAPT analysis types in " + mode)
        for kind in sorted(cpu_analysis[mode]):
            left_terms = cpu_analysis[mode][kind]
            right_terms = gpu_analysis[mode][kind]
            if set(left_terms) != set(right_terms):
                raise ValueError(f"Mismatched SAPT terms in {mode}/{kind}")
            for term in sorted(left_terms):
                if set(left_terms[term]) != set(right_terms[term]):
                    raise ValueError(f"Mismatched FragA keys in {mode}/{kind}/{term}")
                for fa in sorted(left_terms[term]):
                    if set(left_terms[term][fa]) != set(right_terms[term][fa]):
                        raise ValueError(f"Mismatched FragB keys in {mode}/{kind}/{term}/{fa}")
                    for fb in sorted(left_terms[term][fa]):
                        a = float(left_terms[term][fa][fb])
                        b = float(right_terms[term][fa][fb])
                        if not (math.isfinite(a) and math.isfinite(b)):
                            raise ValueError(f"Nonfinite F-SAPT cell {mode}/{kind}/{term}/{fa}/{fb}")
                        err = abs(b - a)
                        if err > atol_kcal + rtol * abs(a):
                            raise AssertionError(
                                f"F-SAPT group mismatch {mode}/{kind}/{term}/{fa}-{fb}: "
                                f"cpu={a:.10g} kcal/mol, gpu={b:.10g} kcal/mol, diff={b-a:.3e}"
                            )
                        cells_checked += 1
                        if err >= max_abs_kcal:
                            max_abs_kcal = err
                            max_cell = f"{mode}/{kind}/{term}/{fa}-{fb}"
    return {
        "cells_checked": cells_checked,
        "max_abs_diff_kcal_mol": max_abs_kcal,
        "max_abs_diff_Eh": max_abs_kcal / HARTR_TO_KCAL,
        "worst_cell": max_cell,
    }


def summarize_stage_status(stages):
    return {name: stage.get("status", "UNKNOWN") for name, stage in stages.items()}


def execute(manifest_path, root):
    manifest = json.loads(manifest_path.read_text())
    if digest(manifest_path) != os.environ["FSAPT_MANIFEST_SHA256"]:
        raise RuntimeError("Submission manifest changed")
    source = Path(manifest["science_source"])
    outer = Path(manifest["helper_source"])
    clean(source, manifest["science_pin"])
    clean(outer, manifest["helper_pin"])
    subprocess.run(
        ["git", "-C", str(outer), "diff", "--exit-code",
         manifest["science_pin"], manifest["helper_pin"], "--",
         "psi4", "psi4_runtime_support"],
        check=True,
    )
    helper = outer / "devtools/disp0/phoenix_validation"
    sys.path.insert(0, str(helper))
    from gpu_runtime import read_config, retirement_channel
    from stage_scaling import require_cpu_runtime_admission
    from run_gate import environment, compare
    from run_full import execute_owned, compare_energy
    from repeat_gauge import compare_repeat
    from full_inputs import load_point
    from prepare_inputs import input_basis_profile
    from sapt0_hardware import PROFILES
    from sealed_h200 import step_scratch
    from split_jobs import validate_allocation
    from validate_cpu_allocation import fields, memory_bytes, seconds

    device_family = manifest.get("device_family", "H200")
    if device_family not in PROFILES:
        raise RuntimeError("Unsupported device family: " + str(device_family))
    profile = PROFILES[device_family]

    os.environ.update(DISP0_SOURCE=str(source), DISP0_PIN=manifest["science_pin"])
    report = {
        "status": "STARTED",
        "manifest_sha256": digest(manifest_path),
        "manifest": manifest,
        "device_family": device_family,
        "stages": {},
        "processes": [],
        "scope": f"SINGLE_JOB_{device_family}_{manifest['qos'].upper()}_FSAPT_AND_SAPT0_TIMING_SUITE",
    }
    checkpoint = lambda: save(root / "suite-result.json", report)
    groups, owner = set(), None
    checkpoint()
    try:
        raw = subprocess.check_output(
            ["/opt/slurm/current/bin/scontrol", "show", "job", "-o", os.environ["SLURM_JOB_ID"]],
            text=True, timeout=30,
        )
        job_fields = fields(raw)
        if device_family == "H200":
            validate_allocation(job_fields, manifest)
        else:
            partitions = set(job_fields.get("Partition", "").split(","))
            allocated = job_fields.get("AllocTRES", "").split(",")
            gres_token = "gres/" + profile["gres"].rsplit(":", 1)[0] + "=1"
            if (
                job_fields.get("Account") != manifest["account"]
                or job_fields.get("QOS") != manifest["qos"]
                or profile["partition"] not in partitions
                or not (partitions <= {profile["partition"], "gpu-v100"})
                or job_fields.get("NumNodes") != "1"
                or job_fields.get("NumCPUs") != "8"
                or job_fields.get("CPUs/Task") != "8"
                or job_fields.get("NumTasks") != "1"
                or seconds(job_fields["TimeLimit"]) != manifest.get("allocation_wall_seconds", 28800)
                or memory_bytes(job_fields["MinMemoryNode"]) != 160 * 1024**3
                or gres_token not in allocated
            ):
                raise RuntimeError(f"{device_family} allocation differs from manifest: {raw}")
        report["scheduler_job"] = raw
        report["node"] = os.uname().nodename
        report["cpu_affinity"] = sorted(os.sched_getaffinity(0))
        if len(report["cpu_affinity"]) < 8:
            raise RuntimeError("Eight allocated CPUs required")
        report["gpu_hardware"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"],
            text=True, timeout=30,
        )
        if profile["name"] not in report["gpu_hardware"]:
            raise RuntimeError(f"{profile['name']} required; got {report['gpu_hardware']!r}")

        if digest(Path(manifest["build_root"]) / "manifest.json") != manifest["build_manifest_sha256"]:
            raise RuntimeError("Prerequisite build manifest changed")
        build = Path(manifest["build_root"]) / ("job-" + manifest["build_job"])
        config_path = root / "runtime-config.json"
        base_path = build / "cpu-sealed-admission/runtime-config.json"
        base, proof = read_config(base_path, digest(base_path))
        if proof["pin"] != manifest["science_pin"] or base["threads"] != 8:
            raise RuntimeError("Wrong build pin or thread policy")
        report["cpu_runtime_admission"] = require_cpu_runtime_admission(base)
        aliases = root / "sealed-workers"
        aliases.mkdir(mode=0o700)
        derived = {**base, "sealed_alias_parent": str(aliases)}
        if device_family != "H200":
            derived["device_family"] = device_family
            derived["device_memory_limit_bytes"] = profile["budget_gib"] * 1024**3
            derived["native_persistent_bytes"] = min(base["native_persistent_bytes"], 8 * 1024**3)
        save(config_path, derived)
        config_sha = digest(config_path)
        config, _ = read_config(config_path, config_sha)
        if device_family == "H200":
            require_cpu_runtime_admission(config)
        else:
            ignored = {"sealed_alias_parent", "device_family", "device_memory_limit_bytes", "native_persistent_bytes"}
            if {k: v for k, v in config.items() if k not in ignored} != {
                k: v for k, v in base.items() if k not in ignored
            }:
                raise RuntimeError("Derived GPU runtime closure diverged from CPU-admitted base")

        inputs = Path(manifest["inputs"])
        if input_basis_profile(json.loads(inputs.read_text())) != "jun-cc-pvdz":
            raise RuntimeError("Qualified jun-cc-pVDZ inputs required")
        dims_by_system = {
            name: load_point(inputs, manifest["inputs_sha256"], name)[1]
            for name in ("peptide", "nanotube", "protein83")
        }
        sapt0_inputs = Path(manifest["sapt0_inputs"])

        owner = step_scratch(root)
        scratch = Path(os.environ["TMPDIR"])
        report["scratch_mount"] = fields(subprocess.check_output(
            ["findmnt", "-T", str(scratch), "-n", "-P", "-o", "SOURCE,FSTYPE,TARGET"],
            text=True, timeout=30,
        ))
        disk = os.statvfs(scratch)
        report["scratch_free_bytes"] = disk.f_bavail * disk.f_frsize
        if report["scratch_free_bytes"] < 32 * 1024**3:
            raise RuntimeError("Scratch headroom below 32 GiB")

        deadline = time.monotonic() + manifest["allocation_wall_seconds"] - RESERVE_SECONDS
        bootstrap = helper / "sealed_bootstrap.py"
        suite_worker = outer / "devtools/fsapt_integration/suite_fsapt_worker.py"
        timing_worker = outer / "devtools/fsapt_integration/timing_worker.py"

        def remaining_seconds():
            return deadline - time.monotonic()

        def run_child(label, argv, env, *, timeout=3600):
            row = {"label": label, "argv": argv, "started_monotonic": time.monotonic()}
            report["processes"].append(row)
            checkpoint()
            rem = remaining_seconds()
            if rem < 60:
                raise TimeoutError("Allocation deadline exhausted before " + label)
            print("START", label, f"(remaining={rem:.0f}s)", flush=True)
            with (root / (label + ".stdout")).open("x") as out, (
                root / (label + ".stderr")
            ).open("x") as err, retirement_channel(env) as channel:
                t0 = time.perf_counter()
                code = execute_owned(
                    argv, env, out, err, min(timeout, rem),
                    groups.add, reaped=groups.discard, pass_fds=channel.pass_fds,
                )
                row.update(
                    returncode=code,
                    wall_seconds=time.perf_counter() - t0,
                    retirement_reason=channel.reason(),
                )
            checkpoint()
            if code or row["retirement_reason"] is not None:
                raise RuntimeError(
                    f"Child {label} failed (rc={code}, retirement={row['retirement_reason']})"
                )
            print("DONE", label, f"({row['wall_seconds']:.1f}s)", flush=True)
            return row

        def psi4_wrapped(payload):
            return [
                "/bin/bash", "-c",
                'set -euo pipefail; export PYTHONPATH="${PYTHONPATH-}"; '
                'eval "$("$1" --psiapi)"; shift; exec "$@"',
                "fsapt-suite", config["stage"] + "/bin/psi4",
                config["dependency_environment"] + "/bin/python", "-B", *payload,
            ]

        def worker_env(label, arm):
            work = scratch / label
            work.mkdir()
            alias = aliases / label
            alias.mkdir(mode=0o700)
            env = environment(config, gpu=(arm == "gpu"), sealed_alias_directory=alias)
            env.update(TMPDIR=str(work), SCRATCH=str(work), PSI_SCRATCH=str(work))
            return env, work

        def run_fragment_worker(label, system, arm, *, instrumented=False, oracle=None, timeout=3600):
            output = root / label
            env, _ = worker_env(label, arm)
            entry = (
                [str(timing_worker), "--bootstrap", str(bootstrap),
                 "--receipt", str(root / (label + "-timers.json"))]
                if instrumented
                else [str(bootstrap)]
            )
            # Nanotube jun-cc-pVDZ has min_eig(S)=2.47e-8 (cond(S)~1e9), giving
            # a ~3e-12 FP64 DIIS Dnorm floor on both CPU MemDFJK and GPU cuESTJK;
            # match sapt0_timing.py's E10/D10 for nanotube while keeping E10/D12
            # for peptide, protein83 (frozen CPU48 parity), and fsapt1.
            scf_dconv = "10" if system == "nanotube" else "12"
            payload = [
                *entry,
                "--config", str(config_path), "--config-sha256", config_sha,
                "--arm", arm, "--scaled", "--benchmark", "--output", str(output),
                "--inputs", str(inputs), "--inputs-sha256", manifest["inputs_sha256"],
                "--system", system, "--block-size", "24", "--factor-cache-bytes", "0",
                "--scf-convergence", "10", "--scf-density-convergence", scf_dconv,
            ]
            if oracle is not None:
                payload += ["--oracle-from", str(oracle)]
            run_child(label, psi4_wrapped(payload), env, timeout=timeout)
            doc = json.loads((output / "result.json").read_text())
            if doc["status"] != "EXECUTED_REQUIRES_INDEPENDENT_COMPARISON":
                raise RuntimeError("Worker status invalid: " + label)
            seconds = None
            if instrumented:
                rec = json.loads((root / (label + "-timers.json")).read_text())
                if rec["status"] != "PASS_OBSERVED_TIMERS":
                    raise RuntimeError("Incomplete timer receipt: " + label)
                seconds = rec["seconds"]
            return output, doc, seconds

        def run_sapt0_case(label, system, *, repeats=3, diagnostics=True, timeout=3600):
            out_dir = root / label
            cmd = [
                sys.executable, "-B", str(helper / "sapt0_timing.py"), "run",
                "--config", str(config_path), "--config-sha256", config_sha,
                "--case", str(sapt0_inputs / (system + ".json")),
                "--output", str(out_dir), "--repeats", str(repeats),
                "--timeout", str(timeout),
            ]
            if diagnostics:
                cmd.append("--diagnostics")
            run_child(label, cmd, dict(os.environ), timeout=timeout * (2 * repeats + 2))
            return json.loads((out_dir / "comparison.json").read_text())

        # =====================================================================
        # STAGE 1: Required small H200 10-process F-SAPT water gate (hard gate)
        # =====================================================================
        gate_dir = root / "stage1-small-gate"
        gate_dir.mkdir()
        run_child(
            "stage1-small-gate",
            [sys.executable, "-B", str(helper / "run_gate.py"),
             "--config", str(config_path), "--config-sha256", config_sha,
             "--output", str(gate_dir)],
            dict(os.environ),
            timeout=1800,
        )
        gate_doc = json.loads((gate_dir / "gate-result.json").read_text())
        if gate_doc["status"] != "PASS_SMALL_H200_ONLY" or not gate_doc["gpu_small_case_correctness_verified"]:
            raise RuntimeError("Small H200 F-SAPT water gate failed")
        report["stages"]["stage1_small_gate"] = {
            "status": "PASS",
            "gate_result_sha256": digest(gate_dir / "gate-result.json"),
        }
        checkpoint()

        # Helper to isolate stages 2..7 so a later failure never discards earlier passes.
        skip_stages = set(manifest.get("skip_stages", ()))

        def run_stage(name, fn):
            if name in skip_stages:
                report["stages"][name] = {"status": "SKIPPED_BY_MANIFEST"}
                print(f"STAGE_SKIPPED {name}", flush=True)
                checkpoint()
                return
            stage_box = {"status": "STARTED"}
            report["stages"][name] = stage_box
            checkpoint()
            try:
                out = fn()
                stage_box.update(out)
                if "status" not in out:
                    stage_box["status"] = "PASS"
            except BaseException as err:
                stage_box["status"] = "FAILED"
                stage_box["error"] = f"{type(err).__name__}: {err}"
                print(f"STAGE_FAILED {name}: {type(err).__name__}: {err}", flush=True)
            finally:
                checkpoint()

        # =====================================================================
        # STAGE 2: F-SAPT-specific test: tests/fsapt1 S22 phenol dimer (OH/PH)
        # =====================================================================
        def stage2_fsapt1():
            def call_fsapt1(label, arm, *, oracle=None):
                out_dir = root / label
                env, _ = worker_env(label, arm)
                payload = [
                    str(suite_worker), "--bootstrap", str(bootstrap),
                    "--config", str(config_path), "--config-sha256", config_sha,
                    "--mode", "fsapt1", "--arm", arm, "--output", str(out_dir),
                    "--block-size", "24", "--factor-cache-bytes", "0",
                    "--scf-convergence", "10", "--scf-density-convergence", "12",
                ]
                if oracle is not None:
                    payload += ["--oracle-from", str(oracle)]
                run_child(label, psi4_wrapped(payload), env, timeout=2400)
                return out_dir, json.loads((out_dir / "result.json").read_text())

            cpu_dir, cpu_doc = call_fsapt1("stage2-fsapt1-cpu", "cpu")
            gpu0_dir, gpu0_doc = call_fsapt1("stage2-fsapt1-gpu-0", "gpu")
            gpu1_dir, gpu1_doc = call_fsapt1("stage2-fsapt1-gpu-1", "gpu")
            _, oracle_doc = call_fsapt1("stage2-fsapt1-oracle", "cpu", oracle=gpu0_dir)

            cpu_gpu_tables = compare_fsapt_tables(
                cpu_doc["fsapt_analysis_kcal_mol"], gpu0_doc["fsapt_analysis_kcal_mol"]
            )
            oracle_gpu_tables = compare_fsapt_tables(
                oracle_doc["fsapt_analysis_kcal_mol"], gpu0_doc["fsapt_analysis_kcal_mol"]
            )
            repeat_gpu_tables = compare_fsapt_tables(
                gpu0_doc["fsapt_analysis_kcal_mol"], gpu1_doc["fsapt_analysis_kcal_mol"],
                atol_eh=1e-9, rtol=1e-9,
            )
            gpu_med = statistics.median([
                gpu0_doc["benchmark_whole_energy_seconds"],
                gpu1_doc["benchmark_whole_energy_seconds"],
            ])
            cpu_t = cpu_doc["benchmark_whole_energy_seconds"]
            disp_cpu = cpu_doc["step_timers"].get("fragment_disp")
            disp_gpu = gpu0_doc.get("fragment_execution", {}).get("dispersion", {}).get(
                "blocked_execution_seconds"
            )
            return {
                "status": "PASS",
                "system": "phenol_dimer_fsapt1",
                "cpu_seconds": cpu_t,
                "gpu_seconds": [
                    gpu0_doc["benchmark_whole_energy_seconds"],
                    gpu1_doc["benchmark_whole_energy_seconds"],
                ],
                "whole_speedup": cpu_t / gpu_med,
                "cpu_fdisp0_seconds": disp_cpu,
                "gpu_fdisp0_blocked_seconds": disp_gpu,
                "energy_diff_Eh": gpu0_doc["energy_Eh"] - cpu_doc["energy_Eh"],
                "cpu_vs_gpu_fsapt_tables": cpu_gpu_tables,
                "oracle_vs_gpu_fsapt_tables": oracle_gpu_tables,
                "gpu_repeat_fsapt_tables": repeat_gpu_tables,
                "same_gauge_max_matrix_err_Eh": max(
                    v["maximum_absolute_difference"] for v in oracle_doc["same_gauge_errors"].values()
                ),
            }

        run_stage("stage2_fsapt1_phenol_dimer", stage2_fsapt1)

        # =====================================================================
        # STAGE 3a: Peptide canonical SAPT0 (water gate + 3x CPU8 vs 3x H200)
        # =====================================================================
        def stage3a_peptide_sapt0():
            water_cmp = run_sapt0_case("stage3a-sapt0-water", "water", repeats=1, diagnostics=False, timeout=600)
            pep_cmp = run_sapt0_case("stage3a-sapt0-peptide", "peptide", repeats=3, diagnostics=True, timeout=1800)
            return {
                "status": "PASS",
                "water_status": water_cmp["status"],
                "peptide_status": pep_cmp["status"],
                "signed_max_error_Eh": pep_cmp["signed_max_error_Eh"],
                "timing": pep_cmp["timing"],
                "diagnostics": pep_cmp.get("diagnostics"),
            }

        run_stage("stage3a_peptide_sapt0", stage3a_peptide_sapt0)

        # =====================================================================
        # STAGE 3b: Peptide F-SAPT (exact protocol of phoenix_timing.py: 3+3)
        # =====================================================================
        def stage3b_peptide_fsapt():
            dims = dims_by_system["peptide"]
            samples = {m: {a: [] for a in ("cpu", "gpu")} for m in ("baseline", "instrumented")}
            checks, refs = [], {}
            for mode in ("baseline", "instrumented"):
                for rep in range(3):
                    pair = {}
                    for arm in (("cpu", "gpu") if rep % 2 == 0 else ("gpu", "cpu")):
                        path, doc, secs = run_fragment_worker(
                            f"stage3b-peptide-{mode}-{arm}-{rep}",
                            "peptide", arm, instrumented=(mode == "instrumented"), timeout=1800,
                        )
                        if arm in refs:
                            checks.append(compare_repeat(refs[arm], path, dimensions=dims, arm=arm))
                        else:
                            refs[arm] = path
                        pair[arm] = doc
                        samples[mode][arm].append(
                            secs if secs is not None else {"total": doc["benchmark_whole_energy_seconds"]}
                        )
                    checks.append({"energy": compare_energy(pair["cpu"], pair["gpu"]),
                                   "signed_terms": compare(pair["cpu"], pair["gpu"])})
            _, oracle_doc, _ = run_fragment_worker(
                "stage3b-peptide-oracle", "peptide", "cpu", oracle=refs["gpu"], timeout=1800,
            )
            return {
                "status": "PASS",
                "baseline": ratios(samples["baseline"]["cpu"], samples["baseline"]["gpu"]),
                "instrumented_steps": ratios(samples["instrumented"]["cpu"], samples["instrumented"]["gpu"]),
                "same_gauge_max_matrix_err_Eh": max(
                    v["maximum_absolute_difference"] for v in oracle_doc["same_gauge_errors"].values()
                ),
                "numerical_checks_count": len(checks),
            }

        run_stage("stage3b_peptide_fsapt", stage3b_peptide_fsapt)

        # =====================================================================
        # STAGE 4a: Nanotube canonical SAPT0 (3x CPU8 vs 3x H200 + warm/prof)
        # =====================================================================
        def stage4a_nanotube_sapt0():
            cmp_doc = run_sapt0_case("stage4a-sapt0-nanotube", "nanotube", repeats=3, diagnostics=True, timeout=2400)
            return {
                "status": "PASS",
                "signed_max_error_Eh": cmp_doc["signed_max_error_Eh"],
                "timing": cmp_doc["timing"],
                "diagnostics": cmp_doc.get("diagnostics"),
            }

        run_stage("stage4a_nanotube_sapt0", stage4a_nanotube_sapt0)

        # =====================================================================
        # STAGE 4b: Nanotube F-SAPT H200 (2x baseline + 1x instrumented)
        # =====================================================================
        nanotube_gpu_ref = {}

        def stage4b_nanotube_fsapt_gpu():
            dims = dims_by_system["nanotube"]
            p0, d0, _ = run_fragment_worker("stage4b-nanotube-gpu-0", "nanotube", "gpu", timeout=2400)
            p1, d1, _ = run_fragment_worker("stage4b-nanotube-gpu-1", "nanotube", "gpu", timeout=2400)
            pt, dt, st = run_fragment_worker(
                "stage4b-nanotube-gpu-inst", "nanotube", "gpu", instrumented=True, timeout=2400,
            )
            nanotube_gpu_ref["path"] = p0
            nanotube_gpu_ref["doc"] = d0
            nanotube_gpu_ref["inst_seconds"] = st
            repeat_check = {"status": "PASS"}
            try:
                repeat_check["detail"] = compare_repeat(p0, p1, dimensions=dims, arm="gpu")
            except BaseException as err:
                repeat_check = {"status": "REPEAT_DRIFT", "error": f"{type(err).__name__}: {err}"}
            return {
                "status": "PASS",
                "gpu_baseline_seconds": [d0["benchmark_whole_energy_seconds"], d1["benchmark_whole_energy_seconds"]],
                "gpu_baseline_median_seconds": statistics.median([
                    d0["benchmark_whole_energy_seconds"], d1["benchmark_whole_energy_seconds"],
                ]),
                "gpu_instrumented_seconds": st,
                "gpu_repeat_check": repeat_check,
                "gpu_dispersion_execution": d0.get("fragment_execution", {}).get("dispersion"),
            }

        run_stage("stage4b_nanotube_fsapt_gpu", stage4b_nanotube_fsapt_gpu)

        # =====================================================================
        # STAGE 5a: Protein83 fixed-HF fdisp0 (jun-cc-pVDZ vs CPU48 + def2-SVP)
        # =====================================================================
        def stage5a_protein83_fixed_fdisp0():
            results = {}
            for key, spec in manifest["fixed_fdisp0_snapshots"].items():
                label = f"stage5a-fixed-fdisp0-{key}"
                out_dir = root / label
                env, _ = worker_env(label, "gpu")
                payload = [
                    str(suite_worker), "--bootstrap", str(bootstrap),
                    "--config", str(config_path), "--config-sha256", config_sha,
                    "--mode", "fixed-fdisp0", "--output", str(out_dir),
                    "--reference-result", spec["result"],
                    "--reference-result-sha256", spec["result_sha256"],
                    "--inputs", spec["inputs"],
                    "--inputs-sha256", spec["inputs_sha256"],
                    "--system", "protein83",
                    "--block-size", "24", "--cache-caps", "0,4294967296,0",
                ]
                run_child(label, psi4_wrapped(payload), env, timeout=1200)
                results[key] = json.loads((out_dir / "result.json").read_text())
            return {"status": "PASS", "snapshots": results}

        run_stage("stage5a_protein83_fixed_fdisp0", stage5a_protein83_fixed_fdisp0)

        # =====================================================================
        # STAGE 5b: Protein83 canonical SAPT0 (1x CPU8 + 1x H200 + warm/prof)
        # =====================================================================
        def stage5b_protein83_sapt0():
            cmp_doc = run_sapt0_case("stage5b-sapt0-protein83", "protein83", repeats=1, diagnostics=True, timeout=3600)
            return {
                "status": "PASS",
                "signed_max_error_Eh": cmp_doc["signed_max_error_Eh"],
                "timing": cmp_doc["timing"],
                "diagnostics": cmp_doc.get("diagnostics"),
            }

        run_stage("stage5b_protein83_sapt0", stage5b_protein83_sapt0)

        # =====================================================================
        # STAGE 5c: Protein83 full H200 F-SAPT (compared vs frozen CPU48 refs)
        # =====================================================================
        def stage5c_protein83_fsapt_gpu():
            dims = dims_by_system["protein83"]
            p0, d0, _ = run_fragment_worker("stage5c-protein83-gpu-0", "protein83", "gpu", timeout=3600)
            p1, d1, _ = run_fragment_worker("stage5c-protein83-gpu-1", "protein83", "gpu", timeout=3600)
            _, _, st = run_fragment_worker(
                "stage5c-protein83-gpu-inst", "protein83", "gpu", instrumented=True, timeout=3600,
            )
            repeat_check = {"status": "PASS"}
            try:
                repeat_check["detail"] = compare_repeat(p0, p1, dimensions=dims, arm="gpu")
            except BaseException as err:
                repeat_check = {"status": "REPEAT_DRIFT", "error": f"{type(err).__name__}: {err}"}
            cpu48_refs = []
            cpu48_checks = []
            for spec in manifest["protein83_cpu48_references"]:
                ref_path = Path(spec["result"])
                if digest(ref_path) != spec["result_sha256"]:
                    raise RuntimeError("Frozen Protein83 CPU48 reference changed")
                ref_doc = json.loads(ref_path.read_text())
                cpu48_refs.append(ref_doc["benchmark_whole_energy_seconds"])
                cpu48_checks.append({
                    "energy": compare_energy(ref_doc, d0),
                    "signed_terms": compare(ref_doc, d0),
                })
            gpu_med = statistics.median([
                d0["benchmark_whole_energy_seconds"], d1["benchmark_whole_energy_seconds"],
            ])
            cpu48_med = statistics.median(cpu48_refs)
            return {
                "status": "PASS",
                "gpu_baseline_seconds": [
                    d0["benchmark_whole_energy_seconds"], d1["benchmark_whole_energy_seconds"],
                ],
                "gpu_baseline_median_seconds": gpu_med,
                "gpu_instrumented_seconds": st,
                "frozen_cpu48_seconds": cpu48_refs,
                "frozen_cpu48_median_seconds": cpu48_med,
                "cpu48_over_h200_whole_speedup": cpu48_med / gpu_med,
                "cpu48_comparison_scope": "CPU48 cpu-gnr frozen refs (15474909/15474910) vs H200+host8",
                "cpu48_numerical_checks": cpu48_checks,
                "gpu_repeat_check": repeat_check,
                "gpu_dispersion_execution": d0.get("fragment_execution", {}).get("dispersion"),
            }

        run_stage("stage5c_protein83_fsapt_gpu", stage5c_protein83_fsapt_gpu)

        # =====================================================================
        # STAGE 6: Protein157 GPU-only bounded probe (canonical SAPT0 + F-SAPT)
        # =====================================================================
        def stage6_protein157_gpu():
            out = {}
            if remaining_seconds() < 1800:
                return {"status": "SKIPPED_LOW_WALLTIME", "remaining_seconds": remaining_seconds()}
            # 6a: Canonical SAPT0 GPU-only worker
            p157_sapt0_dir = root / "stage6a-sapt0-protein157-gpu"
            env, _ = worker_env("stage6a-sapt0-protein157-gpu", "gpu")
            payload = [
                str(bootstrap), "--sapt0",
                "--config", str(config_path), "--config-sha256", config_sha,
                "--case", str(sapt0_inputs / "protein157.json"),
                "--output", str(p157_sapt0_dir), "--arm", "gpu",
            ]
            try:
                run_child("stage6a-sapt0-protein157-gpu", psi4_wrapped(payload), env, timeout=2700)
                out["sapt0_gpu"] = json.loads((p157_sapt0_dir / "result.json").read_text())
            except BaseException as err:
                out["sapt0_gpu"] = {"status": "FAILED", "error": f"{type(err).__name__}: {err}"}

            # 6b: F-SAPT GPU-only probe if >= 45 min remain
            if remaining_seconds() >= 2700:
                p157_fsapt_dir = root / "stage6b-fsapt-protein157-gpu"
                env2, _ = worker_env("stage6b-fsapt-protein157-gpu", "gpu")
                case_path = sapt0_inputs / "protein157.json"
                payload2 = [
                    str(suite_worker), "--bootstrap", str(bootstrap),
                    "--config", str(config_path), "--config-sha256", config_sha,
                    "--mode", "custom-fsapt", "--output", str(p157_fsapt_dir),
                    "--case", str(case_path), "--case-sha256", digest(case_path),
                    "--block-size", "24", "--factor-cache-bytes", "0",
                    "--scf-convergence", "10", "--scf-density-convergence", "12",
                ]
                try:
                    run_child("stage6b-fsapt-protein157-gpu", psi4_wrapped(payload2), env2, timeout=3600)
                    out["fsapt_gpu"] = json.loads((p157_fsapt_dir / "result.json").read_text())
                except BaseException as err:
                    out["fsapt_gpu"] = {"status": "FAILED", "error": f"{type(err).__name__}: {err}"}
            else:
                out["fsapt_gpu"] = {"status": "SKIPPED_LOW_WALLTIME", "remaining_seconds": remaining_seconds()}
            ok = any(
                v.get("status") in ("PASS", "PASS_CUSTOM_FSAPT_GPU")
                for v in out.values()
                if isinstance(v, dict)
            )
            out["status"] = "PASS" if ok else "BOUNDED_FAILURE"
            return out

        run_stage("stage6_protein157_gpu_probe", stage6_protein157_gpu)

        # =====================================================================
        # STAGE 7: Nanotube CPU8 F-SAPT reference + oracle (time-permitting)
        # =====================================================================
        def stage7_nanotube_fsapt_cpu():
            if "doc" not in nanotube_gpu_ref:
                return {"status": "SKIPPED_NO_GPU_PRODUCER"}
            rem = remaining_seconds()
            if rem < 2400:
                return {"status": "SKIPPED_LOW_WALLTIME", "remaining_seconds": rem}
            cpu_timeout = max(1800, int(rem - 600))
            _, cpu_doc, cpu_secs = run_fragment_worker(
                "stage7-nanotube-cpu-inst", "nanotube", "cpu",
                instrumented=True, timeout=cpu_timeout,
            )
            gpu_doc = nanotube_gpu_ref["doc"]
            gpu_secs = nanotube_gpu_ref["inst_seconds"]
            step_ratios = {}
            for key in sorted((set(cpu_secs) & set(gpu_secs)) - {"other", "gpu_fragment_overhead"}):
                if gpu_secs[key] > 0:
                    step_ratios[key] = {
                        "cpu_seconds": cpu_secs[key],
                        "gpu_seconds": gpu_secs[key],
                        "speedup": cpu_secs[key] / gpu_secs[key],
                    }
            res = {
                "status": "PASS",
                "cpu_seconds": cpu_doc["benchmark_whole_energy_seconds"],
                "gpu_seconds": gpu_doc["benchmark_whole_energy_seconds"],
                "whole_speedup": (
                    cpu_doc["benchmark_whole_energy_seconds"] / gpu_doc["benchmark_whole_energy_seconds"]
                ),
                "step_ratios": step_ratios,
                "energy_check": compare_energy(cpu_doc, gpu_doc),
                "signed_terms_check": compare(cpu_doc, gpu_doc),
            }
            if remaining_seconds() > cpu_doc["benchmark_whole_energy_seconds"] * 1.3 + 300:
                _, oracle_doc, _ = run_fragment_worker(
                    "stage7-nanotube-oracle", "nanotube", "cpu",
                    oracle=nanotube_gpu_ref["path"],
                    timeout=max(1800, int(remaining_seconds() - 300)),
                )
                res["same_gauge_max_matrix_err_Eh"] = max(
                    v["maximum_absolute_difference"] for v in oracle_doc["same_gauge_errors"].values()
                )
            else:
                res["oracle_status"] = "SKIPPED_LOW_WALLTIME"
            return res

        run_stage("stage7_nanotube_fsapt_cpu", stage7_nanotube_fsapt_cpu)

        report["stage_summary"] = summarize_stage_status(report["stages"])
        core_passed = all(
            report["stages"].get(k, {}).get("status") == "PASS"
            for k in (
                "stage1_small_gate",
                "stage2_fsapt1_phenol_dimer",
                "stage3a_peptide_sapt0",
                "stage3b_peptide_fsapt",
            )
        )
        report["status"] = "PASS_SUITE" if core_passed else "PARTIAL_SUITE"
    except BaseException as error:
        report["status"] = "FAILED"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            if owner is not None:
                report["scratch_cleanup"] = owner.cleanup(groups)
        except BaseException as error:
            report["status"] = "FAILED"
            report["cleanup_error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            checkpoint()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    sys.path.insert(0, str(Path(manifest["helper_source"]) / "devtools/disp0/phoenix_validation"))
    from owned_process import termination_signals
    with termination_signals():
        execute(args.manifest, args.output)
