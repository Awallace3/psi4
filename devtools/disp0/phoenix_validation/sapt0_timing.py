"""Matched canonical SAPT0(+delta-HF) through energy('sapt(dft)').

Fresh processes, one energy per process, alternating CPU/GPU order. No F-SAPT
localization, empirical dispersion, fitted correction, or timing subtraction
for audits/JIT. GPU runtime admission is outside the energy timer; provider
construction, JIT, native audits and cleanup inside energy remain charged.
"""
import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

SYSTEMS = ("peptide", "nanotube", "protein83", "protein157")
EXPECTED_NBF = dict(water=56, peptide=250, nanotube=548, protein83=922, protein157=1786)
RAW = ("Elst10,r", "Exch10", "Ind20,r", "Exch-Ind20,r",
       "Disp20,u", "Exch-Disp20,u", "Delta HF Correction")
TERMS = ("elst", "exch", "ind", "disp", "delta_hf")
TIMER = {k: "SAPT(DFT):" + k for k in TERMS[:4]}
ASSEMBLY = "SAPT(DFT):delta HF assembly"
DIMER = "SAPT(DFT):Dimer SCF"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)


def timer_delta(before, after, name):
    """Only actual matching tree records; missing is never measured zero."""
    keys = [k for k in after if k == name or k.endswith(";" + name)]
    if not keys:
        raise ValueError("Missing required timer: " + name)
    seconds = sum(after[k]["wall_time"] - before.get(k, {}).get("wall_time", 0.)
                  for k in keys)
    calls = sum(after[k]["n_calls"] - before.get(k, {}).get("n_calls", 0) for k in keys)
    if calls != 1 or not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"Invalid timer interval {name}: {seconds}, calls={calls}")
    return seconds


def term_times(before, after, total):
    result = {k: timer_delta(before, after, v) for k, v in TIMER.items()}
    correction = timer_delta(before, after, ASSEMBLY)
    result["ind"] -= correction
    result["delta_hf"] = timer_delta(before, after, DIMER) + correction
    result["shared_setup_other"] = total - sum(result.values())
    result["total"] = total
    if any(not math.isfinite(x) or x < 0 for x in result.values()):
        raise ValueError("Negative/nonfinite disjoint timing")
    return result


def options(gpu):
    return {
        "basis": "jun-cc-pvdz", "puream": True, "reference": "rhf", "scf_type": "df",
        # Same explicit spherical auxiliary sets in both arms.
        "df_basis_scf": "aug-cc-pvdz-jkfit", "df_basis_sapt": "aug-cc-pvdz-ri",
        "df_basis_mp2": "aug-cc-pvdz-ri", "df_fitting_condition": 1e-12,
        "use_cuest": gpu, "cuest_mixed_precision": False, "cuest_gemm_min_dim": 0,
        "cuest_gridk": False, "cuest_directj": False,
        "cuest_device_cg": False, "cuest_diis": False, "cuest_response_profile": False,
        "guess": "sad", "sad_scf_type": "df", "df_scf_guess": False,
        "e_convergence": 1e-10, "d_convergence": 1e-10, "ints_tolerance": 1e-16,
        "maxiter": 100, "freeze_core": True, "orbital_optimizer_package": "INTERNAL",
        "sapt_dft_functional": "HF", "sapt_dft_do_fsapt": "NONE",
        "sapt_dft_use_einsums": True, "sapt_dft_mp2_disp_alg": "SAPT",
        "sapt_dft_einsums_gpu": False, "sapt_dft_do_ddft": False,
        "sapt_dft_do_disp": True, "sapt_dft_do_dhf": True,
        "sapt_dft_induction_type": "CPHF", "cphf_r_convergence": 1e-10,
        "sapt_dft_grac_compute": "NONE", "sapt_dft_grac_shift_a": 0.,
        "sapt_dft_grac_shift_b": 0., "sapt_dft_d3_ie": False, "sapt_dft_d4_ie": False,
    }


def prepare(suite, output):
    source = suite.resolve(strict=True) / "geometries/systems.py"
    spec = importlib.util.spec_from_file_location("timing_suite_inputs", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output.mkdir()
    for name in SYSTEMS:
        text = module.SYSTEMS[name]
        save(output / (name + ".json"), {
            "schema": 1, "system": name, "geometry": text,
            "geometry_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "suite_source": str(source), "suite_source_sha256": sha(source)})
    # Independent small public-driver smoke before timing the suite.
    from fragment_gate import WATER
    save(output / "water.json", {
        "schema": 1, "system": "water", "geometry": WATER,
        "geometry_sha256": hashlib.sha256(WATER.encode()).hexdigest(),
        "suite_source": "small correctness gate, not a timing-suite dimer"})


def forbid_cpu_dispersion(stack):
    """Patch the actual Einsums module, not sapt_proc's local `sapt_mp2` alias."""
    from unittest.mock import patch
    from psi4.driver.procrouting.sapt import sapt_mp2_terms_ein
    for name in ("df_mp2_sapt_dispersion", "df_mp2_einsums_dispersion",
                 "df_mp2_fisapt_dispersion"):
        stack.enter_context(patch.object(
            sapt_mp2_terms_ein, name, side_effect=RuntimeError("CPU dispersion fallback forbidden")))


def worker(args):
    from gpu_runtime import AuditedRuntime, read_config
    config, build = read_config(args.config, args.config_sha256)
    import psi4
    from psi4 import core
    if not Path(core.__file__).resolve().is_relative_to(Path(config["stage"]).resolve()):
        raise RuntimeError("Foreign core")
    case = json.loads(args.case.read_text())
    if hashlib.sha256(case["geometry"].encode()).hexdigest() != case["geometry_sha256"]:
        raise ValueError("Changed geometry")
    args.output.mkdir()
    result = {"status": "FAILED", "arm": args.arm, "source_pin": config["source_pin"],
              "input": case, "input_sha256": sha(args.case), "options": options(args.arm == "gpu"),
              "threads": config["threads"], "host_memory": config["host_memory"],
              "host": platform.node(), "job_id": os.environ.get("SLURM_JOB_ID"),
              "runtime_config_sha256": args.config_sha256}
    runtime = None
    try:
        core.set_output_file(str(args.output / "psi4.out"), False)
        psi4.set_num_threads(config["threads"])
        psi4.set_memory(config["host_memory"])
        psi4.set_options(result["options"])
        molecule = psi4.geometry(case["geometry"])
        molecule.reset_point_group("c1")
        molecule.update_geometry()
        if molecule.nfragments() != 2:
            raise ValueError("Exactly two original fragments required")
        if args.arm == "gpu":
            runtime = AuditedRuntime(args.config, args.config_sha256)
        result["runtime_before"] = dict(core.cuest_runtime_info())
        kwargs = {}
        if runtime:
            kwargs = {"cuda_provider_factory": runtime.provider,
                      "cuda_dispersion_block_size": 32,
                      "cuda_dispersion_factor_cache_bytes": 4 * 1024**3}
        with ExitStack() as stack:
            if runtime:
                forbid_cpu_dispersion(stack)
            before = dict(core.get_timer_records(False))
            started = time.perf_counter()
            energy, wfn = psi4.energy("sapt(dft)", molecule=molecule, return_wfn=True, **kwargs)
            elapsed = time.perf_counter() - started
            after = dict(core.get_timer_records(False))
        result["seconds"] = term_times(before, after, elapsed)
        result["timers_before"], result["timers_after"] = before, after
        result["raw_Eh"] = {name: float(wfn.variable(name)) for name in RAW}
        raw = result["raw_Eh"]
        result["term_Eh"] = dict(zip(TERMS, (
            raw["Elst10,r"], raw["Exch10"], raw["Ind20,r"] + raw["Exch-Ind20,r"],
            raw["Disp20,u"] + raw["Exch-Disp20,u"], raw["Delta HF Correction"])))
        if not all(math.isfinite(x) for x in (*raw.values(), energy)):
            raise ValueError("Nonfinite energy")
        if abs(sum(result["term_Eh"].values()) - energy) > 1e-10:
            raise ValueError("Disjoint SAPT0 energy terms do not reconstruct total")
        result["total_Eh"] = energy
        result["nbf"] = wfn.basisset().nbf()
        if result["nbf"] != EXPECTED_NBF[case["system"]]:
            raise ValueError("Unexpected spherical jun-cc-pVDZ AO count")
        result["dimer_nmo"] = wfn.nmo()
        result["runtime_after"] = dict(core.cuest_runtime_info())
        if runtime:
            result["dispersion_execution"] = wfn._sapt_cuda_dispersion_execution
            if not runtime.report["plans"] or not result["runtime_after"]["initialized"]:
                raise RuntimeError("No actual GPU execution")
            for plan in runtime.report["plans"]:
                if (not plan["session_cleanup"]["complete"]
                        or plan["session_cleanup"]["live_bytes"]
                        or plan["session_cleanup"]["native_owners"]):
                    raise RuntimeError("Incomplete GPU cleanup")
            if result["runtime_after"]["native_math_mode"] != 1:
                raise RuntimeError("Native FP64 was not retained")
            runtime.verify_optional_modules()
        elif any(result["runtime_after"][key] for key in
                 ("initialized", "has_cuest", "has_cublas", "has_cusolver", "has_stream")):
            raise RuntimeError("CPU arm initialized a GPU")
        result["status"] = "PASS"
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if runtime:
            result["runtime"] = runtime.report
        core.close_outfile()
        save(args.output / "result.json", result)


def compare(records):
    """Qualify every signed raw term and every repetition before any speedup."""
    if not records:
        raise ValueError("No completed samples")
    cpu = [r for r in records if r["arm"] == "cpu"]
    gpu = [r for r in records if r["arm"] == "gpu"]
    if not cpu or len(cpu) != len(gpu):
        raise ValueError("Incomplete paired samples")
    first = cpu[0]
    for row in records:
        if row["status"] != "PASS":
            raise ValueError("Failed sample")
        for field in ("input_sha256", "source_pin", "threads", "host", "nbf", "dimer_nmo", "host_memory",
                      "runtime_config_sha256"):
            if row[field] != first[field]:
                raise ValueError("Unmatched " + field)
        expected = dict(first["options"], use_cuest=row["arm"] == "gpu")
        if row["options"] != expected:
            raise ValueError("Unmatched scientific options")
    errors = {}
    for key in (*RAW, "total"):
        value = lambda r: r["total_Eh"] if key == "total" else r["raw_Eh"][key]
        if not all(math.isfinite(value(row)) for row in records):
            raise ValueError("Nonfinite signed component: " + key)
        reference = value(first)
        errors[key] = max(abs(value(r) - reference) for r in gpu)
        for row in gpu:
            if not math.isclose(value(row), reference, rel_tol=1e-8, abs_tol=2e-8):
                raise ValueError("CPU/GPU signed component mismatch: " + key)
        for arm in (cpu, gpu):
            for row in arm:
                if not math.isclose(value(row), value(arm[0]), rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError("Fresh-process repetition mismatch: " + key)
    summary = {}
    for term in (*TERMS, "shared_setup_other", "total"):
        times = {arm: [r["seconds"][term] for r in rows] for arm, rows in (("cpu", cpu), ("gpu", gpu))}
        if any(not math.isfinite(x) or x < 0 for values in times.values() for x in values):
            raise ValueError("Invalid timing sample")
        c, g = statistics.median(times["cpu"]), statistics.median(times["gpu"])
        summary[term] = {"samples_seconds": times, "cpu_median_seconds": c,
                         "gpu_median_seconds": g, "cpu_over_gpu": c / g if g > 0 else None}
    return {"status": "PASS_MATCHED_SAPT0", "signed_max_error_Eh": errors, "timing": summary,
            "repeats_per_arm": len(cpu), "scope": "canonical SAPT0+delta-HF; not F-SAPT"}


def run(args):
    from gpu_runtime import read_config
    from run_gate import environment
    config, build = read_config(args.config, args.config_sha256)
    if config.get("identity_contract", "fresh-bytes-v1") != "fresh-bytes-v1":
        raise ValueError("This first campaign requires the existing fresh-byte runtime")
    args.output.mkdir()
    records, children = [], []
    result = {"status": "FAILED", "case": str(args.case), "children": children,
              "helper_sha256": sha(__file__), "cold_fresh_processes": True}
    try:
        for repeat in range(args.repeats):
            for arm in (("cpu", "gpu") if repeat % 2 == 0 else ("gpu", "cpu")):
                label = f"{arm}-{repeat}"
                child = args.output / label
                scratch = Path(os.environ["TMPDIR"]) / (args.output.name + "-" + label)
                scratch.mkdir()
                env = environment(config, gpu=arm == "gpu")
                env.update(TMPDIR=str(scratch), SCRATCH=str(scratch), PSI_SCRATCH=str(scratch),
                           OMP_NUM_THREADS=str(config["threads"]), MKL_NUM_THREADS=str(config["threads"]),
                           OPENBLAS_NUM_THREADS=str(config["threads"]), PYTHONNOUSERSITE="1",
                           PYTHONDONTWRITEBYTECODE="1", OMP_DYNAMIC="FALSE")
                command = ["/bin/bash", "-c",
                    'set -eo pipefail; activation=$("$1" --psiapi); eval "$activation"; '
                    'set -u; exec "$2" -B "$3" "${@:4}"',
                    "sapt0-worker", config["stage"] + "/bin/psi4",
                    config["dependency_environment"] + "/bin/python", str(Path(__file__).resolve()),
                    "worker", "--config", str(args.config), "--config-sha256", args.config_sha256,
                    "--case", str(args.case), "--output", str(child), "--arm", arm]
                with (args.output / (label + ".stdout")).open("x") as out, (
                        args.output / (label + ".stderr")).open("x") as err:
                    completed = subprocess.run(command, env=env, cwd=scratch, stdout=out, stderr=err,
                                               timeout=args.timeout)
                children.append({"label": label, "returncode": completed.returncode,
                                 "command": command, "scratch": str(scratch)})
                if completed.returncode:
                    raise RuntimeError("Worker failed: " + label)
                receipt = child / "result.json"
                children[-1]["result_sha256"] = sha(receipt)
                records.append(json.loads(receipt.read_text()))
        result.update(compare(records))
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save(args.output / "comparison.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--suite", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    for name in ("worker", "run"):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--config-sha256", required=True)
        p.add_argument("--case", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        if name == "worker":
            p.add_argument("--arm", choices=("cpu", "gpu"), required=True)
        else:
            p.add_argument("--repeats", type=int, choices=(1, 3), default=3)
            p.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.suite, args.output)
    elif args.mode == "worker":
        worker(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
