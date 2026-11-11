"""Observe an authenticated worker without replacing installed scientific code.

All step timings are instrumented wall times, not CUDA event/kernel timings.
Separate untraced workers supply the end-to-end baseline.
"""
import argparse
import json
from pathlib import Path
import runpy
import sys
import time


def ledger(before, after, total, gpu_trace=None):
    import math

    def timer(name):
        keys = [key for key in after if key == name or key.endswith(";" + name)]
        if len(keys) != 1:
            raise ValueError("Exactly one timer path required: " + name)
        key = keys[0]
        old = before.get(key, {})
        value = after[key]["wall_time"] - old.get("wall_time", 0.)
        if (after[key]["n_calls"] - old.get("n_calls", 0) != 1
                or not math.isfinite(value) or value < 0):
            raise ValueError("Invalid timer: " + name)
        return value

    rows = {label: timer("SAPT(DFT):" + name) for label, name in (
        ("scf_dimer", "Dimer SCF"), ("scf_A", "Monomer A SCF"),
        ("scf_B", "Monomer B SCF"), ("canonical_prepare_JK", "Build JK"),
        ("canonical_elst", "elst"), ("canonical_exch", "exch"),
        ("canonical_ind", "ind"), ("delta_hf_assembly", "delta HF assembly"))}
    rows["canonical_ind"] -= rows["delta_hf_assembly"]
    if gpu_trace is None:
        rows["fragment_localization"] = sum(timer("SAPT(DFT):" + name) for name in (
            "Localize Orbitals", "Partition", " F-SAPT Localization (IBO)"))
        rows.update({"fragment_" + label: timer("SAPT(DFT): F-SAPT " + name)
                     for label, name in (("elst", "Electrostatics"), ("exch", "Exchange"),
                                         ("ind", "Induction"), ("disp", "Dispersion"))})
    else:
        rows.update({"fragment_" + label: window["wall_seconds"]
                     for label, window in gpu_trace["workflow"]["windows"].items()})
        required = {"localization", "elst", "exch", "ind", "disp"}
        if set(gpu_trace["workflow"]["windows"]) != required:
            raise ValueError("Incomplete GPU fragment windows")
        fragment = sum(rows["fragment_" + label] for label in required)
        rows["gpu_fragment_overhead"] = timer("SAPT(DFT): GPU F-SAPT") - fragment
    rows["other"] = total - sum(rows.values())
    rows["total"] = total
    if any(not math.isfinite(value) or value < 0 for value in rows.values()):
        raise ValueError("Negative/nonfinite disjoint timing")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--arm", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    bootstrap = args.bootstrap.resolve(strict=True)
    sys.path.insert(0, str(bootstrap.parent))
    sys.path.insert(0, str(bootstrap.parent.parent / "cachy"))
    from gpu_runtime import read_config, digest
    from fragment_term_trace import TermTrace
    _, build = read_config(args.config, args.config_sha256)
    if build["provenance_sha256"].get(str(bootstrap)) != digest(bootstrap):
        raise RuntimeError("Authenticated original bootstrap required")
    package = Path(build["python_package"])
    driver = package / "driver/driver.py"
    workflow = package / "driver/procrouting/sapt/sapt_fisapt_workflow_cuda.py"
    for path in (driver, workflow):
        if digest(path) != build["scientific_python_sha256"][str(path.relative_to(package))]:
            raise RuntimeError("Installed scientific source changed")
    terms = TermTrace(workflow, workflow.read_text(), include_localization=True)
    calls = []

    def observe(frame, event, arg):
        if (event == "call" and frame.f_code.co_name == "energy"
                and frame.f_code.co_filename == str(driver)
                and frame.f_locals.get("name") == "sapt(dft)"):
            core = sys.modules["psi4.core"]
            row = {"before": dict(core.get_timer_records(False))}
            calls.append(row)
            started = time.perf_counter()

            def energy_event(current, event, value):
                if event == "return":
                    row["wall_seconds"] = time.perf_counter() - started
                    row["after"] = dict(core.get_timer_records(False))
                    row["completed"] = value is not None
                    return None
                return energy_event

            # No line tracing inside the driver; only its final return.
            frame.f_trace_lines = False
            return energy_event
        return terms.global_trace(frame, event, arg) if args.arm == "gpu" else None

    report = {"status": "STARTED", "arm": args.arm,
              "scope": "INSTRUMENTED_SYNCHRONIZED_WALL_NOT_KERNEL_TIMING"}
    if args.receipt.exists() or sys.gettrace() is not None:
        raise RuntimeError("Fresh untraced process and receipt required")
    try:
        sys.argv = [str(bootstrap), "--config", str(args.config),
                    "--config-sha256", args.config_sha256, "--arm", args.arm,
                    "--output", str(args.output), *remaining]
        sys.settrace(observe)
        try:
            runpy.run_path(str(bootstrap), run_name="__main__")
        finally:
            sys.settrace(None)
        if len(calls) != 1 or not calls[0].get("completed"):
            raise RuntimeError("One completed public energy call required")
        report["energy"] = calls[0]
        trace = terms.report() if args.arm == "gpu" else None
        report["fragment_trace"] = trace
        result = json.loads((args.output / "result.json").read_text())
        if (result["status"] != "EXECUTED_REQUIRES_INDEPENDENT_COMPARISON"
                or result["arm"] != args.arm or not result["benchmark"]):
            raise RuntimeError("Completed benchmark worker required")
        # Preserve the same outer guarded-call scope as the untraced baseline.
        # Audit work surrounding psi4.energy remains charged to the residual.
        total = result["benchmark_whole_energy_seconds"]
        if total < calls[0]["wall_seconds"]:
            raise RuntimeError("Public energy interval exceeds its enclosing timer")
        report["seconds"] = ledger(calls[0]["before"], calls[0]["after"],
                                   total, trace)
        report["status"] = "PASS_OBSERVED_TIMERS"
    except BaseException as error:
        report.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        with args.receipt.open("x") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)


if __name__ == "__main__":
    main()
