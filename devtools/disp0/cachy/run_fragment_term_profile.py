"""Two GPU-only follow-ups around an unchanged, previously checked whole stage."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from profile_sapt_steps import (
    admit_device, checkpoint, device_state, digest, parse_timers,
)


def decomposition(trace, full_seconds):
    workflow = trace["workflow"]
    parts = {name: workflow["windows"][name]["wall_seconds"]
             for name in ("elst", "exch", "ind", "disp")}
    parts["workflow_other"] = workflow["workflow_other_seconds"]
    parts["outer_factory_other"] = full_seconds-workflow["wall_seconds"]
    if any(v < 0 for v in parts.values()):
        raise RuntimeError("Invalid full-fragment timing residual")
    if abs(sum(parts.values())-full_seconds) > 1e-7:
        raise RuntimeError("Fragment term breakdown does not close")
    return {"full_fragment_seconds": full_seconds, "parts_seconds": parts,
            "scope": "INSTRUMENTED_GPU_FRAGMENT_WALL_NOT_BASELINE_OR_KERNEL_TIMING"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, reference = args.root.resolve(strict=True), args.reference.resolve(strict=True)
    output = args.output.resolve()
    if (root != Path.home()/"gits/psi4-disp0-perf-cachy"
            or output.parent != root/"runs" or output.exists() or output.is_symlink()):
        raise ValueError("Fresh output in the owned private performance root required")
    if digest(reference) != args.reference_sha256:
        raise RuntimeError("Completed reference receipt changed")
    previous = json.loads(reference.read_text())
    if (previous["status"] != "PASS_SMALL_RTX3090_DRIVER_DIAGNOSTICS_NOT_H200_SPEEDUP"
            or any(w["returncode"] or not w["owned_group_reaped"] for w in previous["workers"])):
        raise RuntimeError("Completed independently checked GPU baselines required")
    source = root/"source"
    sys.path[:0] = [str(source/"devtools/disp0/cachy"),
                    str(source/"devtools/disp0/phoenix_validation")]
    from coherent_stage11 import lease, clean
    from gpu_runtime import read_config
    from owned_process import execute
    from repeat_gauge import compare_repeat
    from response_reuse_probe import resource_scope
    from run_gate import environment
    output.mkdir(mode=0o700)
    here = Path(__file__).resolve().parent
    report = output/"term-profile-result.json"
    result = {
        "status": "STARTED", "scope": "SYNCHRONIZED_RTX3090_TERM_DIAGNOSTIC_NOT_H200_SPEEDUP",
        "reference_sha256": args.reference_sha256, "resource_scope": resource_scope(),
        "science_pin": previous["science_pin"], "cases": {},
        "helper_pin": subprocess.check_output([
            "git", "-C", str(here.parents[2]), "rev-parse", "HEAD"], text=True).strip(),
        "helper_sha256": {p.name: digest(p) for p in (
            Path(__file__), here/"fragment_term_trace.py", here/"profile_sapt_steps.py")},
    }
    checkpoint(report, result)
    deadline = time.monotonic()+1200
    try:
        with lease(root) as fd:
            clean(source, previous["science_pin"])
            for case in ("water", "peptide"):
                record = {"status": "STARTED"}
                result["cases"][case] = record
                config_path = reference.parent/(case+"-runtime-config.json")
                baseline_worker = next(w for w in previous["workers"] if w["label"] == case+"-gpu")
                if digest(config_path) != baseline_worker["config_sha256"]:
                    raise RuntimeError("Unchanged case configuration required")
                config, _ = read_config(config_path, digest(config_path))
                if config["source_pin"] != previous["science_pin"] or config["threads"] != 4:
                    raise RuntimeError("Original four-thread scientific policy required")
                aliases = Path(config["sealed_alias_parent"])/(output.name+"-"+case)
                aliases.mkdir(mode=0o700)
                # Fresh children only; no prior diagnostic alias/data is repointed.
                env = environment(config, gpu=True, sealed_alias_directory=aliases)
                scratch = output/(case+"-scratch")
                scratch.mkdir(mode=0o700)
                env.update(TMPDIR=str(scratch), SCRATCH=str(scratch), PSI_SCRATCH=str(scratch))
                worker, trace_path = output/(case+"-gpu"), output/(case+"-term-windows.json")
                payload = [
                    str(here/"fragment_term_trace.py"),
                    "--bootstrap", str(source/"devtools/disp0/cachy/bootstrap_gate.py"),
                    "--config", str(config_path), "--config-sha256", digest(config_path),
                    "--trace-output", str(trace_path), "--arm", "gpu", "--scaled",
                    "--benchmark", "--output", str(worker),
                    "--block-size", "24" if case == "peptide" else "5",
                ]
                if case == "peptide":
                    inputs = reference.parent/"inputs/inputs.json"
                    if digest(inputs) != previous["inputs_sha256"]:
                        raise RuntimeError("Original peptide input manifest changed")
                    payload += ["--inputs", str(inputs), "--inputs-sha256", digest(inputs),
                                "--system", "peptide", "--factor-cache-bytes", str(64*1024**2)]
                argv = ["/bin/bash", "-c",
                        'set -euo pipefail; export PYTHONPATH="${PYTHONPATH-}"; '
                        'eval "$("$1" --psiapi)"; exec "$2" -B "${@:3}"',
                        "sapt-term-profile", config["stage"]+"/bin/psi4",
                        config["dependency_environment"]+"/bin/python", *payload]
                record.update(argv=argv, runtime_config_sha256=digest(config_path),
                              device_before=device_state(config["local_lease"]["expected_uuid"]))
                admit_device(record["device_before"], config)
                checkpoint(report, result)
                seconds = min(600, deadline-time.monotonic())
                if seconds < 60:
                    raise TimeoutError("Bounded GPU term-profile deadline exhausted")
                print("START", case, flush=True)
                started = time.perf_counter()
                with (output/(case+".stdout")).open("x") as out, (
                        output/(case+".stderr")).open("x") as err:
                    code = execute(argv, env, out, err, seconds, fd, config["local_lease"], cwd=root)
                record.update(returncode=code, owned_group_reaped=True,
                              inclusive_seconds=time.perf_counter()-started)
                checkpoint(report, result)
                if code:
                    raise RuntimeError("Term-profile worker failed: "+case)
                original = reference.parent/(case+"-gpu")
                if digest(original/"result.json") != baseline_worker["result_sha256"]:
                    raise RuntimeError("Original baseline result changed")
                document = json.loads((worker/"result.json").read_text())
                if document["source_pin"] != previous["science_pin"]:
                    raise RuntimeError("Scientific source changed")
                trace = json.loads(trace_path.read_text())
                if trace["status"] != "PASS_OBSERVED_TERM_BOUNDARIES":
                    raise RuntimeError("Incomplete term tracing")
                timers = parse_timers((worker/"timer.dat").read_text())
                full = timers["driver_steps"]["SAPT(DFT): GPU F-SAPT"]["wall_seconds"]
                record.update(
                    repeat_check=compare_repeat(original, worker,
                                                dimensions=document["orbital_dimensions"], arm="gpu"),
                    decomposition=decomposition(trace["trace"], full),
                    trace=trace["trace"], trace_sha256=digest(trace_path),
                    result_sha256=digest(worker/"result.json"),
                    timer_sha256=digest(worker/"timer.dat"),
                    energy_seconds=document["benchmark_whole_energy_seconds"],
                    status="PASS_FOUR_TERM_WINDOWS_AND_STRICT_FRESH_REPEAT",
                )
                checkpoint(report, result)
                print("DONE", case, json.dumps(record["decomposition"]), flush=True)
            clean(source, previous["science_pin"])
            result["status"] = "PASS_RTX3090_FOUR_TERM_DIAGNOSTICS"
    except BaseException as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        checkpoint(report, result)


if __name__ == "__main__":
    main()
