"""Bounded RTX3090 water/peptide driver diagnostics, not H200 speedup evidence.

Reuse an unchanged, CPU-qualified whole Cachy installation and its authenticated
bootstrap. This outer experiment launcher never overlays scientific modules.
Baseline workers retain their own timer.dat; a separate cProfile worker exposes
inclusive Python/native-call windows without pretending they are kernel times.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pstats
import re
import subprocess
import sys
import time

SCOPE = "RTX3090_DEBUG_ONLY_NOT_H200_SPEEDUP"
GIB = 1024**3
HEADER = re.compile(r"^Module\s+User\s+System\s+Wall\s+Calls\s*$")
TIMER = re.compile(
    r"^(.*?)\s*:\s*([\d.]+)u\s+([\d.]+)s\s+([\d.]+)w\s+(\d+) calls")
DRIVER_TIMERS = (
    "SAPT(DFT):Dimer SCF", "SAPT(DFT):Monomer A SCF", "SAPT(DFT):Monomer B SCF",
    "SAPT(DFT):Build JK", "SAPT(DFT):elst", "SAPT(DFT):exch", "SAPT(DFT):ind",
    "SAPT(DFT): GPU F-SAPT", "SAPT(DFT):Localize Orbitals", "SAPT(DFT):Partition",
    "SAPT(DFT): F-SAPT Localization (IBO)", "SAPT(DFT): F-SAPT Electrostatics",
    "SAPT(DFT): F-SAPT Exchange", "SAPT(DFT): F-SAPT Induction",
    "SAPT(DFT):disp", "SAPT(DFT): F-SAPT Dispersion",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def checkpoint(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")
    temporary.replace(path)


def parse_timers(text):
    """One nonempty flat epoch only; never add the duplicate nested tree."""
    state, current, epochs = "seeking", {}, []
    for line in text.splitlines():
        stripped = line.strip()
        if HEADER.fullmatch(stripped):
            if current:
                epochs.append(current)
            state, current = "flat", {}
        elif state == "flat" and len(stripped) > 20 and set(stripped) == {"-"}:
            state = "tree"
        elif stripped and set(stripped) == {"*"}:
            if current:
                epochs.append(current)
            state, current = "seeking", {}
        elif state == "flat":
            match = TIMER.match(line)
            if not match:
                continue
            name = match[1].strip()
            if not name or name.startswith("|"):
                continue
            if name in current:
                raise ValueError("Duplicate flat timer within one epoch: "+name)
            current[name] = {
                "user_seconds": float(match[2]), "system_seconds": float(match[3]),
                "wall_seconds": float(match[4]), "calls": int(match[5]),
            }
    if current:
        epochs.append(current)
    if len(epochs) != 1:
        raise ValueError("Exactly one nonempty timer epoch required, got "+str(len(epochs)))
    timers = epochs[0]
    if "SAPT(DFT) Energy" not in timers or timers["SAPT(DFT) Energy"]["calls"] != 1:
        raise ValueError("One completed SAPT driver call required")
    return {
        "scope": "FLAT_TIMER_WALL_COLUMNS_NOT_SUMMED_WITH_PARENT_OR_NESTED_TREE",
        "nonempty_epochs": 1, "all_flat_timers": timers,
        "driver_steps": {name: timers.get(name) for name in DRIVER_TIMERS},
        "driver_energy": timers["SAPT(DFT) Energy"],
    }


def case_policy(base, case, alias_parent):
    if case not in ("water", "peptide"):
        raise ValueError("Only the explicitly bounded water and peptide cases are selected")
    result = dict(base)
    result.update(threads=4, host_memory="2 GiB" if case == "water" else "6 GiB",
                  device_memory_limit_bytes=GIB if case == "water" else 20*GIB,
                  native_persistent_bytes=GIB//2 if case == "water" else 12*GIB,
                  sealed_alias_parent=str(alias_parent))
    return result


def device_state(uuid):
    raw = subprocess.check_output([
        "/usr/bin/nvidia-smi", "--id="+uuid,
        "--query-gpu=name,uuid,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True, timeout=20)
    name, actual_uuid, total, free, utilization = (v.strip() for v in raw.strip().split(","))
    if actual_uuid != uuid or "RTX 3090" not in name:
        raise RuntimeError("Actual selected desktop GPU changed")
    return {"name": name, "uuid": actual_uuid, "total_MiB": int(total),
            "free_MiB": int(free), "utilization_percent": int(utilization),
            "scope": "POINT_SAMPLE_NOT_PEAK_OR_EXCLUSIVE_DEVICE"}


def admit_device(state, config):
    if state["free_MiB"]*1024**2 < config["device_memory_limit_bytes"]+GIB//2:
        raise MemoryError("Insufficient free VRAM for bounded budget plus desktop margin")
    if state["utilization_percent"] > 10:
        raise RuntimeError("Desktop GPU is busy; do not compete with the user's work")


def profile_windows(path):
    rows = []
    for (filename, line, function), (primitive, calls, own, inclusive, _) in pstats.Stats(
            str(path)).stats.items():
        if "/sapt/" not in filename or not Path(filename).name.startswith("sapt_"):
            continue
        rows.append({"file": filename, "line": line, "function": function,
                     "primitive_calls": primitive, "calls": calls,
                     "own_seconds": own, "inclusive_seconds": inclusive})
    return {
        "scope": "SEPARATE_PROFILED_WORKER_INCLUSIVE_OVERLAPPING_WINDOWS_NOT_KERNEL_TIMES",
        "largest_sapt_windows": sorted(rows, key=lambda r: -r["inclusive_seconds"])[:60],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True,
                        help="Existing frozen scientific/helper checkout, not this outer launcher")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--suite-module", type=Path, required=True)
    parser.add_argument("--suite-module-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--total-seconds", type=int, default=5400)
    parser.add_argument("--profile-peptide", action="store_true")
    args = parser.parse_args()
    root, source = args.root.resolve(strict=True), args.source.resolve(strict=True)
    output = args.output.resolve()
    if (root != Path.home()/"gits/psi4-disp0-perf-cachy"
            or source != root/"source" or output.parent != root/"runs"
            or output.exists() or output.is_symlink()):
        raise ValueError("Fresh run inside the existing private performance root required")
    if not 60 <= args.seconds <= 1800 or not args.seconds <= args.total_seconds <= 5400:
        raise ValueError("Bounded per-child/total diagnostic deadlines required")
    if digest(args.suite_module) != args.suite_module_sha256:
        raise RuntimeError("Timing-suite input copy changed")
    # These are the original authenticated helpers, not new scientific imports.
    sys.path[:0] = [str(source/"devtools/disp0/cachy"),
                    str(source/"devtools/disp0/phoenix_validation")]
    from coherent_stage11 import lease, clean
    from gpu_runtime import read_config
    from response_reuse_probe import resource_scope
    from owned_process import execute
    from run_gate import environment, compare
    from run_full import compare_energy
    from prepare_inputs import prepare
    from repeat_gauge import compare_repeat
    limits = resource_scope()
    output.mkdir(mode=0o700)
    result = {
        "status": "STARTED", "scope": SCOPE, "cases": {}, "workers": [],
        "resource_scope": limits, "selected_cases": ["water", "peptide"],
        "excluded": ["nanotube", "protein83", "protein157"],
        "excluded_reason": "Not admitted to this bounded <=30-atom desktop diagnostic",
        "outer_launcher": str(Path(__file__).resolve()),
        "outer_launcher_sha256": digest(__file__),
        "outer_helper_pin": subprocess.check_output([
            "git", "-C", str(Path(__file__).resolve().parents[3]), "rev-parse", "HEAD"],
            text=True).strip(),
        "suite_module_sha256": args.suite_module_sha256,
        "accepted_h200_speedups": 0,
    }
    report = output/"profile-result.json"
    deadline = time.monotonic()+args.total_seconds
    checkpoint(report, result)
    try:
        with lease(root) as fd:
            base, _ = read_config(args.config, args.config_sha256)
            if base.get("identity_contract") != "process-owned-kernel-sealed-snapshots-v1":
                raise ValueError("Previously gated sealed whole stage required")
            clean(source, base["source_pin"])
            result.update(science_pin=base["source_pin"], stage=base["stage"],
                          reference_config_sha256=args.config_sha256)
            h200_pin = "927283f8ed0a94d42179c35dd2a1a5bce674347e"
            result["h200_comparison_scope"] = {
                "h200_science_pin": h200_pin,
                "policy": "SAME_WATER_OR_PEPTIDE_METHOD_INPUTS_NOT_IDENTICAL_WHOLE_BUILD",
                "changed_science_or_support_paths": subprocess.check_output([
                    "git", "-C", str(source), "diff", "--name-only",
                    base["source_pin"]+".."+h200_pin, "--",
                    "psi4/driver/procrouting/sapt", "psi4_runtime_support"],
                    text=True).splitlines(),
            }
            prepare(args.suite_module.resolve(strict=True),
                    Path(base["stage"])/"share/psi4/basis", output/"inputs")
            inputs = output/"inputs/inputs.json"
            result["inputs_sha256"] = digest(inputs)

            def child(case, arm, config_path, *, oracle=None, profiled=False):
                label = case+"-"+("oracle" if oracle else arm)+("-profiled" if profiled else "")
                worker = output/label
                config, _ = read_config(config_path, digest(config_path))
                aliases = Path(config["sealed_alias_parent"])/label
                aliases.mkdir(mode=0o700)
                env = environment(config, gpu=arm == "gpu", sealed_alias_directory=aliases)
                scratch = output/(label+"-scratch")
                scratch.mkdir(mode=0o700)
                env.update(TMPDIR=str(scratch), SCRATCH=str(scratch), PSI_SCRATCH=str(scratch))
                script = source/"devtools/disp0/cachy/bootstrap_gate.py"
                payload = [str(script), "--config", str(config_path),
                           "--config-sha256", digest(config_path), "--arm", arm,
                           "--scaled", "--benchmark", "--output", str(worker),
                           "--block-size", "24" if case == "peptide" else "5"]
                if case == "peptide":
                    payload += ["--inputs", str(inputs), "--inputs-sha256", digest(inputs),
                                "--system", "peptide"]
                    if arm == "gpu":
                        payload += ["--factor-cache-bytes", str(64*1024**2)]
                if oracle:
                    payload += ["--oracle-from", str(oracle)]
                if profiled:
                    payload = ["-m", "cProfile", "-o", str(worker/"profile.pstats"), *payload]
                argv = ["/bin/bash", "-c",
                        'set -euo pipefail; export PYTHONPATH="${PYTHONPATH-}"; '
                        'eval "$("$1" --psiapi)"; exec "$2" -B "${@:3}"',
                        "sapt-steps", config["stage"]+"/bin/psi4",
                        config["dependency_environment"]+"/bin/python", *payload]
                record = {"label": label, "arm": arm, "profiled": profiled,
                          "argv": argv, "config_sha256": digest(config_path)}
                result["workers"].append(record)
                if arm == "gpu":
                    record["device_before"] = device_state(config["local_lease"]["expected_uuid"])
                    admit_device(record["device_before"], config)
                checkpoint(report, result)
                seconds = min(args.seconds, deadline-time.monotonic())
                if seconds < 60:
                    raise TimeoutError("Bounded diagnostic total deadline exhausted")
                print("START", label, flush=True)
                started = time.perf_counter()
                with (output/(label+".stdout")).open("x") as stdout, (
                        output/(label+".stderr")).open("x") as stderr:
                    code = execute(argv, env, stdout, stderr, seconds, fd, config["local_lease"],
                                   cwd=root)
                record.update(returncode=code, owned_group_reaped=True,
                              inclusive_seconds=time.perf_counter()-started)
                checkpoint(report, result)
                if code:
                    raise RuntimeError("Scientific worker failed: "+label)
                document = json.loads((worker/"result.json").read_text())
                if (document["status"] != "EXECUTED_REQUIRES_INDEPENDENT_COMPARISON"
                        or document["source_pin"] != base["source_pin"]
                        or document["runtime_config_sha256"] != digest(config_path)
                        or document["arm"] != arm or not document["benchmark"]):
                    raise RuntimeError("Incomplete/mismatched whole worker: "+label)
                for filename, checksum in document["artifact_sha256"].items():
                    if digest(worker/filename) != checksum:
                        raise RuntimeError("Worker matrix artifact changed: "+label)
                record.update(result_sha256=digest(worker/"result.json"),
                              energy_seconds=document["benchmark_whole_energy_seconds"],
                              energy_Eh=document["energy_Eh"],
                              timer_sha256=digest(worker/"timer.dat"),
                              timers=parse_timers((worker/"timer.dat").read_text()))
                if profiled:
                    record["profile"] = profile_windows(worker/"profile.pstats")
                    record["profile_sha256"] = digest(worker/"profile.pstats")
                checkpoint(report, result)
                print("DONE", label, "energy_s", record["energy_seconds"], flush=True)
                return worker, document

            for case in ("water", "peptide"):
                aliases = output/(case+"-aliases")
                aliases.mkdir(mode=0o700)
                config_path = output/(case+"-runtime-config.json")
                save(config_path, case_policy(base, case, aliases))
                cpu_path, cpu = child(case, "cpu", config_path)
                gpu_path, gpu = child(case, "gpu", config_path)
                oracle_path, oracle = child(case, "cpu", config_path, oracle=gpu_path)
                result["cases"][case] = {
                    "status": "PASS_SINGLE_DIAGNOSTIC_AND_INDEPENDENT_CPU_ORACLE",
                    "cpu_result": str(cpu_path/"result.json"),
                    "gpu_result": str(gpu_path/"result.json"),
                    "oracle_result": str(oracle_path/"result.json"),
                    "energy_check": compare_energy(cpu, gpu),
                    "component_checks": compare(cpu, gpu),
                    "same_gauge_fields": oracle["same_gauge_compared_fields"],
                    "same_gauge_errors": oracle["same_gauge_errors"],
                }
                checkpoint(report, result)
                if case == "peptide" and args.profile_peptide:
                    profiled_path, _ = child(case, "gpu", config_path, profiled=True)
                    result["cases"][case]["separate_profile_repeat"] = compare_repeat(
                        gpu_path, profiled_path,
                        dimensions=gpu["orbital_dimensions"], arm="gpu")
                    checkpoint(report, result)
            result["status"] = "PASS_SMALL_RTX3090_DRIVER_DIAGNOSTICS_NOT_H200_SPEEDUP"
    except BaseException as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        checkpoint(report, result)


if __name__ == "__main__":
    main()
