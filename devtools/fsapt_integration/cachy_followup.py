"""Bounded post-gate cachy diagnostics: peptide parity or water sanitizers.

Reuses the exact installed science and authenticated bootstrap. This outer
launcher is separately recorded, never overlaid on the scientific install.
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import time


def require_release(release, pin, config_sha):
    if (release.get("status") != "PASS_SMALL_CACHY_ONLY"
            or release.get("source_pin") != pin
            or release.get("runtime_config_sha256") != config_sha
            or not release.get("gpu_small_case_correctness_verified")
            or len(release.get("cases", [])) != 10
            or any(c.get("returncode") != 0 for c in release["cases"])):
        raise RuntimeError("Exact-source completed small GPU gate required")


def require_clean_sanitizer(text):
    counts = re.findall(r"ERROR SUMMARY:\s+(\d+) errors?", text)
    if not counts or any(int(count) for count in counts):
        raise RuntimeError("Missing or nonzero compute-sanitizer error summary")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pin", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--inputs-sha256")
    parser.add_argument("--mode", choices=("peptide", "sanitizers"), required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--scf-convergence", type=int, choices=(10, 12), default=10)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if (root != Path.home() / "gits/psi4-fsapt-cpks-integration-cachy"
            or not args.attempt.replace("-", "").isalnum()):
        raise ValueError("Explicit integration root and fresh simple attempt required")
    source = root / "source"
    sys.path[:0] = [str(source / "devtools/disp0/cachy"),
                    str(source / "devtools/disp0/phoenix_validation")]
    from coherent_stage11 import clean, git, lease, save
    from gpu_runtime import digest, read_config
    from response_reuse_probe import resource_scope
    from profile_sapt_steps import admit_device, case_policy, device_state
    from run_gate import compare, environment
    from run_full import compare_energy
    from repeat_gauge import compare_repeat
    from owned_process import execute
    from full_inputs import load_point
    from prepare_inputs import input_basis_profile

    scope = resource_scope()
    clean(source, args.pin)
    helper = Path(__file__).resolve().parents[2]
    helper_pin = git(helper, "rev-parse", "HEAD")
    clean(helper, helper_pin)
    if digest(args.release) != args.release_sha256:
        raise RuntimeError("Small gate receipt changed")
    release = json.loads(args.release.read_text())
    require_release(release, args.pin, args.config_sha256)
    output = root / "runs" / args.attempt
    output.mkdir(mode=0o700)
    report = {"status": "STARTED", "mode": args.mode, "science_pin": args.pin,
              "helper_pin": helper_pin, "launcher_sha256": digest(__file__),
              "resource_scope": scope, "release_sha256": args.release_sha256,
              "workers": [], "accepted_speedups": 0}
    deadline = time.monotonic() + 3000
    try:
        with lease(root) as fd:
            base, build = read_config(args.config, args.config_sha256)
            if base["source_pin"] != args.pin:
                raise RuntimeError("Installed science differs from selected pin")
            bootstrap = helper / "devtools/disp0/cachy/bootstrap_gate.py"
            gate = helper / "devtools/disp0/phoenix_validation/fragment_gate.py"
            for path in (bootstrap, gate):
                if build["provenance_sha256"].get(str(path)) != digest(path):
                    raise RuntimeError("Follow-up bootstrap/helper lacks matching CPU qualification")
            report["scf_convergence"] = args.scf_convergence
            config_path = args.config
            if args.mode == "peptide":
                if args.inputs is None or not args.inputs_sha256:
                    raise ValueError("Frozen practical input manifest required")
                _, dimensions = load_point(args.inputs, args.inputs_sha256, "peptide")
                if input_basis_profile(json.loads(args.inputs.read_text())) != "jun-cc-pvdz":
                    raise ValueError("Explicit matched jun-cc-pVDZ profile required")
                aliases = output / "sealed-workers"
                aliases.mkdir(mode=0o700)
                config_path = output / "runtime-config.json"
                save(config_path, case_policy(base, "peptide", aliases))
                report["inputs_sha256"] = args.inputs_sha256
            config_sha = digest(config_path)
            config, _ = read_config(config_path, config_sha)
            report["runtime_config_sha256"] = config_sha

            def child(label, arm, *, oracle=None, sanitizer=None):
                worker = output / label
                alias = Path(config["sealed_alias_parent"]) / (args.attempt + "-" + label)
                alias.mkdir(mode=0o700)
                env = environment(config, gpu=arm == "gpu", sealed_alias_directory=alias)
                scratch = output / (label + "-scratch")
                scratch.mkdir()
                env.update(TMPDIR=str(scratch), SCRATCH=str(scratch), PSI_SCRATCH=str(scratch))
                payload = [config["dependency_environment"] + "/bin/python", "-B",
                           str(bootstrap),
                           "--config", str(config_path), "--config-sha256", config_sha,
                           "--arm", arm, "--scaled", "--output", str(worker),
                           "--block-size", "24" if args.mode == "peptide" else "5",
                           "--scf-convergence", str(args.scf_convergence)]
                if args.mode == "peptide":
                    payload += ["--inputs", str(args.inputs), "--inputs-sha256",
                                args.inputs_sha256, "--system", "peptide"]
                if oracle:
                    payload += ["--oracle-from", str(oracle)]
                if sanitizer:
                    log = output / (sanitizer + ".log")
                    tool = ["/opt/cuda/bin/compute-sanitizer", "--tool", sanitizer,
                            "--error-exitcode", "9", "--target-processes", "all",
                            "--log-file", str(log)]
                    if sanitizer == "memcheck":
                        tool += ["--leak-check", "full"]
                    payload = tool + payload
                argv = ["/bin/bash", "-c",
                        'set -euo pipefail; export PYTHONPATH="${PYTHONPATH-}"; '
                        'eval "$("$1" --psiapi)"; shift; exec "$@"',
                        "fsapt-followup", config["stage"] + "/bin/psi4", *payload]
                record = {"label": label, "arm": arm, "argv": argv}
                report["workers"].append(record)
                if arm == "gpu":
                    record["device_before"] = device_state(config["local_lease"]["expected_uuid"])
                    admit_device(record["device_before"], config)
                save(output / "result.json", report)
                seconds = min(1200, deadline - time.monotonic())
                if seconds < 60:
                    raise TimeoutError("Bounded diagnostic deadline exhausted")
                print("START", label, flush=True)
                with (output / (label + ".stdout")).open("x") as out, (
                        output / (label + ".stderr")).open("x") as err:
                    code = execute(argv, env, out, err, seconds, fd, config["local_lease"], cwd=root)
                record["returncode"] = code
                if code:
                    raise RuntimeError("Worker failed: " + label)
                document = json.loads((worker / "result.json").read_text())
                if (document["status"] != "EXECUTED_REQUIRES_INDEPENDENT_COMPARISON"
                        or document["source_pin"] != args.pin
                        or document["runtime_config_sha256"] != config_sha):
                    raise RuntimeError("Incomplete/mismatched worker")
                for name, checksum in document["artifact_sha256"].items():
                    if digest(worker / name) != checksum:
                        raise RuntimeError("Changed worker artifact: " + name)
                record["result_sha256"] = digest(worker / "result.json")
                if sanitizer:
                    require_clean_sanitizer(log.read_text())
                    record["sanitizer_log_sha256"] = digest(log)
                print("DONE", label, flush=True)
                return worker, document

            if args.mode == "peptide":
                _, cpu = child("cpu", "cpu")
                gpu_path, gpu = child("gpu", "gpu")
                _, oracle = child("oracle", "cpu", oracle=gpu_path)
                report.update(energy_check=compare_energy(cpu, gpu),
                              signed_aggregate_checks=compare(cpu, gpu),
                              same_gauge_fields=oracle["same_gauge_compared_fields"],
                              same_gauge_errors=oracle["same_gauge_errors"])
                repeat, _ = child("gpu-repeat", "gpu")
                report["repeat"] = compare_repeat(gpu_path, repeat, dimensions=dimensions)
                report["status"] = "PASS_PEPTIDE_CACHY_ONLY"
            else:
                for tool in ("memcheck", "initcheck"):
                    worker, _ = child(tool, "gpu", sanitizer=tool)
                    report[tool + "_repeat"] = compare_repeat(
                        args.release.parent / "gpu-scaled", worker)
                report["status"] = "PASS_WATER_SANITIZERS_ONLY"
            clean(source, args.pin)
    except BaseException as error:
        report.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save(output / "result.json", report)
    print(report["status"], output, flush=True)


if __name__ == "__main__":
    main()
