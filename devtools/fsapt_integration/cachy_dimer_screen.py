"""Bounded fresh-process small SCF screen using an existing authenticated install."""
import argparse
import json
import os
from pathlib import Path
import pwd
import sys
import time


def worker_alias(parent, label):
    directory = parent / label
    directory.mkdir(mode=0o700)
    return directory


def dedicated_root():
    # Scientific workers deliberately have a private HOME. Workspace identity
    # belongs to the account, not to that per-process configuration sandbox.
    return Path(pwd.getpwuid(os.getuid()).pw_dir) / "gits/psi4-small-dimer-cachy"


def case_specs(mode, replay_only=False):
    if mode == "plans":
        cases = [("retained", "peptide", "gpu", "energy"),
                 ("cpu-replay", "peptide", "cpu", "energy"),
                 ("gpu-replay", "peptide", "gpu", "energy")]
        return cases[1:] if replay_only else cases
    if mode == "beta-scf" and not replay_only:
        return [(label, "peptide", "gpu", "energy")
                for label in ("beta1-0", "beta0-0", "beta0-1", "beta1-1")]
    if mode == "beta-entrypoints" and not replay_only:
        return [(f"beta{beta}-{entrypoint}-{repeat}", "peptide", "gpu", entrypoint)
                for repeat, betas, entries in ((0, (1, 0), ("energy", "sapt")),
                                               (1, (0, 1), ("sapt", "energy")))
                for beta in betas for entrypoint in entries]
    if mode != "screen":
        raise ValueError("Explicit diagnostic mode required")
    if replay_only:
        raise ValueError("Replay continuation requires plans mode")
    return [(f"{system}-{backend}-{entrypoint}-{repeat}", system, backend, entrypoint)
            for system in ("hcl", "peptide") for backend in ("cpu", "gpu")
            for repeat, order in enumerate((("energy", "sapt"), ("sapt", "energy")))
            for entrypoint in order]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runtime-source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--backend", choices=("cpu", "gpu"))
    parser.add_argument("--entrypoint", choices=("energy", "sapt"))
    parser.add_argument("--system", choices=("hcl", "peptide"))
    parser.add_argument("--mode", choices=("screen", "plans", "beta-scf", "beta-entrypoints"),
                        default="screen")
    parser.add_argument("--beta", type=int, choices=(0, 1))
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reference-sha256")
    args = parser.parse_args()
    if bool(args.reference) != bool(args.reference_sha256):
        parser.error("Reference and its hash must be supplied together")
    if args.reference and args.mode != "plans":
        parser.error("Reference is only valid for plans mode")
    beta_mode = args.mode in ("beta-scf", "beta-entrypoints")
    if args.worker and beta_mode and args.beta is None:
        parser.error("Beta-SCF worker requires an explicit beta value")
    root = args.root.resolve(strict=True)
    if root != dedicated_root():
        raise RuntimeError("Dedicated small-dimer workspace required")
    source = args.runtime_source.resolve(strict=True)
    sys.path[:0] = [str(source / "devtools/disp0/cachy"),
                    str(source / "devtools/disp0/phoenix_validation")]
    from coherent_stage11 import clean, git, lease, save
    from gpu_runtime import read_config, digest
    from response_reuse_probe import resource_scope

    scope = resource_scope()
    helper = Path(__file__).resolve().parents[2]
    helper_pin = git(helper, "rev-parse", "HEAD")
    clean(helper, helper_pin)
    output = args.output.resolve()
    if not output.is_relative_to(root):
        raise RuntimeError("Output must remain in the dedicated workspace")
    if args.worker:
        from bootstrap_gate import bootstrap
        from gpu_runtime import AuditedRuntime
        config, build = read_config(args.config, args.config_sha256)
        owners, events = bootstrap(config, build)
        runtime = AuditedRuntime(args.config, args.config_sha256)
        if beta_mode:
            from peptide_plan_probe import beta_scf_run
            result = runtime.guarded_call(beta_scf_run, runtime.psi4, args.inputs, output,
                                          args.beta, args.entrypoint, runtime.provider)
        elif args.mode == "plans":
            from peptide_plan_probe import run
            result = runtime.guarded_call(run, runtime.psi4, args.inputs, output,
                                          args.backend, args.reference, args.reference_sha256)
        else:
            from small_dimer_probe import run
            result = runtime.guarded_call(
                run, runtime.psi4, args.inputs, output, args.backend,
                args.entrypoint, args.system, 4,
                runtime.provider if args.backend == "gpu" else None)
        if runtime.report["plans"]:
            raise RuntimeError("Dimer screen unexpectedly executed a fragment provider")
        result.update(science_pin=config["source_pin"], helper_pin=helper_pin,
                      runtime_config_sha256=args.config_sha256,
                      resource_scope=scope, bootstrap_events=events)
        save(output / "result.json", result)
        return

    from owned_process import execute
    from run_gate import environment
    from profile_sapt_steps import admit_device, device_state
    from small_dimer_probe import INPUT_SHA
    if digest(args.inputs) != INPUT_SHA:
        raise RuntimeError("Frozen input manifest required")
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="STARTED", mode=args.mode, helper_pin=helper_pin, resource_scope=scope,
                  workers=[], scope="SMALL_SCF_NUMERICAL_SCREEN_NOT_TIMING",
                  accepted_speedups=0)
    deadline = time.monotonic() + 2400
    try:
        with lease(root) as fd:
            base, build = read_config(args.config, args.config_sha256)
            aliases = output / "aliases"
            aliases.mkdir(mode=0o700)
            config_path = output / "runtime-config.json"
            save(config_path, {**base, "threads": 4, "host_memory": "6 GiB",
                               "sealed_alias_parent": str(aliases)})
            config_sha = digest(config_path)
            config, _ = read_config(config_path, config_sha)
            report.update(science_pin=config["source_pin"], stage=config["stage"],
                          runtime_config_sha256=config_sha)
            # Interleave entrypoints and reverse repeat order; each is a fresh
            # interpreter, never reusing a previous calculation's JK or scratch.
            if args.reference:
                if digest(args.reference) != args.reference_sha256:
                    raise RuntimeError("Completed producer changed before continuation")
                report.update(reference=str(args.reference), reference_sha256=args.reference_sha256)
            for label, system, backend, entrypoint in case_specs(
                    args.mode, replay_only=args.reference is not None):
                alias = worker_alias(aliases, label)
                scratch = output / (label + "-scratch")
                scratch.mkdir()
                env = environment(config, gpu=backend == "gpu",
                                  sealed_alias_directory=alias)
                env.update(TMPDIR=str(scratch), SCRATCH=str(scratch), PSI_SCRATCH=str(scratch))
                argv = ["/bin/bash", "-c",
                        'set -euo pipefail; export PYTHONPATH="${PYTHONPATH-}"; '
                        'eval "$("$1" --psiapi)"; shift; exec "$@"',
                        "small-dimer", config["stage"] + "/bin/psi4",
                        config["dependency_environment"] + "/bin/python", "-B",
                        str(Path(__file__).resolve()), "--root", str(root),
                        "--runtime-source", str(source), "--config", str(config_path),
                        "--config-sha256", config_sha, "--inputs", str(args.inputs),
                        "--output", str(output / label), "--worker",
                        "--backend", backend, "--entrypoint", entrypoint,
                        "--system", system, "--mode", args.mode]
                if beta_mode:
                    argv += ["--beta", label.split("-", 1)[0].removeprefix("beta")]
                if args.mode == "plans" and label != "retained":
                    reference = args.reference or output / "retained/result.json"
                    checksum = (args.reference_sha256 if args.reference
                                else report["workers"][0]["result_sha256"])
                    if digest(reference) != checksum:
                        raise RuntimeError("Retained-plan result changed before replay")
                    argv += ["--reference", str(reference), "--reference-sha256", checksum]
                row = dict(label=label, argv=argv)
                report["workers"].append(row)
                if backend == "gpu":
                    row["device_before"] = device_state(config["local_lease"]["expected_uuid"])
                    admit_device(row["device_before"], config)
                save(output / "screen-result.json", report)
                seconds = min(600, deadline - time.monotonic())
                if seconds < 60:
                    raise TimeoutError("Small-screen time budget exhausted")
                print("START", label, flush=True)
                with (output / (label + ".stdout")).open("x") as out, (
                        output / (label + ".stderr")).open("x") as err:
                    code = execute(argv, env, out, err, seconds, fd,
                                   config["local_lease"], cwd=scratch)
                row["returncode"] = code
                if code:
                    raise RuntimeError("Infrastructure/worker failure: " + label)
                result_path = output / label / "result.json"
                result = json.loads(result_path.read_text())
                row.update(result_sha256=digest(result_path), scientific_status=result["status"])
                allowed = (("EXECUTED_PLAN_REPLAY_DIAGNOSTIC_ONLY",) if args.mode == "plans"
                           else ("CONVERGED_DIAGNOSTIC_ONLY", "SCF_NONCONVERGENCE"))
                if result["status"] not in allowed:
                    raise RuntimeError("Incomplete scientific outcome")
                text = (output / label / "psi4.out").read_text()
                if backend == "gpu" and (
                        "Math mode: Native FP64" not in " ".join(text.split())
                        or result["jk_name"] != "cuESTJK"):
                    raise RuntimeError("Native FP64 cuEST execution not confirmed")
                print("DONE", label, result["status"], flush=True)
                # Preserve per-worker scratch for investigation.
                save(output / "screen-result.json", report)
            report["status"] = "EXECUTED_DIAGNOSTIC_ONLY_NOT_PARITY_GATE"
    except BaseException as error:
        report.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save(output / "screen-result.json", report)


if __name__ == "__main__":
    main()
