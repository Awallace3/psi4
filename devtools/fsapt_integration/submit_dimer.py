"""Submit one CPU-only and one H200 standalone dimer diagnostic on embers."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from dimer_energy_probe import INPUT_SHA
from phoenix_dimer import resources
from phoenix_timing import clean, digest, save


def command(helper, directory, backend):
    policy = resources(backend)
    argv = [
        "sbatch", "--parsable", "--job-name=dimer-energy-" + backend,
        "--account=gts-cs207-chemx", "--qos=embers",
        "--partition=" + policy["partition"], "--constraint=" + policy["constraint"],
        "--nodes=1", "--ntasks=1", "--cpus-per-task=" + str(policy["threads"]),
        "--mem=" + str(policy["memory_GiB"]) + "G",
        "--time=" + ("00:30:00" if backend == "gpu" else "01:00:00"),
        "--no-requeue", "--signal=TERM@120", "--export=ALL",
        "--chdir=" + str(directory),
        "--output=" + str(directory / "logs/%x-%j.out"),
        "--error=" + str(directory / "logs/%x-%j.err"),
    ]
    if backend == "gpu":
        argv.append("--gres=gpu:h200:1")
    return argv + [str(helper / "devtools/fsapt_integration/phoenix_timing.sbatch")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=("energy", "fixed-jk", "trace-energy", "replay-trace",
                                          "beta-energy"), default="energy")
    parser.add_argument("--beta", type=int, choices=(0, 1))
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reference-sha256")
    parser.add_argument("--trace-sha256")
    parser.add_argument("--backends", nargs="+", choices=("cpu", "gpu"), default=("cpu", "gpu"))
    args = parser.parse_args()
    if len(set(args.backends)) != len(args.backends):
        parser.error("Select each backend at most once")
    if args.mode == "beta-energy":
        if args.beta is None or args.backends != ["gpu"]:
            parser.error("Beta-energy requires --beta 0|1 and --backends gpu")
    elif args.beta is not None:
        parser.error("--beta only applies to beta-energy")
    helper = Path(__file__).resolve().parents[2]
    helper_pin = subprocess.check_output(
        ["git", "-C", str(helper), "rev-parse", "HEAD"], text=True).strip()
    clean(helper, helper_pin)
    base = json.loads(args.config.resolve(strict=True).read_text())
    source = Path(base["source"])
    clean(source, base["source_pin"])
    sys.path.insert(0, str(source / "devtools/disp0/phoenix_validation"))
    from stage_scaling import require_cpu_runtime_admission
    admission = require_cpu_runtime_admission(base)
    inputs = args.inputs.resolve(strict=True)
    if digest(inputs) != INPUT_SHA:
        raise RuntimeError("Wrong frozen input manifest")
    reference_fields = {"beta": args.beta} if args.mode == "beta-energy" else {}
    if args.mode == "fixed-jk":
        if args.reference is None or args.reference_sha256 is None:
            parser.error("Fixed-JK mode requires a frozen CPU reference and SHA256")
        from dimer_fixed_jk import reference
        ref = args.reference.resolve(strict=True)
        reference(ref, args.reference_sha256)
        reference_fields = dict(reference=str(ref), reference_sha256=args.reference_sha256)
    if args.mode == "replay-trace":
        if args.reference is None or args.reference_sha256 is None or args.trace_sha256 is None:
            parser.error("Trace replay requires frozen result and trace hashes")
        from dimer_trace_replay import validate_reference
        ref = args.reference.resolve(strict=True)
        validate_reference(ref, args.reference_sha256, args.trace_sha256)
        reference_fields = dict(reference=str(ref), reference_sha256=args.reference_sha256,
                                trace_sha256=args.trace_sha256)
    if not args.root.is_absolute():
        raise ValueError("Absolute new run directory required")
    root = args.root
    root.mkdir(mode=0o700)
    aliases = root / "sealed-workers"
    aliases.mkdir(mode=0o700)
    python = Path(base["dependency_environment"]) / "bin/python"
    if not python.is_file():
        raise RuntimeError("Pinned dependency Python missing")
    env_base = {k: v for k, v in os.environ.items() if not k.startswith("SBATCH_")}
    for backend in args.backends:
        directory = root / backend
        directory.mkdir()
        (directory / "logs").mkdir()
        config = {**base, "threads": resources(backend)["threads"],
                  "sealed_alias_parent": str(aliases)}
        config_path = directory / "runtime-config.json"
        save(config_path, config)
        argv = command(helper, directory, backend)
        save(directory / "manifest.json", {
            **reference_fields, "mode": args.mode,
            "science_source": str(source), "science_pin": base["source_pin"],
            "helper_source": str(helper), "helper_pin": helper_pin,
            "config": str(config_path), "config_sha256": digest(config_path),
            "inputs": str(inputs), "inputs_sha256": INPUT_SHA,
            "backend": backend, "cpu_admission": admission,
            "resources": resources(backend), "command": argv,
            "scope": ("PROTEIN83_ZERO_OMEGA_BETA_ONLY_SCF_CONTROL"
                      if args.mode == "beta-energy" else
                      "GPU_TRAJECTORY_MEMDFJK_VS_CUEST_REPLAY_NO_SCF"
                      if args.mode == "replay-trace" else
                      "FIXED_CPU_DIMER_COEFFICIENT_JK_REPEATS_NO_SCF"
                      if args.mode == "fixed-jk" else
                      "STANDALONE_DIMER_HF_ENERGY_TRACE_NOT_TIMING" if args.mode == "trace-energy"
                      else "ONE_FRESH_PROCESS_STANDALONE_DIMER_HF_ENERGY"),
            "native_source_overlay": False,
            "sanitizers": "OUTSTANDING_LOADER_INCOMPATIBILITY_NO_NEW_KERNELS",
        })
        test = subprocess.run([argv[0], "--test-only", *argv[1:]], env=env_base,
                              text=True, capture_output=True, timeout=60)
        save(directory / "test-only.json", vars(test))
        print(backend, test.stdout + test.stderr, flush=True)
        if test.returncode or resources(backend)["partition"] not in test.stdout + test.stderr:
            raise RuntimeError("Route rejected before any submissions")
    submissions = dict(status="SUBMITTING", jobs={})
    save(root / "submissions.json", submissions)
    for backend in args.backends:
        directory = root / backend
        env = {**env_base, "FSAPT_SOURCE": str(helper), "FSAPT_PYTHON": str(python),
               "FSAPT_RUN_ROOT": str(directory),
               "FSAPT_MANIFEST_SHA256": digest(directory / "manifest.json"),
               "FSAPT_DRIVER": str(helper / "devtools/fsapt_integration/phoenix_dimer.py"),
               "FSAPT_THREADS": str(resources(backend)["threads"])}
        try:
            (directory / "SUBMISSION_ATTEMPTED").write_text("No automatic resubmission.\n")
            result = subprocess.run(command(helper, directory, backend), env=env,
                                    text=True, capture_output=True, timeout=60)
            save(directory / "submission.json", vars(result))
            job = result.stdout.strip().split(";")[0]
            if result.returncode or not job.isdecimal():
                raise RuntimeError("Failed/ambiguous submission: " + result.stdout + result.stderr)
            (directory / "job-id.txt").write_text(job + "\n")
            submissions["jobs"][backend] = job
            save(root / "submissions.json", submissions)
            print("SUBMITTED", backend, job, flush=True)
        except BaseException as error:
            submissions.update(status="PARTIAL_OR_AMBIGUOUS", failed_backend=backend,
                               error=str(error))
            save(root / "submissions.json", submissions)
            raise
    submissions["status"] = "SUBMITTED"
    save(root / "submissions.json", submissions)


if __name__ == "__main__":
    main()
