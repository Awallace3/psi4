"""One fresh explicit-embers CPU submission; no hidden retry or GPU requests."""
import argparse
import json
import os
from pathlib import Path
import subprocess

P = Path("/storage/project/r-cs207-0/awallace43")
OLD = P / "runs/saptdft-cuest-disp0/opt064-770ecf90108aa7783da766bf0402d8c267dcbbed-14254131-0"


def clean_environment(environment):
    return {k: v for k, v in environment.items() if not k.startswith("SBATCH_")}


def command(source, run):
    return ["sbatch", "--parsable", "--job-name=disp0-current-build",
            "--account=gts-cs207-chemx", "--partition=cpu-small", "--qos=embers",
            "--nodes=1", "--ntasks=1", "--cpus-per-task=8", "--mem=48G",
            "--time=03:00:00", "--no-requeue", "--signal=TERM@90",
            "--chdir=" + str(run),
            "--output=" + str(run / "logs/%x-%j.out"),
            "--error=" + str(run / "logs/%x-%j.err"),
            "--export=ALL", str(source / "devtools/disp0/phoenix_validation/job.sbatch")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--pin", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    source, run = args.source.resolve(strict=True), args.run_root
    if not run.is_absolute():
        raise RuntimeError("Run root must be absolute")
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip()
    if actual != args.pin or dirty:
        raise RuntimeError("Submission source must be clean and exactly pinned")
    for path in (OLD / "environment/bin/python", OLD / "einsums-stage/lib/einsums",
                 P / "software/toolchains/cuda13/bin/nvcc",
                 P / "software/cuest-0.2.50-a06e8148/include/cuest.h"):
        if not path.exists():
            raise RuntimeError("Missing read-only dependency: " + str(path))
    run.mkdir(mode=0o700)  # Refuse reusing ANY previous attempt.
    (run / "logs").mkdir()
    env = clean_environment(os.environ)
    env.update(DISP0_SOURCE=str(source), DISP0_PIN=args.pin, DISP0_RUN_ROOT=str(run),
               DISP0_DEPENDENCIES=str(OLD / "environment"),
               DISP0_E2_PREFIX=str(OLD / "einsums-stage"),
               DISP0_CUDA_SOURCE=str(P / "software/toolchains/cuda13"),
               DISP0_CUEST_ROOT=str(P / "software/cuest-0.2.50-a06e8148"))
    argv = command(source, run)
    manifest = {"scope": "Fresh native cuEST-enabled Psi4 build and CPU validation only",
                "source_pin": args.pin, "source": str(source), "command": argv,
                "read_only_inputs": {k: v for k, v in env.items() if k.startswith("DISP0_")},
                "qos": "embers", "account": "gts-cs207-chemx", "gpu_count": 0,
                "cpu_count": 8, "memory_GiB": 48, "time_limit": "03:00:00"}
    with (run / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
    test = subprocess.run([argv[0], "--test-only", *argv[1:]], env=env,
                          capture_output=True, text=True, timeout=60)
    with (run / "test-only.json").open("x") as stream:
        json.dump({"returncode": test.returncode, "stdout": test.stdout,
                   "stderr": test.stderr}, stream, indent=2)
    print(test.stdout + test.stderr, flush=True)
    if test.returncode:
        raise RuntimeError("Exact-resource test-only failed; nothing submitted")
    # Durable reservation BEFORE scheduler call. Never retry an ambiguous send.
    with (run / "SUBMISSION_ATTEMPTED").open("x") as stream:
        stream.write("Exactly one scheduler call; inspect queue before any recovery.\n")
    result = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=60)
    with (run / "submission.json").open("x") as stream:
        json.dump({"returncode": result.returncode, "stdout": result.stdout,
                   "stderr": result.stderr}, stream, indent=2)
    print(result.stdout + result.stderr, flush=True)
    if result.returncode:
        raise RuntimeError("Submission failed or ambiguous; do NOT retry automatically")
    job = result.stdout.strip().split(";")[0]
    if not job.isdecimal():
        raise RuntimeError("Unparseable job ID; inspect saved scheduler response")
    with (run / "job-id.txt").open("x") as stream:
        stream.write(job + "\n")
    print("SUBMITTED_JOB=" + job, flush=True)
    print("OUTPUT_ROOT=" + str(run), flush=True)


if __name__ == "__main__":
    main()
