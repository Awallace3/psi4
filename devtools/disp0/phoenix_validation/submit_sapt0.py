"""One explicit embers H200 timing job after a successful coherent build."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from sapt0_timing import SYSTEMS, save, sha
from submit import clean_environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--pin", required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--system", choices=SYSTEMS, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--hours", type=int, choices=(2, 4, 8), default=2)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    run = args.run_root
    if not run.is_absolute():
        raise ValueError("Absolute fresh run root required")
    if subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"],
                               text=True).strip() != args.pin:
        raise ValueError("Source is not pinned")
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"],
                               text=True).strip():
        raise ValueError("Source is dirty")
    build_root, inputs = args.build_root.resolve(strict=True), args.inputs.resolve(strict=True)
    job = (build_root / "job-id.txt").read_text().strip()
    build_run = build_root / ("job-" + job)
    status = (build_run / "job-status.env").read_text()
    if "exit_code=0\n" not in status or "term_received=0\n" not in status:
        raise RuntimeError("Build failed or was interrupted")
    proof_path = build_run / "result.json"
    proof = json.loads(proof_path.read_text())
    if proof["status"] != "PASS":
        raise RuntimeError("Build qualification missing")
    # Helper-only evolution is allowed; scientific code must match the native build.
    subprocess.run(["git", "-C", str(source), "diff", "--exit-code", proof["pin"],
                    args.pin, "--", "psi4/"], check=True)
    input_hashes = {name: sha(inputs / (name + ".json")) for name in ("water", args.system)}
    env = clean_environment(os.environ)
    env.update(SAPT0_SOURCE=str(source), SAPT0_PIN=args.pin, SAPT0_BUILD_ROOT=str(build_root),
               SAPT0_INPUTS=str(inputs), SAPT0_SYSTEM=args.system, SAPT0_RUN_ROOT=str(run),
               SAPT0_PYTHON=proof["dependency_environment"] + "/bin/python")
    run.mkdir(mode=0o700)
    (run / "logs").mkdir()
    argv = ["sbatch", "--parsable", "--job-name=sapt0-" + args.system,
            "--account=gts-cs207-chemx", "--partition=gpu-h200", "--qos=embers",
            "--nodes=1", "--ntasks=1", "--cpus-per-task=8", "--gres=gpu:h200:1",
            "--mem=160G", f"--time={args.hours:02}:00:00", "--exclusive",
            "--no-requeue", "--signal=TERM@90", "--chdir=" + str(run),
            "--output=" + str(run / "logs/%x-%j.out"),
            "--error=" + str(run / "logs/%x-%j.err"),
            "--export=ALL", str(source / "devtools/disp0/phoenix_validation/sapt0_job.sbatch")]
    save(run / "manifest.json", {
        "source_pin": args.pin, "source": str(source), "build_proof": str(proof_path),
        "build_proof_sha256": sha(proof_path), "native_pin": proof["pin"],
        "input_hashes": input_hashes, "command": argv, "qos": "embers",
        "system": args.system, "cpu_threads_each_arm": 8,
        "repeats_each_arm": 3, "gate": "water CPU/GPU first",
        "environment": {k: v for k, v in env.items() if k.startswith("SAPT0_")}})
    test = subprocess.run([argv[0], "--test-only", *argv[1:]], env=env, capture_output=True,
                          text=True, timeout=60)
    save(run / "test-only.json", dict(returncode=test.returncode, stdout=test.stdout, stderr=test.stderr))
    print(test.stdout + test.stderr, flush=True)
    if test.returncode:
        raise RuntimeError("Exact route rejected; nothing submitted")
    with (run / "SUBMISSION_ATTEMPTED").open("x") as stream:
        stream.write("One scheduler call. Do not retry an ambiguous submission.\n")
    submitted = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=60)
    save(run / "submission.json", dict(returncode=submitted.returncode,
                                     stdout=submitted.stdout, stderr=submitted.stderr))
    print(submitted.stdout + submitted.stderr, flush=True)
    if submitted.returncode:
        raise RuntimeError("Submission failed/ambiguous; do not retry")
    job = submitted.stdout.strip().split(";")[0]
    if not job.isdecimal():
        raise RuntimeError("Unknown scheduler response")
    with (run / "job-id.txt").open("x") as stream:
        stream.write(job + "\n")
    print("SUBMITTED_JOB=" + job, flush=True)


if __name__ == "__main__":
    main()
