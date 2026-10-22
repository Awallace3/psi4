"""One-shot Phoenix admission/submission; UNKNOWN is never an automatic retry."""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import shlex
import subprocess


def durable(path, text):
    with path.open("x") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def accepted_id(text):
    if not re.fullmatch(r"[1-9][0-9]*(?:;[A-Za-z0-9_.-]+)?\n?", text):
        raise ValueError("Ambiguous sbatch receipt; do not resubmit")
    return text.strip().split(";")[0]


def main():
    parser = argparse.ArgumentParser()
    for name in ("source", "pin", "harness", "harness-pin", "env", "run"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    source, harness, run = (Path(s).resolve() for s in (args.source, args.harness, args.run))
    assert all(p.is_dir() for p in (source, harness, run))
    for root, pin in ((source, args.pin), (harness, args.harness_pin)):
        assert re.fullmatch("[0-9a-f]{40}", pin)
        assert subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip() == pin
        assert not subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True)
    attempt = run / "submission"
    attempt.mkdir()  # No restart, even after timeout/unknown delivery.
    queue = subprocess.run(["squeue", "-u", os.environ["USER"], "-h", "-r",
                            "-o", "%i|%j|%q|%T|%b"], capture_output=True, text=True, timeout=20)
    durable(attempt / "queue.stdout", queue.stdout)
    durable(attempt / "queue.stderr", queue.stderr)
    queue.check_returncode()
    rows = [line.split("|") for line in queue.stdout.splitlines() if line.strip()]
    assert all(len(row) == 5 for row in rows)
    assert sum(row[2] == "embers" for row in rows) < 50
    # Only this owner's campaign. Other agents' resources remain untouched.
    assert not any("fdds" in row[1].lower() and "gpu" in row[4].lower() for row in rows), rows
    logs = run / "logs"
    logs.mkdir()
    payload = ["bash", str(harness / "devtools/benchmarks/fdds_canonical_job.sh"),
               str(source), args.pin, str(harness), args.harness_pin, args.env, str(run)]
    command = ["sbatch", "--parsable", "--account=gts-cs207-chemx", "--qos=embers",
               "--partition=gpu-h200", "--gres=gpu:h200:1", "--nodes=1", "--ntasks=1",
               "--cpus-per-task=8", "--mem=128G", "--time=04:00:00", "--no-requeue",
               "--signal=TERM@120", "--job-name=fdds-canonical-0250",
               "--comment=fdds-3a7820-canonical-20261022-r1",
               "--chdir=" + str(run), "--output=" + str(logs / "%x-%j.out"),
               "--error=" + str(logs / "%x-%j.err"),
               "--wrap=" + shlex.join(["srun", "--exclusive", "--exact", *payload])]
    durable(attempt / "command.json", json.dumps(command, indent=2) + "\n")
    dry = subprocess.run([*command[:1], "--test-only", *command[1:]], capture_output=True, text=True, timeout=30)
    durable(attempt / "dry-run.stdout", dry.stdout)
    durable(attempt / "dry-run.stderr", dry.stderr)
    dry.check_returncode()
    assert "gpu-h200" in dry.stdout + dry.stderr, "Dry-run did not prove H200 route"
    durable(attempt / "SUBMISSION_INVOKED", datetime.datetime.now(datetime.timezone.utc).isoformat() + "\n")
    # A timeout here leaves the invocation marker, no repeat path and no made-up ID.
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    durable(attempt / "sbatch.stdout", result.stdout)
    durable(attempt / "sbatch.stderr", result.stderr)
    result.check_returncode()
    job = accepted_id(result.stdout)
    durable(attempt / "job-id.txt", job + "\n")
    print("ACCEPTED_JOB_ID=" + job, flush=True)
    state = subprocess.run(["scontrol", "show", "job", "-o", job], capture_output=True, text=True, timeout=15)
    durable(attempt / "effective-job.stdout", state.stdout)
    durable(attempt / "effective-job.stderr", state.stderr)
    state.check_returncode()
    assert "QOS=embers" in state.stdout and "Requeue=0" in state.stdout
    assert "NumCPUs=8" in state.stdout and "MinMemoryNode=128G" in state.stdout
    assert "gpu:h200:1" in state.stdout.lower()
    durable(attempt / "receipt.json", json.dumps({"job_id": int(job), "source_pin": args.pin,
            "harness_pin": args.harness_pin, "qos": "embers", "run_directory": str(run)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
