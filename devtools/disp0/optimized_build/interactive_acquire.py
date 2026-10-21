"""Acquire ONE bounded CPU allocation. No build, retry, release or GPU request."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

import submit as admission

WAIT_SECONDS = 600


def accepted_id(line):
    match = re.search(r"salloc: (?:Pending job allocation|Granted job allocation) ([1-9][0-9]*)\b", line)
    return match.group(1) if match else None


def validate_controller(text, job):
    fields = dict(token.split("=", 1) for token in text.split() if "=" in token)
    expected = {"JobId": job, "Account": "gts-cs207-chemx", "QOS": "embers",
                "JobState": "RUNNING", "Requeue": "0", "Restarts": "0",
                "NumCPUs": "8", "NumTasks": "1", "TimeLimit": "02:00:00",
                "Partition": "cpu-small", "MinMemoryNode": "48G", "BatchFlag": "0"}
    admission.require(all(fields.get(k) == v for k, v in expected.items()),
                      f"Wrong effective interactive CPU allocation: {fields}")
    admission.require(fields.get("NodeList") not in (None, "", "(null)")
                      and "gres/gpu" not in fields.get("ReqTRES", ""), "No CPU-only grant proof")
    return fields


def acquire(source, pin, attempt, quota, quota_sha):
    _, intent, environment = admission.preflight(
        source, pin, attempt, quota, quota_sha, interactive=True)
    # salloc has no --no-requeue option. Interactive allocations have no batch
    # script to replay; effective Requeue=0/Restarts=0 is mandatory before GO.
    resources = [value for value in admission.RESOURCES if value != "--no-requeue"]
    argv = ["salloc", "--no-shell", f"--immediate={WAIT_SECONDS}",
            "--job-name=disp0-opt064-i", *resources]
    admission.record(attempt, "dispatch-intent.json",
                     {"argv": argv, "timestamp_unix": time.time(),
                      "queue_wait_seconds": WAIT_SECONDS, "no_retry": True,
                      "release": "NO automatic release/job changes by this acquisition helper"})
    job = None
    # One noninteractive invocation. Every received line is durable before
    # acceptance is printed or any controller is queried.
    with (attempt / "salloc.raw").open("xb") as raw:
        admission.fsync_dir(attempt)
        process = subprocess.Popen(argv, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        for line in process.stdout:
            admission.require(len(line.encode()) <= 16384, "Oversized allocation response; UNKNOWN")
            raw.write(line.encode())
            raw.flush()
            os.fsync(raw.fileno())
            candidate = accepted_id(line)
            if candidate:
                admission.require(job in (None, candidate), "Conflicting allocation IDs; UNKNOWN")
                if job is None:
                    job = candidate
                    admission.durable(attempt / "accepted-job-id.txt", (job + "\n").encode())
            print(line, end="", flush=True)
        status = process.wait()
    admission.record(attempt, "acquisition-exit.json", {"exit": status, "accepted_id": job})
    admission.require(status == 0 and job is not None, "No verified grant; NEVER replay acquisition")
    response = admission.capture(["scontrol", "show", "job", job, "-o"], timeout=10)
    admission.record(attempt, "controller.json", response)
    fields = validate_controller(admission.checked(response), job)
    admission.record(attempt, "allocation.json",
                     {"status": "ALLOCATED_CPU_ONLY", "job_id": job, "fields": fields,
                      "source": str(source), "recipe_pin": pin, "intent": intent,
                      "scientific_readiness": False, "gpu_correctness": False})
    print(json.dumps({"status": "ALLOCATED_CPU_ONLY", "job_id": job,
                      "manifest": str(attempt / "allocation.json"), "fields": fields}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--pin", required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--quota-proof", type=Path, required=True)
    parser.add_argument("--quota-sha256", required=True)
    args = parser.parse_args()
    os.umask(0o077)
    acquire(args.source, args.pin, args.attempt, args.quota_proof, args.quota_sha256)


if __name__ == "__main__":
    main()
