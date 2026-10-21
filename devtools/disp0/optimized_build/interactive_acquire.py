"""Acquire ONE bounded CPU allocation. No build, retry, release or GPU request."""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import time

import submit as admission

WAIT_SECONDS = 600


def validate_tres(value):
    admission.require(isinstance(value, str) and value and value != "(null)", "Missing TRES proof")
    fields = {}
    for token in value.split(","):
        pair = token.split("=", 1)
        admission.require(len(pair) == 2 and pair[0] and pair[1] and pair[0] not in fields,
                          "Malformed/duplicate TRES proof")
        fields[pair[0]] = pair[1]
    admission.require(not any("gpu" in key.lower() or key.startswith(("gres/mps", "gres/shard"))
                              for key in fields), "GPU resource in CPU allocation")
    admission.require(fields.get("cpu") == "8" and fields.get("node") == "1", "Wrong CPU/node TRES")
    memory = re.fullmatch(r"([0-9]+)([KMGT]?)", fields.get("mem", ""))
    admission.require(memory is not None, "Malformed memory TRES")
    factor = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}[memory.group(2)]
    admission.require(int(memory.group(1)) * factor == 48 * 1024**3, "Wrong memory TRES")
    return fields


def accepted_id(line):
    match = re.search(r"salloc: (?:Pending job allocation|Granted job allocation) ([1-9][0-9]*)\b", line)
    return match.group(1) if match else None


def validate_controller(text, job, *, before_approved_correction=False):
    fields = dict(token.split("=", 1) for token in text.split() if "=" in token)
    expected = {"JobId": job, "Account": "gts-cs207-chemx", "QOS": "embers",
                "JobName": "disp0-opt064-i",
                "JobState": "RUNNING", "Restarts": "0",
                "NumCPUs": "8", "NumTasks": "1", "NumNodes": "1", "TimeLimit": "02:00:00",
                "Partition": "cpu-small", "MinMemoryNode": "48G", "BatchFlag": "0"}
    admission.require(all(fields.get(k) == v for k, v in expected.items()),
                      f"Wrong effective interactive CPU allocation: {fields}")
    owner = re.fullmatch(r"[^()]+\(([0-9]+)\)", fields.get("UserId", ""))
    admission.require(owner is not None and int(owner.group(1)) == os.getuid(),
                      "Allocation is not owned by this user")
    allowed = ("0", "1") if before_approved_correction else ("0",)
    admission.require(fields.get("Requeue") in allowed, "Wrong effective requeue policy")
    admission.require(fields.get("NodeList") not in (None, "", "(null)"), "No node grant proof")
    validate_tres(fields.get("ReqTRES"))
    validate_tres(fields.get("AllocTRES"))
    times = {}
    for key in ("StartTime", "EndTime"):
        value = fields.get(key, "")
        admission.require(re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}",
                                       value) is not None, f"Missing/invalid {key} proof")
        try:
            times[key] = datetime.datetime.fromisoformat(value)
        except ValueError as exc:
            raise RuntimeError(f"Invalid {key} calendar value") from exc
    admission.require(times["EndTime"] > times["StartTime"], "Invalid allocation time interval")
    return fields


def acquire(source, pin, attempt, quota, quota_sha, *, correct_requeue=False, approval_reference=None):
    admission.require(not correct_requeue or approval_reference,
                      "Explicit new-own-ID flag correction approval required")
    _, intent, environment = admission.preflight(
        source, pin, attempt, quota, quota_sha, interactive=True)
    if correct_requeue:
        admission.record(attempt, "flag-correction-approval.json",
                         {"reference": approval_reference, "scope": "ONE Requeue=0 on this NEW own ID only"})
    # salloc has no --no-requeue option. Interactive allocations have no batch
    # script to replay; effective Requeue=0/Restarts=0 is mandatory before GO.
    resources = [value for value in admission.RESOURCES if value != "--no-requeue"]
    argv = ["salloc", "--no-shell", f"--immediate={WAIT_SECONDS}",
            "--job-name=disp0-opt064-i", *resources]
    admission.record(attempt, "dispatch-intent.json",
                     {"argv": argv, "timestamp_unix": time.time(),
                      "queue_wait_seconds": WAIT_SECONDS, "no_retry": True,
                      "release": "NO automatic release; only explicitly approved own-ID flag correction"})
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
    fields = validate_controller(admission.checked(response), job,
                                 before_approved_correction=correct_requeue)
    if fields["Requeue"] == "1":
        admission.require(correct_requeue and approval_reference, "No flag update authority")
        argv = ["scontrol", "update", f"JobId={job}", "Requeue=0"]
        admission.record(attempt, "flag-update-intent.json",
                         {"argv": argv, "approval": approval_reference, "timestamp_unix": time.time()})
        update = admission.capture(argv, timeout=10)
        admission.record(attempt, "flag-update-response.json", update)
        admission.checked(update)
        after = admission.capture(["scontrol", "show", "job", job, "-o"], timeout=10)
        admission.record(attempt, "controller-after-flag.json", after)
        final = validate_controller(admission.checked(after), job)
        for key in ("Account", "QOS", "Partition", "NumCPUs", "NumTasks", "NumNodes",
                    "MinMemoryNode", "TimeLimit", "ReqTRES", "AllocTRES", "StartTime",
                    "EndTime", "NodeList", "Restarts"):
            admission.require(final[key] == fields[key], f"Unexpected allocation change: {key}")
        fields = final
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
    parser.add_argument("--correct-requeue", action="store_true")
    parser.add_argument("--approval-reference")
    args = parser.parse_args()
    os.umask(0o077)
    acquire(args.source, args.pin, args.attempt, args.quota_proof, args.quota_sha256,
            correct_requeue=args.correct_requeue, approval_reference=args.approval_reference)


if __name__ == "__main__":
    main()
