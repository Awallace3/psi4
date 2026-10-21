"""One-shot CPU build admission/submission. Existing attempts ALWAYS reject.

Invoke as one noninteractive configured-SSH command, never a persistent stdin
session across model review. No submission retry, including after reply loss.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

P = Path("/storage/project/r-cs207-0/awallace43")
SCIENCE = "587e172a33afaadb00eb7f574a5089533f7fc755"
E2 = "06404fa69ce9346443d7d4f19a250f6db9f0f62c"
LOCK_SHA = "3f9a508fd601306f548a9088f5cd304367a5fde3988e8bbd4fe00d9e6efb0205"
RESOURCES = ["--account=gts-cs207-chemx", "--partition=cpu-small", "--qos=embers",
             "--nodes=1", "--ntasks=1", "--cpus-per-task=8", "--mem=48G",
             "--time=02:00:00", "--no-requeue", "--gres=none"]


def scheduler_environment():
    # SBATCH_* overrides script directives. Neither test-only nor the actual
    # RPC may inherit another experiment's paid QoS/resources.
    return {name: value for name, value in os.environ.items() if not name.startswith("SBATCH_")}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def durable(path, data):
    with open(path, "xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    fsync_dir(Path(path).parent)


def record(attempt, name, value):
    durable(attempt / name, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def capture(argv, *, timeout=15, env=None):
    result = subprocess.run(argv, capture_output=True, timeout=timeout, env=env)
    return {"argv": argv, "exit": result.returncode,
            "stdout": result.stdout.decode(errors="replace"),
            "stderr": result.stderr.decode(errors="replace")}


def checked(result):
    require(result["exit"] == 0, f"Command failed: {result}")
    return result["stdout"].strip()


def quota_admission(proof, expected_sha, now):
    require(sha(proof) == expected_sha, "Quota evidence changed")
    value = json.loads(Path(proof).read_text())
    require(0 <= now - value["timestamp_unix"] <= 600, "Quota proof stale/future")
    require(value["scope"] == "HUMAN_REVIEWED_PROJECT_AND_SCRATCH_QUOTA", "Wrong quota proof scope")
    for item in ("project", "scratch"):
        require(value[item]["free_gib"] >= 64 and value[item]["free_inodes"] >= 50000,
                f"Insufficient {item} quota")
        raw = Path(value[item]["raw_path"])
        require(raw.is_relative_to(proof.parent) and sha(raw) == value[item]["raw_sha256"],
                "Raw quota proof must be retained and hash-bound")
    return value


def submit(source, pin, attempt, quota, quota_sha):
    source, attempt, quota = Path(source), Path(attempt), Path(quota)
    require(re.fullmatch("[0-9a-f]{40}", pin), "Full immutable recipe pin required")
    require(source.is_absolute() and source.resolve() == source, "Canonical source required")
    require(source.is_relative_to(P / "gits/psi4.disp0-opt064-20261021"),
            "Not the new dedicated source namespace")
    require(attempt == source.parent / "submission-attempt", "Wrong attempt namespace")
    require(not attempt.exists() and not attempt.is_symlink(), "Attempt already exists: NEVER replay")
    require(attempt.parent.stat().st_uid == os.getuid()
            and not attempt.parent.stat().st_mode & 0o077, "Private owned parent required")
    attempt.mkdir(mode=0o700)
    fsync_dir(attempt.parent)
    # BEFORE any scheduler request, even test-only: permanent attempt identity.
    batch = source / "devtools/disp0/optimized_build/build.sbatch"
    payload = source / "devtools/disp0/optimized_build/payload.sh"
    intent = {"recipe_pin": pin, "source": str(source), "batch_sha256": sha(batch),
              "payload_sha256": sha(payload), "einsums_fork": "https://github.com/Awallace3/Einsums.git",
              "einsums_pin": E2, "scientific_pin": SCIENCE,
              "old_input_submission": "UNKNOWN_RESERVED_UNTOUCHED_NOT_REPLAYED",
              "authorization": "2026-10-21 user: start the build",
              "resources": {"nodes": 1, "tasks": 1, "cpus": 8, "memory_gib": 48,
                            "seconds": 7200, "gpus": 0, "qos": "embers",
                            "account": "gts-cs207-chemx", "requeue": False},
              "timestamp_unix": time.time()}
    record(attempt, "intent.json", intent)
    def run(name, argv, **kwargs):
        result = capture(argv, **kwargs)
        record(attempt, name + ".json", result)
        return checked(result)
    require(run("head", ["git", "-C", str(source), "rev-parse", "HEAD"]) == pin, "Wrong source")
    require(not run("clean", ["git", "-C", str(source), "status", "--porcelain"]), "Dirty source")
    run("science", ["git", "-C", str(source), "diff", "--exit-code", SCIENCE, pin, "--", "psi4/"])
    lock = P / f"runs/saptdft-cuest-disp0/build-{SCIENCE}-14196330/environment-explicit.txt"
    require(sha(lock) == LOCK_SHA, "Original read-only dependency lock differs")
    record(attempt, "quota-admission.json", quota_admission(quota, quota_sha, time.time()))
    for name, directory in (("project", source.parent),
                            ("scratch", Path("/storage/scratch1/3/awallace43"))):
        stat = os.statvfs(directory)
        require(stat.f_bavail * stat.f_frsize >= 64 * 1024**3
                and stat.f_favail >= 50000, "Insufficient filesystem bytes/inodes")
        record(attempt, name + "-statvfs.json",
               {"directory": str(directory), "free_bytes": stat.f_bavail * stat.f_frsize,
                "free_inodes": stat.f_favail})
    rows = run("queue", ["squeue", "-h", "-r", "-u", os.environ["USER"], "-o", "%i|%j|%q"])
    queue = [line.split("|") for line in rows.splitlines() if line]
    require(all(len(row) == 3 for row in queue), "Malformed queue proof")
    require(sum(row[2] == "embers" for row in queue) < 50, "No embers headroom")
    require(not any(row[1] == "disp0-opt064" for row in queue), "Duplicate optimized build")
    scheduler_env = scheduler_environment()
    route = capture(["sbatch", "--test-only", *RESOURCES, str(batch)], env=scheduler_env)
    record(attempt, "test-only.json", route)  # preserve both streams, including simulated ID
    checked(route)
    text = route["stdout"] + "\n" + route["stderr"]
    require(re.search(r"\bpartition[ =:]cpu-small\b", text, re.I), "Unexpected/unproved CPU route")
    require(sha(batch) == intent["batch_sha256"] and sha(payload) == intent["payload_sha256"],
            "Recipe bytes changed during preflight")
    argv = ["sbatch", "--parsable", *RESOURCES,
            f"--export=ALL,DISP0_OPT_SOURCE={source},DISP0_OPT_PIN={pin}", str(batch)]
    # Durable marker and parent fsync BEFORE the sole actual sbatch RPC.
    record(attempt, "dispatch-intent.json", {"argv": argv, "timestamp_unix": time.time()})
    response = capture(argv, timeout=20, env=scheduler_env)
    record(attempt, "raw-response.json", response)
    # Any exception/lost response/nonzero/unparseable response stays UNKNOWN.
    require(response["exit"] == 0, "Submission ambiguous/rejected; NEVER replay")
    accepted = re.fullmatch(r"([1-9][0-9]*)(?:;([A-Za-z0-9_.-]+))?\n?", response["stdout"])
    require(accepted is not None, "Unparseable submission reply; NEVER replay")
    job = accepted.group(1)
    durable(attempt / "accepted-job-id.txt", (job + "\n").encode())
    # Identity is durable BEFORE any accounting/controller query.
    print(json.dumps({"accepted_job_id": job, "attempt": str(attempt)}, sort_keys=True), flush=True)
    result = capture(["scontrol", "show", "job", job, "-o"], timeout=10)
    record(attempt, "controller.json", result)
    print(json.dumps({"accepted_job_id": job, "attempt": str(attempt), "controller": result},
                     sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--pin", required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--quota-proof", type=Path, required=True)
    parser.add_argument("--quota-sha256", required=True)
    args = parser.parse_args()
    os.umask(0o077)
    submit(args.source, args.pin, args.attempt, args.quota_proof, args.quota_sha256)


if __name__ == "__main__":
    main()
