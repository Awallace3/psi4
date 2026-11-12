"""Phoenix-only preflight/submission; no job-control operations or dependencies."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("campaign", type=Path)
parser.add_argument("mode", choices=["build", "science", "replay"])
parser.add_argument("--test-only", action="store_true")
parser.add_argument("--indices", nargs="+", type=int, help="Submit only these science cases when embers slots are limited")
args = parser.parse_args()
root = args.campaign.resolve()
harness = Path(__file__).resolve().parent
config = json.loads((harness / "config.json").read_text())
assert root.is_dir()
assert not subprocess.check_output(["git", "-C", str(harness), "status", "--porcelain"])
commit = subprocess.check_output(["git", "-C", str(harness), "rev-parse", "HEAD"], text=True).strip()
for arm in ["base", "head"]:
    source = root / "sources" / arm
    assert subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() == config[arm]
    if args.mode != "build":
        validated = json.loads((root / "builds" / arm / "validated.json").read_text())
        assert validated["commit"] == config[arm]
        assert (root / "builds" / arm / "stage/bin/psi4").is_file()
logs = root / "logs"
logs.mkdir(exist_ok=True)
indices = list(range(len(config["cases"]))) if args.indices is None else args.indices
assert len(set(indices)) == len(indices)
assert all(0 <= i < len(config["cases"]) for i in indices)
assert args.mode == "science" or args.indices is None
plan = [("build", 96, "04:00:00")] if args.mode == "build" else [
    (str(i), config["cases"][i]["slurm_gib"], "08:00:00") for i in indices]
if args.mode == "replay":
    plan = [("replay", 32, "02:00:00")]
queue = subprocess.check_output(["squeue", "-h", "-r", "-u", os.environ["USER"], "-o", "%q"], text=True)
used = sum(q.strip() == "embers" for q in queue.splitlines())
assert used + len(plan) <= 50, f"embers slots: {used} + {len(plan)} > 50"
manifest = {
    "cluster": "phoenix", "account": "gts-cs207-chemx", "qos": "embers",
    "requested_partition": "cpu-small", "harness_commit": commit,
    "config_sha256": hashlib.sha256((harness / "config.json").read_bytes()).hexdigest(),
    "config": config, "campaign": str(root), "harness": str(harness),
}
manifest_path = root / f"manifest-{commit}.json"
if manifest_path.exists():
    assert json.loads(manifest_path.read_text()) == manifest
elif not args.test_only:
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
commands = []
for index, memory, wall in plan:
    command = ["sbatch", "--parsable", "--account=gts-cs207-chemx", "--qos=embers",
               "--partition=cpu-small", "--cpus-per-task=24", f"--mem={memory}G",
               f"--time={wall}", f"--job-name=pr44-{args.mode}-{index}",
               f"--output={logs}/%x-%j.out", f"--error={logs}/%x-%j.err",
               str(harness / "job.sbatch"), args.mode, str(harness), str(root), index]
    subprocess.run([command[0], "--test-only", *command[1:]], check=True)
    commands.append(command)
if args.test_only:
    raise SystemExit(0)
failed = False
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
receipt = root / f"submissions-{args.mode}-{stamp}.json"
records = []
for command in commands:
    result = subprocess.run(command, capture_output=True, text=True)
    record = {"command": command, "returncode": result.returncode,
              "stdout": result.stdout, "stderr": result.stderr}
    if result.returncode == 0:
        job_id = result.stdout.strip().split(";")[0]
        assert job_id.isdigit(), record
        record["job_id"] = job_id
    else:
        failed = True
    records.append(record)
    receipt.write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    if result.returncode != 0:
        # Especially for a socket timeout, inspect squeue before retrying.
        break
print(f"RECEIPT {receipt}", flush=True)
raise SystemExit(1 if failed else 0)
