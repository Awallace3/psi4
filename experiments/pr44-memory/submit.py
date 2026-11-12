"""Explicit embers-only diagnostic submission, durable IDs, no automatic retry."""
import argparse
import json
import os
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("campaign", type=Path)
parser.add_argument("mode", choices=["build", "science"])
parser.add_argument("--test-only", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
campaign = args.campaign.resolve()
assert not subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"])
commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
if args.mode == "science":
    validated = json.loads((campaign / "build/validated.json").read_text())
    assert validated["commit"] == commit
plan = ["build"] if args.mode == "build" else ["1200", "800"]
queue = subprocess.check_output(["squeue", "-h", "-r", "-u", os.environ["USER"], "-o", "%q"], text=True)
assert sum(q.strip() == "embers" for q in queue.splitlines()) + len(plan) <= 50
(campaign / "logs").mkdir(parents=True, exist_ok=True)
receipt = campaign / f"{args.mode}-submissions.json"
assert not receipt.exists(), f"Inspect existing receipt before retry: {receipt}"
commands = []
for item in plan:
    script = root / "experiments/pr44-memory" / ("build.sbatch" if item == "build" else "job.sbatch")
    cmd = ["sbatch", "--parsable", f"--job-name=pr44-mem-{item}",
           f"--output={campaign}/logs/%x-%j.out", f"--error={campaign}/logs/%x-%j.err",
           str(script), str(root), str(campaign)]
    cmd += [commit] if item == "build" else [item, commit]
    subprocess.run([cmd[0], "--test-only", *cmd[1:]], check=True)
    commands.append(cmd)
if args.test_only:
    print(json.dumps(commands, indent=2))
    raise SystemExit(0)
with receipt.open("x") as handle:
    json.dump({"source_commit": commit, "commands": commands, "jobs": []}, handle)
records = []
for cmd in commands:
    record = {"command": cmd, "state": "submitting"}
    records.append(record)
    receipt.write_text(json.dumps({"source_commit": commit, "jobs": records}, indent=2))
    result = subprocess.run(cmd, capture_output=True, text=True)
    record.update(state="returned", returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    if result.returncode == 0:
        record["job_id"] = result.stdout.strip().split(";")[0]
    receipt.write_text(json.dumps({"source_commit": commit, "jobs": records}, indent=2))
    print(json.dumps(record), flush=True)
    if result.returncode or not record.get("job_id", "").isdigit():
        raise SystemExit("Submission failed/uncertain. Inspect receipt and queue, do not blindly retry.")
