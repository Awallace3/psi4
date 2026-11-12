"""Submit the approved pair only. Run inferno_account.sh immediately beforehand."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("campaign", type=Path)
parser.add_argument("--test-only", action="store_true")
args = parser.parse_args()
harness = Path(__file__).resolve().parent
root = args.campaign.resolve()
config = json.loads((harness / "config.json").read_text())
assert not subprocess.check_output(["git", "-C", str(harness), "status", "--porcelain"])
commit = subprocess.check_output(["git", "-C", str(harness), "rev-parse", "HEAD"], text=True).strip()
build_campaign = Path(config["build_campaign"])
assert (build_campaign / "runtime-pydantic2/bin/python").is_file()
for arm in ["base", "head"]:
    source = build_campaign / "sources" / arm
    assert subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() == config[arm]
    assert not subprocess.check_output(["git", "-C", str(source), "diff", "HEAD"])
    validated = json.loads((build_campaign / "builds" / arm / "validated.json").read_text())
    assert validated["commit"] == config[arm]
    assert (build_campaign / "builds" / arm / "stage/bin/psi4").is_file()
root.mkdir(parents=True, exist_ok=True)
(root / "logs").mkdir(exist_ok=True)
receipt = root / "submissions.json"
assert not receipt.exists(), f"Submission already attempted: inspect {receipt}, do not duplicate jobs"
manifest = {
    "cluster": "phoenix", "account": "gts-cs207", "qos": "inferno",
    "partition": "cpu-gnr", "constraint": "graniterapids", "exclusive": True,
    "cpus": 192, "slurm_memory_gib": 1400, "tmp_gib": 2600, "walltime": "12:00:00",
    "harness_commit": commit, "harness": str(harness), "campaign": str(root), "config": config,
    "file_sha256": {str(p.relative_to(harness.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [harness / "config.json", harness / "protein157.xyz",
                              harness / "run.py", harness / "job.sbatch", harness.parent / "pr44/case.py"]},
}
commands = []
for arm in ["base", "head"]:
    cmd = ["sbatch", "--parsable", f"--job-name=pr44-p157-{arm}",
           f"--output={root}/logs/%x-%j.out", f"--error={root}/logs/%x-%j.err",
           str(harness / "job.sbatch"), str(harness), str(root), arm, commit]
    subprocess.run([cmd[0], "--test-only", *cmd[1:]], check=True)
    commands.append(cmd)
if args.test_only:
    print(json.dumps(manifest, indent=2))
    raise SystemExit(0)
with (root / "manifest.json").open("x") as handle:
    json.dump(manifest, handle, indent=2)
# Reserve the receipt before submission, so an interrupted submit is never blindly retried.
with receipt.open("x") as handle:
    json.dump([], handle)
records = []
for cmd in commands:
    record = {"command": cmd, "submitted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "state": "submitting"}
    records.append(record)
    receipt.write_text(json.dumps(records, indent=2) + "\n")
    result = subprocess.run(cmd, capture_output=True, text=True)
    record.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr, state="returned")
    if result.returncode == 0:
        record["job_id"] = result.stdout.strip().split(";")[0]
    receipt.write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    if result.returncode or not record.get("job_id", "").isdigit():
        raise SystemExit("Submission uncertain or failed: inspect receipt and squeue before any retry")
print(f"RECEIPT {receipt}", flush=True)
