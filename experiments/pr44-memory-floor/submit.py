"""One bounded bracketing round. No job dependencies or automatic retries."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("campaign", type=Path)
parser.add_argument("--test-only", action="store_true")
args = parser.parse_args()
here = Path(__file__).resolve().parent
root = here.parents[1]
config = json.loads((here / "config.json").read_text())
assert config["qos"] == "embers"
commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
assert not subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"])
assert not subprocess.check_output(["git", "-C", str(root), "diff", config["native_commit"], "HEAD", "--", "psi4/"])
build = Path(config["build_campaign"]) / "build"
validated = json.loads((build / "validated.json").read_text())
assert validated["commit"] == config["native_commit"] and validated["core_sha256"] == config["core_sha256"]
assert (build / "stage/bin/psi4").is_file()
checkpoint = Path(config["checkpoint"])
assert (checkpoint / "fdds-cache.npz").is_file()
assert json.loads((checkpoint / "fdds-cache.json").read_text())["sha256"] == config["checkpoint_sha256"]
queue = subprocess.check_output(["squeue", "-h", "-r", "-u", os.environ["USER"], "-o", "%q"], text=True)
assert sum(q.strip() == "embers" for q in queue.splitlines()) + len(config["cases"]) <= 50
campaign = args.campaign.resolve()
(campaign / "logs").mkdir(parents=True, exist_ok=True)
receipt = campaign / "submissions.json"
assert not receipt.exists(), f"Prior submission attempted: inspect {receipt} and squeue before any retry"
commands = []
for case in config["cases"]:
    cmd = ["sbatch", "--parsable", f"--job-name=pr44-floor-{case['id']}",
           f"--account={config['account']}", f"--qos={config['qos']}", f"--partition={config['partition']}",
           f"--cpus-per-task={config['threads']}", f"--mem={config['slurm_gib']}G",
           f"--tmp={config['tmp_gib']}G", f"--time={config['walltime']}",
           f"--output={campaign}/logs/%x-%j.out", f"--error={campaign}/logs/%x-%j.err",
           str(here / "job.sbatch"), str(root), str(campaign), case["id"], case["mode"],
           str(case["memory_gb"]), commit, config["build_campaign"], config["checkpoint"]]
    subprocess.run([cmd[0], "--test-only", *cmd[1:]], check=True)
    commands.append(cmd)
manifest = {
    "harness_commit": commit, "campaign": str(campaign), "config": config,
    "file_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [here / "config.json", here / "job.sbatch", here / "submit.py",
                              here.parent / "pr44-memory/run.py", here.parent / "pr44/case.py"]},
}
if args.test_only:
    print(json.dumps(manifest, indent=2))
    raise SystemExit(0)
with (campaign / "manifest.json").open("x") as handle:
    json.dump(manifest, handle, indent=2)
with receipt.open("x") as handle:
    json.dump([], handle)
records = []
for case, cmd in zip(config["cases"], commands):
    record = {"case": case, "command": cmd, "state": "submitting"}
    records.append(record)
    receipt.write_text(json.dumps(records, indent=2))
    result = subprocess.run(cmd, capture_output=True, text=True)
    record.update(state="returned", returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    if result.returncode == 0:
        record["job_id"] = result.stdout.strip().split(";")[0]
    receipt.write_text(json.dumps(records, indent=2))
    print(json.dumps(record), flush=True)
    if result.returncode or not record.get("job_id", "").isdigit():
        raise SystemExit("Submission failed/uncertain. Inspect receipt and queue; do not blindly repeat.")
