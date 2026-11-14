"""Bounded old-binary tests. No automatic retries; fresh approval for inferno."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

BASE = "8f46e95cba0ea6c6bdb56453eeabb2f7560d62b7"
CORE_HASH = "bfd8cbebf6a3c2ae0d69bdb6344f9f446204cfdfcedbe24296b11cc51eaa8fe8"
BUILD = Path("/storage/scratch1/3/awallace43/psi4-pr44-283ae89a32-embers-v1")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("campaign", type=Path)
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--memory-gb", type=int, nargs="+")
parser.add_argument("--slurm-gib", type=int, default=1400)
parser.add_argument("--qos", choices=["embers", "inferno"], default="embers")
parser.add_argument("--account", choices=["gts-cs207", "gts-cs207-chemx"],
                    help="Inferno account from the fresh quote and explicit user approval")
parser.add_argument("--time", default="08:00:00")
parser.add_argument("--test-only", action="store_true")
parser.add_argument("--approved-inferno", action="store_true")
args = parser.parse_args()
assert 1 <= args.slurm_gib <= 1400
if args.smoke:
    assert not args.memory_gb
    budgets, threads, memory, walltime = [0], 2, 8, "00:15:00"
else:
    assert args.memory_gb and len(args.memory_gb) <= 2
    assert len(set(args.memory_gb)) == len(args.memory_gb)
    assert all(1 <= value <= 1200 for value in args.memory_gb)
    budgets, threads, memory, walltime = args.memory_gb, 192, args.slurm_gib, args.time
parts = walltime.split(":")
assert len(parts) == 3 and all(p.isdigit() for p in parts), "Walltime must be HH:MM:SS"
hours, minutes, seconds = map(int, parts)
assert 0 <= minutes < 60 and 0 <= seconds < 60
limit_seconds = 3600 * hours + 60 * minutes + seconds
assert 0 < limit_seconds <= (16 if args.qos == "inferno" else 8) * 3600, "Walltime exceeds round ceiling"
if args.qos == "inferno" and not args.test_only:
    assert args.approved_inferno, "Fresh user approval required for this exact round"
account = args.account or ("gts-cs207" if args.qos == "inferno" else "gts-cs207-chemx")
if args.qos == "embers":
    assert account == "gts-cs207-chemx"
here = Path(__file__).resolve().parent
root = here.parents[1]
commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
assert not subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"])
assert subprocess.check_output(
    ["git", "-C", str(BUILD / "sources/base"), "rev-parse", "HEAD"], text=True).strip() == BASE
assert not subprocess.check_output(["git", "-C", str(BUILD / "sources/base"), "diff", "HEAD"])
validated = json.loads((BUILD / "builds/base/validated.json").read_text())
assert validated["commit"] == BASE and validated["core_sha256"] == CORE_HASH
assert (BUILD / "runtime-pydantic2/bin/python").is_file()
assert (BUILD / "builds/base/stage/bin/psi4").is_file()
if args.qos == "embers":
    queue = subprocess.check_output(
        ["squeue", "-h", "-r", "-u", os.environ["USER"], "-o", "%q"], text=True)
    assert sum(q.strip() == "embers" for q in queue.splitlines()) + len(budgets) <= 50
campaign = args.campaign.resolve()
(campaign / "logs").mkdir(parents=True, exist_ok=True)
receipt = campaign / "submissions.json"
assert not receipt.exists(), "Prior submission attempted: inspect receipt/queue, never retry blindly"
commands = []
for budget in budgets:
    case = "base-smoke" if args.smoke else f"base-{budget}GB-cap{memory}GiB"
    cmd = ["sbatch", "--parsable", f"--job-name=pr44-old-{case}", f"--account={account}",
           f"--qos={args.qos}", f"--cpus-per-task={threads}", f"--mem={memory}G",
           f"--time={walltime}", f"--output={campaign}/logs/%x-%j.out",
           f"--error={campaign}/logs/%x-%j.err"]
    cmd += (["--partition=cpu-small"] if args.smoke else
            ["--partition=cpu-gnr", "--constraint=graniterapids", "--exclusive", "--tmp=2600G"])
    cmd += [str(here / "job.sbatch"), str(root), str(campaign), case, str(budget), commit,
            "smoke" if args.smoke else "full"]
    subprocess.run([cmd[0], "--test-only", *cmd[1:]], check=True)
    commands.append(cmd)
files = [*here.glob("*.py"), here / "job.sbatch",
         root / "experiments/pr44-protein157/run.py",
         root / "experiments/pr44-protein157/config.json",
         root / "experiments/pr44-protein157/protein157.xyz",
         root / "experiments/pr44/case.py",
         * (root / "experiments/pr44-memory").glob("*.py")]
manifest = {
    "harness_commit": commit, "base_commit": BASE, "core_sha256": CORE_HASH,
    "account": account, "qos": args.qos, "threads": threads, "slurm_gib": memory,
    "walltime": walltime, "budgets_GB": budgets, "namespace_workaround": True,
    "energy_tolerance_Eh": 1e-7, "approved_inferno": args.approved_inferno,
    "file_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    "commands": commands,
}
if args.test_only:
    print(json.dumps(manifest, indent=2))
    raise SystemExit(0)
with (campaign / "manifest.json").open("x") as handle:
    json.dump(manifest, handle, indent=2)
with receipt.open("x") as handle:
    json.dump([], handle)
records = []
for cmd in commands:
    record = {"command": cmd, "state": "submitting"}
    records.append(record)
    receipt.write_text(json.dumps(records, indent=2))
    result = subprocess.run(cmd, capture_output=True, text=True)
    record.update(state="returned", returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    if result.returncode == 0:
        record["job_id"] = result.stdout.strip().split(";")[0]
    receipt.write_text(json.dumps(records, indent=2))
    print(json.dumps(record), flush=True)
    if result.returncode or not record.get("job_id", "").isdigit():
        raise SystemExit("Submission uncertain/failed: inspect receipt and queue before any retry")
