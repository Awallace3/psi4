"""Five-minute read-only artifact checks; official watch_jobs owns Slurm truth.

Exit on an unleased allocation to wake the agent for actual-state/cleanup
reconciliation. Never launch, cancel, or release any job from this observer.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time


REMOTE = r"""
import fcntl,json,pathlib,time
r=pathlib.Path(MANIFEST_LITERAL).parent
m=json.loads((r/'allocation.json').read_text())
out=dict(manifest_status=m['status'],job_id=m.get('job_id'),quarantined=(r/'QUARANTINED').exists())
try:
 with (r/'gpu-step.lock').open() as f:
  try:
   fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);out['lease']='free'
  except BlockingIOError:out['lease']='held'
except FileNotFoundError:out['lease']='unknown'
out['steps']=[]
for p in sorted((r/'steps').glob('*')):
 out['steps'].append(dict(name=p.name,terminal=(p/'TERMINAL').exists(),
  cleanup=(p/'GPU_CLEANUP_VERIFIED').exists(),
  exit_code=(p/'exit-code.txt').read_text().strip() if (p/'exit-code.txt').exists() else None))
print(json.dumps(out,sort_keys=True))
"""


def remote_script(manifest):
    path = Path(manifest)
    prefix = Path("/storage/project/r-cs207-0/awallace43/runs/psi4-cuest-shared")
    if path.name != "allocation.json" or path.parent.parent != prefix:
        raise ValueError("Manifest must belong to one approved campaign child")
    return REMOTE.replace("MANIFEST_LITERAL", repr(str(path)))


def action(snapshot):
    if snapshot.get("quarantined"):
        return "ALERT_QUARANTINED"
    if snapshot.get("manifest_status") != "allocated":
        return "HOLD_MANIFEST_NOT_ALLOCATED"
    if snapshot.get("lease") == "free":
        return "ALERT_NO_ACTIVE_LEASE_RECONCILE_ACTUAL_STATE"
    if snapshot.get("lease") != "held":
        return "ALERT_LEASE_UNKNOWN"
    if snapshot.get("steps") and all(s["terminal"] for s in snapshot["steps"]):
        return "ALERT_HELD_LEASE_WITH_NO_UNFINISHED_STEP"
    return "ACTIVE_LEASE_CPU_CONTROL_OR_GPU_WORK"


def official_terminal(log, job_id=14214205):
    # Only a positive official-watcher event counts; absent/failed queries do not.
    return any(line.split("ALL-TERMINAL ", 1)[-1].split()[0] == str(job_id)
               for line in log.splitlines() if "ALL-TERMINAL " in line
               and line.split("ALL-TERMINAL ", 1)[-1].split())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phx", type=Path, required=True)
    p.add_argument("--lifetime-log", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--job-id", type=int, required=True)
    p.add_argument("--interval", type=int, default=300)
    p.add_argument("--initial-delay", type=int, choices=(0, 300), default=0,
                   help="Allow a bounded handoff before the first artifact check.")
    args = p.parse_args()
    assert args.job_id > 0
    script = remote_script(args.manifest)
    assert args.interval >= 300
    deadline = time.monotonic() + args.initial_delay
    if args.initial_delay:
        print("HANDOFF_FIRST_CHECK_IN_300_SECONDS", flush=True)
        time.sleep(args.initial_delay)
    while True:
        if args.lifetime_log.exists() and official_terminal(args.lifetime_log.read_text(), args.job_id):
            print("OFFICIAL_ALLOCATION_TERMINAL_REQUIRE_ARTIFACT_AUDIT", flush=True)
            return
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            result = subprocess.run(
                [str(args.phx), "python3 -"], input=script,
                text=True, capture_output=True, timeout=120,
            )
            if result.returncode:
                raise RuntimeError(f"SSH exit{result.returncode}: {result.stderr[-500:]}")
            snapshot = json.loads(result.stdout)
            if int(snapshot["job_id"]) != args.job_id:
                raise ValueError("Manifest job differs from pinned observer job")
            decision = action(snapshot)
            print(stamp, decision, json.dumps(snapshot, sort_keys=True), flush=True)
            if decision.startswith("ALERT_"):
                # bg_run completion notification wakes the owner; it must verify
                # actual Slurm state and cleanup, not assume idle/granted/released.
                return
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as e:
            print(stamp, "PROGRESS_UNKNOWN_NO_STATE_INFERENCE", str(e), flush=True)
        deadline += args.interval
        time.sleep(max(0, deadline - time.monotonic()))


if __name__ == "__main__":
    main()
