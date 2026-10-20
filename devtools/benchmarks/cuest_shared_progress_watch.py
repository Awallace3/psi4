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
import fcntl,json,pathlib
r=pathlib.Path('/storage/project/r-cs207-0/awallace43/runs/psi4-cuest-shared/20261020-interactive-02')
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
base=pathlib.Path('/storage/project/r-cs207-0/awallace43/runs/psi4-fdds/20261020-interactive-01/matched-14214205')
out['results']={}
for system in ['peptide','nanotube','protein83']:
 for arm in ['cpu','gpu']:
  for i in [0,1]:
   p=base/f'{system}-{arm}-{i}'/'result.json'
   if p.exists():
    d=json.loads(p.read_text())
    out['results'][f'{system}-{arm}-{i}']=dict(ok=d.get('ok'),seconds=d.get('fdds_seconds'))
print(json.dumps(out,sort_keys=True))
"""


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


def official_terminal(log):
    # Only a positive official-watcher event counts; absent/failed queries do not.
    return any("ALL-TERMINAL 14214205" in line for line in log.splitlines())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phx", type=Path, required=True)
    p.add_argument("--lifetime-log", type=Path, required=True)
    p.add_argument("--interval", type=int, default=300)
    p.add_argument("--initial-delay", type=int, choices=(0, 300), default=0,
                   help="Allow a bounded handoff before the first artifact check.")
    args = p.parse_args()
    assert args.interval >= 300
    deadline = time.monotonic() + args.initial_delay
    if args.initial_delay:
        print("HANDOFF_FIRST_CHECK_IN_300_SECONDS", flush=True)
        time.sleep(args.initial_delay)
    while True:
        if args.lifetime_log.exists() and official_terminal(args.lifetime_log.read_text()):
            print("OFFICIAL_ALLOCATION_TERMINAL_REQUIRE_ARTIFACT_AUDIT", flush=True)
            return
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            result = subprocess.run(
                [str(args.phx), "python3 -"], input=REMOTE,
                text=True, capture_output=True, timeout=120,
            )
            if result.returncode:
                raise RuntimeError(f"SSH exit{result.returncode}: {result.stderr[-500:]}")
            snapshot = json.loads(result.stdout)
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
