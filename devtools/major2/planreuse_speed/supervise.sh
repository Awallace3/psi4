#!/usr/bin/env bash
# Owned specialization of native_plan/supervise.sh, tested with host fake payloads.
set -euo pipefail
umask 027
root=$1; shift
test -d "$root"
test ! -e "$root/COMPLETED"
test ! -e "$root/FAILED"
test ! -e "$root/batch-exit.json"
child= signalled=0
forward() {
  [[ -z $child ]] && return 0
  kill -TERM -- "-$child" 2>/dev/null || true
  kill -TERM "$child" 2>/dev/null || true
}
stop() { signalled=1; forward; }
finish() {
  rc=$?
  trap - EXIT
  (( signalled )) && rc=143
  "$marker_python" -B -c '
import json,os,sys
from pathlib import Path
root=Path(sys.argv[1]); rc=int(sys.argv[2]); signalled=int(sys.argv[3])
def write(name,data):
    with (root/name).open("x") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    (root/name).chmod(0o444)
    fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)
if rc==0:
    try:
        v=json.loads((root/"SPEED-COMPLETE.json").read_text())
        assert v==dict(processes=6,computes=66,stored_arrays=24,matched_actual_D=True,plan_reuse_benchmark_only=True)
        assert (root/"SPEED-REPORT.json").is_file()
    except BaseException: rc=1
write("batch-exit.json",json.dumps(dict(exit=rc,signalled=signalled))+"\n")
write("COMPLETED" if rc==0 else "FAILED","fixed-work plan reuse only\n")
sys.exit(rc)
' "$root" "$rc" "$signalled"
}
marker_python=$1
trap stop TERM INT HUP
trap finish EXIT
setsid "$@" &
child=$!
if (( signalled )); then forward; fi
set +e
wait "$child"; rc=$?
while kill -0 "$child" 2>/dev/null; do wait "$child"; rc=$?; done
set -e
(( signalled )) && rc=143
exit "$rc"
