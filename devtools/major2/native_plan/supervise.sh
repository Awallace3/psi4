#!/usr/bin/env bash
# Owned TERM-safe wrapper. Completion is fsynced only after true unsignalled success.
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
  # Python argument is explicit and remains the original controller interpreter.
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
        v=json.loads((root/"ENDPOINT-COMPLETE.json").read_text())
        assert v==dict(processes=3,outputs=18,pairs=153,accepted_speedup_data=False,compiled_diagnostic_only=True)
        assert (root/"NATIVE-DIAGNOSTIC.json").is_file()
    except BaseException: rc=1
write("batch-exit.json",json.dumps(dict(exit=rc,signalled=signalled))+"\n")
write("COMPLETED" if rc==0 else "FAILED","native diagnostic only\n")
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
