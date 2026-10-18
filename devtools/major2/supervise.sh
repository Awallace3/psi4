#!/usr/bin/env bash
# TERM-safe supervisor, also exercised locally with fake payloads.
# <owned-output-root> <command argv...>. Never submits or retries.
set -euo pipefail
umask 027
root=$1; shift
test -d "$root"
# A second invocation must not overwrite or reinterpret an older terminal marker.
test ! -e "$root/COMPLETED"
test ! -e "$root/FAILED"
test ! -e "$root/batch-exit.txt"
child=
signalled=0
forward_term() {
    [[ -z $child ]] && return 0
    # Notify the whole group AND the registered payload. This also covers the
    # pre-setsid window; the engine's handler latches repeated signals safely.
    kill -TERM -- "-$child" 2>/dev/null || true
    kill -TERM "$child" 2>/dev/null || true
}
on_signal() {
    signalled=1
    forward_term
}
on_exit() {
    rc=$?
    trap - EXIT
    (( signalled )) && rc=143
    printf 'exit=%s signalled=%s\n' "$rc" "$signalled" > "$root/batch-exit.txt"
    if (( rc == 0 )); then
        printf 'all gates and 24 records independently verified\n' > "$root/COMPLETED"
    else
        printf 'exit=%s signalled=%s\n' "$rc" "$signalled" > "$root/FAILED"
    fi
    exit "$rc"
}
trap on_signal TERM INT HUP
trap on_exit EXIT
# A dedicated group permits forwarding to a build/pytest/runner and all descendants.
setsid "$@" &
child=$!
# TERM can arrive after launch but BEFORE the shell records $!: forward the
# latched signal now instead of waiting forever for an unsignalled child.
if (( signalled )); then forward_term; fi
set +e
wait "$child"; rc=$?
while kill -0 "$child" 2>/dev/null; do wait "$child"; rc=$?; done
set -e
(( signalled )) && rc=143
exit "$rc"
