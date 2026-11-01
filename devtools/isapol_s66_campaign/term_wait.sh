# Source from an absolute pinned operational path, not the Slurm spool path.
# The payload is a setsid process-group leader (srun on Phoenix).
ccasp_wait_child() {
    local child=$1 grace=$2 rc=0 watchdog=''
    local interrupted=0
    [[ "$child" =~ ^[0-9]+$ && "$grace" =~ ^[0-9]+$ && "$grace" -ge 1 && "$grace" -le 60 ]] || return 2
    ccasp_forward_term() {
        [[ "$interrupted" -eq 0 ]] || return 0
        interrupted=1
        kill -TERM -- "-$child" 2>/dev/null || true
        # Separate group so both the watchdog and its sleep can be cancelled
        # after a prompt cleanup. Never leave an unbounded wait after TERM.
        setsid /bin/bash -c \
            'sleep "$1"; kill -KILL -- "-$2" 2>/dev/null || true' _ "$grace" "$child" &
        watchdog=$!
    }
    trap ccasp_forward_term TERM INT
    while true; do
        if wait "$child"; then rc=0; else rc=$?; fi
        # A trapped signal interrupts Bash wait before the child has exited.
        # Reap again rather than ending the allocation during Python cleanup.
        kill -0 "$child" 2>/dev/null || break
    done
    if [[ -n "$watchdog" ]]; then
        kill -TERM -- "-$watchdog" 2>/dev/null || true
        wait "$watchdog" 2>/dev/null || true
    fi
    trap - TERM INT
    unset -f ccasp_forward_term
    [[ "$interrupted" -eq 0 ]] || return 143
    return "$rc"
}
