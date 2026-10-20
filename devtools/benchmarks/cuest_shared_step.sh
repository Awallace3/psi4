#!/usr/bin/env bash
# Run on Phoenix. All participating agents must use this owner-provided lock.
# Usage: bash cuest_shared_step.sh MANIFEST LABEL CWD TIMEOUT_SECONDS -- COMMAND...
set -euo pipefail
if (($# < 6)) || [[ "$5" != -- ]]; then
    echo "Usage: $0 MANIFEST LABEL CWD TIMEOUT_SECONDS -- COMMAND..." >&2
    exit 2
fi
manifest=$1 label=$2 cwd=$3 seconds=$4
shift 5
[[ "$manifest" = /* && "$cwd" = /* && -d "$cwd" ]]
[[ "$label" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]
[[ "$seconds" =~ ^[0-9]+$ ]] && ((seconds > 0 && seconds <= 10800))
mapfile -t allocation < <(python3 - "$manifest" <<'PY'
import json, pathlib, sys
m = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert m["status"] == "allocated", "Allocation is not ready"
assert m["gpus"] == 1 and m["cpus"] == 8 and m["memory_gib"] == 128
root = pathlib.Path(m["run_directory"])
assert root.is_absolute() and root.is_dir()
print(int(m["job_id"]))
print(root)
PY
)
((${#allocation[@]} == 2))
job=${allocation[0]} root=${allocation[1]}
# --exclusive alone is not a substitute for the cross-agent lock: hold both.
# Bound lock waiting too. A busy allocation does not authorize another request.
exec 9>"$root/gpu-step.lock"
flock --exclusive --wait 120 9 || {
    echo "Allocation busy; coordinate another window with the owner." >&2
    exit 75
}
state=$(squeue -h -j "$job" -o %T)
[[ "$state" == RUNNING ]] || {
    printf 'Refusing step: job%s state=%s\n' "$job" "$state" >&2
    exit 3
}
step="$root/steps/$label"
mkdir -p "$root/steps"
mkdir "$step" # Never overwrite a prior step, including a failed one.
printf '%q ' "$@" > "$step/command.txt"
printf '\n' >> "$step/command.txt"
printf '%s\n' "$cwd" > "$step/working-directory.txt"
date -u +%FT%TZ > "$step/STARTED"
printf 'job=%s label=%s bound_seconds=%s\n' "$job" "$label" "$seconds"
rc=0
srun --jobid="$job" --nodes=1 --ntasks=1 --cpus-per-task=8 \
    --gres=gpu:1 --mem=128G --exclusive --exact --unbuffered \
    --chdir="$cwd" \
    timeout --signal=TERM --kill-after=30s "$seconds" "$@" \
    > "$step/stdout.log" 2> "$step/stderr.log" || rc=$?
printf '%s\n' "$rc" > "$step/exit-code.txt"
date -u +%FT%TZ > "$step/TERMINAL"
# This means the command exited zero, not that its scientific claims were audited.
if ((rc == 0)); then date -u +%FT%TZ > "$step/COMMAND_EXITED_ZERO"; fi
printf 'job=%s label=%s exit=%s logs=%s\n' "$job" "$label" "$rc" "$step"
exit "$rc"
