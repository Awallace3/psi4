#!/usr/bin/env bash
# The owning agent invokes once after approval, duplicate/account/dry-run checks.
# salloc --no-shell preserves the granted allocation for serialized srun steps.
set -Eeuo pipefail
trap 'rc=$?; printf "Shared allocation controller failed at line%s (exit%s)\n" "$LINENO" "$rc" >&2; exit "$rc"' ERR
: "${CUEST_SHARED_RUN_ROOT:?}" "${CUEST_SHARED_ACCOUNT:?}"
: "${CUEST_SHARED_HARNESS_PIN:?}"
root=$(git -C "$(dirname "$(realpath "$0")")" rev-parse --show-toplevel)
test "$(git -C "$root" rev-parse HEAD)" = "$CUEST_SHARED_HARNESS_PIN"
test -z "$(git -C "$root" status --porcelain --untracked-files=no)"
run=$CUEST_SHARED_RUN_ROOT
[[ "$run" == /storage/project/r-cs207-0/awallace43/runs/psi4-cuest-shared/* ]]
mkdir "$run" # A fresh run, never overwrite even an ungranted request.
export PYTHONDONTWRITEBYTECODE=1
python3 - "$run" "$CUEST_SHARED_HARNESS_PIN" <<'PY'
import json, pathlib, sys
r = pathlib.Path(sys.argv[1])
(r / "allocation.json").write_text(json.dumps(dict(
    status="pending", harness_pin=sys.argv[2], run_directory=str(r),
    gpus=1, cpus=8, memory_gib=128, walltime="08:00:00",
    owner="saptdft-cuest-disp", qos="inferno",
)) + "\n")
PY
salloc --no-shell --account="$CUEST_SHARED_ACCOUNT" --qos=inferno \
    --partition=gpu-h200 --gres=gpu:h200:1 --nodes=1 --ntasks=1 \
    --cpus-per-task=8 --mem=128G --time=08:00:00 \
    --job-name=cuest-shared-interactive 2>&1 | tee "$run/salloc.log"
job=$(awk '/Granted job allocation/ {print $NF}' "$run/salloc.log" | tail -n 1)
[[ "$job" =~ ^[0-9]+$ ]]
printf 'GRANTED_JOB=%s\n' "$job"
scontrol show job -o "$job" > "$run/scontrol-granted.txt"
python3 - "$run/scontrol-granted.txt" "$CUEST_SHARED_ACCOUNT" <<'PY'
import pathlib, sys
text = pathlib.Path(sys.argv[1]).read_text()
fields = dict(s.split("=", 1) for s in text.split() if "=" in s)
assert fields["Account"] == sys.argv[2]
assert fields["QOS"] == "inferno" and fields["JobState"] == "RUNNING"
assert fields["NumCPUs"] == "8" and fields["TimeLimit"] == "08:00:00"
PY
srun --jobid="$job" --nodes=1 --ntasks=1 --cpus-per-task=8 \
    --gres=gpu:1 --mem=128G --exclusive --exact \
    timeout --signal=TERM --kill-after=10s 60 \
    python3 "$root/devtools/benchmarks/cuest_allocation_identity.py" "$run/hardware.json"
python3 - "$run" "$job" "$CUEST_SHARED_ACCOUNT" <<'PY'
import json, os, pathlib, sys
r = pathlib.Path(sys.argv[1])
m = json.loads((r / "allocation.json").read_text())
h = json.loads((r / "hardware.json").read_text())
assert h["job_id"] == int(sys.argv[2])
m.update(h, status="allocated", account=sys.argv[3],
         lock_path=str(r / "gpu-step.lock"),
         co_tenancy="serialized agents only; whole-step exclusive timing")
tmp = r / ".allocation.json.tmp"
tmp.write_text(json.dumps(m, indent=2) + "\n")
os.replace(tmp, r / "allocation.json")
PY
printf 'ALLOCATION_READY=%s/allocation.json\n' "$run"
