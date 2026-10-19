#!/usr/bin/env bash
# Parent custody ONLY. Never invoked by preparation or any Slurm payload.
# Arguments: absolute generated bundle directory, optional absolute phx path.
set -euo pipefail
bundle=${1:?absolute reviewed bundle directory required}
phx=${2:-/home/awallace43/.pi/agent/skills/phoenix-slurm-experiments/scripts/phx}
[[ "$bundle" == /* && "$phx" == /* ]]
test -x "$phx"
test -n "${MAJOR2_NATIVE_PARENT_REVIEW:-}"
here=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
exec /home/awallace43/miniconda3/envs/p4qca/bin/python -B "$here/handoff.py" parent-local "$bundle" "$phx"
