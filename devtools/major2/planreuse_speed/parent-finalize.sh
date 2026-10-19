#!/usr/bin/env bash
# PARENT ONLY. UNEXECUTED by preparation; permanent local AND remote attempts.
set -euo pipefail
bundle=${1:?absolute reviewed frozen bundle directory required}
phx=${2:-/home/awallace43/.pi/agent/skills/phoenix-slurm-experiments/scripts/phx}
[[ "$bundle" == /* && "$phx" == /* ]]
test -x "$phx"
test -n "${MAJOR2_SPEED_PARENT_REVIEW:-}"
here=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
exec /home/awallace43/miniconda3/envs/p4qca/bin/python -B \
  "$here/custody.py" parent-local "$bundle" "$phx"
