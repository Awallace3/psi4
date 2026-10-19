#!/usr/bin/env bash
# UNEXECUTED: parent alone may review and invoke ONCE. No automatic submission.
set -euo pipefail
here=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
bundle=${1:?absolute frozen bundle required}
phx=/home/awallace43/.pi/agent/skills/phoenix-slurm-experiments/scripts/phx
: "${MAJOR2_WHOLE_PARENT_REVIEW:?parent must set exact manifest SHA256 after review}"
exec /home/awallace43/miniconda3/envs/p4qca/bin/python -B \
  "$here/custody.py" parent-local "$bundle" "$phx"
