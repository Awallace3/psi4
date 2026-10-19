#!/usr/bin/env bash
# ONLY the six real JK children use this wrapper. No ENV value is unset/rewritten.
set -euo pipefail
H="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt/bin/python -B \
  "$H/workspace_control.py" --child-shell-check
exec bash "$H/arm.sh" "$@"
