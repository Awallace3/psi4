#!/usr/bin/env bash
# Called by absolute path, never through the sbatch spool location.
set -euo pipefail
exec python3 "$(dirname "$(realpath "$0")")/fdds_canonical_controller.py" "$@"
