#!/usr/bin/env bash
# Compatibility entrypoint. The editable implementation now lives at repo root.
set -euo pipefail
SOURCE=$(git -C "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)
exec bash "$SOURCE/build.sh" "$@"
