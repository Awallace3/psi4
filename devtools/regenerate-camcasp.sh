#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
export LC_ALL=C
export LANG=C
export TZ=UTC

LOCALIZATION_LIMIT=3
WSM_LIMIT=3
HYDROGEN_LIMIT=3
PFIT_WEIGHT=4
PFIT_WEIGHT_COEFF=0.001
PFIT_CUTOFF=0.0001
N_FREQUENCIES=10
FREQUENCY_SCALE=0.5

# L1 supports dispersion through C6, L2 through C10, and L3 through C12.
# WSM_LIMIT must not exceed LOCALIZATION_LIMIT.
# Reducing HYDROGEN_LIMIT truncates higher-order H-containing coefficients.
# Rank or penalty changes can alter even the fitted dipole-dipole subset.
# Values from a changed protocol must not replace pytest literals without
# reviewing the complete protocol change.

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd -P)"
REFERENCE_ROOT="$REPO_ROOT/.camcasp-reference"
CURRENT_STAGE="startup"

fail() {
    printf '[%s] %s\n' "$CURRENT_STAGE" "$*" >&2
    return 1
}

run_logged() {
    local stage="$1"
    local log="$2"
    shift 2
    CURRENT_STAGE="$stage"
    mkdir -p "$(dirname "$log")"
    set +e
    "$@" >"$log" 2>&1
    local rc=$?
    set -e
    if (( rc != 0 )); then
        printf '[%s] failed with exit status %d; retained log: %s\n' \
            "$stage" "$rc" "$log" >&2
        return "$rc"
    fi
}

on_exit() {
    local rc=$?
    if (( rc != 0 )); then
        printf '[%s] reference generation stopped; logs remain under %s/logs\n' \
            "$CURRENT_STAGE" "$REFERENCE_ROOT" >&2
    fi
}

main() {
    trap on_exit EXIT
    printf 'CamCASP reference root: %s\n' "$REFERENCE_ROOT"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
