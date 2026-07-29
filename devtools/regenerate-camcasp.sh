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

CAMCASP_URL="https://github.com/ajmisquitta/camcasp-bin.git"
CAMCASP_COMMIT="b4744425233a61786052832e1db4f109959c1ce9"
CAMCASP_VERSION_PATTERN="VERSION 7.2.2|7.2.2"
CAMCASP_PROGRAMS=(camcasp cluster process pfit casimir)
ORIENT_URL="https://gitlab.com/anthonyjs/orient.git"

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

require_safe_generated_path() {
    local candidate
    local camcasp_root="${CAMCASP:-$REPO_ROOT/camcasp-bin}"
    candidate="$(realpath -m "$1")"
    camcasp_root="$(realpath -m "$camcasp_root")"
    case "$candidate" in
        /|"$HOME"|"$REPO_ROOT"|"$camcasp_root"|"$REPO_ROOT/orient")
            fail "refusing unsafe generated path: $candidate"
            ;;
        "$REFERENCE_ROOT"|"$REFERENCE_ROOT"/*)
            ;;
        *)
            fail "generated path must be inside $REFERENCE_ROOT: $candidate"
            ;;
    esac
}

require_executable() {
    local name="$1"
    local path="$2"
    [[ -f "$path" && -x "$path" ]] ||
        fail "$name is not executable: $path"
}

preflight() {
    CURRENT_STAGE="preflight"
    PSI4_EXE="${PSI4_EXE:-$REPO_ROOT/build_camcasp/stage/bin/psi4}"
    CAMCASP="${CAMCASP:-$REPO_ROOT/camcasp-bin}"
    ORIENT_REF="${ORIENT_REF:-d8d861098c8f548e2cf230c387c8431d9418650a}"

    PSI4_EXE="$(realpath -m "$PSI4_EXE")"
    CAMCASP="$(realpath -m "$CAMCASP")"
    require_executable PSI4_EXE "$PSI4_EXE"

    [[ "$LOCALIZATION_LIMIT" -eq 3 ]]
    [[ "$WSM_LIMIT" -eq 3 && "$WSM_LIMIT" -le "$LOCALIZATION_LIMIT" ]]
    [[ "$HYDROGEN_LIMIT" -eq 3 ]]
    [[ "$PFIT_WEIGHT" -eq 4 ]]
    [[ "$PFIT_CUTOFF" == "0.0001" ]]
    [[ "$N_FREQUENCIES" -eq 10 ]]
    [[ "$FREQUENCY_SCALE" == "0.5" ]]
}

install_camcasp_program() {
    local program="$1"
    local archive="$CAMCASP/x86-64/gfortran/$program.gz"
    local exe_dir="$CAMCASP/x86-64/gfortran/exe"
    local target="$exe_dir/$program"
    local temp="$target.tmp.$$"

    [[ -f "$archive" ]] || fail "missing CamCASP archive: $archive"
    gzip -t "$archive"
    mkdir -p "$exe_dir" "$CAMCASP/bin"
    if [[ ! -x "$target" ]]; then
        gzip -dc "$archive" >"$temp"
        chmod 0755 "$temp"
        mv -f "$temp" "$target"
    fi
    ln -sfn "../x86-64/gfortran/exe/$program" "$CAMCASP/bin/$program"
    require_executable "$program" "$target"
}

provision_camcasp() {
    CURRENT_STAGE="provision-camcasp"
    if [[ ! -d "$CAMCASP/.git" ]]; then
        [[ "$CAMCASP" == "$REPO_ROOT/camcasp-bin" ]] ||
            fail "CAMCASP override must already be a Git checkout: $CAMCASP"
        git clone --no-checkout "$CAMCASP_URL" "$CAMCASP"
        git -C "$CAMCASP" fetch --depth 1 origin "$CAMCASP_COMMIT"
        git -C "$CAMCASP" checkout --detach "$CAMCASP_COMMIT"
    fi
    [[ "$(git -C "$CAMCASP" rev-parse HEAD)" == "$CAMCASP_COMMIT" ]] ||
        fail "CamCASP checkout is not pinned to $CAMCASP_COMMIT"
    grep -Eq '7\.2\.2|VERSION 7\.2\.2' "$CAMCASP/VERSION" ||
        fail "unexpected CamCASP version in $CAMCASP/VERSION"
    git -C "$CAMCASP" diff --quiet
    git -C "$CAMCASP" diff --cached --quiet
    local program
    for program in "${CAMCASP_PROGRAMS[@]}"; do
        install_camcasp_program "$program"
    done
}

smoke_orient() {
    local exe="$1"
    local log="$2"
    printf 'UNITS BOHR\nFINISH\n' |
        run_logged orient-smoke "$log" "$exe"
    if grep -Eiq 'fatal|segmentation fault|cannot open shared object|error stop' "$log"; then
        fail "Orient smoke test reported an error: $log"
    fi
}

provision_orient() {
    CURRENT_STAGE="provision-orient"
    local checkout="$REPO_ROOT/orient"
    local candidate

    if [[ -n "${ORIENT_EXE:-}" ]]; then
        candidate="$(realpath -m "$ORIENT_EXE")"
    else
        if [[ ! -d "$checkout/.git" ]]; then
            git clone "$ORIENT_URL" "$checkout"
        fi
        git -C "$checkout" fetch --tags origin
        git -C "$checkout" checkout --detach "$ORIENT_REF"
        [[ "$(git -C "$checkout" rev-parse HEAD)" == "$ORIENT_REF" ]] ||
            fail "Orient checkout is not pinned to $ORIENT_REF"
        candidate="$checkout/x86-64/gfortran/exe/orient-5.0.11-ng"
    fi

    require_executable ORIENT_EXE "$candidate"
    if command -v ldd >/dev/null && ! ldd "$candidate" \
        >"$REFERENCE_ROOT/logs/orient-ldd.log" 2>&1; then
        fail "Orient binary is incompatible; build with 'make OPENGL=no'; see orient-ldd.log"
    fi

    ORIENT_BIN_DIR="$REFERENCE_ROOT/tools/orient/bin"
    mkdir -p "$ORIENT_BIN_DIR"
    ln -sfn "$candidate" "$ORIENT_BIN_DIR/orient"
    ORIENT_EXE="$ORIENT_BIN_DIR/orient"
    smoke_orient "$ORIENT_EXE" "$REFERENCE_ROOT/logs/orient-smoke.log"
}

write_psi4_wrapper() {
    CURRENT_STAGE="psi4-wrapper"
    local wrapper="$CAMCASP/bin/psi4.sh"
    cat >"$wrapper" <<EOF
#!/usr/bin/env bash
set -euo pipefail
if (( \${3:-1} > 1 )); then
    exec "$PSI4_EXE" -n "\$3" "\$1" "\$2"
else
    exec "$PSI4_EXE" "\$1" "\$2"
fi
EOF
    chmod 0755 "$wrapper"
    run_logged psi4-version "$REFERENCE_ROOT/logs/psi4-version.log" \
        "$PSI4_EXE" --version
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

    MODE="full"
    case "${1:-}" in
        "") ;;
        --preflight-only) MODE="preflight" ;;
        *) CURRENT_STAGE="arguments"; fail "unknown argument: $1" ;;
    esac

    preflight
    printf 'CamCASP reference root: %s\n' "$REFERENCE_ROOT"
    if [[ "$MODE" == "preflight" ]]; then
        return
    fi

    mkdir -p "$REFERENCE_ROOT/logs"
    provision_camcasp
    provision_orient
    write_psi4_wrapper

    export CAMCASP
    export ARCH=x86-64
    export PATH="$ORIENT_BIN_DIR:$CAMCASP/bin:$PATH"
    export PSIPATH="$CAMCASP/basis/psi4:$CAMCASP/basis/psi4/for-psi4-lib"
    export SCRATCH="$REFERENCE_ROOT/scratch/camcasp"
    export PSI_SCRATCH="$REFERENCE_ROOT/scratch/psi4"
    mkdir -p "$SCRATCH" "$PSI_SCRATCH"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
