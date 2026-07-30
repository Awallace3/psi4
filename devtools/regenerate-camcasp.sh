#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
export LC_ALL=C
export LANG=C
export TZ=UTC
export PYTHONDONTWRITEBYTECODE=1

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

write_sha256_record() {
    local input="$1"
    local output="$2"
    local label="$3"
    local digest
    local temp="$output.tmp.$$"
    digest="$(sha256sum "$input")"
    digest="${digest%% *}"
    printf '%s  %s\n' "$digest" "$label" >"$temp"
    mv -f "$temp" "$output"
}

checksum_log() {
    local log="$1"
    write_sha256_record "$log" "$log.sha256" "$(basename "$log")"
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
    checksum_log "$log"
    if (( rc != 0 )); then
        printf '[%s] failed with exit status %d; retained log: %s\n' \
            "$stage" "$rc" "$log" >&2
        return "$rc"
    fi
}

require_safe_generated_path() {
    local candidate
    local camcasp_source="${CAMCASP_SOURCE_ROOT:-${CAMCASP:-$REPO_ROOT/camcasp-bin}}"
    candidate="$(realpath -m "$1")"
    camcasp_source="$(realpath -m "$camcasp_source")"
    case "$candidate" in
        /|"$HOME"|"$REPO_ROOT"|"$camcasp_source"|"$REPO_ROOT/orient")
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

parse_arguments() {
    CURRENT_STAGE="arguments"
    MODE="full"
    if (( $# > 1 )); then
        fail "expected at most one argument"
        return
    fi
    case "${1:-}" in
        "") ;;
        --preflight-only) MODE="preflight" ;;
        *) fail "unknown argument: $1" ;;
    esac
}

verify_psi4_source_root() {
    local source_root candidate expected checkout status
    source_root="$(realpath -m "$1")"
    candidate="$(realpath -m "$2")"
    expected="$source_root/build_camcasp/stage/bin/psi4"

    [[ "$candidate" == "$expected" ]] || {
        fail "unexpected Psi4 executable: $candidate; expected $expected"
        return
    }
    checkout="$(git -C "$source_root" rev-parse --show-toplevel 2>/dev/null)" || {
        fail "Psi4 source root is not a Git checkout: $source_root"
        return
    }
    [[ "$(realpath -m "$checkout")" == "$source_root" ]] || {
        fail "Psi4 source root is not the checkout root: $source_root"
        return
    }
    status="$(git -C "$source_root" status --porcelain --untracked-files=all)" || {
        fail "could not inspect Psi4 source checkout status: $source_root"
        return
    }
    [[ -z "$status" ]] || {
        fail "Psi4 source checkout is not clean: $source_root"
        return
    }
}

validate_camcasp_root_separation() {
    CAMCASP_SOURCE_ROOT="$(realpath -m "$CAMCASP_SOURCE_ROOT")"
    CAMCASP="$(realpath -m "$CAMCASP")"
    if [[ "$CAMCASP_SOURCE_ROOT" == "$CAMCASP" ||
          "$CAMCASP_SOURCE_ROOT" == "$CAMCASP"/* ||
          "$CAMCASP" == "$CAMCASP_SOURCE_ROOT"/* ]]; then
        fail "CamCASP source and runtime must not contain one another: source=$CAMCASP_SOURCE_ROOT runtime=$CAMCASP"
        return
    fi
}

preflight() {
    CURRENT_STAGE="preflight"
    PSI4_SOURCE_ROOT="${PSI4_SOURCE_ROOT:-$REPO_ROOT}"
    PSI4_SOURCE_ROOT="$(realpath -m "$PSI4_SOURCE_ROOT")"
    PSI4_EXE="${PSI4_EXE:-$PSI4_SOURCE_ROOT/build_camcasp/stage/bin/psi4}"
    CAMCASP_SOURCE_ROOT="${CAMCASP_SOURCE_ROOT:-${CAMCASP:-$REPO_ROOT/camcasp-bin}}"
    ORIENT_REF="${ORIENT_REF:-d8d861098c8f548e2cf230c387c8431d9418650a}"

    PSI4_EXE="$(realpath -m "$PSI4_EXE")"
    CAMCASP_SOURCE_ROOT="$(realpath -m "$CAMCASP_SOURCE_ROOT")"
    CAMCASP="$(realpath -m "$REFERENCE_ROOT/tools/camcasp-runtime")"
    validate_camcasp_root_separation || return
    require_executable PSI4_EXE "$PSI4_EXE"
    verify_psi4_source_root "$PSI4_SOURCE_ROOT" "$PSI4_EXE" || return
    if [[ -n "${ORIENT_EXE:-}" ]]; then
        bind_orient_checkout "$ORIENT_EXE" || return
    fi

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
    gzip -dc "$archive" >"$temp"
    chmod 0755 "$temp"
    if [[ ! -x "$target" ]] || ! cmp -s "$temp" "$target"; then
        mv -f "$temp" "$target"
    else
        rm -f "$temp"
    fi
    ln -sfn "../x86-64/gfortran/exe/$program" "$CAMCASP/bin/$program"
    require_executable "$program" "$target"
}

verify_camcasp_source() {
    local source_root="$(realpath -m "$1")"
    local checkout status marker="$source_root/bin/no_psi4"
    checkout="$(git -C "$source_root" rev-parse --show-toplevel 2>/dev/null)" || {
        fail "CamCASP source root is not a Git checkout: $source_root"
        return
    }
    [[ "$(realpath -m "$checkout")" == "$source_root" ]] || {
        fail "CamCASP source root is not the checkout root: $source_root"
        return
    }
    [[ "$(git -C "$source_root" rev-parse HEAD)" == "$CAMCASP_COMMIT" ]] || {
        fail "CamCASP checkout is not pinned to $CAMCASP_COMMIT"
        return
    }
    grep -Eq "$CAMCASP_VERSION_PATTERN" "$source_root/VERSION" || {
        fail "unexpected CamCASP version in $source_root/VERSION"
        return
    }
    status="$(git -C "$source_root" status --porcelain --untracked-files=all)" || {
        fail "could not inspect CamCASP source checkout: $source_root"
        return
    }
    [[ -z "$status" ]] || {
        fail "CamCASP source checkout is not clean: $source_root"
        return
    }
    git -C "$source_root" ls-files --error-unmatch bin/no_psi4 \
        >/dev/null 2>&1 || {
        fail "CamCASP source requires tracked empty bin/no_psi4"
        return
    }
    [[ -f "$marker" && ! -s "$marker" ]] || {
        fail "CamCASP source requires tracked empty bin/no_psi4"
        return
    }
    [[ "$(git -C "$source_root" cat-file -s HEAD:bin/no_psi4)" == 0 ]] || {
        fail "CamCASP source requires tracked empty bin/no_psi4"
        return
    }
}

extract_and_verify_camcasp_archive() {
    local source_root="$1" commit="$2" archive="$3" target="$4"
    local archived_commit
    archived_commit="$(git get-tar-commit-id <"$archive")" || {
        fail "could not read CamCASP archive commit"
        return
    }
    [[ "$archived_commit" == "$commit" ]] || {
        fail "CamCASP archive commit mismatch: $archived_commit"
        return
    }
    mkdir -p "$target"
    tar -xf "$archive" -C "$target" || {
        fail "could not extract CamCASP source archive"
        return
    }
    git --git-dir="$source_root/.git" --work-tree="$target" \
        diff --exit-code "$commit" -- || {
        fail "CamCASP runtime extraction differs from pinned archive"
        return
    }
    printf 'source=%s\ncommit=%s\nruntime=%s\n' \
        "$source_root" "$commit" "$target"
}

verify_camcasp_runtime() {
    local expected="$(realpath -m "$REFERENCE_ROOT/tools/camcasp-runtime")"
    [[ "$(realpath -m "$CAMCASP")" == "$expected" ]] || {
        fail "unexpected CamCASP runtime path: $CAMCASP"
        return
    }
    [[ "$(realpath -m "$CAMCASP_SOURCE_ROOT")" != "$expected" ]] || {
        fail "CamCASP source and runtime must be distinct"
        return
    }
    [[ -f "$CAMCASP/VERSION" ]] || {
        fail "missing CamCASP runtime VERSION"
        return
    }
    cmp -s "$CAMCASP_SOURCE_ROOT/VERSION" "$CAMCASP/VERSION" || {
        fail "CamCASP runtime VERSION differs from source"
        return
    }
    [[ ! -e "$CAMCASP/bin/no_psi4" ]] || {
        fail "CamCASP runtime sentinel unexpectedly exists: $CAMCASP/bin/no_psi4"
        return
    }
    [[ ! -e "$CAMCASP/.git" ]] || {
        fail "CamCASP runtime unexpectedly contains Git metadata"
        return
    }
}

materialize_camcasp_runtime() {
    CURRENT_STAGE="camcasp-materialize"
    validate_camcasp_root_separation || return
    local archive="$REFERENCE_ROOT/logs/camcasp-source.tar"
    local runtime_temp="$REFERENCE_ROOT/tools/.camcasp-runtime.tmp.$$"
    local source_marker="$CAMCASP_SOURCE_ROOT/bin/no_psi4"
    local runtime_marker="$runtime_temp/bin/no_psi4"
    require_safe_generated_path "$CAMCASP" || return
    [[ "$CAMCASP" == "$(realpath -m "$REFERENCE_ROOT/tools/camcasp-runtime")" ]] || {
        fail "refusing noncanonical CamCASP runtime: $CAMCASP"
        return
    }
    rm -rf "$CAMCASP" "$runtime_temp"
    mkdir -p "$REFERENCE_ROOT/logs" "$REFERENCE_ROOT/tools"
    run_logged camcasp-archive "$REFERENCE_ROOT/logs/camcasp-archive.log" \
        git -C "$CAMCASP_SOURCE_ROOT" archive --format=tar \
            --output="$archive" "$CAMCASP_COMMIT" || return
    write_sha256_record "$archive" "$archive.sha256" camcasp-source.tar
    if ! run_logged camcasp-materialize \
        "$REFERENCE_ROOT/logs/camcasp-materialization.log" \
        extract_and_verify_camcasp_archive "$CAMCASP_SOURCE_ROOT" \
            "$CAMCASP_COMMIT" "$archive" "$runtime_temp"; then
        rm -rf "$runtime_temp"
        return 1
    fi
    [[ -f "$runtime_marker" && ! -s "$runtime_marker" ]] || {
        rm -rf "$runtime_temp"
        fail "archived CamCASP runtime lacks attested empty bin/no_psi4"
        return
    }
    cmp -s "$source_marker" "$runtime_marker" || {
        rm -rf "$runtime_temp"
        fail "runtime bin/no_psi4 differs from pinned source marker"
        return
    }
    rm -f "$runtime_marker"
    [[ ! -e "$runtime_marker" ]] || {
        rm -rf "$runtime_temp"
        fail "runtime sentinel unexpectedly survived removal"
        return
    }
    mv "$runtime_temp" "$CAMCASP"
    verify_camcasp_runtime || return
    write_sha256_record "$source_marker" \
        "$REFERENCE_ROOT/logs/camcasp-source-no_psi4.sha256" \
        camcasp-source/bin/no_psi4
}

provision_camcasp() {
    CURRENT_STAGE="provision-camcasp"
    validate_camcasp_root_separation || return
    if [[ ! -d "$CAMCASP_SOURCE_ROOT/.git" ]]; then
        [[ "$CAMCASP_SOURCE_ROOT" == "$REPO_ROOT/camcasp-bin" ]] || {
            fail "CAMCASP source override must already be a Git checkout: $CAMCASP_SOURCE_ROOT"
            return
        }
        git clone --no-checkout "$CAMCASP_URL" "$CAMCASP_SOURCE_ROOT"
        git -C "$CAMCASP_SOURCE_ROOT" fetch --depth 1 origin "$CAMCASP_COMMIT"
        git -C "$CAMCASP_SOURCE_ROOT" checkout --detach "$CAMCASP_COMMIT"
    fi
    verify_camcasp_source "$CAMCASP_SOURCE_ROOT" || return
    materialize_camcasp_runtime || return
    local program
    for program in "${CAMCASP_PROGRAMS[@]}"; do
        install_camcasp_program "$program" || return
    done
    verify_camcasp_runtime
}

smoke_orient() {
    local exe="$1"
    local log="$2"
    local rc
    if printf 'UNITS BOHR\nFINISH\n' |
        run_logged orient-smoke "$log" "$exe"; then
        :
    else
        rc=$?
        fail "Orient smoke test failed with exit status $rc; build with 'make OPENGL=no'; retained log: $log"
    fi
    if grep -Eiq 'fatal|segmentation fault|cannot open shared object|error stop' "$log"; then
        fail "Orient smoke test reported an error; build with 'make OPENGL=no'; retained log: $log"
    fi
}

read_orient_version() {
    local checkout="$1"
    local version_file="$checkout/VERSION"
    local -a versions=() patchlevels=()
    [[ -f "$version_file" ]] || {
        fail "missing Orient VERSION: $version_file"
        return
    }
    mapfile -t versions < <(
        sed -nE 's/^[[:space:]]*VERSION[[:space:]]*(:=|=)?[[:space:]]*([0-9]+\.[0-9]+)[[:space:]]*$/\2/p' "$version_file"
    )
    mapfile -t patchlevels < <(
        sed -nE 's/^[[:space:]]*PATCHLEVEL[[:space:]]*(:=|=)?[[:space:]]*([0-9]+)[[:space:]]*$/\2/p' "$version_file"
    )
    if (( ${#versions[@]} != 1 || ${#patchlevels[@]} != 1 )); then
        fail "malformed Orient VERSION: $version_file"
        return
    fi
    ORIENT_VERSION="${versions[0]}.${patchlevels[0]}-ng"
    ORIENT_RELATIVE_EXE="x86-64/gfortran/exe/orient-$ORIENT_VERSION"
}

materialize_orient_product_inventory() {
    local checkout="$1"
    local snapshot="$2"
    : >"$snapshot" || {
        fail "could not create Orient product inventory: $snapshot"
        return
    }
    if ! git -C "$checkout" ls-files --others --exclude-standard -z >"$snapshot"; then
        rm -f "$snapshot"
        fail "Orient ordinary product inventory failed: $checkout"
        return
    fi
    if ! git -C "$checkout" ls-files --others --ignored \
        --exclude-standard -z >>"$snapshot"; then
        rm -f "$snapshot"
        fail "Orient ignored product inventory failed: $checkout"
        return
    fi
}

validate_orient_product_snapshot() {
    local snapshot="$1"
    local path
    while IFS= read -r -d '' path; do
        case "$path" in
            src/version.f90|"$ORIENT_RELATIVE_EXE") ;;
            *)
                if [[ ! "$path" =~ ^x86-64/gfortran/exe/[^/]+\.(o|mod)$ ]]; then
                    fail "unexpected untracked/ignored Orient product: $path"
                    return
                fi
                ;;
        esac
    done <"$snapshot"
}

prepare_orient_products() (
    local checkout="$1"
    local snapshot
    mkdir -p "$REFERENCE_ROOT/logs"
    snapshot="$(mktemp "$REFERENCE_ROOT/logs/.orient-products.pre.XXXXXX")" || {
        fail "could not allocate pre-build Orient product inventory"
        return
    }
    trap 'rm -f "$snapshot"' EXIT
    materialize_orient_product_inventory "$checkout" "$snapshot" || return
    validate_orient_product_snapshot "$snapshot" || return
    local path
    while IFS= read -r -d '' path; do
        rm -f "$checkout/$path"
    done <"$snapshot"
)

validate_current_orient_products() (
    local checkout="$1"
    local label="$2"
    local snapshot
    mkdir -p "$REFERENCE_ROOT/logs"
    snapshot="$(mktemp "$REFERENCE_ROOT/logs/.orient-products.$label.XXXXXX")" || {
        fail "could not allocate $label Orient product inventory"
        return
    }
    trap 'rm -f "$snapshot"' EXIT
    materialize_orient_product_inventory "$checkout" "$snapshot" || return
    validate_orient_product_snapshot "$snapshot"
)

verify_orient_checkout() {
    local checkout="$1"
    local candidate="$2"
    local product_check="${3:-with-products}"

    [[ "$(git -C "$checkout" rev-parse HEAD)" == "$ORIENT_REF" ]] || {
        fail "Orient checkout is not pinned to $ORIENT_REF"
        return
    }
    git -C "$checkout" diff --quiet || {
        fail "Orient tracked source checkout is not clean: $checkout"
        return
    }
    git -C "$checkout" diff --cached --quiet || {
        fail "Orient tracked source checkout index is not clean: $checkout"
        return
    }
    read_orient_version "$checkout" || return
    [[ "$(realpath -m "$candidate")" == "$(realpath -m "$checkout/$ORIENT_RELATIVE_EXE")" ]] || {
        fail "unexpected Orient artifact: $candidate"
        return
    }
    if [[ "$product_check" == "with-products" ]]; then
        validate_current_orient_products "$checkout" verify || return
    fi
}

record_orient_executable() {
    local candidate="$1"
    local log="$REFERENCE_ROOT/logs/orient-executable.log"
    mkdir -p "$REFERENCE_ROOT/logs"
    printf 'selected executable: %s\n' "$candidate" >"$log"
    checksum_log "$log"
    write_sha256_record "$candidate" \
        "$REFERENCE_ROOT/logs/orient-executable.sha256" orient
}

bind_orient_checkout() {
    local candidate checkout
    local product_check="${2:-with-products}"
    candidate="$(realpath -m "$1")"
    checkout="$(git -C "$(dirname "$candidate")" rev-parse --show-toplevel 2>/dev/null)" || {
        fail "Orient override must be inside a Git checkout: $candidate"
        return
    }
    checkout="$(realpath -m "$checkout")"
    verify_orient_checkout "$checkout" "$candidate" "$product_check" || return
    ORIENT_SOURCE_ROOT="$checkout"
}

provision_orient() {
    CURRENT_STAGE="provision-orient"
    local checkout="$REPO_ROOT/orient"
    local candidate target

    if [[ -n "${ORIENT_EXE:-}" ]]; then
        candidate="$(realpath -m "$ORIENT_EXE")"
    else
        if [[ ! -d "$checkout/.git" ]]; then
            git clone "$ORIENT_URL" "$checkout"
        fi
        git -C "$checkout" fetch --tags origin
        git -C "$checkout" checkout --detach "$ORIENT_REF"
        read_orient_version "$checkout" || return
        candidate="$checkout/$ORIENT_RELATIVE_EXE"
    fi

    bind_orient_checkout "$candidate" without-products || return
    if git -C "$ORIENT_SOURCE_ROOT" ls-files --error-unmatch \
        "$ORIENT_RELATIVE_EXE" >/dev/null 2>&1; then
        fail "derived Orient build artifact must not be tracked: $ORIENT_RELATIVE_EXE"
        return
    fi
    prepare_orient_products "$ORIENT_SOURCE_ROOT" || return
    target="orient-$ORIENT_VERSION"
    (
        cd "$ORIENT_SOURCE_ROOT/x86-64/gfortran/exe"
        run_logged orient-build "$REFERENCE_ROOT/logs/orient-build.log" \
            env -u MAKEFLAGS -u MFLAGS make -j1 \
                -f "$ORIENT_SOURCE_ROOT/Makefile" \
                OPENGL=no BASE="$ORIENT_SOURCE_ROOT" "$target"
    )
    verify_orient_checkout "$ORIENT_SOURCE_ROOT" "$candidate" without-products || return
    validate_current_orient_products "$ORIENT_SOURCE_ROOT" post || return
    [[ -f "$candidate" && -x "$candidate" ]] || {
        fail "missing built Orient artifact: $candidate"
        return
    }
    require_executable ORIENT_EXE "$candidate"
    record_orient_executable "$candidate"
    if command -v ldd >/dev/null; then
        local ldd_log="$REFERENCE_ROOT/logs/orient-ldd.log"
        local ldd_rc
        set +e
        ldd "$candidate" >"$ldd_log" 2>&1
        ldd_rc=$?
        set -e
        checksum_log "$ldd_log"
        if (( ldd_rc != 0 )); then
            fail "Orient binary is incompatible; build with 'make OPENGL=no'; retained log: $ldd_log"
        fi
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
if [[ "\${1:-}" == "--version" ]]; then
    exec "$PSI4_EXE" --version
fi
input="\${1:-}"
output="\${2:-}"
[[ -n "\$input" && -n "\$output" ]] || { echo "psi4.sh requires input and output paths" >&2; exit 64; }
resolved_input="\$(realpath -e -- "\$input")" || { echo "psi4.sh input does not exist: \$input" >&2; exit 66; }
if [[ -v CAMCASP_PSI4_EVIDENCE_DIR && -z "\$CAMCASP_PSI4_EVIDENCE_DIR" ]]; then
    echo "psi4.sh evidence directory is empty" >&2
    exit 65
fi
evidence_dir="\${CAMCASP_PSI4_EVIDENCE_DIR:-}"
python -P - "\$resolved_input" "\$evidence_dir" "$REFERENCE_ROOT" <<'PY'
import os
import re
import stat
import sys
import tempfile
from pathlib import Path

path = Path(sys.argv[1])
evidence_text = sys.argv[2]
reference_root = Path(sys.argv[3]).resolve(strict=True)
lines = path.read_text().splitlines(keepends=True)
starts = []
for index, line in enumerate(lines):
    stripped = line.strip()
    if not stripped.startswith("#") and re.match(
        r"^molecule(?:\s+\S+)?\s*\{\s*$", stripped, re.IGNORECASE
    ):
        starts.append(index)
if len(starts) != 1:
    raise SystemExit(f"psi4.sh requires exactly one molecule block, found {len(starts)}")
start = starts[0]
ends = [index for index in range(start + 1, len(lines)) if lines[index].strip() == "}"]
if not ends:
    raise SystemExit("psi4.sh molecule block is unterminated")
end = ends[0]
symmetries = []
for index, line in enumerate(lines):
    stripped = line.split("#", 1)[0].strip()
    if not stripped:
        continue
    match = re.match(r"^symmetry\s+(\S+)\s*$", stripped, re.IGNORECASE)
    if match:
        symmetries.append((index, match.group(1)))
if len(symmetries) > 1:
    raise SystemExit("psi4.sh rejects duplicate symmetry directives")
if symmetries and symmetries[0][1].casefold() != "c1":
    raise SystemExit(f"psi4.sh rejects conflicting symmetry {symmetries[0][1]!r}")
if symmetries and not (start < symmetries[0][0] < end):
    raise SystemExit("psi4.sh rejects symmetry outside the molecule block")
if not symmetries:
    lines.insert(start + 1, "  symmetry c1\n")
    mode = stat.S_IMODE(path.stat().st_mode)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.writelines(lines)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_name, mode)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise

if evidence_text:
    evidence_raw = Path(evidence_text)
    if not evidence_raw.is_absolute():
        raise SystemExit("psi4.sh evidence directory must be absolute")
    try:
        evidence_dir = evidence_raw.resolve(strict=True)
    except (FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(f"psi4.sh evidence directory is invalid: {evidence_raw}") from exc
    if not evidence_dir.is_dir():
        raise SystemExit(f"psi4.sh evidence path is not a directory: {evidence_dir}")
    if Path(os.path.abspath(evidence_raw)) != evidence_dir:
        raise SystemExit("psi4.sh evidence directory must not traverse symlinks")
    safe_root = (reference_root / "work").resolve(strict=True)
    try:
        evidence_dir.relative_to(safe_root)
    except ValueError as exc:
        raise SystemExit(
            f"psi4.sh evidence directory must be inside {safe_root}"
        ) from exc
    evidence = evidence_dir / "H2O_A.executed.in"
    if evidence == path:
        raise SystemExit("psi4.sh evidence path must differ from executed input")
    payload = path.read_bytes()
    mode = stat.S_IMODE(path.stat().st_mode)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{evidence.name}.", suffix=".tmp", dir=evidence_dir
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_name, mode)
        os.replace(temporary_name, evidence)
        directory_descriptor = os.open(evidence_dir, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    if evidence.read_bytes() != path.read_bytes():
        raise SystemExit("psi4.sh executed-input evidence differs from resolved input")
PY
if (( \${3:-1} > 1 )); then
    exec "$PSI4_EXE" -n "\$3" "\$resolved_input" "\$output"
else
    exec "$PSI4_EXE" "\$resolved_input" "\$output"
fi
EOF
    chmod 0755 "$wrapper"
    mkdir -p "$REFERENCE_ROOT/logs"
    write_sha256_record "$wrapper" \
        "$REFERENCE_ROOT/logs/camcasp-psi4-wrapper.sha256" \
        camcasp-runtime/bin/psi4.sh
    run_logged psi4-version "$REFERENCE_ROOT/logs/psi4-version.log" \
        "$wrapper" --version
}

camcasp_runtime_attestation_payload() {
    local action="$1"
    local attestation="$REFERENCE_ROOT/logs/camcasp-runtime-attestation.json"
    python -P - "$action" "$CAMCASP_SOURCE_ROOT" "$CAMCASP" \
        "$REFERENCE_ROOT/logs/camcasp-source.tar" "$CAMCASP_COMMIT" \
        "$attestation" <<'PY'
import gzip
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

(action, source_text, runtime_text, archive_text, expected_commit, output_text) = sys.argv[1:]
source = Path(source_text).resolve()
runtime = Path(runtime_text).resolve()
archive = Path(archive_text).resolve()
output = Path(output_text)
programs = ("camcasp", "cluster", "process", "pfit", "casimir")


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def record(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"missing file: {path}")
    return {"path": str(path.resolve()), "sha256": digest_bytes(path.read_bytes())}


def archive_commit(path):
    result = subprocess.run(
        ["git", "get-tar-commit-id"],
        input=path.read_bytes(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("could not revalidate git archive embedded commit")
    return result.stdout.decode().strip()


def current_payload():
    embedded_commit = archive_commit(archive)
    if embedded_commit != expected_commit:
        raise ValueError(
            f"git archive embedded commit {embedded_commit!r} != {expected_commit!r}"
        )
    sentinel = source / "bin" / "no_psi4"
    source_sentinel = record(sentinel)
    if sentinel.read_bytes() != b"":
        raise ValueError("source sentinel is not empty")
    runtime_sentinel = runtime / "bin" / "no_psi4"
    archives = {}
    executables = {}
    symlinks = {}
    for name in programs:
        compressed = runtime / "x86-64" / "gfortran" / f"{name}.gz"
        executable = runtime / "x86-64" / "gfortran" / "exe" / name
        archive_record = record(compressed)
        executable_record = record(executable)
        with gzip.open(compressed, "rb") as handle:
            decompressed = handle.read()
        decompressed_digest = digest_bytes(decompressed)
        if decompressed_digest != executable_record["sha256"]:
            raise ValueError(f"{name} executable differs from preserved archive")
        archive_record["decompressed_sha256"] = decompressed_digest
        archives[name] = archive_record
        executables[name] = executable_record
        link = runtime / "bin" / name
        if not link.is_symlink():
            raise ValueError(f"missing bin symlink: {link}")
        symlinks[name] = {
            "path": str(link.absolute()),
            "target": os.readlink(link),
        }
    return {
        "schema_version": 1,
        "git_archive": {
            **record(archive),
            "embedded_commit": embedded_commit,
        },
        "source_sentinel": source_sentinel,
        "runtime_sentinel": {
            "path": str(runtime_sentinel.absolute()),
            "absent": not runtime_sentinel.exists() and not runtime_sentinel.is_symlink(),
        },
        "runtime_version": record(runtime / "VERSION"),
        "psi4_wrapper": record(runtime / "bin" / "psi4.sh"),
        "archives": archives,
        "executables": executables,
        "bin_symlinks": symlinks,
        "drivers": {
            name: record(runtime / "bin" / name)
            for name in ("runcamcasp.py", "localize.py")
        },
    }


try:
    current = current_payload()
    if not current["runtime_sentinel"]["absent"]:
        raise ValueError("runtime sentinel unexpectedly exists")
    if action == "write":
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
        temporary.write_text(
            json.dumps(current, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    elif action == "verify":
        recorded = json.loads(output.read_text(encoding="utf-8"))
        if recorded != current:
            raise ValueError("recorded install-time state differs from current runtime")
    else:
        raise ValueError(f"unknown attestation action: {action}")
except Exception as exc:
    print(f"runtime attestation mismatch: {exc}", file=sys.stderr)
    raise SystemExit(1)
PY
}

write_camcasp_runtime_attestation() {
    CURRENT_STAGE="camcasp-runtime-attestation"
    verify_camcasp_source "$CAMCASP_SOURCE_ROOT" || return
    verify_camcasp_runtime || return
    camcasp_runtime_attestation_payload write || {
        fail "could not create CamCASP runtime attestation"
        return
    }
    write_sha256_record \
        "$REFERENCE_ROOT/logs/camcasp-runtime-attestation.json" \
        "$REFERENCE_ROOT/logs/camcasp-runtime-attestation.json.sha256" \
        camcasp-runtime-attestation.json
}

verify_camcasp_runtime_attestation() {
    CURRENT_STAGE="camcasp-runtime-attestation"
    local attestation="$REFERENCE_ROOT/logs/camcasp-runtime-attestation.json"
    local checksum="$attestation.sha256"
    local expected actual
    [[ -f "$attestation" && -f "$checksum" ]] || {
        fail "runtime attestation mismatch: attestation or checksum is missing"
        return
    }
    expected="$(awk 'NR == 1 {print $1}' "$checksum")"
    actual="$(sha256sum "$attestation")"
    actual="${actual%% *}"
    [[ "$expected" =~ ^[0-9a-f]{64}$ && "$actual" == "$expected" ]] || {
        fail "runtime attestation mismatch: attestation checksum changed"
        return
    }
    verify_camcasp_runtime || {
        fail "runtime attestation mismatch: runtime invariants changed"
        return
    }
    camcasp_runtime_attestation_payload verify || {
        fail "runtime attestation mismatch: sealed runtime changed"
        return
    }
}

prepare_layout() {
    CURRENT_STAGE="layout"
    require_safe_generated_path "$REFERENCE_ROOT"
    rm -f \
        "$REFERENCE_ROOT/atomic-polarizabilities.json" \
        "$REFERENCE_ROOT/atomic-polarizabilities.json.sha256"
    rm -rf \
        "$REFERENCE_ROOT/inputs" \
        "$REFERENCE_ROOT/work" \
        "$REFERENCE_ROOT/scratch" \
        "$REFERENCE_ROOT/logs"
    mkdir -p \
        "$REFERENCE_ROOT/inputs" \
        "$REFERENCE_ROOT/work" \
        "$REFERENCE_ROOT/scratch/camcasp" \
        "$REFERENCE_ROOT/scratch/psi4" \
        "$REFERENCE_ROOT/logs" \
        "$REFERENCE_ROOT/tools"
}

export_reference_environment() {
    CURRENT_STAGE="environment"
    export CAMCASP
    export ARCH=x86-64
    export PATH="$ORIENT_BIN_DIR:$CAMCASP/bin:$PATH"
    export PSIPATH="$CAMCASP/basis/psi4:$CAMCASP/basis/psi4/for-psi4-lib"
    export SCRATCH="$REFERENCE_ROOT/scratch/camcasp"
    export PSI_SCRATCH="$REFERENCE_ROOT/scratch/psi4"
    mkdir -p "$SCRATCH" "$PSI_SCRATCH"
}

write_inputs() {
    CURRENT_STAGE="inputs"
    cat >"$REFERENCE_ROOT/inputs/H2O.clt" <<'EOF'
! Canonical CamCASP/Psi4 H2O atomic-polarizability reference.
Global
  Units Bohr Degrees
  Overwrite Yes
End

Molecule H2O
  I.P.  12.62063 eV
  HOMO -0.3989
  O   8.0    0.0000000000    0.0000000000    0.0000000000  Type O
  H1  1.0   -1.4536519600    0.0000000000   -1.1216873200  Type H
  H2  1.0    1.4536519600    0.0000000000   -1.1216873200  Type H
End

Run-type properties
  Molecule H2O
  Basis aVTZ
  SCFcode psi4
  Method DFT
  Functional PBE0
  Kernel ALDA+CHF
  Options Tests
  Orient file
  Process file
  Sites file
End

Finish
EOF

    cat >"$REFERENCE_ROOT/inputs/H2O.axes" <<'EOF'
Axes
  H1  z global Z x from H2 to H1
  H2  z global Z x from H1 to H2
End
EOF
    write_sha256_record "$REFERENCE_ROOT/inputs/H2O.clt" \
        "$REFERENCE_ROOT/inputs/H2O.clt.sha256" H2O.clt
    write_sha256_record "$REFERENCE_ROOT/inputs/H2O.axes" \
        "$REFERENCE_ROOT/inputs/H2O.axes.sha256" H2O.axes
}

declare -ag NL4_FORMAT_A_FILES=()
declare -ag NL4_FILES=()
declare -ag P2P_FILES=()
declare -ag PSI4_INPUT_FILES=()
declare -ag PSI4_EXECUTED_INPUT_FILES=()
declare -ag PSI4_OUTPUT_FILES=()

discover_camcasp_artifacts() {
    local job_dir="$1"
    local -a all_nl4_files=()
    mapfile -t all_nl4_files < <(
        find "$job_dir/OUT" -maxdepth 1 -type f -name '*NL4*.pol' -print | sort
    )
    mapfile -t NL4_FORMAT_A_FILES < <(
        find "$job_dir/OUT" -maxdepth 1 -type f -name '*NL4_fmtA.pol' -print | sort
    )
    mapfile -t NL4_FILES < <(
        find "$job_dir/OUT" -maxdepth 1 -type f -name '*NL4_fmtB.pol' -print | sort
    )
    mapfile -t P2P_FILES < <(
        find "$job_dir/OUT" -maxdepth 1 -type f -name '*.p2p' -print | sort
    )
    mapfile -t PSI4_INPUT_FILES < <(
        find "$job_dir" -maxdepth 2 -type f \
            \( -name 'H2O_A.in' -o -name 'H2O_A.dat' \) -print | sort
    )
    mapfile -t PSI4_EXECUTED_INPUT_FILES < <(
        find "$job_dir" -maxdepth 2 -type f -name 'H2O_A.executed.in' -print | sort
    )
    mapfile -t PSI4_OUTPUT_FILES < <(
        find "$job_dir" -maxdepth 2 -type f -name 'H2O_A.out' -print | sort
    )
    (( ${#NL4_FORMAT_A_FILES[@]} == 1 )) || {
        fail "expected exactly one NL4 format-A polarizability file, found ${#NL4_FORMAT_A_FILES[@]}"
        return
    }
    (( ${#NL4_FILES[@]} == 1 )) || {
        fail "expected exactly one NL4 format-B polarizability file, found ${#NL4_FILES[@]}"
        return
    }
    (( ${#all_nl4_files[@]} == 2 )) || {
        fail "expected exactly two total NL4 polarizability files, found ${#all_nl4_files[@]}"
        return
    }
    [[ -s "${NL4_FORMAT_A_FILES[0]}" ]] || {
        fail "NL4 format-A polarizability file is empty: ${NL4_FORMAT_A_FILES[0]}"
        return
    }
    [[ -s "${NL4_FILES[0]}" ]] || {
        fail "NL4 format-B polarizability file is empty: ${NL4_FILES[0]}"
        return
    }
    (( ${#P2P_FILES[@]} == 1 )) || {
        fail "expected exactly one p2p file, found ${#P2P_FILES[@]}"
        return
    }
    (( ${#PSI4_INPUT_FILES[@]} == 1 )) || {
        fail "expected exactly one generated Psi4 input, found ${#PSI4_INPUT_FILES[@]}"
        return
    }
    [[ -s "${PSI4_INPUT_FILES[0]}" ]] || {
        fail "generated Psi4 input is empty: ${PSI4_INPUT_FILES[0]}"
        return
    }
    (( ${#PSI4_EXECUTED_INPUT_FILES[@]} == 1 )) || {
        fail "expected exactly one executed Psi4 input, found ${#PSI4_EXECUTED_INPUT_FILES[@]}"
        return
    }
    [[ -s "${PSI4_EXECUTED_INPUT_FILES[0]}" ]] || {
        fail "executed Psi4 input is empty: ${PSI4_EXECUTED_INPUT_FILES[0]}"
        return
    }
    (( ${#PSI4_OUTPUT_FILES[@]} == 1 )) || {
        fail "expected exactly one generated Psi4 output, found ${#PSI4_OUTPUT_FILES[@]}"
        return
    }
    [[ -s "${PSI4_OUTPUT_FILES[0]}" ]] || {
        fail "generated Psi4 output is empty: ${PSI4_OUTPUT_FILES[0]}"
        return
    }
}

run_camcasp() {
    CURRENT_STAGE="camcasp"
    local job_dir="$REFERENCE_ROOT/work/H2O"
    local inputs_dir="$REFERENCE_ROOT/inputs"
    require_safe_generated_path "$job_dir"
    rm -rf "$job_dir"
    (
        cd "$inputs_dir"
        export CAMCASP_PSI4_EVIDENCE_DIR="$job_dir"
        run_logged camcasp "$REFERENCE_ROOT/logs/camcasp.log" \
            "$CAMCASP/bin/runcamcasp.py" H2O \
            --clt H2O.clt \
            --directory "$job_dir" \
            --ifexists delete \
            --scfcode psi4 \
            --queue none \
            --scratch "$SCRATCH" \
            --cores "${CAMCASP_REFERENCE_CORES:-1}" \
            --verbosity 1 \
            --debug
    ) || return
    [[ ! -e "$inputs_dir/H2O.clt.clout" ]] || {
        fail "unexpected input-directory CamCASP output: $inputs_dir/H2O.clt.clout"
        return
    }
    cp "$inputs_dir/H2O.axes" "$job_dir/H2O.axes"

    local generated
    for generated in H2O.cks H2O.clt.clout H2O.ornt H2O.prss \
        H2O_casimir.prss H2O.sites; do
        [[ -s "$job_dir/$generated" ]] || fail "missing generated $generated"
    done

    discover_camcasp_artifacts "$job_dir" || return
}

attest_generated_protocol() {
    CURRENT_STAGE="attest-protocol"
    local job_dir="$REFERENCE_ROOT/work/H2O"
    (( ${#PSI4_INPUT_FILES[@]} == 1 &&
       ${#PSI4_EXECUTED_INPUT_FILES[@]} == 1 &&
       ${#PSI4_OUTPUT_FILES[@]} == 1 )) ||
        fail "generated/executed Psi4 input and output cardinality was not established"
    run_logged attest-protocol "$REFERENCE_ROOT/logs/attest-protocol.log" \
        python -P "$REPO_ROOT/devtools/camcasp_reference.py" attest-protocol \
        --clt "$REFERENCE_ROOT/inputs/H2O.clt" \
        --cks "$job_dir/H2O.cks" \
        --camcasp-log "$REFERENCE_ROOT/logs/camcasp.log" \
        --psi4-input "${PSI4_EXECUTED_INPUT_FILES[0]}" \
        --psi4-output "${PSI4_OUTPUT_FILES[0]}"
}

checksum_job_files() {
    CURRENT_STAGE="checksum-artifacts"
    local job_dir="$1"
    local artifact
    while IFS= read -r -d '' artifact; do
        write_sha256_record "$artifact" "$artifact.sha256" "${artifact#"$job_dir"/}"
    done < <(
        find "$job_dir" -type f ! -name '*.sha256' -print0 | sort -z
    )
}

run_localize() {
    CURRENT_STAGE="localize-refine-dispersion"
    local job_dir="$REFERENCE_ROOT/work/H2O"
    (( ${#NL4_FILES[@]} == 1 )) || fail "NL4 artifact cardinality is not one"
    (
        cd "$job_dir"
        run_logged localize-refine-dispersion \
            "$REFERENCE_ROOT/logs/localize-refine-dispersion.log" \
            "$CAMCASP/bin/localize.py" H2O \
                --axes H2O.axes \
                --polfile "${NL4_FILES[0]}" \
                --format NEW \
                --limit "$LOCALIZATION_LIMIT" \
                --wsmlimit "$WSM_LIMIT" \
                --hlimit "$HYDROGEN_LIMIT" \
                --loc LW \
                --weight "$PFIT_WEIGHT" \
                --weightcoeff "$PFIT_WEIGHT_COEFF" \
                --cutoff "$PFIT_CUTOFF" \
                --force loc refine disp \
                --debug
    )
    run_logged validate-artifacts "$REFERENCE_ROOT/logs/validate-artifacts.log" \
        python -P "$REPO_ROOT/devtools/camcasp_reference.py" \
        validate-artifacts --work-dir "$job_dir" --job H2O

    checksum_job_files "$job_dir"
}

validate_hydrogen_model() {
    local pdef="$1"
    grep -Eiq '^\s*H1\s+H1\s+' "$pdef" ||
        fail "generated model has no H1 parameter definitions: $pdef"
    grep -Eiq '^\s*H2\s+H2\s+COPY\s+H1\s+H1\s*$' "$pdef" ||
        fail "generated model lacks the required H2 COPY H1 edit: $pdef"
}

require_reviewed_pdef() {
    CURRENT_STAGE="review-pdef"
    local pdef="$REFERENCE_ROOT/work/H2O/H2O.pdef"
    local checksum_file="$REFERENCE_ROOT/work/H2O/H2O.pdef.sha256"
    local digest
    [[ -s "$pdef" ]] || fail "missing generated model: $pdef"
    validate_hydrogen_model "$pdef"
    digest="$(sha256sum "$pdef")"
    digest="${digest%% *}"
    printf '%s  %s\n' "$digest" "$pdef" >"$checksum_file"

    if [[ "${CAMCASP_PDEF_SHA256:-}" != "$digest" ]]; then
        cat >&2 <<EOF
[review-pdef] generated L3 model requires scientific review.
Inspect:
  $pdef
  $REFERENCE_ROOT/logs/localize-refine-dispersion.log
Verify O/H site types, H1/H2 COPY constraints, L3 entries on every atom,
weight 4, weight coefficient 0.001, and cutoff 0.0001.
Recorded checksum:
  $digest
After review, rerun with CAMCASP_PDEF_SHA256 set to that exact digest.
No reference JSON has been written.
EOF
        return 78
    fi
}

write_manifest() {
    CURRENT_STAGE="manifest"
    local job_dir="$REFERENCE_ROOT/work/H2O"
    local manifest="$REFERENCE_ROOT/work/manifest.json"
    local temp="$manifest.tmp.$$"
    (( ${#NL4_FORMAT_A_FILES[@]} == 1 && ${#NL4_FILES[@]} == 1 &&
       ${#P2P_FILES[@]} == 1 && ${#PSI4_INPUT_FILES[@]} == 1 &&
       ${#PSI4_EXECUTED_INPUT_FILES[@]} == 1 &&
       ${#PSI4_OUTPUT_FILES[@]} == 1 )) ||
        fail "source artifact cardinalities are incomplete"
    verify_orient_checkout "$ORIENT_SOURCE_ROOT" "$ORIENT_EXE" without-products || return
    validate_current_orient_products "$ORIENT_SOURCE_ROOT" manifest || return
    verify_psi4_source_root "$PSI4_SOURCE_ROOT" "$PSI4_EXE" || return
    verify_camcasp_source "$CAMCASP_SOURCE_ROOT" || return
    verify_camcasp_runtime_attestation || return
    run_logged write-manifest "$REFERENCE_ROOT/logs/write-manifest.log" \
        python -P - "$REPO_ROOT" "$SCRIPT_PATH" "$CAMCASP_SOURCE_ROOT" "$CAMCASP" \
        "$ORIENT_SOURCE_ROOT" "$ORIENT_EXE" \
        "$PSI4_SOURCE_ROOT" "$PSI4_EXE" "$REFERENCE_ROOT" \
        "${PSI4_INPUT_FILES[0]}" "${PSI4_EXECUTED_INPUT_FILES[0]}" \
        "${PSI4_OUTPUT_FILES[0]}" "${P2P_FILES[0]}" \
        "${NL4_FORMAT_A_FILES[0]}" "$temp" <<'PY'
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

(repo, generator, camcasp_source, camcasp, orient_source, orient_exe,
 psi4_source, psi4_exe, reference, psi4_generated_input,
 psi4_executed_input, psi4_output, p2p, nl4_format_a, output) = map(
    Path, sys.argv[1:]
)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def record(path):
    path = path.resolve()
    return {"path": str(path), "sha256": sha(path)}

def git_at(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

def orient_version(root):
    text = (root / "VERSION").read_text()
    version = re.findall(
        r"^\s*VERSION\s*(?::=|=)?\s*([0-9]+\.[0-9]+)\s*$", text, re.MULTILINE
    )
    patch = re.findall(
        r"^\s*PATCHLEVEL\s*(?::=|=)?\s*([0-9]+)\s*$", text, re.MULTILINE
    )
    if len(version) != 1 or len(patch) != 1:
        raise ValueError(f"malformed Orient VERSION: {root / 'VERSION'}")
    return f"{version[0]}.{patch[0]}-ng"

job = reference / "work" / "H2O"
inputs_dir = reference / "inputs"
programs = ("camcasp", "cluster", "process", "pfit", "casimir")
repo_status = git_at(repo, "status", "--porcelain", "--untracked-files=no")
psi_status = git_at(psi4_source, "status", "--porcelain", "--untracked-files=all")
psi_version = (reference / "logs" / "psi4-version.log").read_text().strip()
document = {
    "schema_version": 1,
    "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "generator": {"path": str(generator.relative_to(repo)), "sha256": sha(generator)},
    "repository": {
        "commit": git_at(repo, "rev-parse", "HEAD"),
        "dirty": bool(repo_status),
    },
    "tools": {
        "camcasp": {
            "version": (camcasp_source / "VERSION").read_text().strip(),
            "commit": git_at(camcasp_source, "rev-parse", "HEAD"),
            "executables": {
                name: record(camcasp / "bin" / name) for name in programs
            },
        },
        "orient": {
            "version": orient_version(orient_source),
            "commit": git_at(orient_source, "rev-parse", "HEAD"),
            "executable": record(orient_exe),
        },
        "psi4": {
            "version": psi_version,
            "commit": git_at(psi4_source, "rev-parse", "HEAD"),
            "dirty": bool(psi_status),
            "executable": record(psi4_exe),
        },
    },
    "scientific_protocol": {
        "geometry": {
            "units": "bohr", "charge": 0, "multiplicity": 1,
            "atom_order": ["O", "H1", "H2"],
            "atoms": [
                {"label": "O", "element": "O", "xyz": [0.0, 0.0, 0.0]},
                {"label": "H1", "element": "H", "xyz": [-1.45365196, 0.0, -1.12168732]},
                {"label": "H2", "element": "H", "xyz": [1.45365196, 0.0, -1.12168732]},
            ],
            "orientation": ["symmetry c1", "no_com", "no_reorient"],
        },
        "electronic_structure": {
            "method": "PBE0", "basis": "aug-cc-pVTZ", "camcasp_basis": "aVTZ",
            "asymptotic_correction": "Psi4 GRAC", "ionization_potential_ev": 12.62063,
            "homo_hartree": -0.3989, "kernel": "ALDA+CHF", "grid": "Options Tests",
        },
        "frequency_grid": {"kind": "Gauss-Legendre", "nonzero_count": 10, "scale_au": 0.5},
        "model": {
            "nonlocal_rank": 4, "localization_method": "LW", "localization_limit": 3,
            "wsm_limit": 3, "hydrogen_limit": 3, "pfit_weight": 4,
            "pfit_weight_coefficient": 0.001, "pfit_cutoff": 0.0001,
        },
    },
    "frequencies": {"units": "hartree", "values": [], "squared_source_values": []},
    "polarizabilities": {
        "units": "atomic units", "spherical_frame": "atom-local real spherical",
        "cartesian_frame": "global Cartesian", "frequency_blocks": [],
    },
    "dispersion": {
        "component": "00 00 0", "atom_order": ["O", "H1", "H2"],
        "units": {
            "C6": "hartree * bohr^6", "C8": "hartree * bohr^8",
            "C10": "hartree * bohr^10", "C12": "hartree * bohr^12",
        },
        "matrices": {},
    },
    "inputs": {
        "H2O.clt": record(inputs_dir / "H2O.clt"),
        "H2O.axes": record(inputs_dir / "H2O.axes"),
    },
    "sources": {
        "cks": record(job / "H2O.cks"),
        "cluster_output": record(job / "H2O.clt.clout"),
        "psi4_generated_input": record(psi4_generated_input),
        "psi4_executed_input": record(psi4_executed_input),
        "psi4_input": record(psi4_executed_input),
        "psi4_output": record(psi4_output),
        "p2p": record(p2p),
        "nonlocal_pol_format_a": record(nl4_format_a),
        "pdef": record(job / "H2O.pdef"),
        "camcasp_no_psi4": record(camcasp_source / "bin" / "no_psi4"),
        "camcasp_psi4_wrapper": record(camcasp / "bin" / "psi4.sh"),
        "camcasp_archive": record(reference / "logs" / "camcasp-source.tar"),
        "camcasp_materialization_log": record(
            reference / "logs" / "camcasp-materialization.log"
        ),
        "camcasp_runtime_attestation": record(
            reference / "logs" / "camcasp-runtime-attestation.json"
        ),
    },
}
with output.open("w", encoding="utf-8", newline="\n") as handle:
    json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")
PY
    mv -f "$temp" "$manifest"
    write_sha256_record "$manifest" "$manifest.sha256" manifest.json
}

build_reference_json() {
    CURRENT_STAGE="build-reference"
    local job_dir="$REFERENCE_ROOT/work/H2O"
    (( ${#NL4_FILES[@]} == 1 )) || fail "NL4 artifact cardinality is not one"
    run_logged build-reference "$REFERENCE_ROOT/logs/build-reference.log" \
        python -P "$REPO_ROOT/devtools/camcasp_reference.py" build \
        --manifest "$REFERENCE_ROOT/work/manifest.json" \
        --frequency-file "${NL4_FILES[0]}" \
        --refined-file "$job_dir/H2O_ref_wt4_L3_0f10.pol" \
        --casimir-file "$job_dir/H2O_ref_wt4_L3_C12.pot" \
        --axes-file "$REFERENCE_ROOT/inputs/H2O.axes" \
        --output "$REFERENCE_ROOT/atomic-polarizabilities.json"
    cat "$REFERENCE_ROOT/logs/build-reference.log"
    write_sha256_record "$REFERENCE_ROOT/atomic-polarizabilities.json" \
        "$REFERENCE_ROOT/atomic-polarizabilities.json.sha256" \
        atomic-polarizabilities.json
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
    parse_arguments "$@"
    preflight
    if [[ "$MODE" == "preflight" ]]; then
        printf 'Preflight passed for %s\n' "$PSI4_EXE"
        return 0
    fi

    printf 'CamCASP reference root: %s\n' "$REFERENCE_ROOT"
    prepare_layout
    provision_orient
    provision_camcasp
    write_psi4_wrapper
    write_camcasp_runtime_attestation
    export_reference_environment
    write_inputs
    verify_camcasp_runtime_attestation
    run_camcasp
    attest_generated_protocol
    run_localize
    require_reviewed_pdef
    write_manifest
    build_reference_json
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
