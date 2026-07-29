# CamCASP Reproducible Reference and Provenance Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an auditable, fail-closed workflow that provisions the approved external tools, runs the canonical CamCASP Figure 1 reference calculation, validates the complete L3 model and C6–C12 outputs, and writes untracked provenance JSON plus copy-ready literals.

**Architecture:** A tracked Bash orchestrator owns safe provisioning, isolated directories, stage execution, logs, and the canonical external workflow. A focused Python module owns deterministic parsing, frame/tensor conversion, validation, JSON serialization, and literal rendering; pure pytest tests exercise it using synthetic CamCASP-format fixtures without invoking external software.

**Tech Stack:** Bash, Python 3 standard library, pytest, Git, CamCASP 7.2.2 patch 003, Orient 5.0.11-ng, staged Psi4.

## Global Constraints

Production pytest must not invoke CamCASP, ORIENT, PFIT, or CASIMIR. It compares Psi4 results against hard-coded values produced by a tracked local regeneration script.

- Production Psi4 must not call external CamCASP, ORIENT, PFIT, or CASIMIR executables.
- Production pytest must not clone software, access the network, or read locally generated JSON.
- Broad plausibility ranges are not substitutes for reference comparisons.
- The historical L2 values near 6.51 a.u. for O and 1.38 a.u. for H are not normative after adopting the complete L3 model.
- This work does not add anisotropic spherical Cn components to the public Psi4 API. The initial Cn API contains the isotropic `00 00 0` atom-pair coefficients.

Use the Psi4-backed CamCASP H2O properties example as the canonical system.

- Geometry, in Bohr:
  - O: `(0.0000000000, 0.0000000000, 0.0000000000)`
  - H1: `(-1.4536519600, 0.0000000000, -1.1216873200)`
  - H2: `( 1.4536519600, 0.0000000000, -1.1216873200)`
- Charge: 0
- Multiplicity: 1
- Psi4 orientation: `symmetry c1`, `no_com`, and `no_reorient`
- Orbital method: PBE0
- Orbital basis: aug-cc-pVTZ (`aVTZ` in the CamCASP input)
- Asymptotic correction: Psi4 GRAC protocol used by CamCASP
- Experimental ionization potential: `12.62063 eV`
- CamCASP H2O input HOMO value: `-0.3989 hartree`
- CamCASP response kernel: ALDA+CHF
- Point grid: CamCASP `Options Tests` grid for a deterministic, tractable regression calculation

The regeneration script must make these choices explicit rather than relying silently on CamCASP defaults.

Use the CamCASP/CASIMIR default quadrature:

- one static point at zero frequency
- ten nonzero Gauss-Legendre imaginary-frequency points
- base frequency parameter `0.5 a.u.`

The exact generated frequencies are stored in local JSON and copied as hard-coded pytest literals.

Use one complete model for all accepted properties:

- nonlocal CamCASP polarizability rank: L4 (`NL4` output), matching the standard properties workflow
- ORIENT localization method: Lillestolen-Wheatley (`LW`)
- ORIENT localization limit: L3
- PFIT WSM limit: L3
- PFIT hydrogen limit: L3
- PFIT penalty weight: 4
- PFIT cutoff: `0.0001`
- local axes: the symmetry-related definitions from `camcasp-bin/tests/H2O_props/psi4/H2O.axes`:

```text
Axes
  H1  z global Z x from H2 to H1
  H2  z global Z x from H1 to H2
End
```

L3 is required on every atom because CamCASP maps L1 to coefficients through C6, L2 through C10, and L3 through C12. Lowering only the hydrogen limit would deliberately truncate higher-order coefficients involving H.

Changing localization rank, WSM rank, hydrogen rank, weighting, cutoff, axes, grid, basis, functional, kernel, or asymptotic correction defines a different reference model and requires intentional regeneration and review of every hard-coded value.

For atom A, convert the spherical dipole-dipole block to a local Cartesian 3x3 matrix using CamCASP's documented real-spherical ordering, then rotate it using the orthonormal local-to-global matrix `R_A`:

```text
alpha_A_global = R_A @ alpha_A_local @ R_A.T
```

The regeneration script must validate `R_A @ R_A.T = I`, preserve right-handed frames, and store the spherical block, local Cartesian block, rotation matrix, and global Cartesian block in the local JSON.

The JSON exists only for local provenance and diagnostics. Add `.camcasp-reference/` and `orient/` to `.gitignore`. Do not commit the JSON.

The script exits nonzero at the first failed stage and retains logs. Error messages identify the failed stage and relevant output path.

It rejects:

- missing or non-executable Psi4
- failed Orient checkout, binary validation, or smoke test
- missing or failed CamCASP executables
- nonzero CamCASP, ORIENT, PFIT, or CASIMIR exits
- absent static or dynamic frequency blocks
- a frequency count other than eleven under the accepted configuration
- unexpected atom labels or order
- incomplete L3 polarizability output
- incomplete C12 output
- non-finite values
- non-orthonormal or left-handed local frames
- nonsymmetric Cartesian dipole-dipole tensors beyond parsing tolerance
- nonsymmetric isotropic atom-pair Cn matrices beyond parsing tolerance

---

## File Map and Interfaces

- Modify: `.gitignore`
  - Preserve the existing broad `*.sh` rule while explicitly unignoring `/devtools/regenerate-camcasp.sh`.
  - Explicitly ignore `/.camcasp-reference/`, `/orient/`, and `/camcasp-bin/`.
- Create: `devtools/regenerate-camcasp.sh`
  - Public interface:
    - `bash devtools/regenerate-camcasp.sh`
    - `bash devtools/regenerate-camcasp.sh --preflight-only`
  - Environment overrides:
    - `PSI4_EXE`
    - `CAMCASP`
    - `ORIENT_EXE`
    - `ORIENT_REF`
    - `CAMCASP_PDEF_SHA256`
    - `CAMCASP_REFERENCE_CORES`
  - Owns provisioning, canonical inputs, stage logs, artifact checks, scientific review gate, and invocation of the Python builder.
- Create: `devtools/camcasp_reference.py`
  - Public Python interfaces:
    - `parse_frequencies(path: Path) -> list[FrequencyPoint]`
    - `parse_refined_polarizabilities(path: Path, atom_labels: Sequence[str], limit: int) -> list[FrequencyBlock]`
    - `parse_axes(text: str, geometry: Mapping[str, Vector3]) -> dict[str, Matrix3]`
    - `dipole_local_cartesian(model: SphericalModel) -> Matrix3`
    - `rotate_tensor(local: Matrix3, rotation: Matrix3) -> Matrix3`
    - `parse_isotropic_cn(path: Path, atom_labels: Sequence[str], atom_types: Mapping[str, str]) -> dict[str, Matrix]`
    - `validate_stage_artifacts(work_dir: Path, job: str) -> dict[str, Path]`
    - `build_reference_document(inputs: BuildInputs) -> dict[str, object]`
    - `validate_reference_document(document: Mapping[str, object]) -> None`
    - `write_atomic_json(path: Path, document: Mapping[str, object]) -> None`
    - `render_python_literals(document: Mapping[str, object]) -> str`
  - CLI:
    - `python devtools/camcasp_reference.py build --manifest ... --output ...`
    - `python devtools/camcasp_reference.py validate FILE`
- Create: `tests/pytests/test_camcasp_reference.py`
  - Pure unit and integration tests for the tracked workflow.
  - Uses only temporary files and synthetic CamCASP-format text.
  - Does not invoke CamCASP, Orient, PFIT, CASIMIR, network access, or local provenance JSON.

The following remain generated and ignored:

- `.camcasp-reference/inputs/`
- `.camcasp-reference/work/`
- `.camcasp-reference/scratch/`
- `.camcasp-reference/logs/`
- `.camcasp-reference/tools/`
- `.camcasp-reference/atomic-polarizabilities.json`
- `orient/`
- `camcasp-bin/`

### Task 1: Make the tracked orchestrator visible and establish fail-closed stage execution

**Files:**
- Modify: `.gitignore:1-110`
- Create: `devtools/regenerate-camcasp.sh`
- Create: `tests/pytests/test_camcasp_reference.py`

**Interfaces:**
- Consumes: repository root and Git ignore rules.
- Produces: sourceable shell function `run_logged STAGE LOG COMMAND...`; tracked executable path `devtools/regenerate-camcasp.sh`.

- [ ] **Step 1: Write the failing ignore and stage-runner tests**

Create `tests/pytests/test_camcasp_reference.py` with:

```python
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "devtools" / "regenerate-camcasp.sh"


def test_regeneration_script_is_trackable():
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", "devtools/regenerate-camcasp.sh"],
        cwd=ROOT,
        check=False,
    )
    assert result.returncode == 1, "devtools/regenerate-camcasp.sh is still ignored"


def test_run_logged_reports_stage_and_retains_log(tmp_path):
    log = tmp_path / "orient.log"
    command = (
        f'source "{SCRIPT}"; '
        'run_logged orient "$1" bash -c \'echo orient-sentinel; exit 23\''
    )
    result = subprocess.run(
        ["bash", "-c", command, "stage-test", str(log)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 23
    assert "[orient] failed with exit status 23" in result.stderr
    assert "orient-sentinel" in log.read_text()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -P -m pytest -vv -x \
  tests/pytests/test_camcasp_reference.py::test_regeneration_script_is_trackable \
  tests/pytests/test_camcasp_reference.py::test_run_logged_reports_stage_and_retains_log
```

Expected: the first test reports that `devtools/regenerate-camcasp.sh` is ignored by `*.sh`, or the second reports that the script/function does not exist.

- [ ] **Step 3: Fix the ignore intent**

Remove the existing trailing bare `camcasp-bin` entry, then add these canonical root-anchored rules immediately after the existing `*.sh` rule in `.gitignore`:

```gitignore
# Tracked CamCASP reference generator; external dependencies and outputs stay local.
!/devtools/regenerate-camcasp.sh
/.camcasp-reference/
/orient/
/camcasp-bin/
```

Verify the negation is after `*.sh`; placing it before the broad rule will not work. Keep exactly one CamCASP ignore rule.

- [ ] **Step 4: Add the shell scaffold and stage runner**

Create `devtools/regenerate-camcasp.sh` with this opening:

```bash
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
```

Do not install the `EXIT` trap while the file is merely sourced by tests. Install it inside `main`:

```bash
main() {
    trap on_exit EXIT
    printf 'CamCASP reference root: %s\n' "$REFERENCE_ROOT"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
```

- [ ] **Step 5: Run syntax and focused tests**

Run:

```bash
bash -n devtools/regenerate-camcasp.sh
python -P -m pytest -vv \
  tests/pytests/test_camcasp_reference.py::test_regeneration_script_is_trackable \
  tests/pytests/test_camcasp_reference.py::test_run_logged_reports_stage_and_retains_log
```

Expected: Bash syntax succeeds and both tests pass. The failure-path test must return 23 and retain its log.

- [ ] **Step 6: Commit**

```bash
git add .gitignore devtools/regenerate-camcasp.sh tests/pytests/test_camcasp_reference.py
git commit -m "devtools: track fail-closed CamCASP generator"
```

### Task 2: Add safe preflight and idempotent external-tool provisioning

**Files:**
- Modify: `devtools/regenerate-camcasp.sh`
- Modify: `tests/pytests/test_camcasp_reference.py`

**Interfaces:**
- Consumes: `PSI4_EXE`, `CAMCASP`, `ORIENT_EXE`, `ORIENT_REF`, compressed CamCASP program archives.
- Produces: verified absolute tool paths, five executable CamCASP links, a tested Psi4 wrapper, an Orient executable, and checksummed provisioning logs.

- [ ] **Step 1: Add failing preflight and non-destructive archive tests**

Append:

```python
import gzip


def test_preflight_rejects_missing_psi4(tmp_path):
    result = subprocess.run(
        ["bash", str(SCRIPT), "--preflight-only"],
        cwd=ROOT,
        env={
            **os.environ,
            "PSI4_EXE": str(tmp_path / "missing-psi4"),
            "CAMCASP": str(tmp_path / "camcasp-bin"),
            "ORIENT_EXE": str(tmp_path / "orient"),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "[preflight] PSI4_EXE is not executable" in result.stderr


def test_install_camcasp_program_preserves_archive(tmp_path):
    camcasp = tmp_path / "camcasp-bin"
    archive_dir = camcasp / "x86-64" / "gfortran"
    archive_dir.mkdir(parents=True)
    (camcasp / "bin").mkdir()
    archive = archive_dir / "camcasp.gz"
    payload = b"#!/usr/bin/env bash\ncat >/dev/null\nexit 0\n"
    with gzip.open(archive, "wb") as handle:
        handle.write(payload)

    command = (
        f'source "{SCRIPT}"; '
        'CAMCASP="$1"; install_camcasp_program camcasp'
    )
    result = subprocess.run(
        ["bash", "-c", command, "archive-test", str(camcasp)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert archive.read_bytes().startswith(b"\x1f\x8b")
    installed = camcasp / "x86-64" / "gfortran" / "exe" / "camcasp"
    assert installed.read_bytes() == payload
    assert os.access(installed, os.X_OK)
    assert (camcasp / "bin" / "camcasp").resolve() == installed.resolve()


def test_safe_path_rejects_repository_root():
    command = (
        f'source "{SCRIPT}"; '
        'require_safe_generated_path "$REPO_ROOT"'
    )
    result = subprocess.run(
        ["bash", "-c", command],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "refusing unsafe generated path" in result.stderr
```

- [ ] **Step 2: Run tests to verify red**

Run:

```bash
python -P -m pytest -vv -x \
  tests/pytests/test_camcasp_reference.py::test_preflight_rejects_missing_psi4 \
  tests/pytests/test_camcasp_reference.py::test_install_camcasp_program_preserves_archive \
  tests/pytests/test_camcasp_reference.py::test_safe_path_rejects_repository_root
```

Expected: failures identify missing `--preflight-only`, `install_camcasp_program`, or `require_safe_generated_path`.

- [ ] **Step 3: Implement safe path and executable checks**

Add:

```bash
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
```

Parse the single supported flag before provisioning:

```bash
MODE="full"
case "${1:-}" in
    "") ;;
    --preflight-only) MODE="preflight" ;;
    *) CURRENT_STAGE="arguments"; fail "unknown argument: $1" ;;
esac
```

- [ ] **Step 4: Implement non-destructive CamCASP provisioning**

Add:

```bash
CAMCASP_URL="https://github.com/ajmisquitta/camcasp-bin.git"
CAMCASP_COMMIT="b4744425233a61786052832e1db4f109959c1ce9"
CAMCASP_VERSION_PATTERN="VERSION 7.2.2|7.2.2"
CAMCASP_PROGRAMS=(camcasp cluster process pfit casimir)

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
```

Do not use `gunzip` in place: the `.gz` provenance originals must remain intact.

- [ ] **Step 5: Implement Orient provisioning and smoke validation**

Add:

```bash
ORIENT_URL="https://gitlab.com/anthonyjs/orient.git"

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
```

If the supplied non-graphical binary is incompatible, stop with the source-build guidance; do not skip localization.

- [ ] **Step 6: Generate and smoke-test the Psi4 wrapper**

Add:

```bash
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
```

Export the isolated environment only after paths have been validated:

```bash
export CAMCASP
export ARCH=x86-64
export PATH="$ORIENT_BIN_DIR:$CAMCASP/bin:$PATH"
export PSIPATH="$CAMCASP/basis/psi4:$CAMCASP/basis/psi4/for-psi4-lib"
export SCRATCH="$REFERENCE_ROOT/scratch/camcasp"
export PSI_SCRATCH="$REFERENCE_ROOT/scratch/psi4"
mkdir -p "$SCRATCH" "$PSI_SCRATCH"
```

- [ ] **Step 7: Run focused tests and preflight**

Run:

```bash
bash -n devtools/regenerate-camcasp.sh
python -P -m pytest -vv \
  tests/pytests/test_camcasp_reference.py::test_preflight_rejects_missing_psi4 \
  tests/pytests/test_camcasp_reference.py::test_install_camcasp_program_preserves_archive \
  tests/pytests/test_camcasp_reference.py::test_safe_path_rejects_repository_root
PSI4_EXE="$PWD/build_camcasp/stage/bin/psi4" \
  bash devtools/regenerate-camcasp.sh --preflight-only
```

Expected: all unit tests pass. Preflight passes only if the staged Psi4 executable exists; it must not clone or unpack anything in preflight-only mode.

- [ ] **Step 8: Commit**

```bash
git add devtools/regenerate-camcasp.sh tests/pytests/test_camcasp_reference.py
git commit -m "devtools: safely provision CamCASP reference tools"
```

### Task 3: Parse exact frequency blocks and the complete refined L3 model

**Files:**
- Create: `devtools/camcasp_reference.py`
- Modify: `tests/pytests/test_camcasp_reference.py`

**Interfaces:**
- Consumes: NL4 format-B `FREQ2` headers and refined `_ref_wt4_L3_0f10.pol`.
- Produces: `FrequencyPoint`, `SphericalModel`, and `FrequencyBlock` records with exact atom/frequency ordering and 16×16 L3 matrices.
- Format evidence: the bundled `camcasp-bin/examples/properties/H2O/output_2/H2O_aTZ_ref_wt4_L2_0f10.pol` contains 9×9 matrices because `(L+1)^2 = 9` for L2; the accepted L3 parser therefore requires `(3+1)^2 = 16` rows and columns.

- [ ] **Step 1: Write failing parser tests**

Append:

```python
import math
import sys

sys.path.insert(0, str(ROOT))
from devtools.camcasp_reference import (  # noqa: E402
    COMPONENTS_L3,
    ReferenceFormatError,
    parse_frequencies,
    parse_refined_polarizabilities,
)


def make_nl4_frequency_text():
    squared = [0.0] + [-(0.01 * index) ** 2 for index in range(1, 11)]
    lines = []
    for value in squared:
        for left, right in (("O", "O"), ("O", "H1"), ("H1", "O")):
            lines.append(
                "POL  SITE-LABELS  "
                f"{left}  {right}  SITE-INDICES  1  1  "
                f"RANK  0 : 4  BY  0 : 4  FREQ2 {value:.16E} CARTSPHER S"
            )
    return "\n".join(lines) + "\n"


def make_l3_refined_text():
    blocks = []
    for frequency_index in range(11):
        blocks.append(f"# INDEX {frequency_index:03d}")
        for atom_index, label in enumerate(("O", "H1", "H2")):
            blocks.append(f"{label} {label}")
            for row in range(16):
                values = [
                    frequency_index + atom_index + row / 100.0 + column / 10000.0
                    for column in range(16)
                ]
                blocks.append(" ".join(f"{value:.8f}" for value in values))
    return "\n".join(blocks) + "\n"


def test_parse_static_plus_ten_frequencies(tmp_path):
    source = tmp_path / "H2O_NL4_fmtB.pol"
    source.write_text(make_nl4_frequency_text())
    points = parse_frequencies(source)
    assert [point.index for point in points] == list(range(11))
    assert points[0].omega == 0.0
    assert [point.omega for point in points[1:]] == [
        index / 100.0 for index in range(1, 11)
    ]
    assert all(points[index].omega < points[index + 1].omega for index in range(10))


def test_parse_complete_l3_model(tmp_path):
    source = tmp_path / "H2O_ref_wt4_L3_0f10.pol"
    source.write_text(make_l3_refined_text())
    blocks = parse_refined_polarizabilities(source, ("O", "H1", "H2"), limit=3)
    assert len(blocks) == 11
    assert COMPONENTS_L3 == (
        "00", "10", "11c", "11s", "20", "21c", "21s", "22c", "22s",
        "30", "31c", "31s", "32c", "32s", "33c", "33s",
    )
    assert tuple(blocks[0].atoms) == ("O", "H1", "H2")
    assert len(blocks[10].atoms["H2"].matrix) == 16
    assert all(len(row) == 16 for row in blocks[10].atoms["H2"].matrix)


def test_rejects_incomplete_l3_model(tmp_path):
    source = tmp_path / "truncated.pol"
    source.write_text(make_l3_refined_text().rsplit("\n", 8)[0] + "\n")
    try:
        parse_refined_polarizabilities(source, ("O", "H1", "H2"), limit=3)
    except ReferenceFormatError as exc:
        assert "frequency 010 atom H2 requires 16 rows" in str(exc)
    else:
        raise AssertionError("truncated L3 model was accepted")


def test_rejects_nonfinite_l3_value(tmp_path):
    source = tmp_path / "nonfinite.pol"
    source.write_text(make_l3_refined_text().replace("0.00000000", "nan", 1))
    try:
        parse_refined_polarizabilities(source, ("O", "H1", "H2"), limit=3)
    except ReferenceFormatError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("non-finite L3 value was accepted")
```

- [ ] **Step 2: Run tests to verify red**

Run:

```bash
python -P -m pytest -vv -x \
  tests/pytests/test_camcasp_reference.py::test_parse_static_plus_ten_frequencies \
  tests/pytests/test_camcasp_reference.py::test_parse_complete_l3_model \
  tests/pytests/test_camcasp_reference.py::test_rejects_incomplete_l3_model \
  tests/pytests/test_camcasp_reference.py::test_rejects_nonfinite_l3_value
```

Expected: collection fails because `devtools.camcasp_reference` or its interfaces do not exist.

- [ ] **Step 3: Add parser data types and component order**

Create `devtools/camcasp_reference.py` with:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

COMPONENTS_BY_RANK = {
    0: ("00",),
    1: ("10", "11c", "11s"),
    2: ("20", "21c", "21s", "22c", "22s"),
    3: ("30", "31c", "31s", "32c", "32s", "33c", "33s"),
}
COMPONENTS_L3 = tuple(
    component
    for rank in range(4)
    for component in COMPONENTS_BY_RANK[rank]
)


class ReferenceFormatError(ValueError):
    pass


@dataclass(frozen=True)
class FrequencyPoint:
    index: int
    squared_source_text: str
    squared_frequency: float
    omega: float


@dataclass(frozen=True)
class SphericalModel:
    components: tuple[str, ...]
    matrix: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class FrequencyBlock:
    index: int
    atoms: dict[str, SphericalModel]
```

- [ ] **Step 4: Implement frequency parsing**

Use source decimal text as provenance and derive `omega` only from the emitted squared frequency:

```python
FREQ2_RE = re.compile(
    r"\bFREQ2\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EDed][-+]?\d+)?)"
)


def _float(text: str, context: str) -> float:
    try:
        value = float(text.replace("D", "E").replace("d", "e"))
    except ValueError as exc:
        raise ReferenceFormatError(f"{context}: invalid float {text!r}") from exc
    if not math.isfinite(value):
        raise ReferenceFormatError(f"{context}: non-finite value {text!r}")
    return value


def parse_frequencies(path: Path) -> list[FrequencyPoint]:
    unique: list[tuple[str, float]] = []
    for match in FREQ2_RE.finditer(path.read_text()):
        raw = match.group(1)
        value = _float(raw, f"{path}: FREQ2")
        if not unique or value != unique[-1][1]:
            unique.append((raw, value))

    if len(unique) != 11:
        raise ReferenceFormatError(
            f"{path}: expected 11 unique frequency blocks, found {len(unique)}"
        )
    if unique[0][1] != 0.0:
        raise ReferenceFormatError(f"{path}: first frequency is not static zero")

    points = []
    for index, (raw, squared) in enumerate(unique):
        if index and squared >= 0.0:
            raise ReferenceFormatError(
                f"{path}: dynamic FREQ2 at index {index} is not negative"
            )
        omega = 0.0 if index == 0 else math.sqrt(-squared)
        points.append(FrequencyPoint(index, raw, squared, omega))

    if any(points[index].omega >= points[index + 1].omega for index in range(10)):
        raise ReferenceFormatError(f"{path}: imaginary frequencies are not increasing")
    return points
```

- [ ] **Step 5: Implement strict refined-model parsing**

Implement an index/header state machine:

```python
INDEX_RE = re.compile(r"^\s*#\s*INDEX\s+(\d{3})\s*$")
ATOM_RE = re.compile(r"^\s*(\S+)\s+\1\s*$")


def parse_refined_polarizabilities(
    path: Path,
    atom_labels: Sequence[str],
    limit: int,
) -> list[FrequencyBlock]:
    if limit != 3:
        raise ReferenceFormatError(f"accepted model requires limit 3, got {limit}")
    lines = path.read_text().splitlines()
    blocks: list[FrequencyBlock] = []
    position = 0

    while position < len(lines):
        match = INDEX_RE.match(lines[position])
        if not match:
            position += 1
            continue
        index = int(match.group(1))
        if index != len(blocks):
            raise ReferenceFormatError(
                f"{path}: expected frequency index {len(blocks):03d}, found {index:03d}"
            )
        position += 1
        atoms: dict[str, SphericalModel] = {}

        for expected_atom in atom_labels:
            while position < len(lines) and not lines[position].strip():
                position += 1
            if position >= len(lines) or lines[position].split() != [
                expected_atom,
                expected_atom,
            ]:
                found = "<end>" if position >= len(lines) else lines[position].strip()
                raise ReferenceFormatError(
                    f"{path}: frequency {index:03d} expected atom "
                    f"{expected_atom}, found {found!r}"
                )
            position += 1
            matrix = []
            for row_index in range(16):
                if position >= len(lines):
                    raise ReferenceFormatError(
                        f"{path}: frequency {index:03d} atom {expected_atom} "
                        "requires 16 rows"
                    )
                fields = lines[position].split()
                if len(fields) != 16:
                    raise ReferenceFormatError(
                        f"{path}: frequency {index:03d} atom {expected_atom} "
                        f"row {row_index} requires 16 values, found {len(fields)}"
                    )
                matrix.append(
                    tuple(
                        _float(
                            field,
                            f"{path}: frequency {index:03d} atom "
                            f"{expected_atom} row {row_index}",
                        )
                        for field in fields
                    )
                )
                position += 1
            atoms[expected_atom] = SphericalModel(COMPONENTS_L3, tuple(matrix))
        blocks.append(FrequencyBlock(index, atoms))

    if len(blocks) != 11:
        raise ReferenceFormatError(
            f"{path}: expected 11 refined blocks, found {len(blocks)}"
        )
    return blocks
```

- [ ] **Step 6: Run parser tests**

Run:

```bash
python -P -m pytest -vv \
  tests/pytests/test_camcasp_reference.py::test_parse_static_plus_ten_frequencies \
  tests/pytests/test_camcasp_reference.py::test_parse_complete_l3_model \
  tests/pytests/test_camcasp_reference.py::test_rejects_incomplete_l3_model \
  tests/pytests/test_camcasp_reference.py::test_rejects_nonfinite_l3_value
```

Expected: all four tests pass.

- [ ] **Step 7: Commit**

```bash
git add devtools/camcasp_reference.py tests/pytests/test_camcasp_reference.py
git commit -m "devtools: parse complete CamCASP L3 references"
```

### Task 4: Build and validate local frames and Cartesian dipole tensors

**Files:**
- Modify: `devtools/camcasp_reference.py`
- Modify: `tests/pytests/test_camcasp_reference.py`

**Interfaces:**
- Consumes: canonical geometry, axes text, and each atom’s complete spherical matrix.
- Produces: right-handed local-to-global matrices, symmetric local Cartesian tensors, and symmetric global Cartesian tensors.

- [ ] **Step 1: Write failing frame and tensor tests**

Append:

```python
from devtools.camcasp_reference import (  # noqa: E402
    build_local_frames,
    dipole_local_cartesian,
    rotate_tensor,
    validate_rotation_matrix,
)


CANONICAL_GEOMETRY = {
    "O": (0.0, 0.0, 0.0),
    "H1": (-1.4536519600, 0.0, -1.1216873200),
    "H2": (1.4536519600, 0.0, -1.1216873200),
}
CANONICAL_AXES = """\
Axes
  H1  z global Z x from H2 to H1
  H2  z global Z x from H1 to H2
End
"""


def test_canonical_frames_are_right_handed():
    frames = build_local_frames(CANONICAL_GEOMETRY, CANONICAL_AXES)
    assert frames["O"] == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    assert frames["H1"] == ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0))
    assert frames["H2"] == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    for frame in frames.values():
        validate_rotation_matrix(frame)


def test_rejects_left_handed_frame():
    try:
        validate_rotation_matrix(
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0))
        )
    except ReferenceFormatError as exc:
        assert "left-handed" in str(exc)
    else:
        raise AssertionError("left-handed frame was accepted")


def test_dipole_mapping_and_hydrogen_c2_signs():
    matrix = [[0.0] * 16 for _ in range(16)]
    indices = {label: COMPONENTS_L3.index(label) for label in ("10", "11c", "11s")}
    matrix[indices["10"]][indices["10"]] = 1.6
    matrix[indices["11c"]][indices["11c"]] = 1.3
    matrix[indices["11s"]][indices["11s"]] = 1.2
    matrix[indices["10"]][indices["11c"]] = -0.25
    matrix[indices["11c"]][indices["10"]] = -0.25
    model = type("Model", (), {
        "components": COMPONENTS_L3,
        "matrix": tuple(tuple(row) for row in matrix),
    })()

    local = dipole_local_cartesian(model)
    assert local == ((1.3, 0.0, -0.25), (0.0, 1.2, 0.0), (-0.25, 0.0, 1.6))

    frames = build_local_frames(CANONICAL_GEOMETRY, CANONICAL_AXES)
    h1 = rotate_tensor(local, frames["H1"])
    h2 = rotate_tensor(local, frames["H2"])
    assert h1[0][2] == 0.25
    assert h2[0][2] == -0.25
    assert h1[0][0] == h2[0][0]
    assert h1[1][1] == h2[1][1]
    assert h1[2][2] == h2[2][2]
```

- [ ] **Step 2: Run tests to verify red**

Run:

```bash
python -P -m pytest -vv -x \
  tests/pytests/test_camcasp_reference.py::test_canonical_frames_are_right_handed \
  tests/pytests/test_camcasp_reference.py::test_rejects_left_handed_frame \
  tests/pytests/test_camcasp_reference.py::test_dipole_mapping_and_hydrogen_c2_signs
```

Expected: imports fail because the frame/tensor functions do not exist.

- [ ] **Step 3: Implement vector and matrix primitives**

Add dependency-free helpers:

```python
Vector3 = tuple[float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]


def _dot(left: Vector3, right: Vector3) -> float:
    return sum(a * b for a, b in zip(left, right))


def _norm(vector: Vector3) -> float:
    return math.sqrt(_dot(vector, vector))


def _normalize(vector: Vector3) -> Vector3:
    length = _norm(vector)
    if length <= 1.0e-14:
        raise ReferenceFormatError("axis direction has zero length")
    return tuple(value / length for value in vector)  # type: ignore[return-value]


def _cross(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _transpose(matrix: Matrix3) -> Matrix3:
    return tuple(zip(*matrix))  # type: ignore[return-value]


def _matmul(left: Matrix3, right: Matrix3) -> Matrix3:
    right_t = _transpose(right)
    return tuple(
        tuple(_dot(row, column) for column in right_t)
        for row in left
    )  # type: ignore[return-value]


def _determinant(matrix: Matrix3) -> float:
    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )
```

- [ ] **Step 4: Implement the canonical axes parser and validation**

The matrix convention is local-to-global with columns `(local x, local y, local z)`:

```python
def validate_rotation_matrix(rotation: Matrix3, tolerance: float = 1.0e-10) -> None:
    product = _matmul(rotation, _transpose(rotation))
    for row in range(3):
        for column in range(3):
            expected = 1.0 if row == column else 0.0
            if abs(product[row][column] - expected) > tolerance:
                raise ReferenceFormatError("local frame is not orthonormal")
    determinant = _determinant(rotation)
    if determinant < 0.0:
        raise ReferenceFormatError("local frame is left-handed")
    if abs(determinant - 1.0) > tolerance:
        raise ReferenceFormatError(
            f"local frame determinant is {determinant}, expected +1"
        )


def build_local_frames(
    geometry: Mapping[str, Vector3],
    axes_text: str,
) -> dict[str, Matrix3]:
    frames: dict[str, Matrix3] = {
        label: ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        for label in geometry
    }
    rule = re.compile(
        r"^\s*(\S+)\s+z\s+global\s+Z\s+x\s+from\s+(\S+)\s+to\s+(\S+)\s*$",
        re.IGNORECASE,
    )
    for line in axes_text.splitlines():
        match = rule.match(line)
        if not match:
            continue
        site, origin, target = match.groups()
        if site not in geometry or origin not in geometry or target not in geometry:
            raise ReferenceFormatError(f"unknown site in axis rule: {line.strip()}")
        local_z = (0.0, 0.0, 1.0)
        direction = tuple(
            geometry[target][index] - geometry[origin][index]
            for index in range(3)
        )
        projected = tuple(
            direction[index] - _dot(direction, local_z) * local_z[index]
            for index in range(3)
        )
        local_x = _normalize(projected)  # type: ignore[arg-type]
        local_y = _normalize(_cross(local_z, local_x))
        rotation = tuple(zip(local_x, local_y, local_z))  # type: ignore[assignment]
        validate_rotation_matrix(rotation)
        frames[site] = rotation

    if set(frames) != set(geometry):
        raise ReferenceFormatError("frame labels do not match geometry labels")
    return frames
```

- [ ] **Step 5: Implement spherical-to-Cartesian extraction and rotation**

Document the convention in code:

```python
# CamCASP real spherical dipoles: 10 -> z, 11c -> x, 11s -> y.
CARTESIAN_TO_SPHERICAL_DIPOLE = ("11c", "11s", "10")


def _validate_symmetric(matrix: Matrix3, context: str, tolerance: float = 1.0e-8) -> None:
    for row in range(3):
        for column in range(3):
            if abs(matrix[row][column] - matrix[column][row]) > tolerance:
                raise ReferenceFormatError(f"{context} is not symmetric")


def dipole_local_cartesian(model: SphericalModel) -> Matrix3:
    index = {label: model.components.index(label) for label in CARTESIAN_TO_SPHERICAL_DIPOLE}
    result = tuple(
        tuple(model.matrix[index[left]][index[right]] for right in CARTESIAN_TO_SPHERICAL_DIPOLE)
        for left in CARTESIAN_TO_SPHERICAL_DIPOLE
    )
    _validate_symmetric(result, "local Cartesian dipole tensor")
    return result  # type: ignore[return-value]


def rotate_tensor(local: Matrix3, rotation: Matrix3) -> Matrix3:
    validate_rotation_matrix(rotation)
    global_tensor = _matmul(_matmul(rotation, local), _transpose(rotation))
    _validate_symmetric(global_tensor, "global Cartesian dipole tensor")
    for row in global_tensor:
        for value in row:
            if not math.isfinite(value):
                raise ReferenceFormatError("global Cartesian tensor contains non-finite value")
    return global_tensor
```

- [ ] **Step 6: Run frame and tensor tests**

Run:

```bash
python -P -m pytest -vv \
  tests/pytests/test_camcasp_reference.py::test_canonical_frames_are_right_handed \
  tests/pytests/test_camcasp_reference.py::test_rejects_left_handed_frame \
  tests/pytests/test_camcasp_reference.py::test_dipole_mapping_and_hydrogen_c2_signs
```

Expected: all pass, including opposite H1/H2 global `xz` signs.

- [ ] **Step 7: Commit**

```bash
git add devtools/camcasp_reference.py tests/pytests/test_camcasp_reference.py
git commit -m "devtools: validate CamCASP frames and tensors"
```

### Task 5: Parse and validate isotropic atom-pair C6, C8, C10, and C12

**Files:**
- Modify: `devtools/camcasp_reference.py`
- Modify: `tests/pytests/test_camcasp_reference.py`

**Interfaces:**
- Consumes: CASIMIR Orient-format `_C12.pot`.
- Produces: four finite symmetric matrices expanded from site types to atom order `O, H1, H2`.

- [ ] **Step 1: Write failing CASIMIR parser tests**

Append:

```python
from devtools.camcasp_reference import parse_isotropic_cn  # noqa: E402


CASIMIR_C12 = """\
  O  O      C6 C7 C8 C9 C10 C11 C12
    00 00 0 20.0 0.0 200.0 0.0 2000.0 0.0 20000.0
  End
  H  O      C6 C7 C8 C9 C10 C11 C12
    00 00 0 4.0 0.0 40.0 0.0 400.0 0.0 4000.0
  End
  H  H      C6 C7 C8 C9 C10 C11 C12
    00 00 0 1.0 0.0 10.0 0.0 100.0 0.0 1000.0
  End
"""


def test_parse_all_isotropic_cn_matrices(tmp_path):
    source = tmp_path / "H2O_ref_wt4_L3_C12.pot"
    source.write_text(CASIMIR_C12)
    matrices = parse_isotropic_cn(
        source,
        ("O", "H1", "H2"),
        {"O": "O", "H1": "H", "H2": "H"},
    )
    assert tuple(matrices) == ("C6", "C8", "C10", "C12")
    assert matrices["C6"] == (
        (20.0, 4.0, 4.0),
        (4.0, 1.0, 1.0),
        (4.0, 1.0, 1.0),
    )
    assert matrices["C12"][0][0] == 20000.0
    assert matrices["C12"][1][0] == 4000.0
    assert matrices["C12"][2][2] == 1000.0


def test_rejects_casimir_output_without_c12(tmp_path):
    source = tmp_path / "C10-only.pot"
    source.write_text(CASIMIR_C12.replace(" C11 C12", "").replace(" 0.0 20000.0", "").replace(" 0.0 4000.0", "").replace(" 0.0 1000.0", ""))
    try:
        parse_isotropic_cn(
            source,
            ("O", "H1", "H2"),
            {"O": "O", "H1": "H", "H2": "H"},
        )
    except ReferenceFormatError as exc:
        assert "missing required C12 column" in str(exc)
    else:
        raise AssertionError("C10-only output was accepted")
```

- [ ] **Step 2: Run tests to verify red**

Run:

```bash
python -P -m pytest -vv -x \
  tests/pytests/test_camcasp_reference.py::test_parse_all_isotropic_cn_matrices \
  tests/pytests/test_camcasp_reference.py::test_rejects_casimir_output_without_c12
```

Expected: import fails because `parse_isotropic_cn` is absent.

- [ ] **Step 3: Implement section and isotropic-row parsing**

Add:

```python
CN_ORDERS = ("C6", "C8", "C10", "C12")
PAIR_HEADER_RE = re.compile(r"^\s*(\S+)\s+(\S+)\s+(C6\b.*)$", re.IGNORECASE)


def parse_isotropic_cn(
    path: Path,
    atom_labels: Sequence[str],
    atom_types: Mapping[str, str],
) -> dict[str, tuple[tuple[float, ...], ...]]:
    lines = path.read_text().splitlines()
    by_types: dict[frozenset[str] | tuple[str, str], dict[str, float]] = {}
    index = 0

    while index < len(lines):
        header = PAIR_HEADER_RE.match(lines[index])
        if not header:
            index += 1
            continue
        left_type, right_type, columns_text = header.groups()
        columns = columns_text.upper().split()
        if "C12" not in columns:
            raise ReferenceFormatError(f"{path}: missing required C12 column")
        column_index = {name: columns.index(name) for name in CN_ORDERS}
        pair_key = tuple(sorted((left_type, right_type)))
        if pair_key in by_types:
            raise ReferenceFormatError(f"{path}: duplicate pair block {pair_key}")
        index += 1
        isotropic = None

        while index < len(lines) and lines[index].strip().lower() != "end":
            fields = lines[index].split()
            if len(fields) >= 3 and fields[:3] == ["00", "00", "0"]:
                if isotropic is not None:
                    raise ReferenceFormatError(
                        f"{path}: duplicate 00 00 0 row for {pair_key}"
                    )
                numeric = fields[3:]
                isotropic = {
                    order: _float(
                        numeric[column_index[order]],
                        f"{path}: {pair_key} {order}",
                    )
                    for order in CN_ORDERS
                }
            index += 1
        if isotropic is None:
            raise ReferenceFormatError(f"{path}: missing 00 00 0 row for {pair_key}")
        by_types[pair_key] = isotropic
        index += 1

    required_pairs = {
        tuple(sorted((atom_types[left], atom_types[right])))
        for left in atom_labels
        for right in atom_labels
    }
    missing = required_pairs - set(by_types)
    if missing:
        raise ReferenceFormatError(f"{path}: missing atom-type pairs {sorted(missing)}")

    matrices = {}
    for order in CN_ORDERS:
        matrix = tuple(
            tuple(
                by_types[tuple(sorted((atom_types[left], atom_types[right])))][order]
                for right in atom_labels
            )
            for left in atom_labels
        )
        for row in range(len(atom_labels)):
            for column in range(len(atom_labels)):
                if not math.isfinite(matrix[row][column]):
                    raise ReferenceFormatError(f"{path}: non-finite {order} value")
                if abs(matrix[row][column] - matrix[column][row]) > 1.0e-8:
                    raise ReferenceFormatError(f"{path}: {order} matrix is not symmetric")
        matrices[order] = matrix
    return matrices
```

When indexing `numeric`, use the C-column’s position relative to `C6`, not the complete token line. The shown `columns.index(order)` is correct because `columns` begins at `C6`.

- [ ] **Step 4: Run Cn tests**

Run:

```bash
python -P -m pytest -vv \
  tests/pytests/test_camcasp_reference.py::test_parse_all_isotropic_cn_matrices \
  tests/pytests/test_camcasp_reference.py::test_rejects_casimir_output_without_c12
```

Expected: both pass; the C10-only historical form is rejected.

- [ ] **Step 5: Commit**

```bash
git add devtools/camcasp_reference.py tests/pytests/test_camcasp_reference.py
git commit -m "devtools: parse isotropic CamCASP C6 through C12"
```

### Task 6: Define, validate, and atomically write the complete provenance schema

**Files:**
- Modify: `devtools/camcasp_reference.py`
- Modify: `tests/pytests/test_camcasp_reference.py`

**Interfaces:**
- Consumes: parsed frequencies/models/frames/tensors/Cn, tool metadata, checksums, and source artifact paths.
- Produces: schema-versioned `.camcasp-reference/atomic-polarizabilities.json` and deterministic copy-ready Python literals.

- [ ] **Step 1: Write failing schema, atomic-write, and literal tests**

Append:

```python
import json

from devtools.camcasp_reference import (  # noqa: E402
    validate_reference_document,
    write_atomic_json,
    render_python_literals,
)


def complete_document():
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    tensor = [[1.0, 0.0, 0.0], [0.0, 1.1, 0.0], [0.0, 0.0, 1.2]]
    atom = {
        "spherical": {
            "components": list(COMPONENTS_L3),
            "matrix": [[float(row == column) for column in range(16)] for row in range(16)],
        },
        "local_cartesian": tensor,
        "local_to_global": identity,
        "global_cartesian": tensor,
    }
    frequency_blocks = [
        {
            "index": index,
            "omega": 0.0 if index == 0 else index / 100.0,
            "atoms": {"O": atom, "H1": atom, "H2": atom},
        }
        for index in range(11)
    ]
    matrix = [[1.0, 0.5, 0.5], [0.5, 0.2, 0.2], [0.5, 0.2, 0.2]]
    return {
        "schema_version": 1,
        "generated_at_utc": "2026-07-29T12:00:00Z",
        "generator": {"path": "devtools/regenerate-camcasp.sh", "sha256": "a" * 64},
        "repository": {"commit": "b" * 40, "dirty": False},
        "tools": {
            "camcasp": {"version": "7.2.2 patch 003", "commit": "b4744425233a61786052832e1db4f109959c1ce9", "executables": {}},
            "orient": {"version": "5.0.11-ng", "commit": "d8d861098c8f548e2cf230c387c8431d9418650a", "executable": "/opt/orient"},
            "psi4": {"version": "1.11a1.dev31", "commit": "c" * 40, "dirty": True, "executable": "/opt/psi4"},
        },
        "scientific_protocol": {
            "geometry": {
                "units": "bohr",
                "charge": 0,
                "multiplicity": 1,
                "atom_order": ["O", "H1", "H2"],
                "atoms": [
                    {"label": label, "element": "O" if label == "O" else "H", "xyz": list(CANONICAL_GEOMETRY[label])}
                    for label in ("O", "H1", "H2")
                ],
                "orientation": ["symmetry c1", "no_com", "no_reorient"],
            },
            "electronic_structure": {
                "method": "PBE0",
                "basis": "aug-cc-pVTZ",
                "camcasp_basis": "aVTZ",
                "asymptotic_correction": "Psi4 GRAC",
                "ionization_potential_ev": 12.62063,
                "homo_hartree": -0.3989,
                "kernel": "ALDA+CHF",
                "grid": "Options Tests",
            },
            "frequency_grid": {"kind": "Gauss-Legendre", "nonzero_count": 10, "scale_au": 0.5},
            "model": {
                "nonlocal_rank": 4,
                "localization_method": "LW",
                "localization_limit": 3,
                "wsm_limit": 3,
                "hydrogen_limit": 3,
                "pfit_weight": 4,
                "pfit_weight_coefficient": 0.001,
                "pfit_cutoff": 0.0001,
            },
        },
        "frequencies": {
            "units": "hartree",
            "values": [0.0] + [index / 100.0 for index in range(1, 11)],
            "squared_source_values": ["0.0"] + [f"-{(index / 100.0) ** 2:.8f}" for index in range(1, 11)],
        },
        "polarizabilities": {
            "units": "atomic units",
            "spherical_frame": "atom-local real spherical",
            "cartesian_frame": "global Cartesian",
            "frequency_blocks": frequency_blocks,
        },
        "dispersion": {
            "component": "00 00 0",
            "atom_order": ["O", "H1", "H2"],
            "units": {
                "C6": "hartree * bohr^6",
                "C8": "hartree * bohr^8",
                "C10": "hartree * bohr^10",
                "C12": "hartree * bohr^12",
            },
            "matrices": {"C6": matrix, "C8": matrix, "C10": matrix, "C12": matrix},
        },
        "inputs": {"H2O.clt": {"sha256": "d" * 64}, "H2O.axes": {"sha256": "e" * 64}},
        "sources": {"refined_pol": {"path": "/work/refined.pol", "sha256": "f" * 64}},
    }


def test_validate_complete_schema():
    validate_reference_document(complete_document())


def test_validator_rejects_each_required_top_level_field():
    for key in (
        "schema_version", "generated_at_utc", "generator", "repository", "tools",
        "scientific_protocol", "frequencies", "polarizabilities", "dispersion",
        "inputs", "sources",
    ):
        document = complete_document()
        del document[key]
        try:
            validate_reference_document(document)
        except ReferenceFormatError as exc:
            assert key in str(exc)
        else:
            raise AssertionError(f"missing required field {key} was accepted")


def test_atomic_json_round_trip(tmp_path):
    output = tmp_path / "atomic-polarizabilities.json"
    document = complete_document()
    write_atomic_json(output, document)
    assert json.loads(output.read_text()) == document
    assert not list(tmp_path.glob("*.tmp"))


def test_literal_output_is_deterministic():
    first = render_python_literals(complete_document())
    second = render_python_literals(complete_document())
    assert first == second
    assert "REFERENCE_FREQUENCIES = np.array([" in first
    assert "REFERENCE_ATOMIC_C12 = np.array([" in first
```

- [ ] **Step 2: Run tests to verify red**

Run:

```bash
python -P -m pytest -vv -x \
  tests/pytests/test_camcasp_reference.py::test_validate_complete_schema \
  tests/pytests/test_camcasp_reference.py::test_validator_rejects_each_required_top_level_field \
  tests/pytests/test_camcasp_reference.py::test_atomic_json_round_trip \
  tests/pytests/test_camcasp_reference.py::test_literal_output_is_deterministic
```

Expected: imports fail because schema functions are absent.

- [ ] **Step 3: Implement explicit schema validation**

Define required fields once and validate their exact invariants:

```python
REQUIRED_TOP_LEVEL = (
    "schema_version",
    "generated_at_utc",
    "generator",
    "repository",
    "tools",
    "scientific_protocol",
    "frequencies",
    "polarizabilities",
    "dispersion",
    "inputs",
    "sources",
)


def _require(mapping: Mapping[str, object], key: str, context: str) -> object:
    if key not in mapping:
        raise ReferenceFormatError(f"{context}: missing required field {key}")
    return mapping[key]


def validate_reference_document(document: Mapping[str, object]) -> None:
    for key in REQUIRED_TOP_LEVEL:
        _require(document, key, "document")
    if document["schema_version"] != 1:
        raise ReferenceFormatError("schema_version must be 1")

    tools = document["tools"]
    if not isinstance(tools, Mapping):
        raise ReferenceFormatError("tools must be an object")
    for tool in ("camcasp", "orient", "psi4"):
        entry = _require(tools, tool, "tools")
        if not isinstance(entry, Mapping):
            raise ReferenceFormatError(f"tools.{tool} must be an object")
        _require(entry, "version", f"tools.{tool}")
        if tool != "psi4":
            _require(entry, "commit", f"tools.{tool}")
    psi4 = tools["psi4"]
    for key in ("commit", "dirty", "executable"):
        _require(psi4, key, "tools.psi4")

    protocol = document["scientific_protocol"]
    for key in ("geometry", "electronic_structure", "frequency_grid", "model"):
        _require(protocol, key, "scientific_protocol")
    geometry = protocol["geometry"]
    if geometry["atom_order"] != ["O", "H1", "H2"]:
        raise ReferenceFormatError("scientific_protocol.geometry.atom_order must be O,H1,H2")
    if geometry["units"] != "bohr" or geometry["charge"] != 0 or geometry["multiplicity"] != 1:
        raise ReferenceFormatError("canonical geometry metadata does not match the approved protocol")

    electronic = protocol["electronic_structure"]
    expected_electronic = {
        "method": "PBE0",
        "basis": "aug-cc-pVTZ",
        "camcasp_basis": "aVTZ",
        "asymptotic_correction": "Psi4 GRAC",
        "ionization_potential_ev": 12.62063,
        "homo_hartree": -0.3989,
        "kernel": "ALDA+CHF",
        "grid": "Options Tests",
    }
    for key, expected in expected_electronic.items():
        if electronic.get(key) != expected:
            raise ReferenceFormatError(f"electronic_structure.{key} must be {expected!r}")

    model = protocol["model"]
    expected_model = {
        "nonlocal_rank": 4,
        "localization_method": "LW",
        "localization_limit": 3,
        "wsm_limit": 3,
        "hydrogen_limit": 3,
        "pfit_weight": 4,
        "pfit_weight_coefficient": 0.001,
        "pfit_cutoff": 0.0001,
    }
    for key, expected in expected_model.items():
        if model.get(key) != expected:
            raise ReferenceFormatError(f"model.{key} must be {expected!r}")

    frequencies = document["frequencies"]
    values = frequencies["values"]
    squared = frequencies["squared_source_values"]
    if frequencies["units"] != "hartree" or len(values) != 11 or len(squared) != 11:
        raise ReferenceFormatError("frequencies must contain eleven hartree values")
    if values[0] != 0.0 or any(values[index] >= values[index + 1] for index in range(10)):
        raise ReferenceFormatError("frequencies must be static zero plus ten increasing values")

    polar = document["polarizabilities"]
    blocks = polar["frequency_blocks"]
    if len(blocks) != 11:
        raise ReferenceFormatError("polarizabilities requires eleven frequency_blocks")
    for expected_index, block in enumerate(blocks):
        if block["index"] != expected_index:
            raise ReferenceFormatError("polarizability blocks are not frequency-major")
        if list(block["atoms"]) != ["O", "H1", "H2"]:
            raise ReferenceFormatError("polarizability atom order must be O,H1,H2")
        for label, atom in block["atoms"].items():
            spherical = atom["spherical"]
            if spherical["components"] != list(COMPONENTS_L3):
                raise ReferenceFormatError(f"{label}: incomplete L3 component ordering")
            matrix = spherical["matrix"]
            if len(matrix) != 16 or any(len(row) != 16 for row in matrix):
                raise ReferenceFormatError(f"{label}: spherical matrix must be 16x16")
            for matrix_key in ("local_cartesian", "local_to_global", "global_cartesian"):
                candidate = atom[matrix_key]
                if len(candidate) != 3 or any(len(row) != 3 for row in candidate):
                    raise ReferenceFormatError(f"{label}.{matrix_key} must be 3x3")
            validate_rotation_matrix(tuple(tuple(row) for row in atom["local_to_global"]))
            _validate_symmetric(
                tuple(tuple(row) for row in atom["local_cartesian"]),
                f"{label}.local_cartesian",
            )
            _validate_symmetric(
                tuple(tuple(row) for row in atom["global_cartesian"]),
                f"{label}.global_cartesian",
            )

    dispersion = document["dispersion"]
    if dispersion["component"] != "00 00 0":
        raise ReferenceFormatError("dispersion component must be 00 00 0")
    if dispersion["atom_order"] != ["O", "H1", "H2"]:
        raise ReferenceFormatError("dispersion atom order must be O,H1,H2")
    for order in CN_ORDERS:
        matrix = dispersion["matrices"][order]
        if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
            raise ReferenceFormatError(f"{order} matrix must be 3x3")
        for row in range(3):
            for column in range(3):
                value = matrix[row][column]
                if not math.isfinite(value):
                    raise ReferenceFormatError(f"{order} contains non-finite values")
                if abs(value - matrix[column][row]) > 1.0e-8:
                    raise ReferenceFormatError(f"{order} matrix is not symmetric")

    for section in ("inputs", "sources"):
        entries = document[section]
        if not entries:
            raise ReferenceFormatError(f"{section} must not be empty")
        for name, entry in entries.items():
            digest = entry.get("sha256")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ReferenceFormatError(f"{section}.{name}.sha256 is invalid")
```

- [ ] **Step 4: Implement atomic JSON and deterministic literals**

Add:

```python
def write_atomic_json(path: Path, document: Mapping[str, object]) -> None:
    validate_reference_document(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _python_array(name: str, values: object) -> str:
    rendered = json.dumps(values, indent=4, allow_nan=False)
    return f"{name} = np.array({rendered}, dtype=float)\n"


def render_python_literals(document: Mapping[str, object]) -> str:
    validate_reference_document(document)
    blocks = document["polarizabilities"]["frequency_blocks"]
    packed = []
    for block in blocks:
        for label in ("O", "H1", "H2"):
            tensor = block["atoms"][label]["global_cartesian"]
            packed.append([
                tensor[0][0], tensor[0][1], tensor[0][2],
                tensor[1][1], tensor[1][2], tensor[2][2],
            ])
    static = packed[:3]
    output = [
        "# Generated by: bash devtools/regenerate-camcasp.sh\n",
        _python_array("REFERENCE_FREQUENCIES", [[value] for value in document["frequencies"]["values"]]),
        _python_array("REFERENCE_STATIC_ATOMIC_POLARIZABILITIES", static),
        _python_array("REFERENCE_DYNAMIC_ATOMIC_POLARIZABILITIES", packed),
    ]
    for order in CN_ORDERS:
        output.append(
            _python_array(
                f"REFERENCE_ATOMIC_{order}",
                document["dispersion"]["matrices"][order],
            )
        )
    return "\n".join(output)
```

- [ ] **Step 5: Run schema and rendering tests**

Run:

```bash
python -P -m pytest -vv \
  tests/pytests/test_camcasp_reference.py::test_validate_complete_schema \
  tests/pytests/test_camcasp_reference.py::test_validator_rejects_each_required_top_level_field \
  tests/pytests/test_camcasp_reference.py::test_atomic_json_round_trip \
  tests/pytests/test_camcasp_reference.py::test_literal_output_is_deterministic
```

Expected: all pass, with no residual temporary file.

- [ ] **Step 6: Commit**

```bash
git add devtools/camcasp_reference.py tests/pytests/test_camcasp_reference.py
git commit -m "devtools: validate CamCASP provenance schema"
```

### Task 7: Materialize and attest the canonical CamCASP/Psi4 calculation

**Files:**
- Modify: `devtools/regenerate-camcasp.sh`
- Modify: `devtools/camcasp_reference.py`
- Modify: `tests/pytests/test_camcasp_reference.py`

**Interfaces:**
- Consumes: provisioned tools and approved canonical constants.
- Produces: explicit `.clt` and `.axes`, generated Psi4/CKS inputs, NL4 and p2p outputs, and checksummed stage artifacts under `.camcasp-reference/`.

- [ ] **Step 1: Write failing canonical-input and artifact-failure tests**

Append:

```python
from devtools.camcasp_reference import (  # noqa: E402
    validate_generated_protocol,
    validate_stage_artifacts,
)


def test_generated_protocol_requires_explicit_canonical_settings():
    clt = """\
Run-type properties
  Basis aVTZ
  SCFcode psi4
  Method DFT
  Functional PBE0
  Kernel ALDA+CHF
  Options Tests
  Localization
End
"""
    cks = """\
SET QUAD
  Type Gauss-Legendre
  Beta 0.5
END
BEGIN Polarizability
  Quad 10
  Rank 4
  Print pols for Orient
END
"""
    cluster_log = "AC options: type = GRAC\nfunctional PBE0\nkernel = ALDA+CHF\n"
    psi4_input = "symmetry c1\nno_com\nno_reorient\n"
    validate_generated_protocol(clt, cks, cluster_log, psi4_input)


def populate_stage_artifacts(work, job="H2O"):
    for index in range(11):
        for name in (
            f"{job}_L3_{index:03d}.out",
            f"{job}_ref_wt4_L3_{index:03d}.out",
            f"{job}_ref_wt4_L3_{index:03d}.pol",
        ):
            (work / name).write_text("Finished\n")
    (work / f"{job}_ref_wt4_L3_0f10.pol").write_text("complete\n")
    (work / f"{job}_ref_wt4_L3_casimir.out").write_text("Dispersion coefficients\n")
    (work / f"{job}_ref_wt4_L3_C12.pot").write_text("C12\n")
    (work / f"{job}.pdef").write_text("Polarizabilities\nEnd\n")


def test_stage_validation_rejects_missing_orient_block(tmp_path):
    populate_stage_artifacts(tmp_path)
    (tmp_path / "H2O_L3_007.out").unlink()
    try:
        validate_stage_artifacts(tmp_path, "H2O")
    except ReferenceFormatError as exc:
        assert "H2O_L3_007.out" in str(exc)
    else:
        raise AssertionError("missing ORIENT block was accepted")


def test_stage_validation_rejects_missing_pfit_block(tmp_path):
    populate_stage_artifacts(tmp_path)
    (tmp_path / "H2O_ref_wt4_L3_010.pol").unlink()
    try:
        validate_stage_artifacts(tmp_path, "H2O")
    except ReferenceFormatError as exc:
        assert "H2O_ref_wt4_L3_010.pol" in str(exc)
    else:
        raise AssertionError("missing PFIT block was accepted")


def test_stage_validation_rejects_missing_c12(tmp_path):
    populate_stage_artifacts(tmp_path)
    (tmp_path / "H2O_ref_wt4_L3_C12.pot").unlink()
    try:
        validate_stage_artifacts(tmp_path, "H2O")
    except ReferenceFormatError as exc:
        assert "C12" in str(exc)
    else:
        raise AssertionError("missing C12 output was accepted")
```

- [ ] **Step 2: Run tests to verify red**

Run:

```bash
python -P -m pytest -vv -x \
  tests/pytests/test_camcasp_reference.py::test_generated_protocol_requires_explicit_canonical_settings \
  tests/pytests/test_camcasp_reference.py::test_stage_validation_rejects_missing_orient_block \
  tests/pytests/test_camcasp_reference.py::test_stage_validation_rejects_missing_pfit_block \
  tests/pytests/test_camcasp_reference.py::test_stage_validation_rejects_missing_c12
```

Expected: missing protocol/artifact validators cause collection failure.

- [ ] **Step 3: Implement generated-protocol attestation**

Add:

```python
def _require_text(text: str, pattern: str, context: str) -> None:
    if re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE) is None:
        raise ReferenceFormatError(f"{context}: missing required pattern {pattern!r}")


def validate_generated_protocol(
    clt_text: str,
    cks_text: str,
    cluster_log_text: str,
    psi4_input_text: str,
) -> None:
    for pattern in (
        r"\bBasis\s+aVTZ\b",
        r"\bSCFcode\s+psi4\b",
        r"\bMethod\s+DFT\b",
        r"\bFunctional\s+PBE0\b",
        r"\bKernel\s+ALDA\+CHF\b",
        r"\bOptions\s+Tests\b",
        r"\bLocalization\b",
    ):
        _require_text(clt_text, pattern, "H2O.clt")
    for pattern in (
        r"Type\s+Gauss-Legendre",
        r"Beta\s+0\.5\b",
        r"Quad\s+10\b",
        r"Rank\s+4\b",
        r"Print\s+pols\s+for\s+Orient",
    ):
        _require_text(cks_text, pattern, "H2O.cks")
    for pattern in (
        r"AC options:\s*type\s*=\s*GRAC",
        r"functional\s+PBE0",
        r"kernel\s*=\s*ALDA\+CHF",
    ):
        _require_text(cluster_log_text, pattern, "cluster output")
    for pattern in (r"symmetry\s+c1", r"\bno_com\b", r"\bno_reorient\b"):
        _require_text(psi4_input_text, pattern, "generated Psi4 input")
```

This deliberately attests GRAC from generated output because CamCASP 7.2.2 selects its Psi4 GRAC protocol internally; an `AC GRAC` cluster line would instead be parsed as an LB94 correction joined with GRAC and must not be inserted.

- [ ] **Step 4: Implement complete artifact checks**

Add:

```python
def validate_stage_artifacts(work_dir: Path, job: str) -> dict[str, Path]:
    required = []
    for index in range(11):
        required.extend(
            (
                work_dir / f"{job}_L3_{index:03d}.out",
                work_dir / f"{job}_ref_wt4_L3_{index:03d}.out",
                work_dir / f"{job}_ref_wt4_L3_{index:03d}.pol",
            )
        )
    required.extend(
        (
            work_dir / f"{job}_ref_wt4_L3_0f10.pol",
            work_dir / f"{job}_ref_wt4_L3_casimir.out",
            work_dir / f"{job}_ref_wt4_L3_C12.pot",
            work_dir / f"{job}.pdef",
        )
    )
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise ReferenceFormatError(f"missing or empty stage artifact: {path}")
    for path in work_dir.glob("*.out"):
        text = path.read_text(errors="replace")
        if re.search(
            r"segmentation fault|fatal error|error stop|pfit_error|orient\.error",
            text,
            flags=re.IGNORECASE,
        ):
            raise ReferenceFormatError(f"error marker in stage output: {path}")
    return {path.name: path for path in required}
```

- [ ] **Step 5: Materialize exact scientific inputs from the shell**

Add `prepare_layout` and `write_inputs`:

```bash
prepare_layout() {
    CURRENT_STAGE="layout"
    require_safe_generated_path "$REFERENCE_ROOT"
    mkdir -p \
        "$REFERENCE_ROOT/inputs" \
        "$REFERENCE_ROOT/work" \
        "$REFERENCE_ROOT/scratch/camcasp" \
        "$REFERENCE_ROOT/scratch/psi4" \
        "$REFERENCE_ROOT/logs" \
        "$REFERENCE_ROOT/tools"
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
  Localization
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
}
```

- [ ] **Step 6: Run CamCASP in an isolated foreground job**

Add:

```bash
run_camcasp() {
    CURRENT_STAGE="camcasp"
    local job_dir="$REFERENCE_ROOT/work/H2O"
    rm -rf "$job_dir"
    run_logged camcasp "$REFERENCE_ROOT/logs/camcasp.log" \
        "$CAMCASP/bin/runcamcasp.py" H2O \
        --clt "$REFERENCE_ROOT/inputs/H2O.clt" \
        --directory "$job_dir" \
        --ifexists delete \
        --scfcode psi4 \
        --queue none \
        --scratch "$SCRATCH" \
        --cores "${CAMCASP_REFERENCE_CORES:-1}" \
        --debug
    cp "$REFERENCE_ROOT/inputs/H2O.axes" "$job_dir/H2O.axes"

    [[ -s "$job_dir/H2O.cks" ]] || fail "missing generated H2O.cks"
    [[ -s "$job_dir/H2O.clt.clout" ]] || fail "missing cluster output"
    [[ -s "$job_dir/H2O.ornt" ]] || fail "missing ORIENT input"
    [[ -s "$job_dir/H2O.prss" ]] || fail "missing PROCESS input"
    [[ -s "$job_dir/H2O_casimir.prss" ]] || fail "missing CASIMIR PROCESS input"
    [[ -s "$job_dir/H2O.sites" ]] || fail "missing site definitions"

    mapfile -t NL4_FILES < <(find "$job_dir/OUT" -maxdepth 1 -type f -name '*NL4*pol' -print)
    mapfile -t P2P_FILES < <(find "$job_dir/OUT" -maxdepth 1 -type f -name '*.p2p' -print)
    (( ${#NL4_FILES[@]} == 1 )) ||
        fail "expected exactly one NL4 polarizability file, found ${#NL4_FILES[@]}"
    (( ${#P2P_FILES[@]} == 1 )) ||
        fail "expected exactly one p2p file, found ${#P2P_FILES[@]}"
}
```

After setup, call a Python `attest-protocol` CLI or an inline import of `validate_generated_protocol` against:

- `.camcasp-reference/inputs/H2O.clt`
- `.camcasp-reference/work/H2O/H2O.cks`
- `.camcasp-reference/work/H2O/H2O.clt.clout`
- the single generated `H2O_A.in` or `H2O_A.dat` Psi4 input discovered by exact filename; reject zero or multiple candidates.

- [ ] **Step 7: Run focused tests**

Run:

```bash
python -P -m pytest -vv \
  tests/pytests/test_camcasp_reference.py::test_generated_protocol_requires_explicit_canonical_settings \
  tests/pytests/test_camcasp_reference.py::test_stage_validation_rejects_missing_orient_block \
  tests/pytests/test_camcasp_reference.py::test_stage_validation_rejects_missing_pfit_block \
  tests/pytests/test_camcasp_reference.py::test_stage_validation_rejects_missing_c12
bash -n devtools/regenerate-camcasp.sh
```

Expected: all tests and shell syntax pass.

- [ ] **Step 8: Commit**

```bash
git add devtools/regenerate-camcasp.sh devtools/camcasp_reference.py tests/pytests/test_camcasp_reference.py
git commit -m "devtools: attest canonical CamCASP protocol"
```

### Task 8: Run LW localization, WSM refinement, CASIMIR, review gate, and final document assembly

**Files:**
- Modify: `devtools/regenerate-camcasp.sh`
- Modify: `devtools/camcasp_reference.py`
- Modify: `tests/pytests/test_camcasp_reference.py`

**Interfaces:**
- Consumes: the attested canonical CamCASP job and exact NL4/p2p outputs.
- Produces: eleven ORIENT and PFIT blocks, reviewed L3 `.pdef`, C12 output, validated JSON, and copy-ready literals.

- [ ] **Step 1: Add a failing end-to-end builder test using synthetic artifacts**

Append:

```python
from devtools.camcasp_reference import build_reference_document  # noqa: E402


def test_builder_combines_all_required_properties(tmp_path):
    nl4 = tmp_path / "NL4_fmtB.pol"
    refined = tmp_path / "H2O_ref_wt4_L3_0f10.pol"
    pot = tmp_path / "H2O_ref_wt4_L3_C12.pot"
    axes = tmp_path / "H2O.axes"
    nl4.write_text(make_nl4_frequency_text())
    refined.write_text(make_l3_refined_text())
    pot.write_text(CASIMIR_C12)
    axes.write_text(CANONICAL_AXES)

    document = build_reference_document(
        frequency_path=nl4,
        refined_path=refined,
        casimir_path=pot,
        axes_path=axes,
        metadata=complete_document(),
    )
    validate_reference_document(document)
    assert len(document["polarizabilities"]["frequency_blocks"]) == 11
    assert tuple(document["dispersion"]["matrices"]) == ("C6", "C8", "C10", "C12")
    assert document["scientific_protocol"]["model"]["hydrogen_limit"] == 3
```

- [ ] **Step 2: Run the builder test to verify red**

Run:

```bash
python -P -m pytest -vv -x \
  tests/pytests/test_camcasp_reference.py::test_builder_combines_all_required_properties
```

Expected: import or signature failure because the document builder is absent.

- [ ] **Step 3: Implement document assembly**

Add the exact builder signature used by the test:

```python
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_reference_document(
    *,
    frequency_path: Path,
    refined_path: Path,
    casimir_path: Path,
    axes_path: Path,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    document = json.loads(json.dumps(metadata))
    frequencies = parse_frequencies(frequency_path)
    models = parse_refined_polarizabilities(
        refined_path, ("O", "H1", "H2"), limit=3
    )
    frames = build_local_frames(CANONICAL_GEOMETRY, axes_path.read_text())
    cn = parse_isotropic_cn(
        casimir_path,
        ("O", "H1", "H2"),
        {"O": "O", "H1": "H", "H2": "H"},
    )

    frequency_blocks = []
    for point, block in zip(frequencies, models):
        atoms = {}
        for label in ("O", "H1", "H2"):
            spherical = block.atoms[label]
            local = dipole_local_cartesian(spherical)
            global_tensor = rotate_tensor(local, frames[label])
            atoms[label] = {
                "spherical": {
                    "components": list(spherical.components),
                    "matrix": [list(row) for row in spherical.matrix],
                },
                "local_cartesian": [list(row) for row in local],
                "local_to_global": [list(row) for row in frames[label]],
                "global_cartesian": [list(row) for row in global_tensor],
            }
        frequency_blocks.append(
            {"index": point.index, "omega": point.omega, "atoms": atoms}
        )

    document["frequencies"] = {
        "units": "hartree",
        "values": [point.omega for point in frequencies],
        "squared_source_values": [
            point.squared_source_text for point in frequencies
        ],
    }
    document["polarizabilities"]["frequency_blocks"] = frequency_blocks
    document["dispersion"]["matrices"] = {
        order: [list(row) for row in cn[order]] for order in CN_ORDERS
    }
    document["sources"].update(
        {
            "nonlocal_pol": {"path": str(frequency_path.resolve()), "sha256": _sha256(frequency_path)},
            "refined_pol": {"path": str(refined_path.resolve()), "sha256": _sha256(refined_path)},
            "casimir_pot": {"path": str(casimir_path.resolve()), "sha256": _sha256(casimir_path)},
            "axes": {"path": str(axes_path.resolve()), "sha256": _sha256(axes_path)},
        }
    )
    validate_reference_document(document)
    return document
```

Define `CANONICAL_GEOMETRY` in the module with the exact design coordinates rather than importing test data.

- [ ] **Step 4: Run the complete local unit suite**

Run:

```bash
python -P -m pytest -vv -ra tests/pytests/test_camcasp_reference.py
```

Expected: all workflow unit tests pass without external software.

- [ ] **Step 5: Add the explicit localization/refinement/CASIMIR command**

Add:

```bash
run_localize() {
    CURRENT_STAGE="localize-refine-dispersion"
    local job_dir="$REFERENCE_ROOT/work/H2O"
    local polfile="${NL4_FILES[0]}"

    (
        cd "$job_dir"
        run_logged localize-refine-dispersion \
            "$REFERENCE_ROOT/logs/localize-refine-dispersion.log" \
            "$CAMCASP/bin/localize.py" H2O \
                --axes H2O.axes \
                --polfile "$polfile" \
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

    python -P "$REPO_ROOT/devtools/camcasp_reference.py" \
        validate-artifacts --work-dir "$job_dir" --job H2O
}
```

Do not pass `--sites`; CamCASP 7.2.2 `localize.py` contains the `arg.sites` typo on that path. The default `H2O.sites` path is valid.

Because `localize.py` does not reliably propagate every ORIENT/PFIT subprocess return code, the subsequent validator must require all eleven ORIENT outputs, all eleven PFIT outputs, the combined L3 file, CASIMIR output, and `_C12.pot`.

- [ ] **Step 6: Add the scientific `.pdef` review gate**

After localization, compute and save the candidate checksum:

```bash
require_reviewed_pdef() {
    CURRENT_STAGE="review-pdef"
    local pdef="$REFERENCE_ROOT/work/H2O/H2O.pdef"
    local checksum_file="$REFERENCE_ROOT/work/H2O/H2O.pdef.sha256"
    local digest
    digest="$(sha256sum "$pdef" | awk '{print $1}')"
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
```

This gate must run before JSON creation. A first successful external calculation is therefore expected to stop with status 78 until a human has inspected the generated model.

- [ ] **Step 7: Add metadata collection and CLI build**

Have the shell write `.camcasp-reference/work/manifest.json` using Python `json.dump(..., allow_nan=False)` with:

- UTC generation timestamp.
- Generator path/checksum.
- repository commit and dirty state.
- CamCASP version, commit, five executable absolute paths/checksums.
- Orient version, commit, executable absolute path/checksum.
- Psi4 version, Git revision, dirty state, executable path/checksum.
- exact scientific protocol object from Task 6.
- input and source paths/checksums.

Then invoke:

```bash
python -P "$REPO_ROOT/devtools/camcasp_reference.py" build \
    --manifest "$REFERENCE_ROOT/work/manifest.json" \
    --frequency-file "${NL4_FILES[0]}" \
    --refined-file "$REFERENCE_ROOT/work/H2O/H2O_ref_wt4_L3_0f10.pol" \
    --casimir-file "$REFERENCE_ROOT/work/H2O/H2O_ref_wt4_L3_C12.pot" \
    --axes-file "$REFERENCE_ROOT/inputs/H2O.axes" \
    --output "$REFERENCE_ROOT/atomic-polarizabilities.json"
```

The CLI must:

1. Load the manifest.
2. Call `build_reference_document`.
3. Call `validate_reference_document`.
4. Call `write_atomic_json`.
5. Print `render_python_literals(document)` only after the atomic write succeeds.
6. Return nonzero without creating/replacing JSON on any parse or validation failure.

- [ ] **Step 8: Wire the full shell main sequence**

Use this order:

```bash
main() {
    trap on_exit EXIT
    parse_arguments "$@"
    preflight
    if [[ "$MODE" == "preflight" ]]; then
        printf 'Preflight passed for %s\n' "$PSI4_EXE"
        return 0
    fi

    prepare_layout
    provision_orient
    provision_camcasp
    write_psi4_wrapper
    export_reference_environment
    write_inputs
    run_camcasp
    attest_generated_protocol
    run_localize
    require_reviewed_pdef
    write_manifest
    build_reference_json
}
```

Do not reorder JSON creation before protocol, stage, `.pdef`, frame, tensor, or C12 validation.

- [ ] **Step 9: Run all local tests and CLI validation tests**

Run:

```bash
bash -n devtools/regenerate-camcasp.sh
python -m py_compile devtools/camcasp_reference.py
python -P -m pytest -vv -ra tests/pytests/test_camcasp_reference.py
git diff --check
```

Expected: syntax/compile checks and all pure workflow tests pass.

- [ ] **Step 10: Commit**

```bash
git add devtools/regenerate-camcasp.sh devtools/camcasp_reference.py tests/pytests/test_camcasp_reference.py
git commit -m "devtools: generate reviewed CamCASP provenance"
```

## Final External Acceptance Gate

This gate is required before canonical L3 references are declared available. It may fail because of network access, Orient binary compatibility, external executable behavior, scientific input mismatch, or `.pdef` review. Such failures are blocking evidence, not permission to substitute historical L2 values.

- [ ] **Step 1: Confirm the intended staged Psi4**

Run:

```bash
export PATH="$PWD/build_camcasp/stage/bin:$PATH"
export PYTHONPATH="$PWD/build_camcasp/stage/lib${PYTHONPATH:+:$PYTHONPATH}"
python -P -c 'import psi4; print(psi4.__version__, psi4.core.__file__)'
```

Expected: the version and module path point to `build_camcasp/stage`, not the source-tree Python package.

- [ ] **Step 2: Run the full external workflow for candidate generation**

Run:

```bash
PSI4_EXE="$PWD/build_camcasp/stage/bin/psi4" \
CAMCASP="$PWD/camcasp-bin" \
bash -x devtools/regenerate-camcasp.sh \
  2>&1 | tee /tmp/regenerate-camcasp.log
test "${PIPESTATUS[0]}" -eq 78
test ! -e .camcasp-reference/atomic-polarizabilities.json
```

Expected after all external scientific stages succeed: status 78 at `review-pdef`, retained logs and candidate `.pdef`, and no JSON. If it stops earlier, retain the named stage log and fix that stage before continuing.

- [ ] **Step 3: Perform the required manual scientific review**

Run:

```bash
cat .camcasp-reference/work/H2O/H2O.pdef
cat .camcasp-reference/work/H2O/H2O.pdef.sha256
grep -nE 'Limit:|WSM-Limit:|H-Limit:|Loc algorithm:|Weight:|Weight coeff:|Pol Cutoff:' \
  .camcasp-reference/work/H2O/H2O_ref_wt4_L3_0f10.pol
```

Acceptance requires inspection confirming:

- O has complete L3 components.
- H1 and H2 have complete L3 components.
- H2 copies the symmetry-equivalent H1 parameter definitions where intended.
- Local axes are the approved symmetry-related axes.
- Localization/WSM/H limits are 3/3/3.
- Method is LW.
- Weight is 4.
- Weight coefficient is 0.001.
- Cutoff is 0.0001.

- [ ] **Step 4: Rerun with the reviewed checksum**

After completing the review, run:

```bash
export CAMCASP_PDEF_SHA256="$(
  awk '{print $1}' .camcasp-reference/work/H2O/H2O.pdef.sha256
)"
PSI4_EXE="$PWD/build_camcasp/stage/bin/psi4" \
CAMCASP="$PWD/camcasp-bin" \
bash devtools/regenerate-camcasp.sh \
  2>&1 | tee /tmp/regenerate-camcasp-reviewed.log
```

Expected: exit status 0, all eleven ORIENT and PFIT blocks validated, C12 present, JSON written atomically, and copy-ready literals printed. If the regenerated `.pdef` checksum changes, the script must stop at review status 78 again.

- [ ] **Step 5: Validate the final JSON independently**

Run:

```bash
python -m json.tool \
  .camcasp-reference/atomic-polarizabilities.json >/dev/null
python -P devtools/camcasp_reference.py validate \
  .camcasp-reference/atomic-polarizabilities.json
python - <<'PY'
import json
from pathlib import Path

path = Path(".camcasp-reference/atomic-polarizabilities.json")
data = json.loads(path.read_text())
assert data["schema_version"] == 1
assert data["scientific_protocol"]["model"] == {
    "nonlocal_rank": 4,
    "localization_method": "LW",
    "localization_limit": 3,
    "wsm_limit": 3,
    "hydrogen_limit": 3,
    "pfit_weight": 4,
    "pfit_weight_coefficient": 0.001,
    "pfit_cutoff": 0.0001,
}
assert len(data["frequencies"]["values"]) == 11
assert data["frequencies"]["values"][0] == 0.0
assert len(data["polarizabilities"]["frequency_blocks"]) == 11
for block in data["polarizabilities"]["frequency_blocks"]:
    assert list(block["atoms"]) == ["O", "H1", "H2"]
    for atom in block["atoms"].values():
        assert len(atom["spherical"]["components"]) == 16
        assert len(atom["spherical"]["matrix"]) == 16
assert list(data["dispersion"]["matrices"]) == ["C6", "C8", "C10", "C12"]
print("validated canonical L3 provenance:", path)
PY
```

Expected: all commands pass and print the validated JSON path.

- [ ] **Step 6: Confirm generated material remains ignored and the tracked tree is clean**

Run:

```bash
git check-ignore -q .camcasp-reference/atomic-polarizabilities.json
git check-ignore -q orient/
git check-ignore -q camcasp-bin/
! git check-ignore --no-index -q devtools/regenerate-camcasp.sh
git diff --check
git diff --cached --name-only
git status --short
```

Expected:

- The JSON and external tool trees are ignored.
- The orchestrator is not ignored.
- `git diff --cached --name-only` is empty.
- `git status --short` is empty after the implementation commits.
- No generated JSON, external binaries, Orient material, or candidate/reference values are committed.

## Dependency Order

1. Task 1 establishes tracking and shared shell failure semantics.
2. Task 2 depends on Task 1’s shell scaffold.
3. Tasks 3–6 depend on the Python module introduced in Task 3.
4. Task 7 depends on provisioning from Task 2 and validation interfaces from Tasks 3–6.
5. Task 8 depends on every preceding task.
6. The final external gate depends on all tracked implementation tasks and external access to the approved toolchain.
7. Production Psi4 response/localization work and hard-coded production parity pytest values remain blocked until the final external gate succeeds and its JSON/protocol receive review.

## Risks

- The supplied Orient binary may be incompatible with the host runtime; the workflow must stop with `make OPENGL=no` guidance.
- CamCASP helper scripts do not propagate every ORIENT/PFIT subprocess failure; artifact and log validation is therefore mandatory.
- CamCASP 7.2.2’s explicit `AC GRAC` parsing does not mean “Psi4 GRAC correction.” The workflow must attest the generated `AC options: type = GRAC` result instead of inserting the misleading cluster directive.
- Exact spherical normalization/order must remain tied to documented CamCASP real-spherical conventions and inspected canonical output. Trace agreement alone cannot validate component signs.
- The generated `.pdef` is a scientific model, not merely a file-existence condition; its checksum gate prevents unreviewed JSON.
- Compiler, BLAS, Psi4, or Orient differences may affect numerical reproducibility. Record versions/checksums and do not relax later production tolerances without scientific review.
- No historical L2/H1, weight-3, or C10-only artifact can satisfy this milestone.
- If any external stage, mapping check, L3 completeness check, or C12 check fails, stop without writing or replacing the canonical JSON.