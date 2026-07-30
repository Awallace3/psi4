import fcntl
import gzip
import hashlib
import os
import subprocess
from pathlib import Path

import pytest

import devtools.camcasp_reference as camcasp_reference

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "devtools" / "regenerate-camcasp.sh"


@pytest.fixture(scope="session", autouse=True)
def set_up_overall():
    """Override Psi4's parent fixture for this dependency-free tooling module."""


@pytest.fixture(autouse=True)
def set_up():
    """Keep the pure tooling tests independent of a staged Psi4 build."""


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


REALCG_BASENAMES = tuple(
    [f"realcg_{left}_{right}" for left in range(1, 5) for right in range(1, 5)]
    + ["realcg_notes"]
)


def _write_realcg_tables(root):
    realcg = root / "data" / "realcg"
    realcg.mkdir(parents=True)
    for name in REALCG_BASENAMES:
        (realcg / name).write_text(f"sealed {name}\n")


def _make_camcasp_source_checkout(path, sentinel=b""):
    (path / "bin").mkdir(parents=True)
    _write_realcg_tables(path)
    (path / "VERSION").write_text("CamCASP VERSION 7.2.2 patch 003\n")
    if sentinel is not None:
        (path / "bin" / "no_psi4").write_bytes(sentinel)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"], check=True
    )
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-qm", "stub CamCASP"],
        check=True,
    )
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _make_attestable_camcasp_source(
    path, *, ignore_reference=False, runcamcasp_text=None
):
    bin_dir = path / "bin"
    archive_dir = path / "x86-64" / "gfortran"
    bin_dir.mkdir(parents=True)
    archive_dir.mkdir(parents=True)
    _write_realcg_tables(path)
    (path / "VERSION").write_text("CamCASP VERSION 7.2.2 patch 003\n")
    (bin_dir / "no_psi4").write_bytes(b"")
    _write_executable(
        bin_dir / "runcamcasp.py",
        runcamcasp_text or "#!/usr/bin/env bash\nexit 0\n",
    )
    _write_executable(bin_dir / "localize.py", "#!/usr/bin/env bash\nexit 0\n")
    for name in ("camcasp", "cluster", "process", "pfit", "casimir"):
        with gzip.open(archive_dir / f"{name}.gz", "wb") as handle:
            handle.write(f"#!/usr/bin/env bash\necho {name}\n".encode())
    if ignore_reference:
        (path / ".gitignore").write_text("reference/\n")
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"], check=True
    )
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-qm", "attestable CamCASP"],
        check=True,
    )
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


@pytest.mark.parametrize("sentinel", (None, b"not empty\n"))
def test_verify_camcasp_source_rejects_missing_or_nonempty_sentinel(
    tmp_path, sentinel
):
    source = tmp_path / "camcasp-source"
    commit = _make_camcasp_source_checkout(source, sentinel)
    result = subprocess.run(
        [
            "bash", "-c",
            f'source "{SCRIPT}"; CAMCASP_COMMIT="$1"; '
            'verify_camcasp_source "$2"',
            "camcasp-sentinel", commit, str(source),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "tracked empty bin/no_psi4" in result.stderr


@pytest.mark.parametrize("failure_kind", ("archive", "materialize"))
def test_camcasp_runtime_materialization_failure_is_fail_closed(
    tmp_path, failure_kind
):
    source = tmp_path / "camcasp-source"
    commit = _make_camcasp_source_checkout(source)
    reference = tmp_path / "reference"
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    real_git = subprocess.check_output(["bash", "-c", "command -v git"], text=True).strip()
    real_tar = subprocess.check_output(["bash", "-c", "command -v tar"], text=True).strip()
    git_wrapper = fake_bin / "git"
    git_wrapper.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$FAILURE_KIND\" == archive && \" $* \" == *\" archive \"* ]]; then exit 88; fi\n"
        f'exec "{real_git}" "$@"\n'
    )
    git_wrapper.chmod(0o755)
    tar_wrapper = fake_bin / "tar"
    tar_wrapper.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$FAILURE_KIND\" == materialize ]]; then exit 89; fi\n"
        f'exec "{real_tar}" "$@"\n'
    )
    tar_wrapper.chmod(0o755)
    command = (
        f'source "{SCRIPT}"; '
        'REFERENCE_ROOT="$1"; CAMCASP_SOURCE_ROOT="$2"; '
        'CAMCASP_COMMIT="$3"; CAMCASP="$REFERENCE_ROOT/tools/camcasp-runtime"; '
        'verify_camcasp_source "$CAMCASP_SOURCE_ROOT"; '
        'materialize_camcasp_runtime'
    )
    result = subprocess.run(
        ["bash", "-c", command, "camcasp-materialize", str(reference), str(source), commit],
        cwd=ROOT,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "FAILURE_KIND": failure_kind,
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert not (reference / "tools" / "camcasp-runtime").exists()
    failure_log = (
        reference / "logs" /
        ("camcasp-archive.log" if failure_kind == "archive" else "camcasp-materialization.log")
    )
    assert failure_log.is_file()
    assert failure_log.with_name(failure_log.name + ".sha256").is_file()
    assert subprocess.check_output(
        [real_git, "-C", str(source), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    ) == ""


def test_verify_camcasp_runtime_rejects_surviving_sentinel(tmp_path):
    source = tmp_path / "camcasp-source"
    commit = _make_camcasp_source_checkout(source)
    reference = tmp_path / "reference"
    runtime = reference / "tools" / "camcasp-runtime"
    command = (
        f'source "{SCRIPT}"; '
        'REFERENCE_ROOT="$1"; CAMCASP_SOURCE_ROOT="$2"; '
        'CAMCASP_COMMIT="$3"; CAMCASP="$4"; '
        'verify_camcasp_source "$CAMCASP_SOURCE_ROOT"; '
        'materialize_camcasp_runtime; : >"$CAMCASP/bin/no_psi4"; '
        'verify_camcasp_runtime'
    )
    result = subprocess.run(
        ["bash", "-c", command, "camcasp-runtime", str(reference), str(source), commit, str(runtime)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "runtime sentinel unexpectedly exists" in result.stderr
    assert (source / "bin" / "no_psi4").read_bytes() == b""


@pytest.mark.parametrize(
    "mutation",
    (
        "executable", "wrapper", "driver", "archive", "symlink", "sentinel",
        "realcg", "realcg-nonregular",
    ),
)
def test_camcasp_runtime_attestation_rejects_post_install_mutation(
    tmp_path, mutation
):
    source = tmp_path / "camcasp-source"
    commit = _make_attestable_camcasp_source(source)
    reference = tmp_path / "reference"
    psi4 = tmp_path / "psi4"
    _write_executable(psi4, "#!/usr/bin/env bash\necho 'Psi4 stub'\n")
    command = f'''source "{SCRIPT}"
REFERENCE_ROOT="$1"
CAMCASP_SOURCE_ROOT="$2"
CAMCASP_COMMIT="$3"
CAMCASP="$REFERENCE_ROOT/tools/camcasp-runtime"
PSI4_EXE="$4"
provision_camcasp
write_psi4_wrapper
write_camcasp_runtime_attestation
case "$5" in
  executable) printf mutation >>"$CAMCASP/x86-64/gfortran/exe/camcasp" ;;
  wrapper) printf mutation >>"$CAMCASP/bin/psi4.sh" ;;
  driver) printf mutation >>"$CAMCASP/bin/runcamcasp.py" ;;
  archive) printf mutation >>"$CAMCASP/x86-64/gfortran/camcasp.gz" ;;
  symlink) ln -sfn ../localize.py "$CAMCASP/bin/camcasp" ;;
  sentinel) : >"$CAMCASP/bin/no_psi4" ;;
  realcg) printf mutation >>"$CAMCASP/data/realcg/realcg_3_3" ;;
  realcg-nonregular) mkdir "$CAMCASP/data/realcg/unexpected-directory" ;;
esac
verify_camcasp_runtime_attestation
: >"$REFERENCE_ROOT/work/manifest.json"
: >"$REFERENCE_ROOT/atomic-polarizabilities.json"
'''
    result = subprocess.run(
        [
            "bash", "-c", command, "runtime-mutation", str(reference),
            str(source), commit, str(psi4), mutation,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "runtime attestation mismatch" in result.stderr
    assert not (reference / "work" / "manifest.json").exists()
    assert not (reference / "atomic-polarizabilities.json").exists()


@pytest.mark.parametrize("relationship", ("source-inside-runtime", "runtime-inside-source"))
def test_camcasp_runtime_rejects_ancestor_roots_before_mutation(
    tmp_path, relationship
):
    if relationship == "source-inside-runtime":
        reference = tmp_path / "reference"
        runtime = reference / "tools" / "camcasp-runtime"
        source = runtime / "source"
        commit = _make_attestable_camcasp_source(source)
    else:
        source = tmp_path / "source"
        commit = _make_attestable_camcasp_source(source, ignore_reference=True)
        reference = source / "reference"
        runtime = reference / "tools" / "camcasp-runtime"
    sentinel = source / "bin" / "no_psi4"
    sentinel_digest = hashlib.sha256(sentinel.read_bytes()).hexdigest()
    status_before = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    )
    command = (
        f'source "{SCRIPT}"; REFERENCE_ROOT="$1"; '
        'CAMCASP_SOURCE_ROOT="$2"; CAMCASP_COMMIT="$3"; CAMCASP="$4"; '
        'materialize_camcasp_runtime'
    )
    result = subprocess.run(
        [
            "bash", "-c", command, "runtime-roots", str(reference),
            str(source), commit, str(runtime),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "must not contain one another" in result.stderr
    assert sentinel.is_file()
    assert hashlib.sha256(sentinel.read_bytes()).hexdigest() == sentinel_digest
    assert subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    ) == status_before
    if relationship == "runtime-inside-source":
        assert not runtime.exists()
    assert not (reference / "logs" / "camcasp-source.tar").exists()


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


def test_run_logged_checksums_retained_failure_log(tmp_path):
    log = tmp_path / "failed.log"
    command = (
        f'source "{SCRIPT}"; '
        'run_logged checksum-failure "$1" bash -c \'echo retained; exit 9\''
    )
    result = subprocess.run(
        ["bash", "-c", command, "checksum-test", str(log)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    digest = hashlib.sha256(log.read_bytes()).hexdigest()
    assert result.returncode == 9
    assert log.with_name("failed.log.sha256").read_text() == (
        f"{digest}  failed.log\n"
    )


def test_install_camcasp_program_replaces_stale_executable(tmp_path):
    camcasp = tmp_path / "camcasp-bin"
    archive_dir = camcasp / "x86-64" / "gfortran"
    target = archive_dir / "exe" / "camcasp"
    target.parent.mkdir(parents=True)
    (camcasp / "bin").mkdir()
    archive = archive_dir / "camcasp.gz"
    payload = b"#!/usr/bin/env bash\necho current\n"
    with gzip.open(archive, "wb") as handle:
        handle.write(payload)
    target.write_bytes(b"#!/usr/bin/env bash\necho stale\n")
    target.chmod(0o755)

    command = (
        f'source "{SCRIPT}"; '
        'CAMCASP="$1"; install_camcasp_program camcasp'
    )
    result = subprocess.run(
        ["bash", "-c", command, "stale-test", str(camcasp)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert target.read_bytes() == payload
    assert archive.read_bytes().startswith(b"\x1f\x8b")


def test_smoke_orient_failure_reports_build_guidance_and_checksum(tmp_path):
    orient = tmp_path / "orient"
    orient.write_text("#!/usr/bin/env bash\necho smoke-failed\nexit 7\n")
    orient.chmod(0o755)
    log = tmp_path / "orient-smoke.log"
    command = f'source "{SCRIPT}"; smoke_orient "$1" "$2"'
    result = subprocess.run(
        ["bash", "-c", command, "orient-test", str(orient), str(log)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "make OPENGL=no" in result.stderr
    assert str(log) in result.stderr
    assert log.with_name("orient-smoke.log.sha256").is_file()


def test_smoke_orient_fatal_log_reports_build_guidance(tmp_path):
    orient = tmp_path / "orient"
    orient.write_text("#!/usr/bin/env bash\necho 'ERROR STOP incompatible'\n")
    orient.chmod(0o755)
    log = tmp_path / "orient-smoke.log"
    command = f'source "{SCRIPT}"; smoke_orient "$1" "$2"'
    result = subprocess.run(
        ["bash", "-c", command, "orient-test", str(orient), str(log)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "make OPENGL=no" in result.stderr
    assert str(log) in result.stderr


def test_psi4_wrapper_smoke_and_forwards_serial_and_parallel(tmp_path):
    camcasp = tmp_path / "camcasp-bin"
    (camcasp / "bin").mkdir(parents=True)
    reference_root = tmp_path / "reference"
    fake_psi4 = tmp_path / "fake-psi4"
    arguments = tmp_path / "arguments.log"
    fake_psi4.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s' \"$1\" >>\"$FAKE_ARGS_LOG\"\n"
        "shift\n"
        "if (( $# )); then printf ' %s' \"$@\" >>\"$FAKE_ARGS_LOG\"; fi\n"
        "printf '\\n' >>\"$FAKE_ARGS_LOG\"\n"
        "echo fake-psi4\n"
    )
    fake_psi4.chmod(0o755)
    molecule = "molecule {\n  no_reorient\n  no_com\n  O 0 0 0\n}\n"
    (tmp_path / "serial.in").write_text(molecule)
    (tmp_path / "parallel.in").write_text(molecule)
    command = (
        f'source "{SCRIPT}"; '
        'CAMCASP="$1"; REFERENCE_ROOT="$2"; PSI4_EXE="$3"; '
        'write_psi4_wrapper; '
        '"$CAMCASP/bin/psi4.sh" serial.in serial.out; '
        '"$CAMCASP/bin/psi4.sh" parallel.in parallel.out 4'
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            command,
            "wrapper-test",
            str(camcasp),
            str(reference_root),
            str(fake_psi4),
        ],
        cwd=tmp_path,
        env={**os.environ, "FAKE_ARGS_LOG": str(arguments)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for name in ("serial.in", "parallel.in"):
        text = (tmp_path / name).read_text()
        assert text.count("symmetry c1") == 1
        assert "no_com" in text
        assert "no_reorient" in text
    assert arguments.read_text().splitlines() == [
        "--version",
        f"{(tmp_path / 'serial.in').resolve()} serial.out",
        f"-n 4 {(tmp_path / 'parallel.in').resolve()} parallel.out",
    ]
    version_log = reference_root / "logs" / "psi4-version.log"
    assert "fake-psi4" in version_log.read_text()
    assert version_log.with_name("psi4-version.log.sha256").is_file()


@pytest.mark.parametrize(
    "symmetry_directive", ("symmetry c1", "symmetry c1 # already canonical")
)
def test_psi4_wrapper_existing_c1_is_idempotent(tmp_path, symmetry_directive):
    camcasp = tmp_path / "camcasp"
    (camcasp / "bin").mkdir(parents=True)
    fake = tmp_path / "psi4"
    _write_executable(fake, "#!/usr/bin/env bash\nexit 0\n")
    input_file = tmp_path / "input.in"
    original = (
        f"molecule {{\n  {symmetry_directive}\n"
        "  no_reorient\n  no_com\n  O 0 0 0\n}\n"
    )
    input_file.write_text(original)
    command = (
        f'source "{SCRIPT}"; CAMCASP="$1"; REFERENCE_ROOT="$2"; '
        'PSI4_EXE="$3"; write_psi4_wrapper; '
        '"$CAMCASP/bin/psi4.sh" "$4" output.out; '
        '"$CAMCASP/bin/psi4.sh" "$4" output.out'
    )
    result = subprocess.run(
        ["bash", "-c", command, "wrapper-idempotent", str(camcasp),
         str(tmp_path / "reference"), str(fake), str(input_file)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert input_file.read_text() == original


def test_psi4_wrapper_invokes_resolved_symlink_target(tmp_path):
    camcasp = tmp_path / "camcasp"
    (camcasp / "bin").mkdir(parents=True)
    fake = tmp_path / "psi4"
    calls = tmp_path / "calls"
    _write_executable(
        fake,
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >>\"$WRAPPER_CALLS\"\n",
    )
    target = tmp_path / "target.in"
    target.write_text("molecule {\n  no_reorient\n  no_com\n  O 0 0 0\n}\n")
    symlink = tmp_path / "input-link.in"
    symlink.symlink_to(target)
    command = (
        f'source "{SCRIPT}"; CAMCASP="$1"; REFERENCE_ROOT="$2"; '
        'PSI4_EXE="$3"; write_psi4_wrapper; '
        '"$CAMCASP/bin/psi4.sh" "$4" linked.out 3'
    )
    result = subprocess.run(
        ["bash", "-c", command, "wrapper-symlink", str(camcasp),
         str(tmp_path / "reference"), str(fake), str(symlink)],
        cwd=ROOT,
        env={**os.environ, "WRAPPER_CALLS": str(calls)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert symlink.is_symlink()
    assert target.read_text().count("symmetry c1") == 1
    assert calls.read_text().splitlines() == [
        "--version",
        f"-n 3 {target.resolve()} linked.out",
    ]


def test_psi4_wrapper_writes_byte_identical_executed_input_evidence(tmp_path):
    reference = tmp_path / "reference"
    evidence_dir = reference / "work" / "H2O"
    evidence_dir.mkdir(parents=True)
    camcasp = tmp_path / "camcasp"
    (camcasp / "bin").mkdir(parents=True)
    fake = tmp_path / "psi4"
    received = tmp_path / "received.in"
    _write_executable(
        fake,
        "#!/usr/bin/env bash\n"
        "[[ \"${1:-}\" == --version ]] && exit 0\n"
        "[[ \"${1:-}\" == -n ]] && shift 2\n"
        "cp \"$1\" \"$WRAPPER_RECEIVED\"\n",
    )
    scratch = tmp_path / "scratch.in"
    scratch.write_bytes(b"molecule {\n  no_reorient\n  no_com\n  O 0 0 0\n}\n")
    command = (
        f'source "{SCRIPT}"; CAMCASP="$1"; REFERENCE_ROOT="$2"; '
        'PSI4_EXE="$3"; write_psi4_wrapper; '
        'CAMCASP_PSI4_EVIDENCE_DIR="$4" '
        '"$CAMCASP/bin/psi4.sh" "$5" evidence.out 2'
    )
    result = subprocess.run(
        ["bash", "-c", command, "wrapper-evidence", str(camcasp),
         str(reference), str(fake), str(evidence_dir), str(scratch)],
        cwd=ROOT,
        env={**os.environ, "WRAPPER_RECEIVED": str(received)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    evidence = evidence_dir / "H2O_A.executed.in"
    assert evidence.read_bytes() == scratch.read_bytes() == received.read_bytes()
    assert evidence.read_text().count("symmetry c1") == 1


@pytest.mark.parametrize("invalid_kind", ("empty", "relative", "outside", "missing"))
def test_psi4_wrapper_rejects_invalid_evidence_directory(tmp_path, invalid_kind):
    reference = tmp_path / "reference"
    (reference / "work").mkdir(parents=True)
    camcasp = tmp_path / "camcasp"
    (camcasp / "bin").mkdir(parents=True)
    calls = tmp_path / "calls"
    fake = tmp_path / "psi4"
    _write_executable(
        fake,
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >>\"$WRAPPER_CALLS\"\n",
    )
    scratch = tmp_path / "scratch.in"
    scratch.write_text("molecule {\n  no_reorient\n  no_com\n  O 0 0 0\n}\n")
    if invalid_kind == "empty":
        evidence_dir = ""
    elif invalid_kind == "relative":
        evidence_dir = "relative-evidence"
    elif invalid_kind == "outside":
        evidence_dir = str(tmp_path / "outside")
        Path(evidence_dir).mkdir()
    else:
        evidence_dir = str(reference / "work" / "missing")
    command = (
        f'source "{SCRIPT}"; CAMCASP="$1"; REFERENCE_ROOT="$2"; '
        'PSI4_EXE="$3"; write_psi4_wrapper; '
        'export CAMCASP_PSI4_EVIDENCE_DIR="$4"; '
        '"$CAMCASP/bin/psi4.sh" "$5" evidence.out'
    )
    result = subprocess.run(
        ["bash", "-c", command, "wrapper-invalid-evidence", str(camcasp),
         str(reference), str(fake), evidence_dir, str(scratch)],
        cwd=ROOT,
        env={**os.environ, "WRAPPER_CALLS": str(calls)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "evidence" in result.stderr.lower()
    assert calls.read_text().splitlines() == ["--version"]
    assert not list((reference / "work").rglob("H2O_A.executed.in"))


@pytest.mark.parametrize(
    "symmetry_lines",
    (
        "  symmetry c2v\n",
        "  symmetry c2v # conflicts with canonical protocol\n",
        "  symmetry c1\n  symmetry c1\n",
        "  symmetry c1 # first\n  symmetry c2v # second\n",
    ),
)
def test_psi4_wrapper_rejects_conflicting_or_duplicate_symmetry(
    tmp_path, symmetry_lines
):
    camcasp = tmp_path / "camcasp"
    (camcasp / "bin").mkdir(parents=True)
    fake = tmp_path / "psi4"
    calls = tmp_path / "calls"
    _write_executable(
        fake,
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >>\"$WRAPPER_CALLS\"\n",
    )
    input_file = tmp_path / "input.in"
    original = (
        "molecule {\n" + symmetry_lines
        + "  no_reorient\n  no_com\n  O 0 0 0\n}\n"
    )
    input_file.write_text(original)
    command = (
        f'source "{SCRIPT}"; CAMCASP="$1"; REFERENCE_ROOT="$2"; '
        'PSI4_EXE="$3"; write_psi4_wrapper; '
        '"$CAMCASP/bin/psi4.sh" "$4" output.out'
    )
    result = subprocess.run(
        ["bash", "-c", command, "wrapper-reject", str(camcasp),
         str(tmp_path / "reference"), str(fake), str(input_file)],
        cwd=ROOT,
        env={**os.environ, "WRAPPER_CALLS": str(calls)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "symmetry" in result.stderr.lower()
    assert input_file.read_text() == original
    assert calls.read_text().splitlines() == ["--version"]


def _make_git_checkout(
    path, tracked_candidate, candidate_text="#!/usr/bin/env bash\n"
):
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"], check=True
    )
    (path / "README").write_text("orient\n")
    (path / "VERSION").write_text("VERSION := 5.0\nPATCHLEVEL := 10\n")
    if tracked_candidate:
        candidate = path / "x86-64" / "gfortran" / "exe" / "orient-5.0.10-ng"
        candidate.parent.mkdir(parents=True)
        candidate.write_text(candidate_text)
        candidate.chmod(0o755)
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "fixture"], check=True)
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _make_pinned_orient_checkout(
    path, *, malformed_version=False, produce_candidate=True,
    produce_wrong_version=False,
):
    exe_dir = path / "x86-64" / "gfortran" / "exe"
    exe_dir.mkdir(parents=True)
    version_text = (
        "VERSION := malformed\nPATCHLEVEL := 10\n"
        if malformed_version
        else "VERSION := 5.0\nPATCHLEVEL := 10\n"
    )
    (path / "VERSION").write_text(version_text)
    (path / "README").write_text("pinned Orient source\n")
    (path / "bin").mkdir()
    old = exe_dir / "orient-5.0.09-ng"
    old.write_text("tracked old binary\n")
    (path / "bin" / "orient").symlink_to(old)
    (exe_dir / ".gitignore").write_text(
        "*.o\n*.mod\n*.cache\norient\norient-5.0.10-ng\n"
    )
    if produce_wrong_version:
        recipe = "\tcp /bin/true orient-5.0.99-ng\n"
    elif produce_candidate:
        recipe = "\tcp /bin/true $@\n"
    else:
        recipe = "\t@echo intentionally omitted product\n"
    (path / "Makefile").write_text(
        ".PHONY: FORCE\n"
        "orient-5.0.10-ng: FORCE\n"
        "\t@case \"$(MAKEFLAGS)\" in *jobserver*|*-j8*|*--jobs=8*) echo inherited parallel build; exit 91;; esac\n"
        "\t@case \" $(MAKEFLAGS) \" in *\" -j1 \"*) :;; *) echo serial -j1 missing; exit 92;; esac\n"
        "\t@test \"$(OPENGL)\" = no\n"
        f"\t@test \"$(BASE)\" = \"{path}\"\n"
        "\t@echo OPENGL=$(OPENGL)\n"
        "\t@echo BASE=$(BASE)\n"
        "\t@echo MAKEFLAGS=$(MAKEFLAGS)\n"
        + recipe
    )
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"], check=True
    )
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-qm", "pinned fixture"], check=True
    )
    commit = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    return commit, exe_dir / "orient-5.0.10-ng", old


def test_provision_orient_builds_derived_version_serially(tmp_path):
    checkout = tmp_path / "orient"
    commit, candidate, old = _make_pinned_orient_checkout(checkout)
    old_digest = hashlib.sha256(old.read_bytes()).hexdigest()
    reference = tmp_path / "reference"
    command = (
        f'source "{SCRIPT}"; '
        'REFERENCE_ROOT="$1"; ORIENT_REF="$2"; ORIENT_EXE="$3"; '
        'provision_orient'
    )
    result = subprocess.run(
        ["bash", "-c", command, "orient-build", str(reference), commit, str(candidate)],
        cwd=ROOT,
        env={**os.environ, "MAKEFLAGS": "-j8", "MFLAGS": "-j8"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert candidate.is_file() and os.access(candidate, os.X_OK)
    assert hashlib.sha256(old.read_bytes()).hexdigest() == old_digest
    subprocess.run(["git", "-C", str(checkout), "diff", "--exit-code"], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "diff", "--cached", "--exit-code"],
        check=True,
    )
    build_log = reference / "logs" / "orient-build.log"
    build_text = build_log.read_text()
    assert "OPENGL=no" in build_text
    assert f"BASE={checkout}" in build_text
    assert "-j1" in build_text
    assert "-j8" not in build_text
    assert "jobserver" not in build_text
    for log_name in (
        "orient-build.log", "orient-ldd.log", "orient-smoke.log",
        "orient-executable.log",
    ):
        assert (reference / "logs" / f"{log_name}.sha256").is_file()
    assert (reference / "logs" / "orient-executable.sha256").read_text().startswith(
        hashlib.sha256(candidate.read_bytes()).hexdigest()
    )
    assert not list((reference / "logs").glob(".orient-products.*"))


def test_preflight_rejects_malformed_orient_version(tmp_path):
    checkout = tmp_path / "orient"
    commit, candidate, _ = _make_pinned_orient_checkout(
        checkout, malformed_version=True
    )
    psi4_source = tmp_path / "psi4-source"
    psi4, _ = _make_psi4_checkout(psi4_source)
    command = (
        f'source "{SCRIPT}"; '
        'PSI4_SOURCE_ROOT="$1"; PSI4_EXE="$2"; '
        'ORIENT_REF="$3"; ORIENT_EXE="$4"; preflight'
    )
    result = subprocess.run(
        [
            "bash", "-c", command, "orient-version", str(psi4_source),
            str(psi4), commit, str(candidate),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "malformed Orient VERSION" in result.stderr


def test_provision_orient_rejects_missing_build_product(tmp_path):
    checkout = tmp_path / "orient"
    commit, candidate, _ = _make_pinned_orient_checkout(
        checkout, produce_candidate=False
    )
    reference = tmp_path / "reference"
    command = (
        f'source "{SCRIPT}"; '
        'REFERENCE_ROOT="$1"; ORIENT_REF="$2"; ORIENT_EXE="$3"; '
        'provision_orient'
    )
    result = subprocess.run(
        ["bash", "-c", command, "orient-product", str(reference), commit, str(candidate)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "missing built Orient artifact" in result.stderr
    assert (reference / "logs" / "orient-build.log.sha256").is_file()
    assert not list((reference / "logs").glob(".orient-products.*"))


def test_provision_orient_rejects_unexpected_ignored_product(tmp_path):
    checkout = tmp_path / "orient"
    commit, candidate, _ = _make_pinned_orient_checkout(checkout)
    unexpected = candidate.parent / "stale.cache"
    unexpected.write_text("ignored stale product\n")
    assert subprocess.run(
        ["git", "-C", str(checkout), "check-ignore", "-q", str(unexpected)],
        check=False,
    ).returncode == 0
    reference = tmp_path / "reference"
    command = (
        f'source "{SCRIPT}"; '
        'REFERENCE_ROOT="$1"; ORIENT_REF="$2"; ORIENT_EXE="$3"; '
        'provision_orient'
    )
    result = subprocess.run(
        ["bash", "-c", command, "orient-ignored", str(reference), commit, str(candidate)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "unexpected untracked/ignored Orient product" in result.stderr
    assert "stale.cache" in result.stderr
    assert not (reference / "logs" / "orient-build.log").exists()
    assert not list((reference / "logs").glob(".orient-products.*"))


@pytest.mark.parametrize("inventory_kind", ("ordinary", "ignored"))
def test_provision_orient_inventory_failure_is_fail_closed(
    tmp_path, inventory_kind
):
    checkout = tmp_path / f"orient-{inventory_kind}"
    commit, candidate, _ = _make_pinned_orient_checkout(checkout)
    candidate.write_bytes(Path("/bin/true").read_bytes())
    candidate.chmod(0o755)
    original_digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    reference = tmp_path / f"reference-{inventory_kind}"
    fake_bin = tmp_path / f"fake-bin-{inventory_kind}"
    fake_bin.mkdir()
    real_git = subprocess.check_output(["bash", "-c", "command -v git"], text=True).strip()
    git_wrapper = fake_bin / "git"
    git_wrapper.write_text(
        "#!/usr/bin/env bash\n"
        "args=\" $* \"\n"
        "if [[ \"$FAIL_INVENTORY\" == ordinary && \"$args\" == *\" ls-files --others --exclude-standard -z \"* ]]; then exit 86; fi\n"
        "if [[ \"$FAIL_INVENTORY\" == ignored && \"$args\" == *\" ls-files --others --ignored --exclude-standard -z \"* ]]; then exit 87; fi\n"
        f'exec "{real_git}" "$@"\n'
    )
    git_wrapper.chmod(0o755)
    command = (
        f'source "{SCRIPT}"; '
        'REFERENCE_ROOT="$1"; ORIENT_REF="$2"; ORIENT_EXE="$3"; '
        'provision_orient'
    )
    result = subprocess.run(
        ["bash", "-c", command, "orient-inventory", str(reference), commit, str(candidate)],
        cwd=ROOT,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "FAIL_INVENTORY": inventory_kind,
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert f"Orient {inventory_kind} product inventory failed" in result.stderr
    assert candidate.exists()
    assert hashlib.sha256(candidate.read_bytes()).hexdigest() == original_digest
    assert not (reference / "logs" / "orient-build.log").exists()
    assert not list((reference / "logs").glob(".orient-products.*"))


def test_provision_orient_rejects_wrong_version_build_product(tmp_path):
    checkout = tmp_path / "orient"
    commit, candidate, _ = _make_pinned_orient_checkout(
        checkout, produce_candidate=False, produce_wrong_version=True
    )
    reference = tmp_path / "reference"
    command = (
        f'source "{SCRIPT}"; '
        'REFERENCE_ROOT="$1"; ORIENT_REF="$2"; ORIENT_EXE="$3"; '
        'provision_orient'
    )
    result = subprocess.run(
        ["bash", "-c", command, "orient-wrong-version", str(reference), commit, str(candidate)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "unexpected untracked/ignored Orient product" in result.stderr
    assert "orient-5.0.99-ng" in result.stderr
    assert not candidate.exists()
    assert (reference / "logs" / "orient-build.log.sha256").is_file()
    assert not list((reference / "logs").glob(".orient-products.*"))


def _make_psi4_checkout(path):
    psi4 = path / "build_camcasp" / "stage" / "bin" / "psi4"
    psi4.parent.mkdir(parents=True)
    psi4.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "echo psi4 >>\"$STUB_CALLS\"\n"
        "if [[ \"${1:-}\" == --version ]]; then echo 'stub Psi4 1.0'; exit 0; fi\n"
        "if [[ \"${1:-}\" == -n ]]; then shift 2; fi\n"
        "input=\"$1\"; output=\"$2\"\n"
        "if [[ -n \"${STUB_PSI4_RECEIVED_INPUT:-}\" ]]; then\n"
        "  cp \"$input\" \"$STUB_PSI4_RECEIVED_INPUT\"\n"
        "  realpath -e -- \"$input\" >\"$STUB_PSI4_RECEIVED_PATH\"\n"
        "fi\n"
        "[[ \"$(grep -Eic '^[[:space:]]*symmetry[[:space:]]+c1[[:space:]]*$' \"$input\")\" == 1 ]]\n"
        "grep -Eiq '^[[:space:]]*basis[[:space:]]+aug-cc-pvtz[[:space:]]*$' \"$input\"\n"
        "grep -Fq \"energy('PBE0'\" \"$input\"\n"
        "printf 'Running in c1 symmetry.\\n=> Composite Functional: PBE0 <=\\n' >\"$output\"\n"
    )
    psi4.chmod(0o755)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"], check=True
    )
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-qm", "stub Psi4"], check=True
    )
    commit = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    return psi4, commit


def test_record_orient_executable_writes_observed_digest(tmp_path):
    orient = tmp_path / "orient"
    orient.write_bytes(b"observed orient bytes\n")
    orient.chmod(0o755)
    reference_root = tmp_path / "reference"
    command = (
        f'source "{SCRIPT}"; '
        'REFERENCE_ROOT="$1"; record_orient_executable "$2"'
    )
    result = subprocess.run(
        ["bash", "-c", command, "orient-record", str(reference_root), str(orient)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    logs = reference_root / "logs"
    digest = hashlib.sha256(orient.read_bytes()).hexdigest()
    assert result.returncode == 0, result.stderr
    assert (logs / "orient-executable.log").read_text() == (
        f"selected executable: {orient}\n"
    )
    assert (logs / "orient-executable.sha256").read_text() == (
        f"{digest}  orient\n"
    )
    assert (logs / "orient-executable.log.sha256").is_file()


def test_verify_orient_checkout_rejects_dirty_checkout(tmp_path):
    checkout = tmp_path / "orient"
    commit = _make_git_checkout(checkout, tracked_candidate=True)
    (checkout / "README").write_text("dirty\n")
    candidate = checkout / "x86-64" / "gfortran" / "exe" / "orient-5.0.10-ng"
    command = (
        f'source "{SCRIPT}"; '
        'ORIENT_REF="$1"; verify_orient_checkout "$2" "$3"'
    )
    result = subprocess.run(
        ["bash", "-c", command, "orient-git", commit, str(checkout), str(candidate)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Orient tracked source checkout is not clean" in result.stderr


def test_verify_orient_checkout_rejects_unrelated_untracked_file(tmp_path):
    checkout = tmp_path / "orient"
    commit = _make_git_checkout(checkout, tracked_candidate=True)
    (checkout / "unrelated.tmp").write_text("untracked\n")
    candidate = checkout / "x86-64" / "gfortran" / "exe" / "orient-5.0.10-ng"
    command = (
        f'source "{SCRIPT}"; '
        'ORIENT_REF="$1"; verify_orient_checkout "$2" "$3"'
    )
    result = subprocess.run(
        ["bash", "-c", command, "orient-git", commit, str(checkout), str(candidate)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "unexpected untracked/ignored Orient product" in result.stderr


def test_verify_orient_checkout_rejects_older_tracked_binary(tmp_path):
    checkout = tmp_path / "orient"
    commit, _, candidate = _make_pinned_orient_checkout(checkout)
    command = (
        f'source "{SCRIPT}"; '
        'ORIENT_REF="$1"; verify_orient_checkout "$2" "$3"'
    )
    result = subprocess.run(
        ["bash", "-c", command, "orient-git", commit, str(checkout), str(candidate)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "unexpected Orient artifact" in result.stderr


def test_verify_orient_checkout_rejects_wrong_location(tmp_path):
    checkout = tmp_path / "orient"
    commit = _make_git_checkout(checkout, tracked_candidate=True)
    candidate = checkout / "README"
    command = (
        f'source "{SCRIPT}"; '
        'ORIENT_REF="$1"; verify_orient_checkout "$2" "$3"'
    )
    result = subprocess.run(
        ["bash", "-c", command, "orient-location", commit, str(checkout), str(candidate)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "unexpected Orient artifact" in result.stderr


def test_verify_orient_checkout_rejects_mismatched_commit(tmp_path):
    checkout = tmp_path / "orient"
    _make_git_checkout(checkout, tracked_candidate=True)
    candidate = checkout / "x86-64" / "gfortran" / "exe" / "orient-5.0.10-ng"
    command = (
        f'source "{SCRIPT}"; '
        'ORIENT_REF="$1"; verify_orient_checkout "$2" "$3"'
    )
    result = subprocess.run(
        ["bash", "-c", command, "orient-commit", "0" * 40, str(checkout), str(candidate)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Orient checkout is not pinned" in result.stderr


def test_provision_orient_rejects_override_outside_git(tmp_path):
    candidate = tmp_path / "orient-5.0.10-ng"
    candidate.write_text("#!/usr/bin/env bash\n")
    candidate.chmod(0o755)
    command = (
        f'source "{SCRIPT}"; '
        'REFERENCE_ROOT="$1"; ORIENT_EXE="$2"; provision_orient'
    )
    result = subprocess.run(
        ["bash", "-c", command, "orient-override", str(tmp_path / "ref"), str(candidate)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Orient override must be inside a Git checkout" in result.stderr


def test_verify_psi4_source_rejects_wrong_executable_location(tmp_path):
    source = tmp_path / "psi4-source"
    expected, _ = _make_psi4_checkout(source)
    candidate = source / "psi4"
    candidate.write_bytes(expected.read_bytes())
    candidate.chmod(0o755)
    command = f'source "{SCRIPT}"; verify_psi4_source_root "$1" "$2"'
    result = subprocess.run(
        ["bash", "-c", command, "psi4-location", str(source), str(candidate)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "unexpected Psi4 executable" in result.stderr


def test_verify_psi4_source_rejects_dirty_checkout(tmp_path):
    source = tmp_path / "psi4-source"
    candidate, _ = _make_psi4_checkout(source)
    (source / "dirty.txt").write_text("untracked\n")
    command = f'source "{SCRIPT}"; verify_psi4_source_root "$1" "$2"'
    result = subprocess.run(
        ["bash", "-c", command, "psi4-dirty", str(source), str(candidate)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Psi4 source checkout is not clean" in result.stderr


import math
import sys

sys.path.insert(0, str(ROOT))
from devtools.camcasp_reference import (  # noqa: E402
    COMPONENTS_L3,
    ReferenceFormatError,
    parse_frequencies,
    parse_refined_polarizabilities,
)


def test_pure_tooling_suite_does_not_import_psi4():
    assert "psi4" not in sys.modules


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
    """Legacy synthetic 16x16 fixture retained for parser compatibility tests."""
    blocks = []
    for frequency_index in range(11):
        blocks.append(f"# INDEX {frequency_index:03d}")
        for atom_index, label in enumerate(("O", "H1", "H2")):
            blocks.append(f"{label} {label}")
            for row in range(16):
                values = [
                    frequency_index + atom_index + (row + column) / 100.0
                    for column in range(16)
                ]
                blocks.append(" ".join(f"{value:.8f}" for value in values))
    return "\n".join(blocks) + "\n"


def make_real_l3_refined_text():
    lines = [
        "#  Localisation settings for H2O",
        "#  Pol file format: NEW",
        "# ",
        "",
    ]
    for frequency_index in range(11):
        lines.append(f"# INDEX {frequency_index:03d}")
        for atom_index, label in enumerate(("O", "H1", "H2")):
            lines.append(
                f"ALPHA  H2O  SITE-NAMES  {label}  {label}  "
                f"RANK 1 TO 3 INDEX   0 FREQSQ       {frequency_index / 100:.7f}"
            )
            for row in range(15):
                values = [
                    frequency_index + atom_index + (row + column) / 100.0
                    for column in range(15)
                ]
                lines.append(" ".join(f"{value:.12f}" for value in values))
        lines.extend((" ENDFILE", ""))
    return "\n".join(lines) + "\n"


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


def test_parse_authoritative_real_l3_model_and_dipole_mapping(tmp_path):
    source = tmp_path / "H2O_ref_wt4_L3_0f10.pol"
    source.write_text(make_real_l3_refined_text())
    blocks = parse_refined_polarizabilities(source, ("O", "H1", "H2"), limit=3)
    assert len(blocks) == 11
    assert tuple(blocks[0].atoms) == ("O", "H1", "H2")
    assert blocks[0].atoms["O"].components == COMPONENTS_L3[1:]
    assert len(blocks[10].atoms["H2"].matrix) == 15
    assert all(len(row) == 15 for row in blocks[10].atoms["H2"].matrix)
    assert dipole_local_cartesian(blocks[0].atoms["O"]) == (
        (0.02, 0.03, 0.01),
        (0.03, 0.04, 0.02),
        (0.01, 0.02, 0.0),
    )


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    (
        ("ALPHA  H2O", "ALPHA  NH3", "job H2O"),
        ("RANK 1 TO 3", "RANK 0 TO 3", "RANK 1 TO 3"),
        ("INDEX   0", "INDEX   1", "header INDEX 0"),
        ("SITE-NAMES  O  O", "SITE-NAMES  O  H1", "SITE-NAMES O O"),
        ("FREQSQ       0.0000000", "FREQSQ       -0.1000000", "nonnegative"),
        ("FREQSQ       0.0000000", "FREQSQ       nan", "header"),
        ("FREQSQ       0.0000000", "FREQSQ       0_0", "header"),
    ),
)
def test_real_l3_rejects_invalid_header_contract(tmp_path, old, new, expected):
    source = tmp_path / "invalid-real-header.pol"
    source.write_text(make_real_l3_refined_text().replace(old, new, 1))
    with pytest.raises(ReferenceFormatError, match=expected):
        parse_refined_polarizabilities(source, ("O", "H1", "H2"), limit=3)


def test_real_l3_requires_matching_freqsq_across_atom_headers(tmp_path):
    source = tmp_path / "invalid-real-freqsq.pol"
    text = make_real_l3_refined_text()
    h1 = "ALPHA  H2O  SITE-NAMES  H1  H1  RANK 1 TO 3 INDEX   0 FREQSQ       0.0000000"
    source.write_text(text.replace(h1, h1.replace("0.0000000", "0.1000000"), 1))
    with pytest.raises(ReferenceFormatError, match="FREQSQ does not match"):
        parse_refined_polarizabilities(source, ("O", "H1", "H2"), limit=3)


@pytest.mark.parametrize("mode", ("missing", "duplicate", "reordered", "mixed"))
def test_real_l3_requires_exact_atom_headers_in_order(tmp_path, mode):
    source = tmp_path / "invalid-real-atoms.pol"
    text = make_real_l3_refined_text()
    h1 = "ALPHA  H2O  SITE-NAMES  H1  H1  RANK 1 TO 3 INDEX   0 FREQSQ       0.0000000"
    if mode == "missing":
        text = text.replace(h1 + "\n", "", 1)
    elif mode == "duplicate":
        text = text.replace(h1, h1.replace("H1  H1", "O  O"), 1)
    elif mode == "reordered":
        text = text.replace(h1, h1.replace("H1  H1", "H2  H2"), 1)
    else:
        text = text.replace(h1, "H1 H1", 1)
    source.write_text(text)
    with pytest.raises(ReferenceFormatError, match="expected ALPHA header for H1"):
        parse_refined_polarizabilities(source, ("O", "H1", "H2"), limit=3)


@pytest.mark.parametrize("mode", ("incomplete", "extra", "malformed", "nonfinite"))
def test_real_l3_requires_exact_finite_15_by_15_matrix(tmp_path, mode):
    source = tmp_path / "invalid-real-matrix.pol"
    text = make_real_l3_refined_text()
    row = " ".join(f"{column / 100:.12f}" for column in range(15))
    fields = row.split()
    if mode == "incomplete":
        replacement = " ".join(fields[:-1])
    elif mode == "extra":
        replacement = row + " 1.0"
    elif mode == "malformed":
        replacement = row.replace(fields[3], "not-a-number", 1)
    else:
        replacement = row.replace(fields[3], "nan", 1)
    source.write_text(text.replace(row, replacement, 1))
    with pytest.raises(ReferenceFormatError, match="row 0"):
        parse_refined_polarizabilities(source, ("O", "H1", "H2"), limit=3)


@pytest.mark.parametrize("mode", ("missing", "early", "trailing", "comment"))
def test_real_l3_requires_strict_endfile_and_block_boundaries(tmp_path, mode):
    source = tmp_path / "invalid-real-endfile.pol"
    text = make_real_l3_refined_text()
    if mode == "missing":
        text = text.replace(" ENDFILE\n", "", 1)
    elif mode == "early":
        first_row = " ".join(f"{column / 100:.12f}" for column in range(15))
        text = text.replace(first_row + "\n", " ENDFILE\n" + first_row + "\n", 1)
    elif mode == "trailing":
        text = text.replace(" ENDFILE\n", " ENDFILE\nunexpected payload\n", 1)
    else:
        text = text.replace("\n# INDEX 001", "\n# unexpected inter-block comment\n# INDEX 001", 1)
    source.write_text(text)
    with pytest.raises(ReferenceFormatError, match="ENDFILE|unexpected|row"):
        parse_refined_polarizabilities(source, ("O", "H1", "H2"), limit=3)


@pytest.mark.parametrize("mode", ("missing", "duplicate", "wrong"))
def test_real_l3_requires_exact_index_sequence(tmp_path, mode):
    source = tmp_path / "invalid-real-index.pol"
    text = make_real_l3_refined_text()
    if mode == "missing":
        text = text.replace("# INDEX 001\n", "", 1)
    elif mode == "duplicate":
        text = text.replace("# INDEX 001", "# INDEX 000", 1)
    else:
        text = text.replace("# INDEX 001", "# INDEX 009", 1)
    source.write_text(text)
    with pytest.raises(ReferenceFormatError, match="frequency index 001"):
        parse_refined_polarizabilities(source, ("O", "H1", "H2"), limit=3)


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


def assert_rejects_atom_labels(tmp_path, atom_labels):
    source = tmp_path / "invalid-atom-labels.pol"
    source.write_text(make_l3_refined_text())
    try:
        parse_refined_polarizabilities(source, atom_labels, limit=3)
    except ReferenceFormatError as exc:
        assert "accepted model requires atom labels ('O', 'H1', 'H2')" in str(exc)
    else:
        raise AssertionError(f"invalid atom labels were accepted: {atom_labels!r}")


def test_rejects_reordered_caller_atom_labels(tmp_path):
    assert_rejects_atom_labels(tmp_path, ("H1", "O", "H2"))


def test_rejects_missing_caller_atom_label(tmp_path):
    assert_rejects_atom_labels(tmp_path, ("O", "H1"))


def test_rejects_duplicate_caller_atom_labels(tmp_path):
    assert_rejects_atom_labels(tmp_path, ("O", "H1", "H1"))


def test_rejects_unrelated_caller_atom_label(tmp_path):
    assert_rejects_atom_labels(tmp_path, ("O", "H1", "X"))


def test_rejects_extra_atom_block(tmp_path):
    source = tmp_path / "extra-atom.pol"
    lines = make_l3_refined_text().splitlines()
    next_index = lines.index("# INDEX 001")
    lines[next_index:next_index] = [
        "X X",
        *(" ".join(["0.0"] * 16) for _ in range(16)),
    ]
    source.write_text("\n".join(lines) + "\n")
    try:
        parse_refined_polarizabilities(source, ("O", "H1", "H2"), limit=3)
    except ReferenceFormatError as exc:
        assert "frequency 000 unexpected content after atom H2" in str(exc)
    else:
        raise AssertionError("extra atom block was accepted")


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


def _dipole_model_with_diagonal_value(value):
    matrix = [[0.0] * 16 for _ in range(16)]
    index = COMPONENTS_L3.index("11c")
    matrix[index][index] = value
    return type("Model", (), {
        "components": COMPONENTS_L3,
        "matrix": tuple(tuple(row) for row in matrix),
    })()


def test_rejects_nan_local_dipole_value():
    try:
        dipole_local_cartesian(_dipole_model_with_diagonal_value(math.nan))
    except ReferenceFormatError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("NaN local dipole value was accepted")


def test_rejects_infinite_local_dipole_value():
    try:
        dipole_local_cartesian(_dipole_model_with_diagonal_value(math.inf))
    except ReferenceFormatError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("infinite local dipole value was accepted")


def test_rejects_nan_frame_entry():
    try:
        validate_rotation_matrix(
            ((math.nan, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        )
    except ReferenceFormatError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("NaN frame entry was accepted")


def test_rejects_infinite_frame_entry():
    try:
        validate_rotation_matrix(
            ((math.inf, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        )
    except ReferenceFormatError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("infinite frame entry was accepted")


def test_rejects_missing_hydrogen_axis_rule():
    axes = CANONICAL_AXES.replace("  H1  z global Z x from H2 to H1\n", "")
    try:
        build_local_frames(CANONICAL_GEOMETRY, axes)
    except ReferenceFormatError as exc:
        assert "missing axis rules for H1" in str(exc)
    else:
        raise AssertionError("missing H1 axis rule was accepted")


def test_rejects_malformed_hydrogen_axis_rule():
    axes = CANONICAL_AXES.replace("H1  z global Z", "H1  z global Y")
    try:
        build_local_frames(CANONICAL_GEOMETRY, axes)
    except ReferenceFormatError as exc:
        assert "invalid axes line" in str(exc)
    else:
        raise AssertionError("malformed H1 axis rule was accepted")


def test_rejects_duplicate_hydrogen_axis_rule():
    rule = "  H1  z global Z x from H2 to H1\n"
    axes = CANONICAL_AXES.replace(rule, rule + rule)
    try:
        build_local_frames(CANONICAL_GEOMETRY, axes)
    except ReferenceFormatError as exc:
        assert "duplicate axis rule for H1" in str(exc)
    else:
        raise AssertionError("duplicate H1 axis rule was accepted")


def test_rejects_zero_projection_hydrogen_axis_rule():
    axes = CANONICAL_AXES.replace("from H2 to H1", "from H1 to H1")
    try:
        build_local_frames(CANONICAL_GEOMETRY, axes)
    except ReferenceFormatError as exc:
        assert "axis direction has zero length" in str(exc)
    else:
        raise AssertionError("zero-projection H1 axis rule was accepted")


def test_rejects_additional_malformed_hydrogen_axis_rule():
    axes = CANONICAL_AXES.replace(
        "End\n",
        "  H1  z global Y x from H2 to H1\nEnd\n",
    )
    try:
        build_local_frames(CANONICAL_GEOMETRY, axes)
    except ReferenceFormatError as exc:
        assert "invalid axes line" in str(exc)
    else:
        raise AssertionError("additional malformed H1 axis rule was accepted")


def test_allows_hash_rule_shaped_comment():
    axes = CANONICAL_AXES.replace(
        "End\n",
        "  # z global Z x from H2 to H1\nEnd\n",
    )
    frames = build_local_frames(CANONICAL_GEOMETRY, axes)
    assert frames["H1"] == (
        (-1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
    )


def test_allows_bang_rule_shaped_comment():
    axes = CANONICAL_AXES.replace(
        "End\n",
        "  ! z global Z x from H1 to H2\nEnd\n",
    )
    frames = build_local_frames(CANONICAL_GEOMETRY, axes)
    assert frames["H2"] == (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )


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


def test_rejects_isotropic_cn_final_block_without_end(tmp_path):
    source = tmp_path / "missing-final-end.pot"
    source.write_text(CASIMIR_C12.rsplit("  End\n", 1)[0] + "\n")
    try:
        parse_isotropic_cn(
            source,
            ("O", "H1", "H2"),
            {"O": "O", "H1": "H", "H2": "H"},
        )
    except ReferenceFormatError as exc:
        assert str(source) in str(exc)
        assert "End" in str(exc)
        assert "('H', 'H')" in str(exc)
    else:
        raise AssertionError("final CASIMIR block without End was accepted")


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


def assert_rejects_isotropic_cn(tmp_path, name, text, expected):
    source = tmp_path / name
    source.write_text(text)
    try:
        parse_isotropic_cn(
            source,
            ("O", "H1", "H2"),
            {"O": "O", "H1": "H", "H2": "H"},
        )
    except ReferenceFormatError as exc:
        assert str(source) in str(exc)
        assert expected in str(exc)
    else:
        raise AssertionError(f"invalid isotropic Cn input was accepted: {name}")


def test_rejects_isotropic_cn_missing_type_pair(tmp_path):
    missing_hh = CASIMIR_C12.replace(
        "  H  H      C6 C7 C8 C9 C10 C11 C12\n"
        "    00 00 0 1.0 0.0 10.0 0.0 100.0 0.0 1000.0\n"
        "  End\n",
        "",
    )
    assert_rejects_isotropic_cn(
        tmp_path,
        "missing-HH.pot",
        missing_hh,
        "missing atom-type pairs [('H', 'H')]",
    )


def test_rejects_isotropic_cn_reversed_duplicate_pair(tmp_path):
    reversed_oh = CASIMIR_C12 + (
        "  O  H      C6 C7 C8 C9 C10 C11 C12\n"
        "    00 00 0 9.0 0.0 90.0 0.0 900.0 0.0 9000.0\n"
        "  End\n"
    )
    assert_rejects_isotropic_cn(
        tmp_path,
        "duplicate-OH.pot",
        reversed_oh,
        "duplicate pair block ('H', 'O')",
    )


def test_rejects_isotropic_cn_missing_isotropic_row(tmp_path):
    missing_row = CASIMIR_C12.replace(
        "    00 00 0 20.0 0.0 200.0 0.0 2000.0 0.0 20000.0",
        "    10 00 0 20.0 0.0 200.0 0.0 2000.0 0.0 20000.0",
        1,
    )
    assert_rejects_isotropic_cn(
        tmp_path,
        "missing-isotropic.pot",
        missing_row,
        "missing 00 00 0 row for ('O', 'O')",
    )


def test_rejects_isotropic_cn_duplicate_isotropic_row(tmp_path):
    row = "    00 00 0 20.0 0.0 200.0 0.0 2000.0 0.0 20000.0\n"
    duplicate_row = CASIMIR_C12.replace(row, row + row, 1)
    assert_rejects_isotropic_cn(
        tmp_path,
        "duplicate-isotropic.pot",
        duplicate_row,
        "duplicate 00 00 0 row for ('O', 'O')",
    )


def test_rejects_isotropic_cn_nonfinite_values(tmp_path):
    for value in ("nan", "inf"):
        nonfinite = CASIMIR_C12.replace("20.0", value, 1)
        assert_rejects_isotropic_cn(
            tmp_path,
            f"nonfinite-{value}.pot",
            nonfinite,
            "('O', 'O') C6: non-finite value",
        )


def test_rejects_isotropic_cn_missing_required_header_columns(tmp_path):
    for order in ("C8", "C10", "C6"):
        incomplete = CASIMIR_C12.replace(
            "C6 C7 C8 C9 C10 C11 C12",
            "C6 C7 C8 C9 C10 C11 C12".replace(f"{order} ", "", 1),
            1,
        )
        assert_rejects_isotropic_cn(
            tmp_path,
            f"missing-{order}.pot",
            incomplete,
            f"missing required {order} column",
        )


def test_rejects_isotropic_cn_truncated_numeric_row(tmp_path):
    truncated = CASIMIR_C12.replace(" 0.0 20000.0", "", 1)
    assert_rejects_isotropic_cn(
        tmp_path,
        "truncated-isotropic.pot",
        truncated,
        "('O', 'O') 00 00 0 row requires 7 numeric values, found 5",
    )


import json

from devtools.camcasp_reference import (  # noqa: E402
    render_python_literals,
    validate_reference_document,
    write_atomic_json,
)


APPROVED_FREQUENCIES = (
    0.0,
    0.0066096015960872435,
    0.036174811998630957,
    0.095447363690348272,
    0.1976442118453127,
    0.3704172128053672,
    0.6749146404580301,
    1.264899172436498,
    2.619244684547324,
    6.910885950408292,
    37.82376235021415,
)
APPROVED_SQUARED_SOURCE_VALUES = (
    "0.0",
    "-4.3686833258999033E-005",
    "-1.3086170231362943E-003",
    "-9.1101992354376132E-003",
    "-3.9063234475954833E-002",
    "-0.1372089115424966",
    "-0.4555097719045920",
    "-1.599969916430539",
    "-6.860442717529413",
    "-47.76034461955072",
    "-1430.636998325477",
)
CAMCASP_EXECUTABLE_NAMES = ("camcasp", "cluster", "process", "pfit", "casimir")


def _executable_record(name, digest_character):
    return {"path": f"/opt/{name}", "sha256": digest_character * 64}


def complete_document():
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    tensor = [[1.0, 0.0, 0.0], [0.0, 1.1, 0.0], [0.0, 0.0, 1.2]]
    atom = {
        "spherical": {
            "components": list(COMPONENTS_L3),
            "matrix": [
                [float(row == column) for column in range(16)]
                for row in range(16)
            ],
        },
        "local_cartesian": tensor,
        "local_to_global": identity,
        "global_cartesian": tensor,
    }
    frequency_blocks = [
        {
            "index": index,
            "omega": omega,
            "atoms": {"O": atom, "H1": atom, "H2": atom},
        }
        for index, omega in enumerate(APPROVED_FREQUENCIES)
    ]
    matrix = [[1.0, 0.5, 0.5], [0.5, 0.2, 0.2], [0.5, 0.2, 0.2]]
    return {
        "schema_version": 1,
        "generated_at_utc": "2026-07-29T12:00:00Z",
        "generator": {
            "path": "devtools/regenerate-camcasp.sh",
            "sha256": "a" * 64,
        },
        "repository": {"commit": "b" * 40, "dirty": False},
        "tools": {
            "camcasp": {
                "version": "7.2.2 patch 003",
                "commit": "b4744425233a61786052832e1db4f109959c1ce9",
                "executables": {
                    name: _executable_record(name, str(index + 1))
                    for index, name in enumerate(CAMCASP_EXECUTABLE_NAMES)
                },
            },
            "orient": {
                "version": "5.0.10-ng",
                "commit": "d8d861098c8f548e2cf230c387c8431d9418650a",
                "executable": _executable_record("orient", "6"),
            },
            "psi4": {
                "version": "1.11a1.dev31",
                "commit": "c" * 40,
                "dirty": True,
                "executable": _executable_record("psi4", "7"),
            },
        },
        "scientific_protocol": {
            "geometry": {
                "units": "bohr",
                "charge": 0,
                "multiplicity": 1,
                "atom_order": ["O", "H1", "H2"],
                "atoms": [
                    {
                        "label": label,
                        "element": "O" if label == "O" else "H",
                        "xyz": list(CANONICAL_GEOMETRY[label]),
                    }
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
            "frequency_grid": {
                "kind": "Gauss-Legendre",
                "nonzero_count": 10,
                "scale_au": 0.5,
            },
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
            "values": list(APPROVED_FREQUENCIES),
            "squared_source_values": list(APPROVED_SQUARED_SOURCE_VALUES),
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
            "matrices": {
                "C6": matrix,
                "C8": matrix,
                "C10": matrix,
                "C12": matrix,
            },
        },
        "inputs": {
            "H2O.clt": {"path": "/inputs/H2O.clt", "sha256": "d" * 64},
            "H2O.axes": {"path": "/inputs/H2O.axes", "sha256": "e" * 64},
        },
        "sources": {
            "refined_pol": {"path": "/work/refined.pol", "sha256": "f" * 64}
        },
    }


def test_validate_complete_schema():
    validate_reference_document(complete_document())


def test_validator_rejects_each_required_top_level_field():
    for key in (
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


def _assert_document_rejected(document, expected):
    try:
        validate_reference_document(document)
    except ReferenceFormatError as exc:
        assert expected in str(exc)
    else:
        raise AssertionError(f"invalid document was accepted: {expected}")


def test_validator_rejects_nonfinite_numbers_recursively():
    mutations = (
        (
            lambda document: document["scientific_protocol"]["geometry"]["atoms"][
                0
            ]["xyz"].__setitem__(0, math.nan),
            "non-finite",
        ),
        (
            lambda document: document["polarizabilities"]["frequency_blocks"][0][
                "atoms"
            ]["O"]["spherical"]["matrix"][0].__setitem__(0, math.inf),
            "non-finite",
        ),
        (
            lambda document: document["dispersion"]["matrices"]["C12"][
                0
            ].__setitem__(0, -math.inf),
            "non-finite",
        ),
    )
    for mutate, expected in mutations:
        document = complete_document()
        mutate(document)
        _assert_document_rejected(document, expected)


def test_validator_rejects_noncanonical_protocol_and_atom_invariants():
    document = complete_document()
    document["scientific_protocol"]["frequency_grid"]["scale_au"] = 0.6
    _assert_document_rejected(document, "frequency_grid.scale_au")

    document = complete_document()
    document["scientific_protocol"]["geometry"]["atoms"][1]["element"] = "O"
    _assert_document_rejected(document, "geometry.atoms")

    document = complete_document()
    atoms = document["polarizabilities"]["frequency_blocks"][0]["atoms"]
    atoms["H3"] = atoms.pop("H2")
    _assert_document_rejected(document, "atom order")


def test_validator_rejects_incomplete_tensors_and_inconsistent_frames():
    document = complete_document()
    document["polarizabilities"]["frequency_blocks"][0]["atoms"]["O"]["spherical"]["matrix"].pop()
    _assert_document_rejected(document, "16x16")

    document = complete_document()
    document["polarizabilities"]["frequency_blocks"][0]["atoms"]["H1"]["local_to_global"][0][0] = -1.0
    _assert_document_rejected(document, "left-handed")

    document = complete_document()
    document["polarizabilities"]["frequency_blocks"][0]["atoms"]["O"]["global_cartesian"][0][1] = 0.1
    _assert_document_rejected(document, "not symmetric")


def test_validator_rejects_frequency_and_dispersion_invariants():
    document = complete_document()
    document["polarizabilities"]["frequency_blocks"][4]["omega"] = 0.041
    _assert_document_rejected(document, "omega")

    document = complete_document()
    del document["dispersion"]["matrices"]["C12"]
    _assert_document_rejected(document, "C12")

    document = complete_document()
    document["dispersion"]["matrices"]["C8"][0][1] = 9.0
    _assert_document_rejected(document, "not symmetric")


def test_validator_rejects_invalid_provenance_checksums():
    document = complete_document()
    document["generator"]["sha256"] = "not-a-digest"
    _assert_document_rejected(document, "generator.sha256")

    document = complete_document()
    document["sources"]["refined_pol"]["sha256"] = "F" * 64
    _assert_document_rejected(document, "sources.refined_pol.sha256")

    document = complete_document()
    document["tools"]["orient"]["executable"]["sha256"] = "short"
    _assert_document_rejected(document, "tools.orient.executable.sha256")


def test_validator_rejects_missing_input_path():
    document = complete_document()
    del document["inputs"]["H2O.clt"]["path"]
    _assert_document_rejected(document, "inputs.H2O.clt")


def test_validator_rejects_invalid_input_path():
    document = complete_document()
    document["inputs"]["H2O.axes"]["path"] = 7
    _assert_document_rejected(document, "inputs.H2O.axes.path")


def test_validator_rejects_bare_tool_executable_paths():
    mutations = (
        ("orient", None),
        ("psi4", None),
        ("camcasp", "camcasp"),
    )
    for tool, executable_name in mutations:
        document = complete_document()
        if executable_name is None:
            document["tools"][tool]["executable"] = f"/opt/{tool}"
            expected = f"tools.{tool}.executable"
        else:
            document["tools"][tool]["executables"][executable_name] = (
                f"/opt/{executable_name}"
            )
            expected = f"tools.{tool}.executables.{executable_name}"
        _assert_document_rejected(document, expected)


def test_validator_rejects_missing_tool_executable_checksums():
    mutations = (
        ("orient", None),
        ("psi4", None),
        ("camcasp", "pfit"),
    )
    for tool, executable_name in mutations:
        document = complete_document()
        if executable_name is None:
            del document["tools"][tool]["executable"]["sha256"]
            expected = f"tools.{tool}.executable"
        else:
            del document["tools"][tool]["executables"][executable_name]["sha256"]
            expected = f"tools.{tool}.executables.{executable_name}"
        _assert_document_rejected(document, expected)


def test_validator_rejects_invalid_tool_executable_checksums():
    mutations = (
        ("orient", None),
        ("psi4", None),
        ("camcasp", "casimir"),
    )
    for tool, executable_name in mutations:
        document = complete_document()
        if executable_name is None:
            document["tools"][tool]["executable"]["sha256"] = "invalid"
            expected = f"tools.{tool}.executable.sha256"
        else:
            document["tools"][tool]["executables"][executable_name][
                "sha256"
            ] = "invalid"
            expected = f"tools.{tool}.executables.{executable_name}.sha256"
        _assert_document_rejected(document, expected)


def test_validator_rejects_empty_or_incomplete_camcasp_executables():
    document = complete_document()
    document["tools"]["camcasp"]["executables"].clear()
    _assert_document_rejected(document, "camcasp")

    for executable_name in CAMCASP_EXECUTABLE_NAMES:
        document = complete_document()
        del document["tools"]["camcasp"]["executables"][executable_name]
        _assert_document_rejected(document, executable_name)


def test_validator_rejects_internally_consistent_noncanonical_grid():
    document = complete_document()
    noncanonical = [0.0] + [index / 100.0 for index in range(1, 11)]
    document["frequencies"]["values"] = noncanonical
    document["frequencies"]["squared_source_values"] = ["0.0"] + [
        f"-{omega ** 2:.8f}" for omega in noncanonical[1:]
    ]
    for block, omega in zip(
        document["polarizabilities"]["frequency_blocks"], noncanonical
    ):
        block["omega"] = omega
    _assert_document_rejected(document, "canonical Gauss-Legendre")


def test_atomic_json_validation_failure_preserves_existing_file(tmp_path):
    output = tmp_path / "atomic-polarizabilities.json"
    output.write_text("stale reference\n")
    document = complete_document()
    document["schema_version"] = 2
    _assert_document_rejected(document, "schema_version")
    try:
        write_atomic_json(output, document)
    except ReferenceFormatError:
        pass
    else:
        raise AssertionError("invalid document was written")
    assert output.read_text() == "stale reference\n"
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_json_replace_failure_preserves_existing_file(tmp_path, monkeypatch):
    output = tmp_path / "atomic-polarizabilities.json"
    output.write_text("stale reference\n")

    def fail_replace(source, destination):
        raise OSError("replacement failed")

    monkeypatch.setattr(os, "replace", fail_replace)
    try:
        write_atomic_json(output, complete_document())
    except OSError as exc:
        assert "replacement failed" in str(exc)
    else:
        raise AssertionError("replacement failure did not propagate")
    assert output.read_text() == "stale reference\n"
    assert not list(tmp_path.glob("*.tmp"))


from devtools.camcasp_reference import (  # noqa: E402
    build_reference_document,
    validate_generated_protocol,
    validate_stage_artifacts,
)


def canonical_generated_protocol_texts():
    clt = """\
Run-type properties
  Basis aVTZ
  SCFcode psi4
  Method DFT
  Functional PBE0
  Kernel ALDA+CHF
  Options Tests
End
"""
    cks = """\
SET Global_data
  XC-func PBE0
END
SET QUAD
  Type Gauss-Legendre
  Beta 0.5
END
SET NEW-PROP
  Kernel ALDA
END
SET PROPAGATOR
  Type CKS
END
SET NEW-PROP
  Kernel ALDA
END
SET PROPAGATOR
  Type CKS
END
BEGIN Polarizability
  Quad 10
  Rank 4
  Print pols for Orient
END
"""
    camcasp_log = """\
AC options: type = GRAC
basis = aug-cc-pvtz
basis type = None
Run-type: properties
SCFcode: psi4
"""
    psi4_input = """\
molecule {
  symmetry c1
  no_com
  no_reorient
  O 0 0 0
}
set {
  basis aug-cc-pvtz
}
energy, wfn = energy('PBE0', return_wfn=True)
"""
    psi4_output = (
        "Running in c1 symmetry.\n"
        "    => Composite Functional: PBE0 <=\n"
    )
    return clt, cks, camcasp_log, psi4_input, psi4_output


def test_generated_protocol_requires_explicit_canonical_settings():
    validate_generated_protocol(*canonical_generated_protocol_texts())


def _assert_protocol_rejected(texts, expected):
    try:
        validate_generated_protocol(*texts)
    except ReferenceFormatError as exc:
        assert expected.lower() in str(exc).lower()
    else:
        raise AssertionError("invalid generated protocol was accepted")


def test_generated_protocol_ignores_comment_only_settings():
    texts = list(canonical_generated_protocol_texts())
    texts[0] = texts[0].replace("  Basis aVTZ", "  ! Basis aVTZ")
    _assert_protocol_rejected(texts, "Basis")


@pytest.mark.parametrize("directive", ("Localization", "No Localization"))
def test_generated_protocol_rejects_active_localization(directive):
    texts = list(canonical_generated_protocol_texts())
    texts[0] = texts[0].replace("  Options Tests", f"  Options Tests\n  {directive}")
    _assert_protocol_rejected(texts, "Localization")


def test_generated_protocol_allows_commented_localization():
    texts = list(canonical_generated_protocol_texts())
    texts[0] = texts[0].replace(
        "  Options Tests",
        "  Options Tests\n  ! Localization\n  # No Localization",
    )
    validate_generated_protocol(*texts)


def test_generated_protocol_rejects_duplicate_active_setting():
    texts = list(canonical_generated_protocol_texts())
    texts[0] = texts[0].replace("  Basis aVTZ", "  Basis aVTZ\n  Basis aVTZ")
    _assert_protocol_rejected(texts, "Basis")


def test_generated_protocol_rejects_conflicting_active_settings():
    mutations = (
        (0, "Basis aVTZ", "Basis aDZ", "Basis"),
        (0, "SCFcode psi4", "SCFcode dalton", "SCFcode"),
        (0, "Method DFT", "Method HF", "Method"),
        (0, "Functional PBE0", "Functional PBE", "Functional"),
        (0, "Kernel ALDA+CHF", "Kernel ALDA", "Kernel"),
        (0, "Options Tests", "Options Production", "Options"),
        (1, "XC-func PBE0", "XC-func PBE", "XC-func"),
        (1, "Type Gauss-Legendre", "Type Euler-Maclaurin", "Type"),
        (1, "Beta 0.5", "Beta 0.7", "Beta"),
        (1, "Kernel ALDA", "Kernel CHF", "Kernel"),
        (1, "Type CKS", "Type CHF", "PROPAGATOR"),
        (1, "Quad 10", "Quad 9", "Quad"),
        (1, "Rank 4", "Rank 3", "Rank"),
        (1, "Print pols for Orient", "Print pols for Molpro", "Print"),
        (2, "type = GRAC", "type = LB94", "AC options"),
        (2, "basis = aug-cc-pvtz", "basis = cc-pvdz", "basis"),
        (2, "Run-type: properties", "Run-type: energy", "Run-type"),
        (2, "SCFcode: psi4", "SCFcode: dalton", "SCFcode"),
        (3, "basis aug-cc-pvtz", "basis cc-pvdz", "basis"),
        (3, "energy('PBE0'", "energy('PBE'", "energy"),
        (3, "symmetry c1", "symmetry c2v", "symmetry"),
        (3, "no_com", "com", "no_com"),
        (3, "no_reorient", "reorient", "no_reorient"),
        (4, "c1 symmetry", "c2v symmetry", "symmetry"),
        (4, "Composite Functional: PBE0", "Composite Functional: PBE", "PBE0"),
    )
    for text_index, old, new, expected in mutations:
        texts = list(canonical_generated_protocol_texts())
        texts[text_index] = texts[text_index].replace(old, new)
        _assert_protocol_rejected(texts, expected)


@pytest.mark.parametrize("mode", ("duplicate", "comment-only"))
def test_protocol_rejects_duplicate_or_comment_only_grac_report(mode):
    texts = list(canonical_generated_protocol_texts())
    if mode == "duplicate":
        texts[2] += "AC options: type = GRAC\n"
    else:
        texts[2] = texts[2].replace(
            "AC options: type = GRAC", "! AC options: type = GRAC"
        )
    _assert_protocol_rejected(texts, "AC options")


def test_protocol_rejects_duplicate_conflicting_actual_basis_report():
    texts = list(canonical_generated_protocol_texts())
    texts[2] += "basis = cc-pvdz\n"
    _assert_protocol_rejected(texts, "basis")


@pytest.mark.parametrize(
    "mode", ("duplicate", "conflicting", "comment-only", "unrelated-token")
)
def test_protocol_requires_one_active_composite_functional_report(mode):
    texts = list(canonical_generated_protocol_texts())
    report = "    => Composite Functional: PBE0 <="
    if mode == "duplicate":
        texts[4] += report + "\n"
    elif mode == "conflicting":
        texts[4] += "    => Composite Functional: PBE <=\n"
    elif mode == "comment-only":
        texts[4] = texts[4].replace(report, "# " + report)
    else:
        texts[4] = texts[4].replace(report, "Unrelated PBE0 token")
    _assert_protocol_rejected(texts, "Composite Functional")


@pytest.mark.parametrize("functional", ("pbe0", "Pbe0"))
def test_protocol_requires_exact_pbe0_functional_case(functional):
    texts = list(canonical_generated_protocol_texts())
    texts[4] = texts[4].replace(
        "Composite Functional: PBE0", f"Composite Functional: {functional}"
    )
    _assert_protocol_rejected(texts, "Composite Functional")


def _refined_block_text(combined_text, index):
    marker = f"# INDEX {index:03d}\n"
    block = combined_text.split(marker, 1)[1]
    if index < 10:
        block = block.split(f"# INDEX {index + 1:03d}\n", 1)[0]
    return block.strip() + "\n"


ORIENT_FINISHED = "Finished at 11:27:54 on 30 Jul 2026 "


def populate_stage_artifacts(work, job="H2O"):
    combined = make_real_l3_refined_text()
    for index in range(11):
        (work / f"{job}_L3_{index:03d}.out").write_text(
            f"ORIENT localization output\n{ORIENT_FINISHED}\n"
        )
        (work / f"{job}_ref_wt4_L3_{index:03d}.out").write_text(
            "PFIT refinement output\nFinished\n"
        )
        (work / f"{job}_ref_wt4_L3_{index:03d}.pol").write_text(
            _refined_block_text(combined, index)
        )
    (work / f"{job}_ref_wt4_L3_0f10.pol").write_text(combined)
    (work / f"{job}_ref_wt4_L3_casimir.out").write_text(
        "Dispersion coefficients\n" + CASIMIR_C12
    )
    (work / f"{job}_ref_wt4_L3_C12.pot").write_text(CASIMIR_C12)
    (work / f"{job}.pdef").write_text(
        "Polarizabilities\n  H1 H1 10 10 = H1_A\n"
        "  H2 H2 COPY H1 H1\nEnd\n"
    )


def test_stage_validation_accepts_completed_parseable_artifacts(tmp_path):
    populate_stage_artifacts(tmp_path)
    artifacts = validate_stage_artifacts(tmp_path, "H2O")
    assert len(artifacts) == 37
    assert "H2O_ref_wt4_L3_C12.pot" in artifacts


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


def test_stage_validation_rejects_truncated_orient_completion(tmp_path):
    populate_stage_artifacts(tmp_path)
    path = tmp_path / "H2O_L3_004.out"
    path.write_text("ORIENT produced nonempty output without completion\n")
    try:
        validate_stage_artifacts(tmp_path, "H2O")
    except ReferenceFormatError as exc:
        assert path.name in str(exc)
        assert "Finished" in str(exc)
    else:
        raise AssertionError("truncated ORIENT output was accepted")


def test_stage_validation_rejects_ambiguous_finished_completion(tmp_path):
    populate_stage_artifacts(tmp_path)
    path = tmp_path / "H2O_L3_002.out"
    path.write_text("Finished\nmore output\nFinished\n")
    try:
        validate_stage_artifacts(tmp_path, "H2O")
    except ReferenceFormatError as exc:
        assert path.name in str(exc)
        assert "unambiguous" in str(exc)
    else:
        raise AssertionError("ambiguous ORIENT completion was accepted")


@pytest.mark.parametrize("suffix", ("", " ", "\t", " \t\t"))
def test_stage_validation_accepts_only_horizontal_orient_completion_tail(
    tmp_path, suffix
):
    populate_stage_artifacts(tmp_path)
    path = tmp_path / "H2O_L3_001.out"
    path.write_text(
        "ORIENT output\nFinished at 11:27:54 on 30 Jul 2026" + suffix + "\n"
    )
    validate_stage_artifacts(tmp_path, "H2O")


@pytest.mark.parametrize("suffix", ("", " ", "\t", " \t\t"))
def test_stage_validation_accepts_only_horizontal_pfit_completion_tail(
    tmp_path, suffix
):
    populate_stage_artifacts(tmp_path)
    path = tmp_path / "H2O_ref_wt4_L3_001.out"
    path.write_text("PFIT output\nFinished" + suffix + "\n")
    validate_stage_artifacts(tmp_path, "H2O")


def test_stage_validation_accepts_blank_tail_after_canonical_orient_completion(tmp_path):
    populate_stage_artifacts(tmp_path)
    path = tmp_path / "H2O_L3_003.out"
    path.write_text(f"ORIENT output\n{ORIENT_FINISHED}\n\n  \n")
    validate_stage_artifacts(tmp_path, "H2O")


@pytest.mark.parametrize(
    "trailer",
    (
        "Finished",
        "finished at 11:27:54 on 30 Jul 2026",
        "Finished At 11:27:54 on 30 Jul 2026",
        "Finished at 1:27:54 on 30 Jul 2026",
        "Finished at 11:27:5 on 30 Jul 2026",
        "Finished at 24:27:54 on 30 Jul 2026",
        "Finished at 11:60:54 on 30 Jul 2026",
        "Finished at 11:27:60 on 30 Jul 2026",
        "Finished at 11:27:54 on 3 Jul 2026",
        "Finished at 11:27:54 on 30 July 2026",
        "Finished at 11:27:54 on 30 jul 2026",
        "Finished at 11:27:54 on 30 Jul 26",
        "Finished at 11:27:54 on 31 Feb 2026",
        "Finished at 11:27:54 on 30 Jul 2026 extra",
        "message Finished at 11:27:54 on 30 Jul 2026",
        " Finished at 11:27:54 on 30 Jul 2026",
    ),
)
def test_stage_validation_rejects_noncanonical_orient_completion(tmp_path, trailer):
    populate_stage_artifacts(tmp_path)
    path = tmp_path / "H2O_L3_005.out"
    path.write_text(f"ORIENT output\n{trailer}\n")
    with pytest.raises(ReferenceFormatError, match="ORIENT.*Finished at"):
        validate_stage_artifacts(tmp_path, "H2O")


@pytest.mark.parametrize(
    "text",
    (
        f"ORIENT output\n{ORIENT_FINISHED}\nmore output\n",
        f"ORIENT output\n{ORIENT_FINISHED}\n{ORIENT_FINISHED}\n",
        f"ORIENT output\nFinished at 10:00:00 on 29 Jul 2026\n{ORIENT_FINISHED}\n",
    ),
)
def test_stage_validation_rejects_nonterminal_or_duplicate_orient_completion(
    tmp_path, text
):
    populate_stage_artifacts(tmp_path)
    path = tmp_path / "H2O_L3_006.out"
    path.write_text(text)
    with pytest.raises(ReferenceFormatError, match="unambiguous terminal ORIENT"):
        validate_stage_artifacts(tmp_path, "H2O")


@pytest.mark.parametrize(
    ("role", "suffix"),
    (
        ("ORIENT", "\u00a0"),
        ("ORIENT", "\u2003"),
        ("ORIENT", "\v"),
        ("ORIENT", "\f"),
        ("ORIENT", "\u0085"),
        ("ORIENT", "\u2028"),
        ("ORIENT", "\u2029"),
        ("ORIENT", "\r"),
        ("PFIT", "\u00a0"),
        ("PFIT", "\u2003"),
        ("PFIT", "\v"),
        ("PFIT", "\f"),
        ("PFIT", "\u0085"),
        ("PFIT", "\u2028"),
        ("PFIT", "\u2029"),
        ("PFIT", "\r"),
    ),
)
def test_stage_validation_rejects_nonhorizontal_completion_tail(
    tmp_path, role, suffix
):
    populate_stage_artifacts(tmp_path)
    if role == "ORIENT":
        path = tmp_path / "H2O_L3_007.out"
        completion = "Finished at 11:27:54 on 30 Jul 2026"
    else:
        path = tmp_path / "H2O_ref_wt4_L3_007.out"
        completion = "Finished"
    path.write_text(f"{role} output\n{completion}{suffix}\n")
    with pytest.raises(ReferenceFormatError, match=f"terminal {role} completion"):
        validate_stage_artifacts(tmp_path, "H2O")


@pytest.mark.parametrize("role", ("ORIENT", "PFIT"))
def test_stage_validation_deliberately_rejects_crlf_globally(tmp_path, role):
    populate_stage_artifacts(tmp_path)
    if role == "ORIENT":
        path = tmp_path / "H2O_L3_008.out"
        completion = b"Finished at 11:27:54 on 30 Jul 2026 "
    else:
        path = tmp_path / "H2O_ref_wt4_L3_008.out"
        completion = b"Finished"
    path.write_bytes(role.encode() + b" output\r\n" + completion + b"\r\n")
    with pytest.raises(ReferenceFormatError, match="non-LF line separator"):
        validate_stage_artifacts(tmp_path, "H2O")


@pytest.mark.parametrize("role", ("ORIENT", "PFIT"))
def test_stage_validation_rejects_bare_cr_completion_record(tmp_path, role):
    populate_stage_artifacts(tmp_path)
    if role == "ORIENT":
        path = tmp_path / "H2O_L3_008.out"
        completion = b"Finished at 11:27:54 on 30 Jul 2026 "
    else:
        path = tmp_path / "H2O_ref_wt4_L3_008.out"
        completion = b"Finished"
    path.write_bytes(role.encode() + b" output\n" + completion + b"\r")
    with pytest.raises(ReferenceFormatError, match=f"terminal {role} completion"):
        validate_stage_artifacts(tmp_path, "H2O")


@pytest.mark.parametrize(
    ("role", "text"),
    (
        (
            "ORIENT",
            f"ORIENT output\nFinished\n{ORIENT_FINISHED}\n",
        ),
        (
            "ORIENT",
            f"ORIENT output\nFinished at 11:27:54 on 30 Jul 2026 extra\n"
            f"{ORIENT_FINISHED}\n",
        ),
        (
            "ORIENT",
            f"ORIENT output\n\u00a0Finished at 11:27:54 on 30 Jul 2026\n"
            f"{ORIENT_FINISHED}\n",
        ),
        (
            "ORIENT",
            f"ORIENT output\n\u2003Finished at 11:27:54 on 30 Jul 2026\n"
            f"{ORIENT_FINISHED}\n",
        ),
        (
            "PFIT",
            f"PFIT output\n{ORIENT_FINISHED}\nFinished\n",
        ),
        (
            "PFIT",
            "PFIT output\n\u00a0Finished\nFinished \t\n",
        ),
        (
            "PFIT",
            "PFIT output\n\u2003Finished\nFinished \t\n",
        ),
    ),
)
def test_stage_validation_rejects_competing_malformed_completion_lines(
    tmp_path, role, text
):
    populate_stage_artifacts(tmp_path)
    path = (
        tmp_path / "H2O_L3_009.out" if role == "ORIENT"
        else tmp_path / "H2O_ref_wt4_L3_009.out"
    )
    path.write_text(text)
    with pytest.raises(ReferenceFormatError, match=f"terminal {role} completion"):
        validate_stage_artifacts(tmp_path, "H2O")


def test_stage_validation_rejects_fatal_orient_output_with_valid_completion(tmp_path):
    populate_stage_artifacts(tmp_path)
    path = tmp_path / "H2O_L3_008.out"
    path.write_text(f"FATAL ERROR in ORIENT\n{ORIENT_FINISHED}\n")
    with pytest.raises(ReferenceFormatError, match="fatal or truncation marker"):
        validate_stage_artifacts(tmp_path, "H2O")


def test_stage_validation_rejects_truncated_pfit_completion(tmp_path):
    populate_stage_artifacts(tmp_path)
    path = tmp_path / "H2O_ref_wt4_L3_009.out"
    path.write_text("PFIT produced nonempty output without completion\n")
    try:
        validate_stage_artifacts(tmp_path, "H2O")
    except ReferenceFormatError as exc:
        assert path.name in str(exc)
        assert "Finished" in str(exc)
    else:
        raise AssertionError("truncated PFIT output was accepted")


@pytest.mark.parametrize(
    "text",
    (
        f"PFIT output\n{ORIENT_FINISHED}\n",
        "PFIT output\nfinished\n",
        "PFIT output\nFinished extra\n",
        "PFIT output\nmessage Finished\n",
        "PFIT output\n Finished\n",
        "PFIT output\nFinished\nmore output\n",
        "PFIT output\nFinished\nFinished\n",
    ),
)
def test_stage_validation_rejects_noncanonical_pfit_completion(tmp_path, text):
    populate_stage_artifacts(tmp_path)
    path = tmp_path / "H2O_ref_wt4_L3_003.out"
    path.write_text(text)
    with pytest.raises(ReferenceFormatError, match="unambiguous terminal PFIT.*Finished"):
        validate_stage_artifacts(tmp_path, "H2O")


def test_stage_validation_rejects_fatal_pfit_output_with_valid_completion(tmp_path):
    populate_stage_artifacts(tmp_path)
    path = tmp_path / "H2O_ref_wt4_L3_004.out"
    path.write_text("PFIT output\nFATAL ERROR\nFinished\n")
    with pytest.raises(ReferenceFormatError, match="fatal or truncation marker"):
        validate_stage_artifacts(tmp_path, "H2O")


def test_stage_validation_rejects_malformed_individual_pfit_pol(tmp_path):
    populate_stage_artifacts(tmp_path)
    path = tmp_path / "H2O_ref_wt4_L3_006.pol"
    path.write_text("nonempty unrelated polarizability output\n")
    try:
        validate_stage_artifacts(tmp_path, "H2O")
    except ReferenceFormatError as exc:
        assert path.name in str(exc)
    else:
        raise AssertionError("unrelated PFIT polarizability was accepted")


@pytest.mark.parametrize("mode", ("header-index", "value", "comment", "index-marker"))
def test_stage_validation_rejects_noncorresponding_real_individual_pol(tmp_path, mode):
    populate_stage_artifacts(tmp_path)
    path = tmp_path / "H2O_ref_wt4_L3_002.pol"
    text = path.read_text()
    if mode == "header-index":
        text = text.replace("INDEX   0", "INDEX   1", 1)
    elif mode == "value":
        text = text.replace("2.000000000000", "2.125000000000", 1)
    elif mode == "comment":
        text = "# unexpected individual comment\n" + text
    else:
        text = "# INDEX 002\n" + text
    path.write_text(text)
    with pytest.raises(ReferenceFormatError, match="INDEX 0|does not match|ALPHA"):
        validate_stage_artifacts(tmp_path, "H2O")


def test_stage_validation_rejects_marker_only_casimir_output(tmp_path):
    populate_stage_artifacts(tmp_path)
    path = tmp_path / "H2O_ref_wt4_L3_casimir.out"
    path.write_text("Dispersion coefficients\n")
    try:
        validate_stage_artifacts(tmp_path, "H2O")
    except ReferenceFormatError as exc:
        assert path.name in str(exc)
        assert "body" in str(exc)
    else:
        raise AssertionError("marker-only CASIMIR output was accepted")


def test_stage_validation_rejects_junk_casimir_body(tmp_path):
    populate_stage_artifacts(tmp_path)
    path = tmp_path / "H2O_ref_wt4_L3_casimir.out"
    path.write_text("Dispersion coefficients\nnonempty junk\n")
    try:
        validate_stage_artifacts(tmp_path, "H2O")
    except ReferenceFormatError as exc:
        assert path.name in str(exc)
        assert "missing atom-type pairs" in str(exc)
    else:
        raise AssertionError("junk CASIMIR body was accepted")


def test_stage_validation_rejects_casimir_output_without_final_end(tmp_path):
    populate_stage_artifacts(tmp_path)
    path = tmp_path / "H2O_ref_wt4_L3_casimir.out"
    body = CASIMIR_C12.rsplit("  End\n", 1)[0] + "\n"
    path.write_text("Dispersion coefficients\n" + body)
    try:
        validate_stage_artifacts(tmp_path, "H2O")
    except ReferenceFormatError as exc:
        assert path.name in str(exc)
        assert "End" in str(exc)
    else:
        raise AssertionError("unterminated CASIMIR output body was accepted")


def test_stage_validation_rejects_casimir_output_potential_mismatch(tmp_path):
    populate_stage_artifacts(tmp_path)
    path = tmp_path / "H2O_ref_wt4_L3_casimir.out"
    path.write_text(
        "Dispersion coefficients\n" + CASIMIR_C12.replace("20.0", "21.0", 1)
    )
    try:
        validate_stage_artifacts(tmp_path, "H2O")
    except ReferenceFormatError as exc:
        assert path.name in str(exc)
        assert "does not match" in str(exc)
    else:
        raise AssertionError("mismatched CASIMIR output and potential were accepted")


def test_stage_validation_rejects_fatal_marker_outside_out(tmp_path):
    populate_stage_artifacts(tmp_path)
    path = tmp_path / "H2O.pdef"
    path.write_text(path.read_text() + "FATAL ERROR model truncated\n")
    try:
        validate_stage_artifacts(tmp_path, "H2O")
    except ReferenceFormatError as exc:
        assert path.name in str(exc)
        assert "marker" in str(exc)
    else:
        raise AssertionError("fatal pdef marker was accepted")


def make_canonical_nl4_frequency_text():
    lines = []
    for value in APPROVED_SQUARED_SOURCE_VALUES:
        lines.append(
            "POL SITE-LABELS O O SITE-INDICES 1 1 RANK 0 : 4 BY 0 : 4 "
            f"FREQ2 {value} CARTSPHER S"
        )
    return "\n".join(lines) + "\n"


def test_builder_combines_all_required_properties(tmp_path):
    nl4 = tmp_path / "NL4_fmtB.pol"
    refined = tmp_path / "H2O_ref_wt4_L3_0f10.pol"
    pot = tmp_path / "H2O_ref_wt4_L3_C12.pot"
    axes = tmp_path / "H2O.axes"
    nl4.write_text(make_canonical_nl4_frequency_text())
    refined.write_text(make_real_l3_refined_text())
    pot.write_text(CASIMIR_C12)
    axes.write_text(CANONICAL_AXES)

    metadata = complete_document()
    document = build_reference_document(
        frequency_path=nl4,
        refined_path=refined,
        casimir_path=pot,
        axes_path=axes,
        metadata=metadata,
    )
    validate_reference_document(document)
    assert len(document["polarizabilities"]["frequency_blocks"]) == 11
    assert tuple(document["dispersion"]["matrices"]) == (
        "C6", "C8", "C10", "C12"
    )


def _write_executable(path, text):
    path.write_text(text)
    path.chmod(0o755)


@pytest.mark.parametrize(
    ("case", "names", "expected"),
    (
        ("missing-a", ("H2O_NL4_fmtB.pol",), "format-A"),
        (
            "duplicate-a",
            ("H2O_NL4_fmtA.pol", "other_NL4_fmtA.pol", "H2O_NL4_fmtB.pol"),
            "format-A",
        ),
        ("missing-b", ("H2O_NL4_fmtA.pol",), "format-B"),
        (
            "duplicate-b",
            ("H2O_NL4_fmtA.pol", "H2O_NL4_fmtB.pol", "other_NL4_fmtB.pol"),
            "format-B",
        ),
        (
            "unexpected-third",
            ("H2O_NL4_fmtA.pol", "H2O_NL4_fmtB.pol", "H2O_NL4_extra.pol"),
            "exactly two total NL4",
        ),
    ),
)
def test_discover_camcasp_artifacts_rejects_invalid_nl4_sets(
    tmp_path, case, names, expected
):
    job = tmp_path / case / "H2O"
    output = job / "OUT"
    output.mkdir(parents=True)
    for name in names:
        (output / name).write_text(f"{name}\n")
    (output / "H2O.p2p").write_text("p2p\n")
    (job / "H2O_A.in").write_text("psi4 input\n")
    command = (
        f'source "{SCRIPT}"; discover_camcasp_artifacts "$1"'
    )
    result = subprocess.run(
        ["bash", "-c", command, "discover-nl4", str(job)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert expected.lower() in result.stderr.lower()


@pytest.mark.parametrize("empty_format", ("A", "B"))
def test_discover_camcasp_artifacts_rejects_empty_nl4(tmp_path, empty_format):
    job = tmp_path / f"empty-{empty_format}" / "H2O"
    output = job / "OUT"
    output.mkdir(parents=True)
    for format_name in ("A", "B"):
        path = output / f"H2O_NL4_fmt{format_name}.pol"
        path.write_text("" if format_name == empty_format else "nonempty\n")
    (output / "H2O.p2p").write_text("p2p\n")
    (job / "H2O_A.in").write_text("psi4 input\n")
    command = f'source "{SCRIPT}"; discover_camcasp_artifacts "$1"'
    result = subprocess.run(
        ["bash", "-c", command, "discover-empty-nl4", str(job)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert f"format-{empty_format}" in result.stderr
    assert "empty" in result.stderr


@pytest.mark.parametrize("mode", ("missing", "duplicate", "empty"))
def test_discover_camcasp_artifacts_requires_unique_psi4_output(tmp_path, mode):
    job = tmp_path / mode / "H2O"
    output = job / "OUT"
    output.mkdir(parents=True)
    (output / "H2O_NL4_fmtA.pol").write_text("A\n")
    (output / "H2O_NL4_fmtB.pol").write_text("B\n")
    (output / "H2O.p2p").write_text("p2p\n")
    (job / "H2O_A.in").write_text("input\n")
    (job / "H2O_A.executed.in").write_text("executed input\n")
    if mode == "duplicate":
        nested = job / "nested"
        nested.mkdir()
        (job / "H2O_A.out").write_text("output\n")
        (nested / "H2O_A.out").write_text("output\n")
    elif mode == "empty":
        (job / "H2O_A.out").write_text("")
    command = f'source "{SCRIPT}"; discover_camcasp_artifacts "$1"'
    result = subprocess.run(
        ["bash", "-c", command, "discover-psi4-output", str(job)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Psi4 output" in result.stderr


@pytest.mark.parametrize("mode", ("missing", "duplicate", "empty"))
def test_discover_camcasp_artifacts_requires_unique_executed_input_evidence(
    tmp_path, mode
):
    job = tmp_path / mode / "H2O"
    output = job / "OUT"
    output.mkdir(parents=True)
    (output / "H2O_NL4_fmtA.pol").write_text("A\n")
    (output / "H2O_NL4_fmtB.pol").write_text("B\n")
    (output / "H2O.p2p").write_text("p2p\n")
    (job / "H2O_A.in").write_text("generated input\n")
    (job / "H2O_A.out").write_text("output\n")
    if mode == "duplicate":
        nested = job / "nested"
        nested.mkdir()
        (job / "H2O_A.executed.in").write_text("executed input\n")
        (nested / "H2O_A.executed.in").write_text("executed input\n")
    elif mode == "empty":
        (job / "H2O_A.executed.in").write_text("")
    publication = tmp_path / mode / "manifest.json"
    command = (
        f'source "{SCRIPT}"; discover_camcasp_artifacts "$1"; : >"$2"'
    )
    result = subprocess.run(
        ["bash", "-c", command, "discover-executed-input", str(job),
         str(publication)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "executed Psi4 input" in result.stderr
    assert not publication.exists()


def _write_casimir_evidence(work, runtime, *, cgdir=None, data_extra=""):
    work.mkdir(parents=True)
    _write_realcg_tables(runtime)
    relative_realcg = os.path.relpath(runtime / "data" / "realcg", work)
    cgdir = relative_realcg if cgdir is None else cgdir
    absolute_runtime = str(runtime.resolve())
    process_prefix = (
        "TITLE PROCESS file to write the CASIMIR input\n"
        "Set Global-data\n"
        "  CamCASP {camcasp}\n"
        "  Units BOHR DEGREE\n"
        "End\n"
        "Read local pols for H2O\n"
        "  Use ascii file {PREFIX}_0f10.pol\n"
        "  Maximum rank {LIMIT}\n"
        "  Limit rank to {HLIMIT} for sites H1 H2\n"
        "  Frequencies STATIC + 10\n"
        "End\nFinish\n"
        "! retained comment after terminal Finish\n\n"
    )
    generated_text = process_prefix.format(
        camcasp=absolute_runtime, PREFIX="{PREFIX}", LIMIT="{LIMIT}", HLIMIT="{HLIMIT}"
    )
    process_text = process_prefix.format(
        camcasp=os.path.relpath(runtime, work),
        PREFIX="{PREFIX}", LIMIT="{LIMIT}", HLIMIT="{HLIMIT}",
    )
    temp_text = process_text.format(PREFIX="H2O_ref_wt4_L3", LIMIT=3, HLIMIT=3)
    (work / "H2O_casimir.generated.prss").write_text(generated_text)
    (work / "H2O_casimir.prss").write_text(process_text)
    (work / "H2O_casimir.temp").write_text(temp_text)
    (work / "H2O_ref_wt4_L3_casimir.data").write_text(
        "Title H2O ... H2O\n"
        "Frequencies 0.5 10\n"
        "Skip 0\n"
        f"CGdir {cgdir}\n"
        "Dispersion 12 H2O\n"
        f"{data_extra}"
        "Finish\n"
        "# retained comment after terminal Finish\n\n"
    )
    return relative_realcg


def _run_patch_casimir_template(work, runtime):
    return subprocess.run(
        [
            "python", "-P", str(ROOT / "devtools" / "camcasp_reference.py"),
            "patch-casimir-template", "--work-dir", str(work),
            "--runtime", str(runtime),
            "--relative-runtime", os.path.relpath(runtime, work),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_patch_casimir_template_preserves_generated_bytes_and_patches_atomically(tmp_path):
    work = tmp_path / "reference" / "work" / "H2O"
    runtime = tmp_path / "reference" / "tools" / "camcasp-runtime"
    relative = _write_casimir_evidence(work, runtime)
    generated = work / "H2O_casimir.generated.prss"
    template = work / "H2O_casimir.prss"
    original = generated.read_bytes()
    generated.unlink()
    template.write_bytes(original)

    result = _run_patch_casimir_template(work, runtime)

    assert result.returncode == 0, result.stderr
    assert generated.read_bytes() == original
    expected = original.replace(
        f"CamCASP {runtime.resolve()}".encode(),
        f"CamCASP {os.path.relpath(runtime, work)}".encode(),
    )
    assert template.read_bytes() == expected
    directive = f"CamCASP {relative.rsplit('/data/realcg', 1)[0]}".encode()
    assert template.read_bytes().count(directive) == 1
    assert not list(work.glob("*.tmp"))


@pytest.mark.parametrize(
    "directive_prefix", (b"\tCamCASP\t\t", b"   CamCASP   ")
)
def test_patch_casimir_template_replaces_only_runtime_token_bytes(
    tmp_path, directive_prefix
):
    work = tmp_path / "reference" / "work" / "H2O"
    runtime = tmp_path / "reference" / "tools" / "camcasp-runtime"
    _write_casimir_evidence(work, runtime)
    generated = work / "H2O_casimir.generated.prss"
    template = work / "H2O_casimir.prss"
    original = generated.read_bytes()
    old_line = b"  CamCASP " + os.fsencode(runtime.resolve())
    absolute = os.fsencode(runtime.resolve())
    changed = original.replace(old_line, directive_prefix + absolute)
    changed = changed.replace(
        b"Finish\n", b"! unrelated retained path " + absolute + b"\nFinish\n", 1
    )
    generated.unlink()
    template.write_bytes(changed)

    result = _run_patch_casimir_template(work, runtime)

    assert result.returncode == 0, result.stderr
    relative = os.fsencode(os.path.relpath(runtime, work))
    token_start = changed.index(directive_prefix + absolute) + len(directive_prefix)
    expected = changed[:token_start] + relative + changed[token_start + len(absolute):]
    assert generated.read_bytes() == changed
    assert template.read_bytes() == expected
    assert changed[:token_start] == expected[:token_start]
    assert changed[token_start + len(absolute):] == expected[token_start + len(relative):]
    assert b"! unrelated retained path " + absolute in template.read_bytes()


def test_casimir_correspondence_uses_exact_token_offsets_with_flexible_spacing(tmp_path):
    work = tmp_path / "reference" / "work" / "H2O"
    runtime = tmp_path / "reference" / "tools" / "camcasp-runtime"
    _write_casimir_evidence(work, runtime)
    generated = work / "H2O_casimir.generated.prss"
    template = work / "H2O_casimir.prss"
    temp = work / "H2O_casimir.temp"
    absolute = os.fsencode(runtime.resolve())
    relative = os.fsencode(os.path.relpath(runtime, work))
    generated_data = generated.read_bytes().replace(
        b"  CamCASP " + absolute, b"\tCamCASP\t\t" + absolute
    )
    generated_data = generated_data.replace(
        b"Finish\n", b"! unrelated retained path " + absolute + b"\nFinish\n", 1
    )
    token_start = generated_data.index(b"\tCamCASP\t\t" + absolute) + len(
        b"\tCamCASP\t\t"
    )
    template_data = (
        generated_data[:token_start]
        + relative
        + generated_data[token_start + len(absolute):]
    )
    generated.write_bytes(generated_data)
    template.write_bytes(template_data)
    temp.write_bytes(
        template_data.decode("ascii").format(
            PREFIX="H2O_ref_wt4_L3", LIMIT=3, HLIMIT=3
        ).encode("ascii")
    )

    result = subprocess.run(
        [
            "python", "-P", str(ROOT / "devtools" / "camcasp_reference.py"),
            "validate-casimir", "--work-dir", str(work), "--runtime", str(runtime),
        ], cwd=ROOT, text=True, capture_output=True, check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "mode", ("crlf", "bare-cr", "non-ascii", "unicode-separator", "no-final-lf", "trailing-space")
)
def test_patch_casimir_template_rejects_noncanonical_byte_stream(tmp_path, mode):
    work = tmp_path / "reference" / "work" / "H2O"
    runtime = tmp_path / "reference" / "tools" / "camcasp-runtime"
    _write_casimir_evidence(work, runtime)
    generated = work / "H2O_casimir.generated.prss"
    template = work / "H2O_casimir.prss"
    data = generated.read_bytes()
    generated.unlink()
    if mode == "crlf":
        data = data.replace(b"\n", b"\r\n")
    elif mode == "bare-cr":
        data = data.replace(b"TITLE PROCESS", b"TITLE\rPROCESS")
    elif mode == "non-ascii":
        data = data.replace(b"TITLE", "TÍTLE".encode())
    elif mode == "unicode-separator":
        data = data.replace(
            b"TITLE PROCESS", b"TITLE" + "\u2028".encode() + b"PROCESS"
        )
    elif mode == "no-final-lf":
        data = data.rstrip(b"\n")
    else:
        data = data.replace(
            os.fsencode(runtime.resolve()) + b"\n",
            os.fsencode(runtime.resolve()) + b" \n",
        )
    template.write_bytes(data)

    result = _run_patch_casimir_template(work, runtime)

    assert result.returncode != 0
    assert not generated.exists()
    assert template.read_bytes() == data


@pytest.mark.parametrize(
    "mode",
    (
        "nested-set", "nested-molecule", "nested-read", "nested-write",
        "nested-global", "unmatched-end", "unbalanced", "duplicate-global-case",
        "noncanonical-global-case", "directive-outside",
    ),
)
def test_patch_casimir_template_rejects_malformed_top_level_structure(tmp_path, mode):
    work = tmp_path / "reference" / "work" / "H2O"
    runtime = tmp_path / "reference" / "tools" / "camcasp-runtime"
    _write_casimir_evidence(work, runtime)
    generated = work / "H2O_casimir.generated.prss"
    template = work / "H2O_casimir.prss"
    data = generated.read_bytes()
    generated.unlink()
    directive = b"  CamCASP " + os.fsencode(runtime.resolve()) + b"\n"
    if mode.startswith("nested-"):
        opener = {
            "nested-set": b"Set Other",
            "nested-molecule": b"Molecule Other",
            "nested-read": b"Read nested block",
            "nested-write": b"Write",
            "nested-global": b"sET gLOBAL-DATA",
        }[mode]
        data = data.replace(directive, b"  " + opener + b"\n" + directive + b"  End\n")
    elif mode == "unmatched-end":
        data = b"End\n" + data
    elif mode == "unbalanced":
        data = data.replace(b"  Units BOHR DEGREE\nEnd\n", b"  Units BOHR DEGREE\n")
    elif mode == "duplicate-global-case":
        block = b"sET gLOBAL-DATA\n  Units BOHR DEGREE\nEnd\n"
        data = data.replace(b"Set Global-data\n", block + b"Set Global-data\n")
    elif mode == "noncanonical-global-case":
        data = data.replace(b"Set Global-data", b"sET gLOBAL-DATA")
    else:
        data = data.replace(directive, b"").replace(
            b"End\nRead local", b"End\n" + directive + b"Read local", 1
        )
    template.write_bytes(data)

    result = _run_patch_casimir_template(work, runtime)

    assert result.returncode != 0
    assert not generated.exists()
    assert template.read_bytes() == data


@pytest.mark.parametrize(
    "mode",
    ("missing", "duplicate", "wrong", "commented", "outside", "early-finish", "symlink"),
)
def test_patch_casimir_template_fails_closed_on_unsafe_generated_input(tmp_path, mode):
    work = tmp_path / "reference" / "work" / "H2O"
    runtime = tmp_path / "reference" / "tools" / "camcasp-runtime"
    _write_casimir_evidence(work, runtime)
    generated = work / "H2O_casimir.generated.prss"
    template = work / "H2O_casimir.prss"
    generated.unlink()
    original = (work / "H2O_casimir.prss").read_text().replace(
        f"CamCASP {os.path.relpath(runtime, work)}", f"CamCASP {runtime.resolve()}"
    )
    if mode == "missing":
        original = original.replace(f"  CamCASP {runtime.resolve()}\n", "")
    elif mode == "duplicate":
        original = original.replace(
            f"  CamCASP {runtime.resolve()}\n",
            f"  CamCASP {runtime.resolve()}\n  CamCASP {runtime.resolve()}\n",
        )
    elif mode == "wrong":
        original = original.replace(str(runtime.resolve()), str((tmp_path / "wrong").resolve()))
    elif mode == "commented":
        original = original.replace("  CamCASP ", "! CamCASP ")
    elif mode == "outside":
        original = original.replace(f"  CamCASP {runtime.resolve()}\n", "")
        original = original.replace(
            "End\nRead local", f"End\nCamCASP {runtime.resolve()}\nRead local", 1
        )
    elif mode == "early-finish":
        original = original.replace("Finish\n", "Finish\nUnexpected active tail\n")
    template.write_text(original)
    if mode == "symlink":
        target = work / "untrusted.prss"
        target.write_text(original)
        template.unlink()
        template.symlink_to(target)
    publication = tmp_path / "manifest.json"

    result = _run_patch_casimir_template(work, runtime)
    if result.returncode == 0:
        publication.write_text("published\n")

    assert result.returncode != 0
    assert not publication.exists()
    assert not generated.exists()


@pytest.mark.parametrize("role", ("generated", "template", "temp"))
@pytest.mark.parametrize("mode", ("missing", "duplicate", "wrong", "commented"))
def test_casimir_evidence_rejects_invalid_camcasp_directive(tmp_path, role, mode):
    work = tmp_path / "reference" / "work" / "H2O"
    runtime = tmp_path / "reference" / "tools" / "camcasp-runtime"
    _write_casimir_evidence(work, runtime)
    paths = {
        "generated": work / "H2O_casimir.generated.prss",
        "template": work / "H2O_casimir.prss",
        "temp": work / "H2O_casimir.temp",
    }
    path = paths[role]
    expected = str(runtime.resolve()) if role == "generated" else os.path.relpath(runtime, work)
    text = path.read_text()
    line = f"  CamCASP {expected}\n"
    if mode == "missing":
        text = text.replace(line, "")
    elif mode == "duplicate":
        text = text.replace(line, line + line)
    elif mode == "wrong":
        text = text.replace(expected, "/wrong/absolute" if role == "generated" else "../wrong")
    else:
        text = text.replace(line, "! CamCASP " + expected + "\n")
    path.write_text(text)
    publication = tmp_path / "manifest.json"
    result = subprocess.run(
        [
            "python", "-P", str(ROOT / "devtools" / "camcasp_reference.py"),
            "validate-casimir", "--work-dir", str(work), "--runtime", str(runtime),
        ],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    if result.returncode == 0:
        publication.write_text("published\n")
    assert result.returncode != 0
    assert "CamCASP" in result.stderr
    assert not publication.exists()


@pytest.mark.parametrize(
    ("role", "expected"),
    (("template", "byte-correspond"), ("temp", "exact expanded")),
)
def test_casimir_evidence_requires_exact_generated_template_and_temp_relations(
    tmp_path, role, expected
):
    work = tmp_path / "reference" / "work" / "H2O"
    runtime = tmp_path / "reference" / "tools" / "camcasp-runtime"
    _write_casimir_evidence(work, runtime)
    path = work / ("H2O_casimir.prss" if role == "template" else "H2O_casimir.temp")
    path.write_text(path.read_text().replace("TITLE PROCESS", "TITLE CHANGED"))
    result = subprocess.run(
        [
            "python", "-P", str(ROOT / "devtools" / "camcasp_reference.py"),
            "validate-casimir", "--work-dir", str(work), "--runtime", str(runtime),
        ], cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert result.returncode != 0
    assert expected in result.stderr


def test_patch_casimir_template_rejects_ambiguous_generated_destination_symlink(tmp_path):
    work = tmp_path / "reference" / "work" / "H2O"
    runtime = tmp_path / "reference" / "tools" / "camcasp-runtime"
    _write_casimir_evidence(work, runtime)
    generated = work / "H2O_casimir.generated.prss"
    template = work / "H2O_casimir.prss"
    original = generated.read_bytes()
    generated.unlink()
    template.write_bytes(original)
    target = work / "untrusted-generated.prss"
    target.write_text("untrusted\n")
    generated.symlink_to(target)

    result = _run_patch_casimir_template(work, runtime)

    assert result.returncode != 0
    assert generated.is_symlink()
    assert target.read_text() == "untrusted\n"
    assert template.read_bytes() == original


def _direct_patch_fixture(tmp_path):
    work = tmp_path / "reference" / "work" / "H2O"
    runtime = tmp_path / "reference" / "tools" / "camcasp-runtime"
    _write_casimir_evidence(work, runtime)
    generated = work / "H2O_casimir.generated.prss"
    source = work / "H2O_casimir.prss"
    original = generated.read_bytes()
    generated.unlink()
    source.write_bytes(original)
    return work, runtime, source, generated, original


def test_patch_casimir_template_rejects_second_cooperative_patcher(tmp_path):
    work, runtime, source, generated, original = _direct_patch_fixture(tmp_path)
    lock_path = work.parent / ".H2O_casimir.template.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(BlockingIOError):
            camcasp_reference.patch_casimir_template(
                work, runtime, os.path.relpath(runtime, work)
            )
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)

    assert source.read_bytes() == original
    assert not generated.exists()


def test_patch_casimir_template_lock_open_is_nofollow_and_regular(tmp_path):
    work, runtime, source, generated, original = _direct_patch_fixture(tmp_path)
    lock_path = work.parent / ".H2O_casimir.template.lock"
    target = work.parent / "untrusted-lock-target"
    target.write_text("untrusted\n")
    lock_path.symlink_to(target)

    result = _run_patch_casimir_template(work, runtime)

    assert result.returncode != 0
    assert lock_path.is_symlink()
    assert target.read_text() == "untrusted\n"
    assert source.read_bytes() == original
    assert not generated.exists()


def test_patch_casimir_template_detects_source_lost_update_before_replace(tmp_path):
    work, runtime, source, generated, original = _direct_patch_fixture(tmp_path)
    concurrent = original.replace(b"TITLE PROCESS", b"TITLE ALTERED")

    class MutatingIO(camcasp_reference._CasimirTemplateIO):
        lstat_calls = 0

        def lstat(self, path):
            self.lstat_calls += 1
            if self.lstat_calls == 3:
                previous = super().lstat(path)
                path.write_bytes(concurrent)
                os.chmod(path, previous.st_mode & 0o777)
                os.utime(
                    path,
                    ns=(previous.st_atime_ns, previous.st_mtime_ns),
                    follow_symlinks=False,
                )
            return super().lstat(path)

    with pytest.raises(camcasp_reference.ReferenceFormatError, match="changed"):
        camcasp_reference.patch_casimir_template(
            work, runtime, os.path.relpath(runtime, work), _io=MutatingIO()
        )

    assert generated.read_bytes() == original
    assert source.read_bytes() == concurrent
    assert not list(work.glob(".*.tmp"))


def test_generated_publish_links_only_complete_staged_inode(tmp_path):
    work, runtime, source, generated, original = _direct_patch_fixture(tmp_path)

    class InspectingIO(camcasp_reference._CasimirTemplateIO):
        def link(self, staged, destination):
            assert Path(staged).read_bytes() == original
            assert not destination.exists()
            return super().link(staged, destination)

    camcasp_reference.patch_casimir_template(
        work, runtime, os.path.relpath(runtime, work), _io=InspectingIO()
    )

    assert generated.read_bytes() == original


def test_generated_publish_never_removes_concurrent_destination(tmp_path):
    work, runtime, source, generated, original = _direct_patch_fixture(tmp_path)

    class ConcurrentIO(camcasp_reference._CasimirTemplateIO):
        def link(self, staged, destination):
            destination.write_bytes(b"concurrent complete evidence\n")
            return super().link(staged, destination)

    with pytest.raises(FileExistsError):
        camcasp_reference.patch_casimir_template(
            work, runtime, os.path.relpath(runtime, work), _io=ConcurrentIO()
        )

    assert generated.read_bytes() == b"concurrent complete evidence\n"
    assert source.read_bytes() == original
    assert not list(work.glob(".*.tmp"))


def test_concurrent_generated_artifact_survives_temp_cleanup_failure(tmp_path):
    work, runtime, source, generated, original = _direct_patch_fixture(tmp_path)

    class ConcurrentCleanupFailureIO(camcasp_reference._CasimirTemplateIO):
        def link(self, staged, destination):
            destination.write_bytes(b"concurrent complete evidence\n")
            return super().link(staged, destination)

        def unlink(self, path):
            raise OSError("injected concurrent cleanup failure")

    with pytest.raises(OSError, match="cleanup"):
        camcasp_reference.patch_casimir_template(
            work, runtime, os.path.relpath(runtime, work),
            _io=ConcurrentCleanupFailureIO(),
        )

    assert generated.read_bytes() == b"concurrent complete evidence\n"
    assert source.read_bytes() == original
    staged = list(work.glob(".*.tmp"))
    assert len(staged) == 1
    assert staged[0].read_bytes() == original


def test_generated_artifact_survives_staging_cleanup_failure(tmp_path):
    work, runtime, source, generated, original = _direct_patch_fixture(tmp_path)

    class CleanupFailureIO(camcasp_reference._CasimirTemplateIO):
        def unlink(self, path):
            raise OSError("injected staging cleanup failure")

    with pytest.raises(OSError, match="cleanup"):
        camcasp_reference.patch_casimir_template(
            work, runtime, os.path.relpath(runtime, work), _io=CleanupFailureIO()
        )

    assert generated.read_bytes() == original
    assert source.read_bytes() == original
    staged = list(work.glob(".*.tmp"))
    assert len(staged) == 1
    assert staged[0].read_bytes() == original


@pytest.mark.parametrize(
    "fault",
    (
        "lock-open", "generated-mkstemp", "generated-fchmod",
        "generated-close", "generated-link", "generated-directory-fsync",
        "generated-verification-read", "source-mkstemp", "source-fchmod",
        "source-close", "unlock", "lock-close",
    ),
)
def test_patch_casimir_template_injected_operation_failures_are_safe(tmp_path, fault):
    work, runtime, source, generated, original = _direct_patch_fixture(tmp_path)

    class OperationFaultIO(camcasp_reference._CasimirTemplateIO):
        phase = ""
        mkstemp_calls = 0
        lock_descriptor = None
        staged_descriptor = None
        close_calls = {}
        fault_descriptor = None
        ambiguous_close_failures = 0

        def open_lock(self, path):
            if fault == "lock-open":
                raise OSError("injected lock open failure")
            descriptor = super().open_lock(path)
            self.lock_descriptor = descriptor
            return descriptor

        def mkstemp(self, path):
            self.mkstemp_calls += 1
            self.phase = "generated" if self.mkstemp_calls == 1 else "source"
            if fault == f"{self.phase}-mkstemp":
                raise OSError("injected mkstemp failure")
            descriptor, name = super().mkstemp(path)
            self.staged_descriptor = descriptor
            return descriptor, name

        def fchmod(self, descriptor, mode):
            if descriptor == self.staged_descriptor and fault == f"{self.phase}-fchmod":
                raise OSError("injected fchmod failure")
            return super().fchmod(descriptor, mode)

        def close(self, descriptor):
            self.close_calls[descriptor] = self.close_calls.get(descriptor, 0) + 1
            is_staged = descriptor == self.staged_descriptor
            is_lock = descriptor == self.lock_descriptor
            result = super().close(descriptor)
            if (
                (is_staged and fault == f"{self.phase}-close")
                or (is_lock and fault == "lock-close")
            ):
                self.fault_descriptor = descriptor
                self.ambiguous_close_failures += 1
                raise OSError("injected ambiguous close failure")
            return result

        def link(self, staged, destination):
            if fault == "generated-link":
                raise OSError("injected link failure")
            return super().link(staged, destination)

        def fsync_directory(self, path):
            if fault == f"{self.phase}-directory-fsync":
                raise OSError("injected directory fsync failure")
            return super().fsync_directory(path)

        def read_bytes(self, path):
            if fault == "generated-verification-read" and path == generated:
                raise OSError("injected verification read failure")
            return super().read_bytes(path)

        def unlock(self, descriptor):
            if fault == "unlock":
                raise OSError("injected unlock failure")
            return super().unlock(descriptor)

    io = OperationFaultIO()
    with pytest.raises(OSError, match="injected"):
        camcasp_reference.patch_casimir_template(
            work, runtime, os.path.relpath(runtime, work), _io=io
        )

    if fault in {
        "lock-open", "generated-mkstemp", "generated-fchmod",
        "generated-close", "generated-link",
    }:
        assert not generated.exists()
        assert source.read_bytes() == original
    elif fault in {
        "generated-directory-fsync", "generated-verification-read",
        "source-mkstemp", "source-fchmod", "source-close",
    }:
        assert generated.read_bytes() == original
        assert source.read_bytes() == original
    else:
        assert generated.read_bytes() == original
        relative = os.fsencode(os.path.relpath(runtime, work))
        assert source.read_bytes() == original.replace(
            os.fsencode(runtime.resolve()), relative
        )
    if io.fault_descriptor is not None:
        assert io.ambiguous_close_failures == 1
    assert not list(work.glob(".*.tmp"))


@pytest.mark.parametrize("phase", ("generated", "source"))
def test_patch_casimir_template_rejects_no_progress_writes(tmp_path, phase):
    work, runtime, source, generated, original = _direct_patch_fixture(tmp_path)

    class NoProgressIO(camcasp_reference._CasimirTemplateIO):
        current_phase = ""
        mkstemp_calls = 0

        def mkstemp(self, path):
            self.mkstemp_calls += 1
            self.current_phase = "generated" if self.mkstemp_calls == 1 else "source"
            return super().mkstemp(path)

        def write(self, descriptor, data):
            if self.current_phase == phase:
                return 0
            return super().write(descriptor, data)

    with pytest.raises(OSError, match="no progress"):
        camcasp_reference.patch_casimir_template(
            work, runtime, os.path.relpath(runtime, work), _io=NoProgressIO()
        )

    if phase == "generated":
        assert not generated.exists()
    else:
        assert generated.read_bytes() == original
    assert source.read_bytes() == original
    assert not list(work.glob(".*.tmp"))


@pytest.mark.parametrize(
    "fault",
    (
        "generated-write", "generated-fsync", "source-write",
        "source-fsync", "source-replace", "source-directory-fsync",
    ),
)
def test_patch_casimir_template_has_defined_failure_terminal_states(tmp_path, fault):
    work, runtime, source, generated, original = _direct_patch_fixture(tmp_path)

    class FaultIO(camcasp_reference._CasimirTemplateIO):
        phase = ""
        mkstemp_calls = 0
        write_calls = {"generated": 0, "source": 0}

        def mkstemp(self, path):
            self.mkstemp_calls += 1
            self.phase = "generated" if self.mkstemp_calls == 1 else "source"
            return super().mkstemp(path)

        def write(self, descriptor, data):
            if fault == f"{self.phase}-write":
                if self.write_calls[self.phase] == 0:
                    self.write_calls[self.phase] += 1
                    return super().write(descriptor, data[:17])
                raise OSError("injected write failure")
            return super().write(descriptor, data)

        def fsync(self, descriptor):
            if fault == f"{self.phase}-fsync":
                raise OSError("injected fsync failure")
            return super().fsync(descriptor)

        def replace(self, source_path, destination_path):
            if fault == "source-replace":
                raise OSError("injected replace failure")
            return super().replace(source_path, destination_path)

        def fsync_directory(self, path):
            if fault == "source-directory-fsync" and self.phase == "source":
                raise OSError("injected directory fsync failure")
            return super().fsync_directory(path)

    relative = os.path.relpath(runtime, work)
    with pytest.raises(OSError, match="injected"):
        camcasp_reference.patch_casimir_template(
            work, runtime, relative, _io=FaultIO()
        )

    assert not list(work.glob(".*.tmp"))
    if fault.startswith("generated-"):
        assert not generated.exists()
        assert source.read_bytes() == original
    elif fault == "source-directory-fsync":
        assert generated.read_bytes() == original
        assert source.read_bytes() == original.replace(
            os.fsencode(runtime.resolve()), os.fsencode(relative)
        )
    else:
        assert generated.read_bytes() == original
        assert source.read_bytes() == original


def test_patch_casimir_template_write_loops_accept_short_writes(tmp_path):
    work, runtime, source, generated, original = _direct_patch_fixture(tmp_path)

    class ShortWriteIO(camcasp_reference._CasimirTemplateIO):
        def write(self, descriptor, data):
            return super().write(descriptor, data[:11])

    relative = os.path.relpath(runtime, work)
    camcasp_reference.patch_casimir_template(
        work, runtime, relative, _io=ShortWriteIO()
    )

    assert generated.read_bytes() == original
    assert source.read_bytes() == original.replace(
        os.fsencode(runtime.resolve()), os.fsencode(relative)
    )


def test_patch_casimir_template_rejects_preexisting_regular_generated_evidence(tmp_path):
    work, runtime, source, generated, original = _direct_patch_fixture(tmp_path)
    generated.write_bytes(b"preexisting evidence\n")

    result = _run_patch_casimir_template(work, runtime)

    assert result.returncode != 0
    assert generated.read_bytes() == b"preexisting evidence\n"
    assert source.read_bytes() == original


@pytest.mark.parametrize("role", ("generated", "template", "temp", "data"))
@pytest.mark.parametrize("mode", ("missing", "empty", "duplicate"))
def test_discover_casimir_artifacts_requires_four_unique_nonempty_roles(tmp_path, role, mode):
    job = tmp_path / "H2O"
    job.mkdir()
    names = {
        "generated": "H2O_casimir.generated.prss",
        "template": "H2O_casimir.prss",
        "temp": "H2O_casimir.temp",
        "data": "H2O_ref_wt4_L3_casimir.data",
    }
    for name in names.values():
        (job / name).write_text("evidence\n")
    path = job / names[role]
    if mode == "missing":
        path.unlink()
    elif mode == "empty":
        path.write_text("")
    else:
        nested = job / "nested"
        nested.mkdir()
        (nested / names[role]).write_text("duplicate\n")
    publication = tmp_path / "manifest.json"
    result = subprocess.run(
        ["bash", "-c", f'source "{SCRIPT}"; discover_casimir_artifacts "$1"; : >"$2"',
         "discover-casimir", str(job), str(publication)],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert result.returncode != 0
    assert "CASIMIR" in result.stderr
    assert not publication.exists()


@pytest.mark.parametrize(
    "mode", ("malformed", "absolute", "wrong-target", "duplicate", "missing")
)
def test_casimir_evidence_rejects_invalid_cgdir_without_publication(tmp_path, mode):
    work = tmp_path / "reference" / "work" / "H2O"
    runtime = tmp_path / "reference" / "tools" / "camcasp-runtime"
    relative = _write_casimir_evidence(work, runtime)
    data = work / "H2O_ref_wt4_L3_casimir.data"
    text = data.read_text()
    if mode == "malformed":
        text = text.replace(f"CGdir {relative}", f"CGdir {relative} ambiguous")
    elif mode == "absolute":
        text = text.replace(f"CGdir {relative}", f"CGdir {(runtime / 'data' / 'realcg').resolve()}")
    elif mode == "wrong-target":
        text = text.replace(f"CGdir {relative}", "CGdir .")
    elif mode == "duplicate":
        text = text.replace(f"CGdir {relative}", f"CGdir {relative}\nCGdir {relative}")
    else:
        text = text.replace(f"CGdir {relative}\n", "")
    data.write_text(text)
    publication = tmp_path / "manifest.json"
    result = subprocess.run(
        [
            "python", "-P", str(ROOT / "devtools" / "camcasp_reference.py"),
            "validate-casimir", "--work-dir", str(work), "--runtime", str(runtime),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        publication.write_text("published\n")
    assert result.returncode != 0
    assert "CGdir" in result.stderr
    assert not publication.exists()


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    (
        ("Frequencies 0.5 10", "Frequencies 1.0 10", "Frequencies"),
        ("Skip 0", "Skip 1", "Skip"),
        ("Dispersion 12 H2O", "Dispersion 10 H2O", "Dispersion"),
    ),
)
def test_casimir_evidence_requires_exact_final_controls(tmp_path, old, new, expected):
    work = tmp_path / "reference" / "work" / "H2O"
    runtime = tmp_path / "reference" / "tools" / "camcasp-runtime"
    _write_casimir_evidence(work, runtime)
    data = work / "H2O_ref_wt4_L3_casimir.data"
    data.write_text(data.read_text().replace(old, new))
    result = subprocess.run(
        [
            "python", "-P", str(ROOT / "devtools" / "camcasp_reference.py"),
            "validate-casimir", "--work-dir", str(work), "--runtime", str(runtime),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert expected in result.stderr


def test_casimir_evidence_accepts_one_terminal_finish_with_trailing_comments(tmp_path):
    work = tmp_path / "reference" / "work" / "H2O"
    runtime = tmp_path / "reference" / "tools" / "camcasp-runtime"
    _write_casimir_evidence(work, runtime)
    result = subprocess.run(
        [
            "python", "-P", str(ROOT / "devtools" / "camcasp_reference.py"),
            "validate-casimir", "--work-dir", str(work), "--runtime", str(runtime),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "filename",
    (
        "H2O_casimir.generated.prss", "H2O_casimir.prss",
        "H2O_casimir.temp", "H2O_ref_wt4_L3_casimir.data",
    ),
)
@pytest.mark.parametrize("mode", ("missing", "duplicate"))
def test_casimir_evidence_requires_exactly_one_active_terminal_finish(
    tmp_path, filename, mode
):
    work = tmp_path / "reference" / "work" / "H2O"
    runtime = tmp_path / "reference" / "tools" / "camcasp-runtime"
    _write_casimir_evidence(work, runtime)
    path = work / filename
    text = path.read_text()
    if mode == "missing":
        text = text.replace("Finish\n", "", 1)
    else:
        text = text.replace("Finish\n", "Finish\nFinish\n", 1)
    path.write_text(text)
    result = subprocess.run(
        [
            "python", "-P", str(ROOT / "devtools" / "camcasp_reference.py"),
            "validate-casimir", "--work-dir", str(work), "--runtime", str(runtime),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Finish" in result.stderr


@pytest.mark.parametrize(
    ("filename", "control"),
    (
        ("H2O_casimir.prss", "Frequencies STATIC + 10"),
        ("H2O_casimir.temp", "Frequencies STATIC + 10"),
        ("H2O_ref_wt4_L3_casimir.data", "Frequencies 0.5 10"),
        ("H2O_ref_wt4_L3_casimir.data", "Skip 0"),
        ("H2O_ref_wt4_L3_casimir.data", "CGdir "),
        ("H2O_ref_wt4_L3_casimir.data", "Dispersion 12 H2O"),
    ),
)
def test_casimir_evidence_rejects_required_control_after_early_finish(
    tmp_path, filename, control
):
    work = tmp_path / "reference" / "work" / "H2O"
    runtime = tmp_path / "reference" / "tools" / "camcasp-runtime"
    _write_casimir_evidence(work, runtime)
    path = work / filename
    text = path.read_text().replace("Finish\n", "", 1)
    text = text.replace(control, "Finish\n" + control, 1)
    path.write_text(text)
    result = subprocess.run(
        [
            "python", "-P", str(ROOT / "devtools" / "camcasp_reference.py"),
            "validate-casimir", "--work-dir", str(work), "--runtime", str(runtime),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "terminal Finish" in result.stderr


def test_casimir_evidence_requires_retained_process_and_temp(tmp_path):
    work = tmp_path / "reference" / "work" / "H2O"
    runtime = tmp_path / "reference" / "tools" / "camcasp-runtime"
    _write_casimir_evidence(work, runtime)
    (work / "H2O_casimir.temp").unlink()
    result = subprocess.run(
        [
            "python", "-P", str(ROOT / "devtools" / "camcasp_reference.py"),
            "validate-casimir", "--work-dir", str(work), "--runtime", str(runtime),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "casimir.temp" in result.stderr


def test_short_camcasp_path_rejects_casimir_record_overflow(tmp_path):
    work = tmp_path / "reference" / "work" / "H2O"
    runtime = tmp_path / ("runtime-" + "x" * 70)
    work.mkdir(parents=True)
    _write_realcg_tables(runtime)
    result = subprocess.run(
        [
            "bash", "-c", f'source "{SCRIPT}"; derive_short_camcasp_path "$1" "$2"',
            "short-camcasp", str(work), str(runtime),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "80-byte" in result.stderr


def test_casimir_evidence_rejects_long_canonical_cgdir(tmp_path):
    work = tmp_path / "reference" / "work" / "H2O"
    runtime = tmp_path / ("runtime-" + "x" * 70)
    _write_casimir_evidence(work, runtime)
    result = subprocess.run(
        [
            "python", "-P", str(ROOT / "devtools" / "camcasp_reference.py"),
            "validate-casimir", "--work-dir", str(work), "--runtime", str(runtime),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "80-byte" in result.stderr


@pytest.mark.parametrize("kind", ("path", "table"))
def test_short_camcasp_path_rejects_non_ascii_components(tmp_path, kind):
    work = tmp_path / "reference" / "work" / "H2O"
    runtime = tmp_path / ("rüntime" if kind == "path" else "runtime")
    work.mkdir(parents=True)
    _write_realcg_tables(runtime)
    if kind == "table":
        (runtime / "data" / "realcg" / "realcg_é").write_text("non-ascii\n")
    result = subprocess.run(
        [
            "bash", "-c", f'source "{SCRIPT}"; derive_short_camcasp_path "$1" "$2"',
            "short-camcasp-ascii", str(work), str(runtime),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "ASCII" in result.stderr


@pytest.mark.parametrize("kind", ("path", "table"))
def test_casimir_evidence_rejects_non_ascii_components(tmp_path, kind):
    work = tmp_path / "reference" / "work" / "H2O"
    runtime = tmp_path / ("rüntime" if kind == "path" else "runtime")
    _write_casimir_evidence(work, runtime)
    if kind == "table":
        (runtime / "data" / "realcg" / "realcg_é").write_text("non-ascii\n")
    result = subprocess.run(
        [
            "python", "-P", str(ROOT / "devtools" / "camcasp_reference.py"),
            "validate-casimir", "--work-dir", str(work), "--runtime", str(runtime),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "ASCII" in result.stderr


def test_dependency_free_stubbed_pipeline_requires_approval_before_json(tmp_path):
    reference = tmp_path / "reference"
    camcasp = tmp_path / "camcasp-bin"
    bin_dir = camcasp / "bin"
    bin_dir.mkdir(parents=True)
    calls = tmp_path / "calls.log"
    runcamcasp_invocation = tmp_path / "runcamcasp-invocation.log"
    absolute_clt_evidence = tmp_path / "absolute-clt-evidence.log"
    localize_polfile = tmp_path / "localize-polfile.log"
    psi4_received_input = tmp_path / "psi4-received.in"
    psi4_received_path = tmp_path / "psi4-received-path.log"
    localize_camcasp = tmp_path / "localize-camcasp.log"

    psi4_source = tmp_path / "psi4-source"
    psi4, psi4_commit = _make_psi4_checkout(psi4_source)
    orient_source = tmp_path / "orient-source"
    orient_commit = _make_git_checkout(
        orient_source,
        tracked_candidate=True,
        candidate_text=(
            "#!/usr/bin/env bash\n"
            "echo orient >>\"$STUB_CALLS\"\n"
        ),
    )
    orient = (
        orient_source
        / "x86-64"
        / "gfortran"
        / "exe"
        / "orient-5.0.10-ng"
    )
    for name in ("pfit", "casimir"):
        executable = bin_dir / name
        _write_executable(
            executable,
            f"#!/usr/bin/env bash\necho {name} >>\"$STUB_CALLS\"\n",
        )
    for name in ("camcasp", "cluster"):
        _write_executable(
            bin_dir / name,
            f"#!/usr/bin/env bash\necho {name} >>\"$STUB_CALLS\"\n",
        )
    _write_executable(
        bin_dir / "process",
        """#!/usr/bin/env bash
set -euo pipefail
input="$(cat)"
mapfile -t paths < <(printf '%s\n' "$input" | awk '$1 == "CamCASP" {print $2}')
[[ "${#paths[@]}" -eq 1 && "${paths[0]}" == "$CAMCASP" ]] || exit 87
echo process >>"$STUB_CALLS"
cat <<EOF
Title H2O ... H2O
Frequencies 0.5 10
Skip 0
CGdir ${paths[0]}/data/realcg
Dispersion 12 H2O
Finish
EOF
""",
    )

    runcamcasp = bin_dir / "runcamcasp.py"
    _write_executable(
        runcamcasp,
        """#!/usr/bin/env bash
set -euo pipefail
[[ ! -e "$CAMCASP/bin/no_psi4" ]] || exit 91
[[ "$0" == "$CAMCASP/bin/runcamcasp.py" ]] || exit 92
job_dir=""
clt_file=""
verbosity=""
while (( $# )); do
    case "$1" in
        --directory) job_dir="$2"; shift 2 ;;
        --clt) clt_file="$2"; shift 2 ;;
        --verbosity) verbosity="$2"; shift 2 ;;
        *) shift ;;
    esac
done
[[ "$verbosity" == 1 ]] || exit 98
printf 'basis = aug-cc-pvtz\nbasis type = None\nruntype = properties\nscfcode = psi4\nAC options: type = GRAC, join = TANH, p1 = 0.0, p2 = 0.0\n'
[[ -n "$clt_file" && -f "$clt_file" ]] || exit 95
printf 'cwd=%s\nclt=%s\n' "$(pwd -P)" "$clt_file" >"$STUB_RUNCAMCASP_INVOCATION"
mkdir -p "$job_dir/OUT"
cp "$clt_file" "$job_dir/H2O.clt"
cd "$job_dir"
cat "$clt_file" >/dev/null
if grep -Eiq '^[[:space:]]*(no[[:space:]]+)?localization[[:space:]]*$' H2O.clt; then
    exit 97
fi
cat >"$clt_file.clout" <<'EOF'
CLUSTER completed canonical H2O setup
EOF
if [[ ! -f "$job_dir/H2O.clt.clout" ]]; then
    cat H2O_A.in >/dev/null 2>&1 || exit 96
fi
printf 'generated\n' >H2O.ornt
printf 'generated\n' >H2O.prss
cat >H2O_casimir.prss <<EOF
TITLE PROCESS file to write the CASIMIR input
Set Global-data
  CamCASP $CAMCASP
  Units BOHR DEGREE
End
Read local pols for H2O
  Use ascii file {PREFIX}_0f10.pol
  Maximum rank {LIMIT}
  Limit rank to {HLIMIT} for sites H1 H2
  Frequencies STATIC + 10
End
Finish
EOF
printf 'generated\n' >H2O.sites
cat >H2O.cks <<'EOF'
SET Global_data
 XC-func PBE0
END
SET QUAD
 Type Gauss-Legendre
 Beta 0.5
END
SET NEW-PROP
 Kernel ALDA
END
SET PROPAGATOR
 Type CKS
END
SET NEW-PROP
 Kernel ALDA
END
SET PROPAGATOR
 Type CKS
END
BEGIN Polarizability
 Quad 10
 Rank 4
 Print pols for Orient
END
EOF
cat >H2O_A.in <<'EOF'
molecule {
  no_reorient
  no_com
  O 0 0 0
}
set {
  basis aug-cc-pvtz
}
energy, wfn = energy('PBE0', return_wfn=True)
EOF
echo camcasp >>"$STUB_CALLS"
mkdir -p "$SCRATCH/psi4-execution"
cp H2O_A.in "$SCRATCH/psi4-execution/H2O_A.in"
"$CAMCASP/bin/psi4.sh" "$SCRATCH/psi4-execution/H2O_A.in" H2O_A.out
printf 'format A polarizability\n' >OUT/H2O_NL4_fmtA.pol
cat >OUT/H2O_NL4_fmtB.pol <<'EOF'
"""
        + make_canonical_nl4_frequency_text()
        + """EOF
printf 'point to point\n' >OUT/H2O.p2p
""",
    )

    localize = bin_dir / "localize.py"
    _write_executable(
        localize,
        """#!/usr/bin/env bash
set -euo pipefail
[[ "$CAMCASP" != /* ]] || exit 90
[[ "$CAMCASP" != *[[:space:]]* ]] || exit 89
resolved_camcasp="$(realpath -e -- "$CAMCASP")"
printf 'value=%s\nresolved=%s\n' "$CAMCASP" "$resolved_camcasp" >"$STUB_LOCALIZE_CAMCASP"
[[ " $* " == *" --force loc refine disp "* ]] || exit 88
polfile=""
args=("$@")
for ((index = 0; index < ${#args[@]}; index++)); do
    if [[ "${args[index]}" == --polfile ]]; then
        polfile="${args[index + 1]}"
    fi
done
[[ "$polfile" == *_NL4_fmtB.pol ]] || exit 94
printf '%s\n' "$polfile" >"$STUB_LOCALIZE_POLFILE"
grep -Fqx "  CamCASP $CAMCASP" H2O_casimir.prss || exit 86
[[ "$(grep -Ec '^[[:space:]]*CamCASP[[:space:]]+' H2O_casimir.prss)" -eq 1 ]] || exit 85
orient </dev/null
pfit </dev/null
python -P - <<'PY'
from pathlib import Path
source = Path("H2O_casimir.prss").read_text()
Path("H2O_casimir.temp").write_text(
    source.format(PREFIX="H2O_ref_wt4_L3", LIMIT=3, HLIMIT=3)
)
PY
process <H2O_casimir.temp >H2O_ref_wt4_L3_casimir.data
casimir </dev/null
for index in $(seq -w 0 10); do
    printf 'ORIENT output\nFinished at 11:27:54 on 30 Jul 2026 \n' >"H2O_L3_0${index}.out"
    printf 'PFIT output\nFinished\n' >"H2O_ref_wt4_L3_0${index}.out"
done
cat >H2O.pdef <<'EOF'
Polarizabilities
  O O 10 10 = O_10_10_A
  H1 H1 10 10 = H1_10_10_A
  H2 H2 COPY H1 H1
End
EOF
cat >H2O_ref_wt4_L3_0f10.pol <<'EOF'
"""
        + make_real_l3_refined_text()
        + """EOF
awk '
/^# INDEX [0-9][0-9][0-9]$/ {
    tag = $3
    output = "H2O_ref_wt4_L3_" tag ".pol"
    next
}
output != "" { print > output }
' H2O_ref_wt4_L3_0f10.pol
cat >H2O_ref_wt4_L3_casimir.out <<'EOF'
Dispersion coefficients
"""
        + CASIMIR_C12
        + """EOF
cat >H2O_ref_wt4_L3_C12.pot <<'EOF'
"""
        + CASIMIR_C12
        + """EOF
""",
    )
    (camcasp / "VERSION").write_text("CamCASP VERSION 7.2.2 patch 003\n")
    (bin_dir / "no_psi4").write_bytes(b"")
    _write_realcg_tables(camcasp)
    archive_dir = camcasp / "x86-64" / "gfortran"
    archive_dir.mkdir(parents=True)
    for name in ("camcasp", "cluster", "process", "pfit", "casimir"):
        with gzip.open(archive_dir / f"{name}.gz", "wb") as handle:
            handle.write((bin_dir / name).read_bytes())
    subprocess.run(["git", "init", "-q", str(camcasp)], check=True)
    subprocess.run(
        ["git", "-C", str(camcasp), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(camcasp), "config", "user.name", "Test"],
        check=True,
    )
    subprocess.run(["git", "-C", str(camcasp), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(camcasp), "commit", "-qm", "stub tools"],
        check=True,
    )
    camcasp_commit = subprocess.check_output(
        ["git", "-C", str(camcasp), "rev-parse", "HEAD"], text=True
    ).strip()
    source_sentinel_digest = hashlib.sha256(
        (bin_dir / "no_psi4").read_bytes()
    ).hexdigest()

    stale_job = reference / "work" / "H2O"
    stale_job.mkdir(parents=True)
    (stale_job / "H2O_casimir.temp").write_text("stale temp\n")
    (stale_job / "H2O_ref_wt4_L3_casimir.data").write_text("stale data\n")
    (reference / "atomic-polarizabilities.json").write_text("stale json\n")
    (reference / "atomic-polarizabilities.json.sha256").write_text("stale digest\n")
    (reference / "work" / "manifest.json").write_text("stale manifest\n")
    stale_runtime = reference / "tools" / "camcasp-runtime"
    stale_runtime.mkdir(parents=True)
    (stale_runtime / "stale-product").write_text("stale\n")

    command = f'''source "{SCRIPT}"
REFERENCE_ROOT="$1"
CAMCASP="$2"
CAMCASP_COMMIT="$7"
PSI4_SOURCE_ROOT="$3"
PSI4_EXE="$PSI4_SOURCE_ROOT/build_camcasp/stage/bin/psi4"
ORIENT_EXE="$4/x86-64/gfortran/exe/orient-5.0.10-ng"
ORIENT_REF="$5"
export STUB_CALLS="$6"
export STUB_RUNCAMCASP_INVOCATION="$8"
export STUB_LOCALIZE_POLFILE="${{10}}"
export STUB_PSI4_RECEIVED_INPUT="${{11}}"
export STUB_PSI4_RECEIVED_PATH="${{12}}"
export STUB_LOCALIZE_CAMCASP="${{13}}"
preflight
bind_orient_checkout "$ORIENT_EXE"
export PSI4_EXE
prepare_layout
ORIENT_BIN_DIR="$REFERENCE_ROOT/tools/orient/bin"
mkdir -p "$ORIENT_BIN_DIR"
ln -s "$ORIENT_EXE" "$ORIENT_BIN_DIR/orient"
provision_camcasp
write_psi4_wrapper
write_camcasp_runtime_attestation
export_reference_environment
write_inputs
absolute_job="$REFERENCE_ROOT/work/absolute-clt-probe"
set +e
(
    cd "$REFERENCE_ROOT/inputs"
    "$CAMCASP/bin/runcamcasp.py" H2O \
        --clt "$REFERENCE_ROOT/inputs/H2O.clt" \
        --directory "$absolute_job" \
        --verbosity 1
)
absolute_rc=$?
set -e
[[ "$absolute_rc" -eq 96 ]]
[[ -s "$REFERENCE_ROOT/inputs/H2O.clt.clout" ]]
[[ ! -e "$absolute_job/H2O.clt.clout" ]]
[[ ! -e "$absolute_job/H2O.cks" ]]
[[ ! -e "$absolute_job/H2O_A.in" ]]
[[ ! -e "$absolute_job/H2O_A.dat" ]]
printf 'rc=%s\ninput_clout=yes\njob_cks=no\njob_psi4_inputs=no\n' \
    "$absolute_rc" >"$9"
rm -f "$REFERENCE_ROOT/inputs/H2O.clt.clout"
rm -rf "$absolute_job"
verify_camcasp_runtime_attestation
run_camcasp
attest_generated_protocol
run_localize
set +e
require_reviewed_pdef
review_rc=$?
set -e
[[ "$review_rc" -eq 78 ]]
[[ ! -e "$REFERENCE_ROOT/atomic-polarizabilities.json" ]]
[[ ! -e "$REFERENCE_ROOT/atomic-polarizabilities.json.sha256" ]]
[[ ! -e "$REFERENCE_ROOT/work/manifest.json" ]]
export CAMCASP_PDEF_SHA256="$(awk '{{print $1}}' "$REFERENCE_ROOT/work/H2O/H2O.pdef.sha256")"
require_reviewed_pdef
write_manifest
build_reference_json
'''
    result = subprocess.run(
        [
            "bash", "-c", command, "stub-pipeline", str(reference),
            str(camcasp), str(psi4_source), str(orient_source),
            orient_commit, str(calls), camcasp_commit,
            str(runcamcasp_invocation), str(absolute_clt_evidence),
            str(localize_polfile), str(psi4_received_input),
            str(psi4_received_path), str(localize_camcasp),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    document = json.loads(
        (reference / "atomic-polarizabilities.json").read_text()
    )
    validate_reference_document(document)
    runtime = reference / "tools" / "camcasp-runtime"
    source_sentinel = bin_dir / "no_psi4"
    assert source_sentinel.is_file()
    assert hashlib.sha256(source_sentinel.read_bytes()).hexdigest() == source_sentinel_digest
    assert subprocess.check_output(
        ["git", "-C", str(camcasp), "ls-files", "--error-unmatch", "bin/no_psi4"],
        text=True,
    ).strip() == "bin/no_psi4"
    assert subprocess.check_output(
        ["git", "-C", str(camcasp), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    ) == ""
    assert not (runtime / "bin" / "no_psi4").exists()
    assert not (runtime / "stale-product").exists()
    assert (runtime / "bin" / "psi4.sh").is_file()
    for name in ("camcasp", "cluster", "process", "pfit", "casimir"):
        assert (runtime / "x86-64" / "gfortran" / f"{name}.gz").read_bytes().startswith(
            b"\x1f\x8b"
        )
    assert calls.read_text().splitlines() == [
        "psi4", "camcasp", "psi4", "orient", "pfit", "process", "casimir"
    ]
    selected_format_b = reference / "work" / "H2O" / "OUT" / "H2O_NL4_fmtB.pol"
    assert localize_polfile.read_text().strip() == str(selected_format_b)
    assert localize_camcasp.read_text().splitlines() == [
        "value=../../tools/camcasp-runtime",
        f"resolved={runtime.resolve()}",
    ]
    assert absolute_clt_evidence.read_text().splitlines() == [
        "rc=96",
        "input_clout=yes",
        "job_cks=no",
        "job_psi4_inputs=no",
    ]
    assert runcamcasp_invocation.read_text().splitlines() == [
        f"cwd={(reference / 'inputs').resolve()}",
        "clt=H2O.clt",
    ]
    assert not (reference / "inputs" / "H2O.clt.clout").exists()
    camcasp_log = reference / "logs" / "camcasp.log"
    assert camcasp_log.with_name("camcasp.log.sha256").is_file()
    assert camcasp_log.read_text().count("AC options: type = GRAC") == 1
    assert "basis = aug-cc-pvtz" in camcasp_log.read_text()
    assert "basis type = None" in camcasp_log.read_text()
    assert "runtype = properties" in camcasp_log.read_text()
    assert "scfcode = psi4" in camcasp_log.read_text()
    assert "AC options" not in (reference / "work" / "H2O" / "H2O.clt.clout").read_text()
    assert (reference / "logs" / "localize-refine-dispersion.log.sha256").is_file()
    assert (reference / "atomic-polarizabilities.json.sha256").is_file()
    job_dir = reference / "work" / "H2O"
    template_lock = reference / "work" / ".H2O_casimir.template.lock"
    assert template_lock.is_file() and not template_lock.is_symlink()
    assert not template_lock.with_name(template_lock.name + ".sha256").exists()
    for representative in (
        job_dir / "OUT" / "H2O.p2p",
        job_dir / "OUT" / "H2O_NL4_fmtA.pol",
        job_dir / "OUT" / "H2O_NL4_fmtB.pol",
        job_dir / "H2O.ornt",
        job_dir / "H2O.cks",
        job_dir / "H2O_casimir.generated.prss",
        job_dir / "H2O_casimir.prss",
        job_dir / "H2O_casimir.temp",
        job_dir / "H2O_ref_wt4_L3_casimir.data",
        job_dir / "H2O_A.in",
        job_dir / "H2O_A.executed.in",
        job_dir / "H2O_A.out",
        job_dir / "H2O_ref_wt4_L3_010.pol",
    ):
        assert representative.with_name(representative.name + ".sha256").is_file()
    retained = [
        path
        for path in job_dir.rglob("*")
        if path.is_file() and not path.name.endswith(".sha256")
    ]
    assert retained
    assert not [
        path
        for path in retained
        if not path.with_name(path.name + ".sha256").is_file()
    ]
    assert set(document["tools"]["camcasp"]["executables"]) == {
        "camcasp", "cluster", "process", "pfit", "casimir"
    }
    assert document["tools"]["camcasp"]["commit"] == camcasp_commit
    assert all(
        record["path"].startswith(str(runtime.resolve()) + os.sep)
        for record in document["tools"]["camcasp"]["executables"].values()
    )
    assert document["sources"]["camcasp_no_psi4"]["path"] == str(
        source_sentinel.resolve()
    )
    assert document["sources"]["camcasp_psi4_wrapper"]["path"] == str(
        (runtime / "bin" / "psi4.sh").resolve()
    )
    assert (reference / "logs" / "camcasp-source.tar.sha256").is_file()
    assert (reference / "logs" / "camcasp-materialization.log.sha256").is_file()
    attestation = reference / "logs" / "camcasp-runtime-attestation.json"
    assert attestation.is_file()
    attestation_document = json.loads(attestation.read_text())
    assert set(attestation_document["realcg_files"]) == set(REALCG_BASENAMES)
    assert all(
        record["path"].startswith(str((runtime / "data" / "realcg").resolve()) + os.sep)
        for record in attestation_document["realcg_files"].values()
    )
    assert attestation.with_name(attestation.name + ".sha256").is_file()
    assert document["sources"]["camcasp_runtime_attestation"] == {
        "path": str(attestation.resolve()),
        "sha256": hashlib.sha256(attestation.read_bytes()).hexdigest(),
    }
    assert document["inputs"]["H2O.clt"]["path"] == str(
        (reference / "inputs" / "H2O.clt").resolve()
    )
    assert document["inputs"]["H2O.axes"]["path"] == str(
        (reference / "inputs" / "H2O.axes").resolve()
    )
    p2p = job_dir / "OUT" / "H2O.p2p"
    assert document["sources"]["p2p"] == {
        "path": str(p2p.resolve()),
        "sha256": hashlib.sha256(p2p.read_bytes()).hexdigest(),
    }
    generated_input = job_dir / "H2O_A.in"
    executed_input = job_dir / "H2O_A.executed.in"
    psi4_output = job_dir / "H2O_A.out"
    assert "symmetry" not in generated_input.read_text().lower()
    assert executed_input.read_text().count("symmetry c1") == 1
    assert "basis aug-cc-pvtz" in executed_input.read_text()
    assert "energy('PBE0'" in executed_input.read_text()
    assert "no_com" in executed_input.read_text()
    assert "no_reorient" in executed_input.read_text()
    assert psi4_received_input.read_bytes() == executed_input.read_bytes()
    assert psi4_received_path.read_text().strip() == str(
        (reference / "scratch" / "camcasp" / "psi4-execution" / "H2O_A.in").resolve()
    )
    assert "Running in c1 symmetry." in psi4_output.read_text()
    assert "=> Composite Functional: PBE0 <=" in psi4_output.read_text()
    generated_record = {
        "path": str(generated_input.resolve()),
        "sha256": hashlib.sha256(generated_input.read_bytes()).hexdigest(),
    }
    executed_record = {
        "path": str(executed_input.resolve()),
        "sha256": hashlib.sha256(executed_input.read_bytes()).hexdigest(),
    }
    assert document["sources"]["psi4_generated_input"] == generated_record
    assert document["sources"]["psi4_executed_input"] == executed_record
    assert document["sources"]["psi4_input"] == executed_record
    assert document["sources"]["psi4_output"] == {
        "path": str(psi4_output.resolve()),
        "sha256": hashlib.sha256(psi4_output.read_bytes()).hexdigest(),
    }
    casimir_process_generated = job_dir / "H2O_casimir.generated.prss"
    casimir_process_template = job_dir / "H2O_casimir.prss"
    casimir_process_input = job_dir / "H2O_casimir.temp"
    casimir_data = job_dir / "H2O_ref_wt4_L3_casimir.data"
    assert f"CamCASP {runtime.resolve()}" in casimir_process_generated.read_text()
    assert "CamCASP ../../tools/camcasp-runtime" in casimir_process_template.read_text()
    assert "CamCASP ../../tools/camcasp-runtime" in casimir_process_input.read_text()
    assert casimir_process_input.read_text() == casimir_process_template.read_text().format(
        PREFIX="H2O_ref_wt4_L3", LIMIT=3, HLIMIT=3
    )
    assert "stale" not in casimir_process_input.read_text()
    assert "stale" not in casimir_data.read_text()
    assert "CGdir ../../tools/camcasp-runtime/data/realcg" in casimir_data.read_text()
    assert "Frequencies 0.5 10" in casimir_data.read_text()
    assert "Skip 0" in casimir_data.read_text()
    assert "Dispersion 12 H2O" in casimir_data.read_text()
    for source_name, source_path in (
        ("casimir_process_generated", casimir_process_generated),
        ("casimir_process_template", casimir_process_template),
        ("casimir_process_input", casimir_process_input),
        ("casimir_input", casimir_data),
    ):
        assert document["sources"][source_name] == {
            "path": str(source_path.resolve()),
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        }
    cluster_output = job_dir / "H2O.clt.clout"
    assert document["sources"]["cluster_output"] == {
        "path": str(cluster_output.resolve()),
        "sha256": hashlib.sha256(cluster_output.read_bytes()).hexdigest(),
    }
    format_a = job_dir / "OUT" / "H2O_NL4_fmtA.pol"
    format_b = job_dir / "OUT" / "H2O_NL4_fmtB.pol"
    assert document["sources"]["nonlocal_pol_format_a"] == {
        "path": str(format_a.resolve()),
        "sha256": hashlib.sha256(format_a.read_bytes()).hexdigest(),
    }
    assert document["sources"]["nonlocal_pol"] == {
        "path": str(format_b.resolve()),
        "sha256": hashlib.sha256(format_b.read_bytes()).hexdigest(),
    }
    assert document["tools"]["orient"]["version"] == "5.0.10-ng"
    assert document["tools"]["orient"]["commit"] == orient_commit
    assert document["tools"]["orient"]["executable"] == {
        "path": str(orient.resolve()),
        "sha256": hashlib.sha256(orient.read_bytes()).hexdigest(),
    }
    assert document["tools"]["psi4"]["commit"] == psi4_commit
    assert document["tools"]["psi4"]["dirty"] is False
    assert document["tools"]["psi4"]["executable"] == {
        "path": str(psi4.resolve()),
        "sha256": hashlib.sha256(psi4.read_bytes()).hexdigest(),
    }


def test_run_camcasp_rejects_input_directory_clout(tmp_path):
    reference = tmp_path / "reference"
    inputs = reference / "inputs"
    runtime_bin = reference / "tools" / "camcasp-runtime" / "bin"
    inputs.mkdir(parents=True)
    runtime_bin.mkdir(parents=True)
    (inputs / "H2O.clt").write_text("Finish\n")
    (inputs / "H2O.axes").write_text("Axes\nEnd\n")
    runner = runtime_bin / "runcamcasp.py"
    _write_executable(
        runner,
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "[[ \"$(pwd -P)\" == \"$EXPECTED_INPUTS\" ]]\n"
        "[[ \" $* \" == *\" --clt H2O.clt \"* ]]\n"
        ": >H2O.clt.clout\n",
    )
    command = (
        f'source "{SCRIPT}"; REFERENCE_ROOT="$1"; CAMCASP="$2"; '
        'SCRATCH="$REFERENCE_ROOT/scratch/camcasp"; run_camcasp'
    )
    result = subprocess.run(
        ["bash", "-c", command, "stray-clout", str(reference), str(runtime_bin.parent)],
        cwd=ROOT,
        env={**os.environ, "EXPECTED_INPUTS": str(inputs.resolve())},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "unexpected input-directory CamCASP output" in result.stderr
    assert (inputs / "H2O.clt.clout").is_file()
    assert (reference / "logs" / "camcasp.log").is_file()
    assert (reference / "logs" / "camcasp.log.sha256").is_file()


def test_run_camcasp_subshell_preserves_distinctive_failure(tmp_path):
    source = tmp_path / "camcasp-source"
    commit = _make_attestable_camcasp_source(
        source,
        runcamcasp_text=(
            "#!/usr/bin/env bash\n"
            "echo distinctive-runcamcasp-failure\n"
            "exit 73\n"
        ),
    )
    reference = tmp_path / "reference"
    inputs = reference / "inputs"
    inputs.mkdir(parents=True)
    (inputs / "H2O.clt").write_text("Finish\n")
    (inputs / "H2O.axes").write_text("Axes\nEnd\n")
    psi4 = tmp_path / "psi4"
    _write_executable(psi4, "#!/usr/bin/env bash\necho 'Psi4 stub'\n")
    command = f'''source "{SCRIPT}"
REFERENCE_ROOT="$1"
CAMCASP_SOURCE_ROOT="$2"
CAMCASP_COMMIT="$3"
CAMCASP="$REFERENCE_ROOT/tools/camcasp-runtime"
PSI4_EXE="$4"
SCRATCH="$REFERENCE_ROOT/scratch/camcasp"
provision_camcasp
write_psi4_wrapper
write_camcasp_runtime_attestation
verify_camcasp_runtime_attestation
set +e
run_camcasp
rc=$?
set -e
exit "$rc"
'''
    result = subprocess.run(
        [
            "bash", "-c", command, "runcamcasp-failure", str(reference),
            str(source), commit, str(psi4),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 73
    log = reference / "logs" / "camcasp.log"
    checksum = log.with_name(log.name + ".sha256")
    assert "distinctive-runcamcasp-failure" in log.read_text()
    assert checksum.is_file()
    assert checksum.read_text().split()[0] == hashlib.sha256(log.read_bytes()).hexdigest()
    job = reference / "work" / "H2O"
    assert not (job / "H2O.clt.clout").exists()
    assert not (job / "H2O.cks").exists()
    assert not (job / "H2O_A.in").exists()
    assert not (job / "H2O_A.dat").exists()


def test_clean_room_removes_stale_publication_before_unapproved_gate(tmp_path):
    reference = tmp_path / "reference"
    for directory in ("inputs", "work/H2O", "scratch", "logs", "tools"):
        (reference / directory).mkdir(parents=True, exist_ok=True)
    (reference / "atomic-polarizabilities.json").write_text("stale json\n")
    (reference / "atomic-polarizabilities.json.sha256").write_text("stale digest\n")
    (reference / "work" / "manifest.json").write_text("stale manifest\n")
    (reference / "inputs" / "stale.clt").write_text("stale\n")
    (reference / "logs" / "stale.log").write_text("stale\n")
    (reference / "scratch" / "stale.tmp").write_text("stale\n")
    (reference / "tools" / "preserved").write_text("tool\n")
    command = f'''source "{SCRIPT}"
REFERENCE_ROOT="$1"
prepare_layout
mkdir -p "$REFERENCE_ROOT/work/H2O"
cat >"$REFERENCE_ROOT/work/H2O/H2O.pdef" <<'EOF'
Polarizabilities
  H1 H1 10 10 = H1_A
  H2 H2 COPY H1 H1
End
EOF
CAMCASP_PDEF_SHA256={'0' * 64} require_reviewed_pdef
'''
    result = subprocess.run(
        ["bash", "-c", command, "clean-room", str(reference)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 78
    assert not (reference / "atomic-polarizabilities.json").exists()
    assert not (reference / "atomic-polarizabilities.json.sha256").exists()
    assert not (reference / "work" / "manifest.json").exists()
    assert not (reference / "inputs" / "stale.clt").exists()
    assert not (reference / "logs" / "stale.log").exists()
    assert not (reference / "scratch" / "stale.tmp").exists()
    assert (reference / "tools" / "preserved").read_text() == "tool\n"


def test_install_pfit_preserves_compressed_archive(tmp_path):
    camcasp = tmp_path / "camcasp-bin"
    archive_dir = camcasp / "x86-64" / "gfortran"
    archive_dir.mkdir(parents=True)
    (camcasp / "bin").mkdir()
    archive = archive_dir / "pfit.gz"
    payload = b"#!/usr/bin/env bash\necho pfit\n"
    with gzip.open(archive, "wb") as handle:
        handle.write(payload)
    result = subprocess.run(
        [
            "bash", "-c",
            f'source "{SCRIPT}"; CAMCASP="$1"; install_camcasp_program pfit',
            "pfit-install", str(camcasp),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert archive.read_bytes().startswith(b"\x1f\x8b")
    assert (archive_dir / "exe" / "pfit").read_bytes() == payload
