import gzip
import hashlib
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
        cwd=ROOT,
        env={**os.environ, "FAKE_ARGS_LOG": str(arguments)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert arguments.read_text().splitlines() == [
        "--version",
        "serial.in serial.out",
        "-n 4 parallel.in parallel.out",
    ]
    version_log = reference_root / "logs" / "psi4-version.log"
    assert "fake-psi4" in version_log.read_text()
    assert version_log.with_name("psi4-version.log.sha256").is_file()


def _make_git_checkout(path, tracked_candidate):
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"], check=True
    )
    (path / "README").write_text("orient\n")
    if tracked_candidate:
        candidate = path / "x86-64" / "gfortran" / "exe" / "orient-5.0.11-ng"
        candidate.parent.mkdir(parents=True)
        candidate.write_text("#!/usr/bin/env bash\n")
        candidate.chmod(0o755)
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "fixture"], check=True)
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


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
    candidate = checkout / "x86-64" / "gfortran" / "exe" / "orient-5.0.11-ng"
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
    assert "Orient checkout is not clean" in result.stderr


def test_verify_orient_checkout_rejects_unrelated_untracked_file(tmp_path):
    checkout = tmp_path / "orient"
    commit = _make_git_checkout(checkout, tracked_candidate=True)
    (checkout / "unrelated.tmp").write_text("untracked\n")
    candidate = checkout / "x86-64" / "gfortran" / "exe" / "orient-5.0.11-ng"
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
    assert "Orient checkout is not clean" in result.stderr


def test_verify_orient_checkout_rejects_untracked_candidate(tmp_path):
    checkout = tmp_path / "orient"
    commit = _make_git_checkout(checkout, tracked_candidate=False)
    candidate = checkout / "x86-64" / "gfortran" / "exe" / "orient-5.0.11-ng"
    candidate.parent.mkdir(parents=True)
    candidate.write_text("#!/usr/bin/env bash\n")
    candidate.chmod(0o755)
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
    assert "expected tracked Orient artifact" in result.stderr
