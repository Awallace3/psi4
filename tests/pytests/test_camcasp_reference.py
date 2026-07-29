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
