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
                    frequency_index + atom_index + (row + column) / 100.0
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
                "version": "5.0.11-ng",
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
            "H2O.clt": {"sha256": "d" * 64},
            "H2O.axes": {"sha256": "e" * 64},
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
    return clt, cks, cluster_log, psi4_input


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
        (0, "Localization", "No Localization", "Localization"),
        (1, "Type Gauss-Legendre", "Type Euler-Maclaurin", "Type"),
        (1, "Beta 0.5", "Beta 0.7", "Beta"),
        (1, "Quad 10", "Quad 9", "Quad"),
        (1, "Rank 4", "Rank 3", "Rank"),
        (1, "Print pols for Orient", "Print pols for Molpro", "Print"),
        (2, "type = GRAC", "type = LB94", "AC options"),
        (2, "functional PBE0", "functional PBE", "functional"),
        (2, "kernel = ALDA+CHF", "kernel = ALDA", "kernel"),
        (3, "symmetry c1", "symmetry c2v", "symmetry"),
        (3, "no_com", "com", "no_com"),
        (3, "no_reorient", "reorient", "no_reorient"),
    )
    for text_index, old, new, expected in mutations:
        texts = list(canonical_generated_protocol_texts())
        texts[text_index] = texts[text_index].replace(old, new)
        _assert_protocol_rejected(texts, expected)


def _refined_block_text(combined_text, index):
    marker = f"# INDEX {index:03d}\n"
    block = combined_text.split(marker, 1)[1]
    if index < 10:
        block = block.split(f"# INDEX {index + 1:03d}\n", 1)[0]
    return block.strip() + "\n"


def populate_stage_artifacts(work, job="H2O"):
    combined = make_l3_refined_text()
    for index in range(11):
        (work / f"{job}_L3_{index:03d}.out").write_text(
            "ORIENT localization output\nFinished\n"
        )
        (work / f"{job}_ref_wt4_L3_{index:03d}.out").write_text(
            "PFIT refinement output\nFinished\n"
        )
        (work / f"{job}_ref_wt4_L3_{index:03d}.pol").write_text(
            _refined_block_text(combined, index)
        )
    (work / f"{job}_ref_wt4_L3_0f10.pol").write_text(combined)
    (work / f"{job}_ref_wt4_L3_casimir.out").write_text(
        "Dispersion coefficients\n"
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
    refined.write_text(make_l3_refined_text())
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


def test_dependency_free_stubbed_pipeline_requires_approval_before_json(tmp_path):
    reference = tmp_path / "reference"
    camcasp = tmp_path / "camcasp-bin"
    bin_dir = camcasp / "bin"
    bin_dir.mkdir(parents=True)
    calls = tmp_path / "calls.log"

    psi4 = tmp_path / "psi4"
    orient = tmp_path / "orient"
    _write_executable(
        psi4,
        "#!/usr/bin/env bash\n"
        "echo psi4 >>\"$STUB_CALLS\"\n"
        "echo 'stub Psi4 1.0'\n",
    )
    for name in ("orient", "pfit", "casimir"):
        executable = orient if name == "orient" else bin_dir / name
        _write_executable(
            executable,
            f"#!/usr/bin/env bash\necho {name} >>\"$STUB_CALLS\"\n",
        )
    for name in ("camcasp", "cluster", "process"):
        _write_executable(
            bin_dir / name,
            f"#!/usr/bin/env bash\necho {name} >>\"$STUB_CALLS\"\n",
        )

    runcamcasp = bin_dir / "runcamcasp.py"
    _write_executable(
        runcamcasp,
        """#!/usr/bin/env bash
set -euo pipefail
echo camcasp >>"$STUB_CALLS"
"$PSI4_EXE" --stub-stage >/dev/null
job_dir=""
while (( $# )); do
    if [[ "$1" == "--directory" ]]; then job_dir="$2"; shift 2; else shift; fi
done
mkdir -p "$job_dir/OUT"
printf 'generated\n' >"$job_dir/H2O.ornt"
printf 'generated\n' >"$job_dir/H2O.prss"
printf 'generated\n' >"$job_dir/H2O_casimir.prss"
printf 'generated\n' >"$job_dir/H2O.sites"
cat >"$job_dir/H2O.cks" <<'EOF'
SET QUAD
 Type Gauss-Legendre
 Beta 0.5
END
BEGIN Polarizability
 Quad 10
 Rank 4
 Print pols for Orient
END
EOF
cat >"$job_dir/H2O.clt.clout" <<'EOF'
AC options: type = GRAC
functional PBE0
kernel = ALDA+CHF
EOF
cat >"$job_dir/H2O_A.in" <<'EOF'
symmetry c1
no_com
no_reorient
EOF
cat >"$job_dir/OUT/H2O_NL4_fmtB.pol" <<'EOF'
"""
        + make_canonical_nl4_frequency_text()
        + """EOF
printf 'point to point\n' >"$job_dir/OUT/H2O.p2p"
""",
    )

    localize = bin_dir / "localize.py"
    _write_executable(
        localize,
        """#!/usr/bin/env bash
set -euo pipefail
orient </dev/null
pfit </dev/null
casimir </dev/null
for index in $(seq -w 0 10); do
    printf 'ORIENT output\nFinished\n' >"H2O_L3_0${index}.out"
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
        + make_l3_refined_text()
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
EOF
cat >H2O_ref_wt4_L3_C12.pot <<'EOF'
"""
        + CASIMIR_C12
        + """EOF
""",
    )
    (camcasp / "VERSION").write_text("CamCASP VERSION 7.2.2 patch 003\n")
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

    (reference / "work").mkdir(parents=True)
    (reference / "atomic-polarizabilities.json").write_text("stale json\n")
    (reference / "atomic-polarizabilities.json.sha256").write_text("stale digest\n")
    (reference / "work" / "manifest.json").write_text("stale manifest\n")

    command = f'''source "{SCRIPT}"
REFERENCE_ROOT="$1"
CAMCASP="$2"
PSI4_EXE="$3"
ORIENT_EXE="$4"
ORIENT_REF=d8d861098c8f548e2cf230c387c8431d9418650a
PATH="$(dirname "$ORIENT_EXE"):$CAMCASP/bin:$PATH"
export PSI4_EXE STUB_CALLS="$5"
prepare_layout
export SCRATCH="$REFERENCE_ROOT/scratch/camcasp"
write_inputs
mkdir -p "$REFERENCE_ROOT/logs"
"$PSI4_EXE" --version >"$REFERENCE_ROOT/logs/psi4-version.log"
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
            str(camcasp), str(psi4), str(orient), str(calls),
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
    assert calls.read_text().splitlines() == [
        "psi4", "camcasp", "psi4", "orient", "pfit", "casimir"
    ]
    assert (reference / "logs" / "camcasp.log.sha256").is_file()
    assert (reference / "logs" / "localize-refine-dispersion.log.sha256").is_file()
    assert (reference / "atomic-polarizabilities.json.sha256").is_file()
    job_dir = reference / "work" / "H2O"
    for representative in (
        job_dir / "OUT" / "H2O.p2p",
        job_dir / "OUT" / "H2O_NL4_fmtB.pol",
        job_dir / "H2O.ornt",
        job_dir / "H2O.cks",
        job_dir / "H2O_A.in",
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
