"""Tiny fake artifacts only: no deployment/build/environment/GPU operation."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


def module():
    path = Path(__file__).resolve().parents[2] / "devtools/disp0/artifact_deploy.py"
    spec = importlib.util.spec_from_file_location("artifact_deploy", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def fixture_source(tmp_path):
    source = tmp_path / "source"
    files = {
        "psi4-build/stage/lib/psi4/core.cpython-313-x86_64-linux-gnu.so": b"fake-psi4",
        "psi4-build/stage/lib/psi4/__init__.py": b"# fake package\n",
        "psi4-build/stage/share/psi4/basis/fake.gbs": b"fake-data",
        "einsums-build/lib/einsums/__init__.py": b"# fake e2\n",
        "einsums-build/lib/einsums/_core.cpython-313-x86_64-linux-gnu.so": b"fake-pycore",
        "einsums-build/lib/libEinsums-abi1.so.2.0.0": b"fake-native",
    }
    for name, data in files.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    lib = source / "einsums-build/lib"
    (lib / "libEinsums-abi1.so.1").symlink_to("libEinsums-abi1.so.2.0.0")
    (lib / "libEinsums-abi1.so").symlink_to("libEinsums-abi1.so.1")
    return source, files


def test_whole_artifacts_copy_unchanged_into_absent_prefix(tmp_path):
    deploy = module()
    source, original = fixture_source(tmp_path)
    destination = tmp_path / "new"
    deadline = time.monotonic() + 10
    records = deploy.inventory(source, destination, deadline)
    deploy.install(records, destination, deadline)
    assert all((source / name).read_bytes() == data for name, data in original.items())
    for relative, record in records.items():
        path = destination / relative
        if record["kind"] == "file":
            assert deploy.digest(path, deadline) == record["sha256"]
        elif record["kind"] == "symlink":
            assert os.readlink(path) == record["target"]
    assert (destination / "psi4-build/stage/share/psi4/basis/fake.gbs").read_bytes() == b"fake-data"
    with pytest.raises(FileExistsError):
        deploy.install(records, destination, deadline)


@pytest.mark.parametrize("kind", ["absolute", "escape"])
def test_inventory_rejects_external_symlink_before_copy(tmp_path, kind):
    deploy = module()
    source, _ = fixture_source(tmp_path)
    outside = tmp_path / "outside"
    outside.write_text("not an approved artifact")
    link = source / "einsums-build/lib/einsums/external"
    link.symlink_to(outside if kind == "absolute" else "../../../../outside")
    destination = tmp_path / "new"
    with pytest.raises(AssertionError, match="symlink|escapes"):
        deploy.inventory(source, destination, time.monotonic() + 10)
    assert not destination.exists()


def test_copy_detects_source_tamper_and_preserves_partial_prefix(tmp_path):
    deploy = module()
    source, _ = fixture_source(tmp_path)
    destination = tmp_path / "new"
    deadline = time.monotonic() + 10
    records = deploy.inventory(source, destination, deadline)
    (source / "einsums-build/lib/einsums/__init__.py").write_text("tampered")
    with pytest.raises(AssertionError):
        deploy.install(records, destination, deadline)
    assert destination.is_dir()


def test_expired_preflight_never_creates_prefix(tmp_path):
    deploy = module()
    source, _ = fixture_source(tmp_path)
    destination = tmp_path / "new"
    with pytest.raises(TimeoutError):
        deploy.inventory(source, destination, time.monotonic() - 1)
    assert not destination.exists()


def test_optimized_python_rejected_before_deployment():
    path = Path(__file__).resolve().parents[2] / "devtools/disp0/artifact_deploy.py"
    result = subprocess.run([sys.executable, "-O", str(path), "--help"],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode != 0 and "optimized Python is forbidden" in result.stderr


def test_symlinked_directory_root_is_rejected_before_writes(tmp_path):
    deploy = module()
    source, _ = fixture_source(tmp_path)
    stage = source / "psi4-build/stage"
    stage.rename(source / "moved-stage")
    stage.symlink_to("../moved-stage", target_is_directory=True)
    destination = tmp_path / "new"
    with pytest.raises(AssertionError, match="Symlinked artifact root"):
        deploy.inventory(source, destination, time.monotonic() + 10)
    assert not destination.exists()


def test_internal_source_link_that_relocates_into_old_source_is_rejected(tmp_path):
    deploy = module()
    source, _ = fixture_source(tmp_path)
    link = source / "einsums-build/lib/einsums/old-stage"
    link.symlink_to("../../../../source/psi4-build/stage", target_is_directory=True)
    assert link.resolve().is_relative_to(source)
    destination = tmp_path / "new"
    with pytest.raises(AssertionError, match="Relocated link escapes"):
        deploy.inventory(source, destination, time.monotonic() + 10)
    assert not destination.exists()


def test_destination_race_does_not_overwrite_foreign_manifest(tmp_path):
    deploy = module()
    source, _ = fixture_source(tmp_path)
    destination = tmp_path / "new"
    deploy.inventory(source, destination, time.monotonic() + 10)
    destination.mkdir()  # concurrent creator after inventory
    manifest = destination / "deployment-manifest.json"
    manifest.write_text("preserve original deployment")
    ownership = None
    try:
        with pytest.raises(FileExistsError):
            ownership = deploy.create_prefix(destination)
    finally:
        assert not deploy.write_owned_manifest(destination, {"status": "FAILED"}, ownership)
    assert manifest.read_text() == "preserve original deployment"


def test_manifest_rejects_replaced_owned_prefix(tmp_path):
    deploy = module()
    destination = tmp_path / "new"
    ownership = deploy.create_prefix(destination)
    destination.rename(tmp_path / "original")
    destination.mkdir()
    with pytest.raises(AssertionError, match="ownership changed"):
        deploy.write_owned_manifest(destination, {}, ownership)
    assert not list(destination.iterdir())


@pytest.mark.parametrize("state", ["tracked", "untracked", "modified"])
def test_recipe_requires_its_own_pinned_git_blob(tmp_path, state):
    deploy = module()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args],
                                       text=True, stderr=subprocess.STDOUT).strip()
    git("init")
    script = tmp_path / "recipe.py"
    script.write_text("# approved fixture\n")
    git("add", "recipe.py")
    git("-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
        "commit", "-m", "fixture")
    pin = git("rev-parse", "HEAD")
    if state == "tracked":
        deploy.verify_recipe(tmp_path, script, pin)
    elif state == "untracked":
        script = tmp_path / "untracked.py"
        script.write_text("# not in the pinned recipe\n")
        with pytest.raises(subprocess.CalledProcessError):
            deploy.verify_recipe(tmp_path, script, pin)
    else:
        script.write_text("# changed fixture\n")
        with pytest.raises(AssertionError, match="pinned Git blob"):
            deploy.verify_recipe(tmp_path, script, pin)


def test_finalization_has_separate_hard_deadline(tmp_path):
    script = Path(__file__).resolve().parents[2] / "devtools/disp0/artifact_deploy.py"
    code = """
import importlib.util, pathlib, sys, time
spec=importlib.util.spec_from_file_location('deploy',sys.argv[1])
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
destination=pathlib.Path(sys.argv[2]); ownership=m.create_prefix(destination)
m.write_owned_manifest=lambda *args: time.sleep(30)
m.finalize_manifest(destination,{},ownership,timeout_seconds=0.1)
"""
    result = subprocess.run([sys.executable, "-c", code, str(script), str(tmp_path / "partial")],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 124
    assert (tmp_path / "partial").is_dir()


def test_prefix_replacement_after_copy_check_cannot_redirect_writes(tmp_path, monkeypatch):
    deploy = module()
    source, original = fixture_source(tmp_path)
    destination = tmp_path / "new"
    moved = tmp_path / "owned-moved"
    deadline = time.monotonic() + 10
    records = deploy.inventory(source, destination, deadline)
    ownership = deploy.create_prefix(destination)
    verify = deploy.verify_owned_prefix
    first = True

    def replace_after_verification(path, token):
        nonlocal first
        verify(path, token)
        if first:
            first = False
            destination.rename(moved)
            destination.symlink_to(source, target_is_directory=True)

    monkeypatch.setattr(deploy, "verify_owned_prefix", replace_after_verification)
    with pytest.raises(AssertionError):
        deploy.install(records, destination, deadline, ownership=ownership)
    assert all((source / name).read_bytes() == data for name, data in original.items())
    assert not (source / "einsums-stage").exists()
    assert (moved / "einsums-stage/lib/einsums/__init__.py").is_file()


def test_prefix_replacement_after_manifest_check_cannot_redirect_open(tmp_path, monkeypatch):
    deploy = module()
    destination = tmp_path / "new"
    ownership = deploy.create_prefix(destination)
    moved, foreign = tmp_path / "moved", tmp_path / "foreign"
    foreign.mkdir()
    manifest = foreign / "deployment-manifest.json"
    manifest.write_text("preserve foreign manifest")
    verify = deploy.verify_owned_prefix
    first = True

    def replace_after_verification(path, token):
        nonlocal first
        verify(path, token)
        if first:
            first = False
            destination.rename(moved)
            destination.symlink_to(foreign, target_is_directory=True)

    monkeypatch.setattr(deploy, "verify_owned_prefix", replace_after_verification)
    with pytest.raises(AssertionError):
        deploy.write_owned_manifest(destination, {"status": "FAILED"}, ownership)
    assert manifest.read_text() == "preserve foreign manifest"
    assert (moved / "deployment-manifest.json").is_file()


def test_whole_process_watchdog_covers_blocked_final_output(tmp_path):
    script = Path(__file__).resolve().parents[2] / "devtools/disp0/artifact_deploy.py"
    code = """
import importlib.util, sys
spec=importlib.util.spec_from_file_location('deploy',sys.argv[1])
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
def blocked_output():
    sys.stdout.write('x' * (1024 * 1024))
    sys.stdout.flush()
m.run_bounded_worker(blocked_output, timeout_seconds=0.2)
"""
    process = subprocess.Popen([sys.executable, "-c", code, str(script)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        # Intentionally do not drain stdout until the watchdog terminates it.
        assert process.wait(timeout=5) == 124
        process.communicate(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
