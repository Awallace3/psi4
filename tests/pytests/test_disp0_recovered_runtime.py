"""Recovery admission fixtures only; no real build, stage or device access."""

import importlib.util
import json
import os
from pathlib import Path
import time

import pytest


def load(name):
    path = Path(__file__).resolve().parents[2] / "devtools/disp0" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def recovery(tmp_path, monkeypatch):
    admission, installer = load("recovered_runtime"), load("artifact_deploy")
    origin = tmp_path / "runs" / f"build-{admission.COMPILE_PIN}-{admission.JOB}"
    anchor = tmp_path / "private-recipe"
    anchor.mkdir(mode=0o700)
    prefix = anchor / "runtime-private" / "image"
    prefix.mkdir(parents=True)
    science = tmp_path / "science"
    science.mkdir()
    for relative in admission.EXPECTED:
        source_relative = relative.replace("einsums-stage/lib/einsums/", "einsums-build/lib/einsums/")
        source_relative = source_relative.replace("einsums-stage/lib/libEinsums", "einsums-build/lib/libEinsums")
        path = origin / source_relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    for name in admission.MODULES:
        relative = Path("psi4/driver/procrouting/sapt") / name
        source = science / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"# complete fixture module {name}\n")
        staged = origin / "psi4-build/stage/lib" / relative
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_bytes(source.read_bytes())
    (origin / "einsums-build/lib/einsums/__init__.py").write_text("# complete fake package\n")
    (origin / "psi4-build/stage/lib/psi4/__init__.py").write_text("# complete fake stage\n")
    native = origin / "einsums-build/lib"
    (native / "libEinsums-abi1.so.1").symlink_to("libEinsums-abi1.so.2.0.0")
    (native / "libEinsums-abi1.so").symlink_to("libEinsums-abi1.so.1")
    records = installer.inventory(origin, prefix, time.monotonic() + 5)
    # Use a separately acquired absent fixture prefix, never a real deployment.
    prefix.rmdir()
    fd = os.open(prefix.parent, os.O_RDONLY | os.O_DIRECTORY)
    prefix.parent.chmod(0o700)
    try:
        owner = installer.create_prefix(prefix, private_parent_fd=fd)
    finally:
        os.close(fd)
    installer.install(records, prefix, time.monotonic() + 5, owner)
    expected = {relative: records[relative]["sha256"] for relative in admission.EXPECTED}
    manifest = {
        "status": "ARTIFACT_BYTES_VERIFIED", "recipe_pin": admission.RECIPE_PIN,
        "source_run": str(origin), "source_job_status": "FAILED; never relabelled",
        "destination": str(prefix), "private_recipe_anchor": str(anchor),
        "selected_binary_sha256": expected, "files_and_links": records,
        "artifact_bytes": sum(r.get("bytes", 0) for r in records.values()),
        "abi": {name: {"origin": ["fixture ABI"], "destination": ["fixture ABI"]}
                for name in expected},
    }
    manifest_path = prefix / "deployment-manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    logs = origin.parent / "logs"
    logs.mkdir()
    status = logs / f"{admission.JOB}.status"
    status.write_text(f"exit_code=1\nterm_received=0\nrun={origin}\nFAILED_OR_SIGNALLED\n")
    proofs = prefix / "recovery-proofs"
    proofs.mkdir()
    native_path = prefix / "einsums-stage/lib/libEinsums-abi1.so.2.0.0"
    imported = {"psi4": str(prefix / "psi4-build/stage/lib/psi4/__init__.py"),
                "einsums": str(prefix / "einsums-stage/lib/einsums/__init__.py")}
    metadata = {
        **imported, "status": "PASS",
        "world": {"git_commit": admission.E2_PIN, "version_major": 2,
                  "compiler_major": 14, "library_path": str(native_path)},
        "binary_sha256": {str(prefix / name): value for name, value in expected.items()},
        "mapped_libraries": [str(native_path), "/cuda/lib/libcudart.so.13",
                             "/cuda/lib/libcublas.so.13"],
    }
    api = {**imported, "status": "PASS", "checks": sorted(admission.CHECKS),
           "tolerances": {"rtol": 1e-11, "atol": 1e-12}}
    (proofs / "runtime-provenance.json").write_text(json.dumps(metadata))
    (proofs / "api-probe.json").write_text(json.dumps(api))
    monkeypatch.setattr(admission, "ORIGIN", origin)
    monkeypatch.setattr(admission, "EXPECTED", expected)
    monkeypatch.setattr(admission, "verify_sources", lambda *args: None)
    monkeypatch.setattr(admission, "compilation_provenance", lambda: {"fixture": "not real provenance"})
    monkeypatch.setattr(admission, "git", lambda *args: admission.RECIPE_PIN if "rev-parse" in args else "")
    return admission, prefix, science, anchor, manifest, metadata, api, status


def validate(fixture):
    module, prefix, science, anchor, *_ = fixture
    return module.validate(prefix, science, anchor, "a" * 40)


def test_recovery_binds_failed_origin_fresh_proofs_and_science(recovery):
    result = validate(recovery)
    assert result["status"] == "RECOVERED_RUNTIME_PREPARATION_ONLY"
    assert result["origin_compilation_job"] == "14196330"
    assert result["actual_gpu_correctness_verified"] is False
    assert "exit_code=1" in result["original_failed_status"]["contents"]
    assert len(result["imported_scientific_modules"]) == 4
    assert set(result["new_prefix_proofs"]) == {"runtime-provenance.json", "api-probe.json"}


@pytest.mark.parametrize("mutation", ["relabel", "wrong-recipe", "partial", "changed-file",
                                    "wrong-proof-prefix", "mixed-native", "old-proof",
                                    "missing-api", "changed-science", "wrong-binary",
                                    "extra-module", "extra-library", "changed-origin",
                                    "changed-origin-link"])
def test_recovery_rejects_incomplete_or_foreign_evidence(recovery, mutation):
    module, prefix, science, anchor, manifest, metadata, api, status = recovery
    if mutation == "relabel":
        status.write_text("exit_code=0\nterm_received=0\nCPU_PREPARATION_EXIT_ZERO\n")
    elif mutation == "wrong-recipe":
        manifest["recipe_pin"] = "0" * 40
    elif mutation == "partial":
        manifest["files_and_links"].pop("einsums-stage/lib/einsums/__init__.py")
    elif mutation == "changed-file":
        (prefix / "einsums-stage/lib/einsums/__init__.py").write_text("# changed\n")
    elif mutation == "wrong-proof-prefix":
        api["psi4"] = "/installed-release/psi4/__init__.py"
    elif mutation == "mixed-native":
        metadata["mapped_libraries"].append("/old/lib/libEinsums.so.1")
    elif mutation == "old-proof":
        os.utime(prefix / "recovery-proofs/api-probe.json", ns=(1, 1))
    elif mutation == "missing-api":
        api["checks"].pop()
    elif mutation == "changed-science":
        (science / "psi4/driver/procrouting/sapt/sapt_proc.py").write_text("# foreign\n")
    elif mutation == "wrong-binary":
        metadata["binary_sha256"][next(iter(metadata["binary_sha256"]))] = "0" * 64
    elif mutation == "extra-module":
        (prefix / "einsums-stage/lib/einsums/foreign.py").write_text("# overlay\n")
    elif mutation == "extra-library":
        (prefix / "einsums-stage/lib/libEinsums.so.1").write_text("old world\n")
    elif mutation == "changed-origin":
        (module.ORIGIN / "einsums-build/lib/einsums/__init__.py").write_text("# changed\n")
    elif mutation == "changed-origin-link":
        path = module.ORIGIN / "einsums-build/lib/libEinsums-abi1.so"
        path.unlink()
        path.symlink_to("libEinsums-abi1.so.2.0.0")
    if mutation in {"wrong-recipe", "partial"}:
        (prefix / "deployment-manifest.json").write_text(json.dumps(manifest))
        # Keep proofs fresh so the intended inventory/recipe check is exercised.
        now = time.time_ns() + 1_000_000
        for name in ("runtime-provenance.json", "api-probe.json"):
            os.utime(prefix / "recovery-proofs" / name, ns=(now, now))
    if mutation in {"wrong-proof-prefix", "missing-api"}:
        (prefix / "recovery-proofs/api-probe.json").write_text(json.dumps(api))
    if mutation in {"mixed-native", "wrong-binary"}:
        (prefix / "recovery-proofs/runtime-provenance.json").write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError):
        validate(recovery)


def test_original_failed_build_admission_remains_strict():
    root = Path(__file__).resolve().parents[2]
    old = (root / "devtools/disp0/prepare_gpu_gate.sh").read_text()
    assert 'assert "exit_code=0\\n" in status' in old
    assert "CPU_PREPARATION_EXIT_ZERO" in old
    assert "recovered_runtime" not in old


@pytest.mark.parametrize("corrupt", [None, "sm90", "mock", "nvcc", "header", "compiler"])
def test_original_compilation_provenance_is_bound(recovery, monkeypatch, corrupt):
    fixture_module = recovery[0]
    module = load("recovered_runtime")
    monkeypatch.setattr(module, "ORIGIN", fixture_module.ORIGIN)
    accounting = ("14196330|FAILED|1:0|00:40:04|gts-cs207-chemx|embers|8|48G|fixture\n"
                  "14196330.batch|FAILED|1:0|00:40:04|gts-cs207-chemx||8||fixture\n")
    monkeypatch.setattr(module.subprocess, "check_output", lambda *args, **kwargs: accounting)
    root = module.ORIGIN
    (root / "environment-explicit.txt").write_text("fixture-gnu14.3-build20\n")
    (root / "nvcc-version.txt").write_text("V13.3.73" if corrupt != "nvcc" else "V12.6")
    (root / "cuda-header-runtime.txt").write_text("13030" if corrupt != "header" else "12060")
    compiler = str(root / "environment/bin/g++") if corrupt != "compiler" else "/usr/bin/g++"
    cache = (f"CMAKE_CXX_COMPILER:FILEPATH={compiler}\n"
             "EINSUMS_WITH_CUDA:BOOL=ON\nEINSUMS_BUILD_PYTHON:BOOL=ON\n"
             f"EINSUMS_WITH_CUDA_ARCHITECTURES:STRING={'80' if corrupt == 'sm90' else '90'}\n"
             f"EINSUMS_WITH_GPU_MOCK_DISCRETE:BOOL={'ON' if corrupt == 'mock' else 'OFF'}\n")
    for directory in ("einsums-build", "psi4-build"):
        (root / directory / "CMakeCache.txt").write_text(cache)
    receipt = root.parent / "logs" / f"submission-{module.COMPILE_PIN[:10]}.receipt"
    receipt.write_text(f"source={module.COMPILE_PIN}\njob={module.JOB}\n")
    if corrupt:
        with pytest.raises(RuntimeError):
            module.compilation_provenance()
    else:
        result = module.compilation_provenance()
        assert len(result["environment-explicit.txt"]["sha256"]) == 64
        assert "build20" in result["toolchain_drift"]


@pytest.mark.parametrize("corrupt", [None, "job-exit", "batch-exit", "qos", "account"])
def test_original_job_and_batch_accounting_retain_failed_exit(corrupt):
    module = load("recovered_runtime")
    text = ("14196330|FAILED|1:0|00:40:04|gts-cs207-chemx|embers|8|48G|fixture\n"
            "14196330.batch|FAILED|1:0|00:40:04|gts-cs207-chemx||8||fixture\n")
    if corrupt == "job-exit":
        text = text.replace("14196330|FAILED|1:0", "14196330|COMPLETED|0:0")
    elif corrupt == "batch-exit":
        text = text.replace("14196330.batch|FAILED|1:0", "14196330.batch|COMPLETED|0:0")
    elif corrupt == "qos":
        text = text.replace("|embers|", "|inferno|")
    elif corrupt == "account":
        text = text.replace("|gts-cs207-chemx|", "|gts-cs207|")
    if corrupt:
        with pytest.raises(RuntimeError):
            module.verify_accounting(text)
    else:
        assert module.verify_accounting(text)["job_and_batch_remain_failed"] is True
