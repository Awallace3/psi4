"""Synthetic timing-control tests, never evidence of native CUDA execution."""

import copy
import importlib.util
from pathlib import Path
import subprocess
import sys
import os

import pytest


def load(name):
    path = Path(__file__).resolve().parents[2] / f"devtools/disp0/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_gate(tmp_path):
    run = load("benchmark_run")
    binaries = {}
    for name in ("psi4-core", "einsums-core", "einsums-native"):
        path = tmp_path / name
        path.write_text(name)
        binaries[str(path)] = run.digest(path)
    modules = {}
    for name in ("sapt_jk_terms_ein.py", "sapt_mp2_terms_ein.py", "sapt_dispersion_gpu.py", "sapt_proc.py"):
        science = tmp_path / "psi4-build/stage/lib/psi4/driver/procrouting/sapt" / name
        science.parent.mkdir(parents=True, exist_ok=True)
        science.write_text("pass\n")
        modules[str(science)] = run.digest(science)
    return run, {
        "status": "PASS", "failures": [], "comparisons": [{"pass": True}],
        "science_build": {"cpu_preparation_job": "14196330",
                          "pin": "587e172a33afaadb00eb7f574a5089533f7fc755"},
        "loaded_binary_sha256": binaries,
        "imported_scientific_modules": modules,
        "cuda_identity": {"header_version": 13030, "runtime_version": 13030},
        "native_cublas_trace_evidence": {"actual_production_cublas_calls": 4},
        "profile_dispatch_evidence": {"observed_cpu_edges": [{}]},
        "native_logger_preflight": {
            "cpu": {"status": "PASS", "policy_device": "cpu", "probe": {"dgemm_mentions": 0},
                    "positive_control": {"dgemm_mentions": 1}},
            "gpu": {"status": "PASS", "policy_device": "cuda", "probe": {"dgemm_mentions": 0},
                    "positive_control": {"dgemm_mentions": 3}}},
    }


def test_timing_accepts_only_complete_matching_gate_fixture(tmp_path):
    run, gate = fake_gate(tmp_path)
    run.require_gate(gate, tmp_path)


@pytest.mark.parametrize("failure", [
    "status", "signed", "binary", "science", "runtime", "job", "native",
    "cpu-edge", "logger-negative", "logger-positive"])
def test_timing_rejects_incomplete_or_changed_gate_fixture(tmp_path, failure):
    run, original = fake_gate(tmp_path)
    gate = copy.deepcopy(original)
    if failure == "status":
        gate["status"] = "RUNNING"
    elif failure == "signed":
        gate["comparisons"][0]["pass"] = False
    elif failure == "binary":
        Path(next(iter(gate["loaded_binary_sha256"]))).write_text("changed")
    elif failure == "science":
        Path(next(iter(gate["imported_scientific_modules"]))).write_text("changed")
    elif failure == "runtime":
        gate["cuda_identity"]["runtime_version"] = 12060
    elif failure == "job":
        gate["science_build"]["cpu_preparation_job"] = "older"
    elif failure == "native":
        gate["native_cublas_trace_evidence"]["actual_production_cublas_calls"] = 0
    elif failure == "cpu-edge":
        gate["profile_dispatch_evidence"]["observed_cpu_edges"] = []
    elif failure == "logger-negative":
        gate["native_logger_preflight"]["cpu"]["probe"]["dgemm_mentions"] = 1
    else:
        gate["native_logger_preflight"]["cpu"]["positive_control"]["dgemm_mentions"] = 0
    with pytest.raises(AssertionError):
        run.require_gate(gate, tmp_path)


@pytest.mark.parametrize("name", [
    "sapt_jk_terms_ein.py", "sapt_mp2_terms_ein.py", "sapt_dispersion_gpu.py", "sapt_proc.py"])
def test_timing_rejects_each_omitted_scientific_module(tmp_path, name):
    run, gate = fake_gate(tmp_path)
    gate["imported_scientific_modules"] = {
        path: value for path, value in gate["imported_scientific_modules"].items()
        if Path(path).name != name}
    with pytest.raises(AssertionError):
        run.require_gate(gate, tmp_path)


@pytest.mark.parametrize("name", [
    "benchmark_run", "benchmark_resources", "benchmark_evidence", "gpu_gate", "runtime_provenance"])
def test_optimized_python_is_rejected_before_import_or_proof(name):
    path = Path(__file__).resolve().parents[2] / f"devtools/disp0/{name}.py"
    result = subprocess.run([sys.executable, "-O", str(path), "--help"],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert "optimized Python is forbidden" in result.stderr


@pytest.mark.parametrize("version", [1, 2])
def test_memory_check_uses_process_job_and_tightest_ancestor(version):
    module = load("benchmark_resources")
    if version == 2:
        membership = "0::/slurm/job_12/step_0\n"
        mountinfo = "1 2 0:4 / /sys/fs/cgroup rw - cgroup2 cgroup rw"
        names = ("memory.max", "memory.current")
    else:
        membership = "5:memory:/slurm/job_12/step_0\n"
        mountinfo = "1 2 0:4 / /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory"
        names = ("memory.limit_in_bytes", "memory.usage_in_bytes")
    mount = Path("/sys/fs/cgroup" + ("/memory" if version == 1 else ""))
    rows = {mount: (1024, 100), mount / "slurm": (800, 100),
            mount / "slurm/job_12": (256, 6), mount / "slurm/job_12/step_0": (512, 4)}
    data = {str(path / field): str(values[index] * module.GiB)
            for path, values in rows.items() for index, field in enumerate(names)}
    result = module.effective_cgroup_memory(membership, mountinfo, lambda path: data[str(path)])
    assert result["available"] == 250 * module.GiB


def test_memory_check_rejects_unproven_namespace_bind_and_unknown_limits():
    module = load("benchmark_resources")
    mountinfo = "1 2 0:4 /slurm/job_12 /sys/fs/cgroup rw - cgroup2 cgroup rw"
    data = {"/sys/fs/cgroup/memory.max": "1000", "/sys/fs/cgroup/memory.current": "10"}
    with pytest.raises(RuntimeError, match="finite process"):
        module.effective_cgroup_memory("0::/\n", mountinfo, lambda path: data[str(path)])
    mountinfo = "1 2 0:4 / /sys/fs/cgroup rw - cgroup2 cgroup rw"
    with pytest.raises(RuntimeError, match="finite process"):
        module.effective_cgroup_memory("0::/\n", mountinfo, lambda path: data[str(path)])
    data["/sys/fs/cgroup/memory.max"] = "max"
    with pytest.raises(RuntimeError, match="finite process"):
        module.effective_cgroup_memory("0::/\n", mountinfo, lambda path: data[str(path)])


def test_real_v2_hierarchy_root_has_no_memory_max_but_job_bound_applies():
    module = load("benchmark_resources")
    data = {"/sys/fs/cgroup/job/memory.max": str(512 * module.GiB),
            "/sys/fs/cgroup/job/memory.current": str(12 * module.GiB)}

    def read(path):
        if str(path) not in data:
            raise FileNotFoundError(path)
        return data[str(path)]

    mountinfo = "1 2 0:4 / /sys/fs/cgroup rw - cgroup2 cgroup rw"
    result = module.effective_cgroup_memory("0::/job\n", mountinfo, read)
    assert result["available"] == 500 * module.GiB
    with pytest.raises(FileNotFoundError):
        module.effective_cgroup_memory("0::/missing/job\n", mountinfo, read)


@pytest.mark.parametrize("parent,child,expected", [
    ("42 P /scratch", "42 - /scratch/probe", 42),
    ("42 - /scratch", "0 - /scratch/probe", 0),
])
def test_project_quota_uses_confirmed_new_inode_policy(parent, child, expected):
    assert load("benchmark_resources").project_policy(parent, child)["new_child_id"] == expected


def test_inherited_project_mismatch_rejected():
    with pytest.raises(AssertionError, match="inheritance"):
        load("benchmark_resources").project_policy("42 P /scratch", "0 - /scratch/probe")


@pytest.mark.parametrize("fail", [False, True])
def test_ownership_probe_is_private_and_removed_even_on_failure(tmp_path, monkeypatch, fail):
    module = load("benchmark_resources")

    def project(command, **kwargs):
        path = Path(command[-1])
        assert command[:3] == ["lfs", "project", "-d"]
        if path == tmp_path:
            return f"42 P {path}\n"
        assert path.is_file() and path.stat().st_size == 0
        return f"{0 if fail else 42} - {path}\n"

    monkeypatch.setattr(module.subprocess, "check_output", project)
    if fail:
        with pytest.raises(AssertionError):
            module.inspect_scratch_policy(tmp_path)
    else:
        info, policy, _, _ = module.inspect_scratch_policy(tmp_path)
        assert info.st_uid == os.geteuid() and info.st_gid == tmp_path.stat().st_gid
        assert policy["new_child_id"] == 42
    assert not list(tmp_path.iterdir())


class Function:
    def __init__(self):
        self.enabled = False
        self.path = None

    def __call__(self, enabled, stdout, stderr, path):
        assert stdout == stderr == 0
        self.enabled = bool(enabled)
        self.path = Path(path.decode()) if path else None
        return 0


@pytest.mark.parametrize("fail", [False, True])
def test_native_logger_is_off_after_untimed_capture_even_on_failure(tmp_path, monkeypatch, fail):
    module = load("benchmark_evidence")
    configure = Function()
    library = type("Library", (), {"cublasLoggerConfigure": configure})()
    monkeypatch.setattr(module.ctypes, "CDLL", lambda _: library)
    logger = module.NativeLogger()
    assert not configure.enabled


    def execute():
        assert configure.enabled
        configure.path.write_text("cublasDgemm_v2: synthetic parser fixture, not a native call\n")
        if fail:
            raise RuntimeError("synthetic failure")

    path = tmp_path / "native.log"
    if fail:
        with pytest.raises(RuntimeError, match="synthetic failure"):
            logger.capture(path, execute, lambda: None)
    else:
        assert logger.capture(path, execute, lambda: None)["dgemm_mentions"] == 1
        with pytest.raises(AssertionError, match="append"):
            logger.capture(path, execute, lambda: None)
    assert not configure.enabled


@pytest.mark.parametrize("row,expected", [
    ("/scratch/user 100 1000 1200 - 20 100 200 -", {"byte_headroom": 900 * 1024, "inode_headroom": 80}),
    ("/scratch/user 100 0 0 - 20 0 0 -", {"byte_headroom": None, "inode_headroom": None}),
    ("/scratch/user\n 100* 1000 1200 - 20 100 200 -", {"byte_headroom": 900 * 1024, "inode_headroom": 80}),
])
def test_quota_parser_preserves_finite_and_unlimited_limits(row, expected):
    module = load("benchmark_resources")
    assert module.quota_headroom("Disk quotas:\nFilesystem kbytes bquota blimit bgrace files iquota ilimit igrace\n"
                                 + row) == expected


@pytest.mark.parametrize("text", ["quota query failed", "/scratch 1 2 3 - 4 5 6 -\n/other 1 2 3 - 4 5 6 -"])
def test_quota_parser_rejects_missing_or_ambiguous_evidence(text):
    with pytest.raises(AssertionError, match="Ambiguous/missing"):
        load("benchmark_resources").quota_headroom(text)


def test_synthetic_probe_dimensions_are_small_and_do_not_mutate_policy():
    module = load("benchmark_evidence")
    policy = {"compute_block_extents": [500, 500], "occupied_A": 222,
              "occupied_B": 68, "naux": 7384}
    original = copy.deepcopy(policy)
    assert module.probe_dimensions(policy) == {"nr": 4, "ns": 4, "na": 8, "nb": 8, "nq": 64}
    assert policy == original
    small = {"compute_block_extents": [1, 2], "occupied_A": 3, "occupied_B": 2, "naux": 11}
    assert module.probe_dimensions(small) == {"nr": 1, "ns": 2, "na": 3, "nb": 2, "nq": 11}
    with pytest.raises(AssertionError):
        module.probe_dimensions({**small, "naux": 0})


def test_worker_has_only_one_full_energy_call_and_no_adapter_retention():
    import ast
    path = Path(__file__).resolve().parents[2] / "devtools/disp0/benchmark_run.py"
    tree = ast.parse(path.read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "energy"]
    assert len(calls) == 1
    assert "capture_adapter" not in path.read_text()
    assert "untimed_reconstructed_call" not in path.read_text()


def test_cpu_graph_full_hf_cphf_matches_native_sapt(monkeypatch, record_property):
    """Small physical protocol regression, not a scaling/timing observation."""
    import importlib
    import json
    import numpy as np
    import psi4

    terms = importlib.import_module("psi4.driver.procrouting.sapt.sapt_jk_terms_ein")
    original = terms.dispersion0
    geometry = """
    0 1
    O 0.0 0.0 0.0
    H 0.0 0.0 0.96
    H 0.91 0.0 -0.24
    --
    0 1
    Ne 2.8 1.1 1.5
    symmetry c1
    no_reorient
    no_com
    """
    keys = ("SAPT ELST ENERGY", "SAPT EXCH ENERGY", "SAPT IND ENERGY",
            "SAPT DISP ENERGY", "SAPT TOTAL ENERGY", "Disp20,u", "Exch-Disp20,u")
    results = {}
    graph_checks = []

    def cpu_graph(*a, **kw):
        kw["_graph_device"] = "cpu"
        result = original(*a, **kw)
        graph_checks.append(result["Disp0 execution"]["planned_gpu_contractions"])
        return result

    monkeypatch.setattr(terms, "dispersion0", cpu_graph)
    try:
        for algorithm in ("SAPT", "EINSUMS"):
            psi4.core.clean()
            psi4.core.clean_options()
            psi4.set_options({
                "basis": "cc-pvdz", "puream": True, "scf_type": "df", "freeze_core": False,
                "SAPT_DFT_FUNCTIONAL": "HF", "SAPT_DFT_DO_DHF": True,
                "SAPT_DFT_INDUCTION_TYPE": "CPHF", "SAPT_DFT_USE_EINSUMS": True,
                "SAPT_DFT_MP2_DISP_ALG": algorithm, "SAPT_DFT_EINSUMS_GPU": False,
                "ORBITAL_OPTIMIZER_PACKAGE": "INTERNAL",
            })
            energy = psi4.energy("sapt(dft)", molecule=psi4.geometry(geometry))
            results[algorithm] = {key: float(psi4.variable(key)) for key in keys}
            np.testing.assert_allclose(energy, results[algorithm]["SAPT TOTAL ENERGY"],
                                       rtol=1e-11, atol=1e-12)
        assert graph_checks == [0]
        deltas = {key: results["EINSUMS"][key] - results["SAPT"][key] for key in keys}
        record_property("full_hf_cphf_component_deltas_Eh", json.dumps(deltas))
        for key in keys:
            np.testing.assert_allclose(results["EINSUMS"][key], results["SAPT"][key],
                                       rtol=1e-8, atol=1e-8)
    finally:
        psi4.core.clean()
        psi4.core.clean_options()
