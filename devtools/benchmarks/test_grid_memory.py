import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("grid_memory", Path(__file__).with_name("grid_memory.py"))
grid_memory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grid_memory)
Boundaries = grid_memory.Boundaries
parse_compute_apps = grid_memory.parse_compute_apps
summarize_samples = grid_memory.summarize_samples


def test_only_own_processes_and_sum_per_device():
    values, refused = parse_compute_apps(
        "11, 100, GPU-a\n12, 200, GPU-a\n13, 9000, GPU-a\n12, 20, GPU-b\n",
        {11, 12})
    assert values == {"GPU-a": 300, "GPU-b": 20}
    assert not refused


@pytest.mark.parametrize("value", ["[N/A]", "[Not Supported]", "-1", "NaN", "inf"])
def test_unavailable_memory_is_not_zero(value):
    used, refused = parse_compute_apps(f"11, {value}, GPU-a", {11})
    assert not used and len(refused) == 1
    report = summarize_samples([dict(by_uuid_mib=used, total_mib=0, unavailable=refused)])
    assert report["sampled_peak_mib"] is None
    assert report["unavailable_samples"] == 1


def test_other_tenant_unavailable_does_not_taint_ours():
    used, refused = parse_compute_apps("12, [N/A], GPU-a\n11, 80, GPU-a", {11})
    assert used == {"GPU-a": 80} and not refused


def test_corrupt_inventory_is_an_error():
    with pytest.raises(ValueError):
        parse_compute_apps("not a CSV process", {11})


def test_empty_inventory_is_not_proof_of_gpu_work():
    report = summarize_samples([dict(by_uuid_mib={}, total_mib=0, unavailable=[])])
    assert report["sampled_peak_mib"] == 0
    assert not report["gpu_observed"]


def test_phases_distinguish_final_coarse_and_basis():
    samples = []
    for initial, nbf, used in [(True, 48, 200), (False, 48, 300), (False, 922, 1000)]:
        samples.append(dict(by_uuid_mib={"GPU-a": used}, total_mib=used, unavailable=[],
                            phase=dict(phase="form_G", stack=["scf_helper:A", "form_G"],
                                       initial_grid=initial, nbf=nbf)))
    report = summarize_samples(samples + [dict(error="query failed")])
    assert len(report["phases"]) == 3
    assert report["sampled_peak_mib"] == 1000
    assert report["failed_samples"] == 1


def test_wrap_restores_inherited_method_even_after_exception(tmp_path):
    class Parent:
        def method(self):
            raise ValueError("real failure")
    class Child(Parent):
        pass
    hook = Boundaries(tmp_path)
    hook.wrap(Child, "method", "phase")
    with pytest.raises(ValueError, match="real failure"):
        Child().method()
    records = [json.loads(line) for line in (tmp_path / "boundaries.jsonl").read_text().splitlines()]
    assert [r["event"] for r in records] == ["enter", "error", "resume"]
    assert hook.stack == []
    owner, name, original, inherited = hook.originals.pop()
    assert inherited
    delattr(owner, name)
    assert Child.method is Parent.method


def test_potential_grid_transition_metadata_does_not_require_wfn():
    grid = SimpleNamespace(npoints=lambda: 4321, max_points=lambda: 256,
                           max_functions=lambda: 40, collocation_size=lambda: 100000)
    potential = SimpleNamespace(
        initial_grid=lambda: False, grid=lambda: grid, nblocks=lambda: 17,
        basis=lambda: SimpleNamespace(nbf=lambda: 50))
    data = Boundaries.describe(potential)
    assert data["initial_grid"] is False
    assert data["grid_points"] == 4321 and data["nbf"] == 50


@pytest.mark.parametrize("memory,exit_code,expected_ok", [
    ("128", 0, True), ("[N/A]", 0, False), ("128", 1, False),
])
def test_supervisor_honors_accounting_and_worker_failure(
        tmp_path, monkeypatch, capsys, memory, exit_code, expected_ok):
    """Fake only hardware/child workload; exercise the real supervisor/thread."""
    real_popen = subprocess.Popen
    child_pid = []
    def fake_popen(command, **kwargs):
        child = real_popen(
            [grid_memory.sys.executable, "-c",
             f"import time;time.sleep(0.2);raise SystemExit({exit_code})"], **kwargs)
        child_pid.append(child.pid)
        return child
    def fake_query(command, **kwargs):
        if any("query-gpu" in arg for arg in command):
            text = "GPU-test, Fake GPU, 8192, fake-driver\n"
        else:
            text = f"{child_pid[0]}, {memory}, GPU-test\n"
        return SimpleNamespace(returncode=0, stdout=text, stderr="")
    monkeypatch.setattr(grid_memory.shutil, "which", lambda _: "/fake/nvidia-smi")
    monkeypatch.setattr(grid_memory.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(grid_memory.subprocess, "run", fake_query)
    directory = tmp_path / "measurement"
    args = SimpleNamespace(output=str(directory), script=__file__, arguments=[],
                           interval=0.02, timeout=3)
    assert grid_memory.run(args) == (0 if expected_ok else 1)
    report = json.loads((directory / "summary.json").read_text())
    assert report["ok"] is expected_ok
    assert report["eligible_for_speedup_bars"] is False
    assert report["returncode"] == exit_code
    assert report["samples"] > 0
    if memory == "[N/A]":
        assert report["sampled_peak_mib"] is None
    capsys.readouterr()
