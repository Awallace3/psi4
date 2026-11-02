"""Refusal, exact-byte loading and unchanged-stage guards; no Psi4 import."""
from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


planner = load('guard_expanded_resource_plan', 'resource_plan_cb63305.py')
legacy = load('guard_legacy_resource_plan', 'resource_plan.py')
HELPER_NAME = 'psi4.driver.procrouting.isapol_pfit_plan'


def dimensions(sites=3):
    h = sites-1
    return planner.Dimensions(sites, sites-1, 46+23*h, 14+9*h, (8+h)//2,
                              136+55*h, 136+55*h, (109,)+(49,)*h, (12,)+(8,)*h)


def predict(*, cap=2*1024**3, replay=10**12, max_bytes=64*1024**3):
    return planner.plan(dimensions(), dict(max_bytes=max_bytes, max_work=128*10**12,
        max_io_bytes=64*1024**3, capacity='extended'),
        max_replay_work=replay, pfit_maximum_work_bytes=cap)


def installed_helper(monkeypatch, tmp_path):
    path = tmp_path/'isapol_pfit_plan.py'
    path.write_bytes((Path(__file__).resolve().parents[2]/planner.PFIT_SOURCE).read_bytes())
    module = ModuleType(HELPER_NAME)
    module.__file__ = str(path)
    monkeypatch.setitem(sys.modules, HELPER_NAME, module)
    return path


def test_installed_source_is_selected_instead_of_repo_fallback(monkeypatch, tmp_path):
    path = installed_helper(monkeypatch, tmp_path)
    observed = planner.shared_pfit_plan()
    assert Path(observed.__file__) == path
    assert observed.native_work_limit() == 256*1024**2


@pytest.mark.parametrize('change', ['missing', 'modified'])
def test_bad_installed_source_refuses_without_repo_fallback(monkeypatch, tmp_path, change):
    path = installed_helper(monkeypatch, tmp_path)
    if change == 'missing':
        path.unlink()
        error = FileNotFoundError
    else:
        path.write_text("raise AssertionError('unverified source executed')")
        error = ValueError
    # The valid repository copy still exists, but must never be substituted.
    with pytest.raises(error):
        planner.shared_pfit_plan()


def test_executes_exact_hashed_bytes_without_a_second_source_read(monkeypatch, tmp_path):
    target = installed_helper(monkeypatch, tmp_path)
    original = Path.read_bytes
    reads = []
    def replace_after_read(path):
        raw = original(path)
        if path == target:
            reads.append(1)
            target.write_text("raise AssertionError('replacement source executed')")
        return raw
    monkeypatch.setattr(Path, 'read_bytes', replace_after_read)
    verified = planner.shared_pfit_plan()
    assert verified.native_work_limit() == 256*1024**2
    assert reads == [1]
    # The next call must detect the changed file; this is not a file lock or
    # permission to keep using a once-verified, now-modified runtime.
    with pytest.raises(ValueError, match='helper differs'):
        planner.shared_pfit_plan()


@pytest.mark.parametrize('cap', [None, True, False, 0, -1, 1., float('inf'), 2**100])
def test_native_cap_must_be_explicit_finite_and_representable(cap):
    with pytest.raises(ValueError):
        predict(cap=cap)


@pytest.mark.parametrize('replay', [None, True, 0, -1, 1., float('inf'), 2*10**12+1])
def test_replay_budget_is_not_promoted_to_the_ceiling(replay):
    with pytest.raises(ValueError, match='max_work'):
        predict(replay=replay)


def test_packed_byte_refusal_propagates_without_a_larger_retry():
    with pytest.raises(ValueError, match='packed design byte resource limit'):
        predict(max_bytes=1)


@pytest.mark.parametrize('sites', range(3, 18))
def test_every_nonfit_stage_is_identical_to_the_frozen_legacy_mirror(sites):
    current = planner.plan(dimensions(sites), dict(max_bytes=64*1024**3,
        max_work=128*10**12, max_io_bytes=64*1024**3, capacity='extended'),
        max_replay_work=10**12, pfit_maximum_work_bytes=2*1024**3)
    old = legacy.plan(legacy.Dimensions(**asdict(dimensions(sites))),
        dict(max_bytes=64*1024**3, max_work=100*10**12,
             max_io_bytes=64*1024**3, capacity='large'), max_replay_work=10**12)
    assert [s for s in current['stages'] if s['name'] != 'complete-cloud refinement'] == [
        s for s in old['stages'] if s['name'] != 'complete-cloud refinement']
