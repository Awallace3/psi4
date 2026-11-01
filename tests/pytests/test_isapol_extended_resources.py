"""Explicit finite work tier; bookkeeping only, no molecular quantum work."""
import numpy as np
import pytest

from psi4.driver.procrouting.isapol_bounded_response import BoundedResources, _Ledger


def test_new_work_tier_is_explicit_finite_and_preserves_legacy_ceilings():
    assert BoundedResources(1, 128_000_000_000_000, 64*1024**3, 'extended').capacity == 'extended'
    for capacity, limit in (('standard', 6*10**12), ('large', 100*10**12),
                            ('extended', 128*10**12)):
        assert BoundedResources(1, limit, 1, capacity).max_work == limit
        with pytest.raises(ValueError, match='max_work'):
            BoundedResources(1, limit+1, 1, capacity)
    for field, bad in (('max_work', True), ('max_work', 128e12),
                       ('max_work', float('inf')), ('max_io_bytes', 64*1024**3+1)):
        values = dict(max_bytes=1, max_work=1, max_io_bytes=1, capacity='extended')
        values[field] = bad
        with pytest.raises(ValueError, match=field):
            BoundedResources(**values)


def test_extended_work_is_shared_and_not_refunded_or_reset():
    total = 105_124_176_150_908  # dimension-policy forecast, not measured molecular execution
    ledger = _Ledger(BoundedResources(1000, total, 1000, 'extended'), reserved=100)
    ledger.admit('synthetic earlier work', 900, 100*10**12)
    ledger.admit('synthetic remaining work', 900, total-100*10**12)
    assert ledger.work == total
    with pytest.raises(ValueError, match='cumulative work'):
        ledger.admit('one beyond allowance', 900, 1)
    assert ledger.work == total and ledger.peak == 1000
    with pytest.raises(ValueError, match='shared numeric'):
        ledger.admit('one byte beyond allowance', 901)


def _request(schema):
    from psi4.driver.procrouting.isapol_molecular import MolecularDeclaration, PROFILE
    value = dict(schema=schema, point_group='c1',
        declaration=MolecularDeclaration(PROFILE, (np.eye(3),)*17, ()).as_dict(),
        resources=dict(max_bytes=64*1024**3, max_work=128*10**12,
                       max_io_bytes=64*1024**3, capacity='extended'),
        max_replay_work=10**12, implementation_sha256=dict(core='0'*64, isapol_python='1'*64))
    if schema.endswith('/3'):
        value['pfit_maximum_work_bytes'] = 2*1024**3
    return value


def test_extended_serialized_requests_require_version3_and_explicit_native_cap():
    from psi4.driver.procrouting.isapol_schema import NativeMonomerRequest
    raw = _request('native-monomer-request/3')
    request = NativeMonomerRequest.from_dict(raw)
    assert request.as_dict() == raw
    assert request.resources.max_work == 128*10**12
    with pytest.raises(ValueError, match='request/3'):
        NativeMonomerRequest.from_dict(_request('native-monomer-request/2'))
    from dataclasses import replace
    with pytest.raises(ValueError, match='request/3'):
        replace(request, pfit_maximum_work_bytes=None)
    missing = {k: v for k, v in raw.items() if k != 'pfit_maximum_work_bytes'}
    with pytest.raises(ValueError, match='versioned'):
        NativeMonomerRequest.from_dict(missing)
    for cap in (None, True, 0, -1, 2.0, float('inf'), 2**100):
        with pytest.raises(ValueError):
            NativeMonomerRequest.from_dict(dict(raw, pfit_maximum_work_bytes=cap))


def test_extended_live_numeric_budget_must_fit_psi4_memory_before_state(monkeypatch, tmp_path):
    from psi4 import core
    from psi4.driver.procrouting import isapol_bounded_response as backend
    from psi4.driver.procrouting.isapol_logging import StageLog
    def forbidden(*args, **kwargs):
        raise AssertionError('memory refusal must precede state or ledger construction')
    class State:
        molecule = forbidden
    messages = []
    kwargs = dict(caller_converged=True, quadrature=None, response_grid=None,
        smoothing=None, shell_cutoff=0., charge_penalty=0., anchor_metric_damping=0.,
        scf_correction='NONE', scratch_directory=tmp_path,
        log=StageLog(3, writer=messages.append))
    monkeypatch.setattr(core, 'get_memory', lambda: 1024)
    monkeypatch.setattr(backend, '_Ledger', forbidden)
    with pytest.raises(ValueError, match='extended capacity numeric budget exceeds'):
        backend.BoundedResponse(State(), None, (),
            resources=BoundedResources(1025, 1, 1, 'extended'), **kwargs)
    # Exact byte allowance passes this gate and reaches the next declaration
    # check, without constructing state, scratch or a ledger.
    with pytest.raises(ValueError, match='explicit AUX recipe'):
        backend.BoundedResponse(State(), None, (),
            resources=BoundedResources(1024, 1, 1, 'extended'), **kwargs)
    assert not messages and not list(tmp_path.iterdir())
