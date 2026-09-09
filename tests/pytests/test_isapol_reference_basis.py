# Psi4: Copyright (c) 2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""Literal/algebraic checks run from source, without staged Psi4 metadata.

The explicitly named native_core tests require the parent's real core. Pure
checks are not substitutes for existing basis/coulomb/partition/response tests.
No fixture generator imports, SCF, large grids or response integrals. The
native-core public test's Wavefunction.build does compute a bounded AO overlap.
"""
from dataclasses import FrozenInstanceError, fields, is_dataclass, replace
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).parent / 'data_isapol/h2o_props_psi4_basis'


def load_source(name):
    spec = importlib.util.spec_from_file_location('_owned_' + name,
        ROOT / 'psi4/driver/procrouting' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


reference = load_source('isapol_reference_basis')
preflight = load_source('isapol_response_preflight')


def test_literal_manifest_and_hash_pins():
    raw = (DATA / 'records.json').read_bytes()
    pin = '640afec4c49dc2dbc76819a985d3840d1cf0459a9c1192806339f396b4dd4938'
    fixture = json.loads(raw)
    evidence = json.loads((DATA / 'manifest.json').read_text())
    assert hashlib.sha256(raw).hexdigest() == evidence['records_sha256'] == pin
    m = reference.h2o_props_psi4_777f904_manifest()
    assert dict(m.source_sha256) == evidence['sources_sha256']
    assert m.reference_revision == '777f90498868d33847de525612628b2dc8448523'
    assert m.inspected_revision == '63b16a22b9bae597fe81ecdb8b8d91c21868c814'
    for e in m.elements:
        for role in ('main', 'aux'):
            actual = [[s.l, list(s.exponents), list(s.source_coefficients)]
                      for s in getattr(e, role + '_shells')]
            assert actual == fixture[e.symbol][role]  # every ordered numeric record
    assert m.expected_main_nfunction == 92
    assert m.expected_aux_nfunction == 246
    o, h = m.elements
    assert [len(e.main_shells) for e in (o, h)] == [14, 9]
    assert [sum(len(s.exponents) for s in e.main_shells) for e in (o, h)] == [36, 13]
    assert len(o.aux_shells) + 2*len(h.aux_shells) == 56
    assert [len(s.exponents) for s in o.main_shells if s.l == 0] == [10, 10, 1, 1, 1]
    assert o.main_shells[0].exponents == o.main_shells[1].exponents
    assert o.main_shells[0].source_coefficients != o.main_shells[1].source_coefficients
    assert o.main_shells[1].source_coefficients[0] == -.000115
    assert [(s.l, s.exponents[0]) for s in o.aux_shells[-6:]] == [
        (4, 2.3270964878), (0, .11211820118), (1, .21354459926),
        (2, .16734574676), (3, .40383704543), (4, .78655757162)]
    assert [(s.l, s.exponents[0]) for s in h.aux_shells[-5:]] == [
        (3, 1.8063060576), (0, .12719436063), (1, .23551289521),
        (2, .43665405053), (3, .25297787763)]


def test_deep_immutability_non_isa_and_missing_export():
    m = reference.h2o_props_psi4_777f904_manifest()
    def immutable(x):
        if is_dataclass(x):
            assert x.__dataclass_params__.frozen
            for f in fields(x):
                immutable(getattr(x, f.name))
        elif isinstance(x, tuple):
            for item in x:
                immutable(item)
        else:
            assert isinstance(x, (str, int, float, type(None)))
    immutable(m)
    assert not callable(m)
    assert not hasattr(m, 'to_partition_recipe')
    assert m.isa_algorithm is None and m.shape == 'shape_not_applicable'
    assert m.atomaux_mode == 'same_as_aux_fallback'
    assert m.historical_scf_export_verified is False
    assert 'historical H2O-A.basis SCF export' in m.missing_artifacts
    assert 'NN' in m.response_route and 'PFIT' in m.localization
    assert m.fixed_grac_shift_hartree_reference == .06490004527520865
    assert m.geometry_bohr[1] == ('H1', (-1.4536519600, 0., -1.1216873200))
    assert dict(m.known_modern_grid_case)['rows'] == 3*(99-1)*590
    assert 'not literal contractions' in m.native_main_comparison
    with pytest.raises(FrozenInstanceError):
        m.elements[0].main_shells[0].l = 4
    with pytest.raises(FrozenInstanceError):
        m.historical_scf_export_verified = True
    with pytest.raises(TypeError):
        replace(m, historical_scf_export_verified=True)


def test_no_hidden_runtime_inputs(monkeypatch, tmp_path):
    # Execute all production module code with file access disallowed, after
    # reading its source. Dependencies are stdlib, not tests or CamCASP paths.
    source = Path(reference.__file__).read_text()
    def forbidden(*args, **kwargs):
        pytest.fail('hidden runtime file read')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('HOME', '/nonexistent-reference-home')
    monkeypatch.setattr('builtins.open', forbidden)
    monkeypatch.setattr(Path, 'read_text', forbidden)
    monkeypatch.setattr(Path, 'read_bytes', forbidden)
    namespace = {'__name__': reference.__name__}
    exec(compile(source, 'self-contained-reference', 'exec'), namespace)
    m = namespace['h2o_props_psi4_777f904_manifest']()
    assert m.expected_aux_nfunction == 246
    assert m.elements[0].aux_shells[0].effective_coefficients[0] != 1.


def moment(k, exponent):
    # Independent analytic integral of x**(2k) exp(-exponent*x*x).
    return math.gamma(k + .5) / exponent**(k + .5)


def test_effective_normalization_and_all_gamint_self_overlaps():
    for e in reference.h2o_props_psi4_777f904_manifest().elements:
        for s in e.main_shells + e.aux_shells:
            coeff = s.effective_coefficients
            overlap = math.fsum(cp*cq * moment(s.l, ap+aq) * moment(0, ap+aq)**2
                               for ap, cp in zip(s.exponents, coeff)
                               for aq, cq in zip(s.exponents, coeff))
            assert overlap == pytest.approx(1., abs=4.e-14)
        for s in e.aux_shells:
            assert s.source_coefficients == (1.,)
            a, c = s.exponents[0], s.effective_coefficients[0]
            assert c != 1.
            want = 1 / math.sqrt(moment(s.l, 2*a) * moment(0, 2*a)**2)
            assert c == pytest.approx(want, rel=3.e-15)
            # Every GAMINT Cartesian component, including mixed d/f/g powers.
            for ix in range(s.l+1):
                for iy in range(s.l-ix+1):
                    iz = s.l-ix-iy
                    df = lambda n: math.prod(range(1, 2*n, 2))
                    factor_squared = df(s.l) / (df(ix)*df(iy)*df(iz))
                    norm = c*c*factor_squared*math.prod(moment(k, 2*a) for k in (ix, iy, iz))
                    assert norm == pytest.approx(1., abs=4.e-14)


def test_preflight_source_case_and_exact_boundary():
    result = preflight.estimate_response_work(92, 92, 5, 3*(99-1)*590)
    assert result.nov == 435 and result.ao_work == 31_163_093_760
    assert result.alda_work == 32_822_968_500
    assert result.max_grid_rows == 10569
    assert result.failures == ('ALDA work resource limit',)
    with pytest.raises(ValueError, match='NativeResponseProvider: ALDA work resource limit'):
        result.require_pass()
    boundary = 2_000_000_000 // 435**2
    assert preflight.estimate_response_work(92, 92, 5, boundary).passes
    assert not preflight.estimate_response_work(92, 92, 5, boundary+1).passes
    assert 'explicit caller' in result.provenance
    assert result.unchecked
    # Existing small cc-pVDZ public factory dimensions still pass at 99/590.
    small = preflight.estimate_response_work(24, 24, 5, 3*(99-1)*590)
    assert small.nov == 95 and small.passes
    with pytest.raises(FrozenInstanceError):
        result.nov = 0


@pytest.mark.parametrize('index', range(4))
@pytest.mark.parametrize('bad', [True, np.bool_(False), 1.5, 2., -1, '92', None, 2**31])
def test_preflight_invalid_types_and_native_int_dimensions(index, bad):
    dims = [92, 92, 5, 100]
    dims[index] = bad
    with pytest.raises(ValueError):
        preflight.estimate_response_work(*dims)


@pytest.mark.parametrize('dims', [(0, 2, 1, 0), (5, 6, 1, 0), (5, 5, 0, 0),
                                 (5, 5, 5, 0), (5, 5, 6, 0)])
def test_preflight_inconsistent_dimensions(dims):
    with pytest.raises(ValueError):
        preflight.estimate_response_work(*dims)


def test_preflight_all_limits_order_and_checked_math():
    assert preflight.estimate_response_work(np.int64(48), 48, 16, 0).nov == 512
    assert preflight.estimate_response_work(48, 48, 16, 0).passes
    assert not preflight.estimate_response_work(48, 48, 16, 0, max_nov=511).passes
    assert preflight.estimate_response_work(174, 174, 3, 1).failures[0].startswith('dense OV')
    boundary = max(n for n in range(48, 257) if 512*n**4 <= 64_000_000_000)
    assert preflight.estimate_response_work(boundary, 48, 16, 0).passes
    assert preflight.estimate_response_work(boundary+1, 48, 16, 0).failures == ('direct JK work resource limit',)
    assert preflight.estimate_response_work(256, 2, 1, 1_000_000).passes
    assert preflight.estimate_response_work(257, 2, 1, 0).failures == ('direct JK work resource limit',)
    assert preflight.estimate_response_work(2, 2, 1, 1_000_001).failures == ('ALDA work resource limit',)
    huge = 2**31-1
    r = preflight.estimate_response_work(huge, huge, huge//2, huge)
    assert r.ao_work == (huge//2)*(huge-huge//2)*huge**4  # no native overflow/wrap
    assert r.failures == ('dense OV resource limit (maximum 512)',
                          'direct JK work resource limit', 'ALDA work resource limit')
    for bad in (True, np.bool_(True), 1.5, 0, -1):
        with pytest.raises(ValueError):
            preflight.estimate_response_work(2, 2, 1, 0, max_nov=bad)


def test_native_core_expected_aux_sampling_and_recipe_refusal():
    from psi4 import core
    from psi4.driver.procrouting import isapol_native as native
    m = reference.h2o_props_psi4_777f904_manifest()
    basis = reference.build_expected_reference_aux()
    assert basis.nfunction == 246
    # A tiny real native sampling check: pure x and mixed xy d components.
    xyz = np.array([[.2, .3, .4]])
    values = np.asarray(basis.evaluate(xyz.tolist()))[0]
    o = m.elements[0]
    index = next(i for i, s in enumerate(o.aux_shells) if s.l == 2)
    offset = sum(s.cartesian_functions for s in o.aux_shells[:index])
    s = o.aux_shells[index]
    radial = s.effective_coefficients[0] * math.exp(-s.exponents[0]*float(xyz[0]@xyz[0]))
    assert values[offset] == pytest.approx(radial*.2**2, rel=2.e-14)
    assert values[offset+3] == pytest.approx(radial*math.sqrt(3)*.2*.3, rel=2.e-14)
    with pytest.raises(TypeError, match='explicit PartitionRecipe required'):
        native.native_properties(None, m, bonds=(), frames=None, caller_converged=True,
            kernel='no_local', exact_exchange=0., local_scale=0., response_grid=None)
    assert m.historical_scf_export_verified is False


@pytest.mark.parametrize('extra', [0, 1])
def test_native_core_public_preflight_before_heavy_work(monkeypatch, extra):
    # Real basis/state dimensions and public generated-recipe path, no SCF.
    # Admission is bypassed ONLY to isolate resource orchestration. A passing
    # estimate must not be mistaken for a converged/scientifically valid state.
    import psi4
    from psi4.driver.procrouting import isapol_oeprop as api
    wfn = psi4.core.Wavefunction.build(psi4.geometry(
        'O\nH 1 1\nH 1 1 2 100\nsymmetry c1'), 'aug-cc-pvtz')
    assert wfn.basisset().nbf() == wfn.nmo() == 92 and wfn.nalpha() == 5
    nov = wfn.nalpha() * (wfn.nmo() - wfn.nalpha())
    rows = 2_000_000_000 // nov**2 + extra
    vector = np.broadcast_to(np.array([.1]), (rows,))
    class SuppliedGrid:
        def __init__(self, *args):
            pass
        def x(self): return vector
        def y(self): return vector
        def z(self): return vector
        def w(self): return vector
    class ReachedSmallBoundary(Exception):
        pass
    def stop(*args, **kwargs):
        assert extra == 0, 'heavy properties construction reached on rejected grid'
        assert kwargs['response_grid'].shape == (rows, 4)
        raise ReachedSmallBoundary
    monkeypatch.setattr(api, 'validate_request', lambda *args: 0.)
    monkeypatch.setattr(psi4.core, 'IsaGrid', SuppliedGrid)
    monkeypatch.setattr(api.n, 'native_properties', stop)
    monkeypatch.setattr(api.p, 'native_partition', lambda *a, **k: pytest.fail('partition entered'))
    expected = ValueError if extra else ReachedSmallBoundary
    with pytest.raises(expected, match='ALDA work resource limit' if extra else None):
        api.run(wfn, ('ATOMIC_POLARIZABILITIES',))
