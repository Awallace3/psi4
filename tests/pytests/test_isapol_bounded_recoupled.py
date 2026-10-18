# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Analytic adapter contracts and one explicitly selected fresh WATER chain.

No external scientific arrays. The factorial oracle is table-free; licensed
fixture replay belongs to the separately selected existing core regressions.
"""
from dataclasses import asdict, replace
import itertools
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import psi4
from psi4 import core
from psi4.driver.procrouting import isapol_bounded as bounded
from psi4.driver.procrouting import isapol_bounded_oeprop as api
from psi4.driver.procrouting import isapol_native_recoupled as adapter
from psi4.driver.procrouting import isapol_refine as refine
from psi4.driver.procrouting.isapol_bounded_response import BoundedResources, _Ledger
from psi4.driver.procrouting.isapol_logging import StageLog
from psi4.driver.procrouting.isapol_native import Quadrature
from psi4.driver.procrouting.isapol_lw import Provenance
from isapol_factorial_oracle import independent_g


TASK = 'ATOMIC_REFINED_RECOUPLED_DISPERSION'
CP_PROVENANCE = Provenance('analytic CP', 'a'*64, 'independent analytic test', 'includes 1/(2pi)')


def budget(nbytes=512*1024**2, work=6_000_000_000_000):
    return _Ledger(BoundedResources(nbytes, work, 64*1024**3))


def analytic(ranks=(2, 1), frequencies=(0., .7, 2.)):
    frame = ((0., -1., 0.), (1., 0., 0.), (0., 0., 1.))
    sites = tuple(refine.RefinementSite(f'S{i}', f'T{i}', (float(i), 2., 3.),
                                       frame, rank) for i, rank in enumerate(ranks))
    tensors = []
    for s in sites:
        n = s.component_count-1
        a = np.arange(1., n*n+1).reshape(n, n)/31
        full = np.zeros((n+1, n+1))
        full[1:, 1:] = a+a.T-np.eye(n)*3  # signed, nonuniform, reciprocal
        tensors.append(full)
    results = []
    for f in frequencies:
        arrays = tuple(t/(1+f*f) for t in tensors)
        model = refine.refinement_model(sites, arrays, frequency_au=f, cutoff=0.,
            weight_type=3, weight_coefficient=1e-3, provenance='analytic adapter test')
        results.append(refine.RefinementResult(model, model.anchors, arrays, 0.,
            core.IsaPfitStatus.Solved, None, None, 'analytic; no PFIT run'))
    return results


def calculate(results=None, weights=(0., .4, .2), order=10, ledger=None):
    return adapter.refined_recoupled_dispersion(analytic() if results is None else results,
        cp_weights=weights, max_order=order, quadrature_provenance=CP_PROVENANCE,
        ledger=budget() if ledger is None else ledger)


@pytest.mark.parametrize('ranks', [(3, 2, 1), (4, 3, 2, 1)])
def test_full_mapping_factorial_oracle_metadata_and_ownership(ranks):
    results = analytic(ranks)
    output = calculate(results)
    source = output.model.source
    assert source.frequencies == [0., .7, 2.]
    assert output.cp_weights == [0., .4, .2]
    assert output.pair_labels == tuple(itertools.product(
        (f'S{i}' for i in range(len(ranks))), repeat=2))
    for i, s in enumerate(source.sites):
        declared = results[0].model.sites[i]
        assert s.label == declared.label
        assert s.origin == list(declared.origin_bohr)
        assert s.frame == [list(row) for row in declared.frame]
        assert s.ranks == list(range(1, declared.rank_limit+1))
        assert s.components == list(refine.COMPONENT_NAMES[1:declared.component_count])
        for f, tensor in enumerate(s.responses):
            np.testing.assert_array_equal(tensor.np, results[f].refined_tensors[i][1:, 1:])
            for l, p in itertools.product(s.ranks, repeat=2):
                if l+p > 6:  # known omitted core tables, not zero tensor input
                    assert not any(b.la == l and b.lap == p for b in output.model.sites[i])
                    continue
                block = tensor.np[l*l-1:(l+1)**2-1, p*p-1:(p+1)**2-1]
                expected = np.einsum('vkq,kq->v', independent_g(l, p), block)
                for t in range((l-p)**2+1, (l+p+1)**2+1):
                    assert output.model.value(i, f, l, p, t) == pytest.approx(
                        expected[t-1], rel=3e-14, abs=3e-14)
    before = output.model.value(0, 0, 1, 1, 6)
    results[0].refined_tensors[0][:] = 999.
    source.sites[0].responses[0].np[:] = -555.
    assert output.model.value(0, 0, 1, 1, 6) == before
    assert output.model.source.sites[0].responses[0].np[0, 1] != -555.
    assert any(abs(c.value) > 1e-7 and (c.t != 1 or c.u != 1)
               for p in output.pairs for c in p.coefficients)
    assert not hasattr(output.native_result, 'truncated_energy')


def test_closed_dipole_no_axis_swap_or_trace_reduction():
    r = analytic((1,), (1.,))
    a = np.array([[2., .3, -.4], [.3, 3., .8], [-.4, .8, 5.]])
    r[0].refined_tensors[0][1:, 1:] = a
    out = calculate(r, weights=(1.,), order=6)
    expected = {1: -10/math.sqrt(3), 5: -4/math.sqrt(6),
                6: .6/math.sqrt(2), 7: -.8/math.sqrt(2),
                8: -2/math.sqrt(2), 9: 1.6/math.sqrt(2)}
    for t, value in expected.items():
        assert out.model.value(0, 0, 1, 1, t) == pytest.approx(value, abs=2e-15)
    for t in (2, 3, 4):
        assert out.model.value(0, 0, 1, 1, t) == 0j
    iso = analytic((1,), (1.,))
    iso[0].refined_tensors[0][1:, 1:] = np.eye(3)*(10/3)
    isotropic = calculate(iso, weights=(1.,), order=6)
    assert out.pairs[0].coefficient(6, 1, 1, 0) == pytest.approx(
        isotropic.pairs[0].coefficient(6, 1, 1, 0))
    assert out.pairs[0].coefficient(6, 5, 1, 2) != pytest.approx(
        isotropic.pairs[0].coefficient(6, 5, 1, 2))


def test_scalar_literals_match_existing_isotropic_without_extra_prefactor():
    r = analytic((3, 3), (1.,))
    for tensor, scalars in zip(r[0].refined_tensors, ([2., 3., 5.], [7., 11., 13.])):
        tensor[:] = 0
        tensor[1:, 1:] = np.diag(np.repeat(scalars, [3, 5, 7]))
    out = calculate(r, weights=(.7,))
    iso = refine.refined_isotropic_dispersion(r, cp_weights=(.7,),
        quadrature_provenance=CP_PROVENANCE, max_order=10)
    pair = out.pairs[1]  # S0/S1 ordered
    assert (pair.site_a, pair.site_b) == (0, 1)
    assert pair.coefficient(6, 1, 1, 0) == pytest.approx(58.8, rel=3e-14)
    assert pair.coefficient(8, 1, 1, 0) == pytest.approx(451.5, rel=3e-14)
    for native_pair, iso_pair in zip(out.pairs, iso.pairs):
        for c in iso_pair.coefficients:
            assert native_pair.coefficient(c.order, 1, 1, 0) == pytest.approx(c.value, rel=3e-14)


@pytest.mark.parametrize('ranks', [(2, 1), (4,)])
def test_actual_rank_table_and_unrestricted_inventory(ranks):
    out = calculate(analytic(ranks, (1.,)), weights=(1.,), order=12)
    declared = {(c.site_a, c.site_b, c.order): c for c in out.declared_model_coverage}
    for pair in out.pairs:
        ra, rb = ranks[pair.site_a], ranks[pair.site_b]
        for cov in pair.coverage:
            all_quads = {q for q in itertools.product(range(1, cov.order-4), repeat=4)
                         if sum(q) == cov.order-2}
            model_quads = {q for q in all_quads
                           if q[0] <= ra and q[1] <= ra and q[2] <= rb and q[3] <= rb}
            table_defined = {q for q in model_quads if q[0]+q[1] <= 6 and q[2]+q[3] <= 6}
            included = {tuple(q) for q in cov.included_rank_quadruples}
            assert included == table_defined
            assert {tuple(q) for q in cov.missing_unrestricted_rank_quadruples} == all_quads-included
            own = declared[pair.site_a, pair.site_b, cov.order]
            assert set(own.missing_declared_rank_quadruples) == model_quads-included
            assert own.declared_model_complete == (model_quads == included)
            assert cov.unrestricted_complete == (all_quads == included)
    if ranks == (4,):
        cov = out.pairs[0].coverage
        assert [len(c.included_rank_quadruples) for c in cov] == [1, 4, 10, 20, 31, 36, 34]
        assert [len(c.missing_table_rank_quadruples) for c in cov] == [0, 0, 0, 0, 0, 4, 10]
        assert not declared[0, 0, 12].declared_model_complete
    else:
        assert declared[0, 0, 8].declared_model_complete
        assert not out.pairs[0].coverage[2].unrestricted_complete


def test_static_zero_weight_and_frame_metadata_only():
    r = analytic((2,))
    out = calculate(r)
    changed = [replace(x, model=replace(x.model, sites=tuple(
        replace(s, frame=tuple(map(tuple, np.eye(3)))) for s in x.model.sites))) for x in r]
    changed[0].refined_tensors[0][1:, 1:] *= 1e150
    other = calculate(changed)
    assert [(c.order, c.t, c.u, c.J, c.value) for c in out.pairs[0].coefficients] == [
        (c.order, c.t, c.u, c.J, c.value) for c in other.pairs[0].coefficients]


@pytest.mark.parametrize('mutation', [
    'shape', 'asymmetry', 'nonfinite', 'charge', 'channels', 'sites', 'variables',
    'rank0', 'frequency_order', 'frequency_duplicate', 'failed_fit',
])
def test_invalid_tensors_declarations_rejected_before_native_allocation(mutation, monkeypatch):
    r = analytic()
    first = r[0]
    if mutation == 'shape':
        r[0] = replace(first, refined_tensors=(np.eye(8), first.refined_tensors[1]))
    elif mutation == 'asymmetry':
        a = first.refined_tensors[0]
        a[1, 2] = np.nextafter(a[2, 1], np.inf)
    elif mutation == 'nonfinite':
        first.refined_tensors[0][1, 1] = np.nan
    elif mutation == 'charge':
        first.refined_tensors[0][0, 1] = first.refined_tensors[0][1, 0] = 1e-100
    elif mutation == 'channels':
        r[0] = replace(first, model=replace(first.model, channel_labels=first.model.channel_labels[::-1]))
    elif mutation == 'sites':
        r[1] = replace(r[1], model=replace(r[1].model, sites=r[1].model.sites[::-1]))
    elif mutation == 'variables':
        r[1] = replace(r[1], model=replace(r[1].model, parameter_labels=('changed',)))
    elif mutation == 'rank0':
        sites = (replace(first.model.sites[0], rank_limit=0),)
        r[0] = replace(first, model=replace(first.model, sites=sites))
    elif mutation == 'frequency_order':
        r = r[::-1]
    elif mutation == 'frequency_duplicate':
        r[1] = replace(r[1], model=replace(r[1].model, frequency_au=0.))
    else:
        r[0] = replace(first, status=core.IsaPfitStatus.NumericalFailure)
    def forbidden(*args, **kwargs):
        pytest.fail('native local model allocated before validation')
    monkeypatch.setattr(core, 'IsaAnisotropicModel', forbidden)
    with pytest.raises((ValueError, TypeError)):
        calculate(r)


@pytest.mark.parametrize('weights', [(1., .4, .2), (0., 0., .2), (0., -.1, .2),
                                     (0., np.inf, .2), (0., np.nan, .2), (0., .4), [[0., .4, .2]]])
def test_invalid_weights_rejected_before_contraction(weights, monkeypatch):
    monkeypatch.setattr(core, 'isa_recoupled_dispersion', lambda *a: pytest.fail('contracted invalid weights'))
    with pytest.raises(ValueError, match='CP weights'):
        calculate(weights=weights)


@pytest.mark.parametrize('kind', ['bytes', 'work', 'cumulative_work'])
def test_shared_ledger_refuses_before_model_allocations(kind, monkeypatch):
    r = analytic()
    numeric, work = adapter.recoupled_plan(r[0].model.sites, len(r), 10)
    ledger = budget(numeric-1 if kind == 'bytes' else numeric,
                    work-1 if kind == 'work' else work)
    if kind == 'cumulative_work':
        ledger.admit('prior response work', 0, 1)
    monkeypatch.setattr(core, 'IsaAnisotropicModel', lambda *a: pytest.fail('allocated before ledger'))
    with pytest.raises(ValueError, match='resource limit'):
        calculate(r, ledger=ledger)


def test_real_driver_optional_budget_refuses_before_factors(monkeypatch, tmp_path):
    """Same finite budget reaches factors ordinarily, but not with recoupling.

    Mock only the wavefunction/state/basis boundary: run the real driver input
    checks, optional planner and shared ledger. The factor sentinel stops the
    admitted old path before any integrals/response/LW/PFIT can execute.
    """
    from types import SimpleNamespace
    from psi4.driver.procrouting.isapol_native_partition import BasisRecipe, ShellRecipe

    sites = analytic()[0].model.sites
    origins = np.asarray([s.origin_bohr for s in sites])
    quadrature = Quadrature((0., .7, 2.), (0., .4, .2), CP_PROVENANCE)
    grid = np.array([[0., 0., 0., 1.]], dtype=np.float64)
    state = SimpleNamespace(nbf=2, nocc=1, nvir=1, planned_bytes=64,
                            orbitals=lambda: np.eye(2),
                            energies=lambda: np.array([-1., 1.]))
    wfn = SimpleNamespace(molecule=lambda: SimpleNamespace(geometry=lambda: origins))
    main = SimpleNamespace(transform=np.eye(2), basis=object())
    recipe = BasisRecipe('analytic AUX', 'mocked driver-budget boundary', 'Cartesian',
                         tuple(map(tuple, origins)), (ShellRecipe(0, 0, (1.,), (1.,)),))
    lattice = core.FitPointsOptions()
    lattice.npoints = 32
    numeric, _ = adapter.recoupled_plan(sites, len(quadrature.frequencies), 10)
    reserved = 2*grid.nbytes+state.planned_bytes+32*state.nbf**2
    resources = BoundedResources(numeric+reserved-1, 6_000_000_000_000, 64*1024**3)
    monkeypatch.setattr(bounded, 'validate_correction', lambda *a, **kw: 'mocked NONE')
    monkeypatch.setattr(bounded, 'require_scf_seal', lambda *a: None)
    monkeypatch.setattr(bounded, 'native_restricted_state_from_wavefunction', lambda *a, **kw: state)
    monkeypatch.setattr(bounded, 'adapt_main', lambda *a, **kw: main)
    monkeypatch.setattr(bounded, '_context', lambda *a: 'mocked-state-no-SCF')
    monkeypatch.setattr(BasisRecipe, 'build', lambda *a: SimpleNamespace(nfunction=1))
    calls = []

    class FactorBoundary(RuntimeError):
        pass

    def factors(*args, **kwargs):
        calls.append(kwargs)
        raise FactorBoundary('admitted to factor construction; no factors computed')

    monkeypatch.setattr(bounded, 'native_plain_df_operators', factors)
    kwargs = dict(caller_converged=True, distribution='df_centre_analytic',
        sites=sites, bonds=[(0, 1)], quadrature=quadrature, response_grid=grid,
        smoothing=api.settings('water', {})['smoothing'], shell_cutoff=1e-8,
        charge_penalty=1000., anchor_metric_damping=.0005, lattice_options=lattice,
        localization_rank_limit=2, weight_type=3, weight_coefficient=1e-3, cutoff=1e-4,
        resources=resources, scf_correction='NONE', max_order=10,
        scratch_directory=tmp_path, log=StageLog(0))
    with pytest.raises(FactorBoundary, match='admitted to factor construction'):
        bounded.bounded_properties(wfn, recipe, **kwargs)
    assert len(calls) == 1
    assert calls[0]['max_bytes'] == resources.max_bytes-reserved
    calls.clear()
    with pytest.raises(ValueError, match='native recoupled preflight: shared numeric byte resource limit'):
        bounded.bounded_properties(wfn, recipe, **kwargs, recoupled_dispersion=True)
    assert calls == []
    assert not list(tmp_path.iterdir())


def test_legacy_seven_positional_fields_and_no_optional_work():
    r = bounded.BoundedProperties((), (), (), 'isotropic', (), {}, {})
    assert r.dispersion == 'isotropic' and r.recoupled_dispersion is None


def test_real_common_final_stage_old_calls_do_not_recouple(monkeypatch):
    r = analytic()
    q = Quadrature((0., .7, 2.), (0., .4, .2), CP_PROVENANCE)
    ledger = budget()
    def forbidden(*args, **kwargs):
        pytest.fail('ordinary request paid optional recoupling cost')
    monkeypatch.setattr(adapter, 'refined_recoupled_dispersion', forbidden)
    monkeypatch.setattr(core, 'IsaAnisotropicModel', forbidden)
    monkeypatch.setattr(core, 'IsaRecoupledModel', forbidden)
    monkeypatch.setattr(core, 'isa_recoupled_dispersion', forbidden)
    iso, angular = bounded._refined_dispersions(r, q, 10, ledger)
    assert iso.pairs and angular is None
    assert ledger.work == ledger.peak == 0


@pytest.mark.parametrize('order', [True, 5, 13, 6., '8'])
def test_invalid_orders_fail_before_contraction(order, monkeypatch):
    monkeypatch.setattr(core, 'IsaAnisotropicModel', lambda *a: pytest.fail('allocated invalid order'))
    with pytest.raises(ValueError, match='max_order'):
        calculate(order=order)


@pytest.mark.parametrize('field,value', [('rank_limit', -1), ('rank_limit', 5),
    ('rank_limit', True), ('frame', np.diag([1., 1., -1.])), ('frame', np.ones((2, 2)))])
def test_invalid_site_rank_or_frame_is_not_admitted(field, value):
    with pytest.raises(ValueError):
        replace(analytic()[0].model.sites[0], **{field: value})


def test_one_ulp_frequency_and_zero_rank_blocks_preserved():
    frequencies = (0., np.nextafter(.7, 1.), 2.)
    r = analytic((2, 1), frequencies)
    for node in r:
        node.refined_tensors[0][4:, :] = 0.
        node.refined_tensors[0][:, 4:] = 0.
    out = calculate(r)
    assert out.frequencies == list(frequencies)
    assert out.frequencies[1] != .7
    assert out.model.source.sites[0].ranks == [1, 2]
    assert (2, 2, 2, 2) in map(tuple, out.pairs[0].coverage[-1].included_rank_quadruples)


@pytest.fixture
def live_unsealed_water():
    return core.Wavefunction.build(psi4.geometry(
        '0 1\nO 0 0 0\nH .757 0 .586\nH -.757 0 .586\nsymmetry c1\nno_com\nno_reorient'), 'sto-3g')


@pytest.mark.parametrize('tasks,expected', [
    (('ATOMIC_REFINED_POLARIZABILITIES',), False),
    (('ATOMIC_REFINED_DISPERSION',), False),
    ((TASK,), True),
    (('ATOMIC_REFINED_POLARIZABILITIES', 'ATOMIC_REFINED_DISPERSION', TASK), True),
])
def test_public_opt_in_combined_chain_once(live_unsealed_water, monkeypatch, tasks, expected):
    calls = []
    def backend(wfn, recipe, **kwargs):
        calls.append(kwargs)
        assert kwargs.get('recoupled_dispersion', False) is expected
        return bounded.BoundedProperties((), (), (), 'isotropic', (), {}, {},
                                         'recoupled' if expected else None)
    monkeypatch.setattr(api, 'bounded_properties', backend)
    assert psi4.oeprop(live_unsealed_water, *tasks, atomic_backend='BOUNDED_DF', preset='water',
        npoints=32, radial_points=20, spherical_points=50, log=StageLog(0)) is None
    assert len(calls) == 1
    result = psi4.atomic_property_result(live_unsealed_water)
    assert result.dispersion == 'isotropic'
    assert (result.recoupled_dispersion is not None) is expected
    if not expected:
        assert 'recoupled_dispersion' not in calls[0]


@pytest.mark.parametrize('tasks,kw', [
    ((TASK,), {}), ((TASK, 'DIPOLE'), {'atomic_backend': 'BOUNDED_DF'}),
    ((TASK, TASK), {'atomic_backend': 'BOUNDED_DF'}),
    (('ATOMIC_RECOUPLED_DISPERSION',), {'atomic_backend': 'BOUNDED_DF'}),
    ((TASK,), {'atomic_backend': 'BOUNDED_DF', 'recoupled_dispersion': True}),
])
def test_request_refusal_invalidates_owned_result(live_unsealed_water, tasks, kw):
    live_unsealed_water._native_atomic_property_result = object()
    with pytest.raises((ValueError, TypeError, psi4.driver.p4util.exceptions.ValidationError)):
        psi4.oeprop(live_unsealed_water, *tasks, preset='water', **kw)
    with pytest.raises(ValueError, match='No native'):
        psi4.atomic_property_result(live_unsealed_water)


def test_late_optional_failure_invalidates_owned_result(live_unsealed_water, monkeypatch):
    # Cross the real public dispatcher and real adapter with analytic PFIT
    # results; inject failure only at the optional native contraction.
    def backend(*args, **kwargs):
        assert kwargs['recoupled_dispersion']
        return calculate()
    monkeypatch.setattr(api, 'bounded_properties', backend)
    def fail(*args):
        raise ValueError('late optional contraction failure')
    monkeypatch.setattr(core, 'isa_recoupled_dispersion', fail)
    live_unsealed_water._native_atomic_property_result = object()
    with pytest.raises(ValueError, match='late optional'):
        psi4.oeprop(live_unsealed_water, TASK, atomic_backend='BOUNDED_DF', preset='water',
                   npoints=32, radial_points=20, spherical_points=50, log=StageLog(0))
    with pytest.raises(ValueError, match='No native'):
        psi4.atomic_property_result(live_unsealed_water)


def test_native_water_public_rank2_chain(tmp_path, monkeypatch):
    """One fresh sealed PBE0 water, no replay arrays, one combined chain.

    Explicit test-size orbital/grid/lattice declarations are NOT new defaults.
    The property AUX, penalties, cutoff, ranks and 10-point CP grid stay preset.
    """
    psi4.set_memory(4*1024**3)
    psi4.set_num_threads(1)
    molecule = psi4.geometry(
        '0 1\nO 0 0 0\nH .757 0 .586\nH -.757 0 .586\n'
        'units angstrom\nsymmetry c1\nno_com\nno_reorient')
    psi4.set_options({'basis': 'sto-3g', 'reference': 'rks', 'scf_type': 'pk',
        'e_convergence': 1e-10, 'd_convergence': 1e-10,
        'dft_radial_points': 50, 'dft_spherical_points': 110})
    energy, wfn = psi4.energy('pbe0', molecule=molecule, return_wfn=True)
    messages, calls = [], []
    real = api.bounded_properties
    def counted(*args, **kwargs):
        calls.append(True)
        return real(*args, **kwargs)
    monkeypatch.setattr(api, 'bounded_properties', counted)
    assert psi4.oeprop(wfn, 'ATOMIC_REFINED_POLARIZABILITIES', 'ATOMIC_REFINED_DISPERSION', TASK,
        atomic_backend='BOUNDED_DF', preset='water', npoints=32,
        radial_points=20, spherical_points=50, scratch_directory=tmp_path,
        log=StageLog(3, writer=messages.append)) is None
    assert len(calls) == 1 and not list(tmp_path.iterdir())
    result = psi4.atomic_property_result(wfn)
    out = result.recoupled_dispersion
    assert out is not None and len(result.frequencies) == 11
    assert out.frequencies == list(result.frequencies)
    assert out.cp_weights[0] == 0 and all(w > 0 for w in out.cp_weights[1:])
    assert out.cp_weights == list(result.provenance['cp_weights'])
    assert [s.ranks for s in out.model.source.sites] == [[1, 2]]*3
    assert len(out.pairs) == 9 and out.max_order == 10
    assert len(result.refined_tensors) == 11
    for node, tensors in enumerate(result.refined_tensors):
        for i, a in enumerate(tensors):
            assert a.shape == (9, 9) and np.isfinite(a).all()
            np.testing.assert_array_equal(a, a.T)
            np.testing.assert_array_equal(a[0], np.zeros(9))
            np.testing.assert_array_equal(out.model.source.sites[i].responses[node].np, a[1:, 1:])
    assert any(np.ptp(np.diag(a)[1:4]) > 1e-5 for a in result.refined_tensors[0])
    assert any(np.max(np.abs(a[1:4, 4:9])) > 1e-5 for a in result.refined_tensors[0])
    assert any((c.t != 1 or c.u != 1) and abs(c.value) > 1e-7
               for p in out.pairs for c in p.coefficients)
    for p, iso in zip(out.pairs, result.dispersion.pairs):
        assert all(np.isfinite(c.value) for c in p.coefficients)
        for c in iso.coefficients:
            assert p.coefficient(c.order, 1, 1, 0) == pytest.approx(c.value, rel=3e-13, abs=1e-13)
        assert [c.order for c in p.coverage] == [6, 7, 8, 9, 10]
        assert p.coverage[0].unrestricted_complete
        assert not p.coverage[2].unrestricted_complete
        assert all(max(q) <= 2 for c in p.coverage for q in c.included_rank_quadruples)
    assert all(c.declared_model_complete for c in out.declared_model_coverage)
    assert max(d['response_residual'] for d in result.diagnostics) < 1e-10
    assert sum(s['stage'] == 'native recoupled dispersion' for s in result.resources['stages']) == 1
    evidence = os.environ.get('CCASP_RECOUPLED_EVIDENCE')
    if evidence:
        payload = dict(method='restricted PBE0/sto-3g; PK; plain DF/aug-cc-pVTZ-RI; Slater/PW92 ALDA',
            units='atomic units; origins bohr; Cn Eh bohr^n; imaginary omega au; CP includes 1/(2pi)',
            energy_Eh=energy, geometry_bohr=molecule.geometry().np.tolist(),
            settings=dict(npoints=32, response_radial_points=20, response_spherical_points=50,
                          scf_radial_points=50, scf_spherical_points=110, preset='water'),
            sites=[dict(label=s.label, ranks=s.ranks, origin=s.origin, frame=s.frame,
                        components=s.components) for s in out.model.source.sites],
            frequencies=out.frequencies, cp_weights=out.cp_weights,
            local_tensors=[t.array.tolist() for t in result.local_tensors],
            refined_tensors=[[a.tolist() for a in node] for node in result.refined_tensors],
            pairs=[dict(site_a=p.site_a, site_b=p.site_b, labels=out.pair_labels[i],
                coefficients=[dict(order=c.order, t=c.t, u=c.u, J=c.J, value=c.value)
                              for c in p.coefficients],
                coverage=[dict(order=c.order, table_complete=c.table_complete,
                    unrestricted_complete=c.unrestricted_complete,
                    included_rank_quadruples=c.included_rank_quadruples,
                    missing_table_rank_quadruples=c.missing_table_rank_quadruples,
                    missing_unrestricted_rank_quadruples=c.missing_unrestricted_rank_quadruples)
                    for c in p.coverage]) for i, p in enumerate(out.pairs)],
            declared_model_coverage=[asdict(c) for c in out.declared_model_coverage],
            diagnostics=result.diagnostics, resources=result.resources, provenance=result.provenance,
            recoupled_provenance=out.provenance, quadrature_provenance=asdict(out.quadrature_provenance))
        Path(evidence, 'water.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
        Path(evidence, 'water-stages.log').write_text(''.join(messages))
