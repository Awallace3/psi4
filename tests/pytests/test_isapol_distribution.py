"""Owned distributed-moment contract; analytic DF-centre and fresh molecular ISA -> LW -> PFIT -> C6."""
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest
import psi4
from psi4 import core
from psi4.driver.procrouting import isapol_df_multipoles as dfm
from psi4.driver.procrouting import isapol_distribution as dist
from psi4.driver.procrouting import isapol_native_partition as isa
from psi4.driver.procrouting import isapol_bounded_response as response
from psi4.driver.procrouting.isapol_basis import BasisRecipe, ShellRecipe
from psi4.driver.procrouting.isapol_bounded_response import _Ledger, BoundedResources
from psi4.driver.procrouting.isapol_native import Quadrature
from psi4.driver.procrouting.isapol_logging import StageLog
from isapol_water_recipe import native_water_recipe


@pytest.fixture
def small_q():
    aux = BasisRecipe('s,p', 'unit primitive test', 'Cartesian', ((0., 0., 0.),),
                      (ShellRecipe(0, 0, (1.,), (1.,)), ShellRecipe(0, 1, (1.,), (1.,))))
    site = core.IsaMultipoleSite()
    site.label, site.origin, site.rank = 'X', [0., 0., 0.], 1
    return aux, [site], dist.analytic_df_moments(aux, [site], 1)


def test_df_contract_preserves_analytic_values_and_contraction(small_q):
    aux, sites, q = small_q
    old = dfm.analytic_df_centre_multipoles(aux, sites, 1)
    np.testing.assert_array_equal(q.values, old.values)
    fit = np.arange(12.).reshape(3, 4)
    np.testing.assert_array_equal(q.anchor_legs(fit), fit @ old.values.T)
    # Racah 10,11c,11s are z,x,y, unlike the AUX's x,y,z columns.
    assert np.argmax(abs(q.values[1, 1:])) == 2
    assert np.argmax(abs(q.values[2, 1:])) == 0
    assert np.argmax(abs(q.values[3, 1:])) == 1
    with pytest.raises(ValueError):
        q.values.setflags(write=True)
    copied = replace(q, values=old.values)
    old.values[:] = 0.
    assert np.any(copied.values)
    q.validate_for(aux, sites, 1, 'arbitrary state: DF is density independent')


@pytest.mark.parametrize('change', [
    {'values': np.ones((4, 3))}, {'values': np.full((4, 4), np.nan)},
    {'values': np.ones((4, 4), dtype=complex)*1j},
    {'labels': ('',)}, {'origins_bohr': ((1., 0., 0.),)}, {'rank': True},
    {'convention': 'Cartesian xyz'}, {'converged': False}, {'provenance': ''},
])
def test_malformed_contract_rejected(small_q, change):
    with pytest.raises((ValueError, TypeError)):
        replace(small_q[2], **change)


def test_column_and_density_identity_not_just_dimensions(small_q):
    aux, sites, q = small_q
    with pytest.raises(ValueError, match='identity mismatch'):
        q.validate_for(replace(aux, shells=aux.shells[::-1]), sites, 1, 'state')
    with pytest.raises(ValueError, match='density state'):
        replace(q, state_sha256='original').validate_for(aux, sites, 1, 'other')
    with pytest.raises(ValueError, match='coefficient'):
        q.anchor_legs(np.ones((3, 3)))
    with pytest.raises(ValueError, match='coefficient'):
        q.anchor_legs(np.full((3, 4), np.inf))


def test_distribution_loads_only_basis_and_df_centre_modules():
    """In a fresh process the Q contract is produced and contracted on its own."""
    import subprocess
    import sys
    subprocess.run([sys.executable, '-c', """
import sys
import numpy as np
from psi4 import core
from psi4.driver.procrouting import isapol_distribution as dist
from psi4.driver.procrouting.isapol_basis import BasisRecipe, ShellRecipe
aux = BasisRecipe('s', 'unit primitive test', 'Cartesian', ((0., 0., 0.),),
                  (ShellRecipe(0, 0, (1.,), (1.,)),))
site = core.IsaMultipoleSite()
site.label, site.origin, site.rank = 'X', [0., 0., 0.], 1
q = dist.analytic_df_moments(aux, [site], 1)
np.testing.assert_allclose(q.values[:, 0], [np.pi**1.5, 0., 0., 0.], rtol=1e-14, atol=0)
np.testing.assert_allclose(q.anchor_legs(np.array([[2.]])), [[2*np.pi**1.5, 0., 0., 0.]], rtol=1e-14)
loaded = {m for m in sys.modules if m.startswith('psi4.driver.procrouting.isapol_')}
assert loaded == {'psi4.driver.procrouting.' + m for m in
                  ('isapol_basis', 'isapol_df_multipoles', 'isapol_distribution')}, loaded
"""], check=True)


def resources():
    return BoundedResources(4*1024**3, 6_000_000_000_000, 64*1024**3)


def multipole_sites(recipe):
    sites = []
    for s in recipe.sites:
        site = core.IsaMultipoleSite()
        site.label, site.origin, site.rank = s.label, s.origin, 4
        sites.append(site)
    return sites


def isa_grid(molecule, radial, spherical):
    options = core.IsaGridOptions()
    options.radial_points, options.spherical_points = radial, spherical
    grid = core.IsaGrid(molecule.clone(), options)
    return np.column_stack((grid.x(), grid.y(), grid.z(), grid.w()))


@pytest.fixture(scope='module')
def molecular_water():
    # import psi4 sets one thread, so this runs serially whatever the runner's
    # OMP/MKL settings; memory is restored after the module.
    core.be_quiet()
    saved_memory = core.get_memory()
    psi4.set_memory('4 GiB')
    mol = psi4.geometry('''0 1
O 0 0 0
H -1.45365196 0 -1.12168732
H 1.45365196 0 -1.12168732
units bohr
symmetry c1
no_com
no_reorient
''')
    psi4.set_options({'basis': 'aug-cc-pvtz', 'puream': True, 'reference': 'rks',
        'scf_type': 'pk', 'e_convergence': 1e-12, 'd_convergence': 1e-12})
    energy, wfn = psi4.energy('pbe0', molecule=mol, return_wfn=True)
    recipe = native_water_recipe(mol)
    try:
        yield energy, wfn, recipe, isa_grid(mol, 100, 200)
    finally:
        psi4.set_memory(saved_memory, quiet=True)


WATER_ROUTE = dict(atomic_backend='BOUNDED_DF', preset='water', distribution='isa',
                   quadrature=Quadrature.from_casimir(core.CasimirGrid(2, .5)), npoints=64,
                   rank_limit=1, hydrogen_rank_limit=1, localization_rank_limit=2,
                   weight_type=4, weight_coefficient=1e-5, max_order=6)


@pytest.mark.parametrize('qgrid', [(200, 590), (300, 974)])
def test_isa_molecular_properties(molecular_water, tmp_path, qgrid):
    energy, wfn, recipe, grid = molecular_water
    integration = isa_grid(wfn.molecule(), *qgrid)
    psi4.oeprop(wfn, 'ATOMIC_REFINED_DISPERSION', **WATER_ROUTE,
        partition_recipe=recipe, partition_grid=integration,
        auxiliary_recipe=recipe.auxiliary, response_grid=grid,
        resources=resources(), scratch_directory=tmp_path, log=StageLog(0))
    result = psi4.atomic_property_result(wfn)
    partition = result.provenance['partition']
    diagnostics = partition['diagnostics']
    assert partition['model'] == 'isa' and dist.ISA_EXPERIMENTAL in partition['provenance']
    assert diagnostics['status'] == 'experimental' and diagnostics['iteration_converged'] is True
    assert diagnostics['property_grid_convergence'] == 'not_established'
    assert diagnostics['q_shape'] == (75, 246)
    assert np.isfinite(diagnostics['q_charge_row_error'])
    assert diagnostics['max_delta'] <= 1e-9
    assert 1 < diagnostics['iterations'] < 120
    assert diagnostics['drho_metric_condition'] > 1e14  # limitation, not an accuracy certificate
    assert abs(diagnostics['grid_charge_error']) < 2e-5
    assert diagnostics['integration_grid_sha256'] == hashlib.sha256(integration.tobytes()).hexdigest()
    stages = [s['stage'] for s in result.resources['stages']]
    assert stages.index('ISA iteration and response-AUX Q') < stages.index('retained distributed moments')
    assert all(r.status == core.IsaPfitStatus.Solved for r in result.refinements)
    assert max(d['response_residual'] for d in result.diagnostics) < 1e-10
    assert max(d['localization_residual'] for d in result.diagnostics) < 1e-6
    assert not list(tmp_path.iterdir())
    c6 = [c.value for pair in result.dispersion.pairs for c in pair.coefficients]
    assert np.isfinite(c6).all() and min(c6) > 0
    # Each explicitly declared quadrature has its own regression. They differ
    # by up to 2.5e-4 relative in C6; passing LW is not quadrature convergence.
    oo, oh, hh = ((24.01768063, 4.07528415, .70069657) if qgrid == (200, 590)
                  else (24.01972453, 4.07494578, .70052119))
    np.testing.assert_allclose(c6, [oo, oh, oh, oh, hh, hh, oh, hh, hh], rtol=2e-6)
    if qgrid == (200, 590):
        gold = np.array(
            [7.00509272, 6.89215490, 7.41700397, 2.26077379, .00155196, .78850087, .86528699])
        parameters = np.asarray(result.refinements[0].parameters)
        others = [0, 1, 2, 3, 5, 6]
        np.testing.assert_allclose(parameters[others], gold[others], rtol=2e-6, atol=2e-7)
        # Index 4, H1_10_11c_A (bohr^3), is the small off-diagonal H dipole
        # polarizability; it does not enter C6. It moves 2e-7 to 5e-7 absolute
        # with the MKL path downstream of Q, so it has an empirical absolute band
        # (one host, oneMKL, OFF/COMPATIBLE), not an accuracy statement.
        assert abs(parameters[4] - gold[4]) <= 1e-6
    print('MOLECULAR_EVIDENCE', json.dumps(dict(energy=energy, partition=diagnostics,
        response=result.diagnostics, c6=c6,
        static_parameters=result.refinements[0].parameters,
        resources=result.resources)))


def test_final_tails_on_a_different_response_aux(molecular_water, monkeypatch):
    _, wfn, recipe, grid = molecular_water
    # Different function space and dimension, not a renamed density AUX.
    auxiliary = replace(recipe.auxiliary, name='response s,p subset',
                        shells=tuple(s for s in recipe.auxiliary.shells if s.l <= 1))
    original_partition, original_sample = isa.native_partition, isa.final_shape_samples
    captured = {}
    def partition(*args, **kwargs):
        assert kwargs['build_multipoles'] is False
        result = original_partition(*args, **kwargs)
        assert result.q is None and result.shape_samples == () and result.converged
        captured['partition'] = result
        return result
    def sample(shapes, state, sites, points, *, tails):
        result = captured['partition']
        fields = lambda values: [(t.defined, t.amplitude, t.exponent, t.cutoff) for t in values]
        assert fields(tails) == fields(result.trajectory.final_tails)
        assert fields(tails) != fields(result.trajectory.state.tails)
        values = original_sample(shapes, state, sites, points, tails=tails)
        captured['samples'] = values
        return values
    monkeypatch.setattr(isa, 'native_partition', partition)
    monkeypatch.setattr(isa, 'final_shape_samples', sample)
    caller_grid = grid.copy()
    q = dist.isa_moments(wfn, recipe, auxiliary, multipole_sites(recipe), 1,
        caller_converged=True, integration_grid=caller_grid, ledger=_Ledger(resources()))
    assert q.values.shape == (12, auxiliary.build('MolecularAux').nfunction)
    assert q.values.shape[1] != recipe.auxiliary.build('MolecularAux').nfunction
    assert q.auxiliary_sha256 != dict(q.diagnostics)['density_auxiliary_sha256']
    # The producer owns its grid and Q: mutating the caller grid afterwards
    # changes neither the returned Q nor its recorded grid identity.
    before = q.values.copy()
    caller_grid[:] = 0.
    np.testing.assert_array_equal(q.values, before)
    assert dict(q.diagnostics)['integration_grid_sha256'] == hashlib.sha256(grid.tobytes()).hexdigest()
    total = np.sum(captured['samples'], axis=0)
    valid = abs(total) > recipe.controller.density_cutoff
    ratios = np.asarray(captured['samples'])[:, valid]/total[valid]
    np.testing.assert_allclose(ratios.sum(axis=0), 1., atol=3e-16)
    # Molecular charge and z,x,y dipoles on the SAME full quadrature, including
    # the native denominator exclusion; no analytic-integral parity assumption.
    points, weights = grid[valid, :3], grid[valid, 3]
    chi = np.asarray(auxiliary.build('MolecularAux').evaluate(points.tolist()))
    global_q = np.column_stack((np.ones(len(points)), points[:, [2, 0, 1]])).T @ (weights[:, None]*chi)
    translate = np.zeros((4, 12))
    for a, origin in enumerate(q.origins_bohr):
        translate[:, 4*a:4*a+4] = np.eye(4)
        translate[1:, 4*a] = np.asarray(origin)[[2, 0, 1]]
    np.testing.assert_allclose(translate @ q.values, global_q, atol=2e-10, rtol=2e-12)


def test_coarse_q_grid_does_not_bypass_lw_gate(molecular_water, tmp_path):
    _, wfn, recipe, grid = molecular_water
    with pytest.raises(RuntimeError, match='postcondition exceeds residual tolerance'):
        psi4.oeprop(wfn, 'ATOMIC_REFINED_DISPERSION', **WATER_ROUTE,
            partition_recipe=recipe, partition_grid=grid, response_grid=grid,
            auxiliary_recipe=recipe.auxiliary,
            resources=resources(), scratch_directory=tmp_path, log=StageLog(0))
    with pytest.raises(ValueError, match='No native'):
        psi4.atomic_property_result(wfn)
    assert not list(tmp_path.iterdir())


def test_public_nonconvergence_stops_before_response(molecular_water, monkeypatch, tmp_path):
    _, wfn, recipe, grid = molecular_water
    failed = replace(recipe, controller=replace(recipe.controller, max_iterations=1))
    def forbidden(*args, **kwargs):
        pytest.fail('response factors built after ISA nonconvergence')
    monkeypatch.setattr(response, 'native_plain_df_operators', forbidden)
    messages = []
    with pytest.raises(RuntimeError, match='ISA-A did not converge'):
        psi4.oeprop(wfn, 'ATOMIC_REFINED_DISPERSION', atomic_backend='BOUNDED_DF', preset='water',
            distribution='isa', partition_recipe=failed, partition_grid=grid, response_grid=grid,
            auxiliary_recipe=recipe.auxiliary, npoints=32, resources=resources(),
            scratch_directory=tmp_path, log=StageLog(1, writer=messages.append))
    assert 'Stage FAILED: Native ISA partition and response-AUX moments' in ''.join(messages)
    with pytest.raises(ValueError, match='No native'):
        psi4.atomic_property_result(wfn)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('stale', ['moved site', 'site order', 'recipe type', 'grid dtype'])
def test_stale_isa_inputs_refused_before_scratch_or_response(molecular_water, monkeypatch, tmp_path, stale):
    _, wfn, recipe, grid = molecular_water
    monkeypatch.setattr(isa, 'native_partition', lambda *a, **k: pytest.fail('ISA ran on stale input'))
    monkeypatch.setattr(response, 'native_plain_df_operators', lambda *a, **k: pytest.fail('response ran'))
    kwargs = dict(partition_recipe=recipe, partition_grid=grid)
    if stale == 'moved site':
        # A complete recipe declared for an earlier geometry (H1 moved 0.05 bohr).
        moved = wfn.molecule().clone()
        xyz = moved.geometry().np.copy()
        xyz[1, 0] += .05
        moved.set_geometry(core.Matrix.from_array(xyz))
        kwargs['partition_recipe'] = native_water_recipe(moved)
    elif stale == 'site order':
        kwargs['partition_recipe'] = replace(recipe, sites=recipe.sites[::-1])
    elif stale == 'recipe type':
        kwargs['partition_recipe'] = recipe.auxiliary
    else:
        kwargs['partition_grid'] = grid.astype(np.float32)
    messages = []
    with pytest.raises((ValueError, TypeError), match='identity mismatch|PartitionRecipe|integration grid'):
        psi4.oeprop(wfn, 'ATOMIC_REFINED_DISPERSION', atomic_backend='BOUNDED_DF', preset='water',
            distribution='isa', response_grid=grid, auxiliary_recipe=recipe.auxiliary, npoints=32,
            resources=resources(), scratch_directory=tmp_path, log=StageLog(1, writer=messages.append),
            **kwargs)
    assert 'Stage FAILED: Bounded input validation' in ''.join(messages)
    with pytest.raises(ValueError, match='No native'):
        psi4.atomic_property_result(wfn)
    assert not list(tmp_path.iterdir())


def test_isa_nonconvergence_and_admission(molecular_water, monkeypatch):
    _, wfn, recipe, grid = molecular_water
    sites = multipole_sites(recipe)
    failed = replace(recipe, controller=replace(recipe.controller, max_iterations=1))
    with pytest.raises(RuntimeError, match='ISA-A did not converge'):
        dist.isa_moments(wfn, failed, recipe.auxiliary, sites, 4, caller_converged=True,
                         integration_grid=grid, ledger=_Ledger(resources()))
    class Admitted(Exception):
        pass
    def admitted(*args, **kwargs):
        raise Admitted
    monkeypatch.setattr(isa, 'native_partition', admitted)
    # Exact plan admits; one byte or one work unit short refuses before ISA runs.
    numeric, work = dist.isa_resource_plan(wfn, recipe, recipe.auxiliary, grid, 4)
    with pytest.raises(Admitted):
        dist.isa_moments(wfn, recipe, recipe.auxiliary, sites, 4, caller_converged=True,
                         integration_grid=grid, ledger=_Ledger(BoundedResources(numeric, work, 1024)))
    for budget, message in ((BoundedResources(numeric-1, work, 1024), 'byte'),
                            (BoundedResources(numeric, work-1, 1024), 'work')):
        with pytest.raises(ValueError, match=message):
            dist.isa_moments(wfn, recipe, recipe.auxiliary, sites, 4, caller_converged=True,
                             integration_grid=grid, ledger=_Ledger(budget))
    for bad in (None, grid[:, :3], grid.astype(np.float32), grid*float('nan')):
        with pytest.raises(ValueError, match='integration grid'):
            dist.isa_moments(wfn, recipe, recipe.auxiliary, sites, 4, caller_converged=True,
                             integration_grid=bad, ledger=_Ledger(resources()))
    with pytest.raises(ValueError, match='identity mismatch'):
        dist.isa_moments(wfn, recipe, recipe.auxiliary, sites[::-1], 4, caller_converged=True,
                         integration_grid=grid, ledger=_Ledger(resources()))
    with pytest.raises(ValueError, match='caller_converged'):
        dist.isa_moments(wfn, recipe, recipe.auxiliary, sites, 4, caller_converged=False,
                         integration_grid=grid, ledger=_Ledger(resources()))
