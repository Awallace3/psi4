"""Owned distributed-moment contract; analytic DF-centre and fresh molecular ISA and MBIS -> LW -> PFIT -> C6."""
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

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


def test_isa_plan_charges_drho_refinement_once_and_to_the_cap(molecular_water, monkeypatch):
    _, wfn, recipe, grid = molecular_water
    # The plan's d is the dimension the ISA stage actually solves, and refinement
    # there stops within the cap the plan charges.
    main = isa.adapt_main(wfn, caller_converged=True)
    coulomb = core.IsaAuxCoulomb(recipe.auxiliary.build('MolecularAux'))
    drho = coulomb.fit_drho_c(main.basis, core.Matrix.from_array(main.occupied), 1000.,
                              max_refinement_iterations=isa.DRHO_REFINEMENT_ITERATIONS)
    assert len(drho.rhs) == len(drho.coefficients) == dist._width(recipe.auxiliary)
    assert 1 <= drho.refinement_iterations <= isa.DRHO_REFINEMENT_ITERATIONS
    def plan(cap, auxiliary):
        monkeypatch.setattr(isa, 'DRHO_REFINEMENT_ITERATIONS', cap)
        # The plan reads only these recipe fields; a smaller AUX need not partition.
        shaped = SimpleNamespace(sites=recipe.sites, grid=recipe.grid,
                                 controller=recipe.controller, auxiliary=auxiliary)
        return dist.isa_resource_plan(wfn, shaped, recipe.auxiliary, grid, 4)
    smaller = replace(recipe.auxiliary, shells=recipe.auxiliary.shells[:-3])
    charged = {}
    for auxiliary in (recipe.auxiliary, smaller):
        d = dist._width(auxiliary)
        base, one, ten, cap = (plan(k, auxiliary) for k in (0, 1, 10, 32))
        # Bytes are held once for every iteration count, and cover what
        # isa_refined_lu_solve owns: x, the plain LU x, the correction (each
        # residual is written into it) and the 134-digit int64 accumulator.
        held = one[0]-base[0]
        assert ten[0]-base[0] == cap[0]-base[0] == held >= 8*3*d+8*134
        # Work is charged for every iteration up to the cap, not the observed count.
        step = one[1]-base[1]
        assert ten[1]-base[1] == 10*step and cap[1]-base[1] == 32*step
        charged[d] = held, step
    (d1, (held1, step1)), (d2, (held2, step2)) = sorted(charged.items(), reverse=True)
    # Both charges grow with the solve dimension: bytes linearly, work with the
    # d*(d+1) residual product terms.
    assert d1 > d2 and held1-held2 >= 8*3*(d1-d2)
    assert step1-step2 >= 256*(d1*(d1+1)-d2*(d2+1))


# MBIS: native all-shell proatoms feed the same response -> LW -> PFIT -> C6 stages.
MBIS_OPTIONS = {'mbis_radial_points': 75, 'mbis_spherical_points': 302,
                'mbis_pruning_scheme': 'robust', 'mbis_d_convergence': 1e-8, 'mbis_maxiter': 500}


def clear_mbis(wfn):
    for name in list(wfn.scalar_variables()):
        if name.startswith('MBIS'):
            wfn.del_scalar_variable(name)
    for name in list(wfn.array_variables()):
        if name.startswith('MBIS'):
            wfn.del_array_variable(name)


@pytest.fixture
def mbis_water(molecular_water):
    """Shared SCF with no MBIS state before or after, and declared MBIS options.

    MBIS reads, but never writes, the density; the seal check proves it is unchanged.
    conftest restores options after every test.
    """
    wfn = molecular_water[1]
    state = dist_state(wfn)
    clear_mbis(wfn)
    psi4.set_options(MBIS_OPTIONS)
    yield molecular_water
    clear_mbis(wfn)
    assert dist_state(wfn) == state


def dist_state(wfn):
    from psi4.driver.procrouting.isapol_native import _context
    return _context(wfn)


def mbis_q(wfn, auxiliary, sites, rank, grid, budget=None):
    return dist.mbis_moments(wfn, auxiliary, sites, rank, caller_converged=True,
                             integration_grid=grid, ledger=_Ledger(budget or resources()))


def rank1_sites(recipe):
    sites = multipole_sites(recipe)
    for s in sites:
        s.rank = 1
    return sites


def test_mbis_all_shell_weights_and_q(mbis_water, monkeypatch):
    _, wfn, recipe, grid = mbis_water
    monkeypatch.setattr(isa, 'native_partition', lambda *a, **k: pytest.fail('MBIS ran ISA'))
    # A response AUX with a different function space from any density-fit AUX.
    auxiliary = replace(recipe.auxiliary, name='response s,p subset',
                        shells=tuple(s for s in recipe.auxiliary.shells if s.l <= 1))
    sites = rank1_sites(recipe)
    caller_grid = grid.copy()
    q = mbis_q(wfn, auxiliary, sites, 1, caller_grid)
    counts, populations, widths = (wfn.array_variable(n).np.copy() for n in dist.MBIS_SNAPSHOT)
    np.testing.assert_array_equal(counts[:, 0], [2, 1, 1])
    diagnostics = dict(q.diagnostics)
    assert diagnostics['shell_counts'] == (2, 1, 1)
    np.testing.assert_array_equal(diagnostics['shell_widths_bohr'], widths)
    np.testing.assert_array_equal(diagnostics['shell_populations'], populations)
    assert diagnostics['excluded_denominators'] == diagnostics['negative_ratios'] == (0, 0, 0)
    assert diagnostics['denominator_cutoff'] == 0.
    assert diagnostics['weight_sum_error'] < 1e-15
    assert dict(diagnostics['native_options'])['MBIS_SPHERICAL_POINTS'] == 302
    assert dict(diagnostics['native_grid_options'])['DFT_GRID_NAME'] == ''
    assert q.model == 'mbis' and q.state_sha256 == dist_state(wfn)
    assert q.values.shape == (12, auxiliary.build('MolecularAux').nfunction) and not q.values.flags.writeable
    q.validate_for(auxiliary, sites, 1, dist_state(wfn))
    with pytest.raises(ValueError, match='identity mismatch'):
        q.validate_for(recipe.auxiliary, sites, 1, dist_state(wfn))
    # Owned: later caller-grid or wavefunction-variable changes alter neither Q
    # nor its recorded identities.
    before = q.values.copy()
    caller_grid[:] = 0.
    wfn.array_variable('MBIS SHELL WIDTHS').np[:] = 1.
    np.testing.assert_array_equal(q.values, before)
    assert diagnostics['integration_grid_sha256'] == hashlib.sha256(grid.tobytes()).hexdigest()
    np.testing.assert_array_equal(dict(q.diagnostics)['shell_widths_bohr'], widths)

    # Independent linear-domain proatoms from the published arrays, every shell.
    points, weights = grid[:, :3], grid[:, 3]
    geometry = wfn.molecule().geometry().np
    distance = np.linalg.norm(points[None] - geometry[:, None], axis=2)
    def proatoms(shells):
        return np.array([sum(populations[a, s]*np.exp(-distance[a]/widths[a, s])/(8*np.pi*widths[a, s]**3)
                             for s in shells(a)) for a in range(3)])
    rho = proatoms(lambda a: range(int(counts[a, 0])))
    total = rho.sum(axis=0)
    valid = total > 1e-300
    chi = np.asarray(auxiliary.build('MolecularAux').evaluate(points.tolist()))
    # Where the linear sum underflows, every AUX function has vanished too.
    assert np.abs(chi[~valid]).max(initial=0.) < 1e-100
    def direct(rho):
        share = rho[:, valid]/rho[:, valid].sum(axis=0)
        rows = []
        for a in range(3):
            r = points[valid] - geometry[a]
            harmonics = np.column_stack((np.ones(valid.sum()), r[:, [2, 0, 1]]))  # Racah 00,10,11c,11s
            rows.append(harmonics.T @ ((weights[valid]*share[a])[:, None]*chi[valid]))
        return np.vstack(rows)
    np.testing.assert_allclose(q.values, direct(rho), atol=1e-12, rtol=1e-10)
    # Site charge and z,x,y dipole rows translate back to the molecular moments
    # on the same quadrature: the weights partition unity.
    global_q = np.column_stack((np.ones(len(points)), points[:, [2, 0, 1]])).T @ (weights[:, None]*chi)
    translate = np.zeros((4, 12))
    for a, origin in enumerate(q.origins_bohr):
        translate[:, 4*a:4*a+4] = np.eye(4)
        translate[1:, 4*a] = np.asarray(origin)[[2, 0, 1]]
    np.testing.assert_allclose(translate @ q.values, global_q, atol=2e-10, rtol=2e-12)
    # A valence-only model is a different Q (observed 7.7e-4), far outside the match above.
    valence = proatoms(lambda a: [int(counts[a, 0]) - 1])
    assert np.abs(direct(valence) - q.values).max() > 1e-4


def test_mbis_log_domain_tail():
    snapshot = dict(counts=(2, 1), populations=((2., 6.), (1.,)), widths_bohr=((.05, .4), (.3,)))
    origins = ((0., 0., 0.), (0., 0., 2.))
    far = np.array([[0., 0., 1e3], [0., 0., -1e4], [1e5, 0., 0.]])
    logs = dist.mbis_log_proatoms(snapshot, origins, far)
    assert np.isfinite(logs).all()
    with np.errstate(under='ignore'):
        assert np.all(np.exp(logs) == 0)  # the linear stockholder ratio would be 0/0 here
    shares = np.exp(logs - logs.max(axis=0))
    shares /= shares.sum(axis=0)
    np.testing.assert_array_equal(shares[0], 1.)  # the most diffuse shell owns the far tail
    near = np.array([[0., 0., 1.]])
    expected = [np.log(2/(8*np.pi*.05**3)*np.exp(-1/.05) + 6/(8*np.pi*.4**3)*np.exp(-1/.4)),
                np.log(1/(8*np.pi*.3**3)*np.exp(-1/.3))]
    np.testing.assert_allclose(dist.mbis_log_proatoms(snapshot, origins, near)[:, 0], expected, rtol=1e-14)
    for bad in (dict(snapshot, widths_bohr=((.05, 0.), (.3,))), dict(snapshot, populations=((2., np.nan), (1.,)))):
        with pytest.raises(ValueError, match='finite and positive'):
            dist.mbis_log_proatoms(bad, origins, near)
    with pytest.raises(ValueError, match='too distant'):
        dist.mbis_log_proatoms(snapshot, origins, np.array([[0., 0., np.inf]]))


@pytest.mark.parametrize('corruption', ['unconverged', 'padding', 'width', 'count', 'residual', 'iterations'])
def test_mbis_snapshot_validation(mbis_water, monkeypatch, corruption):
    _, wfn, recipe, grid = mbis_water
    real = core.OEProp
    class Corrupted:
        def __init__(self, wfn):
            self.wfn, self.oe = wfn, real(wfn)
        def add(self, name):
            self.oe.add(name)
        def compute(self):
            self.oe.compute()
            w = self.wfn
            if corruption == 'unconverged':
                w.set_scalar_variable('MBIS CONVERGED', 0.)
            elif corruption == 'residual':
                w.set_scalar_variable('MBIS DENSITY RESIDUAL', 1e-7)
            elif corruption == 'iterations':
                w.set_scalar_variable('MBIS ITERATIONS', 0.)
            else:
                name, row, col, value = {'padding': ('MBIS SHELL POPULATIONS', 1, 1, 1e-3),
                                         'width': ('MBIS SHELL WIDTHS', 0, 0, 0.),
                                         'count': ('MBIS SHELL COUNTS', 1, 0, 1.5)}[corruption]
                array = w.array_variable(name).clone()
                array.set(row, col, value)
                w.set_array_variable(name, array)
    monkeypatch.setattr(core, 'OEProp', Corrupted)
    with pytest.raises(RuntimeError, match='native MBIS'):
        mbis_q(wfn, recipe.auxiliary, rank1_sites(recipe), 1, grid)


def test_mbis_failed_retry_never_reuses_q(mbis_water, monkeypatch, tmp_path):
    _, wfn, recipe, grid = mbis_water
    sites = rank1_sites(recipe)
    first = mbis_q(wfn, recipe.auxiliary, sites, 1, grid)
    psi4.set_options({'mbis_maxiter': 2})  # genuine native nonconvergence after a success
    with pytest.raises(RuntimeError, match='native MBIS failed'):
        mbis_q(wfn, recipe.auxiliary, sites, 1, grid)
    assert wfn.scalar_variable('MBIS CONVERGED') == 0
    assert wfn.scalar_variable('MBIS DENSITY RESIDUAL') > 1e-8
    assert not any(wfn.has_array_variable(n) for n in dist.MBIS_SNAPSHOT)
    monkeypatch.setattr(response, 'native_plain_df_operators',
                        lambda *a, **k: pytest.fail('response factors built after MBIS failure'))
    messages = []
    with pytest.raises(RuntimeError, match='native MBIS failed'):
        psi4.oeprop(wfn, 'ATOMIC_REFINED_DISPERSION', atomic_backend='BOUNDED_DF', preset='water',
            distribution='mbis', partition_grid=grid, response_grid=grid,
            auxiliary_recipe=recipe.auxiliary, npoints=32, resources=resources(),
            scratch_directory=tmp_path, log=StageLog(1, writer=messages.append))
    assert 'Stage FAILED: Native MBIS partition and response-AUX moments' in ''.join(messages)
    with pytest.raises(ValueError, match='No native'):
        psi4.atomic_property_result(wfn)
    assert not list(tmp_path.iterdir())
    # A native postprocessing failure after convergence also leaves no usable snapshot.
    psi4.set_options({'mbis_maxiter': 500})
    real = core.OEProp
    class Postprocessing:
        def __init__(self, wfn):
            self.oe = real(wfn)
            self.oe.add('MBIS_VOLUME_RATIOS')  # no free-atom volumes: native postprocessing refuses
        def add(self, name):
            assert name == 'MBIS_CHARGES'
        def compute(self):
            self.oe.compute()
    monkeypatch.setattr(core, 'OEProp', Postprocessing)
    with pytest.raises(RuntimeError, match='(?s)native MBIS failed .*FREE ATOM O VOLUME'):
        mbis_q(wfn, recipe.auxiliary, sites, 1, grid)
    assert wfn.scalar_variable('MBIS CONVERGED') == 0
    assert wfn.scalar_variable('MBIS DENSITY RESIDUAL') < 1e-8
    assert not any(wfn.has_array_variable(n) for n in dist.MBIS_SNAPSHOT)
    monkeypatch.setattr(core, 'OEProp', real)
    np.testing.assert_array_equal(mbis_q(wfn, recipe.auxiliary, sites, 1, grid).values, first.values)


@pytest.mark.parametrize('scheme, spherical', [('ROBUST', 302), ('ROBUST', 6), ('TREUTLER', 14),
                                               ('NONE', 110), ('P_SLATER', 110)])
def test_mbis_plan_bounds_the_native_grid(mbis_water, scheme, spherical):
    _, wfn, recipe, grid = mbis_water
    psi4.set_options({'mbis_pruning_scheme': scheme, 'mbis_spherical_points': spherical})
    sphere = dist._mbis_settings()[1]
    mol, radial = wfn.molecule(), MBIS_OPTIONS['mbis_radial_points']
    native = core.DFTGrid.build(mol, wfn.basisset(),
        {'DFT_RADIAL_POINTS': radial, 'DFT_SPHERICAL_POINTS': spherical},
        {'DFT_PRUNING_SCHEME': scheme}).npoints()
    assert native <= mol.natom()*radial*sphere
    if scheme == 'TREUTLER':
        # Region pruning's fixed inner orders exceed a small requested sphere,
        # so the requested size alone would undercharge.
        assert native > mol.natom()*radial*spherical
    (numeric, work), (sampling, sampling_work) = dist.mbis_resource_plan(wfn, recipe.auxiliary, grid, 4)
    nbf = wfn.basisset().nbf()
    assert numeric >= 8*native*(23+7*mol.natom())+64*nbf*nbf+32*len(grid)+dist.MBIS_SCRATCH_BYTES
    assert work >= 499*native*mol.natom()*dist.MBIS_MAX_SHELLS*128
    assert sampling >= 32*75*246+32*len(grid)+dist.MBIS_SCRATCH_BYTES


def test_mbis_arguments_and_admission(mbis_water, monkeypatch, tmp_path):
    _, wfn, recipe, grid = mbis_water
    sites = multipole_sites(recipe)
    public = dict(atomic_backend='BOUNDED_DF', preset='water', distribution='mbis', response_grid=grid,
                  auxiliary_recipe=recipe.auxiliary, resources=resources(), log=StageLog(0),
                  scratch_directory=tmp_path)
    monkeypatch.setattr(dist, 'run_native_mbis', lambda w: pytest.fail('native MBIS ran'))
    monkeypatch.setattr(isa, 'native_partition', lambda *a, **k: pytest.fail('MBIS ran ISA'))
    monkeypatch.setattr(response, 'native_plain_df_operators', lambda *a, **k: pytest.fail('response ran'))
    for extra, message in (({'partition_grid': grid, 'partition_recipe': recipe}, 'partition_recipe'),
                           ({'partition_grid': grid, 'distributed_moments': dist.analytic_df_moments(
                               recipe.auxiliary, sites, 4)}, 'distributed_moments'),
                           ({}, 'integration grid'),
                           ({'partition_grid': grid.astype(np.float32)}, 'integration grid')):
        with pytest.raises(ValueError, match=message):
            psi4.oeprop(wfn, 'ATOMIC_REFINED_DISPERSION', **public, **extra)
    with pytest.raises(ValueError, match='partition_grid requires'):
        psi4.oeprop(wfn, 'ATOMIC_REFINED_DISPERSION', **dict(public, distribution='supplied'),
                    partition_grid=grid)
    # Native settings that cannot converge or have no point-count bound fail
    # before any native, scratch or response work, publicly and directly.
    for options, message in (({'mbis_maxiter': 1}, 'MBIS_MAXITER'), ({'mbis_maxiter': 0}, 'MBIS_MAXITER'),
                             ({'mbis_d_convergence': 0.}, 'MBIS_D_CONVERGENCE'),
                             ({'mbis_d_convergence': float('nan')}, 'MBIS_D_CONVERGENCE'),
                             ({'mbis_radial_points': 0}, 'MBIS_RADIAL_POINTS'),
                             ({'dft_grid_name': 'SG1'}, 'DFT_GRID_NAME'),
                             ({'mbis_pruning_scheme': 'p_slater', 'dft_pruning_alpha': -1.}, 'bound unavailable')):
        psi4.set_options(options)
        with pytest.raises(ValueError, match=message):
            psi4.oeprop(wfn, 'ATOMIC_REFINED_DISPERSION', **public, partition_grid=grid)
        with pytest.raises(ValueError, match=message):
            mbis_q(wfn, recipe.auxiliary, sites, 4, grid)
        psi4.core.clean_options()
        psi4.set_options(MBIS_OPTIONS)
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError, match='caller_converged'):
        dist.mbis_moments(wfn, recipe.auxiliary, sites, 4, caller_converged=False,
                          integration_grid=grid, ledger=_Ledger(resources()))
    with pytest.raises(ValueError, match='identity mismatch'):
        mbis_q(wfn, recipe.auxiliary, sites[::-1], 4, grid)
    with pytest.raises(TypeError, match='BasisRecipe'):
        mbis_q(wfn, recipe, sites, 4, grid)
    # Exact plans admit (native MBIS is then dispatched); one byte or one work
    # unit short of either stage refuses before dispatch.
    class Admitted(Exception):
        pass
    def admitted(w):
        raise Admitted
    monkeypatch.setattr(dist, 'run_native_mbis', admitted)
    (numeric, work), (sampling, sampling_work) = dist.mbis_resource_plan(wfn, recipe.auxiliary, grid, 4)
    exact = BoundedResources(max(numeric, sampling), work+sampling_work, 1024)
    with pytest.raises(Admitted):
        mbis_q(wfn, recipe.auxiliary, sites, 4, grid, exact)
    stage = 'native MBIS partition' if numeric >= sampling else 'MBIS stockholder response-AUX Q'
    for budget, message in ((BoundedResources(max(numeric, sampling)-1, work+sampling_work, 1024), stage+': .*byte'),
                            (BoundedResources(max(numeric, sampling), work-1, 1024), 'native MBIS partition: .*work'),
                            (BoundedResources(max(numeric, sampling), work+sampling_work-1, 1024),
                             'MBIS stockholder response-AUX Q: .*work')):
        with pytest.raises(ValueError, match=message):
            mbis_q(wfn, recipe.auxiliary, sites, 4, grid, budget)


@pytest.fixture(scope='module')
def unsupported_mbis_scfs():
    wfns = {}
    water = 'O 0 0 0\nH -1.45365196 0 -1.12168732\nH 1.45365196 0 -1.12168732'
    for name, state, geometry, basis in (('ghost', '0 1', 'He 0 0 0\n@He 0 0 3', 'cc-pvdz'),
                                         ('ecp', '0 1', 'H 0 0 0\nI 0 0 3.04', 'def2-svp'),
                                         ('uks', '1 2', water, 'cc-pvdz')):
        mol = psi4.geometry(f'{state}\n{geometry}\nunits bohr\nsymmetry c1\nno_com\nno_reorient')
        psi4.set_options({'basis': basis, 'reference': 'uks' if name == 'uks' else 'rhf',
                          'scf_type': 'pk', 'e_convergence': 1e-10, 'd_convergence': 1e-8})
        method = 'pbe0' if name == 'uks' else 'hf'
        wfns[name] = psi4.energy(method, molecule=mol, return_wfn=True)[1]
        psi4.core.clean_options()
    assert wfns['ecp'].basisset().has_ECP()
    return wfns


def toy_mbis_inputs(wfn):
    geometry = tuple(map(tuple, wfn.molecule().geometry().np))
    auxiliary = BasisRecipe('s', 'unit primitive test', 'Cartesian', geometry,
                            tuple(ShellRecipe(a, 0, (1.,), (1.,)) for a in range(len(geometry))))
    sites = []
    for a, origin in enumerate(geometry):
        site = core.IsaMultipoleSite()
        site.label, site.origin, site.rank = f'X{a}', list(origin), 0
        sites.append(site)
    return auxiliary, sites, np.array([[0., 0., 1., 1.]])


@pytest.mark.parametrize('case, message', [('ghost', 'ghost'), ('ecp', 'ECP')])
def test_mbis_native_refusals(unsupported_mbis_scfs, case, message):
    wfn = unsupported_mbis_scfs[case]
    auxiliary, sites, grid = toy_mbis_inputs(wfn)
    try:
        with pytest.raises(RuntimeError, match='(?s)native MBIS failed .*' + message):
            mbis_q(wfn, auxiliary, sites, 0, grid)
        assert wfn.scalar_variable('MBIS CONVERGED') == 0
        assert wfn.scalar_variable('MBIS ITERATIONS') == 0
        assert not any(wfn.has_array_variable(n) for n in dist.MBIS_SNAPSHOT)
    finally:
        clear_mbis(wfn)


def test_mbis_unrestricted_refused_before_native(unsupported_mbis_scfs, monkeypatch, tmp_path):
    # Spin-polarized SCFs carry no restricted seal, so the direct producer
    # refuses them, and the public route's canonical-functional check refuses
    # UKS PBE0 before any MBIS dispatch.
    wfn = unsupported_mbis_scfs['uks']
    monkeypatch.setattr(dist, 'run_native_mbis', lambda w: pytest.fail('native MBIS ran for UKS'))
    auxiliary, sites, grid = toy_mbis_inputs(wfn)
    with pytest.raises(ValueError, match='Successful SCF convergence evidence'):
        mbis_q(wfn, auxiliary, sites, 0, grid)
    with pytest.raises(ValueError, match='canonical PBE0'):
        psi4.oeprop(wfn, 'ATOMIC_REFINED_DISPERSION', atomic_backend='BOUNDED_DF', preset='water',
            distribution='mbis', partition_grid=grid, response_grid=grid, npoints=32,
            resources=resources(), scratch_directory=tmp_path, log=StageLog(0))
    assert not wfn.has_scalar_variable('MBIS CONVERGED')
    assert not list(tmp_path.iterdir())


MBIS_ROUTE = dict(WATER_ROUTE, distribution='mbis')


def test_mbis_coarse_q_grid_does_not_bypass_lw_gate(mbis_water, tmp_path):
    _, wfn, recipe, grid = mbis_water
    with pytest.raises(RuntimeError, match='postcondition exceeds residual tolerance'):
        psi4.oeprop(wfn, 'ATOMIC_REFINED_DISPERSION', **MBIS_ROUTE, partition_grid=grid,
            auxiliary_recipe=recipe.auxiliary, response_grid=grid,
            resources=resources(), scratch_directory=tmp_path, log=StageLog(0))
    with pytest.raises(ValueError, match='No native'):
        psi4.atomic_property_result(wfn)
    assert not list(tmp_path.iterdir())


def test_mbis_molecular_properties(mbis_water, tmp_path):
    energy, wfn, recipe, grid = mbis_water
    # 100x200 fails the unchanged LW gate (test above); 200x590 is the declared
    # acceptance Q grid.
    integration = isa_grid(wfn.molecule(), 200, 590)
    psi4.oeprop(wfn, 'ATOMIC_REFINED_DISPERSION', **MBIS_ROUTE, partition_grid=integration,
        auxiliary_recipe=recipe.auxiliary, response_grid=grid,
        resources=resources(), scratch_directory=tmp_path, log=StageLog(0))
    result = psi4.atomic_property_result(wfn)
    partition = result.provenance['partition']
    diagnostics = partition['diagnostics']
    assert partition['model'] == 'mbis' and 'MBIS_SPHERICAL_POINTS=302' in partition['provenance']
    assert diagnostics['q_shape'] == (75, 246) and diagnostics['shell_counts'] == (2, 1, 1)
    assert diagnostics['excluded_denominators'] == diagnostics['negative_ratios'] == (0, 0, 0)
    assert diagnostics['density_residual'] < 1e-8 and 1 <= diagnostics['iterations'] < 500
    assert abs(diagnostics['grid_electrons'] - 10) < 1e-6
    assert diagnostics['q_charge_row_error'] < 2e-7
    assert diagnostics['integration_grid_sha256'] == hashlib.sha256(integration.tobytes()).hexdigest()
    stages = [s['stage'] for s in result.resources['stages']]
    assert (stages.index('native MBIS partition') < stages.index('MBIS stockholder response-AUX Q')
            < stages.index('retained distributed moments') < stages.index('native factors'))
    assert all(r.status == core.IsaPfitStatus.Solved for r in result.refinements)
    assert max(d['response_residual'] for d in result.diagnostics) < 1e-10
    assert max(d['localization_residual'] for d in result.diagnostics) < 1e-6
    assert not list(tmp_path.iterdir())
    c6 = [c.value for pair in result.dispersion.pairs for c in pair.coefficients]
    oo, oh, hh = 25.66199432, 3.79986053, .56625895
    np.testing.assert_allclose(c6, [oo, oh, oh, oh, hh, hh, oh, hh, hh], rtol=2e-6)
    np.testing.assert_allclose(result.refinements[0].parameters,
        [7.30577867, 7.19150699, 7.71292004, 2.08822061, -.00364998, .65936187, .71967940],
        rtol=2e-6, atol=2e-7)
    # Equality that is expected: a fresh native retry reproduces the consumed Q bitwise.
    retry = mbis_q(wfn, recipe.auxiliary, multipole_sites(recipe), 4, integration)
    assert hashlib.sha256(retry.values.tobytes()).hexdigest() == partition['q_sha256']
    print('MBIS_MOLECULAR_EVIDENCE', json.dumps(dict(energy=energy, partition=diagnostics,
        response=result.diagnostics, c6=c6, static_parameters=result.refinements[0].parameters,
        resources=result.resources)))
