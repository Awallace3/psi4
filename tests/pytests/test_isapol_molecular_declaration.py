"""Generic declaration/setup tests only; no molecular SCF or campaign run."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from psi4.driver.procrouting.isapol_molecular import (
    MolecularDeclaration, PROFILE, DECLARATION_SCHEMA, independent_sites,
    full_rank2_variables, plan_molecular_refinement, export_molecular_result,
)


def declaration(n=3):
    return MolecularDeclaration(PROFILE, tuple(np.eye(3) for _ in range(n)),
                                tuple((i, i+1) for i in range(n-1)))


def test_declaration_roundtrips_json_and_owns_its_inputs():
    frame = np.eye(3)
    value = MolecularDeclaration(PROFILE, (frame, frame, frame), ((0, 1), (0, 2)))
    frame[0, 0] = 100.
    assert value.frames[0][0][0] == 1.
    raw = json.loads(json.dumps(value.as_dict()))
    assert raw['schema'] == DECLARATION_SCHEMA
    assert MolecularDeclaration.from_dict(raw) == value
    with pytest.raises(ValueError, match='versioned'):
        MolecularDeclaration.from_dict(dict(raw, reference_energy=-2.))


@pytest.mark.parametrize('bonds', [((0, 0),), ((0, 3),), ((0, 1), (1, 0)), ((True, 1),)])
def test_molecular_bonds_must_be_explicit_valid_edges(bonds):
    with pytest.raises(ValueError, match='bond'):
        MolecularDeclaration(PROFILE, (np.eye(3),)*3, bonds)


def test_mirrored_and_nonorthogonal_frames_are_refused():
    for invalid in (np.diag([-1., 1., 1.]), np.diag([2., 1., 1.])):
        with pytest.raises(ValueError, match='proper orthogonal'):
            MolecularDeclaration(PROFILE, (np.eye(3), np.eye(3), invalid), ())


def test_all_methanol_atoms_are_independently_typed_not_element_copy():
    import psi4
    from psi4.driver.procrouting.isapol_refine import refinement_model
    molecule = psi4.geometry(
        '0 1\nC 0 0 0\nH -1 0 0\nH 0 -1 0\nH 0 0 -1\nO 1 0 0\nH 2 0 0\n'
        'symmetry c1\nno_com\nno_reorient\nunits bohr')
    sites = independent_sites(molecule, declaration(6))
    assert [s.label for s in sites] == ['C1', 'H1', 'H2', 'H3', 'O1', 'H4']
    assert all(s.site_type == s.label for s in sites)
    variables = full_rank2_variables(sites)
    assert len(variables) == 216 and not any('_00_' in name for name in variables)
    model = refinement_model(sites, [np.zeros((9, 9)) for _ in sites],
                             declared_variables=variables)
    assert model.parameter_count == 216
    assert model.reference_sites == tuple(range(6))
    assert model.equivalent_sites == tuple((i,) for i in range(6))
    assert all(len(entries) == 1 for entries in model.parameter_entries)


def test_largest_inventory_preflight_is_explicitly_refinement_only():
    from psi4.driver.procrouting.isapol_bounded import BoundedResources
    resources = BoundedResources(4*1024**3, 100_000_000_000_000, 64*1024**3, capacity='large')
    plan = plan_molecular_refinement(declaration(17), resources, max_replay_work=1_000_000_000_000)
    assert plan['parameters'] == 612 and plan['planned_work'] == 718_632_624_000
    assert plan['scope'] == 'refinement_only_not_complete_molecular_admission'


def test_molecular_preflight_refuses_a_budget_that_cannot_cover_refinement_alone():
    from psi4.driver.procrouting.isapol_bounded import BoundedResources
    resources = BoundedResources(4*1024**3, 100_000_000_000, 64*1024**3, capacity='large')
    with pytest.raises(ValueError, match='complete refinement.*work'):
        plan_molecular_refinement(declaration(17), resources, max_replay_work=1_000_000_000_000)


def test_incomplete_export_fails_before_creating_scientific_output(tmp_path):
    import psi4
    from psi4 import core
    molecule = psi4.geometry(
        '0 1\nO 0 0 0\nH 1 0 0\nH 0 1 0\nsymmetry c1\nno_com\nno_reorient\nunits bohr')
    wfn = core.Wavefunction.build(molecule, 'sto-3g')
    incomplete = SimpleNamespace(frequencies=(0.,), refinements=(), diagnostics=(), local_tensors=())
    wfn._native_atomic_property_result = incomplete  # owned negative-export fixture only
    destination = tmp_path/'not-created'
    with pytest.raises(RuntimeError, match='incomplete molecular frequency'):
        export_molecular_result(wfn, incomplete, declaration(), output=destination)
    assert not destination.exists()


@pytest.mark.parametrize('corrupt_c10', [False, True])
def test_export_handles_real_native_coefficient_coverage(tmp_path, monkeypatch, corrupt_c10):
    """Synthetic tensors/native CP kernel only; NOT molecular acceptance."""
    from dataclasses import replace
    import psi4
    from psi4 import core
    from psi4.driver.procrouting import isapol_molecular as api
    from psi4.driver.procrouting.isapol_bounded import BoundedProperties, refinement_models
    from psi4.driver.procrouting.isapol_lw import ArraySnapshot, Coefficient
    from psi4.driver.procrouting.isapol_native import Quadrature
    from psi4.driver.procrouting.isapol_refine import RefinementResult, refined_isotropic_dispersion
    molecule = psi4.geometry(
        '0 1\nO 0 0 0\nH 1 0 0\nH 0 1 0\nsymmetry c1\nno_com\nno_reorient\nunits bohr')
    wfn = core.Wavefunction.build(molecule, 'sto-3g')  # descriptor only, no SCF
    declared = declaration()
    sites = independent_sites(molecule, declared)
    anchors = tuple(np.diag([0.] + [1.]*8) for _ in sites)
    quadrature = Quadrature.from_casimir(core.CasimirGrid(10, .5))
    models = refinement_models(
        sites, [(omega, anchors) for omega in quadrature.frequencies],
        cutoff=1e-4, weight_type=3, weight_coefficient=.001,
        declared_variables=full_rank2_variables(sites), provenance='SYNTHETIC UNIT FIXTURE')
    nodes = tuple(RefinementResult(
        model=model, parameters=model.anchors, refined_tensors=anchors,
        anchor_shift_maxabs=0., status=core.IsaPfitStatus.Solved,
        diagnostics=SimpleNamespace(native_verified=False), result=None,
        refinement_status='SYNTHETIC UNIT FIXTURE, NOT A MOLECULAR FIT')
        for model in models)
    dispersion = refined_isotropic_dispersion(
        nodes, cp_weights=quadrature.cp_weights,
        quadrature_provenance=quadrature.provenance, max_order=10)
    native_c10 = dispersion.pairs[0].coefficients[-1]
    assert isinstance(native_c10, Coefficient)
    assert native_c10.missing_rank_pairs == ((1, 3), (3, 1))
    if corrupt_c10:
        pair = dispersion.pairs[0]
        wrong = replace(native_c10, missing_rank_pairs=((1, 3),))
        dispersion = replace(dispersion, pairs=(
            replace(pair, coefficients=pair.coefficients[:-1] + (wrong,)),) + dispersion.pairs[1:])
    result = BoundedProperties(
        frequencies=quadrature.frequencies,
        local_tensors=tuple(ArraySnapshot.of(np.zeros((1, 3, 15, 15))) for _ in nodes),
        refinements=nodes, dispersion=dispersion,
        diagnostics=tuple(dict(response_residual=0., localization_residual=0.) for _ in nodes),
        resources=dict(synthetic_fixture=True), provenance=dict(synthetic_fixture=True))
    wfn._native_atomic_property_result = result  # fixture ownership only
    # The real seal is never manufactured; the unit fixture explicitly stubs
    # this runtime boundary and may not be used as accepted molecular evidence.
    monkeypatch.setattr(api, 'require_scf_seal', lambda _: None)
    destination = tmp_path/'synthetic-export'
    if corrupt_c10:
        with pytest.raises(RuntimeError, match='partial C10 coverage'):
            export_molecular_result(wfn, result, declared, output=destination)
        assert not destination.exists()
    else:
        record = export_molecular_result(wfn, result, declared, output=destination)
        assert record['provenance']['synthetic_fixture']
        assert len(record['isotropic_pairs']) == 27
        assert record['tensor_layout']['local_shape'] == [11, 3, 15, 15]
        assert all(not c['unrestricted_complete'] and c['missing_rank_pairs'] == [[1, 3], [3, 1]]
                   for c in record['isotropic_pairs'] if c['order'] == 10)


def test_molecular_request_clears_stale_results_and_requires_a_real_scf_seal():
    import psi4
    from psi4 import core
    from psi4.driver.procrouting.isapol_bounded import BoundedResources
    from psi4.driver.procrouting.isapol_molecular import run_molecular_properties
    molecule = psi4.geometry(
        '0 1\nO 0 0 0\nH 1 0 0\nH 0 1 0\nsymmetry c1\nno_com\nno_reorient\nunits bohr')
    wfn = core.Wavefunction.build(molecule, 'sto-3g')
    wfn._native_atomic_property_result = object()
    resources = BoundedResources(1024**3, 100_000_000_000_000, 64*1024**3, capacity='large')
    with pytest.raises(ValueError, match='Successful SCF convergence evidence'):
        run_molecular_properties(wfn, declaration(), resources,
            max_replay_work=1_000_000_000_000, scratch_directory=None)
    with pytest.raises(ValueError, match='No native atomic property'):
        psi4.atomic_property_result(wfn)


def test_campaign_water_is_a_separate_uniform_tail_recipe_not_a_relabelled_regression():
    import psi4
    from psi4 import core
    from psi4.driver.procrouting.isapol_recipes import native_water_recipe, native_monomer_recipe
    core.clean_options()
    molecule = psi4.geometry(
        '0 1\nO 0 0 0\nH 1 0 0\nH 0 1 0\nsymmetry c1\nno_com\nno_reorient\nunits bohr')
    legacy = native_water_recipe(molecule)
    campaign = native_monomer_recipe(molecule, profile=PROFILE)
    assert [s.tail_cutoff for s in legacy.sites] == [1.5, 1.5, 1.5]
    assert [s.tail_cutoff for s in campaign.sites] == [
        1.5*.60/.529177249, 1.5*.50/.529177249, 1.5*.50/.529177249]
    assert [s.label for s in legacy.sites] == ['O', 'H1', 'H2']
    assert [s.label for s in campaign.sites] == ['O1', 'H1', 'H2']
