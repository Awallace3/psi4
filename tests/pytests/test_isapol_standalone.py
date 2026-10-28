# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Installed interface/compatibility checks; no oracle or saved molecular input."""
import dataclasses
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import psi4

from psi4.driver.procrouting import isapol_basis as basis
from psi4.driver.procrouting import isapol_native_partition as partition
from psi4.driver.procrouting import isapol_native_correction as correction
from psi4.driver.procrouting.scf_proc import _scf_seal


def test_basis_types_have_one_definition_and_legacy_imports_work():
    assert partition.BasisRecipe is basis.BasisRecipe
    assert partition.ShellRecipe is basis.ShellRecipe
    assert partition.MainAdaptation is basis.MainAdaptation
    assert partition.adapt_main is basis.adapt_main


def test_scf_seal_helpers_are_shared_without_losing_declared_ac_policy():
    assert correction.component_definition is _scf_seal.component_definition
    assert correction.correction_state is _scf_seal.correction_state
    assert correction.grac_component_definition is _scf_seal.grac_component_definition
    assert 'DECLARED_MULTPOLE_AC' in correction.POLICIES
    assert callable(psi4.atomic_asymptotic_correction)


def test_legacy_ac_reporting_is_retained_lazily():
    from psi4.driver.procrouting import isapol_logging as log
    from psi4.driver.procrouting import isapol_ac_logging as ac
    assert log.AcIterationLog is ac.AcIterationLog
    assert log.report_ac_convergence is ac.report_ac_convergence
    assert log.ac_application_parameters is ac.ac_application_parameters
    with pytest.raises(AttributeError):
        getattr(log, 'not_a_reporter')


def test_declared_bragg_slater_multiplier_is_not_absolute_cutoff():
    oxygen = partition.bragg_slater_tail_cutoff(8, 1.5)
    hydrogen = partition.bragg_slater_tail_cutoff(1, 1.5)
    assert oxygen == 1.5 * .60 / .529177249
    assert hydrogen == 1.5 * .50 / .529177249
    assert oxygen != hydrogen and oxygen != 1.5
    for Z, multiplier in ((55, 1.5), (True, 1.5), (8, 0.)):
        with pytest.raises(ValueError):
            partition.bragg_slater_tail_cutoff(Z, multiplier)


def test_acceptance_requires_exact_installed_build_before_any_scf(tmp_path):
    from psi4.driver.procrouting.isapol_acceptance import _verify_build, _plain
    with pytest.raises(ValueError, match='clean-source fresh-build'):
        _verify_build({})
    with pytest.raises(ValueError, match='not the declared installed'):
        _verify_build(dict(clean_source=True, fresh_build=True,
                           installed_package=str(tmp_path)))
    assert _plain(dict(values=np.array([1., 2.]), flag=np.bool_(True))) == dict(
        values=[1., 2.], flag=True)


def test_installed_production_recipe_import_needs_no_test_or_supplied_helper():
    code = """
import json, sys
from psi4.driver.procrouting.isapol_recipes import native_water_recipe
from psi4.driver.procrouting.isapol_basis import ShellRecipe
from psi4.driver.procrouting.isapol_native_partition import ShellRecipe as legacy
assert legacy is ShellRecipe
assert 'isapol_water_recipe' not in sys.modules
assert not any('oracle' in name for name in sys.modules)
assert not any(name.endswith('isapol_supplied') for name in sys.modules)
print(json.dumps({'recipe': native_water_recipe.__module__, 'standalone_import': True}))
"""
    p = subprocess.run([sys.executable, '-I', '-B', '-c', code],
                       text=True, capture_output=True, check=True)
    record = json.loads(p.stdout.strip().splitlines()[-1])
    assert record['standalone_import'] is True
    assert record['recipe'] == 'psi4.driver.procrouting.isapol_recipes'


@pytest.mark.parametrize('width,rank', [(15, 2), (24, 4)])
def test_acceptance_tensor_artifact_is_numeric_and_pickle_free(tmp_path, width, rank):
    from psi4.driver.procrouting.isapol_acceptance import _write_tensor_artifact
    from psi4.driver.procrouting.isapol_lw import ArraySnapshot
    local = np.arange(2*3*width*width, dtype=np.float64).reshape(2, 3, width, width)
    keep = (rank+1)**2-1
    local[..., keep:, :] = 0
    local[..., :, keep:] = 0
    refined = np.arange(2*3*9*9, dtype=np.float64).reshape(2, 3, 9, 9)
    result = SimpleNamespace(
        frequencies=(0., .5),
        local_tensors=tuple(ArraySnapshot.of(a[None]) for a in local),
        refined_tensors=refined,
        refinements=(SimpleNamespace(model=SimpleNamespace(sites=(None,)*3)),)*2,
    )
    path = tmp_path / 'tensors.npz'
    layout = _write_tensor_artifact(path, result, localization_rank_limit=rank)
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['local'], local)
        np.testing.assert_array_equal(saved['refined'], refined)
        assert saved['local'].dtype == saved['refined'].dtype == np.float64
    assert layout['local_shape'] == (2, 3, width, width)
    assert layout['localization_rank_limit'] == rank
    assert 'no00' in layout['local_components']
    assert 'includes00' in layout['refined_components']
    result.refined_tensors = refined.astype(object)
    with pytest.raises(RuntimeError, match='float64'):
        _write_tensor_artifact(tmp_path / 'object.npz', result, localization_rank_limit=rank)
    result.refined_tensors = np.full_like(refined, np.nan)
    with pytest.raises(RuntimeError, match='finite'):
        _write_tensor_artifact(tmp_path / 'nonfinite.npz', result, localization_rank_limit=rank)


def test_acceptance_angular_export_preserves_coverage_and_conventions():
    from psi4.driver.procrouting.isapol_acceptance import _angular_export, _plain
    from psi4.driver.procrouting.isapol_native_recoupled import DeclaredPairCoverage
    missing = ((1, 1, 1, 5),)
    coverage = SimpleNamespace(
        order=10, table_complete=True, unrestricted_complete=False,
        included_rank_quadruples=((2, 2, 2, 2),),
        missing_table_rank_quadruples=(), missing_unrestricted_rank_quadruples=missing,
    )
    angular = SimpleNamespace(
        convention='local-axis real Racah/no-CS Cn(t,u,J); no dimer placement or energy',
        charge_policy='00 row/column exactly zero; omitted, not rank1',
        frequencies=(0., .5), cp_weights=(0., .1), max_order=10,
        pair_labels=(('O', 'O'),),
        declared_model_coverage=(DeclaredPairCoverage(0, 0, 10, True, ()),),
        quadrature_provenance='synthetic export unit test', provenance='synthetic',
        pairs=(SimpleNamespace(site_a=0, site_b=0, coefficients=(), coverage=(coverage,)),),
    )
    exported = json.loads(json.dumps(_plain(_angular_export(angular)), allow_nan=False))
    assert exported['declared_model_coverage'][0]['declared_model_complete']
    cov = exported['pairs'][0]['coverage'][0]
    assert cov['table_complete'] and not cov['unrestricted_complete']
    assert cov['missing_unrestricted_rank_quadruples'] == [[1, 1, 1, 5]]
    assert exported['cp_weights'][0] == 0.
    assert 'no dimer placement or energy' in exported['convention']


def test_acceptance_isotropic_export_uses_actual_coefficient_contract():
    from psi4.driver.procrouting.isapol_acceptance import _isotropic_export, _plain
    from psi4.driver.procrouting.isapol_lw import Coefficient, DispersionPair
    coefficients = (
        Coefficient(6, 1., ((1, 1),), (), True),
        Coefficient(8, 1.5, ((1, 2), (2, 1)), (), True),
        Coefficient(10, 2., ((2, 2),), ((1, 3), (3, 1)), False),
    )
    pair = DispersionPair(0, 0, 'O', 'O', coefficients)
    result = SimpleNamespace(labels=('O',), pairs=(pair,))
    exported = json.loads(json.dumps(_plain(_isotropic_export(result, max_order=10))))
    assert exported[0]['unrestricted_complete']
    assert not exported[2]['unrestricted_complete']
    assert exported[2]['missing_rank_pairs'] == [[1, 3], [3, 1]]
    assert exported[2]['order'] == 10 and exported[2]['value'] == 2.
    for pairs in ((), (pair, pair), (dataclasses.replace(pair, coefficients=coefficients[:1]),)):
        with pytest.raises(RuntimeError, match='coverage'):
            _isotropic_export(SimpleNamespace(labels=('O',), pairs=pairs), max_order=10)


def test_acceptance_exports_actual_native_lw_rank2_packing(tmp_path):
    from psi4.driver.procrouting import isapol_lw as lw
    from psi4.driver.procrouting.isapol_acceptance import _write_tensor_artifact
    raw = np.zeros((1, 1, 1, 25, 25))
    np.fill_diagonal(raw[0, 0, 0, 1:, 1:], 1.)
    native = lw.supplied_nonlocal_properties(
        labels=('X',), origins=np.zeros((1, 3)), bonds=(), frequencies=(0.,),
        tensors=raw, input_rank=4, truncation=lw.TRUNCATE_RANK4,
        localization_rank_limit=2,
        provenance=lw.Provenance('synthetic', '0'*64, 'native test', 'packing regression'),
    )
    assert native.raw_local.shape == (1, 1, 15, 15)
    result = SimpleNamespace(
        frequencies=(0.,), local_tensors=(native.raw_local,),
        refined_tensors=np.zeros((1, 1, 9, 9)),
        refinements=(SimpleNamespace(model=SimpleNamespace(sites=(None,))),),
    )
    layout = _write_tensor_artifact(tmp_path / 'native.npz', result,
                                    localization_rank_limit=2)
    assert layout['local_shape'] == (1, 1, 15, 15)
    assert layout['packing_rank'] == 3
    assert layout['localization_rank_limit'] == 2


def test_rank2_acceptance_requires_angular_table_and_declared_coverage():
    from psi4.driver.procrouting.isapol_acceptance import _require_angular_coverage
    from psi4.driver.procrouting.isapol_native_recoupled import DeclaredPairCoverage
    with pytest.raises(RuntimeError, match='requires native'):
        _require_angular_coverage(None, nsite=1, max_order=10)
    coverage = [SimpleNamespace(order=n, table_complete=True, unrestricted_complete=False,
                                missing_table_rank_quadruples=())
                for n in range(6, 11)]
    declared = tuple(DeclaredPairCoverage(0, 0, n, True, ()) for n in range(6, 11))
    pair = SimpleNamespace(site_a=0, site_b=0, coverage=coverage, coefficients=(object(),))
    angular = SimpleNamespace(pairs=(pair,), declared_model_coverage=declared)
    _require_angular_coverage(angular, nsite=1, max_order=10)
    coverage[-1].table_complete = False
    with pytest.raises(RuntimeError, match='coverage'):
        _require_angular_coverage(angular, nsite=1, max_order=10)
    coverage[-1].missing_table_rank_quadruples = ((1, 1, 2, 4),)
    _require_angular_coverage(angular, nsite=1, max_order=10)
    coverage[-1].table_complete = True
    coverage[-1].missing_table_rank_quadruples = ()
    angular.declared_model_coverage = declared[:-1]
    with pytest.raises(RuntimeError, match='coverage'):
        _require_angular_coverage(angular, nsite=1, max_order=10)
