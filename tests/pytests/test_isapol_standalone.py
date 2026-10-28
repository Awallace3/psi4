# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Installed interface/compatibility checks; no oracle or saved molecular input."""
import json
import subprocess
import sys

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
