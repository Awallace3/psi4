"""Owned distributed-moment contract on the analytic DF-centre producer."""
from dataclasses import replace

import numpy as np
import pytest
from psi4 import core
from psi4.driver.procrouting import isapol_distribution as dist
from psi4.driver.procrouting.isapol_basis import BasisRecipe, ShellRecipe


@pytest.fixture
def small_q():
    aux = BasisRecipe('s,p', 'unit primitive test', 'Cartesian', ((0., 0., 0.),),
                      (ShellRecipe(0, 0, (1.,), (1.,)), ShellRecipe(0, 1, (1.,), (1.,))))
    site = core.IsaMultipoleSite()
    site.label, site.origin, site.rank = 'X', [0., 0., 0.], 1
    return aux, [site], dist.analytic_df_moments(aux, [site], 1)


def test_df_contract_preserves_analytic_values_and_contraction(small_q):
    aux, sites, q = small_q
    old = dist.df_centre_multipoles('df_centre_analytic', aux, sites, 1)
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
