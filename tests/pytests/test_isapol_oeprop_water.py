"""Fresh restricted C1 PBE0/cc-pVDZ endpoint; no archived molecular input.

This intentionally exercises the full generated H/O demonstration recipe. It is
not a CamCASP parity oracle. Expected runtime is measured in the native handoff.
"""
import numpy as np
import pytest
import psi4
from psi4.driver.procrouting import isapol_oeprop as api


@pytest.fixture(scope='module')
def water():
    psi4.core.clean_options()
    psi4.core.be_quiet()
    mol = psi4.geometry('''0 1
O 0 0 0
H -1.45365196 0 -1.12168732
H 1.45365196 0 -1.12168732
units bohr
symmetry c1
no_com
no_reorient
''')
    psi4.set_options({'basis':'cc-pvdz','reference':'rks','scf_type':'pk','puream':True,
                      'e_convergence':1e-10,'d_convergence':1e-10,
                      'dft_radial_points':99,'dft_spherical_points':590})
    _, w = psi4.energy('pbe0',molecule=mol,return_wfn=True)
    return w


def test_fresh_water_partition_only(water,monkeypatch):
    monkeypatch.setattr(psi4,'energy',lambda *a,**k: pytest.fail('hidden SCF'))
    monkeypatch.setattr(api.n,'native_properties',lambda *a,**k: pytest.fail('partition requested response'))
    assert psi4.oeprop(water,'ATOMIC_PARTITION') is None
    r = psi4.atomic_property_result(water)
    assert r.partition.converged and r.properties is None
    assert r.partition.drho.charge_penalty == 1000.
    assert r.partition.recipe.name == 'GENERATED_JKFIT_ISA_A'


def test_fresh_water_static_only(water,monkeypatch):
    monkeypatch.setattr(psi4,'energy',lambda *a,**k: pytest.fail('hidden SCF'))
    monkeypatch.setattr(api.n.Quadrature,'from_casimir',lambda *a: pytest.fail('static quadrature'))
    assert psi4.oeprop(water,'ATOMIC_POLARIZABILITIES') is None
    r = psi4.atomic_property_result(water)
    assert r.properties.frequencies == (0.,) and r.dispersion is None
    assert r.properties.ov_fit is None and r.properties.distributed.partition.representation == 'direct_ov'
    assert not r.properties.failures
    assert r.atomic_scalars.shape == (1,3,3)
    assert np.isfinite(r.atomic_scalars).all()
    # Analytic Mints MO dipoles and independent static H1 inversion, not Q-based oracle.
    c = np.asarray(water.Ca()); no = water.nalpha()
    moments = np.array([(c[:,:no].T @ np.asarray(mu) @ c[:,no:]).T.reshape(-1)
                       for mu in psi4.core.MintsHelper(water.basisset()).ao_dipole()])
    h1 = np.asarray(r.properties.context.response.provider.h1())
    expected = 4*moments @ np.linalg.solve(h1,moments.T)
    actual = r.properties.local.global_dipoles.array[0].sum(axis=0)
    np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=0)


def test_fresh_water_dispersion_only_and_owned_results(water,monkeypatch):
    monkeypatch.setattr(psi4,'energy',lambda *a,**k: pytest.fail('hidden SCF'))
    assert psi4.oeprop(water,'ATOMIC_DISPERSION') is None
    r = psi4.atomic_property_result(water)
    assert r.properties.partition.converged and not r.properties.failures
    assert len(r.properties.frequencies) == 11 and len(r.dispersion.pairs) == 9
    np.testing.assert_allclose(r.atomic_scalars[:,1],r.atomic_scalars[:,2],atol=1e-7,rtol=0)
    for pair in r.dispersion.pairs:
        c6 = 6*np.dot(r.properties.quadrature.cp_weights,
                     r.atomic_scalars[:,pair.site_a,0]*r.atomic_scalars[:,pair.site_b,0])
        np.testing.assert_allclose(pair.coefficients[0].value,c6,atol=2e-12,rtol=0)
        assert [v.order for v in pair.coefficients] == [6,8,10,12]
    old_scalars = r.atomic_scalars.copy()
    assert psi4.oeprop(water,'ATOMIC_PARTITION') is None
    assert psi4.atomic_property_result(water) is not r
    assert np.array_equal(r.atomic_scalars,old_scalars)
