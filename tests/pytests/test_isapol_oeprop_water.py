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


# Reduced literals for the public refined path, guarded by
# `agent_scratch/pytests/test_isapol_refined_dispersion_full.py`, which holds the
# complete fixture (every refined tensor, every coefficient, every QCVariable).
# These are this endpoint's own numbers and NOT a CamCASP parity reference: the
# public path's distributed multipoles are the ISA-A shape-partitioned ones,
# while the archived aug-cc-pVTZ audit used analytic DF-centre multipoles.
REFINED_LATTICE = (500, 1327, '693d2c092b36f85171da0acd08702e6c1fd24deb95c7b1146a0d83487a2c80aa')
REFINED_SITE_RANKS = [(1, 2), (1,), (1,)]
REFINED_COEFFICIENTS = {
    'O1 O1 6': 8.894640189857054, 'O1 O1 8': 56.34886849092746,
    'O1 O1 10': 187.13499877010736,
    'O1 H2 6': 1.8996992531739563, 'O1 H2 8': 5.691735724163029,
    'H2 H2 6': 0.4157409071711345,
}


def test_fresh_water_refined_dispersion(water,monkeypatch):
    """The public PFIT path: one refinement per node, its own C_n, its own names.

    The refined and unrefined coefficients are two models of the same molecule,
    so both are published and neither is quoted as the other's error.
    """
    monkeypatch.setattr(psi4,'energy',lambda *a,**k: pytest.fail('hidden SCF'))
    assert psi4.oeprop(water,'ATOMIC_DISPERSION','ATOMIC_REFINED_DISPERSION') is None
    r = psi4.atomic_property_result(water)
    ref = r.refinement
    assert dict(r.options)['ATOMIC_REFINEMENT_SEED'] == 1
    assert (ref.lattice.npoints,ref.lattice.ncandidates,ref.lattice.sha256) == REFINED_LATTICE
    assert ref.lattice.limit_units == 'multiples of the van der Waals radius'
    assert len(ref.refinements) == len(r.properties.frequencies) == 11
    assert ref.solved and set(ref.solver_status) == {'Solved'}
    # Declared per site type, exactly as a .pdef's `Limit rank to 1 for sites H`.
    assert list(ref.site_types) == ['O','H','H'] and list(ref.rank_limits) == [2,1,1]
    d = r.refined_dispersion
    assert [tuple(x) for x in d.site_ranks] == REFINED_SITE_RANKS
    assert len(d.pairs) == 9 and not isinstance(d,type(r.dispersion))
    values = {f'{p.label_a} {p.label_b} {c.order}': c for p in d.pairs for c in p.coefficients}
    for key, expected in REFINED_COEFFICIENTS.items():
        np.testing.assert_allclose(values[key].value,expected,rtol=1e-9,atol=0)
        assert values[key].included_rank_pairs
    # An order no site pair carries is exactly zero and flagged, never an estimate.
    for key in ('O1 O1 12','O1 H2 10','H2 H2 8'):
        assert values[key].value == 0. and not values[key].unrestricted_complete
        assert values[key].missing_rank_pairs and not values[key].included_rank_pairs
    # H3 COPY H2: one declared variable set, so the two hydrogens agree exactly.
    assert values['H2 H2 6'].value == values['H3 H3 6'].value == values['H2 H3 6'].value
    assert values['O1 H2 6'].value == values['O1 H3 6'].value
    # Separate stems: an unrefined variable can never be read as a refined one.
    assert water.variable('ATOM O1 C6 REFINED DISPERSION COEFFICIENT') == \
           values['O1 O1 6'].value
    assert water.variable('ATOM O1 C6 DISPERSION COEFFICIENT') != values['O1 O1 6'].value
    assert water.variable('ATOMIC REFINEMENT FIT POINT SEED') == 1.
    plain = {(p.label_a,p.label_b): p.coefficients[0].value for p in r.dispersion.pairs}
    assert all(plain[(p.label_a,p.label_b)] != p.coefficients[0].value for p in d.pairs)
