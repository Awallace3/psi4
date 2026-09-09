"""Explicit-orbital Psi4 FDDS construction, NOT CamCASP response parity."""
import numpy as np
import pytest
import psi4
from psi4 import core
from psi4.driver.procrouting.sapt.fdds_response import (
    FDDSMonomerResponse, _compute_fxc, prepare_fdds_hybrid_transform, solve_fdds_response,
)


@pytest.fixture(scope='module')
def orbital_data():
    mol = psi4.geometry('''
    0 1
    O 0 0 0
    H .757 0 .586
    H -.757 0 .586
    symmetry c1
    no_reorient
    no_com
    ''')
    psi4.set_options({'basis': 'cc-pvdz', 'scf_type': 'pk', 'e_convergence': 1e-10,
                      'd_convergence': 1e-10})
    _, wfn = psi4.energy('hf', molecule=mol, return_wfn=True)
    # Explicit RI basis, not a silent default/JKFIT substitution.
    aux = core.BasisSet.build(mol, 'DF_BASIS_SAPT', 'cc-pvdz-ri')
    return (wfn.basisset(), aux, wfn.Ca_subset('AO', 'OCC'), wfn.Ca_subset('AO', 'VIR'),
            wfn.epsilon_a_subset('MO', 'OCC'), wfn.epsilon_a_subset('MO', 'VIR'))


def copied(data):
    return (*data[:2], *(x.clone() for x in data[2:]))


def close(x, y):
    np.testing.assert_allclose(x.to_array() if hasattr(x, 'to_array') else x,
                               y.to_array() if hasattr(y, 'to_array') else y,
                               rtol=1e-10, atol=1e-11)


@pytest.mark.parametrize('hybrid', [False, True])
def test_native_monomer_matches_pair_intermediates(orbital_data, hybrid):
    primary, aux, occ, vir, eo, ev = copied(orbital_data)
    monomer = core.FDDS_Monomer(primary, aux, occ, vir, eo, ev, hybrid)
    # Identical A/B inputs are a comparison fixture only, not the monomer API.
    matrices = {'Cocc_A': occ, 'Cvir_A': vir, 'Cocc_B': occ, 'Cvir_B': vir}
    vectors = {'eps_occ_A': eo, 'eps_vir_A': ev, 'eps_occ_B': eo, 'eps_vir_B': ev}
    pair = core.FDDS_Dispersion(primary, aux, matrices, vectors, hybrid)
    close(monomer.metric(), pair.metric())
    close(monomer.metric_inv(), pair.metric_inv())
    close(monomer.aux_overlap(), pair.aux_overlap())
    density = core.Matrix.from_array(occ.np @ occ.np.T)
    projected = pair.project_densities([density])[0]
    close(monomer.project_density(density), projected)
    alpha = .25 if hybrid else 0.
    provider = FDDSMonomerResponse(primary, aux, occ, vir, eo, ev, density,
                                   is_hybrid=hybrid, x_alpha=alpha)
    half = pair.aux_overlap().clone(); half.power(-.5, 1e-12)
    halfp = pair.aux_overlap().clone(); halfp.power(.5, 1e-12)
    W = pair.metric().clone()
    W.axpy(1., _compute_fxc(projected, half, halfp, alpha, 1e-8))
    if hybrid:
        close(monomer.R(), pair.R_A())
    for omega in [0., .4, 1000.]:
        close(monomer.form_unc_amplitude(omega), pair.form_unc_amplitude('A', omega))
        if hybrid:
            single = monomer.form_aux_matrices(omega)
            h = pair.form_aux_matrices('A', omega)
            assert single.keys() == h.keys()
            for key in single:
                close(single[key], h[key])
            h = {k: v.to_array() for k, v in h.items()}
            u = h.pop('amp')
            h['Rtinv'] = prepare_fdds_hybrid_transform(pair.R_A().to_array())
        else:
            u = pair.form_unc_amplitude('A', omega)
            u.scale(-1.)
            u = u.to_array()
            h = None
        expected = solve_fdds_response(metric=pair.metric().to_array(), metric_inv=pair.metric_inv().to_array(),
                                        W=W.to_array(), uncoupled=u, hybrid=h, x_alpha=alpha)
        actual = provider.at_frequency(omega)
        close(actual.raw_uncoupled, expected.raw_uncoupled)
        close(actual.raw_coupled, expected.raw_coupled)
        close(actual.coupled, expected.coupled)
        assert actual.representation == 'fdds_coulomb_auxiliary'
    # Repeating a hybrid frequency must not accumulate X/Y transformations.
    close(provider.at_frequency(.4).coupled, provider.at_frequency(.4).coupled)


def test_native_monomer_owns_inputs_and_getters(orbital_data):
    data = copied(orbital_data)
    obj = core.FDDS_Monomer(*data, False)
    before = obj.form_unc_amplitude(.3).to_array()
    j = obj.metric().to_array()
    obj.metric().zero()
    obj.metric_inv().zero()
    obj.aux_overlap().zero()
    data[2].zero(); data[3].zero(); data[4].zero(); data[5].zero()
    np.testing.assert_array_equal(obj.metric().to_array(), j)
    np.testing.assert_array_equal(obj.form_unc_amplitude(.3).to_array(), before)
    with pytest.raises(Exception, match='hybrid'):
        obj.R()
    with pytest.raises(Exception, match='hybrid'):
        obj.form_aux_matrices(.3)
    for frequency in [-.1, float('nan'), float('inf')]:
        with pytest.raises(Exception, match='frequency'):
            obj.form_unc_amplitude(frequency)
    with pytest.raises(Exception, match='density'):
        obj.project_density(core.Matrix.from_array(np.eye(2)))


@pytest.mark.parametrize('case', ['rows', 'nan_coefficient', 'energy_size', 'nan_energy', 'gap', 'qr'])
def test_native_monomer_rejects_bad_inputs(orbital_data, case):
    primary, aux, occ, vir, eo, ev = copied(orbital_data)
    hybrid = False
    pattern = 'FDDS_Monomer'
    if case == 'rows':
        occ = core.Matrix.from_array(np.ones((1, occ.np.shape[1])))
    elif case == 'nan_coefficient':
        occ.np[0, 0] = np.nan
    elif case == 'energy_size':
        eo = core.Vector.from_array(np.ones(eo.dim()+1))
    elif case == 'nan_energy':
        ev.np[0] = np.nan
    elif case == 'gap':
        ev.np[0] = eo.np[0]-1
    else:
        occ = core.Matrix.from_array(np.ones((primary.nbf(), 1)))
        vir = occ.clone()
        eo = core.Vector.from_array(np.array([-1.]))
        ev = core.Vector.from_array(np.array([1.]))
        hybrid = True
        pattern = 'nov >= naux'
    with pytest.raises(Exception, match=pattern):
        core.FDDS_Monomer(primary, aux, occ, vir, eo, ev, hybrid)
