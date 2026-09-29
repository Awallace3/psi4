"""Native FDDS monomer behavior; deliberately small auxiliary bases test QR shapes."""

import gc

import numpy as np
import pytest

import psi4
from psi4.driver.procrouting.sapt.sapt_mp2_terms import fdds_coupled_amplitudes

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.saptdft]


@pytest.fixture(scope="module")
def orbitals():
    # Cache only immutable SCF data, not FDHelper files or global driver state.
    psi4.core.clean_options()
    psi4.core.clean_variables()
    mol = psi4.geometry("""
        O 0.0 0.0 0.0
        H 0.0 0.0 0.96
        H 0.90 0.0 -0.24
        symmetry c1
    """)
    psi4.set_options({"basis": "cc-pvdz", "scf_type": "df", "e_convergence": 1.e-10, "d_convergence": 1.e-10})
    _, wfn = psi4.energy("hf", molecule=mol, return_wfn=True)
    data = (wfn.basisset(), psi4.core.BasisSet.build(mol, "ORBITAL", "sto-3g"),
            wfn.Ca_subset("AO", "OCC").to_array(), wfn.Ca_subset("AO", "VIR").to_array(),
            wfn.epsilon_a_subset("AO", "OCC").to_array(), wfn.epsilon_a_subset("AO", "VIR").to_array(),
            wfn.Da().to_array())
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    yield data
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()


@pytest.fixture(autouse=True)
def isolate_state(orbitals):
    psi4.core.clean_options()
    psi4.core.clean_variables()
    yield
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()


def native_inputs(orbitals, nocc=None, nvir=None, shift=0.0, zero_virtual=False):
    primary, auxiliary, co, cv, eo, ev, _ = orbitals
    cv = cv[:, :nvir].copy()
    if zero_virtual:
        cv[:] = 0.0
    return (primary, auxiliary, psi4.core.Matrix.from_array(co[:, :nocc].copy()),
            psi4.core.Matrix.from_array(cv), psi4.core.Vector.from_array(eo[:nocc].copy()),
            psi4.core.Vector.from_array(ev[:nvir].copy() + shift))


def dimer(a, b, hybrid):
    matrices = {f"{key}_{label}": data[i] for label, data in (("A", a), ("B", b))
                for key, i in (("Cocc", 2), ("Cvir", 3))}
    vectors = {f"{key}_{label}": data[i] for label, data in (("A", a), ("B", b))
               for key, i in (("eps_occ", 4), ("eps_vir", 5))}
    return psi4.core.FDDS_Dispersion(a[0], a[1], matrices, vectors, hybrid)


@pytest.mark.parametrize("hybrid", [False, True])
def test_monomer_dimer_equivalence(orbitals, hybrid):
    a = native_inputs(orbitals)
    b = native_inputs(orbitals, nocc=4, nvir=11, shift=0.17)
    b[2].np[:] = orbitals[2][:, -4:]
    b[3].np[:] = orbitals[3][:, -11:]
    b[4].np[:] = orbitals[4][-4:]
    b[5].np[:] = orbitals[5][-11:] + 0.17
    pair = dimer(a, b, hybrid)
    density = psi4.core.Matrix.from_array(orbitals[-1])
    projected = pair.project_densities([density])[0].to_array()
    for label, data in (("A", a), ("B", b)):
        mono = psi4.core.FDDS_Monomer(*data, hybrid)
        for name in ("metric", "metric_inv", "aux_overlap"):
            np.testing.assert_allclose(getattr(mono, name)().np, getattr(pair, name)().np, atol=1.e-12, rtol=1.e-12)
        np.testing.assert_allclose(mono.project_densities([density])[0].np, projected, atol=1.e-12, rtol=1.e-12)
        metric, inv = mono.metric().to_array(), mono.metric_inv().to_array()
        kernel = metric + 0.03 * mono.aux_overlap().to_array()  # Explicit test policy, not an XC model.
        if hybrid:
            pair_r = getattr(pair, f"R_{label}")().to_array()
            np.testing.assert_allclose(mono.R().np, pair_r, atol=1.e-12, rtol=1.e-12)
            rtinv = np.linalg.pinv(mono.R().np, rcond=1.e-13).T
        for omega in (0.0, 0.4, 2.0):
            positive = mono.form_unc_amplitude(omega).to_array()
            np.testing.assert_allclose(positive, pair.form_unc_amplitude(label, omega).np, atol=1.e-12, rtol=1.e-12)
            assert np.linalg.eigvalsh(positive).min() > -1.e-12
            if hybrid:
                aux = {k: v.to_array() for k, v in mono.form_aux_matrices(omega).items()}
                other = {k: v.to_array() for k, v in pair.form_aux_matrices(label, omega).items()}
                for key in aux:
                    np.testing.assert_allclose(aux[key], other[key], atol=1.e-11, rtol=1.e-11)
                np.testing.assert_allclose(aux["amp"], -positive, atol=1.e-12, rtol=1.e-12)
                result = fdds_coupled_amplitudes(aux["amp"], metric, inv, kernel, exchange=aux,
                                                 x_alpha=0.25, Rtinv=rtinv)
                expected = fdds_coupled_amplitudes(other["amp"], metric, inv, kernel, exchange=other,
                                                   x_alpha=0.25, Rtinv=np.linalg.pinv(pair_r, rcond=1.e-13).T)
                np.testing.assert_allclose(result, expected, atol=1.e-11, rtol=1.e-11)
                no_exchange = fdds_coupled_amplitudes(aux["amp"], metric, inv, kernel, exchange=aux,
                                                       x_alpha=0.0, Rtinv=rtinv)
                np.testing.assert_allclose(no_exchange, fdds_coupled_amplitudes(-positive, metric, inv, kernel),
                                           atol=1.e-11, rtol=1.e-11)
            else:
                unc, coupled = fdds_coupled_amplitudes(-positive, metric, inv, kernel)
                # Independent full-rank Dyson solve, before the documented symmetrization.
                raw = np.linalg.solve(np.eye(len(metric)) - unc @ inv @ kernel @ inv, unc)
                np.testing.assert_allclose(coupled, 0.5 * (raw + raw.T), atol=1.e-11, rtol=1.e-11)
                with pytest.raises(RuntimeError, match="require hybrid"):
                    mono.form_aux_matrices(omega)
        with pytest.raises(RuntimeError, match="nonnegative"):
            mono.form_unc_amplitude(-1.0)
    assert np.linalg.norm(pair.form_unc_amplitude("A", 0.4).np - pair.form_unc_amplitude("B", 0.4).np) > 1.e-5


@pytest.mark.parametrize("nocc,nvir", [(1, 7), (5, 2), (1, 2)])
def test_hybrid_qr_supported_shapes(orbitals, nocc, nvir):
    data = native_inputs(orbitals, nocc=nocc, nvir=nvir)
    mono = psi4.core.FDDS_Monomer(*data, True)
    pair = dimer(data, data, True)
    assert mono.R().shape == (7, 7)
    for key, value in mono.form_aux_matrices(0.4).items():
        assert np.isfinite(value.np).all()
        np.testing.assert_allclose(value.np, pair.form_aux_matrices("A", 0.4)[key].np, atol=1.e-11, rtol=1.e-11)
    primary, auxiliary, co, cv = data[:4]
    naux, nbf = auxiliary.nbf(), primary.nbf()
    zero = psi4.core.BasisSet.zero_ao_basis_set()
    mints = psi4.core.MintsHelper(primary)
    ao = mints.ao_eri(auxiliary, zero, primary, primary).np.reshape(naux, nbf, nbf)
    metric = mints.ao_eri(auxiliary, zero, auxiliary, zero).np.reshape(naux, naux)
    ov = np.einsum("Pmn,mi,na->iaP", ao, co.np, cv.np)
    q_tensor = pair.get_tensor_pqQ("QarQ", (nocc, nvir, naux))
    assert (q_tensor.rows(), q_tensor.cols()) == (nocc * nvir, naux)
    # DFHelper retains (occupied, virtual, auxiliary) metadata in the NumPy view.
    q = q_tensor.np
    assert q.shape == ov.shape == (nocc, nvir, naux)
    r = mono.R().np
    np.testing.assert_allclose(r.T @ r, ov.reshape(-1, naux).T @ ov.reshape(-1, naux),
                               atol=1.e-11, rtol=1.e-11)
    np.testing.assert_allclose(q @ r, ov, atol=1.e-11, rtol=1.e-11)
    if nocc * nvir < naux:
        np.testing.assert_array_equal(r[nocc*nvir:], 0)
        np.testing.assert_array_equal(q[..., nocc*nvir:], 0)

    # Explicit DF exchange contraction checks the v < o blocking independently.
    eig, vectors = np.linalg.eigh(metric)
    half_inv = (vectors / np.sqrt(eig)) @ vectors.T
    oo = np.einsum("Pmn,mi,nj->ijP", ao, co.np, co.np) @ half_inv
    vv = np.einsum("Pmn,ma,nb->abP", ao, cv.np, cv.np) @ half_inv
    expected_y = np.einsum("ijP,abP,jbQ->iaQ", oo, vv, ov)
    y_tensor = pair.get_tensor_pqQ("YarQ", (nocc, nvir, naux))
    assert (y_tensor.rows(), y_tensor.cols()) == (nocc * nvir, naux)
    actual_y = y_tensor.np
    assert actual_y.shape == expected_y.shape == (nocc, nvir, naux)
    np.testing.assert_allclose(actual_y, expected_y, atol=1.e-11, rtol=1.e-11)


@pytest.mark.parametrize("zero_rank", [False, True])
def test_rank_deficiency_retains_pseudoinverse_policy(orbitals, zero_rank):
    data = native_inputs(orbitals, zero_virtual=zero_rank)
    if not zero_rank:
        data[3].np[:, 1:] = 0.0
    mono = psi4.core.FDDS_Monomer(*data, True)
    rank = np.linalg.matrix_rank(mono.R().np)
    assert rank == 0 if zero_rank else 0 < rank < 7
    aux = {k: v.to_array() for k, v in mono.form_aux_matrices(0.4).items()}
    result = fdds_coupled_amplitudes(aux["amp"], mono.metric().np, mono.metric_inv().np, mono.metric().np,
                                     exchange=aux, x_alpha=0.25, Rtinv=np.linalg.pinv(mono.R().np, rcond=1.e-13).T)
    assert np.isfinite(result).all()
    if zero_rank:
        np.testing.assert_array_equal(result, np.zeros((2, 7, 7)))
    else:
        pair = dimer(data, data, True)
        for key, value in pair.form_aux_matrices("A", 0.4).items():
            np.testing.assert_allclose(aux[key], value.np, atol=1.e-11, rtol=1.e-11)


def test_invalid_shapes_fail_cleanly(orbitals):
    data = list(native_inputs(orbitals))
    data[4] = psi4.core.Vector(1)
    with pytest.raises(RuntimeError, match="orbital dimensions"):
        psi4.core.FDDS_Monomer(*data, False)
    data = list(native_inputs(orbitals))
    data[2].np[0, 0] = np.nan
    with pytest.raises(RuntimeError, match="must be finite"):
        psi4.core.FDDS_Monomer(*data, False)
    data = list(native_inputs(orbitals))
    data[2] = psi4.core.Matrix(data[0].nbf() - 1, 5)
    with pytest.raises(RuntimeError, match="orbital dimensions"):
        psi4.core.FDDS_Monomer(*data, False)
    mono = psi4.core.FDDS_Monomer(*native_inputs(orbitals), False)
    with pytest.raises(RuntimeError, match="densities must"):
        mono.project_densities([psi4.core.Matrix(2, 2)])
    with pytest.raises(RuntimeError, match="require hybrid"):
        mono.R()
    metric = mono.metric().np
    with pytest.raises(ValueError, match="naux by naux"):
        fdds_coupled_amplitudes(np.zeros((2, 2)), metric, metric, metric)
    with pytest.raises(ValueError, match="requires hybrid"):
        fdds_coupled_amplitudes(metric, metric, metric, metric, x_alpha=0.25)
    for nocc, nvir in ((0, None), (None, 0)):
        with pytest.raises(RuntimeError, match="orbital dimensions"):
            psi4.core.FDDS_Monomer(*native_inputs(orbitals, nocc=nocc, nvir=nvir), False)
    for value in (np.nan, np.inf):
        data = list(native_inputs(orbitals))
        data[5].np[0] = value
        with pytest.raises(RuntimeError, match="energies must be finite"):
            psi4.core.FDDS_Monomer(*data, False)
        with pytest.raises(RuntimeError, match="frequency must be finite"):
            mono.form_unc_amplitude(value)
    for matrices in ({}, {"Cocc_A": None}):
        with pytest.raises(RuntimeError, match="missing orbital data"):
            psi4.core.FDDS_Dispersion(orbitals[0], orbitals[1], matrices, {}, False)
    retained = psi4.core.FDDS_Monomer(*native_inputs(orbitals), False)
    gc.collect()  # The temporary input wrappers are gone; native shared ownership remains.
    assert np.isfinite(retained.form_unc_amplitude(0.4).np).all()


def test_hybrid_algebra_and_input_ownership():
    metric = np.array([[2., 1.], [1., 3.]])
    inverse = np.array([[3., -1.], [-1., 2.]]) / 5
    unc = np.array([[-3., .5], [.5, -2.]])
    kernel = np.array([[.7, -.3], [-.3, 1.1]])
    rtinv = np.array([[1., .5], [0., 1.]])
    alpha = .5
    exchange = {
        "K1LD": np.array([[.2, .4], [.1, -.3]]),
        "K2LD": np.array([[-.4, .2], [.3, .1]]),
        "K2L": np.array([[.25, .75], [-.5, 1.]]),
    }
    x = unc - alpha * exchange["K2L"]
    assert np.linalg.norm(metric @ x - x @ metric) > .1
    for operator, pinv in (
        (np.array([[3., 1.], [0., 2.]]), np.array([[1/3, -1/6], [0., .5]])),
        (np.array([[1., 2.], [2., 4.]]), np.array([[1., 2.], [2., 4.]]) / 25),
    ):
        # Set J-XSW to an operator with an independently known pseudoinverse.
        correction = 4 * (metric - operator - x @ inverse @ kernel)
        exchange["K21L"] = (
            np.linalg.solve((rtinv @ metric).T, correction.T).T
            + alpha * (exchange["K1LD"] + exchange["K2LD"])
        ) / alpha**2
        inputs = [unc, metric, inverse, kernel, rtinv, *exchange.values()]
        saved = [array.copy() for array in inputs]
        result = fdds_coupled_amplitudes(unc, metric, inverse, kernel,
                                         exchange=exchange, x_alpha=alpha, Rtinv=rtinv)
        expected = metric @ pinv @ x + (np.eye(2) - operator @ pinv) @ x
        np.testing.assert_allclose(result[0], unc, rtol=0, atol=1.e-13)
        np.testing.assert_allclose(result[1], (expected + expected.T) / 2, rtol=1.e-12, atol=1.e-12)
        for array, snapshot in zip(inputs, saved):
            np.testing.assert_array_equal(array, snapshot)
