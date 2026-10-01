"""Native FDDS monomer behavior; deliberately small auxiliary bases test QR shapes."""

import gc
import os
import re

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


def test_wide_hybrid_response_matches_qr_free_reference(orbitals):
    """Wide (ar|P) (nov=2 < naux=7): Q-bearing terms against an oracle with no QR.

    For any valid Q R = A of full-row-rank A, Q pinv(R)^T = pinv(A)^T, and the driver uses
    Q only through K @ pinv(R)^T. Every operand below comes from Mints. The pre-fix QR
    (invalid DORGQR, out-of-bounds R copy) missed this oracle by O(1).
    Uses only the two-monomer API.
    """
    nocc, nvir, omega, x_alpha = 1, 2, 0.4, 0.25
    data = native_inputs(orbitals, nocc=nocc, nvir=nvir)
    primary, auxiliary, co, cv, eo, ev = data
    naux, nbf, nov = auxiliary.nbf(), primary.nbf(), nocc * nvir
    zero = psi4.core.BasisSet.zero_ao_basis_set()
    mints = psi4.core.MintsHelper(primary)
    ao = mints.ao_eri(auxiliary, zero, primary, primary).np.reshape(naux, nbf, nbf)
    metric = mints.ao_eri(auxiliary, zero, auxiliary, zero).np.reshape(naux, naux)
    eig, vectors = np.linalg.eigh(metric)
    half_inv = (vectors / np.sqrt(eig)) @ vectors.T
    A = np.einsum("Pmn,mi,na->iaP", ao, co.np, cv.np).reshape(nov, naux)
    assert np.linalg.matrix_rank(A) == nov and np.linalg.cond(A) < 10
    B = A @ half_inv
    oo = np.einsum("Pmn,mi,nj->ijP", ao, co.np, co.np) @ half_inv
    vv = np.einsum("Pmn,ma,nb->abP", ao, cv.np, cv.np) @ half_inv
    Vx = B @ B.T                                                           # (ia|jb)
    Vy = np.einsum("ijP,abP->iajb", oo, vv).reshape(nov, nov)              # (ij|ab)
    QRt = np.linalg.pinv(A).T                                              # Q pinv(R)^T
    delta = (ev.np[None, :] - eo.np[:, None]).ravel()
    lam = -4.0 / (delta**2 + omega**2)
    LDA, Yp = (delta * lam)[:, None] * A, (Vy - Vx) @ A
    ref = {"K1LD": LDA.T @ (Vx + Vy) @ QRt, "K2LD": LDA.T @ (Vy - Vx) @ QRt,
           "K21L": (lam[:, None] * Yp).T @ (Vx + Vy) @ QRt}
    kernel = metric + 0.03 * mints.ao_overlap(auxiliary, auxiliary).np  # Explicit test policy.
    exchange = dict(ref, K2L=A.T @ (lam[:, None] * Yp), amp=LDA.T @ A)  # amp is already negative.
    expected = fdds_coupled_amplitudes(exchange["amp"], metric, np.linalg.inv(metric), kernel,
                                       exchange=exchange, x_alpha=x_alpha, Rtinv=np.eye(naux))

    runs = []
    for _ in range(2):
        pair = dimer(data, data, True)
        rtinv = np.linalg.pinv(pair.R_A().np, rcond=1.e-13).T
        aux = {k: v.to_array() for k, v in pair.form_aux_matrices("A", omega).items()}
        assert all(np.isfinite(v).all() for v in [rtinv, *aux.values()])
        for key, value in ref.items():
            np.testing.assert_allclose(aux[key] @ rtinv, value, atol=1.e-11, rtol=1.e-11)
        for key in ("K2L", "amp"):
            np.testing.assert_allclose(aux[key], exchange[key], atol=1.e-11, rtol=1.e-11)
        result = fdds_coupled_amplitudes(aux["amp"], pair.metric().np, pair.metric_inv().np, kernel,
                                         exchange=aux, x_alpha=x_alpha, Rtinv=rtinv)
        np.testing.assert_allclose(result, expected, atol=1.e-11, rtol=1.e-11)
        runs.append(result)
        del pair
    np.testing.assert_array_equal(runs[0], runs[1])


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


# ==> Declared-basis path with explicit resources <==


def declared(data, hybrid, scratch, T=None, extra=0, disk_extra=0, subalgo="OUT_OF_CORE", nthread=1):
    primary, auxiliary = data[:2]
    naux = auxiliary.nbf() if T is None else T.shape[0]
    req = psi4.core.FDDS_Monomer.requirement(primary, auxiliary, data[2].cols(), data[3].cols(), naux, hybrid,
                                             subalgo, nthread)
    return psi4.core.FDDS_Monomer(*data, hybrid, memory_bytes=req["memory_bytes"] + extra,
                                  disk_bytes=req["disk_bytes"] + disk_extra, scratch_dir=str(scratch),
                                  nthread=nthread, subalgo=subalgo,
                                  aux_transform=None if T is None else psi4.core.Matrix.from_array(T))


def truncated_power(J, alpha):
    w, U = np.linalg.eigh(J)
    keep = np.abs(w) >= 1.e-12 * np.abs(w).max()
    return (U[:, keep] * w[keep]**alpha) @ U[:, keep].T


def ov_reference(data, omega, x_alpha, kernel, T=None):
    """Explicit n x n OV-space (S1) response from Mints integrals: sym(b^T (I - S)^-1 N b), b = B J^-1."""
    primary, auxiliary, co, cv, eo, ev = (d.np if hasattr(d, "np") else d for d in data)
    naux, nbf, o, v = auxiliary.nbf(), primary.nbf(), co.shape[1], cv.shape[1]
    zero = psi4.core.BasisSet.zero_ao_basis_set()
    mints = psi4.core.MintsHelper(primary)
    ao = mints.ao_eri(auxiliary, zero, primary, primary).np.reshape(naux, nbf, nbf)
    J = mints.ao_eri(auxiliary, zero, auxiliary, zero).np.reshape(naux, naux)
    if T is not None:
        ao, J = np.einsum("dP,Pmn->dmn", T, ao), T @ J @ T.T
    B = np.einsum("Pmn,mi,na->iaP", ao, co, cv).reshape(o * v, -1)
    half = truncated_power(J, -0.5)
    C = (B @ half).reshape(o, v, -1)
    oo = np.einsum("Pmn,mi,nj->ijP", ao, co, co) @ half
    vv = np.einsum("Pmn,ma,nb->abP", ao, cv, cv) @ half
    x = np.einsum("jaP,ibP->iajb", C, C).reshape(o * v, -1)            # (ib|ja)
    y = np.einsum("ijP,abP->iajb", oo, vv).reshape(o * v, -1)          # (ij|ab)
    U, s, _ = np.linalg.svd(B, full_matrices=False)
    P = U[:, s > 1.e-13 * s.max()] @ U[:, s > 1.e-13 * s.max()].T     # native QR range projector
    delta = (ev[None, :] - eo[:, None]).ravel()
    lam = -4.0 / (delta**2 + omega**2)
    N = lam[:, None] * (np.diag(delta) - x_alpha * (y - x))
    M = -x_alpha * (lam * delta)[:, None] * (2 * y) + x_alpha**2 * (y - x) @ (lam[:, None] * (y + x))
    b, bt = np.linalg.solve(J, B.T).T, B @ truncated_power(J, -1.0)
    chi = b.T @ np.linalg.solve(np.eye(len(B)) - N @ bt @ kernel @ b.T - 0.25 * M @ P, N @ b)
    return 0.5 * (chi + chi.T)


def rel(a, b):
    return np.abs(a - b).max() / np.abs(b).max()


@pytest.mark.parametrize("hybrid", [False, True])
def test_declared_identity_matches_legacy(orbitals, hybrid, tmp_path):
    data = native_inputs(orbitals)
    legacy = psi4.core.FDDS_Monomer(*data, hybrid)
    mono = declared(data, hybrid, tmp_path)
    for name in ("metric", "metric_inv", "aux_overlap"):
        np.testing.assert_array_equal(getattr(mono, name)().np, getattr(legacy, name)().np)
    np.testing.assert_allclose(mono.form_unc_amplitude(0.4).np, legacy.form_unc_amplitude(0.4).np, atol=1.e-11)
    if hybrid:
        np.testing.assert_allclose(mono.R().np, legacy.R().np, atol=1.e-11)
        other = legacy.form_aux_matrices(0.4)
        for key, value in mono.form_aux_matrices(0.4).items():
            np.testing.assert_allclose(value.np, other[key].np, atol=1.e-11)
    # Minimal admitted budget, a generous one, INCORE AOs and an explicit two-thread instance
    # (accounted for two threads, whatever the process setting) agree. With sto-3g auxiliaries the
    # minimal budget still holds every occupied orbital in one response block; multi-block responses
    # are compared with the OV reference in the cc-pVDZ-RI and JKFIT tests below.
    kernel = psi4.core.Matrix.from_array(mono.metric().np + 0.03 * mono.aux_overlap().np)
    alpha = 0.25 if hybrid else 0.0
    base = mono.form_coefficient_response(0.4, alpha, kernel)["response"].np
    others = (declared(data, hybrid, tmp_path, extra=10**8), declared(data, hybrid, tmp_path, subalgo="INCORE"),
              declared(data, hybrid, tmp_path, nthread=2))
    assert others[2].model()["nthread"] == 2
    assert others[2].model()["required_memory_bytes"] > mono.model()["required_memory_bytes"]
    for other in others:
        np.testing.assert_allclose(other.form_coefficient_response(0.4, alpha, kernel)["response"].np, base,
                                   rtol=0, atol=1.e-12 * np.abs(base).max())


@pytest.mark.parametrize("case", ["tall", "wide", "rank_deficient", "nonhybrid"])
def test_declared_response_matches_ov_reference(orbitals, case, tmp_path):
    data = native_inputs(orbitals, **({"nocc": 1, "nvir": 2} if case == "wide" else {}))
    if case == "rank_deficient":
        data[3].np[:, 1:] = 0.0
    hybrid = case != "nonhybrid"
    alpha = 0.25 if hybrid else 0.0
    mono = declared(data, hybrid, tmp_path)
    model = mono.model()
    assert model["qr_rank"] == {"tall": 7, "wide": 2, "rank_deficient": 5, "nonhybrid": 0}[case]
    kernel = mono.metric().np + 0.03 * mono.aux_overlap().np  # Explicit test policy, not an XC model.
    for omega in (0.0, 0.4, 2.0):
        result = mono.form_coefficient_response(omega, alpha, psi4.core.Matrix.from_array(kernel))
        assert min(result["native_dyson_ratio"], result["s2_dyson_ratio"]) > model["dyson_refusal"]
        assert result["solve_residual"] < 1.e-13
        chi = result["response"].np
        np.testing.assert_array_equal(chi, chi.T)
        assert np.linalg.eigvalsh(chi).max() < 1.e-12 * np.abs(chi).max()
        assert rel(chi, ov_reference(data, omega, alpha, kernel)) < 1.e-10


def test_nonhybrid_multiblock_response_matches_ov_reference(orbitals, tmp_path):
    """Non-hybrid S2 with cc-pVDZ-RI: the minimal budget streams one occupied orbital per block."""
    data = native_inputs(orbitals)
    data = (data[0], psi4.core.BasisSet.build(data[0].molecule(), "DF_BASIS_MP2", "cc-pvdz-ri", "RIFIT",
                                              "cc-pvdz"), *data[2:])
    minimal, generous = declared(data, False, tmp_path), declared(data, False, tmp_path, extra=10**8)
    assert minimal.model()["metric_dropped"] == 0
    kernel = minimal.metric().np + 0.03 * minimal.aux_overlap().np  # Explicit test policy.
    chi = minimal.form_coefficient_response(0.4, 0.0, psi4.core.Matrix.from_array(kernel))["response"].np
    chi_generous = generous.form_coefficient_response(0.4, 0.0, psi4.core.Matrix.from_array(kernel))["response"].np
    np.testing.assert_allclose(chi, chi_generous, rtol=0, atol=1.e-12 * np.abs(chi).max())
    assert rel(chi, ov_reference(data, 0.4, 0.0, kernel)) < 1.e-10


def test_ill_conditioned_response_is_stable(orbitals, tmp_path):
    """def2-universal-JKFIT on water: cond(J) = 2.5e7, no metric truncation.

    One-ulp metric perturbations move the OV reference by at most 2.4e-9 here, so 1e-7 is above
    the operand floor. Native amplitudes converted by a J^-1 C J^-1 sandwich miss it by 2e-6 to 2e-5.
    """
    data = native_inputs(orbitals)
    data = (data[0], psi4.core.BasisSet.build(data[0].molecule(), "DF_BASIS_SCF", "def2-universal-jkfit"), *data[2:])
    mono = declared(data, True, tmp_path)
    assert mono.model()["metric_dropped"] == 0
    legacy = psi4.core.FDDS_Monomer(*data, True)
    J, Jplus = legacy.metric().np, legacy.metric_inv().np
    kernel = J + 0.03 * legacy.aux_overlap().np
    reference = ov_reference(data, 0.4, 0.25, kernel)
    chi = mono.form_coefficient_response(0.4, 0.25, psi4.core.Matrix.from_array(kernel))["response"].np
    assert rel(chi, reference) < 1.e-7
    aux = {k: v.to_array() for k, v in legacy.form_aux_matrices(0.4).items()}
    _, coupled = fdds_coupled_amplitudes(aux["amp"], J, Jplus, kernel, exchange=aux, x_alpha=0.25,
                                         Rtinv=np.linalg.pinv(legacy.R().np, rcond=1.e-13).T)
    sandwich = np.linalg.solve(J, np.linalg.solve(J, coupled).T).T
    assert rel(sandwich, reference) > 1.e-6


def test_transform_is_applied_before_inversion(orbitals, tmp_path):
    data = native_inputs(orbitals)
    kernel = lambda m: psi4.core.Matrix.from_array(m.metric().np + 0.03 * m.aux_overlap().np)
    # Square: permutation times positive diagonal; the model is covariant (no truncation fires).
    T = np.diag(np.linspace(0.5, 2.0, 7))[[3, 0, 6, 1, 5, 2, 4]]
    raw, mono = declared(data, True, tmp_path), declared(data, True, tmp_path, T=T)
    np.testing.assert_allclose(mono.metric().np, T @ raw.metric().np @ T.T, rtol=1.e-14, atol=1.e-14)
    assert rel(mono.metric_inv().np, np.linalg.inv(T @ raw.metric().np @ T.T)) < 1.e-10
    chi = mono.form_coefficient_response(0.4, 0.25, kernel(mono))["response"].np
    chi_raw = raw.form_coefficient_response(0.4, 0.25, kernel(raw))["response"].np
    Tinv = np.linalg.inv(T)
    assert rel(chi, Tinv.T @ chi_raw @ Tinv) < 1.e-10

    # Rectangular: Cartesian cc-pVDZ-RI mapped onto its spherical subspace equals the native spherical basis.
    mol, primary = data[0].molecule(), data[0]
    sph = psi4.core.BasisSet.build(mol, "DF_BASIS_MP2", "cc-pvdz-ri", "RIFIT", "cc-pvdz", 1)
    cart = psi4.core.BasisSet.build(mol, "DF_BASIS_MP2", "cc-pvdz-ri", "RIFIT", "cc-pvdz", 0)
    mints = psi4.core.MintsHelper(primary)
    T = np.linalg.solve(mints.ao_overlap(cart, cart).np, mints.ao_overlap(sph, cart).np.T).T
    assert T.shape == (84, 96)
    mapped = declared((primary, cart, *data[2:]), True, tmp_path, T=T)
    native = declared((primary, sph, *data[2:]), True, tmp_path)
    cartesian = declared((primary, cart, *data[2:]), True, tmp_path)
    for name in ("metric", "metric_inv"):
        assert rel(getattr(mapped, name)().np, getattr(native, name)().np) < 1.e-10
    chi = mapped.form_coefficient_response(0.4, 0.25, kernel(native))["response"].np
    chi_native = native.form_coefficient_response(0.4, 0.25, kernel(native))["response"].np
    assert rel(chi, chi_native) < 1.e-10
    # Mapping the finished Cartesian response (integral space) is a different model.
    Jr, Jd = cartesian.metric().np, native.metric().np
    chi_cart = cartesian.form_coefficient_response(0.4, 0.25, kernel(cartesian))["response"].np
    projected = np.linalg.solve(Jd, np.linalg.solve(Jd, T @ Jr @ chi_cart @ Jr @ T.T).T).T
    assert rel(projected, chi_native) > 1.e-5


def planted_kernel(mono, ratio, omega=0.4, x_alpha=0.25):
    """Kernel making the natively formed (legacy helper order) J - XSW equal J with its
    smallest eigenvalue scaled to ratio."""
    J, Jplus = mono.metric().np, mono.metric_inv().np
    aux = {k: v.to_array() for k, v in mono.form_aux_matrices(omega).items()}
    X = aux["amp"] - x_alpha * aux["K2L"]
    K = -x_alpha * aux["K1LD"] - x_alpha * aux["K2LD"] + x_alpha * x_alpha * aux["K21L"]
    KRS = K @ np.linalg.pinv(mono.R().np, rcond=1.e-13).T @ J
    w, U = np.linalg.eigh(J)
    D = (U * (w.max() * np.r_[ratio, np.ones(len(w) - 1)])) @ U.T
    return psi4.core.Matrix.from_array(np.linalg.solve(X @ Jplus, J - 0.25 * KRS - D))


def test_dyson_admission_policy(orbitals, tmp_path):
    """Both J - XSW formations must keep sigma_min/sigma_max > 2e-13 (empirical policy, not an accuracy bound)."""
    data = native_inputs(orbitals)
    threshold = declared(data, True, tmp_path).model()["dyson_refusal"]

    # A planted singular value on either side of 2e-13; on this well-conditioned case the S2
    # formation agrees, and the native check refuses first.
    mono = declared(data, True, tmp_path)
    for ratio, admitted in ((5.e-14, False), (2.e-12, True)):
        kernel = planted_kernel(mono, ratio)
        if admitted:
            result = mono.form_coefficient_response(0.4, 0.25, kernel)
            for key in ("native_dyson_ratio", "s2_dyson_ratio"):
                assert result[key] == pytest.approx(ratio, rel=1.e-2)
        else:
            with pytest.raises(RuntimeError, match=r"native J - XSW check.*S2 J - J\*A check was not evaluated"):
                mono.form_coefficient_response(0.4, 0.25, kernel)

    # Near-duplicate declared functions make J_d singular to machine precision (T1 drops one
    # direction at ~3e-16). Both ratios then sit at a rounding floor and the outcomes differ between
    # BLAS code paths, so only the decision rule is asserted: admission needs both ratios above
    # threshold; a refusal names its check. Admitted responses here are not accurate (or always
    # negative semidefinite); this test does not check them. A finite-ratio native-pass/S2-refuse
    # node is not required: the formations are equal in exact arithmetic and differ only by
    # rounding. The nonfinite S2 refusal is asserted deterministically in the overflow test below.
    outcomes = []
    for delta in (1.25e-7, 2.5e-7, 3.2e-6):
        T = np.eye(7)
        T[1] = T[0] + delta * T[1]
        redundant = declared(data, True, tmp_path, T=T)
        assert redundant.model()["metric_dropped"] == 1
        kernel = psi4.core.Matrix.from_array(redundant.metric().np + 0.03 * redundant.aux_overlap().np)
        for omega in (0.0, 0.4, 2.0):
            try:
                result = redundant.form_coefficient_response(omega, 0.25, kernel)
            except RuntimeError as err:
                msg = str(err)
                check, value = re.search(r"refused by the (.*?) check: sigma_min/sigma_max = (\S+) <=", msg).groups()
                assert float(value) <= threshold or value == "nan"
                if check == "native J - XSW":
                    assert "S2 J - J*A check was not evaluated" in msg
                else:
                    assert check == "S2 J - J*A"
                    assert float(re.search(r"check passed with (\S+)\.", msg).group(1)) > threshold
                outcomes.append(check)
            else:
                assert min(result["native_dyson_ratio"], result["s2_dyson_ratio"]) > threshold
                assert np.isfinite(result["response"].np).all()
                outcomes.append("admitted")
    assert "native J - XSW" in outcomes


def test_dyson_s2_overflow_refusal(orbitals, tmp_path):
    """Native check passes, fused S2 formation overflows: refused as nan by the S2 check.

    T = s I is covariant: it leaves X J+ (all the native products see) unchanged and scales
    chi0_in = J_d^-1 X J+ by 1/s^2. The finite kernel is sized so that every native product stays
    1e3 below DBL_MAX, while chi0_in W exceeds it by about 1e2 (s = 1e-3) or stays finite (s = 1).
    """
    data = native_inputs(orbitals)
    big = np.finfo(float).max

    def kernel(mono):
        J = mono.metric().np
        aux = {k: v.to_array() for k, v in mono.form_aux_matrices(0.4).items()}
        XJ = (aux["amp"] - 0.25 * aux["K2L"]) @ mono.metric_inv().np
        W0 = J + 0.03 * mono.aux_overlap().np
        return psi4.core.Matrix.from_array((big / (1.e3 * len(J) * np.abs(XJ).max())) * (W0 / np.abs(W0).max()))

    control = declared(data, True, tmp_path, T=np.eye(7))
    result = control.form_coefficient_response(0.4, 0.25, kernel(control))
    threshold = control.model()["dyson_refusal"]
    assert min(result["native_dyson_ratio"], result["s2_dyson_ratio"]) > threshold
    assert np.isfinite(result["response"].np).all()

    scaled = declared(data, True, tmp_path, T=1.e-3 * np.eye(7))
    with pytest.raises(RuntimeError) as err:
        scaled.form_coefficient_response(0.4, 0.25, kernel(scaled))
    passed = re.search(r"refused by the S2 J - J\*A check: sigma_min/sigma_max = nan <= 2e-13 at omega = 0.4; "
                       r"the native J - XSW check passed with (\S+)\.", str(err.value))
    assert passed and float(passed.group(1)) > threshold
    # The refusal leaves the instance usable.
    ordinary = psi4.core.Matrix.from_array(scaled.metric().np + 0.03 * scaled.aux_overlap().np)
    assert np.isfinite(scaled.form_coefficient_response(0.4, 0.25, ordinary)["response"].np).all()


def test_declared_invalid_inputs(orbitals, tmp_path):
    data = native_inputs(orbitals)
    for T, match in ((np.eye(7)[:, :6], "aux_transform"), (np.full((7, 7), np.nan), "aux_transform must be finite")):
        with pytest.raises(RuntimeError, match=match):
            declared(data, True, tmp_path, T=T)
    with pytest.raises(RuntimeError, match="subalgo"):
        psi4.core.FDDS_Monomer(*data, True, memory_bytes=10**9, disk_bytes=10**9, scratch_dir=str(tmp_path),
                               nthread=1, subalgo="AUTO")
    with pytest.raises(RuntimeError, match="scratch_dir"):
        declared(data, True, tmp_path / "missing")
    shifted = native_inputs(orbitals, shift=-1.0)  # lowest virtual below the HOMO
    with pytest.raises(RuntimeError, match="every virtual energy"):
        declared(shifted, True, tmp_path)
    mono, nonhybrid = declared(data, True, tmp_path), declared(data, False, tmp_path)
    good = psi4.core.Matrix.from_array(mono.metric().np)
    for args, match in (((0.4, 0.25, psi4.core.Matrix(6, 6)), "kernel"),
                        ((0.4, 0.25, psi4.core.Matrix.from_array(np.full((7, 7), np.inf))), "kernel"),
                        ((-1.0, 0.25, good), "nonnegative"), ((0.4, np.nan, good), "x_alpha")):
        with pytest.raises(RuntimeError, match=match):
            mono.form_coefficient_response(*args)
    with pytest.raises(RuntimeError, match="x_alpha"):
        nonhybrid.form_coefficient_response(0.4, 0.25, good)
    legacy = psi4.core.FDDS_Monomer(*data, True)
    with pytest.raises(RuntimeError, match="declared"):
        legacy.form_coefficient_response(0.4, 0.25, good)
    with pytest.raises(RuntimeError, match="declared"):
        legacy.model()
    with pytest.raises(RuntimeError, match="declared path"):
        mono.project_densities([psi4.core.Matrix.from_array(orbitals[-1])])
    with pytest.raises(RuntimeError, match="nthread"):
        psi4.core.FDDS_Monomer(*data, True, memory_bytes=10**9, disk_bytes=10**9, scratch_dir=str(tmp_path),
                               nthread=0)


def test_declared_resources_and_private_scratch(orbitals, tmp_path):
    data = native_inputs(orbitals)
    psio = psi4.core.IOManager.shared_object()
    default = psio.get_default_path()
    watched = lambda: {f for f in os.listdir(default) if "dfh" in f}
    gc.collect()
    before = (psi4.core.get_memory(), default, psi4.core.get_global_option("SCF_SUBTYPE"),
              psi4.core.has_global_option_changed("SCF_SUBTYPE"), psi4.core.get_num_threads(), watched())
    size = lambda: sum(os.path.getsize(tmp_path / f) for f in os.listdir(tmp_path))
    req = psi4.core.FDDS_Monomer.requirement(data[0], data[1], 5, 19, 7, True, "OUT_OF_CORE", 1)
    assert req["memory_bytes"] > req["resident_bytes"] and req["disk_bytes"] >= req["disk:steady"]

    # Admission refuses one byte short, before any integral or file.
    for extra, disk_extra, match in ((-1, 0, "memory_bytes"), (0, -1, "disk_bytes")):
        with pytest.raises(RuntimeError, match=match):
            declared(data, True, tmp_path, extra=extra, disk_extra=disk_extra)
        assert os.listdir(tmp_path) == []

    # Global options, memory and path do not enter the explicit path.
    first = declared(data, True, tmp_path)
    kernel = psi4.core.Matrix.from_array(first.metric().np + 0.03 * first.aux_overlap().np)
    expected = first.form_coefficient_response(0.4, 0.25, kernel)["response"].np
    assert size() == req["disk:steady"]  # every stream has been read back, so no stdio buffering remains
    psi4.set_options({"scf_subtype": "incore"})
    psi4.core.set_memory_bytes(10**6)  # far below what the legacy path would need
    second = declared(data, True, tmp_path)
    np.testing.assert_array_equal(second.form_coefficient_response(0.4, 0.25, kernel)["response"].np, expected)
    assert size() == 2 * req["disk:steady"] and watched() == before[-1]
    psi4.core.set_memory_bytes(before[0])
    psi4.core.clean_options()

    # A refusal leaves the instance usable; each instance removes only its own files.
    with pytest.raises(RuntimeError, match="Dyson admission refused"):
        first.form_coefficient_response(0.4, 0.25, planted_kernel(first, 5.e-14))
    np.testing.assert_array_equal(first.form_coefficient_response(0.4, 0.25, kernel)["response"].np, expected)
    del first
    gc.collect()
    assert size() == req["disk:steady"]
    np.testing.assert_array_equal(second.form_coefficient_response(0.4, 0.25, kernel)["response"].np, expected)
    del second
    gc.collect()
    assert os.listdir(tmp_path) == [] and tmp_path.is_dir()
    assert (psi4.core.get_memory(), psio.get_default_path(), psi4.core.get_global_option("SCF_SUBTYPE"),
            psi4.core.has_global_option_changed("SCF_SUBTYPE"), psi4.core.get_num_threads(), watched()) == before
