"""Frozen-orbital checks of FDDS storage modes and response work buffers."""

import numpy as np
import pytest

import psi4


pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.saptdft]


@pytest.fixture
def fdds_input():
    """Use identical, fixed monomer orbitals so only the FDDS implementation changes."""
    old_memory = psi4.core.get_memory()
    old_threads = psi4.core.get_num_threads()
    psi4.core.set_memory_bytes(256 * 1024**2, True)
    psi4.set_num_threads(1)
    try:
        mol = psi4.geometry("""
        O 0.0 0.0 0.0
        H 0.0 0.0 1.0
        H 0.0 1.0 0.0
        symmetry c1
        """)
        psi4.set_options({"basis": "cc-pvdz", "scf_type": "pk"})
        _, wfn = psi4.energy("hf", molecule=mol, return_wfn=True)
        auxiliary = psi4.core.BasisSet.build(mol, "DF_BASIS_MP2", "cc-pvdz-ri", "RIFIT", "cc-pvdz")
        matrices, vectors = {}, {}
        for monomer in ("A", "B"):
            matrices[f"Cocc_{monomer}"] = wfn.Ca_subset("AO", "OCC")
            matrices[f"Cvir_{monomer}"] = wfn.Ca_subset("AO", "VIR")
            vectors[f"eps_occ_{monomer}"] = wfn.epsilon_a_subset("AO", "OCC")
            vectors[f"eps_vir_{monomer}"] = wfn.epsilon_a_subset("AO", "VIR")
        yield wfn.basisset(), auxiliary, matrices, vectors
    finally:
        psi4.core.set_memory_bytes(old_memory, True)
        psi4.set_num_threads(old_threads)


def _aux_matrices(obj, monomer, frequency, is_lrc):
    return {k: v.np.copy() for k, v in obj.form_aux_matrices(
        monomer, frequency, 0.25, 0.75 if is_lrc else 0.0).items()}


@pytest.mark.parametrize("omega", [0.0, 0.1, 0.4])
def test_fdds_storage_mode_parity(fdds_input, omega):
    """Finite-range exchange must not become Coulomb exchange out of core.

    Comparing every auxiliary response matrix catches the wrong three-index
    operator even when an integrated energy could mask it. omega=0 exercises
    the unchanged global-hybrid path; the finite-omega cases include mixed
    full-range/LR exchange products. Also covers STORE vs DIRECT/DIRECT_IAQ/LEGACY,
    SAPT_FDDS_DISP_AO_STORAGE, and split_monomers toggling.
    """
    primary, auxiliary, matrices, vectors = fdds_input
    is_lrc = omega != 0.0
    cases = [
        ("store_incore_split", {"scf_subtype": "INCORE", "sapt_fdds_disp_df_algorithm": "STORE",
                                "sapt_fdds_disp_ao_storage": "INCORE", "sapt_fdds_disp_split_monomers": True}),
        ("store_ooc_split", {"scf_subtype": "AUTO", "sapt_fdds_disp_df_algorithm": "STORE",
                             "sapt_fdds_disp_ao_storage": "OUT_OF_CORE", "sapt_fdds_disp_split_monomers": True}),
        ("store_ooc_nosplit", {"scf_subtype": "OUT_OF_CORE", "sapt_fdds_disp_df_algorithm": "STORE",
                               "sapt_fdds_disp_ao_storage": "AUTO", "sapt_fdds_disp_split_monomers": False}),
        ("direct_incore", {"scf_subtype": "INCORE", "sapt_fdds_disp_df_algorithm": "DIRECT",
                           "sapt_fdds_disp_ao_storage": "AUTO", "sapt_fdds_disp_split_monomers": True}),
        ("direct_ooc", {"scf_subtype": "OUT_OF_CORE", "sapt_fdds_disp_df_algorithm": "DIRECT",
                        "sapt_fdds_disp_ao_storage": "AUTO", "sapt_fdds_disp_split_monomers": True}),
        ("legacy_ooc", {"scf_subtype": "OUT_OF_CORE", "sapt_fdds_disp_df_algorithm": "LEGACY",
                        "sapt_fdds_disp_ao_storage": "AUTO", "sapt_fdds_disp_split_monomers": True}),
    ]
    results = {}
    for label, opts in cases:
        psi4.set_options(opts)
        obj = psi4.core.FDDS_Dispersion(primary, auxiliary, matrices, vectors, True, is_lrc, omega)
        results[label] = {
            (monomer, frequency): _aux_matrices(obj, monomer, frequency, is_lrc)
            for monomer in ("A", "B") for frequency in (0.0, 0.3)
        }
        del obj
    ref_results = results["store_incore_split"]
    for label, res in results.items():
        if label == "store_incore_split":
            continue
        for case, reference in ref_results.items():
            for key, value in reference.items():
                np.testing.assert_allclose(
                    res[case][key], value, rtol=1.e-9, atol=1.e-10,
                    err_msg=f"omega={omega}, mode={label}, {case}, {key}",
                )


@pytest.mark.parametrize("is_lrc", [False, True])
def test_fdds_aux_memory_blocking(fdds_input, is_lrc):
    """Reject a budget below one block; one/two-occupied blocks match unblocked.

    Lower memory only after constructing FDDS so this specifically tests the
    form_aux_matrices work buffers, not earlier integral/QR memory requirements.
    """
    primary, auxiliary, matrices, vectors = fdds_input
    obj = psi4.core.FDDS_Dispersion(primary, auxiliary, matrices, vectors, True, is_lrc, 0.3)
    reference = {monomer: _aux_matrices(obj, monomer, 0.3, is_lrc) for monomer in ("A", "B")}
    nocc, nvir = matrices["Cocc_A"].np.shape[1], matrices["Cvir_A"].np.shape[1]
    naux = auxiliary.nbf()
    assert nocc > 2 and nocc % 2  # two-occupied blocking also has a partial last block
    fixed = 2 * nocc * nvir + 6 * naux**2
    # Three slices (work, arQL, arQLD) for both global hybrid and LRC.
    per_occ = 3 * nvir * naux

    # 80% of the global byte setting / sizeof(double) = byte setting / 10.
    too_small = fixed + 2 * nvir * naux
    psi4.core.set_memory_bytes(10 * too_small, True)
    with pytest.raises(RuntimeError, match=r"Too little memory for FDDS_Dispersion::form_aux_matrices"):
        obj.form_aux_matrices("A", 0.3, 0.25, 0.75 if is_lrc else 0.0)

    for block in (1, 2):
        psi4.core.set_memory_bytes(10 * (fixed + block * per_occ + 1), True)
        for monomer in ("A", "B"):
            actual = _aux_matrices(obj, monomer, 0.3, is_lrc)
            for key, value in reference[monomer].items():
                np.testing.assert_allclose(actual[key], value, rtol=1.e-10, atol=1.e-11,
                                           err_msg=f"{monomer}, block={block}, {key}")


def _mat_power(mat, exp, cutoff=1.0e-12):
    evals, evecs = np.linalg.eigh(mat)
    keep = evals > cutoff * evals[-1]
    return (evecs[:, keep] * (evals[keep] ** exp)) @ evecs[:, keep].T


def _dense_local_fxc(rho_aux, s_aux, x_alpha, x_beta, omega):
    sh = _mat_power(s_aux, 0.5)
    si = _mat_power(s_aux, -0.5)
    rho, u = np.linalg.eigh(si @ rho_aux @ si)
    cut = psi4.core.get_option("SAPT", "SAPT_FDDS_V2_RHO_CUTOFF")
    n = len(rho)
    inp = {"RHO_A": psi4.core.Vector.from_array(np.maximum(rho, cut))}

    def _deriv(name, w=None):
        func = psi4.core.LibXCFunctional(name, True)
        if w is not None:
            func.set_omega(w)
        out = {k: psi4.core.Vector(n) for k in ("V", "V_RHO_A", "V_RHO_A_RHO_A")}
        func.compute_functional(inp, out, n, 2)
        return out["V_RHO_A_RHO_A"].np.copy()

    val = (1.0 - x_alpha - x_beta) * _deriv("XC_LDA_X") + _deriv("XC_LDA_C_VWN")
    if x_beta != 0.0:
        val = val + x_beta * _deriv("XC_LDA_X_ERF", omega)
    return sh @ (u * np.where(rho >= cut, val, 0.0)) @ u.T @ sh


def _dense_monomer_tddft_setup(wfn, aux, x_alpha, x_beta, omega):
    """Build exact dense singlet TDDFT operators (D, G, P, N, B, M) matching FDDS."""
    prim = wfn.basisset()
    mints = psi4.core.MintsHelper(prim)
    zero = psi4.core.BasisSet.zero_ao_basis_set()
    naux, nbf = aux.nbf(), prim.nbf()
    co = wfn.Ca_subset("AO", "OCC").np.copy()
    cv = wfn.Ca_subset("AO", "VIR").np.copy()
    eo = wfn.epsilon_a_subset("AO", "OCC").np.copy()
    ev = wfn.epsilon_a_subset("AO", "VIR").np.copy()
    no, nv = co.shape[1], cv.shape[1]
    nov = no * nv

    def _df_exchange_blocks(lr):
        f_m = psi4.core.IntegralFactory(aux, zero, aux, zero)
        f_t = psi4.core.IntegralFactory(aux, zero, prim, prim)
        m = (mints.ao_erf_eri(omega, f_m) if lr else mints.ao_eri(f_m)).np.reshape(naux, naux).copy()
        t = (mints.ao_erf_eri(omega, f_t) if lr else mints.ao_eri(f_t)).np.reshape(naux, nbf, nbf).copy()
        eri = np.einsum("Ppq,PQ,Qrs->pqrs", t, _mat_power(m, -1.0), t, optimize=True)
        x = np.einsum("pqrs,pi,qb,rj,sa->iajb", eri, co, cv, co, cv, optimize=True).reshape(nov, nov)
        y = np.einsum("pqrs,pi,qj,ra,sb->iajb", eri, co, co, cv, cv, optimize=True).reshape(nov, nov)
        return x + y, y - x, m, t

    P0, N0, metric, raw = _df_exchange_blocks(False)
    if x_beta != 0.0:
        Pw, Nw, _, _ = _df_exchange_blocks(True)
    else:
        Pw = np.zeros_like(P0)
        Nw = np.zeros_like(N0)

    B = np.einsum("Ppq,pi,qa->iaP", raw, co, cv, optimize=True).reshape(nov, naux)
    metric_inv = _mat_power(metric, -1.0)
    s_aux = mints.ao_overlap(aux, aux).np.copy()
    triple = mints.ao_3coverlap(aux, aux, aux).np.reshape(naux, naux, naux).copy()
    rho_aux = 2.0 * np.einsum("PQR,R->PQ", triple, metric_inv @ np.einsum("Ppq,pq->P", raw, wfn.Da().np))
    fxc = _dense_local_fxc(rho_aux, s_aux, x_alpha, x_beta, omega)
    G = B @ metric_inv @ (metric + fxc) @ metric_inv @ B.T
    D = np.diag((ev[None, :] - eo[:, None]).ravel())
    P = x_alpha * P0 + x_beta * Pw
    N = x_alpha * N0 + x_beta * Nw
    Hp = D + 4.0 * G - P
    Hm = D - N
    return {
        "Hp": Hp, "Hm": Hm, "B": B, "metric": metric, "metric_inv": metric_inv,
        "nov": nov, "naux": naux,
    }


def _dense_chi(setup, u):
    Hp, Hm, B, nov = setup["Hp"], setup["Hm"], setup["B"], setup["nov"]
    eye = np.eye(nov)
    h = np.block([[Hp, u * eye], [-u * eye, Hm]])
    rhs = np.vstack([B, np.zeros_like(B)])
    z = np.linalg.solve(h, rhs)[:nov]
    return -4.0 * (B.T @ z)


@pytest.mark.parametrize(
    "functional,lr_on",
    [
        ("pbe0", False),
        ("wb97x", True),
        ("wb97x", False),
        ("lc-wpbe", True),
        ("hf", False),
    ],
)
def test_fdds_dense_tddft_susceptibility_parity(functional, lr_on):
    """In a full-rank auxiliary basis, raw and symmetrized FDDS chi(iu) must
    match exact dense singlet TDDFT to machine precision across frequencies,
    and static HF (u=0) must match -wfn.cphf_Hx.
    """
    from unittest.mock import patch
    from psi4.driver.procrouting.sapt import sapt_mp2_terms as prod

    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()

    # Distorted water in STO-3G with cc-pVDZ auxiliary basis (Nov=10 < Naux=24: full rank)
    mol = psi4.geometry("""
    0 1
    O 0.0 0.0 0.0
    H 0.0 0.78 0.56
    H 0.12 -0.71 0.62
    symmetry c1
    no_reorient
    no_com
    """)
    psi4.set_options({
        "basis": "sto-3g",
        "scf_type": "pk",
        "e_convergence": 1e-11,
        "d_convergence": 1e-10,
    })
    _, wfn = psi4.energy(functional, molecule=mol, return_wfn=True)
    prim = wfn.basisset()
    aux = psi4.core.BasisSet.build(mol, "BASIS", "cc-pvdz")

    func = wfn.functional()
    x_alpha = func.x_alpha()
    x_beta = func.x_beta() if (func.is_x_lrc() and lr_on) else 0.0
    omega = func.x_omega() if (func.is_x_lrc() and lr_on) else 0.0
    is_lrc = func.is_x_lrc() and lr_on
    is_hybrid = (x_alpha != 0.0) or is_lrc

    co, cv = wfn.Ca_subset("AO", "OCC"), wfn.Ca_subset("AO", "VIR")
    eo, ev = wfn.epsilon_a_subset("AO", "OCC"), wfn.epsilon_a_subset("AO", "VIR")
    cache = {
        f"{k}_{m}": v
        for m in ("A", "B")
        for k, v in (("Cocc", co), ("Cvir", cv), ("eps_occ", eo), ("eps_vir", ev), ("D", wfn.Da()))
    }

    setup = _dense_monomer_tddft_setup(wfn, aux, x_alpha, x_beta, omega)
    assert np.linalg.matrix_rank(setup["B"]) == setup["nov"]

    freqs = np.array([0.0, 0.1, 0.3, 1.0])
    captured = []
    real_sym = prod._symmetrize

    def _tap(mat):
        captured.append(mat.copy())
        return real_sym(mat)

    with patch.object(prod, "_symmetrize", _tap), patch.object(
        np.polynomial.legendre, "leggauss", return_value=((0.3 - freqs) / (0.3 + freqs), np.ones(len(freqs)))
    ):
        prod.df_fdds_dispersion(prim, aux, cache, is_hybrid, x_alpha, x_beta, is_lrc, omega, do_print=False)

    assert len(captured) == 4 * len(freqs)
    for ix, u in enumerate(freqs):
        ref = _dense_chi(setup, u)
        raw_coupled = captured[4 * ix + 2]
        sym_coupled = real_sym(raw_coupled)
        rel_raw = np.linalg.norm(raw_coupled - ref) / np.linalg.norm(ref)
        rel_sym = np.linalg.norm(sym_coupled - ref) / np.linalg.norm(ref)
        assert rel_raw < 1.0e-11, f"{functional} (lr_on={lr_on}) u={u}: raw rel_err={rel_raw:.3e}"
        assert rel_sym < 1.0e-11, f"{functional} (lr_on={lr_on}) u={u}: sym rel_err={rel_sym:.3e}"

    if functional == "hf":
        # Independent static u=0 check directly against Psi4's CPHF Hessian action (-wfn.cphf_Hx).
        # Build a DF JK matching the cc-pvdz auxiliary basis and zero out the ALDA correlation kernel
        # so FDDS evaluates pure TDHF (H_+ = D + 4J - K - K^T = -cphf_Hx) with the exact same DF integrals.
        jk = psi4.core.JK.build(prim, aux=aux, jk_type="MEM_DF", do_wK=False, memory=int(1e8))
        jk.set_do_J(True)
        jk.set_do_K(True)
        jk.initialize()
        wfn.set_jk(jk)
        no, nv = co.np.shape[1], cv.np.shape[1]
        nov = no * nv
        H_cphf = np.zeros((nov, nov))
        for col in range(nov):
            e_vec = np.zeros((no, nv))
            e_vec.ravel()[col] = 1.0
            hx = wfn.cphf_Hx([psi4.core.Matrix.from_array(e_vec)])[0].np
            H_cphf[:, col] = -hx.ravel()
        B = setup["B"]
        chi_cphf = -4.0 * (B.T @ np.linalg.solve(H_cphf, B))

        captured_hf = []
        zero_fxc = psi4.core.Matrix(aux.nbf(), aux.nbf())

        def _tap_hf(mat):
            captured_hf.append(mat.copy())
            return real_sym(mat)

        with patch.object(prod, "_symmetrize", _tap_hf), patch.object(
            prod, "_compute_fxc", return_value=zero_fxc
        ), patch.object(np.polynomial.legendre, "leggauss", return_value=(np.array([1.0]), np.array([1.0]))):
            prod.df_fdds_dispersion(prim, aux, cache, True, 1.0, 0.0, False, 0.0, do_print=False)

        raw_u0 = captured_hf[2]
        rel_cphf = np.linalg.norm(raw_u0 - chi_cphf) / np.linalg.norm(chi_cphf)
        assert rel_cphf < 1.0e-11, f"HF static FDDS vs -wfn.cphf_Hx rel_err={rel_cphf:.3e}"
        jk.finalize()


@pytest.mark.parametrize("functional", ["pbe0", "wb97x", "lc-wpbe", "hf"])
def test_fdds_dimer_dispersion_full_and_rank_deficient_parity(functional):
    """Check coupled FDDS dimer E^(2)_disp against matched dense TDDFT on both
    a full-rank dimer (He2/aug-cc-pVDZ: exact parity) and a rank-deficient dimer
    ((H2O)2/cc-pVDZ: ~3.5x-4.1x error reduction over the buggy LDN/NLP ordering).
    """
    from psi4.driver.procrouting.sapt import sapt_mp2_terms as prod

    def _run_dimer(mol_str, basis, ri_basis, max_rel_err):
        psi4.core.clean()
        psi4.core.clean_options()
        psi4.core.clean_variables()
        dimer = psi4.geometry(mol_str)
        dimer, monA, monB = psi4.driver.proc_util.prepare_sapt_molecule(dimer, "dimer")
        psi4.set_options({
            "basis": basis,
            "scf_type": "df",
            "e_convergence": 1e-10,
            "d_convergence": 1e-9,
        })
        _, wfnA = psi4.energy(functional, molecule=monA, return_wfn=True)
        _, wfnB = psi4.energy(functional, molecule=monB, return_wfn=True)
        prim = wfnA.basisset()
        aux = psi4.core.BasisSet.build(dimer, "DF_BASIS_MP2", ri_basis, "RIFIT", basis)

        func = wfnB.functional()
        x_alpha = func.x_alpha()
        is_lrc = func.is_x_lrc()
        x_beta = func.x_beta() if is_lrc else 0.0
        omega = func.x_omega() if is_lrc else 0.0
        is_hybrid = (x_alpha != 0.0) or is_lrc

        cache = {
            "Cocc_A": wfnA.Ca_subset("AO", "OCC"),
            "Cvir_A": wfnA.Ca_subset("AO", "VIR"),
            "eps_occ_A": wfnA.epsilon_a_subset("AO", "OCC"),
            "eps_vir_A": wfnA.epsilon_a_subset("AO", "VIR"),
            "D_A": wfnA.Da(),
            "Cocc_B": wfnB.Ca_subset("AO", "OCC"),
            "Cvir_B": wfnB.Ca_subset("AO", "VIR"),
            "eps_occ_B": wfnB.epsilon_a_subset("AO", "OCC"),
            "eps_vir_B": wfnB.epsilon_a_subset("AO", "VIR"),
            "D_B": wfnB.Da(),
        }
        res = prod.df_fdds_dispersion(
            prim, aux, cache, is_hybrid, x_alpha, x_beta, is_lrc, omega, leg_points=10, leg_lambda=0.3, do_print=False
        )
        e_fdds = res["Disp20"]

        setup_A = _dense_monomer_tddft_setup(wfnA, aux, x_alpha, x_beta, omega)
        setup_B = _dense_monomer_tddft_setup(wfnB, aux, x_alpha, x_beta, omega)
        minv = setup_A["metric_inv"]
        total_dense = 0.0
        for pt, wt in zip(*np.polynomial.legendre.leggauss(10)):
            u = 0.3 * (1.0 - pt) / (1.0 + pt)
            jac = (2.0 * 0.3) / ((pt + 1.0) ** 2)
            chiA = _dense_chi(setup_A, u)
            chiB = _dense_chi(setup_B, u)
            val = np.sum((minv @ chiA @ minv) * chiB)
            total_dense += val * wt * jac
        e_dense = -total_dense / (2.0 * np.pi)

        rel_err = abs(e_fdds - e_dense) / abs(e_dense)
        assert rel_err < max_rel_err, (
            f"{functional} ({basis}): e_fdds={e_fdds:.8f}, e_dense={e_dense:.8f}, rel_err={rel_err:.4%}"
        )

    # Full-rank He2 (Nov=17 < Naux=26): exact parity with dense TDDFT
    _run_dimer(
        """
        0 1
        He 0.0 0.0 0.0
        --
        0 1
        He 0.0 0.0 3.0
        units angstrom
        symmetry c1
        no_reorient
        no_com
        """,
        "aug-cc-pvdz",
        "aug-cc-pvdz-ri",
        1.0e-10,
    )

    # Rank-deficient (H2O)2 (Nov=215 > Naux=168): pure DF Q-projection residual only
    # (was 0.62% PBE0, 1.18% wB97X, 1.28% LC-wPBE, 2.22% HF before the fix;
    # drops 3.5x-4.1x to 0.17%, 0.33%, 0.34%, 0.54%).
    rank_def_bounds = {"pbe0": 2.5e-3, "wb97x": 4.0e-3, "lc-wpbe": 4.0e-3, "hf": 6.0e-3}
    _run_dimer(
        """
        0 1
        O -0.702196054 -0.056060256  0.009942262
        H -1.022193224  0.846775782 -0.011488714
        H  0.257521062  0.042121496  0.005218999
        --
        0 1
        O  2.268880784  0.026340101  0.000508029
        H  2.645502399 -0.412039965  0.766632411
        H  2.641145101 -0.449872874 -0.744894473
        units angstrom
        symmetry c1
        no_reorient
        no_com
        """,
        "cc-pvdz",
        "cc-pvdz-ri",
        rank_def_bounds[functional],
    )
