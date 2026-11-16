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


def _aux_matrices(obj, monomer, frequency, is_lrc, auxiliary_basis=False):
    result = {k: v.np.copy() for k, v in obj.form_aux_matrices(
        monomer, frequency, 0.25, 0.75 if is_lrc else 0.0).items()}
    if auxiliary_basis:
        # These three matrices carry one right-hand QR-coordinate index.
        # QR signs (and the null-space completion) can change with roundoff
        # across integral algorithms. Multiplying by R maps Q back to the
        # common auxiliary basis, since (ar|P) = Q R, without a pseudoinverse.
        R = (obj.R_A() if monomer == "A" else obj.R_B()).np
        for key in ("K1LD", "K2LD", "K21L"):
            result[key] = result[key] @ R
    return result


@pytest.mark.parametrize("omega", [0.0, 0.1, 0.4])
@pytest.mark.parametrize("algorithm", ["STORE", "LEGACY", "DIRECT_IAQ"])
@pytest.mark.parametrize("split_monomers", [False, True])
@pytest.mark.parametrize("asymmetric", [False, True])
def test_fdds_storage_mode_parity(fdds_input, omega, algorithm, split_monomers, asymmetric):
    """Finite-range exchange must not become Coulomb exchange out of core.

    Comparing every auxiliary response matrix catches the wrong three-index
    operator even when an integrated energy could mask it. omega=0 exercises
    the unchanged global-hybrid path; the finite-omega cases include mixed
    full-range/LR exchange products.
    """
    primary, auxiliary, matrices, vectors = fdds_input
    if asymmetric:
        # Move one B occupied orbital into its virtual space. This is a
        # frozen-orbital algebra probe, not a self-consistent ion calculation.
        # Different monomer shapes also exercise wide versus tall QR factors.
        matrices["Cocc_B"] = psi4.core.Matrix.from_array(matrices["Cocc_A"].np[:, :-1])
        matrices["Cvir_B"] = psi4.core.Matrix.from_array(np.column_stack(
            (matrices["Cocc_A"].np[:, -1:], matrices["Cvir_A"].np)))
        vectors["eps_occ_B"] = psi4.core.Vector.from_array(vectors["eps_occ_A"].np[:-1])
        vectors["eps_vir_B"] = psi4.core.Vector.from_array(np.concatenate(
            (vectors["eps_occ_A"].np[-1:], vectors["eps_vir_A"].np)))
    is_lrc = omega != 0.0
    # A common reference also checks the new STORE/split path against the
    # published in-core DIRECT/DIRECT_iaQ implementation.
    psi4.set_options({
        "scf_subtype": "INCORE",
        "sapt_fdds_disp_df_algorithm": "LEGACY",
        "sapt_fdds_disp_split_monomers": False,
    })
    obj = psi4.core.FDDS_Dispersion(primary, auxiliary, matrices, vectors, True, is_lrc, omega)
    reference = {
        (monomer, frequency): _aux_matrices(obj, monomer, frequency, is_lrc, auxiliary_basis=True)
        for monomer in ("A", "B") for frequency in (0.0, 0.3)
    }
    del obj
    psi4.set_options({
        "sapt_fdds_disp_df_algorithm": algorithm,
        "sapt_fdds_disp_split_monomers": split_monomers,
    })
    results = {}
    for mode in ("INCORE", "OUT_OF_CORE"):
        psi4.set_options({"scf_subtype": mode})
        obj = psi4.core.FDDS_Dispersion(primary, auxiliary, matrices, vectors, True, is_lrc, omega)
        results[mode] = {
            (monomer, frequency): _aux_matrices(obj, monomer, frequency, is_lrc, auxiliary_basis=True)
            for monomer in ("A", "B") for frequency in (0.0, 0.3)
        }
        del obj
    for mode, cases in results.items():
        for case, expected in reference.items():
            for key, value in expected.items():
                np.testing.assert_allclose(
                    cases[case][key], value, rtol=1.e-9, atol=1.e-10,
                    err_msg=f"{algorithm}, split={split_monomers}, {mode}, omega={omega}, {case}, {key}",
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
    per_occ = (12 if is_lrc else 7) * nvir * naux

    # 80% of the global byte setting / sizeof(double) = byte setting / 10.
    # Nine slices fit the old (incorrect) seven-slice LRC budget, not twelve.
    too_small = fixed + (9 if is_lrc else 6) * nvir * naux
    resident_bytes = 8 * psi4.core.memory_committed()
    psi4.core.set_memory_bytes(resident_bytes + 10 * too_small, True)
    with pytest.raises(RuntimeError, match=r"Too little memory for FDDS_Dispersion::form_aux_matrices"):
        obj.form_aux_matrices("A", 0.3, 0.25, 0.75 if is_lrc else 0.0)

    for block in (1, 2):
        psi4.core.set_memory_bytes(resident_bytes + 10 * (fixed + block * per_occ + 1), True)
        for monomer in ("A", "B"):
            actual = _aux_matrices(obj, monomer, 0.3, is_lrc)
            for key, value in reference[monomer].items():
                np.testing.assert_allclose(actual[key], value, rtol=1.e-10, atol=1.e-11,
                                           err_msg=f"{monomer}, block={block}, {key}")
