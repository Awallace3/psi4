import numpy as np
import psi4
import pytest

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]


@pytest.mark.parametrize("subtype", ["INCORE", "OUT_OF_CORE"])
@pytest.mark.parametrize("range_separated", [False, True])
def test_diskdfjk_reinitialize(subtype, range_separated):
    """Retiring integral engines must preserve repeated J/K/wK contractions."""
    psi4.set_num_threads(2)
    molecule = psi4.geometry("""
        0 1
        O
        H 1 1.0
        H 1 1.0 2 103.1
        symmetry c1
    """)
    primary = psi4.core.BasisSet.build(molecule, "ORBITAL", "cc-pvdz")
    auxiliary = psi4.core.BasisSet.build(molecule, "ORBITAL", "cc-pvdz-jkfit")
    rng = np.random.default_rng(43)
    coefficients = psi4.core.Matrix.from_array(rng.random((primary.nbf(), 5)))

    def build(kind, sub):
        psi4.set_options({"scf_type": kind, "scf_subtype": sub})
        jk = psi4.core.JK.build_JK(primary, auxiliary)
        jk.set_memory(8_000_000)
        jk.set_omp_nthread(2)
        jk.set_do_wK(range_separated)
        jk.set_omega(0.3)
        return jk

    def contract(jk):
        jk.C_clear()
        jk.C_left_add(coefficients)
        jk.compute()
        matrices = [jk.J()[0], jk.K()[0]]
        if range_separated:
            matrices.append(jk.wK()[0])
        return [np.asarray(matrix).copy() for matrix in matrices]

    reference = build("MEM_DF", "INCORE")
    reference.initialize()
    try:
        expected = contract(reference)
    finally:
        reference.finalize()

    jk = build("DISK_DF", subtype)
    for _ in range(2):
        jk.initialize()
        try:
            for _ in range(2):
                for actual, wanted in zip(contract(jk), expected):
                    np.testing.assert_allclose(actual, wanted, rtol=1e-9, atol=1e-9)
        finally:
            jk.finalize()
