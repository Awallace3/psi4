import numpy as np
import psi4
import pytest

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]


@pytest.mark.parametrize("symmetric", [True, False])
def test_dfhelper_incore_jk_small_blocks(symmetric):
    """Small auxiliary blocks preserve J/K and do not change in-core storage."""
    molecule = psi4.geometry("""
        0 1
        O
        H 1 1.0
        H 1 1.0 2 103.1
        symmetry c1
    """)
    primary = psi4.core.BasisSet.build(molecule, "ORBITAL", "cc-pvdz")
    auxiliary = psi4.core.BasisSet.build(molecule, "ORBITAL", "cc-pvdz-jkfit")
    psi4.set_options({"scf_type": "mem_df", "scf_subtype": "incore"})
    jk = psi4.core.JK.build_JK(primary, auxiliary)
    jk.set_omp_nthread(1)
    jk.set_memory(8_000_000)
    jk.initialize()
    try:
        nbf, nocc = primary.nbf(), 5
        rng = np.random.default_rng(43)
        jk.C_left_add(psi4.core.Matrix.from_array(rng.random((nbf, nocc))))
        if not symmetric:
            jk.C_right_add(psi4.core.Matrix.from_array(rng.random((nbf, nocc))))
        jk.compute()
        expected = [np.asarray(jk.J()[0]).copy(), np.asarray(jk.K()[0]).copy()]

        helper = jk.dfh()
        assert helper.get_AO_core()
        ao_size = helper.get_AO_size()
        block_aux = 2 * auxiliary.max_function_per_shell()
        assert block_aux < auxiliary.nbf() // 2  # Force several auxiliary blocks.
        c_buffer = nbf * nbf
        t1 = nbf * nocc * block_aux
        t2 = nbf * nbf if symmetric else t1
        # Isolate iteration blocking from the larger integral-initialization budget.
        helper.set_memory(ao_size + c_buffer + t1 + t2)
        jk.compute()
        assert helper.get_AO_core()
        for reference, actual in zip(expected, (jk.J()[0], jk.K()[0])):
            np.testing.assert_allclose(np.asarray(actual), reference, rtol=1e-10, atol=1e-10)

        helper.set_memory(ao_size)
        with pytest.raises(RuntimeError, match="not enough memory for JK blocking"):
            jk.compute()
    finally:
        jk.finalize()
