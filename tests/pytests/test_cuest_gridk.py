"""DF-J + Grid-K: exact-integral checks, plan lifecycle, and CPKS response."""

import re

import numpy as np
import pytest
import psi4

from addons import using

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick, pytest.mark.cuest,
              *using("cuest"), *using("cuda_cc8")]


@pytest.fixture
def gridk_system():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.set_num_threads(2)
    mol = psi4.geometry("""
        O 0 0 0
        H 0 1.4 1.0
        H 0 -1.4 1.0
        units bohr
        symmetry c1
        no_reorient
        no_com
    """)
    psi4.set_options({
        "basis": "cc-pvdz", "df_basis_scf": "cc-pvdz-jkfit",
        "scf_type": "df", "use_cuest": True, "cuest_gridk": True,
        "cuest_xc": False, "cuest_mixed_precision": False,
        "dft_radial_points": 99, "dft_spherical_points": 590,
        "dft_pruning_scheme": "none",
    })
    primary = psi4.core.BasisSet.build(mol, "ORBITAL", "cc-pvdz")
    aux = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", "", "JKFIT", "cc-pvdz")
    yield mol, primary, aux
    psi4.set_options({"cuest_gridk": False, "use_cuest": False})


def _builder(primary, aux, operator=(1.0, 0.0, 0.0), do_j=True, do_k=True):
    jk = psi4.core.JK.build_JK(primary, aux)
    assert jk.name() == "cuESTJK"
    jk.set_print(3)
    jk.set_do_J(do_j)
    jk.set_do_K(do_k)
    jk.set_do_wK(operator[1] != 0)
    jk.set_omega_alpha(operator[0])
    jk.set_omega_beta(operator[1])
    jk.set_omega(operator[2])
    try:
        jk.initialize()
    except RuntimeError as exc:
        jk.finalize()
        if "CUEST_GRIDK requires cuEST" in str(exc):
            pytest.skip("Linked cuEST SDK predates Grid-K")
        raise
    return jk


def _compute(jk, pairs):
    jk.C_clear()
    for left, right in pairs:
        jk.C_left_add(psi4.core.Matrix.from_array(left))
        if right is not None:
            jk.C_right_add(psi4.core.Matrix.from_array(right))
    jk.compute()
    return ([m.np.copy() for m in jk.J()], [m.np.copy() for m in jk.K()])


def _pairs(nbf, symmetric=False):
    rng = np.random.default_rng(50250)
    return [(rng.normal(size=(nbf, rank)) / np.sqrt(nbf),
             None if symmetric else -rng.normal(size=(nbf, rank)) / np.sqrt(nbf))
            for rank in (1, 4, 0, 2)]


@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("operator", [
    (1.0, 0.0, 0.0), (0.25, 0.0, 0.0), (0.0, 1.0, 0.3),
    (0.2, 0.6, 0.3), (1.0, -1.0, 0.3),
])
def test_cuest_gridk_exact_exchange(gridk_system, symmetric, operator):
    """Grid quadrature is compared with four-center ERIs, not another Grid-K call."""
    _, primary, aux = gridk_system
    pairs = _pairs(primary.nbf(), symmetric)
    mints = psi4.core.MintsHelper(primary)
    eri = operator[0] * np.asarray(mints.ao_eri()).reshape((primary.nbf(),) * 4)
    if operator[1]:
        eri += operator[1] * np.asarray(mints.ao_erf_eri(operator[2])).reshape(eri.shape)
    jk = _builder(primary, aux, operator)
    try:
        _, exchange = _compute(jk, pairs)
        for (left, right), got in zip(pairs, exchange):
            density = left @ (left if right is None else right).T
            expected = np.einsum("prqs,rs->pq", eri, density)
            # Unlike DF-K, Grid-K has a quadrature error. This dense grid is
            # deliberately independent of the SCF/XC grid backend.
            np.testing.assert_allclose(got, expected, atol=2.e-5, rtol=0)
            if symmetric:
                np.testing.assert_allclose(got, got.T, atol=1.e-12, rtol=0)
        np.testing.assert_array_equal(exchange[2], 0)
        if not symmetric:
            assert np.max(np.abs(exchange[1] - exchange[1].T)) > 1.e-3
    finally:
        jk.finalize()


def test_cuest_gridk_plan_reuse_and_switch(gridk_system, tmp_path):
    """Grid/operator/backend changes rebuild; unchanged initialize does not."""
    _, primary, aux = gridk_system
    trace = tmp_path / "gridk-plan.out"
    psi4.core.set_output_file(str(trace), False)
    pairs = _pairs(primary.nbf())

    def creations():
        psi4.core.flush_outfile()
        return len(re.findall(r"cuESTJK plan created #\d+", trace.read_text()))

    jk = _builder(primary, aux)
    try:
        j_ref, k_ref = _compute(jk, pairs)
        assert creations() == 1
        jk.initialize()
        _compute(jk, pairs)
        assert creations() == 1
        jk.set_omega_alpha(0.25)
        _, weighted = _compute(jk, pairs)
        assert creations() == 2
        for ref, got in zip(k_ref, weighted):
            np.testing.assert_allclose(got, 0.25 * ref, atol=1.e-10, rtol=0)
        jk.set_omega_alpha(1.0)
        assert creations() == 3
        psi4.set_options({"dft_radial_points": 100})
        _compute(jk, pairs)  # Detect a live grid-option edit without initialize().
        assert creations() == 4
        psi4.set_options({"cuest_gridk": False})
        j_df, k_df = _compute(jk, pairs)
        assert creations() == 5
        for ref, got in zip(j_ref, j_df):
            np.testing.assert_allclose(got, ref, atol=1.e-10, rtol=0)
        psi4.set_options({"use_cuest": False})
        cpu = psi4.core.JK.build_JK(primary, aux)
        try:
            cpu.initialize()
            j_cpu, k_cpu = _compute(cpu, pairs)
            for ref, got in zip(j_cpu + k_cpu, j_df + k_df):
                np.testing.assert_allclose(got, ref, atol=1.e-8, rtol=0)
        finally:
            cpu.finalize()
        psi4.set_options({"use_cuest": True, "cuest_gridk": True})
        jk.initialize()
        assert creations() == 6
        jk.finalize()
        jk.finalize()
        with pytest.raises(RuntimeError, match="not initialized"):
            _compute(jk, pairs)
    finally:
        jk.finalize()


@pytest.mark.parametrize("do_j,do_k", [(True, False), (False, True)])
def test_cuest_gridk_single_task(gridk_system, do_j, do_k):
    _, primary, aux = gridk_system
    jk = _builder(primary, aux, do_j=do_j, do_k=do_k)
    try:
        j, k = _compute(jk, _pairs(primary.nbf()))
        assert bool(j) == do_j
        assert bool(k) == do_k
        assert all(np.isfinite(m).all() for m in j + k)
    finally:
        jk.finalize()


def test_cuest_gridk_failed_grid_rebuild(gridk_system):
    _, primary, aux = gridk_system
    jk = _builder(primary, aux)
    pairs = _pairs(primary.nbf())
    try:
        reference = _compute(jk, pairs)
        psi4.set_options({"dft_spherical_points": 123})
        with pytest.raises(RuntimeError, match="Invalid number of spherical points"):
            jk.initialize()
        with pytest.raises(RuntimeError, match="not initialized"):
            _compute(jk, pairs)
        psi4.set_options({"dft_spherical_points": 590})
        jk.initialize()
        got = _compute(jk, pairs)
        for ref, result in zip(reference[0] + reference[1], got[0] + got[1]):
            np.testing.assert_allclose(result, ref, atol=1.e-10, rtol=0)
    finally:
        jk.finalize()


def test_cuest_gridk_gradient_guard(gridk_system):
    mol, primary, aux = gridk_system
    # Establish SDK availability before asking the SCF driver for a gradient.
    probe = _builder(primary, aux)
    probe.finalize()
    psi4.set_options({"guess": "core", "reference": "rhf"})
    with pytest.raises(RuntimeError, match="Analytic exchange gradients with CUEST_GRIDK"):
        psi4.gradient("hf", molecule=mol)


@pytest.mark.parametrize("unrestricted", [False, True])
def test_cuest_gridk_cpks(gridk_system, unrestricted):
    """Fixed CPU orbitals/XC isolate Grid-K and check the actual solver residual."""
    mol, _, _ = gridk_system
    if unrestricted:
        mol.set_molecular_charge(1)
        mol.set_multiplicity(2)
    psi4.set_options({
        "use_cuest": False, "cuest_gridk": False, "save_jk": True,
        "reference": "uks" if unrestricted else "rks",
        "e_convergence": 1.e-11, "d_convergence": 1.e-10,
    })
    wfn = psi4.energy("pbe0", molecule=mol, return_wfn=True)[1]
    rng = np.random.default_rng(250)
    rhs = []
    for _ in range(2):
        for spin in (("a", "b") if unrestricted else ("a",)):
            subset = getattr(wfn, f"C{spin}_subset")
            nocc = subset("SO", "OCC").ncol()
            nvir = subset("SO", "VIR").ncol()
            rhs.append(psi4.core.Matrix.from_array(rng.normal(size=(nocc, nvir)) * 0.01))
    original = wfn.jk()
    expected = [m.np.copy() for m in wfn.cphf_Hx(rhs)]
    psi4.set_options({"use_cuest": True, "cuest_gridk": True})
    jk = _builder(wfn.basisset(), wfn.get_basisset("DF_BASIS_SCF"), (0.25, 0.0, 0.0))
    try:
        wfn.set_jk(jk)
        for ref, got in zip(expected, wfn.cphf_Hx(rhs)):
            np.testing.assert_allclose(got.np, ref, atol=2.e-5, rtol=0)
        solved = wfn.cphf_solve(rhs, 1.e-9, 100, 0)
        assert wfn.cphf_converged()
        for ref, got in zip(rhs, wfn.cphf_Hx(solved)):
            np.testing.assert_allclose(got.np, ref.np, atol=1.e-9, rtol=0)
        # SAPT can borrow the same builder for bare exchange before CPKS.
        jk.set_omega_alpha(1.0)
        for ref, got in zip(expected, wfn.cphf_Hx(rhs)):
            np.testing.assert_allclose(got.np, ref, atol=2.e-5, rtol=0)
        assert jk.get_omega_alpha() == 1.0
    finally:
        wfn.set_jk(original)
        jk.finalize()
