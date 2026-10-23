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
    psi4.set_options({"cuest_gridk": False, "cuest_directj": False, "use_cuest": False})


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
        if any(message in str(exc) for message in
               ("CUEST_GRIDK requires cuEST", "CUEST_DIRECTJ requires cuEST")):
            pytest.skip("Linked cuEST SDK predates the requested J/K API")
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
@pytest.mark.parametrize("directj", [False, True])
@pytest.mark.parametrize("operator", [
    (1.0, 0.0, 0.0), (0.25, 0.0, 0.0), (0.0, 1.0, 0.3),
    (0.2, 0.6, 0.3), (1.0, -1.0, 0.3),
])
def test_cuest_gridk_exact_exchange(gridk_system, symmetric, operator, directj):
    """Grid quadrature is compared with four-center ERIs, not another Grid-K call."""
    _, primary, aux = gridk_system
    pairs = _pairs(primary.nbf(), symmetric)
    psi4.set_options({"cuest_directj": directj})
    mints = psi4.core.MintsHelper(primary)
    eri_full = np.asarray(mints.ao_eri()).reshape((primary.nbf(),) * 4)
    eri = operator[0] * eri_full
    if operator[1]:
        eri += operator[1] * np.asarray(mints.ao_erf_eri(operator[2])).reshape(eri.shape)
    jk = _builder(primary, aux, operator)
    try:
        coulomb, exchange = _compute(jk, pairs)
        for (left, right), got_j, got in zip(pairs, coulomb, exchange):
            density = left @ (left if right is None else right).T
            if directj:
                np.testing.assert_allclose(got_j, np.einsum("pqrs,rs->pq", eri_full, density),
                                           atol=1.e-8, rtol=0)
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
@pytest.mark.parametrize("directj", [False, True])
def test_cuest_gridk_cpks(gridk_system, unrestricted, directj):
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
            nocc = sum(subset("SO", "OCC").coldim())
            nvir = sum(subset("SO", "VIR").coldim())
            rhs.append(psi4.core.Matrix.from_array(rng.normal(size=(nocc, nvir)) * 0.01))
    original = wfn.jk()
    expected = [m.np.copy() for m in wfn.cphf_Hx(rhs)]
    psi4.set_options({"use_cuest": True, "cuest_gridk": True, "cuest_directj": directj})
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


def test_cuest_directj_switch_and_no_df_plan(gridk_system, tmp_path):
    """Direct-J + Grid-K stores no DF tensor; a live J-mode switch rebuilds."""
    _, primary, aux = gridk_system
    trace = tmp_path / "directj-plan.out"
    psi4.core.set_output_file(str(trace), False)
    psi4.set_options({"cuest_directj": True})
    jk = _builder(primary, aux)
    pairs = _pairs(primary.nbf())
    try:
        j_direct, k_grid = _compute(jk, pairs)
        psi4.core.flush_outfile()
        text = trace.read_text()
        assert "Device memory, Direct-J plan:" in text
        assert "Device memory, DF int plan:" not in text
        jk.initialize()
        psi4.core.flush_outfile()
        assert len(re.findall(r"cuESTJK plan created #\d+", trace.read_text())) == 1
        psi4.set_options({"cuest_directj": False})
        j_df, k_again = _compute(jk, pairs)
        psi4.core.flush_outfile()
        assert "Device memory, DF int plan:" in trace.read_text()
        for ref, got in zip(k_grid, k_again):
            np.testing.assert_allclose(got, ref, atol=1.e-10, rtol=0)
        psi4.set_options({"cuest_directj": True})
        j_restored, _ = _compute(jk, pairs)
        for ref, got in zip(j_direct, j_restored):
            np.testing.assert_allclose(got, ref, atol=1.e-10, rtol=0)
        psi4.core.flush_outfile()
        assert len(re.findall(r"cuESTJK plan created #\d+", trace.read_text())) == 3
    finally:
        jk.finalize()


@pytest.mark.parametrize("gridk", [False, True])
@pytest.mark.parametrize("do_j,do_k", [(True, False), (False, True), (True, True)])
def test_cuest_directj_task_combinations(gridk_system, gridk, do_j, do_k):
    """Direct-J is independent of DF-K versus Grid-K and of the task mask."""
    _, primary, aux = gridk_system
    psi4.set_options({"cuest_directj": True, "cuest_gridk": gridk})
    jk = _builder(primary, aux, do_j=do_j, do_k=do_k)
    pairs = _pairs(primary.nbf())
    try:
        j, k = _compute(jk, pairs)
        eri = np.asarray(psi4.core.MintsHelper(primary).ao_eri()).reshape((primary.nbf(),) * 4)
        df_k = None
        if do_k and not gridk:
            psi4.set_options({"use_cuest": False})
            cpu = psi4.core.JK.build_JK(primary, aux)
            try:
                cpu.initialize()
                _, df_k = _compute(cpu, pairs)
            finally:
                cpu.finalize()
                psi4.set_options({"use_cuest": True})
        for n, (left, right) in enumerate(pairs):
            density = left @ right.T
            if do_j:
                np.testing.assert_allclose(j[n], np.einsum("pqrs,rs->pq", eri, density),
                                           atol=1.e-8, rtol=0)
            if do_k:
                expected_k = np.einsum("prqs,rs->pq", eri, density) if gridk else df_k[n]
                np.testing.assert_allclose(k[n], expected_k, atol=2.e-5 if gridk else 1.e-8, rtol=0)
        assert bool(j) == do_j
        assert bool(k) == do_k
    finally:
        jk.finalize()


def test_cuest_directj_antisymmetric_density(gridk_system):
    """J discards antisymmetric density, but nonsymmetric Grid-K must retain it."""
    _, primary, aux = gridk_system
    psi4.set_options({"cuest_directj": True})
    left, right = _pairs(primary.nbf())[1]
    anti_left = np.column_stack((left, right))
    anti_right = np.column_stack((right, -left))
    pairs = [(left, right), (right, left), (anti_left, anti_right)]
    jk = _builder(primary, aux)
    try:
        j, k = _compute(jk, pairs)
        np.testing.assert_allclose(j[0], j[1], atol=1.e-10, rtol=0)
        np.testing.assert_allclose(j[2], 0, atol=1.e-10, rtol=0)
        np.testing.assert_allclose(k[2], k[0] - k[1], atol=1.e-10, rtol=0)
        assert np.max(np.abs(k[2])) > 1.e-3
        jk.set_cutoff(0.0)
        j_unscreened, k_unscreened = _compute(jk, pairs)
        for ref, got in zip(j + k, j_unscreened + k_unscreened):
            np.testing.assert_allclose(got, ref, atol=1.e-9, rtol=0)
    finally:
        jk.finalize()


def test_cuest_directj_gradient_guard(gridk_system):
    mol, primary, aux = gridk_system
    psi4.set_options({"cuest_directj": True, "guess": "core", "reference": "rhf"})
    probe = _builder(primary, aux)
    probe.finalize()
    with pytest.raises(RuntimeError, match="Analytic Coulomb gradients with CUEST_DIRECTJ"):
        psi4.gradient("hf", molecule=mol)


@pytest.mark.parametrize("zero_aux", [False, True])
def test_cuest_directj_gridk_unused_auxiliary(gridk_system, zero_aux):
    """An unused Cartesian auxiliary must never enter a hidden DF plan."""
    mol, primary, _ = gridk_system
    aux = (psi4.core.BasisSet.zero_ao_basis_set() if zero_aux else
           psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", "cc-pvdz-jkfit",
                                   "JKFIT", "cc-pvdz", puream=0))
    if not zero_aux:
        assert not aux.has_puream()
    psi4.set_options({"cuest_directj": True})
    jk = _builder(primary, aux)
    pairs = _pairs(primary.nbf())
    try:
        reference = _compute(jk, pairs)
        psi4.set_options({"cuest_directj": False})
        with pytest.raises(RuntimeError, match="spherical-harmonic auxiliary|same molecular centers"):
            jk.initialize()
        with pytest.raises(RuntimeError, match="not initialized"):
            _compute(jk, pairs)
        psi4.set_options({"cuest_directj": True})
        jk.initialize()
        got = _compute(jk, pairs)
        for ref, result in zip(reference[0] + reference[1], got[0] + got[1]):
            np.testing.assert_allclose(result, ref, atol=1.e-10, rtol=0)
    finally:
        jk.finalize()
