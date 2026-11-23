"""cuEST Direct-J (four-center Coulomb) and Grid-K (numerical-grid exchange) in cuESTJK.

Every J/K check is against exact four-center AO integrals from MintsHelper, never
against another cuEST call, and covers symmetric SCF-like densities as well as
signed, nonsymmetric densities of the kind SAPT and CPHF/CPKS hand to JK.
"""

import numpy as np
import pytest
import psi4

from addons import using

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick, *using("cuest"), *using("cuda_cc8")]

WATER = """
O 0 0 0
H 0 1.4 1.0
H 0 -1.4 1.0
units bohr
symmetry c1
no_reorient
no_com
"""

# A dense, unpruned grid so the Grid-K quadrature error sits well under the tolerances.
GRID = {"dft_radial_points": 99, "dft_spherical_points": 590, "dft_pruning_scheme": "none"}

UNSUPPORTED = ("CUEST_K_BUILD GRID requires cuEST", "CUEST_J_BUILD DIRECT requires cuEST")


@pytest.fixture
def water():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.set_num_threads(2)
    mol = psi4.geometry(WATER)
    psi4.set_options({
        "basis": "cc-pvdz", "df_basis_scf": "cc-pvdz-jkfit", "scf_type": "df",
        "use_cuest": True, "cuest_mixed_precision": False, **GRID,
    })
    primary = psi4.core.BasisSet.build(mol, "ORBITAL", "cc-pvdz")
    aux = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", "", "JKFIT", "cc-pvdz")
    yield mol, primary, aux
    psi4.core.clean_options()


def _builder(primary, aux, operator=(1.0, 0.0, 0.0), do_j=True, do_k=True):
    """A cuESTJK for alpha / r + beta erf(omega r) / r; skips if the SDK lacks the API."""
    jk = psi4.core.JK.build_JK(primary, aux)
    assert jk.name() == "cuESTJK"
    jk.set_do_J(do_j)
    jk.set_do_K(do_k)
    jk.set_do_wK(do_k and operator[1] != 0)
    jk.set_omega_alpha(operator[0])
    jk.set_omega_beta(operator[1])
    jk.set_omega(operator[2])
    try:
        jk.initialize()
    except RuntimeError as exc:
        if any(message in str(exc) for message in UNSUPPORTED):
            pytest.skip("linked cuEST SDK does not provide the Direct-J/Grid-K API (needs cuEST >= 0.3)")
        raise
    return jk


def _pairs(nbf, symmetric):
    """(C_left, C_right) factor pairs of rank 1, 4, 0 and 2; C_right is None when symmetric."""
    rng = np.random.default_rng(50250)
    return [(rng.normal(size=(nbf, rank)) / np.sqrt(nbf),
             None if symmetric else -rng.normal(size=(nbf, rank)) / np.sqrt(nbf))
            for rank in (1, 4, 0, 2)]


def _compute(jk, pairs):
    jk.C_clear()
    for left, right in pairs:
        jk.C_left_add(psi4.core.Matrix.from_array(left))
        if right is not None:
            jk.C_right_add(psi4.core.Matrix.from_array(right))
    jk.compute()
    return [m.np.copy() for m in jk.J()], [m.np.copy() for m in jk.K()]


def _density(left, right):
    return left @ (left if right is None else right).T


@pytest.mark.parametrize("symmetric", [True, False], ids=["sym", "nonsym"])
def test_cuest_directj_matches_four_center_coulomb(water, symmetric):
    """Direct-J is the exact four-center J: no DF error, even for nonsymmetric D."""
    _, primary, aux = water
    psi4.set_options({"cuest_j_build": "direct", "ints_tolerance": 0.0})
    eri = np.asarray(psi4.core.MintsHelper(primary).ao_eri()).reshape((primary.nbf(),) * 4)
    pairs = _pairs(primary.nbf(), symmetric)
    jk = _builder(primary, aux, do_k=False)
    try:
        coulomb, _ = _compute(jk, pairs)
    finally:
        jk.finalize()
    max_err = 0.0
    for (left, right), got in zip(pairs, coulomb):
        expected = np.einsum("pqrs,rs->pq", eri, _density(left, right))
        max_err = max(max_err, np.max(np.abs(got - expected)))
        np.testing.assert_allclose(got, expected, atol=1.e-10, rtol=0)
    np.testing.assert_array_equal(coulomb[2], 0)
    print(f"Direct-J {'sym' if symmetric else 'nonsym'} max|J - J_4c| = {max_err:.2e}")


# Worst |K - K_4c| allowed per Grid-K variant.
GRIDK_ATOL = {"grid_fit": 2.e-5, "grid": 2.e-5}


@pytest.mark.parametrize("k_build", ["grid_fit", "grid"])
@pytest.mark.parametrize("symmetric", [True, False], ids=["sym", "nonsym"])
@pytest.mark.parametrize("directj", [False, True], ids=["dfj", "directj"])
@pytest.mark.parametrize("operator", [
    (1.0, 0.0, 0.0),    # HF
    (0.25, 0.0, 0.0),   # global hybrid (PBE0)
    (0.0, 1.0, 0.3),    # pure long-range
    (0.2, 0.6, 0.3),    # range-separated hybrid
    (1.0, -1.0, 0.3),   # short-range only (HSE-like sign)
], ids=["hf", "hybrid", "lrc", "rsh", "srx"])
def test_cuest_gridk_matches_four_center_exchange(water, operator, directj, symmetric, k_build):
    """Grid-K reproduces exact alpha K + beta wK; Direct-J alongside it stays exact."""
    _, primary, aux = water
    psi4.set_options({"cuest_k_build": k_build,
                      "cuest_j_build": "direct" if directj else "df", "ints_tolerance": 0.0})
    mints = psi4.core.MintsHelper(primary)
    eri = np.asarray(mints.ao_eri()).reshape((primary.nbf(),) * 4)
    operator_eri = operator[0] * eri
    if operator[1]:
        operator_eri = operator_eri + operator[1] * np.asarray(mints.ao_erf_eri(operator[2])).reshape(eri.shape)
    pairs = _pairs(primary.nbf(), symmetric)
    jk = _builder(primary, aux, operator)
    try:
        coulomb, exchange = _compute(jk, pairs)
    finally:
        jk.finalize()
    max_err = 0.0
    for (left, right), got_j, got_k in zip(pairs, coulomb, exchange):
        density = _density(left, right)
        if directj:
            np.testing.assert_allclose(got_j, np.einsum("pqrs,rs->pq", eri, density), atol=1.e-10, rtol=0)
        expected = np.einsum("prqs,rs->pq", operator_eri, density)
        max_err = max(max_err, np.max(np.abs(got_k - expected)))
        # Grid-K carries a quadrature error, unlike DF-K's fitting error.
        np.testing.assert_allclose(got_k, expected, atol=GRIDK_ATOL[k_build], rtol=0)
        if symmetric:
            np.testing.assert_allclose(got_k, got_k.T, atol=1.e-12, rtol=0)
    np.testing.assert_array_equal(exchange[2], 0)
    if not symmetric:
        # The nonsymmetric kernel ran: K of a nonsymmetric D is itself nonsymmetric.
        assert np.max(np.abs(exchange[1] - exchange[1].T)) > 1.e-3
    print(f"{k_build} {operator} directj={directj} sym={symmetric} max|K - K_4c| = {max_err:.2e}")


@pytest.mark.parametrize("do_j,do_k", [(True, False), (False, True)], ids=["j_only", "k_only"])
def test_cuest_directj_gridk_single_task(water, do_j, do_k):
    """J-only and K-only builds under Direct-J + Grid-K need only one of the two plans."""
    _, primary, aux = water
    psi4.set_options({"cuest_k_build": "grid", "cuest_j_build": "direct", "ints_tolerance": 0.0})
    eri = np.asarray(psi4.core.MintsHelper(primary).ao_eri()).reshape((primary.nbf(),) * 4)
    pairs = _pairs(primary.nbf(), False)
    jk = _builder(primary, aux, do_j=do_j, do_k=do_k)
    try:
        coulomb, exchange = _compute(jk, pairs)
    finally:
        jk.finalize()
    for (left, right), got_j, got_k in zip(pairs, coulomb, exchange):
        density = _density(left, right)
        if do_j:
            np.testing.assert_allclose(got_j, np.einsum("pqrs,rs->pq", eri, density), atol=1.e-10, rtol=0)
        if do_k:
            np.testing.assert_allclose(got_k, np.einsum("prqs,rs->pq", eri, density), atol=GRIDK_ATOL["grid_fit"], rtol=0)


# Worst |E - E(PK)| allowed per Grid-K variant, in Eh.
SCF_ATOL = {"grid_fit": 1.e-5, "grid": 1.e-5}


# Reference energies are CPU SCF_TYPE PK on the same geometry and basis.
@pytest.mark.parametrize("k_build", ["grid_fit", "grid"])
@pytest.mark.parametrize("method,reference", [
    ("hf", "rhf"), ("hf", "uhf"), ("pbe0", "rks"), ("wb97x", "rks"),
])
def test_cuest_directj_gridk_scf_energy(water, method, reference, k_build):
    """Direct-J + Grid-K SCF reproduces the exact-integral (PK) CPU energy."""
    mol, primary, aux = water
    if reference == "uhf":
        mol.set_molecular_charge(1)
        mol.set_multiplicity(2)
    psi4.set_options({"reference": reference, "d_convergence": 1.e-8, "e_convergence": 1.e-10})

    psi4.set_options({"use_cuest": False, "scf_type": "pk"})
    e_ref = psi4.energy(method, molecule=mol)

    psi4.core.clean()
    psi4.set_options({"use_cuest": True, "scf_type": "df", "cuest_j_build": "direct", "cuest_k_build": k_build})
    probe = _builder(primary, aux)  # skip cleanly if the SDK lacks the API
    probe.finalize()
    e_gpu, wfn = psi4.energy(method, molecule=mol, return_wfn=True)

    print(f"{method}/{reference} {k_build}: E(Direct-J+Grid-K) - E(PK) = {e_gpu - e_ref:+.3e} Eh")
    assert abs(e_gpu - e_ref) < SCF_ATOL[k_build]


def test_cuest_directj_gridk_ignores_auxiliary_basis(water):
    """With both direct modes no DF plan is built: the auxiliary basis cannot affect the energy."""
    mol, primary, aux = water
    probe_aux = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", "", "JKFIT", "cc-pvdz")
    psi4.set_options({"cuest_k_build": "grid", "cuest_j_build": "direct"})
    _builder(primary, probe_aux).finalize()
    energies = {}
    for df_basis in ("cc-pvdz-jkfit", "def2-universal-jkfit"):
        psi4.core.clean()
        psi4.set_options({"df_basis_scf": df_basis})
        energies[df_basis] = psi4.energy("hf", molecule=mol)
    assert energies["cc-pvdz-jkfit"] == pytest.approx(energies["def2-universal-jkfit"], abs=1.e-10)


@pytest.mark.parametrize("option,value,message", [
    ("cuest_j_build", "direct", "Analytic Coulomb gradients with CUEST_J_BUILD DIRECT"),
    ("cuest_k_build", "grid", "Analytic exchange gradients with CUEST_K_BUILD GRID"),
    ("cuest_k_build", "grid_fit", "Analytic exchange gradients with CUEST_K_BUILD GRID"),
])
def test_cuest_directj_gridk_gradient_refused(water, option, value, message):
    """No analytic gradient exists for either mode; refuse instead of returning a DF gradient."""
    mol, primary, aux = water
    psi4.set_options({option: value})
    _builder(primary, aux).finalize()
    with pytest.raises(RuntimeError, match=message):
        psi4.gradient("hf", molecule=mol)


@pytest.mark.parametrize("budget", [0, -1])
def test_cuest_gridk_rejects_nonpositive_workspace(water, budget):
    _, primary, aux = water
    psi4.set_options({"cuest_k_build": "grid", "cuest_gridk_workspace_mb": budget})
    with pytest.raises(RuntimeError, match="CUEST_GRIDK_WORKSPACE_MB must be positive"):
        _builder(primary, aux)


@pytest.mark.parametrize("option,value,message", [
    ("cuest_gridk_fit_ortho_cutoff", 0.0, "CUEST_GRIDK_FIT_ORTHO_CUTOFF must be positive"),
    ("cuest_gridk_fit_cutoff", 0.0, "CUEST_GRIDK_FIT_CUTOFF must be in"),
    ("cuest_gridk_fit_cutoff", 1.0, "CUEST_GRIDK_FIT_CUTOFF must be in"),
])
def test_cuest_gridk_rejects_bad_fit_cutoffs(water, option, value, message):
    _, primary, aux = water
    psi4.set_options({"cuest_k_build": "grid_fit", option: value})
    with pytest.raises(RuntimeError, match=message):
        _builder(primary, aux)


def test_cuest_gridk_workspace_does_not_change_result(water):
    """The scratch budget only changes tiling inside cuEST, never K."""
    _, primary, aux = water
    pairs = _pairs(primary.nbf(), False)
    results = []
    for budget in (16, 256):
        psi4.set_options({"cuest_k_build": "grid", "cuest_gridk_workspace_mb": budget})
        jk = _builder(primary, aux)
        try:
            results.append(_compute(jk, pairs))
        finally:
            jk.finalize()
    for small, large in zip(results[0][0] + results[0][1], results[1][0] + results[1][1]):
        np.testing.assert_allclose(small, large, atol=1.e-12, rtol=0)


@pytest.mark.parametrize("grid", [
    {"dft_grid_name": "SG0"},
    {"dft_grid_name": "SG1"},
    {"dft_radial_scheme": "BECKE", "dft_pruning_scheme": "TREUTLER"},
    {"dft_radial_scheme": "EM", "dft_pruning_scheme": "ROBUST"},
    {"dft_radial_scheme": "TREUTLER", "dft_pruning_scheme": "TREUTLER"},
], ids=["sg0", "sg1", "becke-treutler", "em-robust", "treutler-treutler"])
def test_cuest_grid_matches_host_grid(water, grid):
    """The radial/angular arrays handed to cuEST build the same quadrature as the host grid.

    cuEST always partitions with Stratmann weights, so the host side is pinned to STRATMANN.
    SG-0 (MultiExp radii, outermost first) and region pruning on nucleus-first radial schemes
    were assigned angular counts in reverse shell order on the cuEST path.
    """
    mol, _, _ = water
    psi4.set_options({"use_cuest": False, "dft_nuclear_scheme": "stratmann", "e_convergence": 1.e-10,
                      "d_convergence": 1.e-8, **{k: v for k, v in GRID.items() if k != "dft_pruning_scheme"}, **grid})
    e_host = psi4.energy("pbe", molecule=mol)
    psi4.core.clean()
    psi4.set_options({"use_cuest": True})
    e_cuest = psi4.energy("pbe", molecule=mol)
    print(f"{grid}: E(cuEST grid) - E(host grid) = {e_cuest - e_host:+.3e} Eh")
    assert abs(e_cuest - e_host) < 1.e-8
