"""Regressions for the SCF/DF optimization paths and their memory budgets."""

from types import SimpleNamespace

import numpy as np
import pytest
from addons import uusing

import psi4
from psi4.driver.procrouting.scf_proc import scf_iterator

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]


def _bases():
    mol = psi4.geometry("""
        O  0.000  0.000  0.117
        H  0.000  0.757 -0.467
        H  0.000 -0.757 -0.467
        symmetry c1
        no_reorient
        no_com
    """)
    primary = psi4.core.BasisSet.build(mol, "ORBITAL", "cc-pvdz")
    # An orbital basis is sufficient for these algorithm comparisons and keeps
    # the maximum AM at d, including for minimally configured SIMINT builds.
    aux = psi4.core.BasisSet.build(mol, "ORBITAL", "cc-pvdz")
    return primary, aux


def _jk(primary, aux, *, package="libint2", omega=None, combined=False, threads=1, subtype="INCORE"):
    psi4.set_options({
        "scf_type": "mem_df", "scf_subtype": subtype,
        "integral_package": package, "scf_df_k_layout": "QI",
    })
    jk = psi4.core.JK.build_JK(primary, aux)
    jk.set_omp_nthread(threads)
    jk.set_do_wK(omega is not None)
    jk.set_wcombine(combined)
    if omega is not None:
        jk.set_omega(omega)
        jk.set_omega_alpha(0.2)
        jk.set_omega_beta(0.8)
    return jk


def _compute(jk, coefficients):
    jk.C_clear()
    # No right-hand coefficients: exercise the symmetric SCF path.
    jk.C_left_add(psi4.core.Matrix.from_array(coefficients))
    jk.compute()
    matrices = [jk.J()[0], jk.K()[0]] + ([jk.wK()[0]] if jk.wK() else [])
    return [matrix.np.copy() for matrix in matrices]


@pytest.mark.parametrize("omega", [None, 0.0, 0.3])
@pytest.mark.parametrize("threads", [1, 2])
@uusing("simint")
def test_memdf_simint_wk(omega, threads):
    """SIMINT's Coulomb plus Libint's erf engines agree with the direct Libint path."""
    primary, aux = _bases()
    coefficients = np.random.default_rng(31).normal(size=(primary.nbf(), 5)) / primary.nbf()
    results = []
    for package in ("libint2", "simint"):
        jk = _jk(primary, aux, package=package, omega=omega, threads=threads)
        jk.set_memory(10_000_000)
        jk.initialize()
        results.append(_compute(jk, coefficients))
        jk.finalize()
    for ref, actual in zip(*results):
        np.testing.assert_allclose(actual, ref, atol=1e-10, rtol=1e-10)


def test_memdf_combined_wk_remains_disabled():
    """Do not bypass the existing guard to exercise the latent combined-wK bug."""
    primary, aux = _bases()
    jk = _jk(primary, aux, omega=0.3)
    with pytest.raises(RuntimeError, match="wcombine option is currently not available"):
        jk.set_wcombine(True)


@pytest.mark.parametrize("threads", [1, 2])
def test_memdf_zero_omega_wk(threads):
    """erf(0*r)/r is zero, not ordinary Coulomb exchange."""
    primary, aux = _bases()
    coefficients = np.random.default_rng(32).normal(size=(primary.nbf(), 5)) / primary.nbf()
    jk = _jk(primary, aux, omega=0.0, threads=threads)
    jk.set_memory(10_000_000)
    jk.initialize()
    _, _, wk = _compute(jk, coefficients)
    assert np.linalg.norm(jk.K()[0].np) > 1e-3
    np.testing.assert_allclose(wk, 0.0, atol=1e-14, rtol=0)
    jk.finalize()


@pytest.mark.parametrize("roomy", [False, True])
def test_memdf_fp32_cache_budget(roomy):
    """A float AO cache must fit beside retained and upcoming JK scratch."""
    primary, aux = _bases()
    jk = _jk(primary, aux)
    estimate = jk.memory_estimate()
    float_size = (jk.dfh().get_AO_size() + 1) // 2
    jk.set_memory((4 * estimate if roomy else estimate) + float_size)
    jk.initialize()
    dfh = jk.dfh()
    dfh.set_K_fp32(True)
    nbf = primary.nbf()
    coefficients, _ = np.linalg.qr(np.random.default_rng(33).normal(size=(nbf, nbf)))

    def compute(nocc):
        jk.C_clear()
        jk.C_left_add(psi4.core.Matrix.from_array(coefficients[:, :nocc].copy()))
        jk.compute()
        assert dfh.get_AO_size() + dfh.JK_scratch_memory() + dfh.K_fp32_memory() <= dfh.get_memory()
        return jk.K()[0].np.copy()

    compute(1)
    assert dfh.K_fp32_memory() > 0
    k_fast = compute(nbf)
    assert dfh.K_fp32()  # tight budgets must use tiled FP32, not disable it
    assert bool(dfh.K_fp32_memory()) == roomy
    # Shrinking the occupation does not shrink the retained scratch.
    compute(1)
    assert bool(dfh.K_fp32_memory()) == roomy
    jk.release_scratch()
    compute(1)
    assert dfh.K_fp32_memory() > 0
    dfh.set_K_fp32(False)
    k_double = compute(nbf)
    np.testing.assert_allclose(k_fast, k_double, atol=2e-6, rtol=2e-6)
    jk.finalize()


@pytest.mark.parametrize("layout", ["QI", "IQ"])
@pytest.mark.parametrize("threads", [1, 2])
@pytest.mark.parametrize("wk", [False, True])
def test_memdf_retained_scratch_budget(layout, threads, wk):
    """Blocking stays within budget across occupation/symmetry changes and wK."""
    primary, aux = _bases()
    jk = _jk(primary, aux, omega=0.3 if wk else None, threads=threads)
    psi4.set_options({"scf_df_k_layout": layout})
    jk.set_memory(jk.memory_estimate())
    jk.initialize()
    dfh = jk.dfh()
    dfh.set_K_fp32(True)
    nbf = primary.nbf()
    rng = np.random.default_rng(34)
    coefficients = rng.normal(size=(nbf, nbf + 3)) / nbf
    right = rng.normal(size=(nbf, nbf + 3)) / nbf
    resident = dfh.get_AO_size() * (3 if wk else 1)
    results = []
    for nocc, symmetric in [(1, True), (nbf + 3, True), (nbf, False), (1, True)]:
        jk.C_clear()
        jk.C_left_add(psi4.core.Matrix.from_array(coefficients[:, :nocc].copy()))
        if not symmetric:
            jk.C_right_add(psi4.core.Matrix.from_array(right[:, :nocc].copy()))
        jk.compute()
        assert resident + dfh.JK_scratch_memory() + dfh.K_fp32_memory() <= dfh.get_memory()
        assert dfh.K_fp32()
        results.append([m.np.copy() for m in (jk.J()[0], jk.K()[0])] +
                       ([jk.wK()[0].np.copy()] if wk else []))
    dfh.set_K_fp32(False)
    jk.release_scratch()
    assert dfh.JK_scratch_memory() == dfh.K_fp32_memory() == 0
    # Independent FP64 build, ample memory, same input coefficient sequence.
    reference = _jk(primary, aux, omega=0.3 if wk else None, threads=threads)
    reference.set_memory(10_000_000)
    reference.initialize()
    for (nocc, symmetric), actual in zip([(1, True), (nbf + 3, True), (nbf, False), (1, True)], results):
        reference.C_clear()
        reference.C_left_add(psi4.core.Matrix.from_array(coefficients[:, :nocc].copy()))
        if not symmetric:
            reference.C_right_add(psi4.core.Matrix.from_array(right[:, :nocc].copy()))
        reference.compute()
        expected = [m.np for m in (reference.J()[0], reference.K()[0])] + (
            [reference.wK()[0].np] if wk else [])
        for ref, result in zip(expected, actual):
            np.testing.assert_allclose(result, ref, atol=2e-6, rtol=2e-6)
    reference.finalize()
    jk.finalize()


@pytest.mark.parametrize("symmetric", [False, True])
def test_memdf_out_of_core_blocking(symmetric):
    """The shared blocking routine also budgets the streamed AO block."""
    primary, aux = _bases()
    rng = np.random.default_rng(35)
    left = psi4.core.Matrix.from_array(rng.normal(size=(primary.nbf(), 16)) / primary.nbf())
    right = psi4.core.Matrix.from_array(rng.normal(size=(primary.nbf(), 16)) / primary.nbf())
    results = []
    for subtype in ("INCORE", "OUT_OF_CORE"):
        jk = _jk(primary, aux, subtype=subtype)
        jk.set_memory(jk.memory_estimate())
        jk.initialize()
        assert jk.dfh().get_AO_core() == (subtype == "INCORE")
        jk.C_left_add(left)
        if not symmetric:
            jk.C_right_add(right)
        jk.compute()
        results.append([m.np.copy() for m in (jk.J()[0], jk.K()[0])])
        jk.finalize()
    for ref, actual in zip(*results):
        np.testing.assert_allclose(actual, ref, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("layout", ["QI", "IQ"])
@pytest.mark.parametrize("fp32", [False, True])
def test_memdf_tiny_basis_memory_estimate(layout, fp32):
    """The minimum estimate covers occupied-column padding even for nbf < 8."""
    mol = psi4.geometry("He 0 0 0\nsymmetry c1")
    basis = psi4.core.BasisSet.build(mol, "ORBITAL", "sto-3g")
    jk = _jk(basis, basis)
    psi4.set_options({"scf_df_k_layout": layout})
    jk.set_memory(jk.memory_estimate())
    jk.initialize()
    dfh = jk.dfh()
    dfh.set_K_fp32(fp32)
    jk.C_left_add(psi4.core.Matrix.from_array(np.ones((1, 1))))
    jk.compute()
    assert dfh.get_AO_size() + dfh.JK_scratch_memory() + dfh.K_fp32_memory() <= dfh.get_memory()
    assert jk.K()[0].get(0, 0) > 0
    jk.finalize()


def test_memdf_out_of_core_metric_alias_budget():
    """The metric's 1.0 and -0.5 map keys can refer to one retained allocation."""
    mol = psi4.geometry("He 0 0 0\nsymmetry c1")
    basis = psi4.core.BasisSet.build(mol, "ORBITAL", "sto-3g")
    jk = _jk(basis, basis, subtype="OUT_OF_CORE")
    jk.dfh().hold_met(True)
    jk.set_memory(100)
    jk.initialize()
    assert not jk.dfh().get_AO_core()
    # One metric, one streamed AO element, and three scratch elements.
    jk.dfh().set_memory(5)
    jk.C_left_add(psi4.core.Matrix.from_array(np.ones((1, 1))))
    jk.compute()
    assert jk.dfh().JK_scratch_memory() == 3
    assert jk.K()[0].get(0, 0) > 0
    # Reject an undersized sole/last shell instead of emitting an empty block.
    jk.dfh().set_memory(4)
    with pytest.raises(RuntimeError, match="not enough memory for JK blocking"):
        jk.compute()
    jk.finalize()


@pytest.mark.parametrize("ansatz", [0, 1, 2])
@pytest.mark.parametrize("jk_estimate", [600, 950, 1200])
@pytest.mark.parametrize("fp32_words", [0, 700])
@pytest.mark.parametrize("final_size,dfh_offset,mem_df", [
    (400, 0, True), (400, 50, True), (800, 400, True),
    (50, 0, True), (400, 0, False), (400, 400, True),
])
def test_xc_grid_switch_preserves_memory_partition(
        monkeypatch, ansatz, jk_estimate, fp32_words, final_size, dfh_offset, mem_df):
    """Use the real initialization/switch functions, without allocating a large grid."""
    events = []
    multiplier = {0: 1, 1: 4, 2: 10}[ansatz]
    grid = SimpleNamespace(collocation_size=lambda: 100 / multiplier)
    potential = SimpleNamespace(
        grid=lambda: grid,
        functional=lambda: SimpleNamespace(ansatz=lambda: ansatz),
        set_grid=lambda name: setattr(grid, "collocation_size", lambda: final_size / multiplier),
        build_collocation_cache=lambda memory: events.append(("cache", memory)),
    )
    class MockJK:
        def memory_estimate(self):
            return jk_estimate

        def set_memory(self, memory):
            events.append(("jk", memory))
            self.memory = memory

        def dfh(self):
            return self.helper

        def release_scratch(self):
            events.append(("release",))
            self.fp32 = 0

    class MockMemDFJK(MockJK):
        pass

    jk = MockMemDFJK() if mem_df else MockJK()
    wfn = SimpleNamespace(
        attempt_number_=2,
        diag_scratch_size=lambda: 0,
        V_potential=lambda: potential,
        jk=lambda: None,
        get_print=lambda: False,
        molecule=lambda: SimpleNamespace(),
        form_D=lambda: None,
        compute_initial_E=lambda: 0.0,
        set_energies=lambda *args: None,
    )
    get_global_option = psi4.core.get_global_option
    monkeypatch.setattr(psi4.core, "get_memory", lambda: 8000)
    monkeypatch.setattr(
        psi4.core, "get_global_option",
        lambda key: 1.0 if key == "SCF_MEM_SAFETY_FACTOR" else get_global_option(key),
    )
    monkeypatch.setattr(scf_iterator, "_build_jk", lambda *args: jk)
    monkeypatch.setattr(psi4.core, "MemDFJK", MockMemDFJK)
    psi4.set_options({"dft_vv10_postscf": False})
    scf_iterator.scf_initialize(wfn)
    initial_jk, initial_cache = wfn.memory_jk_, wfn.memory_collocation_
    jk.memory = initial_jk
    jk.fp32 = fp32_words
    jk.helper = SimpleNamespace(
        memory=initial_jk - dfh_offset,
        K_fp32_memory=lambda: jk.fp32,
        get_memory=lambda: jk.helper.memory,
    )
    def set_dfh_memory(memory):
        events.append(("dfh", memory))
        jk.helper.memory = memory
    jk.helper.set_memory = set_dfh_memory
    wfn.jk = lambda: jk
    transfer = min(max(0, final_size - initial_cache),
                   max(0, initial_jk - dfh_offset - jk_estimate)) if mem_df else 0
    scf_iterator._xc_grid_switch(wfn, shift_diis=False)
    assert wfn.memory_jk_ == jk.memory == initial_jk - transfer
    assert jk.helper.memory == initial_jk - dfh_offset - transfer
    assert wfn._collocation_budget == initial_cache + transfer
    assert wfn.memory_collocation_ == min(final_size, initial_cache + transfer)
    assert wfn.memory_jk_ + wfn.memory_collocation_ <= 1000
    if transfer:
        assert events[0] == ("release",)
        assert jk.fp32 == 0
        assert jk.helper.memory >= jk_estimate
    else:
        assert events == [("cache", wfn.memory_collocation_)]
        assert jk.fp32 == fp32_words
    assert events[-1] == ("cache", wfn.memory_collocation_)


@pytest.mark.parametrize("layout", ["QI", "IQ"])
@pytest.mark.parametrize("wk", [False, True])
def test_xc_grid_switch_replans_native_jk(layout, wk):
    """A real initialized JK survives donation down to its conservative minimum."""
    primary, aux = _bases()
    jk = _jk(primary, aux, omega=0.3 if wk else None, threads=2)
    psi4.set_options({"scf_df_k_layout": layout})
    minimum = jk.memory_estimate()
    initial_jk = 4 * minimum
    jk.set_memory(initial_jk)
    jk.initialize()
    dfh = jk.dfh()
    dfh.set_K_fp32(True)
    coefficients = np.random.default_rng(36).normal(size=(primary.nbf(), 5)) / primary.nbf()
    before = _compute(jk, coefficients)
    assert dfh.JK_scratch_memory() > 0
    assert dfh.K_fp32_memory() > 0
    offset = initial_jk - dfh.get_memory()

    def build_cache(memory):
        # Allocation cannot overlap the previous large contraction buffers.
        assert dfh.JK_scratch_memory() == dfh.K_fp32_memory() == 0
        assert dfh.get_memory() == minimum
        assert wfn.memory_jk_ + memory == initial_jk + 100

    potential = SimpleNamespace(
        set_grid=lambda name: None,
        grid=lambda: SimpleNamespace(collocation_size=lambda: 10 * minimum),
        functional=lambda: SimpleNamespace(ansatz=lambda: 0),
        build_collocation_cache=build_cache,
    )
    wfn = SimpleNamespace(
        V_potential=lambda: potential, jk=lambda: jk,
        memory_jk_=initial_jk, memory_collocation_=100, _collocation_budget=100,
    )
    scf_iterator._xc_grid_switch(wfn, shift_diis=False)
    assert wfn.memory_jk_ == minimum + offset
    assert dfh.K_fp32()  # Releasing the mirror must not disable tiled FP32.
    after = _compute(jk, coefficients)
    resident = dfh.get_AO_size() * (3 if wk else 1)
    assert resident + dfh.JK_scratch_memory() + dfh.K_fp32_memory() <= dfh.get_memory()
    for ref, actual in zip(before, after):
        np.testing.assert_allclose(actual, ref, atol=2e-6, rtol=2e-6)
    jk.finalize()
