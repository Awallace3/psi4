"""Real blocked FDDS contractions; no SCF/GPU-JK differences in the fixtures."""
import time

import numpy as np
import pytest
import psi4

from addons import uusing

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]


@pytest.fixture
def fdds_factory():
    psi4.set_num_threads(1)
    psi4.set_memory("500 MiB")
    psi4.set_options({"use_cuest": False, "cuest_xc": False,
                      "cuest_mixed_precision": False, "cuest_gemm_min_dim": 0})
    mol = psi4.geometry("""
        He 0 0 0
        He 0 0 4
        symmetry c1
        no_reorient
        no_com
    """)
    primary = psi4.core.BasisSet.build(mol, "BASIS", "cc-pvdz")
    # A small, legitimate Coulomb auxiliary basis also satisfies nov >= naux
    # for hybrid Householder QR (not guaranteed by typical fitting bases).
    auxiliary = psi4.core.BasisSet.build(mol, "BASIS", "cc-pvdz")
    rng = np.random.default_rng(729)
    c, _ = np.linalg.qr(rng.normal(size=(primary.nbf(), primary.nbf())))

    def make(hybrid=False, empty=False):
        occ = 0 if empty else 3
        mc, vc = {}, {}
        for monomer in ("A", "B"):
            mc["Cocc_" + monomer] = psi4.core.Matrix.from_array(c[:, :occ])
            mc["Cvir_" + monomer] = psi4.core.Matrix.from_array(c[:, 3:])
            vc["eps_occ_" + monomer] = psi4.core.Vector.from_array(np.linspace(-1., -.5, occ))
            vc["eps_vir_" + monomer] = psi4.core.Vector.from_array(np.linspace(.1, 1., primary.nbf() - 3))
        cache = {**mc, **vc}
        for monomer in ("A", "B"):
            co = mc["Cocc_" + monomer]
            cache["D_" + monomer] = psi4.core.doublet(co, co, False, True)
        make.inputs = primary, auxiliary, cache
        return psi4.core.FDDS_Dispersion(primary, auxiliary, mc, vc, hybrid)

    yield make
    psi4.core.set_memory_bytes(500 * 1024**2)
    psi4.set_options({"use_cuest": False, "cuest_gemm_min_dim": 256})


def _values(obj, hybrid, monomer, omega):
    if hybrid:
        return {k: m.to_array() for k, m in obj.form_aux_matrices(monomer, omega).items()}
    return {"amp": obj.form_unc_amplitude(monomer, omega).to_array()}


def _timer_calls():
    return sum(v["n_calls"] for k, v in psi4.core.get_timer_dict().items()
               if "cuEST FDDS: contractions" in k)


@pytest.mark.parametrize("hybrid", [False, True])
def test_fdds_native_contractions(fdds_factory, hybrid):
    obj = fdds_factory(hybrid)
    omega = .37
    q = obj.metric().nrow()
    ovq = obj.get_tensor_pqQ("arQ", (3, 7, q)).to_array()
    gaps = np.linspace(.1, 1., 7)[None, :] - np.linspace(-1., -.5, 3)[:, None]
    ld = (-4. * gaps / (gaps**2 + omega**2)).reshape(-1, 1)
    expected = ovq.T @ (ovq * (ld if hybrid else -ld))
    got = _values(obj, hybrid, "A", omega)
    np.testing.assert_allclose(got["amp"], expected, rtol=0, atol=2.e-12)
    if hybrid:
        x = obj.get_tensor_pqQ("XarQ", (3, 7, q)).to_array()
        y = obj.get_tensor_pqQ("YarQ", (3, 7, q)).to_array()
        qx = obj.get_tensor_pqQ("QXarQ", (3, 7, q)).to_array()
        qy = obj.get_tensor_pqQ("QYarQ", (3, 7, q)).to_array()
        yl = (y - x) * (-4. / (gaps**2 + omega**2)).reshape(-1, 1)
        refs = {"K1LD": (ovq * ld).T @ (qx + qy),
                "K2LD": (ovq * ld).T @ (qy - qx),
                "K2L": ovq.T @ yl, "K21L": yl.T @ (qx + qy)}
        for k, ref in refs.items():
            np.testing.assert_allclose(got[k], ref, rtol=0, atol=2.e-11)
    assert obj.gpu_contraction_blocks() == 0
    with pytest.raises(RuntimeError, match="Monomer must be A or B"):
        _values(obj, hybrid, "C", omega)


def test_fdds_empty_native(fdds_factory):
    obj = fdds_factory(empty=True)
    assert not obj.form_unc_amplitude("A", .1).np.any()
    # Early empty return must not read missing hybrid disk tensors.
    assert all(not m.np.any() for m in obj.form_aux_matrices("A", .1).values())
    assert obj.gpu_contraction_blocks() == 0


@pytest.mark.parametrize("hybrid", [False, True])
def test_fdds_memory_guard(fdds_factory, hybrid):
    obj = fdds_factory(hybrid)
    psi4.core.set_memory_bytes(8)
    with pytest.raises(RuntimeError, match="Too little memory"):
        _values(obj, hybrid, "A", .1)
    assert obj.gpu_contraction_blocks() == 0


@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.cuest
@pytest.mark.parametrize("hybrid", [False, True])
def test_fdds_gpu_contributions(fdds_factory, hybrid):
    obj = fdds_factory(hybrid)
    # Force a 2+1 occupied split, including a partial tail. Host and GPU use
    # the same memory budget. These low budgets apply only after DF setup.
    q, v, o = 10, 7, 3
    fixed = 2 * o * v + 6 * q * q if hybrid else o * v + q * q
    per_occ = 7 * v * q if hybrid else v * q
    psi4.core.set_memory_bytes(int((fixed + 2 * per_occ + 1) * 8 / .8))
    ref = {(m, w): _values(obj, hybrid, m, w) for m in ("A", "B") for w in (.0, .37, 2.)}
    for enabled, threshold in ((True, 10**9), (False, 0), (True, 0)):
        psi4.set_options({"use_cuest": enabled, "cuest_gemm_min_dim": threshold})
        psi4.core.clean_timers()
        before = obj.gpu_contraction_blocks()
        start = time.perf_counter()
        for (m, w), expected in ref.items():
            got = _values(obj, hybrid, m, w)
            for k in expected:
                np.testing.assert_allclose(got[k], expected[k], rtol=0, atol=2.e-11)
        elapsed = time.perf_counter() - start
        routed = enabled and threshold == 0
        assert obj.gpu_contraction_blocks() - before == (12 if routed else 0)
        assert _timer_calls() == (6 if routed else 0)
        print(f"FDDS hybrid={hybrid} enabled={enabled} threshold={threshold} elapsed={elapsed:.6f}s")


@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.cuest
def test_fdds_gpu_empty_and_preflight(fdds_factory):
    obj = fdds_factory(empty=True)
    psi4.set_options({"use_cuest": True, "cuest_gemm_min_dim": 0})
    psi4.core.clean_timers()
    assert not obj.form_unc_amplitude("A", .1).np.any()
    assert all(not m.np.any() for m in obj.form_aux_matrices("A", .1).values())
    assert obj.gpu_contraction_blocks() == 0
    assert _timer_calls() == 0


@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.cuest
def test_fdds_gpu_negative_threshold(fdds_factory):
    obj = fdds_factory()
    psi4.set_options({"use_cuest": True, "cuest_gemm_min_dim": -1})
    with pytest.raises(RuntimeError, match="must be nonnegative"):
        obj.form_unc_amplitude("A", .1)
    assert obj.gpu_contraction_blocks() == 0


@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.cuest
@pytest.mark.parametrize("hybrid", [False, True])
def test_fdds_integrated_gpu_energy(fdds_factory, hybrid):
    """Actual ALDA kernel, hybrid terms, pseudoinverse and default quadrature."""
    from psi4.driver.procrouting.sapt.sapt_mp2_terms import df_fdds_dispersion
    fdds_factory()
    primary, auxiliary, cache = fdds_factory.inputs
    results = []
    for enabled, threshold in ((False, 0), (True, 10**9), (True, 0)):
        psi4.set_options({"use_cuest": enabled, "cuest_gemm_min_dim": threshold})
        psi4.core.clean_timers()
        start = time.perf_counter()
        result = df_fdds_dispersion(primary, auxiliary, cache, hybrid, .25 if hybrid else 0.,
                                    do_print=False)
        elapsed = time.perf_counter() - start
        assert _timer_calls() == (20 if enabled and threshold == 0 else 0)
        assert set(result) == {"Disp20,FDDS (unc)", "Disp20"}
        assert all(np.isfinite(v) for v in result.values())
        results.append(result)
        errors = {k: abs(result[k] - results[0][k]) for k in result}
        timings = {k: v["wall_time"] for k, v in psi4.core.get_timer_dict().items()
                   if any(t in k for t in ("Form FDDS object", "Form xc kernel", "Time Integration",
                                           "cuEST FDDS: contractions"))}
        print(f"FDDS integrated hybrid={hybrid} enabled={enabled} threshold={threshold} "
              f"elapsed={elapsed:.6f}s energy={result} abs_errors={errors} timers={timings}")
    for result in results[1:]:
        for k in result:
            assert abs(result[k] - results[0][k]) < 1.e-10


@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.cuest
@pytest.mark.saptdft
def test_saptdft_fdds_native_xc_gpu(tmp_path):
    """Small full SAPT(DFT), native supported LDA response, no GPU-XC claim."""
    results = []
    for enabled, threshold in ((False, 0), (True, 10**9), (True, 0)):
        psi4.core.clean()
        psi4.core.clean_options()
        psi4.set_num_threads(1)
        psi4.set_memory("500 MiB")
        path = tmp_path / f"sapt-{enabled}-{threshold}.out"
        psi4.core.set_output_file(str(path), False)
        psi4.geometry("""
            0 1
            He 0 0 0
            --
            0 1
            He 0 0 4
            symmetry c1
            no_reorient
            no_com
        """)
        psi4.set_options({"basis": "cc-pvdz", "scf_type": "df",
                          "sapt_dft_functional": "svwn",
                          "sapt_dft_grac_shift_a": .1, "sapt_dft_grac_shift_b": .1,
                          "sapt_dft_use_einsums": False,
                          "e_convergence": 1.e-10, "d_convergence": 1.e-10,
                          "use_cuest": enabled, "cuest_xc": False,
                          "cuest_mixed_precision": False,
                          "cuest_gemm_min_dim": threshold})
        psi4.core.clean_timers()
        psi4.energy("sapt(dft)")
        keys = ("SAPT DISP ENERGY", "SAPT TOTAL ENERGY")
        result = {k: psi4.variable(k) for k in keys}
        assert _timer_calls() == (20 if enabled and threshold == 0 else 0)
        results.append(result)
        psi4.core.close_outfile()
        assert "Disp20" in path.read_text()
        print(f"SAPT(DFT) enabled={enabled} threshold={threshold} values={result}")
    for result in results[1:]:
        for k in result:
            assert abs(result[k] - results[0][k]) <= 1.e-5
    psi4.set_options({"use_cuest": False})
