"""Real blocked FDDS contractions; no SCF/GPU-JK differences in the fixtures."""
import time
import math

import numpy as np
import pytest
import psi4

from addons import uusing

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]


@pytest.fixture
def fdds_factory():
    psi4.set_num_threads(1)
    psi4.set_memory("500 MiB")
    previous_subtype = psi4.core.get_global_option("SCF_SUBTYPE")
    psi4.set_options({"use_cuest": False, "cuest_xc": False,
                      "cuest_mixed_precision": False, "cuest_gemm_min_dim": 0,
                      "cuest_fdds_pinv": False,
                      "scf_subtype": "auto"})
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

    def make(hybrid=False, empty=False, empty_space="occ", occupied=(3, 3)):
        mc, vc = {}, {}
        make.dimensions = {}
        for index, monomer in enumerate(("A", "B")):
            nominal_occ = occupied[index]
            is_empty = empty is True or empty == monomer
            occ = 0 if is_empty and empty_space == "occ" else nominal_occ
            vir = 0 if is_empty and empty_space == "vir" else primary.nbf() - nominal_occ
            mc["Cocc_" + monomer] = psi4.core.Matrix.from_array(c[:, :occ])
            mc["Cvir_" + monomer] = psi4.core.Matrix.from_array(c[:, nominal_occ:nominal_occ + vir])
            vc["eps_occ_" + monomer] = psi4.core.Vector.from_array(np.linspace(-1., -.5, occ))
            vc["eps_vir_" + monomer] = psi4.core.Vector.from_array(np.linspace(.1, 1., vir))
            make.dimensions[monomer] = occ, vir
        cache = {**mc, **vc}
        for monomer in ("A", "B"):
            co = mc["Cocc_" + monomer]
            cache["D_" + monomer] = psi4.core.doublet(co, co, False, True)
        make.inputs = primary, auxiliary, cache
        return psi4.core.FDDS_Dispersion(primary, auxiliary, mc, vc, hybrid)

    yield make
    psi4.core.set_memory_bytes(500 * 1024**2)
    psi4.set_options({"use_cuest": False, "cuest_gemm_min_dim": 256,
                      "cuest_fdds_pinv": False,
                      "scf_subtype": previous_subtype})


def _values(obj, hybrid, monomer, omega):
    if hybrid:
        return {k: m.to_array() for k, m in obj.form_aux_matrices(monomer, omega).items()}
    return {"amp": obj.form_unc_amplitude(monomer, omega).to_array()}


def _timer_calls():
    return sum(v["n_calls"] for k, v in psi4.core.get_timer_dict().items()
               if "cuEST FDDS: contractions" in k)

def _product_timer_calls():
    return sum(v["n_calls"] for k, v in psi4.core.get_timer_dict().items()
               if "cuEST FDDS: response products" in k)

def _pinv_timer_calls():
    return sum(v["n_calls"] for k, v in psi4.core.get_timer_dict().items()
               if "cuEST FDDS: pseudoinverse" in k)


@pytest.mark.parametrize("kind", ["square", "rectangular", "strided", "float32", "empty", "rankdef"])
def test_fdds_native_pseudoinverse(fdds_factory, kind):
    from psi4.driver.procrouting.sapt.sapt_mp2_terms import _fdds_pinv
    a = np.random.default_rng(119).normal(size=(7, 7))
    if kind == "rectangular":
        a = a[:4]
    elif kind == "strided":
        a = a.T
    elif kind == "float32":
        a = a.astype(np.float32)
    elif kind == "empty":
        a = np.empty((0, 0))
    elif kind == "rankdef":
        a[:, 2] = a[:, 0] - a[:, 1]
    original = a.copy()
    psi4.core.clean_timers()
    for opted_in in (False, True):
        psi4.set_options({"use_cuest": False, "cuest_fdds_pinv": opted_in})
        np.testing.assert_array_equal(_fdds_pinv(a, rcond=1.e-13), np.linalg.pinv(a, rcond=1.e-13))
    np.testing.assert_array_equal(a, original)
    assert _pinv_timer_calls() == 0


def test_fdds_pseudoinverse_error_never_retries_cpu(fdds_factory, monkeypatch):
    from psi4.driver.procrouting.sapt.sapt_mp2_terms import _fdds_pinv
    monkeypatch.setattr(psi4.core, "get_global_option",
                        lambda key: {"USE_CUEST": True, "CUEST_FDDS_PINV": True,
                                     "CUEST_GEMM_MIN_DIM": 0}[key])
    def failed_endpoint(*args):
        raise RuntimeError("injected endpoint failure")
    def forbidden_retry(*args, **kwargs):
        pytest.fail("An endpoint failure must not trigger a NumPy retry")
    monkeypatch.setattr(psi4.core, "_cuest_fdds_pinv", failed_endpoint, raising=False)
    monkeypatch.setattr(np.linalg, "pinv", forbidden_retry)
    with pytest.raises(RuntimeError, match="injected endpoint failure"):
        _fdds_pinv(np.eye(5))


@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.cuest
@pytest.mark.parametrize("kind", ["random5", "random17", "rankdef", "zero", "cutoff"])
def test_fdds_gpu_pseudoinverse(fdds_factory, kind):
    from psi4.driver.procrouting.sapt.sapt_mp2_terms import _fdds_pinv
    rng = np.random.default_rng(763)
    for enabled, opted_in, threshold in ((False, True, 0), (True, False, 0),
                                         (True, True, 10**9), (True, True, 0)):
        psi4.set_options({"use_cuest": enabled, "cuest_fdds_pinv": opted_in,
                          "cuest_gemm_min_dim": threshold})
        for repeat in range(2):  # changed inputs, no stale device factors
            q = 17 if kind == "random17" else 5
            a = rng.normal(size=(q, q))
            if kind == "rankdef":
                a = np.outer(a[:, 0], a[0])
            elif kind == "zero":
                a.fill(0)
            elif kind == "cutoff":
                a = np.diag([1., 2.e-13, 1.e-13, 5.e-14, 0.])
            original = a.copy()
            a.flags.writeable = False
            psi4.core.clean_timers()
            actual = _fdds_pinv(a, rcond=1.e-13)
            reference = np.linalg.pinv(a, rcond=1.e-13)
            np.testing.assert_allclose(actual, reference, rtol=2.e-11, atol=5.e-12)
            assert _pinv_timer_calls() == int(enabled and opted_in and threshold == 0)
            np.testing.assert_array_equal(a, original)
            assert not np.shares_memory(a, actual)
            if kind == "cutoff":
                # Equality at 1e-13 is discarded; 2e-13 is retained.
                assert np.count_nonzero(np.diag(actual)) == 2


@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.cuest
@pytest.mark.parametrize("kind", ["rectangular", "strided", "float32", "empty"])
def test_fdds_gpu_pseudoinverse_fallback(fdds_factory, kind):
    from psi4.driver.procrouting.sapt.sapt_mp2_terms import _fdds_pinv
    a = np.random.default_rng(119).normal(size=(7, 7))
    if kind == "rectangular":
        a = a[:4]
    elif kind == "strided":
        a = a.T
    elif kind == "float32":
        a = a.astype(np.float32)
    else:
        a = np.empty((0, 0))
    psi4.set_options({"use_cuest": True, "cuest_fdds_pinv": True, "cuest_gemm_min_dim": 0})
    psi4.core.clean_timers()
    np.testing.assert_array_equal(_fdds_pinv(a), np.linalg.pinv(a, rcond=1.e-13))
    assert _pinv_timer_calls() == 0


@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.cuest
def test_fdds_gpu_pseudoinverse_preflight_errors(fdds_factory):
    endpoint = psi4.core._cuest_fdds_pinv
    a = np.eye(5)
    psi4.core.clean_timers()
    unaligned = np.ndarray((5, 5), dtype=np.float64, buffer=bytearray(201), offset=1)
    assert unaligned.flags.c_contiguous and not unaligned.flags.aligned
    assert endpoint(unaligned, 1.e-13, 0) is None
    with pytest.raises(RuntimeError, match="negative threshold"):
        endpoint(a, 1.e-13, -1)
    for cutoff in (-1., np.nan, np.inf):
        with pytest.raises(RuntimeError, match="rcond"):
            endpoint(a, cutoff, 0)
    a[0, 0] = np.nan
    with pytest.raises(RuntimeError, match="nonfinite input"):
        endpoint(a, 1.e-13, 0)
    assert _pinv_timer_calls() == 0


@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.cuest
@pytest.mark.parametrize("hybrid", [False, True])
def test_fdds_gpu_pseudoinverse_integrated(fdds_factory, hybrid):
    from psi4.driver.procrouting.sapt.sapt_mp2_terms import df_fdds_dispersion
    fdds_factory(occupied=(2, 3))
    primary, auxiliary, cache = fdds_factory.inputs
    reference = None
    for enabled, opted_in in ((False, False), (True, False), (True, True)):
        psi4.set_options({"use_cuest": enabled, "cuest_fdds_pinv": opted_in,
                          "cuest_gemm_min_dim": 0})
        psi4.core.clean_timers()
        actual = df_fdds_dispersion(primary, auxiliary, cache, hybrid, .25 if hybrid else 0.,
                                    do_print=False)
        assert _pinv_timer_calls() == ((22 if hybrid else 20) if opted_in else 0)
        if reference is None:
            reference = actual
        errors = {key: abs(actual[key] - value) for key, value in reference.items()}
        assert max(errors.values()) < 1.e-10, errors
        print("FDDS pseudoinverse integrated", hybrid, enabled, opted_in, errors)


@pytest.mark.parametrize("kind", ["square", "rectangular", "float32", "strided", "empty"])
def test_fdds_native_response_products(fdds_factory, kind):
    from psi4.driver.procrouting.sapt.sapt_mp2_terms import _fdds_dot3
    rng = np.random.default_rng(524)
    a, b, c = (rng.normal(size=(17, 17)) for _ in range(3))
    if kind == "rectangular":
        a = a[:5]
    elif kind == "float32":
        a, b, c = (x.astype(np.float32) for x in (a, b, c))
    elif kind == "strided":
        a, b, c = (x.T for x in (a, b, c))
    elif kind == "empty":
        a, b, c = (np.empty((0, 0)) for _ in range(3))
    originals = [x.copy() for x in (a, b, c)]
    psi4.core.clean_timers()
    result = _fdds_dot3(a, b, c)
    np.testing.assert_array_equal(result, a.dot(b).dot(c))
    for actual, expected in zip((a, b, c), originals):
        np.testing.assert_array_equal(actual, expected)
    assert _product_timer_calls() == 0


@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.cuest
@pytest.mark.parametrize("q", [5, 17])
def test_fdds_gpu_response_products(fdds_factory, q):
    from psi4.driver.procrouting.sapt.sapt_mp2_terms import _fdds_dot3
    rng = np.random.default_rng(871)
    for enabled, threshold in ((False, 0), (True, 10**9), (True, 0)):
        psi4.set_options({"use_cuest": enabled, "cuest_gemm_min_dim": threshold})
        for _ in range(3):
            a, b, c = (rng.normal(size=(q, q)) for _ in range(3))
            originals = [x.copy() for x in (a, b, c)]
            psi4.core.clean_timers()
            actual = _fdds_dot3(a, b, c)
            np.testing.assert_allclose(actual, a.dot(b).dot(c), rtol=2.e-13, atol=2.e-13)
            assert _product_timer_calls() == int(enabled and threshold == 0)
            for value, original in zip((a, b, c), originals):
                np.testing.assert_array_equal(value, original)


@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.cuest
@pytest.mark.parametrize("kind", ["rectangular", "float32", "strided", "empty"])
def test_fdds_gpu_response_product_fallback(fdds_factory, kind):
    psi4.set_options({"use_cuest": True, "cuest_gemm_min_dim": 0})
    # Reuse the exact public helper/native contract, but force GPU eligibility
    # controls: unsupported layouts/dtypes still execute original NumPy dots.
    test_fdds_native_response_products(fdds_factory, kind)

@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.cuest
def test_fdds_gpu_response_product_alias_and_preflight(fdds_factory):
    from psi4.driver.procrouting.sapt.sapt_mp2_terms import _fdds_dot3
    psi4.set_options({"use_cuest": True, "cuest_gemm_min_dim": 0})
    a = np.random.default_rng(777).normal(size=(17, 17))
    original = a.copy()
    a.flags.writeable = False
    with pytest.raises(RuntimeError, match="negative threshold"):
        psi4.core._cuest_fdds_dot3(a, a, a, -1)
    psi4.core.clean_timers()
    result = _fdds_dot3(a, a, a)
    assert _product_timer_calls() == 1
    np.testing.assert_allclose(result, a.dot(a).dot(a), rtol=2.e-13, atol=2.e-13)
    np.testing.assert_array_equal(a, original)
    assert not np.shares_memory(result, a)

def _contribution_budget(naux, dimensions, hybrid):
    """Small host budget; predict whole-occupied blocking before device limits."""
    fixed = {}
    per_occ = {}
    for monomer, (o, v) in dimensions.items():
        fixed[monomer] = (2 * o * v + 6 * naux**2) if hybrid else (o * v + naux**2)
        per_occ[monomer] = (7 if hybrid else 1) * v * naux
    # Keep B's 2+1 occupied split, but also satisfy each monomer's preflight:
    # uncoupled requires TWO q*v blocks even when A has only two occupied rows;
    # hybrid requires one seven-buffer block. Unequal uncoupled A is one block.
    minimum = max(fixed[m] + (1 if hybrid else 2) * per_occ[m] for m in dimensions)
    doubles = max(fixed["B"] + 2 * per_occ["B"], minimum) + 1
    memory_bytes = math.ceil(doubles * 8 / .8)
    available = int(.8 * (memory_bytes // 8))
    blocks = {}
    for monomer, (o, _) in dimensions.items():
        block_size = min(o, (available - fixed[monomer]) // per_occ[monomer])
        blocks[monomer] = math.ceil(o / block_size)
    return memory_bytes, blocks


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("occupied", [(3, 3), (2, 3)], ids=["equal", "unequal"])
def test_fdds_contribution_fixture_budget(fdds_factory, hybrid, occupied):
    """The GPU comparison's CPU reference must fit both real monomer shapes."""
    obj = fdds_factory(hybrid, occupied=occupied)
    q = obj.metric().np.shape[0]
    memory_bytes, blocks = _contribution_budget(q, fdds_factory.dimensions, hybrid)
    psi4.core.set_memory_bytes(memory_bytes)
    for monomer in ("A", "B"):
        got = _values(obj, hybrid, monomer, .37)
        refs = _reference_contractions(obj, hybrid, monomer,
                                       fdds_factory.dimensions[monomer],
                                       fdds_factory.inputs[2], .37)
        for name in refs:
            np.testing.assert_allclose(got[name], refs[name], rtol=0,
                                       atol=2.e-12 if name == "amp" else 2.e-11)
    assert blocks == ({"A": 1, "B": 2} if occupied == (2, 3) and not hybrid
                      else {"A": 2, "B": 2})
    assert obj.gpu_contraction_blocks() == 0


def _reference_contractions(obj, hybrid, monomer, dimensions, cache, omega):
    o, v = dimensions
    q = obj.metric().np.shape[0]
    suffix = "arQ" if monomer == "A" else "bsQ"
    ovq = obj.get_tensor_pqQ(suffix, (o, v, q)).to_array().reshape(-1, q)
    gaps = cache["eps_vir_" + monomer].np[None, :] - cache["eps_occ_" + monomer].np[:, None]
    ld = (-4. * gaps / (gaps**2 + omega**2)).reshape(-1, 1)
    refs = {"amp": ovq.T @ (ovq * (ld if hybrid else -ld))}
    if hybrid:
        x = obj.get_tensor_pqQ("X" + suffix, (o, v, q)).to_array().reshape(-1, q)
        y = obj.get_tensor_pqQ("Y" + suffix, (o, v, q)).to_array().reshape(-1, q)
        qx = obj.get_tensor_pqQ("QX" + suffix, (o, v, q)).to_array().reshape(-1, q)
        qy = obj.get_tensor_pqQ("QY" + suffix, (o, v, q)).to_array().reshape(-1, q)
        yl = (y - x) * (-4. / (gaps**2 + omega**2)).reshape(-1, 1)
        refs.update({"K1LD": (ovq * ld).T @ (qx + qy),
                     "K2LD": (ovq * ld).T @ (qy - qx),
                     "K2L": ovq.T @ yl, "K21L": yl.T @ (qx + qy)})
    return refs


def _exchange_tensors(obj, monomer, dimensions):
    o, v = dimensions
    q = obj.metric().np.shape[0]
    suffix = "arQ" if monomer == "A" else "bsQ"
    return {prefix: obj.get_tensor_pqQ(prefix + suffix, (o, v, q)).np.copy()
            for prefix in ("X", "QX", "Y", "QY")}


def _reference_exchange_tensors(obj, monomer, dimensions):
    """Independent contractions of the actual transformed integral tensors."""
    o, v = dimensions
    q = obj.metric().np.shape[0]
    names = (("arR", "arQ", "QarQ", "aaR", "rrR", "raQ", "QraQ")
             if monomer == "A" else
             ("bsR", "bsQ", "QbsQ", "bbR", "ssR", "sbQ", "QsbQ"))
    shapes = ((o, v, q), (o, v, q), (o, v, q), (o, o, q),
              (v, v, q), (v, o, q), (v, o, q))
    b, raw, orth, aa, rr, ra, qra = (
        obj.get_tensor_pqQ(name, shape).np.copy() for name, shape in zip(names, shapes))
    # Preserve the implemented index order, including nonsymmetric X/QX.
    return {
        "X": np.einsum("prk,ask,psq->arq", b, b, raw),
        "QX": np.einsum("prk,ask,psq->arq", b, b, orth),
        "Y": np.einsum("srk,apk,spq->arq", rr, aa, ra),
        "QY": np.einsum("srk,apk,spq->arq", rr, aa, qra),
    }


def _assert_exchange_partial_blocks(obj, dimensions, memory_bytes):
    """Check that the small budget exercises partial tails in both X and Y."""
    if memory_bytes is None:
        return
    q = obj.metric().np.shape[0]
    available = int(.8 * (memory_bytes // 8))
    x_tail, y_tail = [], []
    for o, v in dimensions.values():
        x_outer = min(o, (available - v * v) // (6 * v * q))
        kov = v // o + 1
        y_block = (available - o * v) // ((kov**2 + 4 * kov + 1) * o * q)
        y_inner = min(v, y_block * max(1, kov - 1))
        assert x_outer > 0 and y_inner > 0
        x_tail.append(x_outer < o and o % x_outer != 0)
        y_tail.append(y_inner < v and v % y_inner != 0)
    assert any(x_tail) and any(y_tail)


@pytest.mark.parametrize("occupied", [(3, 3), (2, 3), (6, 6)], ids=["equal", "unequal", "dense-occ"])
@pytest.mark.parametrize("memory_bytes", [None, 12800], ids=["whole", "blocked"])
def test_fdds_native_exchange_preprocessing(fdds_factory, occupied, memory_bytes):
    if memory_bytes is not None:
        psi4.core.set_memory_bytes(memory_bytes)
    obj = fdds_factory(True, occupied=occupied)
    _assert_exchange_partial_blocks(obj, fdds_factory.dimensions, memory_bytes)
    for monomer in ("A", "B"):
        dims = fdds_factory.dimensions[monomer]
        refs = _reference_exchange_tensors(obj, monomer, dims)
        for key, got in _exchange_tensors(obj, monomer, dims).items():
            np.testing.assert_allclose(got, refs[key], rtol=0, atol=2.e-11)


@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.cuest
@pytest.mark.parametrize("occupied", [(3, 3), (2, 3), (6, 6)], ids=["equal", "unequal", "dense-occ"])
@pytest.mark.parametrize("memory_bytes", [None, 12800], ids=["whole", "blocked"])
def test_fdds_gpu_exchange_preprocessing(fdds_factory, occupied, memory_bytes):
    if memory_bytes is not None:
        psi4.core.set_memory_bytes(memory_bytes)
    obj = fdds_factory(True, occupied=occupied)
    _assert_exchange_partial_blocks(obj, fdds_factory.dimensions, memory_bytes)
    refs = {m: _reference_exchange_tensors(obj, m, fdds_factory.dimensions[m])
            for m in ("A", "B")}
    del obj
    for enabled, threshold in ((False, 0), (True, 10**9), (True, 0)):
        psi4.set_options({"use_cuest": enabled, "cuest_gemm_min_dim": threshold})
        psi4.core.clean_timers()
        obj = fdds_factory(True, occupied=occupied)
        calls = sum(v["n_calls"] for k, v in psi4.core.get_timer_dict().items()
                    if "cuEST FDDS: hybrid exchange" in k)
        assert calls == (4 if enabled and threshold == 0 else 0)
        for monomer in ("A", "B"):
            for key, got in _exchange_tensors(obj, monomer, fdds_factory.dimensions[monomer]).items():
                np.testing.assert_allclose(got, refs[monomer][key], rtol=0, atol=2.e-11)
        del obj
    psi4.set_options({"use_cuest": False})


@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.cuest
@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("occupied", [(3, 3), (2, 3)], ids=["equal", "unequal"])
@pytest.mark.parametrize("memory_bytes", [None, 12800], ids=["whole", "blocked"])
def test_fdds_gpu_metric_fold(fdds_factory, hybrid, occupied, memory_bytes):
    """Compare actual FDDS pqQ and Qpq metric folds, including opt-out controls."""
    if memory_bytes is not None:
        psi4.core.set_memory_bytes(memory_bytes)
    obj = fdds_factory(hybrid, occupied=occupied)
    q = obj.metric().np.shape[0]
    shapes = {}
    for monomer, (o, v) in fdds_factory.dimensions.items():
        a, r = ("a", "r") if monomer == "A" else ("b", "s")
        shapes[a + r + "Q"] = (o, v, q)
        if hybrid:
            shapes[r + a + "Q"] = (v, o, q)
            shapes["Q" + a + r + "Q"] = (o, v, q)
            shapes[a + a + "R"] = (o, o, q)
            shapes[a + r + "R"] = (o, v, q)
            shapes[r + r + "R"] = (v, v, q)
    refs = {name: obj.get_tensor_pqQ(name, shape).np.copy()
            for name, shape in shapes.items()}
    del obj
    for enabled, threshold in ((False, 0), (True, 10**9), (True, 0)):
        psi4.set_options({"use_cuest": enabled, "cuest_gemm_min_dim": threshold})
        psi4.core.clean_timers()
        obj = fdds_factory(hybrid, occupied=occupied)
        calls = sum(v["n_calls"] for k, v in psi4.core.get_timer_dict().items()
                    if "cuEST FDDS: metric fold" in k)
        assert (calls > 0) == (enabled and threshold == 0)
        for name, shape in shapes.items():
            got = obj.get_tensor_pqQ(name, shape).np
            np.testing.assert_allclose(got, refs[name], rtol=0, atol=2.e-12)
        del obj
    psi4.set_options({"use_cuest": False})


@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.cuest
@pytest.mark.parametrize("layout", ["pqQ", "Qpq", "pQq", "Qpq_to_pqQ"])
def test_fdds_gpu_metric_layout(layout):
    """Nonsymmetric metric, unequal dimensions, tail and repeated-input freshness."""
    rng = np.random.default_rng(401)
    q, r = 5, 3
    metric = rng.normal(size=(q, q))  # physical symmetric metrics hide transpose mistakes
    inputs, expected = [], []
    for p in (3, 1, 3, 0):
        shape = ((p * r, q) if layout == "pqQ" else
                 (p * q, r) if layout == "pQq" else (q, p * r))
        a = rng.normal(size=shape)
        inputs.append(psi4.core.Matrix.from_array(a))
        if layout == "pqQ":
            expected.append(a @ metric)
        elif layout == "Qpq":
            expected.append(metric @ a)
        elif layout == "pQq":
            expected.append(np.einsum("ij,pjk->pik", metric, a.reshape(p, q, r)).reshape(shape))
        else:
            expected.append(a.T @ metric)
    met = psi4.core.Matrix.from_array(metric)
    saved = [a.np.copy() for a in inputs]
    got = psi4.core._cuest_df_metric_fold(met, inputs, layout, r)
    for out, ref, a, before in zip(got, expected, inputs, saved):
        np.testing.assert_allclose(out.np, ref, rtol=0, atol=2.e-13)
        np.testing.assert_array_equal(a.np, before)
    # Invalid requests fail before upload; a following valid call still works.
    with pytest.raises(RuntimeError, match="unsupported layout"):
        psi4.core._cuest_df_metric_fold(met, inputs, "invalid", r)
    with pytest.raises(RuntimeError, match="positive r"):
        psi4.core._cuest_df_metric_fold(met, inputs, layout, 0)
    repeated = psi4.core._cuest_df_metric_fold(met, inputs, layout, r)
    for out, ref in zip(repeated, expected):
        np.testing.assert_allclose(out.np, ref, rtol=0, atol=2.e-13)


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("occupied", [(3, 3), (2, 3)], ids=["equal", "unequal"])
def test_fdds_native_contractions(fdds_factory, hybrid, occupied):
    obj = fdds_factory(hybrid, occupied=occupied)
    omega = .37
    cache = fdds_factory.inputs[2]
    for monomer in ("A", "B"):
        refs = _reference_contractions(obj, hybrid, monomer, fdds_factory.dimensions[monomer], cache, omega)
        got = _values(obj, hybrid, monomer, omega)
        errors = {}
        for k, ref in refs.items():
            np.testing.assert_allclose(got[k], ref, rtol=0, atol=2.e-12 if k == "amp" else 2.e-11)
            errors[k] = float(np.max(np.abs(got[k] - ref)))
        print(f"FDDS native hybrid={hybrid} monomer={monomer} dimensions={fdds_factory.dimensions[monomer]} "
              f"max_abs_errors={errors}")
    assert obj.gpu_contraction_blocks() == 0
    with pytest.raises(RuntimeError, match="Monomer must be A or B"):
        _values(obj, hybrid, "C", omega)
    primary, auxiliary, cache = fdds_factory.inputs
    mc = {k: cache[k] for k in ("Cocc_A", "Cvir_A", "Cocc_B", "Cvir_B")}
    vc = {k: cache[k] for k in ("eps_occ_A", "eps_vir_A", "eps_occ_B", "eps_vir_B")}
    with pytest.raises(RuntimeError, match="Missing values in the matrix_cache"):
        psi4.core.FDDS_Dispersion(primary, auxiliary, {k: v for k, v in mc.items() if k != "Cocc_A"}, vc, hybrid)
    with pytest.raises(RuntimeError, match="Missing values in the vector_cache"):
        psi4.core.FDDS_Dispersion(primary, auxiliary, mc, {k: v for k, v in vc.items() if k != "eps_occ_A"}, hybrid)


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("empty", [True, "A", "B"])
def test_fdds_empty_native(fdds_factory, hybrid, empty):
    obj = fdds_factory(hybrid, empty=empty)
    for monomer in ("A", "B"):
        if empty is True or empty == monomer:
            assert not obj.form_unc_amplitude(monomer, .1).np.any()
            # Early return must not read missing hybrid disk tensors.
            assert all(not m.np.any() for m in obj.form_aux_matrices(monomer, .1).values())
            if hybrid:
                assert not getattr(obj, "R_" + monomer)().np.any()
        else:
            assert np.isfinite(obj.form_unc_amplitude(monomer, .1).np).all()
            if hybrid:
                assert all(np.isfinite(m.np).all() for m in obj.form_aux_matrices(monomer, .1).values())
    assert obj.gpu_contraction_blocks() == 0


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("empty", [True, "A", "B"])
@pytest.mark.parametrize("empty_space", ["occ", "vir"])
def test_fdds_empty_output_budget_and_recovery(fdds_factory, hybrid, empty, empty_space):
    obj = fdds_factory(hybrid, empty=empty, empty_space=empty_space)
    q = obj.metric().np.shape[0]
    for monomer in ("A", "B") if empty is True else (empty,):
        try:
            psi4.core.set_memory_bytes(8)
            with pytest.raises(RuntimeError, match="Too little memory"):
                obj.form_unc_amplitude(monomer, .1)
            with pytest.raises(RuntimeError, match="Too little memory"):
                obj.form_aux_matrices(monomer, .1)
            # Enough for one Q x Q output, but not the five auxiliary outputs.
            psi4.core.set_memory_bytes(int(2 * q * q * 8 / .8))
            assert not obj.form_unc_amplitude(monomer, .1).np.any()
            with pytest.raises(RuntimeError, match="Too little memory"):
                obj.form_aux_matrices(monomer, .1)
        finally:
            psi4.core.set_memory_bytes(500 * 1024**2)
        assert not obj.form_unc_amplitude(monomer, .1).np.any()
        assert all(not m.np.any() for m in obj.form_aux_matrices(monomer, .1).values())
    assert obj.gpu_contraction_blocks() == 0


def _tensor_helper(obj, method, name, dimensions, path):
    if method == "get":
        return obj.get_tensor_pqQ(name, dimensions)
    return obj.print_tensor_pqQ(name, str(path), dimensions)


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("empty_space", ["occ", "vir"])
@pytest.mark.parametrize("method", ["get", "print"])
def test_fdds_all_empty_tensor_helpers(fdds_factory, tmp_path, hybrid, empty_space, method):
    obj = fdds_factory(hybrid, empty=True, empty_space=empty_space)
    o, v = fdds_factory.dimensions["A"]
    path = tmp_path / "empty-tensor.txt"
    with pytest.raises(RuntimeError, match="no DFHelper tensors"):
        _tensor_helper(obj, method, "arQ", (o, v, 10), path)
    assert not path.exists()


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("method", ["get", "print"])
def test_fdds_tensor_helper_preflight(fdds_factory, tmp_path, hybrid, method):
    obj = fdds_factory(hybrid, empty="A")
    path = tmp_path / "tensor.txt"
    with pytest.raises(RuntimeError, match="not found"):
        # Missing tensor must win even over an absurd requested allocation.
        _tensor_helper(obj, method, "arQ", (2**63, 2, 10), path)
    assert not path.exists()
    with pytest.raises(RuntimeError, match="overflow"):
        _tensor_helper(obj, method, "bsQ", (2**63, 2, 10), path)
    with pytest.raises(RuntimeError, match="dimensions"):
        _tensor_helper(obj, method, "bsQ", (3, 6, 10), path)
    with pytest.raises(RuntimeError, match="nonempty"):
        _tensor_helper(obj, method, "bsQ", (0, 7, 10), path)
    with pytest.raises(RuntimeError, match="Too little memory"):
        try:
            psi4.core.set_memory_bytes(8)
            _tensor_helper(obj, method, "bsQ", (3, 7, 10), path)
        finally:
            psi4.core.set_memory_bytes(500 * 1024**2)
    assert not path.exists()
    # Real disk tensors from hybrid preprocessing must remain available.
    name = "QYbsQ" if hybrid else "bsQ"
    got = _tensor_helper(obj, method, name, (3, 7, 10), path)
    if method == "get":
        assert got.np.shape == (3, 7, 10)
        assert np.isfinite(got.np).all()
    else:
        reference = obj.get_tensor_pqQ(name, (3, 7, 10)).np.flatten()
        np.testing.assert_allclose(np.loadtxt(path), reference, rtol=0, atol=1.e-14)


@pytest.mark.parametrize("hybrid", [False, True])
def test_fdds_memory_guard(fdds_factory, hybrid):
    obj = fdds_factory(hybrid)
    psi4.core.set_memory_bytes(8)
    with pytest.raises(RuntimeError, match="Too little memory"):
        _values(obj, hybrid, "A", .1)
    assert obj.gpu_contraction_blocks() == 0
    psi4.core.set_memory_bytes(500 * 1024**2)
    assert all(np.isfinite(v).all() for v in _values(obj, hybrid, "A", .1).values())


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("occupied", [(3, 3), (2, 3)], ids=["equal", "unequal"])
def test_fdds_integrated_native_energy(fdds_factory, hybrid, occupied):
    """CPU-only builds exercise the full ALDA/coupling/quadrature interface."""
    from psi4.driver.procrouting.sapt.sapt_mp2_terms import df_fdds_dispersion
    fdds_factory(occupied=occupied)
    primary, auxiliary, cache = fdds_factory.inputs
    psi4.core.clean_timers()
    start = time.perf_counter()
    result = df_fdds_dispersion(primary, auxiliary, cache, hybrid, .25 if hybrid else 0.,
                                do_print=False)
    assert set(result) == {"Disp20,FDDS (unc)", "Disp20"}
    assert all(np.isfinite(v) and v < 0. for v in result.values())
    assert _timer_calls() == 0
    print(f"FDDS native integrated hybrid={hybrid} occupied={occupied} elapsed={time.perf_counter() - start:.6f}s "
          f"energy={result}")


@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.cuest
@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("occupied", [(3, 3), (2, 3)], ids=["equal", "unequal"])
def test_fdds_gpu_contributions(fdds_factory, hybrid, occupied):
    obj = fdds_factory(hybrid, occupied=occupied)
    q = obj.metric().np.shape[0]
    memory_bytes, blocks = _contribution_budget(q, fdds_factory.dimensions, hybrid)
    psi4.core.set_memory_bytes(memory_bytes)
    frequencies = (.0, .37, 2.)
    ref = {(m, w): _values(obj, hybrid, m, w) for m in ("A", "B") for w in frequencies}
    for enabled, threshold in ((True, 10**9), (False, 0), (True, 0)):
        psi4.set_options({"use_cuest": enabled, "cuest_gemm_min_dim": threshold})
        psi4.core.clean_timers()
        before = obj.gpu_contraction_blocks()
        start = time.perf_counter()
        errors = {k: 0. for k in next(iter(ref.values()))}
        for (m, w), expected in ref.items():
            got = _values(obj, hybrid, m, w)
            for k in expected:
                np.testing.assert_allclose(got[k], expected[k], rtol=0, atol=2.e-11)
                errors[k] = max(errors[k], float(np.max(np.abs(got[k] - expected[k]))))
        elapsed = time.perf_counter() - start
        routed = enabled and threshold == 0
        assert obj.gpu_contraction_blocks() - before == (sum(blocks.values()) * len(frequencies)
                                                        if routed else 0)
        assert _timer_calls() == (6 if routed else 0)
        print(f"FDDS hybrid={hybrid} occupied={occupied} enabled={enabled} threshold={threshold} elapsed={elapsed:.6f}s "
              f"max_abs_errors={errors}")


@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.cuest
def test_fdds_gpu_empty_and_preflight(fdds_factory):
    for hybrid in (False, True):
        for empty in (True, "A", "B"):
            psi4.set_options({"use_cuest": False})
            obj = fdds_factory(hybrid, empty=empty)
            psi4.set_options({"use_cuest": True, "cuest_gemm_min_dim": 0})
            psi4.core.clean_timers()
            for monomer in ("A", "B"):
                if empty is True or empty == monomer:
                    assert all(not m.np.any() for m in obj.form_aux_matrices(monomer, .1).values())
                    assert not obj.form_unc_amplitude(monomer, .1).np.any()
            assert obj.gpu_contraction_blocks() == 0
            assert _timer_calls() == 0
            if empty is not True:
                other = "B" if empty == "A" else "A"
                assert all(np.isfinite(v).all() for v in _values(obj, hybrid, other, .1).values())
                assert obj.gpu_contraction_blocks() > 0
                assert _timer_calls() == 1


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
@pytest.mark.parametrize("occupied", [(3, 3), (2, 3)], ids=["equal", "unequal"])
def test_fdds_integrated_gpu_energy(fdds_factory, hybrid, occupied, monkeypatch):
    """Actual ALDA kernel, hybrid terms, pseudoinverse and default quadrature."""
    from psi4.driver.procrouting.sapt.sapt_mp2_terms import df_fdds_dispersion
    fdds_factory(occupied=occupied)
    primary, auxiliary, cache = fdds_factory.inputs
    results = []
    pinv_reference = None
    original_pinv = np.linalg.pinv
    for enabled, threshold in ((False, 0), (True, 10**9), (True, 0)):
        pinv_inputs = []

        def captured_pinv(a, *args, **kwargs):
            assert not args and kwargs == {"rcond": 1.e-13}
            pinv_inputs.append(a.copy())
            return original_pinv(a, *args, **kwargs)

        monkeypatch.setattr(np.linalg, "pinv", captured_pinv)
        psi4.set_options({"use_cuest": enabled, "cuest_gemm_min_dim": threshold})
        psi4.core.clean_timers()
        start = time.perf_counter()
        result = df_fdds_dispersion(primary, auxiliary, cache, hybrid, .25 if hybrid else 0.,
                                    do_print=False)
        elapsed = time.perf_counter() - start
        assert _timer_calls() == (20 if enabled and threshold == 0 else 0)
        # Hybrid Rtinv is a transpose view: preserve native KRS products rather
        # than silently packing it. Other six chains per point are eligible.
        assert _product_timer_calls() == (60 if enabled and threshold == 0 else 0)
        assert len(pinv_inputs) == (22 if hybrid else 20)
        if pinv_reference is None:
            pinv_reference = pinv_inputs
        else:
            for actual_input, reference_input in zip(pinv_inputs, pinv_reference):
                np.testing.assert_allclose(actual_input, reference_input, rtol=2.e-12, atol=2.e-12)
        assert set(result) == {"Disp20,FDDS (unc)", "Disp20"}
        assert all(np.isfinite(v) for v in result.values())
        results.append(result)
        errors = {k: abs(result[k] - results[0][k]) for k in result}
        timings = {k: v["wall_time"] for k, v in psi4.core.get_timer_dict().items()
                   if any(t in k for t in ("Form FDDS object", "Form xc kernel", "Time Integration",
                                           "cuEST FDDS: contractions"))}
        print(f"FDDS integrated hybrid={hybrid} occupied={occupied} enabled={enabled} threshold={threshold} "
              f"elapsed={elapsed:.6f}s energy={result} abs_errors={errors} timers={timings}")
    for result in results[1:]:
        for k in result:
            assert abs(result[k] - results[0][k]) < 1.e-10


def _run_saptdft_fdds(enabled, threshold, path):
    previous_output = psi4.core.outfile_name()
    # set_num_threads/set_memory print through the global native output stream.
    # Open it before setup, including when a previous caller closed its file.
    psi4.core.set_output_file(str(path), False)
    try:
        psi4.core.clean()
        psi4.core.clean_options()
        psi4.set_num_threads(1)
        psi4.set_memory("500 MiB")
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
        keys = ("SAPT ELST ENERGY", "SAPT EXCH ENERGY", "SAPT IND ENERGY",
                "SAPT DISP ENERGY", "SAPT TOTAL ENERGY", "CURRENT ENERGY")
        result = {k: psi4.variable(k) for k in keys}
        assert all(np.isfinite(v) for v in result.values())
        print(f"SAPT(DFT) enabled={enabled} threshold={threshold} values={result}")
        return result
    finally:
        psi4.core.close_outfile()
        psi4.core.set_output_file(previous_output, True)


@pytest.mark.saptdft
def test_saptdft_fdds_native_xc_cpu(tmp_path):
    path = tmp_path / "sapt-cpu.out"
    _run_saptdft_fdds(False, 0, path)
    assert _timer_calls() == 0
    assert "Disp20" in path.read_text()


@pytest.mark.saptdft
def test_saptdft_fdds_sequential_cpu_output(tmp_path):
    """Repeated backend controls must restore a usable caller output stream."""
    previous = psi4.core.outfile_name()
    results = []
    try:
        for index in range(2):
            path = tmp_path / f"sapt-sequential-{index}.out"
            results.append(_run_saptdft_fdds(False, 0, path))
            assert psi4.core.outfile_name() == previous
            # This native print catches a closed stream, even on CPU-only builds.
            psi4.set_num_threads(1)
            assert "Disp20" in path.read_text()
        for key in results[0]:
            assert abs(results[0][key] - results[1][key]) < 1.e-12
    finally:
        # A failed regression assertion must not leave pytest's native stream closed.
        psi4.core.set_output_file(previous, True)


@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.cuest
@pytest.mark.saptdft
def test_saptdft_fdds_native_xc_gpu(tmp_path):
    """Small full SAPT(DFT), native supported LDA response, no GPU-XC claim."""
    results = []
    for enabled, threshold in ((False, 0), (True, 10**9), (True, 0)):
        path = tmp_path / f"sapt-{enabled}-{threshold}.out"
        result = _run_saptdft_fdds(enabled, threshold, path)
        # Forced FDDS checks the shared handle's actual HOST pointer mode after
        # the real cuEST J/K builds; an incompatible mode raises, not CPU retry.
        assert _timer_calls() == (20 if enabled and threshold == 0 else 0)
        if enabled:
            policy = psi4.core._cuest_runtime_policy()
            # Native FP64 mode1 is not cuBLAS DEFAULT_MATH mode0. Observe the
            # real policy after JK, rather than infer it from the option/header.
            assert policy["native_math_mode"] == 1, policy
            assert policy["cublas_pointer_mode"] == 0, policy
            assert policy["shared_stream"] is True, policy
            assert type(policy["cublas_math_mode"]) is int, policy
            assert psi4.core._cuest_runtime_version() == (0, 2, 50)
            print(f"SAPT(DFT) live_gpu_policy={policy}")
        results.append(result)
        assert "Disp20" in path.read_text()
        print(f"SAPT(DFT) abs_errors={ {k: abs(result[k] - results[0][k]) for k in result} }")
    for result in results[1:]:
        for k in result:
            assert abs(result[k] - results[0][k]) <= 1.e-5
    psi4.set_options({"use_cuest": False})
