"""cuBLAS matrix-chain multiplication (``core.cuest_chain_gemm``).

The SAPT tensor code uses ``chain_gemm_einsums`` for left-associated matrix
products. Its eligible chains dispatch to ``cuest_chain_gemm``
(psi4/src/cuest_gemm.cc), which uploads each operand once, keeps intermediate
products on the device, and downloads only requested products.

The delicate part is that Psi4 stores matrices row-major while cuBLAS reads
column-major, which the implementation handles by reversing the operand order
rather than transposing anything.  These tests pin that convention against NumPy
for every combination of transpose flags, because getting it wrong produces a
matrix of the right shape filled with the wrong numbers.
"""

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

import psi4

from addons import uusing

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]


def _reference(arrays, transposes, prefactors_AB):
    """Chain the products on the host, with ``chain_gemm_einsums`` semantics.

    Only the first left operand honours its transpose flag; every later left
    operand is an intermediate that is already oriented.
    """
    products = []
    A = arrays[0].T if transposes[0] == "T" else arrays[0]
    for i, B in enumerate(arrays[1:]):
        if transposes[i + 1] == "T":
            B = B.T
        A = prefactors_AB[i] * (A @ B)
        products.append(A)
    return products


def _run(arrays, transposes, prefactors_AB=None, return_tensors=None):
    n_links = len(arrays) - 1
    if prefactors_AB is None:
        prefactors_AB = [1.0] * n_links
    if return_tensors is None:
        return_tensors = [False] * (n_links - 1) + [True]
    tensors = [psi4.core.Matrix.from_array(a) for a in arrays]
    out = psi4.core.cuest_chain_gemm(tensors, transposes, prefactors_AB, return_tensors)
    return [np.array(m) for m in out]


@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.cuest
@pytest.mark.parametrize("t1", ["N", "T"])
@pytest.mark.parametrize("t2", ["N", "T"])
def test_cuest_chain_gemm_transposes(t1, t2):
    """Every transpose combination must match NumPy, on non-square operands."""
    rng = np.random.default_rng(20260905)
    # Deliberately all different, so a swapped dimension cannot pass silently.
    m, k, n = 7, 5, 3
    A = rng.standard_normal((k, m) if t1 == "T" else (m, k))
    B = rng.standard_normal((n, k) if t2 == "T" else (k, n))

    (out,) = _run([A, B], [t1, t2])
    ref = _reference([A, B], [t1, t2], [1.0])[-1]

    assert out.shape == (m, n)
    assert np.allclose(out, ref, atol=1.0e-13), f"{t1}{t2} product disagrees with NumPy"


@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.cuest
def test_cuest_chain_gemm_multilink_and_prefactors():
    """A four-tensor chain, each link scaled, with a transposed head."""
    rng = np.random.default_rng(11)
    A = rng.standard_normal((6, 9))  # used transposed: 9 x 6
    B = rng.standard_normal((6, 4))
    C = rng.standard_normal((5, 4))  # used transposed: 4 x 5
    D = rng.standard_normal((5, 8))
    transposes = ["T", "N", "T", "N"]
    prefactors = [2.0, -0.5, 3.0]

    (out,) = _run([A, B, C, D], transposes, prefactors)
    ref = _reference([A, B, C, D], transposes, prefactors)[-1]

    assert out.shape == (9, 8)
    assert np.allclose(out, ref, atol=1.0e-12)


@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.cuest
def test_cuest_chain_gemm_returns_requested_intermediates():
    """The flags select which products come back, in chain order."""
    rng = np.random.default_rng(3)
    arrays = [rng.standard_normal((4, 4)) for _ in range(4)]
    transposes = ["N"] * 4
    prefactors = [1.0] * 3
    ref = _reference(arrays, transposes, prefactors)

    out = _run(arrays, transposes, prefactors, return_tensors=[True, False, True])

    assert len(out) == 2
    assert np.allclose(out[0], ref[0], atol=1.0e-12)
    assert np.allclose(out[1], ref[2], atol=1.0e-12)


@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.cuest
def test_cuest_chain_gemm_rejects_nonconformable():
    """A bad chain is caught on the host, before any device allocation."""
    A = psi4.core.Matrix.from_array(np.zeros((3, 4)))
    B = psi4.core.Matrix.from_array(np.zeros((5, 6)))
    with pytest.raises(RuntimeError, match="conformable"):
        psi4.core.cuest_chain_gemm([A, B], ["N", "N"], [1.0], [True])


def _sapt_chain_module():
    pytest.importorskip("einsums")
    return importlib.import_module("psi4.driver.procrouting.sapt.sapt_jk_terms_ein")


@pytest.mark.parametrize("t1", ["N", "T"])
@pytest.mark.parametrize("t2", ["N", "T"])
def test_sapt_chain_work_transposes(t1, t2):
    """The threshold uses the largest link's m*k*n, not a matrix-size bound."""
    module = _sapt_chain_module()
    m, k, n = 7, 5, 3
    tensors = [
        SimpleNamespace(shape=(k, m) if t1 == "T" else (m, k)),
        SimpleNamespace(shape=(n, k) if t2 == "T" else (k, n)),
        SimpleNamespace(shape=(11, n)),
    ]
    assert module._chain_gemm_work(tensors, [t1, t2, "T"]) == max(m * k * n, m * n * 11)


@pytest.mark.parametrize(
    "available,use_cuest,min_dim,beta,expected",
    [
        (True, True, 10, 0.0, True),
        (True, True, 11, 0.0, False),
        (True, True, 0, 0.0, True),
        (True, True, 0, 1.0, False),
        (True, False, 0, 0.0, False),
        (False, True, 0, 0.0, False),
    ],
)
def test_sapt_chain_gpu_eligibility(available, use_cuest, min_dim, beta, expected, monkeypatch):
    module = _sapt_chain_module()
    options = {"USE_CUEST": use_cuest, "CUEST_GEMM_MIN_DIM": min_dim}
    monkeypatch.setattr(module, "_HAVE_CUEST_GEMM", available)
    monkeypatch.setattr(module.core, "get_global_option", lambda name: options[name])
    tensors = [SimpleNamespace(shape=(10, 10)), SimpleNamespace(shape=(10, 10))]
    assert module._use_cuest_gemm(tensors, ["N", "N"], [beta]) is expected


@pytest.mark.parametrize("return_flags", [None, [False, False, False], [True, False, True], [False, True]])
def test_sapt_chain_dispatch_return_contract(return_flags, monkeypatch):
    """Exercise GPU argument/return dispatch without requiring CUDA hardware."""
    module = _sapt_chain_module()
    rng = np.random.default_rng(11)
    arrays = [
        rng.standard_normal((6, 9)),
        rng.standard_normal((6, 4)),
        rng.standard_normal((5, 4)),
        rng.standard_normal((5, 8)),
    ]
    tensors = [psi4.core.Matrix.from_array(a) for a in arrays]
    transposes = ["T", "N", "T", "N"]
    prefactors = [2.0, -0.5, 3.0]
    calls = []

    def gpu_chain(inputs, operations, scales, flags):
        calls.append(flags)
        products = _reference([a.np for a in inputs], operations, scales)
        return [psi4.core.Matrix.from_array(p) for p, requested in zip(products, flags) if requested]

    monkeypatch.setattr(module.core, "cuest_chain_gemm", gpu_chain, raising=False)
    monkeypatch.setattr(module, "_use_cuest_gemm", lambda *args: True)
    gpu = module.chain_gemm_einsums(tensors, transposes, prefactors_AB=prefactors, return_tensors=return_flags)
    monkeypatch.setattr(module, "_use_cuest_gemm", lambda *args: False)
    cpu = module.chain_gemm_einsums(tensors, transposes, prefactors_AB=prefactors, return_tensors=return_flags)

    expected_flags = [False, False, True] if return_flags is None else return_flags + [False] * (3 - len(return_flags))
    assert calls == [expected_flags]
    if return_flags is None:
        gpu, cpu = [gpu], [cpu]
    assert len(gpu) == len(cpu)
    for actual, reference in zip(gpu, cpu):
        np.testing.assert_allclose(actual.np, reference.np, atol=1.e-12, rtol=0)
