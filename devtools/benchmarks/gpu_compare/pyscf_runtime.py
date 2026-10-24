"""Explicit reviewed CUDA12 bootstrap and no-fallback cuTENSOR admission.

No GPU packages are imported here. Bootstrap maps only the independently
approved absolute libcudart path; it calls no device/handle APIs. The existing
passive mapped-file audit remains mandatory before imports and device use.
"""
import ctypes
import os
from pathlib import Path

from cuest_sdk import read_cuda12_receipt, runtime_identity

_runtime_handles = {}


def reject_contraction_override():
    if "CONTRACT_ENGINE" in os.environ:
        raise RuntimeError("CONTRACT_ENGINE overrides are forbidden; native cuTENSOR required")


def bootstrap_pyscf_runtime():
    reject_contraction_override()
    receipt = read_cuda12_receipt()
    selected = receipt["libraries"]["libcudart.so.12"]
    path = selected["path"]
    key = (path, selected["sha256"])
    if key not in _runtime_handles:
        _runtime_handles[key] = ctypes.CDLL(path, mode=os.RTLD_NOW | os.RTLD_LOCAL)
    # No guessed vendor-version ABI and no device initialization. Mapping the
    # exact reviewed DSO makes this audit meaningful even for static CuPy.
    proof = runtime_identity("pyscf")
    return dict(bootstrap="explicit-reviewed-libcudart", library=dict(selected),
                sdk_identity=proof, device_api_called=False)


def audit_cupy_runtime(cp):
    runtime_identity("pyscf")
    compiled = cp.cuda.runtime.runtimeGetVersion()
    shared = cp.cuda.get_local_runtime_version()
    if any(type(value) is not int or value // 1000 != 12 for value in (compiled, shared)):
        raise RuntimeError("CuPy compiled and shared runtimes must both be CUDA12")
    proof = runtime_identity("pyscf")
    return dict(compiled_runtime_version=compiled, shared_runtime_version=shared,
                sdk_identity=proof)


def require_pyscf_cutensor(module):
    reject_contraction_override()
    if (not hasattr(module, "contract_engine") or module.contract_engine is not None or
            getattr(module, "cutensor", None) is None or
            getattr(module, "cutensor_backend", None) is None):
        raise RuntimeError("Native cuTENSOR route required; contraction fallback rejected")
    receipt = read_cuda12_receipt()
    selected = receipt["libraries"].get("libcutensor.so.2")
    if selected is None:
        raise RuntimeError("Reviewed cuTENSOR provider absent from inventory")
    proof = runtime_identity("pyscf")
    path = str(Path(selected["path"]).resolve(strict=True))
    if path not in proof["cuda_linkage_identity"]:
        raise RuntimeError("Reviewed cuTENSOR provider is not actually mapped")
    return dict(verified=True, engine="cutensor", library=dict(selected), sdk_identity=proof)


def contraction_canary(cp, module):
    """Diagnostic-only nonempty FP64 contraction, with native dispatch proof."""
    require_pyscf_cutensor(module)
    a = cp.asarray([[1., 2.], [3., 4.]], dtype=cp.float64)
    b = cp.asarray([[5., 6.], [7., 8.]], dtype=cp.float64)
    backend = module.cutensor_backend
    original = backend.contract
    calls = 0

    def audited(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    backend.contract = audited
    try:
        result = module.contract("ij,jk->ik", a, b)
        cp.cuda.runtime.deviceSynchronize()
        actual = cp.asnumpy(result).tolist()
    finally:
        backend.contract = original
    if calls < 1 or actual != [[19., 22.], [43., 50.]]:
        raise RuntimeError("Nonempty native cuTENSOR contraction gate failed")
    return dict(passed=True, nonempty=True, dtype="float64", native_backend_calls=calls,
                result=actual, route=require_pyscf_cutensor(module))
