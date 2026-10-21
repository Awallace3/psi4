"""CPU equation/transport fixtures ONLY: no CuPy/CUDA/physical gate claim."""
import ctypes
import importlib
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

SAPT = Path(__file__).resolve().parents[2] / "psi4/driver/procrouting/sapt"
PACKAGE = "_disp0_cupy_fixtures"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(SAPT)]
sys.modules[PACKAGE] = package
kernel = importlib.import_module(f"{PACKAGE}.sapt_dispersion_cupy")
dfmo = importlib.import_module(f"{PACKAGE}.sapt_cuest_dfmo")


class Array(np.ndarray):
    @property
    def data(self):
        return types.SimpleNamespace(ptr=np.asarray(self).ctypes.data)

    @property
    def device(self):
        return types.SimpleNamespace(id=0)


class FakeCP:
    """Clearly CPU-only stand-in, never admitted by production imports."""
    float64, uint8 = np.float64, np.uint8
    ndarray = Array
    sum = staticmethod(np.sum)

    def __init__(self):
        self.syncs = 0
        self.device = 0
        self.version = 13030
        self.fail_syncs = set()
        self.used, self.reserved, self.free = 512, 2048, 10**10
        runtime = types.SimpleNamespace(
            is_hip=False, getDeviceCount=lambda: 1, getDevice=lambda: self.device,
            runtimeGetVersion=lambda: self.version, deviceSynchronize=self.synchronize,
            memGetInfo=lambda: (self.free, 10**10))
        self.cuda = types.SimpleNamespace(runtime=runtime, get_current_stream=lambda: self)

    def synchronize(self):
        self.syncs += 1
        if self.syncs in self.fail_syncs:
            raise RuntimeError("fixture synchronization failure")

    @staticmethod
    def asarray(a, **kw):
        return np.asarray(a, **kw).view(Array)

    @staticmethod
    def concatenate(arrays, **kw):
        return np.concatenate(arrays, **kw).view(Array)

    @staticmethod
    def empty(*a, **kw):
        return np.empty(*a, **kw).view(Array)

    @staticmethod
    def zeros(*a, **kw):
        return np.zeros(*a, **kw).view(Array)

    @staticmethod
    def zeros_like(a):
        return np.zeros_like(a).view(Array)

    @staticmethod
    def asnumpy(a):
        return np.array(a)

    def get_default_memory_pool(self):
        return types.SimpleNamespace(used_bytes=lambda: self.used,
                                     total_bytes=lambda: self.reserved)


class FakeCE:
    CuestStatus = types.SimpleNamespace(CUEST_STATUS_SUCCESS=0)
    CuestParametersType = types.SimpleNamespace(CUEST_DFMOINTEGRALSCOMPUTE_PARAMETERS=1)
    cuestDFMOIntegralsComputeParameters = staticmethod(object)
    Pointer = staticmethod(int)

    def __init__(self, tensor, order="C"):
        self.tensor, self.order = tensor, order
        self.created = self.destroyed = self.computed = 0
        self.query_code = 0
        self.compute_code = self.destroy_code = 0

    def cuestParametersCreate(self, **kw):
        self.created += 1
        return 0

    def cuestParametersDestroy(self, **kw):
        self.destroyed += 1
        return self.destroy_code

    def cuestDFMOIntegralsComputeWorkspaceQuery(self, **kw):
        descriptor = dfmo._Descriptor.from_address(kw["temporaryWorkspaceDescriptor"])
        descriptor.host_bytes, descriptor.device_bytes = 128, 256
        return self.query_code

    def cuestDFMOIntegralsCompute(self, **kw):
        if self.compute_code:
            return self.compute_code
        nao = self.tensor.shape[1]
        nl, nr = kw["numLeftOrbitals"], kw["numRightOrbitals"]

        def view(pointer, count):
            return np.ctypeslib.as_array((ctypes.c_double * count).from_address(pointer))

        left = view(kw["leftCoefficientMatrices"], sum(nl) * nao).reshape(-1, nao)
        right = view(kw["rightCoefficientMatrices"], sum(nr) * nao).reshape(-1, nao)
        out = view(kw["outTensors"], self.tensor.shape[0] * sum(l * r for l, r in zip(nl, nr)))
        lo = ro = offset = 0
        for l, r in zip(nl, nr):
            factor = np.einsum("Qmn,lm,rn->Qlr", self.tensor,
                               left[lo:lo+l], right[ro:ro+r])
            out[offset:offset+factor.size] = factor.ravel(order=self.order)
            lo += l
            ro += r
            offset += factor.size
        self.computed += 1
        return 0


def inputs(seed=19):
    rng = np.random.default_rng(seed)
    nao, q, a, r, b, s = 7, 9, 2, 5, 3, 4
    sizes = {"a": a, "r": r, "b": b, "s": s}
    spaces = {name: rng.normal(size=(nao, sizes[name[0]]))
              for name in ("a", "r", "b", "s", "r1", "s1", "a2", "b2", "r3", "s3", "a4", "b4")}
    corrections = {name: rng.normal(size=shape) for name, shape in {
        "Qar": (a,r), "SBar": (a,r), "Qbr": (b,r), "Sbr": (b,r),
        "SAbs": (b,s), "Qbs": (b,s), "Sas": (a,s), "Qas": (a,s)}.items()}
    energies = {"a": -rng.uniform(.7, 1.2, a), "b": -rng.uniform(.8, 1.3, b),
                "r": rng.uniform(.1, .6, r), "s": rng.uniform(.1, .7, s)}
    return rng.normal(size=(q, nao, nao)), spaces, corrections, energies


def oracle(ao, spaces, c, e):
    """Independent scalar loops, no packing/reshape/cross-permutation reuse."""
    def f(left, right):
        return np.einsum("Qmn,mi,nj->Qij", ao, spaces[left], spaces[right])
    ar, bs = f("r","a"), f("s","b")
    fr, fs = f("r3","a")+f("r","a4"), f("s3","b")+f("s","b4")
    bas, cas, bbr, cbr = f("s1","a"), f("s","a2"), f("r1","b"), f("r","b2")
    d = np.zeros((e["a"].size,e["b"].size)); x = np.zeros_like(d)
    for a in range(d.shape[0]):
        for b in range(d.shape[1]):
            for r in range(e["r"].size):
                for s in range(e["s"].size):
                    v = ar[:,r,a] @ bs[:,s,b]
                    t = v/(e["a"][a]+e["b"][b]-e["r"][r]-e["s"][s])
                    direct = (ar[:,r,a] @ fs[:,s,b]+fr[:,r,a] @ bs[:,s,b]
                              +c["Qar"][a,r]*c["SAbs"][b,s]+c["SBar"][a,r]*c["Qbs"][b,s])
                    cross = (bas[:,s,a] @ bbr[:,r,b]+cas[:,s,a] @ cbr[:,r,b]
                             +c["Sas"][a,s]*c["Qbr"][b,r]+c["Qas"][a,s]*c["Sbr"][b,r])
                    d[a,b] += 4*t*v
                    x[a,b] -= 2*t*(direct+cross)
    return d, x


class CuPyContracts(unittest.TestCase):
    def test_block_equations_edges_changed_inputs_and_component_sums(self):
        # Declared before comparisons; no tolerance tuning to results.
        rtol, atol = 1e-11, 1e-12
        for seed in (19, 20):
            ao, spaces, corrections, energies = inputs(seed)
            expected = oracle(ao, spaces, corrections, energies)
            for block in (1, 3, 500):
                cp = FakeCP()
                class Producer:
                    nao, naux = ao.shape[1], ao.shape[0]
                    def check_memory(self, n):
                        self.request = n
                    def compute(self, pairs):
                        return {name: np.einsum("Qmn,mi,nj->Qij", ao, cl, cr).view(Array)
                                for name,(cl,cr) in pairs.items()}
                    def report(self):
                        return {"scope": "CPU TEST DOUBLE"}
                producer = Producer(); producer.cp = cp
                with patch.object(kernel, "require_cupy", return_value=cp):
                    result = kernel.blocked_dispersion(producer, spaces, corrections,
                                                       energies, block_size=block)
                for label, pair_label, reference in (
                    ("Disp20,u","Disp20 occupied pairs",expected[0]),
                    ("Exch-Disp20,u","Exch-Disp20 occupied pairs",expected[1])):
                    np.testing.assert_allclose(result[pair_label], reference, rtol=rtol, atol=atol)
                    np.testing.assert_allclose(result[label], reference.sum(), rtol=rtol, atol=atol)
                self.assertFalse(result["Disp0 execution"]["actual_gpu_execution_verified"])

    def make_provider(self, order="C"):
        ao, spaces, _, _ = inputs()
        cp, ce = FakeCP(), FakeCE(ao, order)
        primary = types.SimpleNamespace(nbf=lambda: ao.shape[1])
        auxiliary = types.SimpleNamespace(nbf=lambda: ao.shape[0])
        with patch.object(dfmo, "require_cupy", return_value=cp), patch.dict(
                sys.modules, {"cuest": types.ModuleType("cuest"), "cuest.bindings": ce}):
            provider = dfmo.CuESTDFMOProvider(
                handle=object(), plan=object(), primary_basis=primary, auxiliary_basis=auxiliary,
                keepalive=(object(),), output_order=order, memory_limit_bytes=10**8,
                native_persistent_bytes=4096, operator="coulomb", metric_power=-0.5)
        return provider, ce, cp, ao, spaces

    def test_real_pointer_workspace_and_batched_output_contract_cpu_fixture(self):
        for order in ("C", "F"):
            p, ce, cp, ao, spaces = self.make_provider(order)
            # Includes strided AO coefficient slabs and nonsquare/edge outputs.
            pairs = {"x": (spaces["r"][:,1:4], spaces["a"]),
                     "y": (spaces["s"][:,::2], spaces["b"])}
            result = p.compute(pairs)
            for name,(cl,cr) in pairs.items():
                np.testing.assert_allclose(result[name], np.einsum("Qmn,mi,nj->Qij",ao,cl,cr),
                                           rtol=1e-12,atol=1e-12)
            self.assertEqual(ce.created, ce.destroyed)
            self.assertEqual(ce.computed, 1)
            self.assertGreaterEqual(cp.syncs, 2)
            self.assertTrue(np.shares_memory(result["x"], result["x"].base))
            p.validate_basis(p.primary_basis,p.auxiliary_basis)
            with self.assertRaises(ValueError):
                p.validate_basis(object(),p.auxiliary_basis)
            p.close()
            with self.assertRaises(RuntimeError):
                p.compute(pairs)

    def test_native_failure_does_not_fallback_and_releases_parameters(self):
        p, ce, cp, _, spaces = self.make_provider()
        ce.query_code = 7
        with self.assertRaisesRegex(RuntimeError, "workspace query"):
            p.compute({"x": (spaces["r"],spaces["a"])})
        self.assertEqual(ce.destroyed, 1)
        self.assertEqual(ce.computed, 0)
        self.assertGreaterEqual(cp.syncs, 2)
        with self.assertRaisesRegex(RuntimeError, "invalidated"):
            p.compute({"x": (spaces["r"],spaces["a"])})

    def test_primary_error_survives_sync_and_destroy_cleanup_failures(self):
        p, ce, cp, _, spaces = self.make_provider()
        ce.compute_code, ce.destroy_code = 9, 8
        cp.fail_syncs = {2}
        with self.assertRaisesRegex(RuntimeError, "compute failed: 9") as caught:
            p.compute({"x":(spaces["r"],spaces["a"])})
        self.assertEqual(ce.destroyed, 1)
        self.assertEqual(len(caught.exception.__notes__), 2)
        self.assertTrue(p.failed)

    def test_cleanup_failure_without_primary_raises_and_context_keeps_body_error(self):
        p, ce, cp, _, spaces = self.make_provider()
        cp.fail_syncs = {2}
        with self.assertRaisesRegex(RuntimeError, "synchronization failure"):
            p.compute({"x":(spaces["r"],spaces["a"])})
        self.assertEqual(ce.destroyed, 1)
        p, ce, cp, _, _ = self.make_provider()
        cp.fail_syncs = {1}
        with self.assertRaisesRegex(ValueError, "original body") as caught:
            with p:
                raise ValueError("original body")
        self.assertIn("context cleanup", caught.exception.__notes__[0])
        self.assertTrue(p.closed)

    def test_cleanup_failure_inside_unrelated_except_is_not_swallowed(self):
        for failure in ("sync","destroy"):
            p, ce, cp, _, spaces = self.make_provider()
            if failure=="sync":
                cp.fail_syncs={2}
            else:
                ce.destroy_code=8
            try:
                raise ValueError("unrelated caller exception")
            except ValueError as outer:
                with self.subTest(failure=failure),self.assertRaises(RuntimeError):
                    p.compute({"x":(spaces["r"],spaces["a"])})
                self.assertFalse(hasattr(outer,"__notes__"))
            self.assertTrue(p.failed)

    def test_live_buffer_estimate_accounts_native_and_reusable_pool(self):
        p, _, cp, _, _ = self.make_provider()
        cp.used, cp.reserved, cp.free = 1000, 4096, 2000
        p.memory_limit = 10000
        p.check_memory(3000)  # Reuses cached CuPy memory, not just free VRAM.
        with self.assertRaises(MemoryError):
            p.check_memory(6000)  # Declared native allocations count too.
        self.assertEqual(p.report()["declared_all_native_persistent_bytes"],4096)
        self.assertEqual(p.report()["pool_reserved_bytes"],4096)

    def test_device_and_budget_fail_before_compute(self):
        p, ce, cp, _, spaces = self.make_provider()
        cp.device = 1
        with self.assertRaises(RuntimeError):
            p.compute({"x":(spaces["r"],spaces["a"])})
        cp.device = 0; p.memory_limit = 1
        with self.assertRaises(MemoryError):
            p.compute({"x":(spaces["r"],spaces["a"])})
        self.assertEqual(ce.computed, 0)

    def test_bad_block_size_missing_corrections_and_gaps_fail(self):
        ao, spaces, c, e = inputs()
        with self.assertRaises(ValueError):
            kernel._validate_inputs(spaces, {}, e)
        e["r"][:] = -3
        with self.assertRaises(ValueError):
            kernel._validate_inputs(spaces,c,e)
        for block in (0,-1,True,1.5):
            with self.assertRaises(ValueError):
                kernel.blocked_dispersion(None,spaces,c,e,block_size=block)

    def test_missing_optional_dependency_and_host_factor_fallback_reject(self):
        with patch.dict(sys.modules, {"cupy": None}):
            with self.assertRaisesRegex(RuntimeError, "optional CuPy"):
                kernel.require_cupy()
        cp = FakeCP()
        pairs = {"x": (np.ones((7,2)),np.ones((7,3)))}
        with self.assertRaisesRegex(ValueError, "FP64 CUDA"):
            kernel._validate_factors(cp, {"x":np.ones((9,2,3))}, pairs, 9)


if __name__ == "__main__":
    unittest.main()
