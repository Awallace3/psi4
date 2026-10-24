"""CPU-only bootstrap/order/fallback regressions; no native CUDA loads."""
import os
import builtins
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pyscf_runtime as admission


class RuntimeAdmission(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runtime = self.root / "libcudart.so.12.6"
        self.tensor = self.root / "libcutensor.so.2"
        self.runtime.write_bytes(b"synthetic, never loaded")
        self.tensor.write_bytes(b"synthetic, never loaded")
        self.receipt = dict(libraries={
            "libcudart.so.12": dict(path=str(self.runtime), sha256="runtime-pin"),
            "libcutensor.so.2": dict(path=str(self.tensor), sha256="tensor-pin")})
        self.proof = dict(verified=True, cuda_linkage_identity={
            str(self.runtime): {}, str(self.tensor): {}})
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, {}, clear=True).start()
        self.pin = patch.object(admission, "read_cuda12_receipt", return_value=self.receipt).start()
        self.audit = patch.object(admission, "runtime_identity", return_value=self.proof).start()
        self.load = patch.object(admission.ctypes, "CDLL", return_value=object()).start()
        admission._runtime_handles.clear()
        self.addCleanup(admission._runtime_handles.clear)

    def test_only_exact_reviewed_runtime_loaded_before_passive_audit(self):
        events = []
        self.pin.side_effect = lambda: (events.append("pin") or self.receipt)
        self.load.side_effect = lambda *a, **k: (events.append("load") or object())
        self.audit.side_effect = lambda *a, **k: (events.append("audit") or self.proof)
        proof = admission.bootstrap_pyscf_runtime()
        self.assertEqual(events, ["pin", "load", "audit"])
        self.load.assert_called_once_with(
            str(self.runtime), mode=os.RTLD_NOW | os.RTLD_LOCAL)
        self.assertIs(proof["device_api_called"], False)
        self.assertEqual(len(admission._runtime_handles), 1)

    def test_missing_or_changed_pin_prevents_native_load(self):
        self.pin.side_effect = RuntimeError("receipt changed")
        with self.assertRaisesRegex(RuntimeError, "receipt changed"):
            admission.bootstrap_pyscf_runtime()
        self.load.assert_not_called()
        self.audit.assert_not_called()

    def test_mapping_failure_cannot_be_bypassed_by_retained_handle(self):
        admission.bootstrap_pyscf_runtime()
        self.audit.side_effect = RuntimeError("wrong provider")
        with self.assertRaisesRegex(RuntimeError, "wrong provider"):
            admission.bootstrap_pyscf_runtime()
        self.assertEqual(self.load.call_count, 1)
        self.assertEqual(self.audit.call_count, 2)

    def test_every_contraction_override_rejected_before_native_load(self):
        for value in ("cupy", "opt_einsum", "cuquantum", ""):
            with self.subTest(value=value), patch.dict(os.environ, CONTRACT_ENGINE=value):
                with self.assertRaisesRegex(RuntimeError, "CONTRACT_ENGINE"):
                    admission.bootstrap_pyscf_runtime()
        self.load.assert_not_called()
        self.pin.assert_not_called()

    def test_direct_entrypoints_reject_override_before_gpu_import(self):
        import compare
        import probe_eigensolver
        original_import = builtins.__import__

        def no_gpu_import(name, *args, **kwargs):
            if name.split(".")[0] in ("cupy", "gpu4pyscf"):
                raise AssertionError("GPU import preceded override rejection")
            return original_import(name, *args, **kwargs)

        cases = [
            (lambda: compare.main(), ["compare", "device"]),
            (lambda: compare.run_pyscf(None, {}, None, {}), ["worker"]),
            (lambda: probe_eigensolver.main(),
             ["probe", "--config", str(self.root / "config.json"),
              "--xyz", str(self.root / "input.xyz"), "--mode", "native",
              "--output", str(self.root / "probe")]),
        ]
        for function, argv in cases:
            with self.subTest(argv=argv), \
                    patch.dict(os.environ, CONTRACT_ENGINE="opt_einsum", SLURM_JOB_ID="1"), \
                    patch.object(sys, "argv", argv), \
                    patch.object(builtins, "__import__", side_effect=no_gpu_import):
                with self.assertRaisesRegex(RuntimeError, "CONTRACT_ENGINE"):
                    function()
        self.load.assert_not_called()

    def cupy(self, compiled=12080, shared=12060):
        return SimpleNamespace(cuda=SimpleNamespace(
            runtime=SimpleNamespace(runtimeGetVersion=Mock(return_value=compiled)),
            get_local_runtime_version=Mock(return_value=shared), Device=Mock()))

    def test_compiled_and_shared_cuda12_minors_recorded_separately(self):
        cp = self.cupy()
        r = admission.audit_cupy_runtime(cp)
        self.assertEqual(r["compiled_runtime_version"], 12080)
        self.assertEqual(r["shared_runtime_version"], 12060)
        self.assertEqual(self.audit.call_count, 2)
        cp.cuda.Device.assert_not_called()

    def test_wrong_runtime_major_or_noninteger_is_never_accepted(self):
        for compiled, shared in ((13030, 12060), (12080, 13030),
                                 (False, 12060), (12080, "12060")):
            cp = self.cupy(compiled, shared)
            with self.subTest(compiled=compiled, shared=shared):
                with self.assertRaisesRegex(RuntimeError, "both be CUDA12"):
                    admission.audit_cupy_runtime(cp)
                cp.cuda.Device.assert_not_called()

    def test_cupy_version_probe_cannot_precede_mapping_admission(self):
        self.audit.side_effect = RuntimeError("unapproved mapping")
        cp = self.cupy()
        with self.assertRaisesRegex(RuntimeError, "unapproved mapping"):
            admission.audit_cupy_runtime(cp)
        cp.cuda.runtime.runtimeGetVersion.assert_not_called()
        cp.cuda.get_local_runtime_version.assert_not_called()

    def module(self, engine=None, frontend=True):
        return SimpleNamespace(contract_engine=engine,
                               cutensor=object() if frontend else None,
                               cutensor_backend=SimpleNamespace(contract=Mock()))

    def test_all_alternate_engines_and_missing_cutensor_rejected(self):
        for module in (self.module("cupy"), self.module("opt_einsum"),
                       self.module("cuquantum"), self.module("", True),
                       self.module(frontend=False), SimpleNamespace()):
            with self.subTest(module=module):
                with self.assertRaisesRegex(RuntimeError, "fallback rejected"):
                    admission.require_pyscf_cutensor(module)

    def test_reviewed_cutensor_must_be_present_and_actually_mapped(self):
        module = self.module()
        self.audit.return_value = dict(cuda_linkage_identity={str(self.runtime): {}})
        with self.assertRaisesRegex(RuntimeError, "not actually mapped"):
            admission.require_pyscf_cutensor(module)
        del self.receipt["libraries"]["libcutensor.so.2"]
        with self.assertRaisesRegex(RuntimeError, "absent from inventory"):
            admission.require_pyscf_cutensor(module)

    def test_wrong_cutensor_provider_mapping_is_not_accepted(self):
        self.audit.side_effect = RuntimeError("outside pinned CUDA12 closure")
        with self.assertRaisesRegex(RuntimeError, "outside pinned"):
            admission.require_pyscf_cutensor(self.module())

    def canary(self, *, dispatch=True, wrong=False, raises=False):
        class Array:
            def __init__(self, values):
                self.values = values

            def tolist(self):
                return self.values

        cp = SimpleNamespace(asarray=lambda v, **k: Array(v),
                             asnumpy=lambda v: v, float64="float64",
                             cuda=SimpleNamespace(runtime=SimpleNamespace(deviceSynchronize=Mock())))
        module = self.module()

        def native(*args, **kwargs):
            if raises:
                raise RuntimeError("native failure")
            return Array([[0., 0.], [0., 0.]] if wrong else [[19., 22.], [43., 50.]])

        module.cutensor_backend.contract = native
        module.contract = (lambda *a: module.cutensor_backend.contract(*a)) if dispatch else native
        return cp, module, native

    def test_canary_requires_native_dispatch_and_correct_nonempty_result(self):
        for dispatch, wrong in ((True, False), (False, False), (True, True)):
            cp, module, native = self.canary(dispatch=dispatch, wrong=wrong)
            if dispatch and not wrong:
                r = admission.contraction_canary(cp, module)
                self.assertTrue(r["passed"])
                self.assertEqual(r["native_backend_calls"], 1)
            else:
                with self.assertRaisesRegex(RuntimeError, "contraction gate failed"):
                    admission.contraction_canary(cp, module)
            self.assertIs(module.cutensor_backend.contract, native)
            cp.cuda.runtime.deviceSynchronize.assert_called_once_with()

    def test_native_hook_restored_on_contraction_failure(self):
        cp, module, native = self.canary(raises=True)
        with self.assertRaisesRegex(RuntimeError, "native failure"):
            admission.contraction_canary(cp, module)
        self.assertIs(module.cutensor_backend.contract, native)


if __name__ == "__main__":
    unittest.main()
