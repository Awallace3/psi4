"""CPU fake-C-ABI/device-buffer tests; no cuEST or CUDA execution proof."""
import ctypes as ct
import importlib
import types
import unittest
from unittest.mock import patch

import numpy as np

from test_cuda_backend import PACKAGE, Kernels, host, runtime, inputs, oracle, backend
import test_cuda_dfmo_transport as transport

setup = importlib.import_module(PACKAGE+".sapt_dispersion_cuda_setup")
dfmo = transport.dfmo


class DeviceSession(transport.Session):
    kernels = None
    scope = runtime.CUDASession.scope
    retain_native_owner = runtime.CUDASession.retain_native_owner
    release_native_owner = runtime.CUDASession.release_native_owner

    def __init__(self):
        super().__init__()
        self.kernels = type(self).kernels
        self.native_owners = []
        self.scope_depth = 0
        self.device = 0
        self.runtime = types.SimpleNamespace(cudaGetDevice=lambda:(0,0))
        self.rt = runtime.CUDASession.rt

    def synchronize_device(self):
        try:
            super().synchronize_device()
        except BaseException:
            self.failed = True
            raise


class DeviceTransport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Kernels.setUpClass.__func__(cls)
        cls.lib.matrix_window.argtypes = [ct.c_longlong]*4+[ct.c_int]+[ct.c_void_p]*2
        cls.lib.matrix_window.restype = None
        DeviceSession.kernels = cls.lib

    @classmethod
    def tearDownClass(cls):
        DeviceSession.kernels = None
        Kernels.tearDownClass.__func__(cls)

    def fixture(self, **changes):
        with patch.object(transport,"Session",DeviceSession):
            p,s,n,ao,sp = transport.Transport().fixture(**changes)
        audit = patch.object(dfmo.identity,"audit_maps",return_value={
            "cuda_mapped_sha256":{s.blas_path:s.blas_sha256}})
        audit.start()
        self.addCleanup(audit.stop)
        return p,s,n,ao,sp

    def test_device_coefficient_pack_layout_batches_changes_and_no_host_upload(self):
        for repeat in range(2):
            p,s,n,ao,sp = self.fixture()
            p.prepare_device_compute()
            matrices = {"x":(sp["r"][:,repeat:repeat+3],sp["a"]),
                        "y":(sp["s"][:,::2],sp["b"])}
            pairs = {name:tuple(s.upload(v) for v in values) for name,values in matrices.items()}
            # Compute may not stage any coefficient or factor through the host.
            s.upload = lambda *_:self.fail("Unexpected DFMO host upload")
            s.download = lambda *_:self.fail("Unexpected DFMO host download")
            with s.scope():
                factors = p.compute_device(pairs)
                for name,(left,right) in matrices.items():
                    np.testing.assert_allclose(host(factors[name]),
                        np.einsum("Qmn,mi,nj->Qij",ao,left,right),rtol=1e-12,atol=1e-12)
                self.assertIs(factors["x"].allocation,factors["y"].allocation)
                self.assertEqual(p._device_pending,[])
                self.assertEqual(p.input_bytes,0)
                self.assertGreater(p.device_input_bytes,0)
                self.assertEqual((n.created,n.destroyed,n.computed),(1,0,1))
            p.close()
            self.assertEqual((n.created,n.destroyed),(1,1))
            self.assertTrue(p.cleanup_complete)
            self.assertEqual(s.native_owners,[])

    def test_prepare_requires_outer_lifetime_and_native_parameter_owner_is_retained(self):
        p,s,n,_,_ = self.fixture()
        with s.scope(),self.assertRaisesRegex(ValueError,"outside temporary"):
            p.prepare_device_compute()
        self.assertEqual(n.created,0)
        p.prepare_device_compute()
        self.assertEqual(s.native_owners,[p])
        with self.assertRaisesRegex(RuntimeError,"already prepared"):
            p.prepare_device_compute()
        p.close()

    def test_native_error_retains_host_descriptors_until_teardown_and_disables_retry(self):
        p,s,n,_,sp = self.fixture()
        p.prepare_device_compute()
        pairs = {"x":(s.upload(sp["r"]),s.upload(sp["a"]))}
        n.compute_code = 9
        with self.assertRaisesRegex(RuntimeError,"device compute failed: 9"):
            p.compute_device(pairs)
        pending = p._device_pending[0]
        self.assertEqual(pending["host"].size,128)
        self.assertEqual(n.destroyed,0)
        with self.assertRaisesRegex(RuntimeError,"prepared live"):
            p.compute_device(pairs)
        p.close()
        self.assertEqual(n.destroyed,1)
        self.assertEqual(p._device_pending,[])

    def test_failed_sync_retains_temporary_device_and_host_buffers_for_cleanup_only(self):
        p,s,n,_,sp = self.fixture()
        p.prepare_device_compute()
        pairs = {"x":(s.upload(sp["r"]),s.upload(sp["a"]))}
        s.fail_syncs = {3,4}  # after compute, then scope finalization
        with self.assertRaisesRegex(RuntimeError,"fixture sync failure"):
            p.compute_device(pairs)
        pending = p._device_pending[0]
        self.assertFalse(pending["packed"][0].allocation.freed)
        self.assertFalse(pending["workspace"][0].allocation.freed)
        self.assertEqual(pending["host"].size,128)
        self.assertEqual(s.native_owners,[p])
        p.close()
        self.assertTrue(p.cleanup_complete)
        self.assertEqual(n.destroyed,1)
        self.assertEqual(p._device_pending,[])
        # Session still owns deferred allocations, and execution cannot resume.
        self.assertTrue(s.allocations)
        self.assertTrue(s.failed)

    def test_ambiguous_native_create_and_destroy_never_blindly_retry(self):
        for operation in ("create","destroy"):
            p,s,n,_,_ = self.fixture()
            if operation == "create":
                def ambiguous(kind,out):
                    out._obj.value = 1234
                    return 7
                n.cuestParametersCreate.side_effect = ambiguous
                with self.assertRaisesRegex(RuntimeError,"create failed"):
                    p.prepare_device_compute()
            else:
                p.prepare_device_compute()
                n.destroy_code = 8
                with self.assertRaisesRegex(RuntimeError,"destroy failed"):
                    p.close()
            calls = n.cuestParametersDestroy.call_count
            with self.assertRaisesRegex(RuntimeError,"ownership quarantined"):
                p.close()
            self.assertEqual(n.cuestParametersDestroy.call_count,calls)
            self.assertEqual(s.native_owners,[p])
            self.assertFalse(p.cleanup_complete)

    def test_device_slabs_match_canonical_pairs_without_host_slicing(self):
        p,s,_,_,sp = self.fixture()
        device = {name:s.upload(value) for name,value in sp.items()}
        common = importlib.import_module(PACKAGE+".sapt_dispersion_device_common")
        for side in ("r","s"):
            actual = setup.coefficient_pairs(s,device,side,1,2)
            expected = common.coefficient_pairs(sp,side,slice(1,3))
            for name,pair in actual.items():
                for value,target in zip(pair,expected[name]):
                    np.testing.assert_array_equal(host(value),target)
        with self.assertRaises(ValueError):
            setup.coefficient_pairs(s,device,"bad",0,1)
        p.close()

    def test_device_layout_and_cross_session_reject_before_compute(self):
        p,s,n,_,sp = self.fixture()
        p.prepare_device_compute()
        value = s.upload(sp["r"])
        for left in (value.view(value.shape,order="F"),DeviceSession().upload(sp["r"])):
            with self.assertRaises(ValueError):
                p.compute_device({"x":(left,s.upload(sp["a"]))})
        self.assertEqual(n.computed,0)
        self.assertFalse(p.failed)
        p.close()

    def test_later_native_lease_blocks_parameter_destruction_until_released(self):
        p,s,n,_,_ = self.fixture()
        p.prepare_device_compute()
        child=types.SimpleNamespace(cleanup_complete=False,close=lambda:None)
        s.retain_native_owner(child)
        with self.assertRaisesRegex(RuntimeError,"Later native leases"):
            p.close()
        self.assertEqual(n.destroyed,0)
        self.assertEqual(s.native_owners,[p,child])
        child.cleanup_complete=True
        s.release_native_owner(child)
        p.close()
        self.assertEqual(n.destroyed,1)

    def test_cleanup_process_device_and_mapped_closure_checks_precede_destructor(self):
        for failure in ("pid","device","maps"):
            p,s,n,_,_ = self.fixture()
            p.prepare_device_compute()
            if failure == "pid":
                p.pid += 1
            elif failure == "device":
                s.runtime.cudaGetDevice = lambda:(0,1)
            else:
                # Nested patch overrides the fixture's normal admitted map.
                guard = patch.object(dfmo.identity,"audit_maps",
                                     side_effect=RuntimeError("unadmitted mapping"))
                guard.start()
                self.addCleanup(guard.stop)
            with self.assertRaises(RuntimeError):
                p.close()
            self.assertEqual(n.destroyed,0)
            self.assertFalse(p.cleanup_complete)
            self.assertEqual(s.native_owners,[p])

    def test_device_block_loop_signed_pairs_globals_blocks_and_changed_inputs(self):
        p,s,n,_,_ = self.fixture()
        p.prepare_device_compute()
        for seed in (19,20):
            ao,sp,c,e = inputs(seed)
            n.ao = ao
            expected = oracle(ao,sp,c,e)
            for block in (1,3,5,500):
                with s.scope():
                    dsp = {name:s.upload(v) for name,v in sp.items()}
                    dc = {name:s.upload(v) for name,v in c.items()}
                    de = {name:s.upload(v) for name,v in e.items()}
                    with patch.object(s,"upload",side_effect=AssertionError("host numerical staging")):
                        result = backend.blocked_device_dispersion(p,dsp,dc,de,block_size=block)
                    for field,target in (("Disp20 occupied pairs",expected[0]),
                                         ("Exch-Disp20 occupied pairs",expected[1]),
                                         ("Disp0 occupied pairs",expected[0]+expected[1])):
                        np.testing.assert_allclose(result[field],target,rtol=1e-11,atol=1e-12)
                    np.testing.assert_allclose(
                        [result["Disp20,u"],result["Exch-Disp20,u"],result["Disp0,u"]],
                        [expected[0].sum(),expected[1].sum(),expected[0].sum()+expected[1].sum()],
                        rtol=1e-11,atol=1e-12)
                    self.assertFalse(result["Disp0 execution"]["actual_gpu_execution_verified"])
                self.assertEqual(s.live_bytes,0)
        self.assertEqual(p.input_bytes,0)
        self.assertGreater(n.computed,0)
        p.close()


if __name__ == "__main__":
    unittest.main()
