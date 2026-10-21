"""CPU ownership spies; no CUDA or native cuEST calls."""
import ctypes as ct
import importlib
import types
import unittest

from test_cuda_backend import PACKAGE

runtime = importlib.import_module(PACKAGE+".sapt_cuda_runtime")


class Owner:
    def __init__(self,session,events,fail=False,name="native close"):
        self.session,self.events,self.fail = session,events,fail
        self.name = name
        self.cleanup_complete = False
    def close(self):
        self.events.append(self.name)
        if self.fail: raise RuntimeError("native destruction failed")
        self.cleanup_complete = True
        self.session.release_native_owner(self)


class NativeOwners(unittest.TestCase):
    def fixture(self):
        session = runtime.CUDASession.__new__(runtime.CUDASession)
        events = []
        def action(label):
            def call(*args):
                events.append(label)
                return (0,)
            return call
        session.runtime = types.SimpleNamespace(
            cudaDeviceSynchronize=action("synchronize"),cudaFree=action("free"),
            cudaStreamDestroy=action("stream destroy"),cudaGetDevice=lambda:(0,0))
        session.driver = types.SimpleNamespace(cuModuleUnload=action("module destroy"))
        session.failed = session.closed = session.cleanup_complete = False
        session.scope_depth = 0
        session.native_owners = []
        session.device = 0
        session.live_bytes = 16
        session.allocations = [runtime._Allocation(session,123,16)]
        session.handle,session.module,session.stream = ct.c_void_p(456),789,7
        session._blas = lambda *args:events.append("BLAS destroy")
        return session,events

    def test_native_destruction_precedes_all_session_resource_release(self):
        session,events = self.fixture()
        owner = Owner(session,events)
        session.retain_native_owner(owner)
        session.close()
        self.assertEqual(events,["native close","synchronize","free",
                                 "BLAS destroy","module destroy","stream destroy"])
        self.assertTrue(session.cleanup_complete)
        self.assertEqual(session.native_owners,[])
        self.assertEqual(session.live_bytes,0)

    def test_failed_native_cleanup_retains_buffers_stream_and_handles_until_explicit_retry(self):
        session,events = self.fixture()
        owner = Owner(session,events,fail=True)
        session.retain_native_owner(owner)
        with self.assertRaisesRegex(RuntimeError,"native destruction failed"):
            session.close()
        self.assertEqual(events,["native close"])
        self.assertEqual(session.live_bytes,16)
        self.assertIsNotNone(session.handle)
        self.assertEqual((session.module,session.stream),(789,7))
        self.assertIs(session.native_owners[0],owner)
        self.assertTrue(session.closed);self.assertTrue(session.failed)
        self.assertFalse(session.cleanup_complete)
        with self.assertRaisesRegex(RuntimeError,"Closed/invalidated"): session.check()
        owner.fail = False
        session.close()  # Explicit cleanup only; cannot resume computation.
        self.assertTrue(session.cleanup_complete)
        self.assertEqual(events.count("free"),1)
        with self.assertRaisesRegex(RuntimeError,"Closed/invalidated"): session.check()

    def test_unreleased_native_owner_blocks_session_even_if_close_returned(self):
        session,events = self.fixture()
        owner = types.SimpleNamespace(cleanup_complete=False,close=lambda:None)
        session.retain_native_owner(owner)
        with self.assertRaisesRegex(RuntimeError,"dependents still retain"):
            session.close()
        self.assertEqual(events,[])
        self.assertEqual(session.live_bytes,16)

    def test_temporary_scope_duplicate_and_premature_release_guards(self):
        session,events = self.fixture()
        owner = Owner(session,events)
        with session.scope():
            self.assertEqual(session.scope_depth,1)
            with self.assertRaisesRegex(ValueError,"outside temporary"):
                session.retain_native_owner(owner)
        self.assertEqual(session.scope_depth,0)
        session.retain_native_owner(owner)
        with self.assertRaisesRegex(ValueError,"already retained"):
            session.retain_native_owner(owner)
        with self.assertRaisesRegex(RuntimeError,"not completed"):
            session.release_native_owner(owner)
        with self.assertRaises(ValueError): session.retain_native_owner(None)
        session.close()

    def test_outer_primary_exception_survives_failed_native_cleanup(self):
        session,events = self.fixture()
        session.retain_native_owner(Owner(session,events,fail=True))
        primary = ValueError("original compute failure")
        self.assertFalse(session.__exit__(ValueError,primary,None))
        self.assertIn("native destruction failed",primary.__notes__[0])
        self.assertEqual(events,["native close"])
        self.assertEqual(session.live_bytes,16)

    def test_multiple_owners_reverse_order_partial_failure_and_no_reclose_of_successes(self):
        session,events = self.fixture()
        owners = [Owner(session,events,name="A"),
                  Owner(session,events,fail=True,name="B"),
                  Owner(session,events,name="C")]
        for owner in owners: session.retain_native_owner(owner)
        with self.assertRaisesRegex(RuntimeError,"native destruction failed"): session.close()
        self.assertEqual(events,["C","B","A"])
        self.assertEqual(session.native_owners,[owners[1]])
        self.assertEqual(session.live_bytes,16)
        self.assertTrue(owners[0].cleanup_complete)
        self.assertTrue(owners[2].cleanup_complete)
        owners[1].fail = False
        session.close()
        self.assertEqual(events[:4],["C","B","A","B"])
        self.assertEqual((events.count("A"),events.count("B"),events.count("C")),(1,2,1))
        self.assertEqual(events.count("free"),1)

    def test_completed_flag_without_unregistration_still_blocks_parent_release(self):
        session,events = self.fixture()
        owner = types.SimpleNamespace(cleanup_complete=False)
        owner.close = lambda:setattr(owner,"cleanup_complete",True)
        session.retain_native_owner(owner)
        with self.assertRaisesRegex(RuntimeError,"dependents still retain"): session.close()
        self.assertTrue(owner.cleanup_complete)
        self.assertIs(session.native_owners[0],owner)
        self.assertEqual(events,[])
        self.assertEqual(session.live_bytes,16)
        session.release_native_owner(owner)
        session.close()
        self.assertEqual(events.count("free"),1)

    def test_nested_scopes_forbid_registration_until_outermost_scope_exits(self):
        session,events = self.fixture()
        owner = Owner(session,events)
        with session.scope():
            with session.scope():
                self.assertEqual(session.scope_depth,2)
                with self.assertRaisesRegex(ValueError,"outside temporary"):
                    session.retain_native_owner(owner)
            self.assertEqual(session.scope_depth,1)
            with self.assertRaisesRegex(ValueError,"outside temporary"):
                session.retain_native_owner(owner)
        self.assertEqual(session.scope_depth,0)
        session.retain_native_owner(owner)
        session.close()

    def test_nested_scope_depth_restores_after_compute_or_cleanup_failure(self):
        for failure in ("compute","cleanup"):
            session,events = self.fixture()
            syncs = []
            def synchronize():
                syncs.append(1)
                return (1,) if failure == "cleanup" or len(syncs) == 1 else (0,)
            session.runtime.cudaDeviceSynchronize = synchronize
            expected = "inner compute failure" if failure == "compute" else "CUDA API failed"
            with self.subTest(failure=failure),self.assertRaisesRegex(RuntimeError,expected) as error:
                with session.scope():
                    try:
                        with session.scope():
                            self.assertEqual(session.scope_depth,2)
                            if failure == "compute": raise RuntimeError("inner compute failure")
                    finally:
                        self.assertEqual(session.scope_depth,1)
                        with self.assertRaises(RuntimeError):
                            session.retain_native_owner(Owner(session,events))
            self.assertEqual(session.scope_depth,0)
            self.assertTrue(session.failed)
            self.assertTrue(error.exception.__notes__)
            self.assertEqual(session.native_owners,[])
            self.assertEqual(session.live_bytes,16)


if __name__ == "__main__":
    unittest.main()
