"""Negative GPU controls must never reuse an invalidated CUDA session."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from fragment_gate import charge_mask_probe


class Session:
    def __init__(self,**kwargs):
        self.cleanup_complete=False
        self.live_bytes=0
        self.native_owners=[]
        self.kernel_launches={}
        self.invalid=False

    def __enter__(self):
        if self.invalid:raise RuntimeError("Invalid session reused")
        return self

    def __exit__(self,*args):
        self.invalid=True
        self.cleanup_complete=True

    def upload(self,value):
        if self.invalid:raise RuntimeError("Invalid session reused")
        return value


class ChargeMaskProbeContracts(unittest.TestCase):
    def test_each_negative_control_has_a_fresh_session(self):
        sessions=[]
        def factory(**kwargs):
            session=Session(**kwargs)
            sessions.append(session)
            return session
        def validate(session,za,zb,active):
            if session.invalid:raise RuntimeError("Invalid session reused")
            session.kernel_launches["fragment_charge_mask_error"]=1
            actual=tuple(frozenset(np.flatnonzero(z)) for z in (za,zb))
            if (not np.isfinite(za).all() or not np.isfinite(zb).all()
                    or actual!=tuple(frozenset(indices) for indices in active)):
                # Model production scope's fail-closed behavior, absent from
                # the serial kernel-equation shim.
                session.invalid=True
                raise ValueError("Active-atom metadata differs from resident nuclear charges")
            return actual
        runtime=SimpleNamespace(cuda=SimpleNamespace(CUDASession=factory),
            blas={"path":"fixture","sha256":"fixture"},
            config={"device_memory_limit_bytes":1024,"native_persistent_bytes":0})
        with patch("fragment_gate.importlib.import_module",
                   return_value=SimpleNamespace(validated_active_atoms=validate)):
            controls=charge_mask_probe(runtime)
        self.assertEqual(len(controls),9)
        self.assertEqual(len({id(session) for session in sessions}),9)
        self.assertTrue(all(s.cleanup_complete for s in sessions))
        self.assertTrue(all(row["passed"] for row in controls))

    def test_unrelated_value_error_is_not_accepted_as_rejection(self):
        runtime=SimpleNamespace(cuda=SimpleNamespace(CUDASession=Session),
            blas={"path":"fixture","sha256":"fixture"},
            config={"device_memory_limit_bytes":1024,"native_persistent_bytes":0})
        calls=[]
        def broken(session,za,zb,active):
            calls.append(active)
            session.kernel_launches["fragment_charge_mask_error"]=1
            if len(calls)==1:return tuple(frozenset(x) for x in active)
            raise ValueError("Unrelated setup failure")
        with patch("fragment_gate.importlib.import_module",
                   return_value=SimpleNamespace(validated_active_atoms=broken)):
            with self.assertRaisesRegex(ValueError,"Unrelated setup failure"):
                charge_mask_probe(runtime)


if __name__=="__main__":
    unittest.main()
