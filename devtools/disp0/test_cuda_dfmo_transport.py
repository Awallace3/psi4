"""CPU fake-C-ABI transport tests; no canonical SDK/native/GPU execution."""
import ctypes as ct
import hashlib
import importlib
import json
import os
import types
import unittest
from unittest.mock import Mock,patch

import numpy as np

from test_cuda_backend import CPUSession,host,inputs,PACKAGE

dfmo=importlib.import_module(f"{PACKAGE}.sapt_cuest_dfmo_cuda")


class Session(CPUSession):
    def __init__(self):
        super().__init__(None)
        self.syncs=0;self.fail_syncs=set()
        self.blas_path="CPU BLAS FIXTURE";self.blas_sha256="0"*64
    def synchronize_device(self):
        self.syncs+=1
        if self.syncs in self.fail_syncs:raise RuntimeError("fixture sync failure")


class Native:
    def __init__(self,ao,order,version=(0,2,50),math_mode=1,math_code=0):
        self.ao,self.order=ao,order
        self.math_mode,self.math_code=math_mode,math_code
        self.created=self.destroyed=self.computed=0
        self.query_code=self.compute_code=self.destroy_code=0
        self.cuestParametersCreate=Mock(side_effect=self.create)
        self.cuestParametersDestroy=Mock(side_effect=self.destroy)
        self.cuestDFMOIntegralsComputeWorkspaceQuery=Mock(side_effect=self.query)
        self.cuestDFMOIntegralsCompute=Mock(side_effect=self.compute)
        self.cuestGetMathMode=Mock(side_effect=self.get_math_mode)
        for name,value in zip(("Major","Minor","Patch"),version):
            def get(handle,out,value=value):
                out._obj.value=value;return 0
            setattr(self,"cuestGet"+name+"Version",Mock(side_effect=get))
    def get_math_mode(self,handle,out):
        assert isinstance(handle,ct.c_void_p) and handle.value==123
        assert isinstance(out._obj,ct.c_int)
        out._obj.value=self.math_mode
        return self.math_code
    def create(self,kind,out):
        assert kind==49
        out._obj.value=1234;self.created+=1;return 0
    def destroy(self,kind,params):
        assert kind==49
        self.destroyed+=1;return self.destroy_code
    def query(self,*args):
        assert len(args)==11
        assert type(args[6])._type_ is ct.c_uint64
        args[4]._obj.host_bytes=128;args[4]._obj.device_bytes=256
        return self.query_code
    def compute(self,*args):
        if self.compute_code:return self.compute_code
        n=int(args[5]);nl,nr=args[6],args[7];nao=self.ao.shape[1];q=self.ao.shape[0]
        def array(pointer,count):
            return np.ctypeslib.as_array((ct.c_double*count).from_address(pointer.value))
        left=array(args[8],sum(nl)*nao).reshape(-1,nao)
        right=array(args[9],sum(nr)*nao).reshape(-1,nao)
        out=array(args[10],q*sum(l*r for l,r in zip(nl,nr)))
        li=ri=offset=0
        for i in range(n):
            l,r=int(nl[i]),int(nr[i])
            value=np.einsum("Qmn,lm,rn->Qlr",self.ao,left[li:li+l],right[ri:ri+r])
            out[offset:offset+value.size]=value.ravel(order=self.order)
            offset+=value.size;li+=l;ri+=r
        self.computed+=1;return 0


class Transport(unittest.TestCase):
    def fixture(self,order="C",version=(0,2,50),audit_error=None,
                math_mode=1,math_code=0,**changes):
        ao,sp,_,_=inputs();session=Session();native=Native(ao,order,version,math_mode,math_code)
        self.native_attempt=native
        primary=types.SimpleNamespace(nbf=lambda:ao.shape[1])
        aux=types.SimpleNamespace(nbf=lambda:ao.shape[0])
        manifest={"scope":"EXPLICIT CPU ADMISSION MOCK, NOT A CUDA MANIFEST"}
        sha=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        kwargs=dict(session=session,sdk_root="CPU SDK FIXTURE",archive="CPU ARCHIVE FIXTURE",
                    cuda_manifest=manifest,cuda_manifest_sha256=sha,handle=123,plan=456,
                    handle_pid=os.getpid(),primary_basis=primary,auxiliary_basis=aux,
                    keepalive=(object(),),plan_policy=dict(dfmo.POLICY),output_order=order)
        kwargs.update(changes)
        with patch.object(dfmo.identity,"verify_files",return_value={"native_path":"CPU FIXTURE"}),patch.object(
                dfmo,"library",return_value=(native,"CPU FIXTURE")),patch.object(
                dfmo.identity,"audit_maps",side_effect=audit_error,
                return_value={"cuda_mapped_sha256":{session.blas_path:session.blas_sha256}}):
            provider=dfmo.CuESTDFMOProvider(**kwargs)
        return provider,session,native,ao,sp

    def test_batched_pointers_layout_and_output_owner_survive_temporary_scope(self):
        for order in ("C","F"):
            p,s,n,ao,sp=self.fixture(order)
            with s.scope():
                pairs={"x":(sp["r"][:,1:4],sp["a"]),"y":(sp["s"][:,::2],sp["b"])}
                factors=p.compute(pairs)
                for name,(cl,cr) in pairs.items():
                    np.testing.assert_allclose(host(factors[name]),
                                               np.einsum("Qmn,mi,nj->Qij",ao,cl,cr),
                                               rtol=1e-12,atol=1e-12)
                self.assertIs(factors["x"].allocation,factors["y"].allocation)
                self.assertEqual(n.created,n.destroyed)
                self.assertEqual(s.syncs,2)
                self.assertEqual(ct.sizeof(dfmo._Descriptor),16)
                self.assertEqual(ct.sizeof(dfmo._Workspace),32)
                self.assertEqual([getattr(dfmo._Workspace,name).offset
                                  for name,_ in dfmo._Workspace._fields_],[0,8,16,24])
            with self.assertRaises(RuntimeError):_=factors["x"].pointer

    def test_native_error_cleanup_and_no_retry(self):
        p,s,n,_,sp=self.fixture()
        n.compute_code,n.destroy_code=9,8;s.fail_syncs={2}
        with self.assertRaisesRegex(RuntimeError,"compute failed: 9") as error:
            p.compute({"x":(sp["r"],sp["a"])})
        self.assertEqual(len(error.exception.__notes__),2)
        self.assertEqual(n.destroyed,1)
        with self.assertRaisesRegex(RuntimeError,"invalidated"):
            p.compute({"x":(sp["r"],sp["a"])})

    def test_old_version_wrong_pid_qr_or_receipt_fail_before_dfmo(self):
        with self.assertRaisesRegex(RuntimeError,"must be0.2.50"):
            self.fixture(version=(0,2,2))
        with self.assertRaises(ValueError):self.fixture(handle_pid=os.getpid()+1)
        policy=dict(dfmo.POLICY);policy["fitting_algorithm"]=0
        with self.assertRaisesRegex(ValueError,"MATRIXPOWER"):
            self.fixture(plan_policy=policy)
        with self.assertRaisesRegex(ValueError,"receipt hash"):
            self.fixture(cuda_manifest_sha256="f"*64)

    def test_mapped_ownership_failure_precedes_native_version_calls(self):
        with self.assertRaisesRegex(RuntimeError,"unadmitted mapping"):
            self.fixture(audit_error=RuntimeError("unadmitted mapping"))
        self.native_attempt.cuestGetMajorVersion.assert_not_called()
        self.native_attempt.cuestGetMinorVersion.assert_not_called()
        self.native_attempt.cuestGetPatchVersion.assert_not_called()
        self.native_attempt.cuestGetMathMode.assert_not_called()

    def test_native_fp64_mode_is_required_not_merely_fp64_output_buffers(self):
        for mode in (0,2,-1):
            with self.subTest(mode=mode),self.assertRaisesRegex(RuntimeError,"FP64"):
                self.fixture(math_mode=mode)
            self.assertEqual(self.native_attempt.created,0)
        with self.assertRaisesRegex(RuntimeError,"math mode query failed: 7"):
            self.fixture(math_code=7)

    def test_changed_math_mode_invalidates_provider_before_dfmo_allocation(self):
        p,s,n,_,sp=self.fixture()
        n.math_mode=0
        with self.assertRaisesRegex(RuntimeError,"FP64"):
            p.compute({"x":(sp["r"],sp["a"])})
        self.assertTrue(p.failed)
        self.assertEqual(n.created,0)
        self.assertEqual(s.live_bytes,0)
        n.math_mode=1
        with self.assertRaisesRegex(RuntimeError,"invalidated"):
            p.compute({"x":(sp["r"],sp["a"])})

    def test_compute_math_query_status_or_exception_invalidates_without_allocation(self):
        for exception in (False,True):
            with self.subTest(exception=exception):
                p,s,n,_,sp=self.fixture()
                n.math_code=7
                if exception:
                    n.cuestGetMathMode.side_effect=RuntimeError("query exception")
                with self.assertRaisesRegex(RuntimeError,"query exception" if exception
                                             else "math mode query failed: 7"):
                    p.compute({"x":(sp["r"],sp["a"])})
                self.assertTrue(p.failed)
                self.assertEqual(n.created,0)
                self.assertEqual(s.live_bytes,0)
                n.math_code=0
                with self.assertRaisesRegex(RuntimeError,"invalidated"):
                    p.compute({"x":(sp["r"],sp["a"])})

    def test_report_reads_actual_mode_and_rejects_drift(self):
        p,s,n,_,_=self.fixture()
        with patch.object(dfmo.identity,"audit_maps",
                          return_value={"cuda_mapped_sha256":{s.blas_path:s.blas_sha256}}):
            self.assertEqual(p.report()["native_math_mode"],1)
            n.math_mode=0
            with self.assertRaisesRegex(RuntimeError,"FP64"):
                p.report()


if __name__=="__main__":
    unittest.main()
