"""CPU native-call spies for owned argument packing; no CUDA execution evidence."""
import ctypes as ct
import importlib
import types
import unittest
from unittest.mock import Mock

from test_cuda_backend import PACKAGE

runtime=importlib.import_module(PACKAGE+".sapt_cuda_runtime")


class OwnedArguments(unittest.TestCase):
    def fixture(self, *, owned=True):
        session=runtime.CUDASession.__new__(runtime.CUDASession)
        session.closed=session.failed=False
        session.device=0
        session.stream=7
        session.handle=ct.c_void_p(8)
        session.functions={"fixture":9}
        session.kernel_launches={}
        session.blas_calls={}
        session.identity_owner=types.SimpleNamespace(check_live=Mock()) if owned else None
        session.runtime=types.SimpleNamespace(cudaGetDevice=Mock(return_value=(0,0)))
        session.driver=types.SimpleNamespace(cuLaunchKernel=Mock(return_value=(0,)))
        session.blas=types.SimpleNamespace(
            cublasDgemm_v2=Mock(return_value=0),
            cublasDgemmStridedBatched=Mock(return_value=0))
        def array(shape,pointer):
            size=8
            for extent in shape:size*=extent
            return runtime.DeviceArray(runtime._Allocation(session,pointer,size),shape)
        return session,array

    def test_owned_launch_has_constant_full_checks_and_unchanged_argument_abi(self):
        for number in (1,3,8):
            session,array=self.fixture()
            arrays=[array((2,),1000+100*i) for i in range(number)]
            arguments=[13,*arrays,None,True]
            session.launch("fixture",2,128,arguments)
            # One admission at entry, one directly before native enqueue, not
            # a full runtime/device check for each pure address calculation.
            self.assertEqual(session.runtime.cudaGetDevice.call_count,2)
            self.assertEqual(session.identity_owner.check_live.call_count,4)
            session.driver.cuLaunchKernel.assert_called_once()
            call=session.driver.cuLaunchKernel.call_args.args
            self.assertEqual(call[:9],(9,2,1,1,128,1,1,0,7))
            # Inspect holders WHILE the production native-call spy runs below,
            # rather than dereferencing their ended lifetimes after launch().
            self.assertEqual(session.kernel_launches,{"fixture":1})

    def test_launch_holders_survive_native_enqueue_and_match_original_types(self):
        session,array=self.fixture()
        left,right=array((2,),1000),array((2,),2000)
        def native(*args):
            parameters=ct.cast(args[9],ct.POINTER(ct.c_void_p))
            self.assertEqual(ct.cast(parameters[0],ct.POINTER(ct.c_longlong))[0],13)
            self.assertEqual(ct.cast(parameters[1],ct.POINTER(ct.c_void_p))[0],1000)
            self.assertEqual(ct.cast(parameters[2],ct.POINTER(ct.c_void_p))[0],2000)
            self.assertIsNone(ct.cast(parameters[3],ct.POINTER(ct.c_void_p))[0])
            self.assertEqual(ct.cast(parameters[4],ct.POINTER(ct.c_int))[0],1)
            return (0,)
        session.driver.cuLaunchKernel.side_effect=native
        session.launch("fixture",2,128,[13,left,right,None,True])

    def test_public_pointer_keeps_its_original_context_and_lifetime_checks(self):
        session,array=self.fixture()
        value=array((2,),1000)
        self.assertEqual(value.pointer,1000)
        self.assertEqual(session.identity_owner.check_live.call_count,2)
        session.runtime.cudaGetDevice.return_value=(0,1)
        with self.assertRaisesRegex(RuntimeError,"changed device"):
            _=value.pointer
        value.allocation.freed=True
        with self.assertRaisesRegex(RuntimeError,"lifetime"):
            _=value.pointer

    def test_argument_coercion_cannot_revoke_identity_or_free_buffers_before_enqueue(self):
        for mode in ("identity","device","freed","offset"):
            session,array=self.fixture()
            value=array((2,),1000)
            class Changed:
                def __index__(self):
                    if mode=="identity":
                        session.identity_owner.check_live.side_effect=RuntimeError("revoked identity")
                    elif mode=="device":
                        session.runtime.cudaGetDevice.return_value=(0,1)
                    elif mode=="freed":
                        value.allocation.freed=True
                    else:
                        value.offset=8
                    return 1
            with self.subTest(mode=mode),self.assertRaises(RuntimeError):
                session.launch("fixture",1,128,[value,Changed()])
            session.driver.cuLaunchKernel.assert_not_called()

    def test_gemm_does_not_repeat_device_checks_for_each_packed_operand(self):
        for method in ("gemm","gemm_at_b","gemm_batched"):
            session,array=self.fixture()
            left,right=array((2,2),1000),array((2,2),2000)
            out=array((1,2,2) if method=="gemm_batched" else (2,2),3000)
            session.empty=Mock(return_value=out)
            result=getattr(session,method)(left,right)
            self.assertIs(result,out)
            self.assertEqual(session.runtime.cudaGetDevice.call_count,2)
            self.assertEqual(session.identity_owner.check_live.call_count,4)
            name="cublasDgemmStridedBatched" if method=="gemm_batched" else "cublasDgemm_v2"
            getattr(session.blas,name).assert_called_once()
            self.assertEqual(session.blas_calls,{name:1})

    def test_launch_descriptor_coercion_also_precedes_native_admission(self):
        for field in ("grid","threads","stream"):
            session,_=self.fixture()
            class Changed:
                def __int__(self):
                    session.identity_owner.check_live.side_effect=RuntimeError("revoked descriptor")
                    return 7
            grid,threads=1,128
            if field=="grid":grid=Changed()
            elif field=="threads":threads=Changed()
            else:session.stream=Changed()
            with self.subTest(field=field),self.assertRaisesRegex(RuntimeError,"revoked descriptor"):
                session.launch("fixture",grid,threads,[])
            session.driver.cuLaunchKernel.assert_not_called()

    def test_healthy_preparation_exceptions_retain_the_original_error(self):
        for kind in ("scalar","iterator","descriptor"):
            session,_=self.fixture()
            primary=ValueError("ordinary "+kind+" input error")
            class Scalar:
                def __index__(self):raise primary
            class Iterator:
                def __iter__(self):raise primary
            class Descriptor:
                def __int__(self):raise primary
            arguments=[Scalar()]
            if kind=="iterator":arguments=Iterator()
            elif kind=="descriptor":
                arguments=[]
                session.stream=Descriptor()
            with self.subTest(kind=kind),self.assertRaises(ValueError) as caught:
                session.launch("fixture",1,128,arguments)
            self.assertIs(caught.exception,primary)
            self.assertGreaterEqual(session.identity_owner.check_live.call_count,3)
            self.assertFalse(session.failed)
            session.driver.cuLaunchKernel.assert_not_called()

    def test_owned_arguments_never_borrow_another_sessions_allocation(self):
        session,_=self.fixture()
        foreign,array=self.fixture()
        value=array((2,),1000)
        with self.assertRaisesRegex(ValueError,"session"):
            session.launch("fixture",1,128,[value])
        session.driver.cuLaunchKernel.assert_not_called()

    def test_legacy_pointer_checks_are_not_replaced_by_an_owned_proof(self):
        session,array=self.fixture(owned=False)
        values=[array((2,),1000+100*i) for i in range(3)]
        session.launch("fixture",1,128,values)
        self.assertEqual(session.runtime.cudaGetDevice.call_count,4)
        session.driver.cuLaunchKernel.assert_called_once()


if __name__=="__main__":unittest.main()
