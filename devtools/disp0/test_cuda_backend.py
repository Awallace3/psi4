"""CPU serial execution of actual CUDA bodies via a C++ qualifier shim.

NOT NVRTC compilation, CUDA parallel execution, cuEST, cuBLAS or physics proof.
"""
import contextlib
import ctypes as ct
import _ctypes
import importlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock

import numpy as np

CUPY_BEFORE_IMPORT = sys.modules.get("cupy")

SAPT=Path(__file__).resolve().parents[2]/"psi4/driver/procrouting/sapt"
PACKAGE="_disp0_cuda_kernel_fixtures"
package=types.ModuleType(PACKAGE);package.__path__=[str(SAPT)];sys.modules[PACKAGE]=package
runtime=importlib.import_module(f"{PACKAGE}.sapt_cuda_runtime")
backend=importlib.import_module(f"{PACKAGE}.sapt_dispersion_cuda")
source=importlib.import_module(f"{PACKAGE}.sapt_dispersion_cuda_kernels")
common=importlib.import_module(f"{PACKAGE}.sapt_dispersion_device_common")


def host(array):
    return np.ndarray(array.shape,dtype=array.dtype,buffer=array.allocation.host,
                      offset=array.offset,order=array.order)


class Allocation:
    def __init__(self,session,size):
        self.session,self.nbytes,self.freed=session,size,False
        self.host=np.full(size,127,dtype=np.uint8)  # poison, never valid zero output
        self.pointer=self.host.ctypes.data
    def free(self):
        if not self.freed:
            self.freed=True;self.session.live_bytes-=self.nbytes


class CPUSession:
    """Explicit CPU test double that executes serial-shim kernel bodies."""
    def __init__(self,kernels):
        self.kernels=kernels;self.allocations=[]
        self.live_bytes=self.peak_live_bytes=self.native_bytes=0
        self.failed=self.closed=False;self.gemms=0
    def check(self):
        if self.closed or self.failed:raise RuntimeError("CPU fixture closed")
    def empty(self,shape,dtype=np.float64,zero=False):
        allocation=Allocation(self,int(np.prod(shape))*np.dtype(dtype).itemsize)
        self.allocations.append(allocation);self.live_bytes+=allocation.nbytes
        self.peak_live_bytes=max(self.peak_live_bytes,self.live_bytes)
        array=runtime.DeviceArray(allocation,shape,dtype=dtype)
        if zero:host(array)[...]=0
        return array
    def upload(self,array):
        result=self.empty(array.shape,dtype=array.dtype);host(result)[...]=array;return result
    def download(self,array):return np.array(host(array))
    @contextlib.contextmanager
    def scope(self):
        start=len(self.allocations)
        try:yield
        finally:
            for allocation in self.allocations[start:]:allocation.free()
            del self.allocations[start:]
    def gemm_at_b(self,left,right):
        result=self.empty((left.shape[1],right.shape[1]))
        host(result)[...]=host(left).T@host(right);self.gemms+=1
        return result
    def launch(self,name,grid,threads,arguments):
        # GPU grid/threads are deliberately NOT simulated. Bodies execute on
        # one CPU thread with blockDim/gridDim=1, testing index equations only.
        converted=[ct.c_void_p(x.pointer if x is not None else 0)
                   if isinstance(x,runtime.DeviceArray) or x is None
                   else ct.c_int(x) if isinstance(x,bool) else ct.c_longlong(x)
                   for x in arguments]
        getattr(self.kernels,name)(*converted)


def inputs(seed=19):
    rng=np.random.default_rng(seed);nao,q,a,r,b,s=7,9,2,5,3,4
    sizes={"a":a,"r":r,"b":b,"s":s}
    spaces={name:rng.normal(size=(nao,sizes[name[0]]))
            for name in ("a","r","b","s","r1","s1","a2","b2","r3","s3","a4","b4")}
    corrections={name:rng.normal(size=shape) for name,shape in {
        "Qar":(a,r),"SBar":(a,r),"Qbr":(b,r),"Sbr":(b,r),
        "SAbs":(b,s),"Qbs":(b,s),"Sas":(a,s),"Qas":(a,s)}.items()}
    energies={"a":-rng.uniform(.7,1.2,a),"b":-rng.uniform(.8,1.3,b),
              "r":rng.uniform(.1,.6,r),"s":rng.uniform(.1,.7,s)}
    return rng.normal(size=(q,nao,nao)),spaces,corrections,energies


def oracle(ao,sp,c,e):
    def f(left,right):return np.einsum("Qmn,mi,nj->Qij",ao,sp[left],sp[right])
    ar,bs=f("r","a"),f("s","b")
    fr,fs=f("r3","a")+f("r","a4"),f("s3","b")+f("s","b4")
    bas,cas,bbr,cbr=f("s1","a"),f("s","a2"),f("r1","b"),f("r","b2")
    d=np.zeros((e["a"].size,e["b"].size));x=np.zeros_like(d)
    for a in range(d.shape[0]):
        for b in range(d.shape[1]):
            for r in range(e["r"].size):
                for s in range(e["s"].size):
                    v=ar[:,r,a]@bs[:,s,b]
                    t=v/(e["a"][a]+e["b"][b]-e["r"][r]-e["s"][s])
                    direct=(ar[:,r,a]@fs[:,s,b]+fr[:,r,a]@bs[:,s,b]
                            +c["Qar"][a,r]*c["SAbs"][b,s]+c["SBar"][a,r]*c["Qbs"][b,s])
                    crossed=(bas[:,s,a]@bbr[:,r,b]+cas[:,s,a]@cbr[:,r,b]
                             +c["Sas"][a,s]*c["Qbr"][b,r]+c["Qas"][a,s]*c["Sbr"][b,r])
                    d[a,b]+=4*t*v;x[a,b]-=2*t*(direct+crossed)
    return d,x


class Kernels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler=shutil.which("c++")
        if compiler is None:raise unittest.SkipTest("CPU C++ qualifier-shim compiler unavailable")
        cls.temp=tempfile.TemporaryDirectory(prefix="disp0-cuda-serial-")
        folder=Path(cls.temp.name)
        shim="""
        #define __global__
        #define __device__
        #define __forceinline__
        #define __shared__
        #define __syncthreads() ((void)0)
        struct Dim { unsigned int x,y,z; };
        static const Dim blockIdx={0,0,0},threadIdx={0,0,0},
                         blockDim={1,1,1},gridDim={1,1,1};
        """
        path=folder/"serial.cc";path.write_text(shim+source.CUDA_SOURCE)
        result=subprocess.run([compiler,"-std=c++14","-shared","-fPIC","-O0",
                               str(path),"-o",str(folder/"serial.so")],
                              capture_output=True,text=True,timeout=20)
        if result.returncode:raise AssertionError(result.stderr)
        cls.lib=ct.CDLL(str(folder/"serial.so"))
        cls.lib.pack_factors.argtypes=[ct.c_longlong]*3+[ct.c_int]+[ct.c_void_p]*7
        cls.lib.reduce_dispersion.argtypes=[ct.c_longlong]*6+[ct.c_void_p]*9
        cls.lib.pack_factors.restype=cls.lib.reduce_dispersion.restype=None

    @classmethod
    def tearDownClass(cls):
        # CDLL does not close its dlopen handle on collection. Unlinking first
        # leaves a deleted mapped .so and correctly trips later identity guards.
        # No kernel callback may escape this serial test class's lifetime.
        _ctypes.dlclose(cls.lib._handle)
        cls.lib = None
        cls.temp.cleanup()

    def test_actual_kernel_bodies_block_edges_orders_changed_inputs_and_sums(self):
        # Predeclared kernel tolerances, no physics/GPU inference.
        rtol,atol=1e-11,1e-12
        for seed in (19,20):
            ao,sp,c,e=inputs(seed);expected=oracle(ao,sp,c,e)
            for order in ("C","F"):
                for block in (1,3,500):
                    session=CPUSession(self.lib)
                    class Provider:
                        nao,naux=ao.shape[1],ao.shape[0]
                        def compute(self,pairs):
                            factors={}
                            for name,(cl,cr) in pairs.items():
                                value=np.einsum("Qmn,mi,nj->Qij",ao,cl,cr)
                                storage=self.session.empty((value.size,))
                                array=storage.view(value.shape,order=order)
                                host(array)[...]=value;factors[name]=array
                            return factors
                        def report(self):return {"scope":"CPU SERIAL TEST DOUBLE"}
                    provider=Provider();provider.session=session
                    result=backend.blocked_dispersion(provider,sp,c,e,block_size=block)
                    for key,pair_key,reference in (
                            ("Disp20,u","Disp20 occupied pairs",expected[0]),
                            ("Exch-Disp20,u","Exch-Disp20 occupied pairs",expected[1])):
                        np.testing.assert_allclose(result[pair_key],reference,rtol=rtol,atol=atol)
                        np.testing.assert_allclose(result[key],reference.sum(),rtol=rtol,atol=atol)
                    self.assertEqual(session.live_bytes,0)
                    self.assertEqual(session.gemms,3*result["Disp0 execution"]["compute_blocks"])
                    self.assertFalse(result["Disp0 execution"]["actual_gpu_execution_verified"])

    def test_no_cupy_import_and_buffer_lifetime_or_host_fallback(self):
        self.assertIs(sys.modules.get("cupy"),CUPY_BEFORE_IMPORT)
        session=CPUSession(self.lib)
        with session.scope():
            array=session.empty((3,4))
            with self.assertRaises(ValueError):array.view((1,),offset=1)
            parent=array.view((6,),offset=16)
            with self.assertRaises(ValueError):parent.view((1,),offset=-8)
        with self.assertRaises(RuntimeError):_ = array.pointer
        ao,sp,_,_=inputs()
        provider=types.SimpleNamespace(naux=ao.shape[0],compute=lambda pairs:{"x":np.zeros((9,2,3))})
        with self.assertRaises(ValueError):
            backend._factor_batch(session,provider,{"x":(sp["a"],sp["b"])})

    def test_row_major_cublas_descriptor_and_scalar_lifetimes(self):
        session=CPUSession(self.lib)
        left=session.empty((9,6));right=session.empty((9,8))
        # Call actual production descriptor builder with a spying CPU BLAS API.
        calls=[]
        session._blas=lambda *args:calls.append(args)
        session.handle=ct.c_void_p(123)
        out=runtime.CUDASession.gemm_at_b(session,left,right)
        args=calls[0]
        self.assertEqual(args[:7],("cublasDgemm_v2",session.handle,0,1,8,6,9))
        self.assertEqual((args[8],args[9],args[10],args[11],args[13],args[14]),
                         (right.pointer,8,left.pointer,6,out.pointer,8))
        self.assertEqual(ct.cast(args[7],ct.POINTER(ct.c_double))[0],1.)
        self.assertEqual(ct.cast(args[12],ct.POINTER(ct.c_double))[0],0.)

    def test_cleanup_inside_unrelated_except_preserves_local_failures(self):
        def fail():raise RuntimeError("local cleanup")
        try:raise ValueError("outer")
        except ValueError as outer:
            with self.assertRaisesRegex(RuntimeError,"local cleanup"):
                runtime.cleanup(None,[fail])
            self.assertFalse(hasattr(outer,"__notes__"))
        original=ValueError("primary")
        runtime.cleanup(original,[fail])
        self.assertIn("local cleanup",original.__notes__[0])

    def test_production_launch_uses_raw_parameter_address_and_live_scalar_holders(self):
        session=CPUSession(self.lib)
        session.stream=7;session.functions={"pack_factors":1234}
        session.rt=runtime.CUDASession.rt
        array=session.empty((2,3))
        arguments=[9,3,2,False,array,None,array,None,array,array,array]
        seen=[]
        def launch(*args):
            self.assertIsInstance(args[9],int)
            addresses=(ct.c_void_p*len(arguments)).from_address(args[9])
            for pointer,value in zip(addresses,arguments):
                kind=ct.c_void_p if isinstance(value,runtime.DeviceArray) or value is None else (
                    ct.c_int if isinstance(value,bool) else ct.c_longlong)
                actual=ct.cast(pointer,ct.POINTER(kind))[0]
                expected=value.pointer if isinstance(value,runtime.DeviceArray) else (
                    None if value is None else int(value))
                self.assertEqual(actual,expected)
            seen.append(args)
            return (0,)
        session.driver=types.SimpleNamespace(cuLaunchKernel=launch)
        runtime.CUDASession.launch(session,"pack_factors",2,256,arguments)
        self.assertEqual(len(seen),1)

    def test_failed_close_retains_resources_for_explicit_cleanup_only_retry(self):
        session=runtime.CUDASession.__new__(runtime.CUDASession)
        frees=[]
        def free(pointer):
            frees.append(pointer)
            return (1,) if len(frees)==1 else (0,)
        session.runtime=types.SimpleNamespace(cudaDeviceSynchronize=lambda:(0,),
                                             cudaFree=free,cudaStreamDestroy=lambda stream:(0,))
        session.driver=types.SimpleNamespace(cuModuleUnload=lambda module:(0,))
        session.failed=session.closed=session.cleanup_complete=False
        session.live_bytes=16;session.allocations=[runtime._Allocation(session,123,16)]
        session.handle=ct.c_void_p(456);session.module=789;session.stream=7
        destroyed=[];session._blas=lambda name,handle:destroyed.append(name)
        with self.assertRaises(RuntimeError):session.close()
        self.assertTrue(session.closed)
        self.assertFalse(session.cleanup_complete)
        self.assertEqual(len(session.allocations),1)
        self.assertIsNone(session.handle);self.assertIsNone(session.module);self.assertIsNone(session.stream)
        session.close()  # Explicit cleanup only, not an execution/allocation retry.
        self.assertTrue(session.cleanup_complete)
        self.assertEqual(session.allocations,[])
        self.assertEqual(session.live_bytes,0)
        self.assertEqual(destroyed,["cublasDestroy_v2"])
        with self.assertRaises(RuntimeError):session.check()

    def test_invalid_host_contracts_and_no_denominator_clamp(self):
        _,sp,c,e=inputs();e["r"][:]=-100
        with self.assertRaises(ValueError):common.validate_inputs(sp,c,e)
        for block in (0,-1,True,1.5):
            with self.assertRaises(ValueError):backend.blocked_dispersion(None,sp,c,e,block_size=block)
        self.assertNotIn("1e20",source.CUDA_SOURCE)
        self.assertNotIn("atomicAdd",source.CUDA_SOURCE)


if __name__=="__main__":
    unittest.main()
