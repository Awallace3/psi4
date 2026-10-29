"""THROWAWAY companion: sealed CUDA binding/kernel/cuEST context compatibility.

No Psi4 import, production identity changes, SCF, benchmark or context reset.
Invoked only by sealed_loader_probe.py --gpu in the approved bounded scope.
"""
import ctypes as ct
import importlib.metadata
import os
from pathlib import Path
import sys

from sealed_loader_probe import digest,symbol_owner,state,F_GET_SEALS
import fcntl


def gpu_policy(plan):
    root=Path(plan["root"])
    sys.path.insert(0,str(root/"source/devtools/disp0/cachy"))
    from coherent_stage11 import policy
    from gpu_runtime import require_local_lease
    from response_reuse_probe import resource_scope
    assert os.environ["CUDA_VISIBLE_DEVICES"]==plan["expected_uuid"]
    assert digest(__file__)==plan["gpu_probe_sha256"]
    require_local_lease(policy(root))
    return resource_scope()


def maps(records):
    """Reject any second/unsealed toolkit, including one loaded lazily."""
    identities={(r["inode"],tuple(r["device"]),r["kernel_path"]):name
                for name,r in records.items()}
    prefixes=("libcudart","libcublas","libcusolver","libcusparse","libnvrtc",
              "libnvJitLink","libcufft","libcurand","libcuest","libcufile",
              "libnvfatbin","libnvjpeg","libnvvm","libnvptxcompiler","libcutensor")
    seen=set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        row=line.split(maxsplit=5)
        if len(row)!=6:continue
        path=row[5]
        if not any(prefix in Path(path).name for prefix in prefixes):continue
        key=(int(row[4]),tuple(int(x,16) for x in row[3].split(":")),path)
        assert key in identities,("Unsealed/foreign toolkit mapping",line)
        seen.add(identities[key])
    for record in records.values():
        assert fcntl.fcntl(record["fd"],F_GET_SEALS)==record["seals"]
    return sorted(seen)


def probe(plan,records,loaded):
    scope=gpu_policy(plan)
    from cuda.bindings import runtime as rt,driver as dr,nvrtc
    selected=Path(plan["optional_python_path"]).resolve()
    for module in (rt,dr,nvrtc):
        assert Path(module.__file__).resolve().is_relative_to(selected)
    assert importlib.metadata.version("cuda-bindings")=="13.3.0"
    report={"resource_scope":scope,"cuda_bindings":"13.3.0",
            "mapped_at_import":maps(records),"cleanup":[],"native_calls":0,
            "cuda_calls":0,"kernel_launches":0}
    def cuda(result):
        report["cuda_calls"]+=1
        assert int(result[0])==0,result
        values=result[1:]
        return values[0] if len(values)==1 else values
    def native(name,*args):
        report["native_calls"]+=1
        function=getattr(cuest,name)
        symbol_owner(function,records["libcuest.so.0.2.50"])
        code=function(*args)
        assert code==0,(name,code)
    assert cuda(rt.cudaRuntimeGetVersion())==13030
    assert tuple(cuda(nvrtc.nvrtcVersion()))==(13,3)
    assert cuda(rt.cudaGetDeviceCount())==1
    assert cuda(rt.cudaGetDevice())==0
    major=cuda(rt.cudaDeviceGetAttribute(rt.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor,0))
    minor=cuda(rt.cudaDeviceGetAttribute(rt.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor,0))
    assert (major,minor)==(8,6)
    report["mapped_after_lazy_binding_calls"]=maps(records)
    stream=module=allocation=None
    parameters,handle=ct.c_void_p(),ct.c_void_p()
    cuest=loaded[plan["preload_order"].index("libcuest.so.0.2.50")]
    H,P=ct.c_void_p,ct.POINTER
    specs={"cuestCreate":[H,P(H)],"cuestDestroy":[H],"cuestSetMathMode":[H,ct.c_int],
           "cuestParametersConfigure":[ct.c_int,H,ct.c_int,H,ct.c_size_t],
           "cuestParametersQuery":[ct.c_int,H,ct.c_int,H,ct.c_size_t]}
    for name,args in specs.items():
        fn=getattr(cuest,name);fn.argtypes,fn.restype=args,ct.c_int
    try:
        stream=cuda(rt.cudaStreamCreate())
        context=int(cuda(dr.cuCtxGetCurrent()));assert context
        report["free_before"]=cuda(rt.cudaMemGetInfo())[0]
        # Reuse the exact528-byte CPU-compiled one-element FP64 kernel.
        ptx=Path(plan["result"]).parent/"probe.ptx"
        module=cuda(dr.cuModuleLoadData(ptx.read_bytes()))
        function=cuda(dr.cuModuleGetFunction(module,b"sealed_probe"))
        allocation=cuda(rt.cudaMalloc(8))
        value=ct.c_double(2.0)
        cuda(rt.cudaMemcpyAsync(allocation,ct.addressof(value),8,
             rt.cudaMemcpyKind.cudaMemcpyHostToDevice,stream))
        cuda(rt.cudaStreamSynchronize(stream))
        holder=H(int(allocation));arguments=(H*1)(ct.cast(ct.pointer(holder),H))
        cuda(dr.cuLaunchKernel(function,1,1,1,1,1,1,0,int(stream),ct.addressof(arguments),0))
        report["kernel_launches"]+=1
        cuda(rt.cudaMemcpyAsync(ct.addressof(value),allocation,8,
             rt.cudaMemcpyKind.cudaMemcpyDeviceToHost,stream))
        cuda(rt.cudaStreamSynchronize(stream))
        assert value.value==3.0,value.value
        report["one_element_fp64_result"]=value.value
        state("sealed_gpu_kernel_passed",value=value.value,mapped=maps(records))
        native("cuestParametersCreate",0,ct.byref(parameters))
        assert parameters.value
        # Exact canonical0.2.50 pointer/value convention; no extra JIT threads.
        for attribute,value in ((0,H(int(stream))),(1,H()),(2,H()),(8,ct.c_int32(1))):
            observed=type(value)()
            native("cuestParametersConfigure",0,parameters,attribute,
                   ct.byref(value),ct.sizeof(value))
            native("cuestParametersQuery",0,parameters,attribute,
                   ct.byref(observed),ct.sizeof(observed))
            assert observed.value==value.value
        native("cuestCreate",parameters,ct.byref(handle))
        assert handle.value
        native("cuestSetMathMode",handle,1)
        version=[]
        for label in ("Major","Minor","Patch"):
            out=ct.c_uint32()
            native("cuestGet"+label+"Version",handle,ct.byref(out));version.append(out.value)
        mode=ct.c_int()
        native("cuestGetMathMode",handle,ct.byref(mode))
        assert version==[0,2,50] and mode.value==1
        report.update(cuest_version=version,math_mode=mode.value,jit_compile_threads=1,
                      private_cache_home=os.environ["HOME"],mapped_after_cuest_create=maps(records))
        assert int(cuda(dr.cuCtxGetCurrent()))==context
        state("sealed_cuest_context_passed",version=version,math_mode=mode.value)
    finally:
        # On a failure still release only explicitly acquired resources.
        actions=[]
        if stream is not None:actions.append(("synchronize",lambda:cuda(rt.cudaDeviceSynchronize())))
        if handle.value:actions.append(("cuestDestroy",lambda:native("cuestDestroy",handle)))
        if parameters.value:
            actions.append(("cuestParametersDestroy",lambda:native("cuestParametersDestroy",0,parameters)))
        if allocation is not None:actions.append(("cudaFree",lambda:cuda(rt.cudaFree(allocation))))
        if module is not None:actions.append(("cuModuleUnload",lambda:cuda(dr.cuModuleUnload(module))))
        if stream is not None:actions.append(("cudaStreamDestroy",lambda:cuda(rt.cudaStreamDestroy(stream))))
        primary=sys.exception();errors=[]
        for label,action in actions:
            try:action();report["cleanup"].append(label)
            except BaseException as error:errors.append(error)
        if errors:
            if primary is not None:
                for error in errors:primary.add_note(str(error))
            else:raise errors[0]
    report["mapped_after_cleanup"]=maps(records)
    report["free_after"]=cuda(rt.cudaMemGetInfo())[0]
    assert int(cuda(dr.cuCtxGetCurrent()))==context
    assert "psi4" not in sys.modules
    report["cleanup_complete"]=True
    report["source_sha256"]=digest(__file__)
    state("sealed_gpu_cleanup_passed",cleanup=report["cleanup"])
    return report
