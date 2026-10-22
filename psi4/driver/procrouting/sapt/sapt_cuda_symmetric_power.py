"""Owned FP64 cuSOLVER symmetric inverse square roots for GPU IAO groundwork.

No eigensolver/CPU arithmetic fallback. The explicit cuEST plan owner retains
the solver handle and supplies the already frozen CUDA mapped-file admission.
This is not a GPU localization implementation or default-driver qualification.
"""
import ctypes as ct
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from . import sapt_cuest250_identity as identity
from .sapt_cuest_plan_owner import DiagnosticPlanOwner
from .sapt_cuda_runtime import library
from .sapt_dispersion_cuda_setup import DeviceMath


@dataclass(frozen=True)
class SymmetricPowerDevice:
    matrix: object
    eigenvalues: object
    rank: object


def reconstruct_inverse_sqrt(session,transpose_vectors,eigenvalues,*,condition=1e-7):
    """From native eigenvectors-as-C-V.T; strict relative spectral cutoff."""
    m=DeviceMath(session);m.require(transpose_vectors);m.require(eigenvalues)
    if (len(eigenvalues.shape)!=1 or transpose_vectors.shape!=(eigenvalues.size,eigenvalues.size)
            or type(condition) not in (float,int) or not np.isfinite(condition) or not 0<=condition<=1):
        raise ValueError("Square FP64 eigensystem and finite relative condition0..1 required")
    N=eigenvalues.size
    output=session.empty((N,N));rank=session.empty((1,))
    with session.scope():
        policy=session.upload(np.array([condition],dtype=np.float64))
        weights=session.empty((N,));stats=session.empty((2,))
        session.launch("inverse_sqrt_weights",1,128,[N,eigenvalues,policy,weights,stats])
        if float(session.download(stats)[1])!=0.:
            raise ValueError("Native symmetric eigensystem contains nonfinite/oversized eigenvalues")
        session.launch("linear_combination",1,256,[1,1,0,stats.view((1,)),stats.view((1,)),rank])
        weighted=session.empty((N,N))
        session.launch("scale_eigenvector_rows",min(65535,(N*N+255)//256),256,
                       [N,transpose_vectors,weights,weighted])
        matrix=session.gemm_at_b(transpose_vectors,weighted)
        session.launch("linear_combination",min(65535,(N*N+255)//256),256,
                       [N*N,1,0,matrix,matrix,output])
    return output,rank


class CuSolverSymmetricPower:
    """Native handle lives in the parent's reverse-order owned resource graph."""
    def __init__(self,owner,*,workspace_limit_bytes=128*1024**2,max_dimension=512):
        if not isinstance(owner,DiagnosticPlanOwner):
            raise TypeError("An explicit admitted native plan owner is required")
        owner._check_construction();owner.assert_live()
        if (type(workspace_limit_bytes) is not int or workspace_limit_bytes<8
                or type(max_dimension) is not int or not 1<=max_dimension<=512):
            raise ValueError("Bounded positive cuSOLVER workspace/dimension required")
        self.owner,self.session=owner,owner.session
        self.workspace_limit_bytes,self.max_dimension=workspace_limit_bytes,max_dimension
        records=[r for r in owner.api.manifest["components"] if Path(r["path"]).name.startswith("libcusolver.so.")]
        if len(records)!=1:raise ValueError("One explicit frozen cuSOLVER DSO required")
        self.native,self.path=library(records[0]["path"],records[0]["sha256"])
        self._bind()
        self.handle=owner._new_native("FP64 symmetric cuSOLVER handle",
                                     self.native.cusolverDnCreate,self.native.cusolverDnDestroy,
                                     (self.native,))
        try:
            owner.api.status("cuSOLVER stream",self.native.cusolverDnSetStream(
                owner._raw(self.handle),ct.c_void_p(int(self.session.stream))))
        except BaseException:
            owner._invalidate();raise

    def _bind(self):
        self.owner.api.assert_identity()
        V,I,D=ct.c_void_p,ct.c_int,ct.POINTER(ct.c_double)
        specs={"cusolverDnCreate":[ct.POINTER(V)],"cusolverDnDestroy":[V],"cusolverDnSetStream":[V,V],
            "cusolverDnDsyevd_bufferSize":[V,I,I,I,D,I,D,ct.POINTER(I)],
            "cusolverDnDsyevd":[V,I,I,I,D,I,D,D,I,ct.POINTER(I)]}
        rows=identity.mapped_rows(Path("/proc/self/maps").read_text())
        for name,args in specs.items():
            fn=getattr(self.native,name);pointer=ct.cast(fn,V).value
            owners={path for lo,hi,path,_,_ in rows if lo<=pointer<hi}
            identity.require(owners=={self.path},"cuSOLVER symbol outside frozen DSO: "+name)
            fn.argtypes,fn.restype=args,I

    def inverse_sqrt(self,value,*,condition=1e-7):
        owner,session=self.owner,self.session;owner.assert_live()
        m=DeviceMath(session);m.require(value)
        if (len(value.shape)!=2 or value.shape[0]!=value.shape[1]
                or value.shape[0]>self.max_dimension
                or type(condition) not in (float,int) or not np.isfinite(condition) or not 0<=condition<=1):
            raise ValueError("Bounded square symmetric input required")
        N=value.shape[0]
        eigenvalues=session.empty((N,))
        result_matrix,result_rank=session.empty((N,N)),session.empty((1,))
        try:
            self._bind()
            with session.scope():
                residual=m.combine(value,m.window(value,transpose=True),1,-1)
                error=session.empty((1,))
                session.launch("matrix_error",1,128,[residual.size,residual,error])
                mismatch=float(session.download(error)[0])
                session.launch("matrix_error",1,128,[value.size,value,error])
                scale=float(session.download(error)[0])
                if not np.isfinite(scale) or scale>=1e300 or mismatch>1e-11*max(1.,scale):
                    raise ValueError("Symmetric power input is nonfinite or asymmetric")
                vectors=m.combine(value,value,1,0)
                raw=ct.cast(vectors.pointer,ct.POINTER(ct.c_double))
                values=ct.cast(eigenvalues.pointer,ct.POINTER(ct.c_double))
                lwork=ct.c_int()
                # Compiler-witnessed EIG_MODE_VECTOR1 / FILL_MODE_LOWER0.
                owner.api.status("syevd buffer size",self.native.cusolverDnDsyevd_bufferSize(
                    owner._raw(self.handle),1,0,N,raw,N,values,ct.byref(lwork)))
                if not 1<=lwork.value<=self.workspace_limit_bytes//8:
                    raise MemoryError("cuSOLVER workspace exceeds explicit bound")
                work=session.empty((lwork.value,));info=session.empty((1,),dtype=np.int32)
                session.rt(session.runtime.cudaMemsetAsync(info.pointer,255,info.nbytes,session.stream))
                owner.api.status("FP64 syevd",self.native.cusolverDnDsyevd(
                    owner._raw(self.handle),1,0,N,raw,N,values,
                    ct.cast(work.pointer,ct.POINTER(ct.c_double)),lwork.value,
                    ct.cast(info.pointer,ct.POINTER(ct.c_int))))
                observed=int(session.download(info)[0])
                if observed!=0:raise RuntimeError(f"Native symmetric eigensolver info={observed}")
                output,rank=reconstruct_inverse_sqrt(session,vectors,eigenvalues,condition=condition)
                # Outputs were allocated BEFORE the eigen-workspace scope.
                session.launch("linear_combination",min(65535,(N*N+255)//256),256,
                               [N*N,1,0,output,output,result_matrix])
                session.launch("linear_combination",1,256,[1,1,0,rank,rank,result_rank])
            owner.assert_live()
            return SymmetricPowerDevice(result_matrix,eigenvalues,result_rank)
        except BaseException:
            owner._invalidate();raise
