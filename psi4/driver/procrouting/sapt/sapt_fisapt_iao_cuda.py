"""Resident IAO construction algebra; ordered IBO Jacobi is a separate stage.

The caller supplies native overlap blocks and an already owned FP64 symmetric
power context. All occupied columns define the IAOs; frozen/active separation
belongs to subsequent localization, not this density construction.
"""
from dataclasses import dataclass
import numpy as np
from .sapt_dispersion_cuda_setup import DeviceMath


@dataclass(frozen=True)
class IAOMetricsDevice:
    session: object
    overlaps: object
    power: object
    condition: float
    primary_inverse: object
    minimal_inverse: object


@dataclass(frozen=True)
class IAODevice:
    a: object
    occupied_iao: object
    coefficients: object
    metrics: object
    work_plan: object = None


def allocation_plan(N,M,O,workspace_bytes):
    """Conservative simultaneous allocation ledger for the unchanged IAO recipe.

    DeviceMath products retain intermediates until their scope ends. Include
    all such matrices and the larger of the two native power scratch regions.
    Already-live overlaps/inverse roots and opaque native reservations belong
    to the session, not this incremental estimate.
    """
    if (any(type(x) is not int or x<1 for x in (N,M,O))
            or type(workspace_bytes) is not int or workspace_bytes<0):
        raise ValueError("Positive IAO dimensions and nonnegative solver workspace required")
    retained=8*(8*N*N+7*N*M+6*N*O+3*O*M+2*O*O+2*M*M+O+M+2)
    D=max(O,M)
    solver_scratch=8*(6*D*D+D+5)+4+workspace_bytes
    return {"retained_intermediate_bytes":retained,"native_spectral_scratch_bytes":solver_scratch,
            "peak_increment_bytes":retained+solver_scratch}


def prepare_iao_metrics(overlaps,power,*,condition=1e-7):
    """Reusable overlap inverse roots; exact overlap object and policy retained."""
    session=power.session;m=DeviceMath(session)
    for value in (overlaps.s11,overlaps.s12,overlaps.s22):m.require(value)
    if (len(overlaps.s11.shape)!=2 or overlaps.s11.shape[0]!=overlaps.s11.shape[1]
            or len(overlaps.s22.shape)!=2 or overlaps.s22.shape[0]!=overlaps.s22.shape[1]
            or overlaps.s12.shape!=(overlaps.s11.shape[0],overlaps.s22.shape[0])
            or type(condition) not in (float,int) or not np.isfinite(condition) or not 0<=condition<=1):
        raise ValueError("Common-session symmetric primary/minimal and rectangular overlaps required")
    h1=power.inverse_sqrt(overlaps.s11,condition=condition).matrix
    h2=power.inverse_sqrt(overlaps.s22,condition=condition).matrix
    return IAOMetricsDevice(session,overlaps,power,float(condition),h1,h2)


def build_iao_device(coefficients,metrics):
    """Exact local2.cc IAO recipe using native powers and CUDA/cuBLAS products.

    Returns AO IAOs A and canonical occupied IAO coordinates C.T*S11*A.
    No overlap, Gram, eigenvector or density matrix is downloaded or generated
    on CPU. Native scalar info/validation controls belong to the power context.
    """
    session=metrics.session;m=DeviceMath(session)
    if metrics.power.session is not session:raise ValueError("IAO power context changed session")
    s11,s12,s22=metrics.overlaps.s11,metrics.overlaps.s12,metrics.overlaps.s22
    h1,h2=metrics.primary_inverse,metrics.minimal_inverse
    for value in (coefficients,s11,s12,s22,h1,h2):m.require(value)
    N,M=s12.shape
    if (len(coefficients.shape)!=2 or coefficients.shape[0]!=N or coefficients.shape[1]>M
            or s11.shape!=(N,N) or s22.shape!=(M,M) or h1.shape!=(N,N) or h2.shape!=(M,M)):
        raise ValueError("Full occupied space/common overlap support required for IAO construction")
    plan=allocation_plan(N,M,coefficients.shape[1],metrics.power.workspace_limit_bytes)
    session.check_memory(plan["peak_increment_bytes"])
    output=session.empty((N,M));occupied_iao=session.empty((coefficients.shape[1],M))
    with session.scope():
        t1=session.gemm(h2,m.window(s12,transpose=True))
        t2=session.gemm(h1,session.gemm_at_b(t1,session.gemm(t1,coefficients)))
        gram=session.gemm_at_b(t2,t2)
        inverse=metrics.power.inverse_sqrt(gram,condition=metrics.condition).matrix
        tilde=m.product(h1,t2,inverse)
        density=session.gemm(coefficients,m.window(coefficients,transpose=True))
        tilde_density=session.gemm(tilde,m.window(tilde,transpose=True))
        operator=m.terms((1,session.gemm(h1,h1)),(2,m.product(density,s11,tilde_density)),
                         (-1,density),(-1,tilde_density))
        an=session.gemm(operator,s12)
        metric=session.gemm_at_b(an,session.gemm(s11,an))
        inverse=metrics.power.inverse_sqrt(metric,condition=metrics.condition).matrix
        a=session.gemm(an,inverse)
        b=session.gemm_at_b(coefficients,session.gemm(s11,a))
        for value,target in ((a,output),(b,occupied_iao)):
            session.launch("linear_combination",min(65535,(value.size+255)//256),256,
                           [value.size,1,0,value,value,target])
    return IAODevice(output,occupied_iao,coefficients,metrics,plan)
