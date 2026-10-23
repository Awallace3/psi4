"""Explicit CPU HF fragment response reference; no GPU or dense Hessian.

Native CPU JK contracts AO densities independently of the device DFMO/GEMM
operator. This is selected only by the opt-in HF coupled fragment flag;
ordinary/default and non-HF routes retain their previous behavior.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CPUResponseSolution:
    amplitudes: object
    iterations: int
    absolute_residual: float
    relative_residual: float


def solve_positive(apply, gap, rhs, *, rtol=1e-10, atol=1e-12, max_iterations=128):
    """Solve H+Y=W by independent CPU PCG and return physical amplitudes -Y."""
    if (not callable(apply) or not isinstance(gap,np.ndarray) or not isinstance(rhs,np.ndarray)
            or gap.dtype!=np.float64 or rhs.dtype!=np.float64 or gap.shape!=rhs.shape
            or gap.ndim!=2 or not gap.size or not np.isfinite(gap).all()
            or not np.isfinite(rhs).all() or np.any(gap<=0)
            or any(type(x) not in (int,float) or not np.isfinite(x) or x<0 for x in (rtol,atol))
            or max(rtol,atol)==0 or type(max_iterations) is not int
            or not 1<=max_iterations<=4096):
        raise ValueError("Finite FP64 response/gaps and explicit positive solver policy required")
    y=np.zeros_like(rhs);residual=rhs.copy()
    norm=float(np.linalg.norm(rhs));threshold=max(atol,rtol*norm)
    def metrics(r):
        absolute=float(np.linalg.norm(r))
        if not np.isfinite(absolute):
            raise RuntimeError("Nonfinite independent CPU response residual")
        return absolute,absolute/norm if norm else absolute
    absolute,relative=metrics(residual)
    if absolute<=threshold:return CPUResponseSolution(-y,0,absolute,relative)
    z=residual/gap;direction=z.copy();rho=float(np.vdot(residual,z).real)
    for iteration in range(1,max_iterations+1):
        product=np.asarray(apply(direction))
        if product.dtype!=np.float64 or product.shape!=rhs.shape or not np.isfinite(product).all():
            raise RuntimeError("Invalid independent CPU Hessian action")
        curvature=float(np.vdot(direction,product).real)
        if not np.isfinite(curvature) or curvature<=0 or not np.isfinite(rho) or rho<=0:
            raise RuntimeError("Independent CPU response has nonpositive/nonfinite curvature")
        alpha=rho/curvature;y+=alpha*direction;residual-=alpha*product
        absolute,relative=metrics(residual)
        if absolute<=threshold:
            true=np.asarray(rhs-apply(y))
            absolute,relative=metrics(true)
            if absolute>threshold:
                raise RuntimeError("Independent CPU response true residual exceeds tolerance")
            return CPUResponseSolution(-y,iteration,absolute,relative)
        z=residual/gap;next_rho=float(np.vdot(residual,z).real)
        if not np.isfinite(next_rho) or next_rho<=0:
            raise RuntimeError("Independent CPU response preconditioner failed")
        direction=z+(next_rho/rho)*direction;rho=next_rho
    raise RuntimeError("Independent CPU response did not converge within its iteration limit")


class NativeCPUResponse:
    """Owned CPU JK matrix-free Hessian, never a CUDA failure fallback."""
    def __init__(self,core,primary,auxiliary,occupied,virtual,eo,ev,*,memory_doubles):
        if core.get_global_option("USE_CUEST"):
            raise ValueError("Independent CPU response requires USE_CUEST=false")
        co,cv,eo,ev=(np.asarray(x) for x in (occupied,virtual,eo,ev))
        if (co.dtype!=np.float64 or cv.dtype!=np.float64 or eo.dtype!=np.float64
                or ev.dtype!=np.float64 or co.ndim!=2 or cv.ndim!=2
                or co.shape[0]!=cv.shape[0] or eo.shape!=(co.shape[1],)
                or ev.shape!=(cv.shape[1],) or min(co.shape+cv.shape)<1
                or not all(np.isfinite(x).all() for x in (co,cv,eo,ev))
                or type(memory_doubles) is not int or memory_doubles<1):
            raise ValueError("Full finite CPU occupied/virtual spaces and explicit JK memory required")
        self.gap=ev[None,:]-eo[:,None]
        if np.any(self.gap<=0):raise ValueError("Independent CPU response requires positive gaps")
        self.core,self.co,self.cv=core,co.copy(),cv.copy()
        self.jk=core.JK.build(primary,auxiliary,jk_type="DF")
        self.closed=False;self.calls=0
        self.jk.set_memory(memory_doubles)
        self.jk.initialize()

    def apply(self,trial):
        if self.closed:raise RuntimeError("Independent CPU response operator is closed")
        if (not isinstance(trial,np.ndarray) or trial.dtype!=np.float64
                or trial.shape!=self.gap.shape or not np.isfinite(trial).all()):
            raise ValueError("Finite full CPU response trial required")
        self.jk.C_clear()
        self.jk.C_left_add(self.core.Matrix.from_array(self.co))
        self.jk.C_right_add(self.core.Matrix.from_array(self.cv@trial.T))
        self.jk.compute()
        js,ks=self.jk.J(),self.jk.K()
        if len(js)!=1 or len(ks)!=1:
            raise RuntimeError("Independent CPU JK returned the wrong density count")
        j,k=np.asarray(js[0].np),np.asarray(ks[0].np)
        self.calls+=1
        return self.gap*trial+self.co.T@(4*j-k-k.T)@self.cv

    def __enter__(self):return self

    def __exit__(self,*exception):
        if not self.closed:
            self.closed=True
            self.jk.finalize()
        return False


def partition_response(core,primary,auxiliary,occupied,virtual,eo,ev,sources,
                       rotation,target,target_exchange,*,memory_doubles):
    """All source columns with canonical solve BEFORE the original IBO rotation."""
    sources,rotation,target,target_exchange=(np.asarray(x) for x in
                                            (sources,rotation,target,target_exchange))
    O,V=occupied.shape[1],virtual.shape[1]
    if (sources.dtype!=np.float64 or sources.ndim!=3 or not sources.shape[0] or sources.shape[1:]!=(O,V)
            or rotation.dtype!=np.float64 or rotation.shape!=(O,O)
            or any(x.dtype!=np.float64 or x.shape!=(O,V) for x in (target,target_exchange))
            or not all(np.isfinite(x).all() for x in (sources,rotation,target,target_exchange))):
        raise ValueError("Complete CPU source/target columns and one occupied gauge required")
    np.testing.assert_allclose(rotation.T@rotation,np.eye(O),rtol=1e-10,atol=1e-10)
    amplitudes=np.empty_like(sources);diagnostics=[]
    with NativeCPUResponse(core,primary,auxiliary,occupied,virtual,eo,ev,
                           memory_doubles=memory_doubles) as operator:
        for column,source in enumerate(sources):
            solution=solve_positive(operator.apply,operator.gap,source)
            amplitudes[column]=solution.amplitudes
            diagnostics.append({"column":column,"iterations":solution.iterations,
                "absolute_residual":solution.absolute_residual,
                "relative_residual":solution.relative_residual})
    induction=2*np.einsum("il,cir,lr->lc",rotation,amplitudes,target,optimize=True)
    exchange=2*np.einsum("il,cir,lr->lc",rotation,amplitudes,target_exchange,optimize=True)
    return induction,exchange,tuple(diagnostics)
