"""Bounded resident singlet HF response, not automatic/coupled FISAPT dispatch.

H+ X = gap*X + 4 J[X] - K[X] - K[X]^T. Native same-gauge (ov,oo,vv)
DFMO factors replace AO JK calls. PCG solves H+Y=W and returns physical x=-Y.
The caller owns the enclosing CUDA scope and the prepared native provider.
"""
from dataclasses import dataclass
import numpy as np

from .sapt_dispersion_cuda_setup import DeviceMath


@dataclass(frozen=True)
class ResponseSolution:
    amplitudes: object
    iterations: int
    absolute_residual: float
    relative_residual: float


class HFResponseOperator:
    """Full occupied response; frozen-core removal is deliberately unsupported.

    Fused exchange is intended for bounded local correctness cases, not a
    protein-scale performance claim. Oversized work fails before native DFMO.
    """
    def __init__(self,provider,occupied,virtual,energies,*,max_exchange_products=64*1024**2):
        self.provider,self.session=provider,provider.session
        m=DeviceMath(self.session)
        if set(energies)!={"o","v"}:
            raise ValueError("Full canonical occupied/virtual energies required")
        for value in (occupied,virtual,*energies.values()):m.require(value)
        if (len(occupied.shape)!=2 or len(virtual.shape)!=2
                or occupied.shape[0]!=provider.nao or virtual.shape[0]!=provider.nao
                or energies["o"].shape!=(occupied.shape[1],) or energies["v"].shape!=(virtual.shape[1],)):
            raise ValueError("Canonical full-occupied/virtual response layouts required")
        self.O,self.V=occupied.shape[1],virtual.shape[1];self.Q=provider.naux
        self.P=self.O*self.V
        if (type(max_exchange_products) is not int or not 1<=max_exchange_products<=64*1024**2
                or self.P>4096 or self.Q*self.P*self.P>max_exchange_products):
            raise ValueError("Response exceeds bounded fused-exchange work admission")
        self.energies=dict(energies)
        provider.validate_basis(provider.primary_basis,provider.auxiliary_basis)
        with self.session.scope():
            error=self.session.empty((1,))
            self.session.launch("single_gap_error",1,128,
                [self.O,self.V,energies["o"],energies["v"],error])
            if float(self.session.download(error)[0])!=0.:
                raise ValueError("Response requires finite positive orbital gaps")
        self.factors=provider.compute_device({"ov":(occupied,virtual),"oo":(occupied,occupied),
                                              "vv":(virtual,virtual)})
        for name,shape in (("ov",(self.Q,self.O,self.V)),("oo",(self.Q,self.O,self.O)),
                           ("vv",(self.Q,self.V,self.V))):
            m.require(self.factors[name])
            if self.factors[name].shape!=shape:
                raise ValueError("Native response factor layout differs")

    def _require(self,trial):
        if self.provider.closed or self.provider.failed:
            raise RuntimeError("Response factors require the original live native provider")
        m=DeviceMath(self.session);m.require(trial)
        for value in (*self.factors.values(),*self.energies.values()):m.require(value)
        if trial.shape!=(self.O,self.V):
            raise ValueError("Wrong canonical occupied-virtual response vector")

    def apply(self,trial):
        self._require(trial);session=self.session
        output=session.empty(trial.shape)
        with session.scope():
            flat=self.factors["ov"].view((self.Q,self.P))
            density=session.gemm(flat,trial.view((self.P,1)))
            coulomb=session.gemm_at_b(flat,density).view(trial.shape)
            session.launch("rhf_response_product",min(65535,self.P),128,
                [self.Q,self.O,self.V,self.factors["ov"],self.factors["oo"],self.factors["vv"],
                 trial,self.energies["o"],self.energies["v"],coulomb,output])
        return output

    def solve(self,rhs,*,rtol=1e-10,atol=1e-12,max_iterations=128):
        """Resident PCG with scalar-only host control and a true-residual check.

        Nonpositive curvature, nonfinite values and nonconvergence fail closed.
        There is no CPU dense/JK solver fallback and no source execution retry.
        """
        self._require(rhs)
        if (type(max_iterations) is not int or not 1<=max_iterations<=4096
                or any(type(v) not in (float,int) or not np.isfinite(v) or v<0 for v in (rtol,atol))
                or max(rtol,atol)==0):
            raise ValueError("Finite nonnegative tolerances/positive iteration limit required")
        session=self.session;m=DeviceMath(session);grid=min(65535,(self.P+255)//256)
        output=session.empty(rhs.shape,zero=True)
        def dot(left,right,dest):
            session.launch("response_dot",1,128,[self.P,left,right,dest])
        with session.scope():
            solution=session.empty(rhs.shape,zero=True)
            residual=m.combine(rhs,rhs,1,0)
            z=session.empty(rhs.shape);direction=session.empty(rhs.shape)
            rho,rho_new,curvature,b2,r2=(session.empty((1,)) for _ in range(5))
            policy=session.upload(np.array([atol,rtol],dtype=np.float64))
            report=session.empty((3,))
            dot(rhs,rhs,b2)
            def metrics():
                session.launch("response_metrics",1,1,[r2,b2,policy,report])
                values=session.download(report)
                if not np.isfinite(values).all():
                    raise RuntimeError("Nonfinite HF response residual")
                return float(values[0]),float(values[1]),bool(values[2]==1.)
            def precondition():
                session.launch("response_precondition",grid,256,
                    [self.O,self.V,self.energies["o"],self.energies["v"],residual,z])
            dot(residual,residual,r2)
            absolute,relative,converged=metrics()
            if converged:return ResponseSolution(output,0,absolute,relative)
            precondition();dot(residual,z,rho)
            session.launch("linear_combination",grid,256,[self.P,1,0,z,z,direction])
            for iteration in range(1,max_iterations+1):
                with session.scope():
                    product=self.apply(direction)
                    dot(direction,product,curvature)
                    observed=float(session.download(curvature)[0])
                    if not np.isfinite(observed) or observed<=0:
                        raise RuntimeError("HF response has nonpositive/nonfinite curvature")
                    session.launch("response_step",grid,256,
                        [self.P,rho,curvature,direction,product,solution,residual])
                    dot(residual,residual,r2)
                    absolute,relative,converged=metrics()
                    if converged:
                        # Recurrence convergence is not sufficient: verify the
                        # actual residual of the positive physical operator.
                        true_residual=m.combine(rhs,self.apply(solution),1,-1)
                        dot(true_residual,true_residual,r2)
                        absolute,relative,converged=metrics()
                        if not converged:
                            raise RuntimeError("HF response true residual exceeds tolerance")
                        session.launch("linear_combination",grid,256,
                            [self.P,-1,0,solution,solution,output])
                        return ResponseSolution(output,iteration,absolute,relative)
                    precondition();dot(residual,z,rho_new)
                    session.launch("response_direction",grid,256,
                        [self.P,rho_new,rho,z,direction])
                    rho,rho_new=rho_new,rho
            raise RuntimeError("HF response did not converge within its iteration limit")
