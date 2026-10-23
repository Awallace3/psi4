"""Memory-budgeted resident singlet HF response with exact batched exchange.

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

    Retains the same-gauge native factors and uses Q-blocked FP64 GEMMs for
    both exchange terms. Admission is by explicit resident/workspace budgets,
    not a small-system P² work cap. Oversized memory fails before native DFMO.
    """
    def __init__(self,provider,occupied,virtual,energies,*,
                 max_workspace_bytes=256*1024**2,max_factor_bytes=None,
                 max_exchange_products=None):
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
        if min(self.Q,self.O,self.V)<1 or max(self.Q,self.O,self.V,self.P)>2**31-1:
            raise ValueError("Response dimensions exceed the native GEMM ABI")
        # An explicit legacy work ceiling remains a caller opt-in, never a
        # hidden admission policy for the scalable implementation.
        if max_exchange_products is not None:
            if (type(max_exchange_products) is not int or max_exchange_products<1
                    or self.Q*self.P*self.P>max_exchange_products):
                raise ValueError("Response exceeds explicit exchange work admission")
        if type(max_workspace_bytes) is not int or max_workspace_bytes<1:
            raise ValueError("Response workspace admission needs a positive byte budget")
        fixed=8*(2*self.P+self.Q)  # Coulomb, exchange accumulator and density.
        per_q=8*(3*self.P+self.O*self.O)  # T1, K1, T2 and K2 batched outputs.
        if max_workspace_bytes<fixed+per_q:
            raise ValueError("Response workspace admission cannot fit one Q block")
        self.q_batch=min(self.Q,(max_workspace_bytes-fixed)//per_q)
        self.factor_bytes=8*self.Q*(self.P+self.O*self.O+self.V*self.V)
        workspace=fixed+self.q_batch*per_q
        # Reserve PCG vectors, raw coefficient packing and one apply output in
        # addition to the bounded exchange workspace. Native DFMO allocations
        # remain independently checked by the owning session at every malloc.
        reserve=workspace+8*(32*self.P+3*provider.nao*(self.O+self.V))
        if max_factor_bytes is not None and (
                type(max_factor_bytes) is not int or max_factor_bytes<1
                or self.factor_bytes>max_factor_bytes):
            raise ValueError("Response resident-factor admission exceeds byte budget")
        self.session.check_memory(self.factor_bytes+reserve)
        self.work_plan={"algorithm":"native-fp64-Q-blocked-strided-batched",
            "resident_factor_bytes":self.factor_bytes,"workspace_bytes":workspace,
            "q_batch":self.q_batch,"reserved_total_bytes":self.factor_bytes+reserve}
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
            exchange=session.empty(trial.shape,zero=True)
            for start in range(0,self.Q,self.q_batch):
                count=min(self.q_batch,self.Q-start)
                with session.scope():
                    def slab(name,left,right):
                        return self.factors[name].view((count,left,right),
                            offset=8*start*left*right)
                    ov=slab("ov",self.O,self.V)
                    oo=slab("oo",self.O,self.O)
                    vv=slab("vv",self.V,self.V)
                    # K1_ia = sum_Qjb OO_Qij X_jb VV_Qab.
                    first=session.gemm_batched(
                        session.gemm_batched(oo,trial),vv,transpose_right=True)
                    # K2_ia = sum_Qjb OV_Qib X_jb OV_Qja, NOT K1.T.
                    second=session.gemm_batched(
                        session.gemm_batched(ov,trial,transpose_right=True),ov)
                    session.launch("response_accumulate_exchange",
                        min(65535,(self.P+255)//256),256,
                        [count,self.P,first,second,exchange])
            session.launch("response_finish",min(65535,(self.P+255)//256),256,
                [self.O,self.V,self.energies["o"],self.energies["v"],
                 trial,coulomb,exchange,output])
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
