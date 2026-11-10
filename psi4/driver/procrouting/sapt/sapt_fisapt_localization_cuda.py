"""Ordered resident IBO Jacobi/reordering/weights, no host numerical fallback.

Matches local2.cc's pair order, p2/p4 angle, strict gradient convergence and
literal permutation orientation. Initial IAO construction is a separate native
overlap/cuSOLVER module. Only scalar convergence/validation is downloaded.
"""
from dataclasses import dataclass
import numpy as np
from .sapt_dispersion_cuda_setup import DeviceMath
from .sapt_fisapt_cuda import validate_rotations


@dataclass(frozen=True)
class JacobiDevice:
    u: object
    f: object
    q: object
    occupied_iao: object
    permutation: object
    converged: bool
    iterations: int
    history: tuple
    nfrozen: int
    work_plan: object = None


@dataclass(frozen=True)
class IBODevice:
    l: object
    a: object
    jacobi: object
    iao: object
    @property
    def u(self):return self.jacobi.u
    @property
    def f(self):return self.jacobi.f
    @property
    def q(self):return self.jacobi.q


def jacobi_localize_device(session,coordinates,energies,atom_indices,*,nfrozen=0,power=4,
                           convergence=1e-12,max_iterations=50,require_converged=True,
                           max_pair_updates=2_000_000):
    m=DeviceMath(session);m.require(coordinates);m.require(energies)
    atoms=np.asarray(atom_indices)
    if (len(coordinates.shape)!=2 or energies.shape!=(coordinates.shape[0],)
            or atoms.dtype!=np.int64 or atoms.shape!=(coordinates.shape[1],)
            or np.any(atoms<0) or np.any(atoms>=coordinates.shape[1])
            or not np.array_equal(np.unique(atoms),np.arange(int(atoms.max())+1))
            or type(nfrozen) is not int or not 0<=nfrozen<=coordinates.shape[0]
            or type(power) is not int or power not in (2,4)
            or type(convergence) not in (float,int) or not np.isfinite(convergence) or convergence<0
            or type(max_iterations) is not int or not 1<=max_iterations<=1000
            or type(require_converged) is not bool):
        raise ValueError("Full occupied IAO coordinates, contiguous int64 atom rows and explicit IBO policy required")
    O,M=coordinates.shape;K=int(atoms.max())+1
    pair_count=nfrozen*(nfrozen-1)//2+(O-nfrozen)*(O-nfrozen-1)//2
    if (type(max_pair_updates) is not int or max_pair_updates<1
            or pair_count*max_iterations>max_pair_updates
            or max(O,M,K)>2**31-1):
        raise ValueError("Ordered IBO pair-update work exceeds bounded admission")
    peak_bytes=8*(9*O*O+3*O*M+K*O+O+M+6)
    session.check_memory(peak_bytes)
    plan={"peak_increment_bytes":peak_bytes,"pair_updates_per_sweep":pair_count,
          "maximum_pair_updates":pair_count*max_iterations,"pair_update_budget":max_pair_updates}
    output_u,output_f=session.empty((O,O)),session.empty((O,O))
    output_b,output_q=session.empty((O,M)),session.empty((K,O))
    output_p=session.empty((O,O))
    history=[];converged=False
    with session.scope():
        error=session.empty((1,))
        for value in (coordinates,energies):
            session.launch("matrix_error",1,128,[value.size,value,error])
            if float(session.download(error)[0])>=1e300:
                raise ValueError("IBO coordinates/energies must be finite and bounded")
        labels=session.upload_indices(atoms)
        policy=session.upload(np.array([convergence],dtype=np.float64))
        rows=m.combine(coordinates,coordinates,1,0)
        rotation=session.empty((O,O))
        session.launch("set_identity",min(65535,(O*O+255)//256),256,[O,rotation])
        gradient=session.empty((1,),zero=True);stats=session.empty((3,))
        for iteration in range(1,max_iterations+1):
            session.launch("ibo_objective",1,128,[O,M,K,power,labels,rows,stats])
            # Native ordered trajectory: frozen block first, active block next,
            # ascending i then j. Stream metadata rather than storing O² pairs.
            for start,stop in ((0,nfrozen),(nfrozen,O)):
                for i in range(start,stop):
                    for j in range(start,i):
                        session.launch("ibo_pair",1,128,[O,M,K,power,i,j,labels,rows,rotation,gradient])
            session.launch("ibo_sweep_finish",1,1,[policy,gradient,stats])
            values=session.download(stats)
            if not np.isfinite(values).all():raise RuntimeError("Nonfinite IBO objective/gradient")
            history.append({"iteration":iteration,"objective":float(values[0]),"gradient":float(values[1])})
            converged=bool(values[2]==1.)
            if converged:break
        if require_converged and not converged:
            raise RuntimeError("IBO Jacobi did not converge within its declared sweep limit")
        diagonal=session.empty((O,))
        session.launch("ibo_fock_diagonal",min(65535,(O+255)//256),256,[O,energies,rotation,diagonal])
        permutation=session.empty((O,O),zero=True)
        session.launch("ibo_permutation",min(65535,(O+255)//256),256,[O,nfrozen,diagonal,permutation])
        u=session.gemm(m.window(rotation,transpose=True),permutation)
        b=session.gemm_at_b(permutation,rows)
        weighted=session.empty((O,O))
        session.launch("scale_eigenvector_rows",min(65535,(O*O+255)//256),256,[O,u,energies,weighted])
        f=session.gemm_at_b(u,weighted)
        session.launch("ibo_charges",min(65535,(K*O+255)//256),256,[O,M,K,labels,b,output_q])
        for value,output in ((u,output_u),(f,output_f),(b,output_b),(permutation,output_p)):
            session.launch("linear_combination",min(65535,(value.size+255)//256),256,
                           [value.size,1,0,value,value,output])
    return JacobiDevice(output_u,output_f,output_q,output_b,output_p,converged,len(history),
                        tuple(history),nfrozen,plan)


def localize_iao_device(iao,energies,atom_indices,**policy):
    """Complete AO IBO outputs from device IAOs; convergence is mandatory."""
    if policy.get("require_converged",True) is not True:
        raise ValueError("AO IBO production assembly requires convergence")
    session=iao.metrics.session;m=DeviceMath(session)
    for value in (iao.coefficients,iao.a,iao.occupied_iao,iao.metrics.overlaps.s11):m.require(value)
    with session.scope():
        gram=session.gemm_at_b(iao.coefficients,session.gemm(iao.metrics.overlaps.s11,iao.coefficients))
        error=session.empty((1,))
        session.launch("rotation_error",1,128,[iao.coefficients.shape[1],gram,error])
        if float(session.download(error)[0])>1e-10:
            raise ValueError("IBO input occupied coefficients are not S-orthonormal")
    value=jacobi_localize_device(session,iao.occupied_iao,energies,atom_indices,**policy)
    validate_rotations(session,value.u,value.u)
    with session.scope():
        error=session.empty((1,))
        session.launch("ibo_population_error",1,128,[value.u.shape[0],value.q.shape[0],value.q,error])
        if float(session.download(error)[0])>1e-8:
            raise ValueError("IAO occupied populations do not span the supplied occupied space")
    return IBODevice(session.gemm(iao.coefficients,value.u),iao.a,value,iao)
