"""Bounded HF fragment workflow: ONE session and ONE resident IBO gauge.

Raw upstream HF arrays are transport inputs, not prepared factors or localized
orbitals. All localization, integral, response and term numerics are native
FP64 GPU work. Only scalar controls and final output formatting use the host.
This is unscaled, two-fragment C-assignment, not general/link FISAPT.
"""
import time
import numpy as np

from .sapt_cuest_dfmo_cuda import CuESTDFMOProvider
from .sapt_cuest_overlap_cuda import CuESTOverlapBlocks
from .sapt_cuest_potential_cuda import CuESTPointPotential
from .sapt_cuda_symmetric_power import CuSolverSymmetricPower
from .sapt_dispersion_cuda import blocked_device_dispersion
from .sapt_dispersion_cuda_setup import DeviceMath, prepare_hf
from .sapt_fisapt_iao_cuda import prepare_iao_metrics, build_iao_device
from .sapt_fisapt_localization_cuda import localize_iao_device
from .sapt_fisapt_elst_cuda import felst_device
from .sapt_fisapt_exch_cuda import fexch_device
from .sapt_fisapt_induction_cuda import coupled_find_device


HF_FIELDS = ("S","D_A","P_A","V_A","J_A","K_A","D_B","P_B","V_B","J_B","K_B","K_O",
             "Cocc_A","Cocc_B","Cvir_A","Cvir_B")
IND_FIELDS = ("S","D_A","D_B","V_A","V_B","J_A","J_B","K_A","K_B","J_O","K_O")


def fragment_terms(provider, rifit, minimal, matrices, energies, xyz, za, zb, *,
                   frozen=(0,0), block_size=24):
    """Consume an unprepared owned JKFIT provider; retain one gauge through RI.

    Returns final host arrays/scalars only. Downloaded U/L is output for
    interpretation/oracle checks, never an input to another production term.
    The caller closes the original provider before destroying its plan.
    """
    if not isinstance(provider,CuESTDFMOProvider):
        raise TypeError("Strict owned cuEST0.2.50 provider required for GPU FISAPT")
    owner = provider.owned_plan()
    session = provider.session
    if type(block_size) is not int or block_size<1 or len(frozen)!=2:
        raise ValueError("Positive block size and two frozen occupied counts required")
    if (set(matrices)!=set(HF_FIELDS)|{"J_O"} or set(energies)!=set("abrs")
            or any(type(n) is not int or n<0 for n in frozen)):
        raise ValueError("Complete raw HF cache and nonnegative frozen counts required")
    for name,value in matrices.items():
        if (not isinstance(value,np.ndarray) or value.dtype!=np.float64
                or value.ndim!=2 or value.shape[0]!=provider.nao
                or value.shape[1]<1 or not np.isfinite(value).all()
                or (name not in ("Cocc_A","Cocc_B","Cvir_A","Cvir_B")
                    and value.shape[1]!=provider.nao)):
            raise ValueError(f"Invalid raw HF matrix {name}")
    for name,key in zip("abrs",("Cocc_A","Cocc_B","Cvir_A","Cvir_B")):
        value=energies[name]
        if (not isinstance(value,np.ndarray) or value.dtype!=np.float64
                or value.shape!=(matrices[key].shape[1],) or not np.isfinite(value).all()):
            raise ValueError("Complete finite native FP64 orbital energies required")
    A,B=(matrices["Cocc_"+side].shape[1] for side in ("A","B"))
    if frozen[0]>=A or frozen[1]>=B:
        raise NotImplementedError("Bounded GPU FISAPT requires nonempty active occupied spaces")
    N=owner.primary_snapshot.ncenter
    if any(not isinstance(v,np.ndarray) or v.dtype!=np.float64
           or not np.isfinite(v).all() for v in (xyz,za,zb)):
        raise ValueError("Finite FP64 coordinates/nuclear assignments required")
    if xyz.shape!=(N,3) or za.shape!=(N,) or zb.shape!=(N,):
        raise ValueError("Full dimer atom/charge metadata required")
    started=time.perf_counter()
    # All native resources precede any temporary CUDA allocation scopes.
    potential=CuESTPointPotential(owner,num_charges=1)
    overlap=CuESTOverlapBlocks(owner,minimal)
    power=CuSolverSymmetricPower(owner)
    atoms=np.array(overlap.local_atom_indices,dtype=np.int64)
    provider.prepare_device_compute()
    with provider.sibling_provider(rifit) as ri:
        if ri.session is not session:
            raise RuntimeError("JKFIT/RIFIT must share the original GPU session")
        with session.scope():
            m=DeviceMath(session)
            raw={name:session.upload(value) for name,value in matrices.items()}
            ed={name:session.upload(value) for name,value in energies.items()}
            coords,charges_a,charges_b=(session.upload(v) for v in (xyz,za,zb))
            metrics=prepare_iao_metrics(overlap.compute(),power,condition=1e-7)
            ibos={}
            for side,key,nf in zip(("A","B"),"ab",frozen):
                iao=build_iao_device(raw["Cocc_"+side],metrics)
                ibos[side]=localize_iao_device(iao,ed[key],atoms,nfrozen=nf,
                    power=4,convergence=1e-12,max_iterations=50)
            la,lb=ibos["A"].l,ibos["B"].l
            state=felst_device(provider,potential,la,lb,coords,charges_a,charges_b,
                               jkfit_basis=provider.auxiliary_basis)
            exch=fexch_device(state,la,lb,raw["Cvir_A"],raw["Cvir_B"],
                {n:raw[n] for n in ("S","V_A","J_A","V_B","J_B")},natoms=N)
            induction=coupled_find_device(state,potential,
                {s:raw["Cocc_"+s] for s in ("A","B")},
                {s:raw["Cvir_"+s] for s in ("A","B")},ed,
                {n:raw[n] for n in IND_FIELDS},coords,charges_a,charges_b)
            rotations=[]
            for side,nf in zip(("A","B"),frozen):
                u=ibos[side].u;O=u.shape[0];active=O-nf
                rotations.append(m.window(u.view((active,O),offset=nf*O*8),nf,active))
            spaces,corrections,eps=prepare_hf(session,{n:matrices[n] for n in HF_FIELDS},
                energies,frozen_a=frozen[0],frozen_b=frozen[1])
            dispersion=blocked_device_dispersion(ri,spaces,corrections,eps,
                block_size=block_size,rotations=tuple(rotations))
            device_outputs={"Elst_AB":state.matrix,"Exch_AB":exch.matrix,
                "IndAB_AB":induction.coupled.combined_ab,"IndBA_AB":induction.coupled.combined_ba}
            for prefix,value in (("Ind20u",induction.uncoupled.ind20u),
                                 ("ExchInd20u",induction.uncoupled.exch_ind20u),
                                 ("Ind20r",induction.coupled.induction),
                                 ("ExchInd20r",induction.coupled.exchange_induction)):
                device_outputs[prefix+"_AB"]=value.ab
                device_outputs[prefix+"_BA"]=value.ba
            for side in ("A","B"):
                device_outputs["Qocc0"+side]=ibos[side].q
                device_outputs["Uocc0"+side]=ibos[side].u
                device_outputs["Locc_"+side]=ibos[side].l
            outputs={name:session.download(value) for name,value in device_outputs.items()}
            localization={s:{"iterations":ibos[s].jacobi.iterations,
                             "converged":ibos[s].jacobi.converged} for s in ("A","B")}
            response=induction.response_diagnostics
        ri_report=ri.report()
    # Final output layout only; pair combination and all sums already ran on GPU.
    full=np.zeros((N+A+1,N+B+1),dtype=np.float64)
    pairs=dispersion["Disp0 occupied pairs"]
    full[N+frozen[0]:N+A,N+frozen[1]:N+B]=pairs
    outputs["Disp_AB"]=full
    outputs["Disp20 occupied pairs"]=dispersion["Disp20 occupied pairs"]
    outputs["Exch-Disp20 occupied pairs"]=dispersion["Exch-Disp20 occupied pairs"]
    if any(not np.isfinite(value).all() for value in outputs.values()):
        raise FloatingPointError("Nonfinite final GPU fragment outputs")
    return outputs,{"Disp20,u":dispersion["Disp20,u"],
        "Exch-Disp20,u":dispersion["Exch-Disp20,u"],
        "FISAPT execution":{"backend":"CUDA_CUEST_SHARED_IBO_EXPERIMENTAL",
            "same_session":True,"gpu_localization":localization,"response":response,
            "jkfit_producer":provider.report(),"rifit_producer":ri_report,
            "dispersion":dispersion["Disp0 execution"],
            "workflow_seconds_excluding_outer_factory_cleanup":time.perf_counter()-started,
            "raw_hf_upload_bytes":sum(v.nbytes for v in matrices.values())+
                sum(v.nbytes for v in energies.values())+
                sum(matrices[n].nbytes for n in HF_FIELDS)+sum(v.nbytes for v in energies.values()),
            "scope":"Two unlinked spherical HF fragments, C-assignment, unscaled S^2 exchange, "
                    "signed coupled/uncoupled induction and dispersion; one resident IBO gauge. "
                    "Raw upstream HF arrays are host inputs; final output layout is host formatting. "
                    "Not a zero-roundtrip SCF handoff, general/link/scaled FISAPT or speedup proof."}}
