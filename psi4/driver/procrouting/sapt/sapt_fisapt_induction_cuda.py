"""Uncoupled directional fragment induction primitives, not full find dispatch.

Nuclear/electronic source assembly and exchange-induction target construction
are separate operations. No coupled response/CPU amplitude fallback is present.
"""
from dataclasses import dataclass
import numpy as np

from .sapt_dispersion_cuda_setup import DeviceMath
from .sapt_fisapt_cuda import validate_rotations
from .sapt_fisapt_elst_cuda import scatter_matrix,sum_values


def projected_coulomb(session,density_factors,target_factors):
    """J[generalized density] in a target MO block, using ONE shared DFMO gauge.

    Inputs are already metric-transformed native factors (Q,k,k) and (Q,o,v).
    Trace over occupied density columns, then contract Q; no AO J matrix, CPU
    factor, spin factor, or second whitening operation is introduced.
    """
    m=DeviceMath(session)
    for value in (density_factors,target_factors):m.require(value)
    if (len(density_factors.shape)!=3 or len(target_factors.shape)!=3
            or density_factors.shape[1]!=density_factors.shape[2]
            or density_factors.shape[0]!=target_factors.shape[0]):
        raise ValueError("Shared-Q square density and occupied-virtual factors required")
    Q,O,_=density_factors.shape
    rho=session.empty((Q,1))
    session.launch("trace_factors",min(65535,Q),128,[Q,O,density_factors,rho])
    _,A,V=target_factors.shape
    result=session.gemm_at_b(rho,target_factors.view((Q,A*V)))
    return result.view((A,V))


def exchange_induction_target(session,matrices,localized,virtual,jp,*,side):
    """Localized S^2 exchange-induction target; JP is already projected to MO.

    Raw upstream AO matrices remain resident; generalized J_P is obtained by
    projected_coulomb() from the same original JKFIT plan. Mixed K_O is not
    symmetric: swap A/B AND transpose it for the B-response direction.
    """
    fields={"S","D_A","D_B","V_A","V_B","J_A","J_B","K_A","K_B","J_O","K_O"}
    if side not in ("A","B") or set(matrices)!=fields:
        raise ValueError("Complete raw shared exchange-induction AO inputs required")
    m=DeviceMath(session)
    for value in (*matrices.values(),localized,virtual,jp):m.require(value)
    if (len(localized.shape)!=2 or len(virtual.shape)!=2
            or localized.shape[0]!=virtual.shape[0]
            or jp.shape!=(localized.shape[1],virtual.shape[1])
            or any(value.shape!=(localized.shape[0],localized.shape[0]) for value in matrices.values())):
        raise ValueError("Common AO and projected-JP target layouts required")
    other="B" if side=="A" else "A"
    S=matrices["S"];da,db=matrices["D_"+side],matrices["D_"+other]
    ka,kb=matrices["K_"+side],matrices["K_"+other]
    wa=m.combine(matrices["V_"+side],matrices["J_"+side],1,2)
    wb=m.combine(matrices["V_"+other],matrices["J_"+other],1,2)
    ko=matrices["K_O"] if side=="A" else m.window(matrices["K_O"],transpose=True)
    ha,hb=m.combine(wa,ka,1,-1),m.combine(wb,kb,1,-1)
    left=m.terms((-1,ha),(1,m.product(S,da,wb)),(1,m.product(wa,db,S)),
                 (-1,m.window(ko,transpose=True)))
    right=m.terms((-1,hb),(1,m.product(wb,da,S)),(-1,ko))
    operator=m.terms((-1,kb),(-2,matrices["J_O"]),(1,ko),
                     (1,m.product(S,db,left)),(1,m.product(right,db,S)))
    base=session.gemm_at_b(localized,session.gemm(operator,virtual))
    return m.combine(base,jp,1,2)


def uncoupled_source(session,source,energies,rotation,targets,outputs,*,column):
    """Divide a canonical source, localize amplitudes, reduce one source column.

    Caller validates full occupied rotation/gaps once before its source loop.
    `source` is canonical (occ,vir); localized `targets['ind']` is wL and
    optional `targets['exch']` is uL. Occupied space includes frozen orbitals.
    `outputs` has corresponding (localizedOcc,sourceCount) device matrices.
    No download or host arithmetic occurs. All returned effects are borrowed
    writes into caller-owned outputs; temporary amplitudes are scope bounded.
    """
    if (set(energies)!={"o","v"} or set(targets) not in ({"ind"},{"ind","exch"})
            or set(outputs)!=set(targets) or type(column) is not int or column<0):
        raise ValueError("Canonical energies and matching directional components required")
    m=DeviceMath(session)
    inputs=(source,rotation,*energies.values(),*targets.values())
    for value in (*inputs,*outputs.values()):m.require(value)
    if len(source.shape)!=2:
        raise ValueError("A canonical occupied-virtual source matrix is required")
    O,V=source.shape
    shape=outputs["ind"].shape
    if (rotation.shape!=(O,O) or energies["o"].shape!=(O,) or energies["v"].shape!=(V,)
            or any(value.shape!=(O,V) for value in targets.values())
            or len(shape)!=2 or shape[0]!=O or column>=shape[1]
            or any(value.shape!=shape for value in outputs.values())
            or any(value.allocation is output.allocation for value in inputs for output in outputs.values())
            or ("exch" in outputs and outputs["ind"].allocation is outputs["exch"].allocation)):
        raise ValueError("Wrong directional induction layout/placement/alias")
    with session.scope():
        canonical=session.empty(source.shape)
        session.launch("make_single_amplitudes",min(65535,(source.size+255)//256),256,
                       [O,V,source,energies["o"],energies["v"],canonical])
        localized=session.gemm_at_b(rotation,canonical)
        session.launch("reduce_induction_source",min(65535,O),128,
                       [O,V,shape[1],column,localized,targets["ind"],targets.get("exch"),
                        outputs["ind"],outputs.get("exch")])


def occupied_rotation(session,canonical,localized,overlap,occupied_energy,virtual_energy):
    """Extract and validate the full occupied rotation on GPU, including frozen.

    Only scalar consistency/gap errors are downloaded; no orbital matrix is
    staged. The caller separately checks the returned rotations' orthogonality.
    """
    m=DeviceMath(session)
    for value in (canonical,localized,overlap,occupied_energy,virtual_energy):m.require(value)
    if (len(canonical.shape)!=2 or canonical.shape!=localized.shape
            or overlap.shape!=(canonical.shape[0],canonical.shape[0])
            or occupied_energy.shape!=(canonical.shape[1],) or len(virtual_energy.shape)!=1):
        raise ValueError("Full occupied/overlap/energy layouts required")
    rotation=session.empty((canonical.shape[1],canonical.shape[1]))
    with session.scope():
        value=session.gemm_at_b(canonical,session.gemm(overlap,localized))
        # Copy through an exact GPU linear combination into the caller scope.
        session.launch("linear_combination",min(65535,(value.size+255)//256),256,
                       [value.size,1,0,value,value,rotation])
        residual=m.combine(session.gemm(canonical,rotation),localized,1,-1)
        error=session.empty((1,))
        session.launch("matrix_error",1,128,[residual.size,residual,error])
        worst=float(session.download(error)[0])
        if not np.isfinite(worst) or worst>1e-10:
            raise ValueError("Localized occupied orbitals are outside the canonical occupied span")
        session.launch("single_gap_error",1,128,
                       [canonical.shape[1],virtual_energy.size,occupied_energy,virtual_energy,error])
        if float(session.download(error)[0])!=0.:
            raise ValueError("Induction requires finite strictly positive occupied-virtual gaps")
    return rotation


@dataclass(frozen=True)
class Ind20uDevice:
    """Two directional component matrices and device scalar sums."""
    ab: object
    ba: object
    total_ab: object
    total_ba: object


@dataclass(frozen=True)
class Find20uDevice:
    ind20u: object
    exch_ind20u: object
    combined_ab: object
    combined_ba: object
    total_ab: object
    total_ba: object


def ind20u_device(elst_state,potential,occupied,virtuals,energies,matrices,xyz,za,zb):
    """Native nuclear/electronic sources and device uncoupled Ind20u directions.
    Explicit two-fragment C/no-external numerical seam. The parent elst state
    retains localized occupied inputs and diagonal JKFIT factors, preventing
    substitution of a different localized density or metric gauge. All occupied
    targets participate; no frozen-core trimming or coupled response occurs.
    """
    return _directional_device(elst_state,potential,occupied,virtuals,energies,matrices,xyz,za,zb,exchange=False)


def find20u_device(elst_state,potential,occupied,virtuals,energies,matrices,xyz,za,zb):
    """Unscaled uncoupled Ind20u AND Exch-Ind20u, same native JKFIT gauge.

    Generalized density coefficients, native DFMO trace factors, projected J_P
    targets, denominators, full occupied rotations, signed pair/source matrices
    and their combined totals all stay on device. This is not coupled response,
    link/external/scaled or automatic FISAPT driver dispatch.
    """
    return _directional_device(elst_state,potential,occupied,virtuals,energies,matrices,xyz,za,zb,exchange=True)


def _directional_device(elst_state,potential,occupied,virtuals,energies,matrices,xyz,za,zb,*,exchange):
    provider=elst_state.provider;elst_state.require_provider(provider)
    session=provider.session;m=DeviceMath(session);owner=potential.owner
    if (potential.session is not session or potential.num_charges!=1
            or provider.handle!=owner.handle or provider.plan!=owner.plan
            or potential.primary_basis is not provider.primary_basis
            or not any(value is owner for value in provider.keepalive)):
        raise ValueError("Induction requires the original owned primary/JKFIT potential context")
    fields={"S","V_A","J_A","V_B","J_B"}
    if exchange:fields|={"D_A","D_B","K_A","K_B","J_O","K_O"}
    if (set(occupied)!={"A","B"} or set(virtuals)!={"A","B"}
            or set(energies)!=set("arbs") or set(matrices)!=fields):
        raise ValueError("Raw shared HF induction inputs required")
    for value in (*occupied.values(),*virtuals.values(),*energies.values(),*matrices.values(),xyz,za,zb):
        m.require(value)
    N=owner.primary_snapshot.ncenter
    local={"A":elst_state.locc_a,"B":elst_state.locc_b}
    for side,o,v in (("A","a","r"),("B","b","s")):
        if (occupied[side].shape!=local[side].shape or occupied[side].shape[0]!=provider.nao
                or len(virtuals[side].shape)!=2 or virtuals[side].shape[0]!=provider.nao
                or energies[o].shape!=(occupied[side].shape[1],)
                or energies[v].shape!=(virtuals[side].shape[1],)):
            raise ValueError("Wrong full-occupied/canonical virtual/energy layouts")
    A,B=local["A"].shape[1],local["B"].shape[1]
    shape=(N+A+1,N+B+1)
    if (elst_state.matrix.shape!=shape or elst_state.qa.shape!=(provider.naux,A)
            or elst_state.qb.shape!=(provider.naux,B) or xyz.shape!=(N,3)
            or za.shape!=(N,) or zb.shape!=(N,)
            or any(value.shape!=(provider.nao,provider.nao) for value in matrices.values())):
        raise ValueError("Wrong atom/source/shared-JKFIT layout")
    ab,ba=session.empty(shape,zero=True),session.empty(shape,zero=True)
    xab,xba=(session.empty(shape,zero=True),session.empty(shape,zero=True)) if exchange else (None,None)
    with session.scope():
        rotations={side:occupied_rotation(session,occupied[side],local[side],matrices["S"],
            energies[o],energies[v]) for side,o,v in (("A","a","r"),("B","b","s"))}
        validate_rotations(session,rotations["A"],rotations["B"])
        pairs={side:(occupied[side],virtuals[side]) for side in ("A","B")}
        if exchange:
            for side,other in (("A","B"),("B","A")):
                prime=m.product(matrices["D_"+side],matrices["S"],
                                matrices["D_"+other],matrices["S"],occupied[side])
                pairs["rho_"+side]=(prime,occupied[side])
                pairs["target_"+side]=(local[side],virtuals[side])
        factors=provider.compute_device(pairs)
        for side,other,diagonal,charges,target_output in (
                ("A","B",elst_state.qb,zb,ab),("B","A",elst_state.qa,za,ba)):
            O,V=occupied[side].shape[1],virtuals[side].shape[1]
            C=N+local[other].shape[1]+1
            o,v=("a","r") if side=="A" else ("b","s")
            with session.scope():
                block=factors[side];m.require(block)
                if block.shape!=(provider.naux,O,V):
                    raise ValueError("Native induction source DFMO layout differs")
                sources=session.gemm_at_b(diagonal,block.view((provider.naux,O*V)))
                sources=m.combine(sources,sources,2,0)
                omega=m.combine(matrices["V_"+other],matrices["J_"+other],1,2)
                target=session.gemm_at_b(local[side],session.gemm(omega,virtuals[side]))
                directional=session.empty((O,C),zero=True)
                targets,outputs={"ind":target},{"ind":directional}
                if exchange:
                    jp=projected_coulomb(session,factors["rho_"+other],factors["target_"+side])
                    targets["exch"]=exchange_induction_target(
                        session,matrices,local[side],virtuals[side],jp,side=side)
                    outputs["exch"]=session.empty((O,C),zero=True)
                negative=m.combine(charges,charges,-1,0)
                for atom in range(N):
                    with session.scope():
                        potential_ao=potential.compute_device(xyz.view((1,3),offset=24*atom),
                                                              negative.view((1,),offset=8*atom))
                        source=session.gemm_at_b(occupied[side],session.gemm(potential_ao,virtuals[side]))
                        uncoupled_source(session,source,{"o":energies[o],"v":energies[v]},rotations[side],
                            targets,outputs,column=atom)
                for k in range(local[other].shape[1]):
                    source=sources.view((O,V),offset=8*k*O*V)
                    uncoupled_source(session,source,{"o":energies[o],"v":energies[v]},rotations[side],
                        targets,outputs,column=N+k)
                if side=="A":
                    scatter_matrix(session,directional,target_output,N,0)
                else:
                    scatter_matrix(session,m.window(directional,transpose=True),target_output,0,N)
                if exchange:
                    if side=="A":scatter_matrix(session,outputs["exch"],xab,N,0)
                    else:scatter_matrix(session,m.window(outputs["exch"],transpose=True),xba,0,N)
    ind=Ind20uDevice(ab,ba,sum_values(session,ab),sum_values(session,ba))
    if not exchange:return ind
    ex=Ind20uDevice(xab,xba,sum_values(session,xab),sum_values(session,xba))
    combined_ab,combined_ba=m.combine(ab,xab),m.combine(ba,xba)
    return Find20uDevice(ind,ex,combined_ab,combined_ba,
                        sum_values(session,combined_ab),sum_values(session,combined_ba))
