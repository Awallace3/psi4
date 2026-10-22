"""Unscaled localized first-order S^2 exchange, shared native JKFIT/GPU algebra.

This is not Exch-Disp20 (that belongs to the dispersion module), scaled
Exch10, default driver dispatch, or link/external-potential qualification.
"""
from dataclasses import dataclass

from .sapt_dispersion_cuda_setup import DeviceMath
from .sapt_fisapt_elst_cuda import scatter_matrix,sum_values


@dataclass(frozen=True)
class FragmentExchangeDevice:
    matrix: object
    total: object


def fexch_device(elst_state,locc_a,locc_b,vir_a,vir_b,matrices,*,natoms):
    """Return native atom/occupied slots and GPU sum, without host intermediates.

    The live electrostatic state identifies the original shared JKFIT provider,
    metric gauge and lifetime. No new SCF/JK engine or CPU factor cache is made.
    Every input matrix is already on this CUDA session. The caller owns the
    enclosing scope, including this result and elst_state's reusable factors.
    """
    provider=elst_state.provider;elst_state.require_provider(provider)
    session=provider.session;m=DeviceMath(session)
    if set(matrices)!={"S","V_A","J_A","V_B","J_B"}:
        raise ValueError("Raw shared upstream S/V/J matrices required")
    for value in (locc_a,locc_b,vir_a,vir_b,*matrices.values()):m.require(value)
    if (type(natoms) is not int or natoms<1
            or any(v.shape!=(provider.nao,provider.nao) for v in matrices.values())
            or any(len(v.shape)!=2 or v.shape[0]!=provider.nao or v.shape[1]<1
                   for v in (locc_a,locc_b,vir_a,vir_b))):
        raise ValueError("Positive common-AO exchange orbital/matrix layouts required")
    A,B=locc_a.shape[1],locc_b.shape[1]
    if elst_state.matrix.shape!=(natoms+A+1,natoms+B+1):
        raise ValueError("Exchange must preserve shared atom/all-occupied slots")
    output=session.empty((natoms+A+1,natoms+B+1),zero=True)
    with session.scope():
        S=matrices["S"]
        wa=m.combine(matrices["V_A"],matrices["J_A"],1,2)
        wb=m.combine(matrices["V_B"],matrices["J_B"],1,2)
        sab=session.gemm_at_b(locc_a,session.gemm(S,locc_b))
        sba=session.gemm_at_b(locc_b,session.gemm(S,locc_a))
        sas=session.gemm_at_b(locc_a,session.gemm(S,vir_b))
        sbr=session.gemm_at_b(locc_b,session.gemm(S,vir_a))
        # Linearity moves virtual contractions to device coefficient spaces,
        # avoiding host disk Bab/Bba tensors and Q*occupied*virtual storage.
        ta=session.gemm(vir_a,m.window(sbr,transpose=True))
        tb=session.gemm(vir_b,m.window(sas,transpose=True))
        wbab=session.gemm_at_b(locc_a,session.gemm(wb,ta))
        waba=session.gemm_at_b(locc_b,session.gemm(wa,tb))
        factors=provider.compute_device({"Bab":(locc_a,ta),"Bba":(locc_b,tb)})
        if (factors["Bab"].shape!=(provider.naux,A,B)
                or factors["Bba"].shape!=(provider.naux,B,A)):
            raise ValueError("Native exchange DFMO factor layout differs")
        for value in factors.values():m.require(value)
        pairs=session.empty((A,B))
        session.launch("reduce_fragment_exchange",min(65535,A*B),128,
                       [provider.naux,A,B,sab,sba,wbab,waba,factors["Bab"],factors["Bba"],pairs])
        scatter_matrix(session,pairs,output,natoms,natoms)
    return FragmentExchangeDevice(output,sum_values(session,output))
