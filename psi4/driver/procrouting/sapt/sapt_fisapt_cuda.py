"""Owned GPU localized-pair dispersion primitives; not full FISAPT dispatch.

Canonical denominators must precede occupied localization. Atom/fragment/link
assignment and localization itself are separate, currently unwired operations.
No CPU factors, arithmetic fallback or CuPy is used.
"""
import numpy as np

from .sapt_dispersion_cuda_setup import DeviceMath


def _rotations(session, ua, ub):
    m=DeviceMath(session)
    for value in (ua,ub):
        m.require(value)
        if len(value.shape)!=2 or value.shape[0]!=value.shape[1]:
            raise ValueError("Square active-occupied rotations required")
    return ua.shape[0],ub.shape[0]


def validate_rotations(session,ua,ub):
    """Validate on device; download two scalar errors only, no matrix staging."""
    _rotations(session,ua,ub)
    for value in (ua,ub):
        with session.scope():
            gram=session.gemm_at_b(value,value)
            error=session.empty((1,))
            session.launch("rotation_error",1,128,[value.shape[0],gram,error])
            worst=float(session.download(error)[0])
        if not np.isfinite(worst) or worst>1e-10:
            raise ValueError("Occupied localization rotation is not finite/orthogonal")


def occupied_transform(session,value,ua,ub,R,S,*,scratch=None,out=None):
    """Y[r,l,s,k] = sum_ab UA[a,l] X[r,a,s,b] UB[b,k], on GPU without permutes.

    Uses a 2D GEMM for UB and a single strided-batched GEMM (batch=R) for UA^T.
    When caller-owned ``scratch`` and ``out`` views are supplied, executes with
    zero allocations and zero scope synchronizations.
    """
    A,B=_rotations(session,ua,ub)
    m=DeviceMath(session);m.require(value)
    if (type(R) is not int or type(S) is not int or min(R,S)<1
            or value.shape!=(R*A,S*B)):
        raise ValueError("Wrong virtual/occupied block layout for localization")
    if out is not None:
        m.require(out)
        if out.shape!=value.shape:
            raise ValueError("Wrong localized transform output shape")
    if scratch is not None:
        m.require(scratch)
        if scratch.size<value.size:
            raise ValueError("Localized transform scratch buffer too small")
        right_view=scratch.view((R*A*S,B))
        output=session.empty(value.shape) if out is None else out
        session.gemm(value.view((R*A*S,B)),ub,out=right_view)
        session.gemm_batched(ua,right_view.view((R,A,S*B)),transpose_left=True,
                             out=output.view((R,A,S*B)))
        return output
    output=session.empty(value.shape) if out is None else out
    with session.scope():
        right=session.gemm(value.view((R*A*S,B)),ub)
        session.gemm_batched(ua,right.view((R,A,S*B)),transpose_left=True,
                             out=output.view((R,A,S*B)))
    return output


def localized_dispersion_block(session,v,direct,crossed,energies,ua,ub,
                               disp,exch,*,R,S,r0=0,s0=0,
                               pre_localized_xw=False,workspace=None):
    """Accumulate signed localized occupied pairs for one canonical (r,s) block.

    Inputs must follow validated HF preparation and validate_rotations(). All
    orbital rotations and amplitude/reduction numerics remain on this session.
    Crossed W retains its independent [s,a,r,b] layout. When ``pre_localized_xw``
    is True, ``direct`` and ``crossed`` are already in the localized occupied
    basis and only ``t`` and ``v`` are transformed.
    """
    A,B=_rotations(session,ua,ub)
    if (any(type(i) is not int for i in (R,S,r0,s0)) or min(R,S)<1 or min(r0,s0)<0
            or set(energies)!=set("arbs") or type(pre_localized_xw) is not bool):
        raise ValueError("Invalid localized dispersion block metadata")
    m=DeviceMath(session)
    for value in (v,direct,crossed,disp,exch,*energies.values()):
        m.require(value)
    if (v.shape!=(R*A,S*B) or direct.shape!=v.shape or crossed.shape!=(S*A,R*B)
            or disp.shape!=(A,B) or exch.shape!=(A,B)
            or energies["a"].shape!=(A,) or energies["b"].shape!=(B,)
            or len(energies["r"].shape)!=1 or len(energies["s"].shape)!=1
            or energies["r"].size<r0+R or energies["s"].size<s0+S):
        raise ValueError("Wrong localized dispersion input shapes")
    reduce_grid=min(65535,A*((B+31)//32))
    if workspace is not None:
        if pre_localized_xw:
            t_buf,tmp1=workspace
            t=t_buf.view(v.shape)
            session.launch("make_amplitudes",min(65535,(v.size+255)//256),256,
                           [A,B,R,S,r0,s0,energies["a"],energies["b"],energies["r"],energies["s"],v,t])
            # Rotate T in place via tmp1, then rotate V in place via tmp1 (non-aliasing).
            session.gemm(t.view((R*A*S,B)),ub,out=tmp1.view((R*A*S,B)))
            session.gemm_batched(ua,tmp1.view((R,A,S*B)),transpose_left=True,out=t.view((R,A,S*B)))
            session.gemm(v.view((R*A*S,B)),ub,out=tmp1.view((R*A*S,B)))
            session.gemm_batched(ua,tmp1.view((R,A,S*B)),transpose_left=True,out=v.view((R,A,S*B)))
            session.launch("reduce_localized_dispersion",reduce_grid,256,
                           [A,B,R,S,t,v,direct,crossed,disp,exch])
            return
        t_buf,tmp1,xl_buf,wl_buf=workspace
        t=t_buf.view(v.shape)
        xl=xl_buf.view(v.shape)
        wl=wl_buf.view(crossed.shape)
        session.launch("make_amplitudes",min(65535,(v.size+255)//256),256,
                       [A,B,R,S,r0,s0,energies["a"],energies["b"],energies["r"],energies["s"],v,t])
        session.gemm(t.view((R*A*S,B)),ub,out=tmp1.view((R*A*S,B)))
        session.gemm_batched(ua,tmp1.view((R,A,S*B)),transpose_left=True,out=t.view((R,A,S*B)))
        session.gemm(v.view((R*A*S,B)),ub,out=tmp1.view((R*A*S,B)))
        session.gemm_batched(ua,tmp1.view((R,A,S*B)),transpose_left=True,out=v.view((R,A,S*B)))
        occupied_transform(session,direct,ua,ub,R,S,scratch=tmp1,out=xl)
        occupied_transform(session,crossed,ua,ub,S,R,scratch=tmp1,out=wl)
        session.launch("reduce_localized_dispersion",reduce_grid,256,
                       [A,B,R,S,t,v,xl,wl,disp,exch])
        return
    with session.scope():
        t=session.empty(v.shape)
        session.launch("make_amplitudes",min(65535,(v.size+255)//256),256,
                       [A,B,R,S,r0,s0,energies["a"],energies["b"],energies["r"],energies["s"],v,t])
        tl=occupied_transform(session,t,ua,ub,R,S)
        vl=occupied_transform(session,v,ua,ub,R,S)
        if pre_localized_xw:
            xl,wl=direct,crossed
        else:
            xl=occupied_transform(session,direct,ua,ub,R,S)
            wl=occupied_transform(session,crossed,ua,ub,S,R)
        session.launch("reduce_localized_dispersion",reduce_grid,256,
                       [A,B,R,S,tl,vl,xl,wl,disp,exch])
