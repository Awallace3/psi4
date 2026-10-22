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


def occupied_transform(session,value,ua,ub,R,S):
    """Y[r,l,s,k] = sum_ab UA[a,l] X[r,a,s,b] UB[b,k], on GPU."""
    A,B=_rotations(session,ua,ub)
    m=DeviceMath(session);m.require(value)
    if (type(R) is not int or type(S) is not int or min(R,S)<1
            or value.shape!=(R*A,S*B)):
        raise ValueError("Wrong virtual/occupied block layout for localization")
    output=session.empty(value.shape)
    with session.scope():
        right=session.gemm(value.view((R*A*S,B)),ub)
        permuted=session.empty((R*S*B,A))
        session.launch("permute_occupied_a",min(65535,(value.size+255)//256),256,
                       [R,A,S,B,False,right,permuted])
        left=session.gemm(permuted,ua)
        session.launch("permute_occupied_a",min(65535,(value.size+255)//256),256,
                       [R,A,S,B,True,left,output])
    return output


def localized_dispersion_block(session,v,direct,crossed,energies,ua,ub,
                               disp,exch,*,R,S,r0=0,s0=0):
    """Accumulate signed localized occupied pairs for one canonical (r,s) block.

    Inputs must follow validated HF preparation and validate_rotations(). All
    orbital rotations and amplitude/reduction numerics remain on this session.
    Crossed W retains its independent [s,a,r,b] layout.
    """
    A,B=_rotations(session,ua,ub)
    if (any(type(i) is not int for i in (R,S,r0,s0)) or min(R,S)<1 or min(r0,s0)<0
            or set(energies)!=set("arbs")):
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
    with session.scope():
        t=session.empty(v.shape)
        session.launch("make_amplitudes",min(65535,(v.size+255)//256),256,
                       [A,B,R,S,r0,s0,energies["a"],energies["b"],energies["r"],energies["s"],v,t])
        tl=occupied_transform(session,t,ua,ub,R,S)
        vl=occupied_transform(session,v,ua,ub,R,S)
        xl=occupied_transform(session,direct,ua,ub,R,S)
        wl=occupied_transform(session,crossed,ua,ub,S,R)
        session.launch("reduce_localized_dispersion",min(65535,A*B),128,
                       [A,B,R,S,tl,vl,xl,wl,disp,exch])
