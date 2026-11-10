"""Experimental CuPy-free Python SAPT0: cuEST + three cuBLAS GEMMs + CUDA."""
import hashlib
import time

import numpy as np

from .sapt_cuda_runtime import DeviceArray
from .sapt_dispersion_cuda_kernels import CUDA_SOURCE
from .sapt_dispersion_device_common import coefficient_pairs, validate_inputs


def _factor_batch(session, provider, pairs, *, device=False):
    factors = provider.compute_device(pairs) if device else provider.compute(pairs)
    if set(factors) != set(pairs):
        raise ValueError("Incomplete DFMO batch")
    for name, (left, right) in pairs.items():
        array = factors[name]
        if (not isinstance(array, DeviceArray) or array.allocation.session is not session
                or array.dtype != np.float64
                or array.shape != (provider.naux, left.shape[1], right.shape[1])):
            raise ValueError("DFMO must return this session's owned FP64 CUDA buffers")
    return factors


def _pack(session, q, factors, first, second, extra_first, extra_second, tail1, tail2, *, out=None):
    f, s = factors[first], factors[second]
    if f.shape != s.shape or f.shape[0] != q or f.order != s.order:
        raise ValueError("Inconsistent DFMO factor layouts")
    _, virtual, occupied = f.shape
    for name in (extra_first, extra_second):
        if name is not None and (factors[name].shape != f.shape or factors[name].order != f.order):
            raise ValueError("Inconsistent combined DFMO factors")
    if tail1.shape != (occupied, virtual) or tail2.shape != tail1.shape:
        raise ValueError("Wrong rank-one correction shapes")
    shape = (2*q+2, virtual*occupied)
    if out is None:
        output = session.empty(shape)
    else:
        if (not isinstance(out,DeviceArray) or out.shape!=shape or out.order!="C"
                or out.dtype!=np.float64 or out.allocation.session is not session):
            raise ValueError("Packed factor output needs an owned C-order FP64 extent")
        out.pointer
        for value in (f,s,factors.get(extra_first),factors.get(extra_second),tail1,tail2):
            if (value is not None and out.allocation is value.allocation
                    and out.offset < value.offset+value.nbytes
                    and value.offset < out.offset+out.nbytes):
                raise ValueError("Packed factor output cannot overlap an input")
        output = out
    session.launch("pack_factors", min(65535, (output.size+255)//256), 256,
                   [q, virtual, occupied, f.order == "F",
                    f, factors.get(extra_first), s, factors.get(extra_second),
                    tail1, tail2, output])
    return output


def blocked_dispersion(provider, spaces, corrections, energies, *, block_size=24):
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise ValueError("CUDA dispersion block_size must be a positive integer")
    validate_inputs(spaces, corrections, energies)
    session = provider.session
    session.check()
    if provider.nao != spaces["a"].shape[0]:
        raise ValueError("Incompatible DFMO AO space")
    started = time.perf_counter()
    a, nr, b, ns = (energies[name].size for name in "arbs")
    q, count = provider.naux, 0
    with session.scope():
        e = {name: session.upload(value) for name, value in energies.items()}
        d = session.empty((a,b), zero=True)
        x = session.empty((a,b), zero=True)
        for r0 in range(0, nr, block_size):
            rs = slice(r0, min(nr, r0+block_size))
            R = rs.stop-rs.start
            with session.scope():
                rf = _factor_batch(session, provider, coefficient_pairs(spaces,"r",rs))
                tails = {name: session.upload(corrections[name][:,rs])
                         for name in ("Qar","SBar","Qbr","Sbr")}
                af = _pack(session,q,rf,"Aar","Dar",None,"Ear",tails["Qar"],tails["SBar"])
                cb = _pack(session,q,rf,"Bbr","Cbr",None,None,tails["Qbr"],tails["Sbr"])
                for s0 in range(0, ns, block_size):
                    ss = slice(s0,min(ns,s0+block_size))
                    S = ss.stop-ss.start
                    with session.scope():
                        sf = _factor_batch(session, provider, coefficient_pairs(spaces,"s",ss))
                        st = {name: session.upload(corrections[name][:,ss])
                              for name in ("SAbs","Qbs","Sas","Qas")}
                        fa = _pack(session,q,sf,"Dbs","Abs","Ebs",None,st["SAbs"],st["Qbs"])
                        bc = _pack(session,q,sf,"Bas","Cas",None,None,st["Sas"],st["Qas"])
                        v = session.gemm_at_b(af.view((q,R*a)),
                                             fa.view((q,S*b),offset=q*S*b*8))
                        direct = session.gemm_at_b(af,fa)
                        crossed = session.gemm_at_b(bc,cb)
                        session.launch("reduce_dispersion",min(65535,a*b),128,
                                       [a,b,R,S,r0,s0,e["a"],e["b"],e["r"],e["s"],
                                        v,direct,crossed,d,x])
                        count += 1
        d_host, x_host = session.download(d), session.download(x)
    if not np.isfinite(d_host).all() or not np.isfinite(x_host).all():
        raise FloatingPointError("Nonfinite signed CUDA dispersion components")
    producer_report = provider.report()
    return {
        "Disp20,u": float(d_host.sum()), "Exch-Disp20,u": float(x_host.sum()),
        "Disp20 occupied pairs": d_host, "Exch-Disp20 occupied pairs": x_host,
        "Disp0 execution": {
            "backend": "CUDA_CUEST_EXPERIMENTAL", "compute_blocks": count,
            "block_size": block_size,
            "blocked_execution_seconds": time.perf_counter()-started,
            "owned_peak_device_bytes_session_lifetime": session.peak_live_bytes,
            "declared_native_persistent_bytes": session.native_bytes,
            "cuda_source_sha256": hashlib.sha256(CUDA_SOURCE.encode()).hexdigest(),
            "actual_gpu_execution_verified": False,
            "scope": "CuPy-free Python/cuEST/cuBLAS plus FP64 packing/reduction kernels; "
                     "excludes upstream orbital setup and borrowed-plan creation/cleanup",
            "producer": producer_report,
        },
    }


def blocked_device_dispersion(provider, spaces, corrections, energies, *, block_size=24, rotations=None,
                              factor_cache_bytes=0):
    """GPU-only numerical continuation of prepare_hf, including global sums.

    All inputs must be device outputs of validated raw-HF preparation on this
    session. The provider must prepare persistent DFMO parameters before the
    caller enters its enclosing setup/compute scope. No host coefficient upload,
    correction staging or CPU numerical fallback is used here.

    An explicit additional byte budget enables per-call packed S-slab reuse.
    Cache lifetime is this call only; neither raw factors nor a second IBO
    gauge are retained. Zero (default) preserves the uncached residency policy.
    """
    from .sapt_dispersion_cuda_setup import DeviceMath, coefficient_pairs as device_pairs
    from .sapt_dispersion_cuda_setup import finish_components
    if type(block_size) is not int or block_size < 1:
        raise ValueError("CUDA dispersion block_size must be a positive integer")
    if type(factor_cache_bytes) is not int or factor_cache_bytes < 0:
        raise ValueError("Dispersion factor cache needs a nonnegative integer byte budget")
    session = provider.session
    m = DeviceMath(session)
    if set(energies) != set("arbs"):
        raise ValueError("Complete device energy vectors required")
    for value in energies.values():
        m.require(value)
        if len(value.shape) != 1:
            raise ValueError("Device energies must be vectors")
    a,nr,b,ns = (energies[name].size for name in "arbs")
    sizes = {"a":a,"r":nr,"b":b,"s":ns}
    needed = ("a","r","b","s","r1","s1","a2","b2","r3","s3","a4","b4")
    shapes = {"Qar":("a","r"),"SBar":("a","r"),"Qbr":("b","r"),"Sbr":("b","r"),
              "SAbs":("b","s"),"Qbs":("b","s"),"Sas":("a","s"),"Qas":("a","s")}
    if set(spaces) != set(needed) or set(corrections) != set(shapes):
        raise ValueError("Complete prepared device spaces/corrections required")
    for name,value in spaces.items():
        m.require(value)
        if value.shape != (provider.nao,sizes[name[0]]):
            raise ValueError("Incompatible prepared device AO coefficients")
    for name,value in corrections.items():
        m.require(value)
        if value.shape != tuple(sizes[index] for index in shapes[name]):
            raise ValueError("Incompatible prepared device correction")
    if rotations is not None:
        from .sapt_fisapt_cuda import validate_rotations,localized_dispersion_block
        if not isinstance(rotations,(tuple,list)) or len(rotations)!=2:
            raise ValueError("Two active occupied rotations required")
        for rotation in rotations:m.require(rotation)
        if rotations[0].shape!=(a,a) or rotations[1].shape!=(b,b):
            raise ValueError("Occupied rotation dimensions differ from active spaces")
        validate_rotations(session,*rotations)
    started = time.perf_counter()
    q,count = provider.naux,0
    # Deterministic budget selection; a short tail may fit even if a full slab
    # does not. Do not allocate cache for a single R slab: there is no reuse.
    layout={};cache_bytes=0
    if nr>block_size and factor_cache_bytes:
        for s0 in range(0,ns,block_size):
            S=min(block_size,ns-s0)
            size=8*(2*q+2)*S*(a+b)
            if size<=factor_cache_bytes-cache_bytes:
                layout[s0]=(S,cache_bytes)
                cache_bytes+=size
    r_batches=s_batches=cache_hits=0
    with session.scope():
        # Allocate BEFORE nested R/S/native scopes, so their cleanup cannot
        # retire cache storage. Fill lazily on the first R pass in the original
        # native-call order; all later contractions keep that same R-major order.
        storage=session.empty((cache_bytes//8,)) if cache_bytes else None
        cached={}
        for s0,(S,offset) in layout.items():
            cached[s0]=(storage.view((2*q+2,S*b),offset=offset),
                        storage.view((2*q+2,S*a),offset=offset+8*(2*q+2)*S*b))
        d,x = session.empty((a,b),zero=True),session.empty((a,b),zero=True)
        for r0 in range(0,nr,block_size):
            R = min(block_size,nr-r0)
            with session.scope():
                rf = _factor_batch(session,provider,device_pairs(session,spaces,"r",r0,R),device=True)
                r_batches+=1
                tails = {name:m.window(corrections[name],r0,R)
                         for name in ("Qar","SBar","Qbr","Sbr")}
                af = _pack(session,q,rf,"Aar","Dar",None,"Ear",tails["Qar"],tails["SBar"])
                cb = _pack(session,q,rf,"Bbr","Cbr",None,None,tails["Qbr"],tails["Sbr"])
                for s0 in range(0,ns,block_size):
                    S = min(block_size,ns-s0)
                    with session.scope():
                        entry=cached.get(s0)
                        if entry is None or r0==0:
                            sf = _factor_batch(session,provider,device_pairs(session,spaces,"s",s0,S),device=True)
                            st = {name:m.window(corrections[name],s0,S)
                                  for name in ("SAbs","Qbs","Sas","Qas")}
                            fa = _pack(session,q,sf,"Dbs","Abs","Ebs",None,st["SAbs"],st["Qbs"],
                                       out=entry[0] if entry is not None else None)
                            bc = _pack(session,q,sf,"Bas","Cas",None,None,st["Sas"],st["Qas"],
                                       out=entry[1] if entry is not None else None)
                            s_batches+=1
                        else:
                            fa,bc=entry
                            cache_hits+=1
                        v = session.gemm_at_b(af.view((q,R*a)),fa.view((q,S*b),offset=q*S*b*8))
                        direct,crossed = session.gemm_at_b(af,fa),session.gemm_at_b(bc,cb)
                        if rotations is None:
                            session.launch("reduce_dispersion",min(65535,a*b),128,
                                           [a,b,R,S,r0,s0,energies["a"],energies["b"],
                                            energies["r"],energies["s"],v,direct,crossed,d,x])
                        else:
                            localized_dispersion_block(session,v,direct,crossed,energies,
                                *rotations,d,x,R=R,S=S,r0=r0,s0=s0)
                        count += 1
        combined,components = finish_components(session,d,x)
        d_host,x_host = session.download(d),session.download(x)
        combined_host,scalar_host = session.download(combined),session.download(components)
    if any(not np.isfinite(value).all() for value in (d_host,x_host,combined_host,scalar_host)):
        raise FloatingPointError("Nonfinite signed CUDA dispersion components")
    return {
        "Disp20,u":float(scalar_host[0]), "Exch-Disp20,u":float(scalar_host[1]),
        "Disp0,u":float(scalar_host[2]),
        "Disp20 occupied pairs":d_host, "Exch-Disp20 occupied pairs":x_host,
        "Disp0 occupied pairs":combined_host,
        "Disp0 execution":{
            "backend":("CUDA_CUEST_LOCALIZED_DEVICE_EXPERIMENTAL" if rotations is not None
                       else "CUDA_CUEST_DEVICE_INPUT_EXPERIMENTAL"),
            "occupied_representation":"localized" if rotations is not None else "canonical",
            "compute_blocks":count,"block_size":block_size,
            "factor_cache":{"policy":"PER_CALL_PACKED_S_SLABS_R_MAJOR",
                "budget_bytes":factor_cache_bytes,"allocated_bytes":cache_bytes,
                "cached_slabs":len(layout),"hits":cache_hits,
                "r_factor_batches":r_batches,"s_factor_batches":s_batches},
            "blocked_execution_seconds":time.perf_counter()-started,
            "owned_peak_device_bytes_session_lifetime":session.peak_live_bytes,
            "declared_native_persistent_bytes":session.native_bytes,
            "cuda_source_sha256":hashlib.sha256(CUDA_SOURCE.encode()).hexdigest(),
            "actual_gpu_execution_verified":False,
            "scope":"Device-input DFMO and GPU packing/contractions/pair/global reductions; "
                    "excludes raw HF upload/setup and plan creation/cleanup",
            "producer":provider.report(),
        },
    }
