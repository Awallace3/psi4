"""Experimental CuPy-free Python SAPT0: cuEST + three cuBLAS GEMMs + CUDA."""
import hashlib
import time

import numpy as np

from .sapt_cuda_runtime import DeviceArray
from .sapt_dispersion_cuda_kernels import CUDA_SOURCE
from .sapt_dispersion_device_common import coefficient_pairs, validate_inputs


def _factor_batch(session, provider, pairs):
    factors = provider.compute(pairs)
    if set(factors) != set(pairs):
        raise ValueError("Incomplete DFMO batch")
    for name, (left, right) in pairs.items():
        array = factors[name]
        if (not isinstance(array, DeviceArray) or array.allocation.session is not session
                or array.dtype != np.float64
                or array.shape != (provider.naux, left.shape[1], right.shape[1])):
            raise ValueError("DFMO must return this session's owned FP64 CUDA buffers")
    return factors


def _pack(session, q, factors, first, second, extra_first, extra_second, tail1, tail2):
    f, s = factors[first], factors[second]
    if f.shape != s.shape or f.shape[0] != q or f.order != s.order:
        raise ValueError("Inconsistent DFMO factor layouts")
    _, virtual, occupied = f.shape
    for name in (extra_first, extra_second):
        if name is not None and (factors[name].shape != f.shape or factors[name].order != f.order):
            raise ValueError("Inconsistent combined DFMO factors")
    if tail1.shape != (occupied, virtual) or tail2.shape != tail1.shape:
        raise ValueError("Wrong rank-one correction shapes")
    output = session.empty((2*q+2, virtual*occupied))
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
