"""Experimental CuPy-free Python SAPT0: cuEST + three cuBLAS GEMMs + CUDA."""
from contextlib import nullcontext as _nullcontext
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


def _grid(count):
    return min(65535, (count + 255) // 256)


def _reduce_grid(a, b):
    return min(65535, a * ((b + 31) // 32))


def _available_device_bytes(session):
    declared = session.limit - session.live_bytes - session.native_bytes
    if hasattr(session, "runtime") and hasattr(session.runtime, "cudaMemGetInfo"):
        free, _ = session.rt(session.runtime.cudaMemGetInfo())
        return max(0, min(declared, int(free)))
    return max(0, declared)


def _dfmo_call(session, provider, pairs, *, variable_bytes=None):
    from .sapt_cuest_dfmo_cuda import CuESTDFMOProvider
    if (variable_bytes is not None and isinstance(provider, CuESTDFMOProvider)
            and variable_bytes != provider.variable_bytes):
        factors = provider.compute_device(pairs, variable_bytes=variable_bytes)
    else:
        factors = provider.compute_device(pairs)
    if set(factors) != set(pairs):
        raise ValueError("Incomplete DFMO batch")
    for name, (left, right) in pairs.items():
        array = factors[name]
        if (not isinstance(array, DeviceArray) or array.allocation.session is not session
                or array.dtype != np.float64
                or array.shape != (provider.naux, left.shape[1], right.shape[1])):
            raise ValueError("DFMO must return this session's owned FP64 CUDA buffers")
    return factors


def _hf_idempotent(session, m, hf, a_space, b_space, nao):
    if not isinstance(hf, dict):
        return False
    needed = ("Cocc_A", "Cocc_B", "S", "D_A", "D_B", "frozen_a", "frozen_b")
    if any(k not in hf for k in needed):
        return False
    CA, CB, S_ao, DA, DB = (hf[k] for k in ("Cocc_A", "Cocc_B", "S", "D_A", "D_B"))
    fa, fb = hf["frozen_a"], hf["frozen_b"]
    try:
        for arr in (CA, CB, S_ao, DA, DB):
            m.require(arr)
    except (ValueError, RuntimeError):
        return False
    if (len(CA.shape) != 2 or len(CB.shape) != 2
            or CA.shape[0] != nao or CB.shape[0] != nao
            or S_ao.shape != (nao, nao) or DA.shape != (nao, nao) or DB.shape != (nao, nao)
            or type(fa) is not int or type(fb) is not int
            or fa < 0 or fb < 0
            or fa + a_space.shape[1] != CA.shape[1]
            or fb + b_space.shape[1] != CB.shape[1]):
        return False
    with session.scope():
        error = session.empty((1,))
        for dens, occ in ((DA, CA), (DB, CB)):
            rebuilt = session.gemm(occ, occ, transpose_right=True)
            diff = m.combine(dens, rebuilt, 1, -1)
            session.launch("matrix_error", 1, 128, [diff.size, diff, error])
            worst = float(session.download(error)[0])
            if not np.isfinite(worst) or worst > 1e-12:
                return False
    return True


def _choose_tile_extents(nr, ns, a, b, q, block_size, localized, avail_bytes):
    n_4idx = 5 if localized else 3
    def tile_bytes(R, S):
        packed = (2 * q + 2) * (R + S) * (a + b) + (q * (R * a + S * b) if localized else 0)
        return 8 * (n_4idx * R * a * S * b + packed)
    # Target ~5120 GEMM rows/cols to saturate H200 FP64 Tensor Cores, bounded by memory.
    target_r = min(nr, max(block_size, (5120 + a - 1) // a))
    target_s = min(ns, max(block_size, (5120 + b - 1) // b))
    budget = min(2 * 1024**3, max(64 * 1024**2, avail_bytes // 6)) if avail_bytes > 0 else 64 * 1024**2
    R_blk, S_blk = target_r, target_s
    while tile_bytes(R_blk, S_blk) > budget and (R_blk > 1 or S_blk > 1):
        if R_blk * a >= S_blk * b and R_blk > 1:
            R_blk = max(1, R_blk // 2)
        elif S_blk > 1:
            S_blk = max(1, S_blk // 2)
        else:
            R_blk = max(1, R_blk // 2)
    # Balance tile sizes evenly across nr and ns to avoid tiny tail tiles.
    n_r = (nr + R_blk - 1) // R_blk
    n_s = (ns + S_blk - 1) // S_blk
    R_blk = (nr + n_r - 1) // n_r
    S_blk = (ns + n_s - 1) // n_s
    while tile_bytes(R_blk, S_blk) > budget and (R_blk > 1 or S_blk > 1):
        if R_blk * a >= S_blk * b and R_blk > 1:
            R_blk = max(1, R_blk - 1)
        elif S_blk > 1:
            S_blk = max(1, S_blk - 1)
        else:
            R_blk = max(1, R_blk - 1)
    return R_blk, S_blk, tile_bytes(R_blk, S_blk)


def _pack_r_tile(session, q, a, b, OA, OB, nr, frozen_a, r0, r_local, R,
                 F_OAr, F_br, F_aO, F_bO, M_a_t, M_a4_t, M_b2_t, X_r, SB_r,
                 Qar_t, SBar_t, Qbr_t, Sbr_t, af_buf, cb_buf, ar_canon_buf, localized):
    af = af_buf.view((2 * q + 2, R * a))
    cb = cb_buf.view((2 * q + 2, R * b))
    af_top = af.view((q, R, a))
    af_bot = af.view((q, R, a), offset=8 * q * R * a)
    cb_top = cb.view((q, R, b))
    cb_bot = cb.view((q, R, b), offset=8 * q * R * b)
    R_df = F_OAr.shape[2]
    if localized:
        ar_canon = ar_canon_buf.view((q, R * a))
        session.launch("transpose_factor_slab", _grid(q * R * a), 256,
                       [q, OA, frozen_a, a, R_df, r_local, R, 1, F_OAr, ar_canon])
        session.gemm_batched(F_OAr, M_a_t, transpose_left=True,
                             left_cols=(r_local, R), out=af_top)
    else:
        session.launch("transpose_factor_slab", _grid(q * R * a), 256,
                       [q, OA, frozen_a, a, R_df, r_local, R, 1, F_OAr, af_top])
        ar_canon = af.view((q, R * a))
    # Dar + Ear into af[Q:2Q]: Ear first (beta=0), then Dar accumulated (beta=1).
    session.gemm_batched(F_OAr, M_a4_t, transpose_left=True,
                         left_cols=(r_local, R), out=af_bot)
    session.gemm_batched(X_r, F_aO, transpose_left=True, transpose_right=True,
                         beta=1.0, left_cols=(r0, R), out=af_bot)
    session.launch("pack_factor_tails", _grid(2 * R * a), 256,
                   [q, nr, r0, R, a, Qar_t, SBar_t, af])
    # Bbr into cb[0:Q]: -F_br^T + SB_r^T @ F_bO_B^T (beta=1).
    session.launch("transpose_factor_slab", _grid(q * R * b), 256,
                   [q, b, 0, b, R_df, r_local, R, -1, F_br, cb_top])
    session.gemm_batched(SB_r, F_bO, transpose_left=True, transpose_right=True,
                         beta=1.0, left_cols=(r0, R), right_cols=(OA, OB), out=cb_top)
    # Cbr into cb[Q:2Q]: F_OAr^T @ M_b2_t.
    session.gemm_batched(F_OAr, M_b2_t, transpose_left=True,
                         left_cols=(r_local, R), out=cb_bot)
    session.launch("pack_factor_tails", _grid(2 * R * b), 256,
                   [q, nr, r0, R, b, Qbr_t, Sbr_t, cb])
    return af, cb, ar_canon


def _pack_s_tile(session, q, a, b, OA, OB, ns, frozen_b, s0, s_local, S,
                 F_OBs, F_as, F_aO, F_bO, M_b_t, M_b4_t, M_a2_t, X_s, SA_s,
                 SAbs_t, Qbs_t, Sas_t, Qas_t, fa_buf, bc_buf, bs_canon_buf, localized):
    fa = fa_buf.view((2 * q + 2, S * b))
    bc = bc_buf.view((2 * q + 2, S * a))
    fa_top = fa.view((q, S, b))
    fa_bot = fa.view((q, S, b), offset=8 * q * S * b)
    bc_top = bc.view((q, S, a))
    bc_bot = bc.view((q, S, a), offset=8 * q * S * a)
    S_df = F_OBs.shape[2]
    # Dbs + Ebs into fa[0:Q]: Ebs first (beta=0), then Dbs accumulated (beta=1).
    session.gemm_batched(F_OBs, M_b4_t, transpose_left=True,
                         left_cols=(s_local, S), out=fa_top)
    session.gemm_batched(X_s, F_bO, transpose_left=True, transpose_right=True,
                         beta=1.0, left_cols=(s0, S), out=fa_top)
    if localized:
        bs_canon = bs_canon_buf.view((q, S * b))
        session.launch("transpose_factor_slab", _grid(q * S * b), 256,
                       [q, OB, frozen_b, b, S_df, s_local, S, 1, F_OBs, bs_canon])
        session.gemm_batched(F_OBs, M_b_t, transpose_left=True,
                             left_cols=(s_local, S), out=fa_bot)
    else:
        session.launch("transpose_factor_slab", _grid(q * S * b), 256,
                       [q, OB, frozen_b, b, S_df, s_local, S, 1, F_OBs, fa_bot])
        bs_canon = fa.view((q, S * b), offset=8 * q * S * b)
    session.launch("pack_factor_tails", _grid(2 * S * b), 256,
                   [q, ns, s0, S, b, SAbs_t, Qbs_t, fa])
    # Bas into bc[0:Q]: -F_as^T + SA_s^T @ F_aO_A^T (beta=1).
    session.launch("transpose_factor_slab", _grid(q * S * a), 256,
                   [q, a, 0, a, S_df, s_local, S, -1, F_as, bc_top])
    session.gemm_batched(SA_s, F_aO, transpose_left=True, transpose_right=True,
                         beta=1.0, left_cols=(s0, S), right_cols=(0, OA), out=bc_top)
    # Cas into bc[Q:2Q]: F_OBs^T @ M_a2_t.
    session.gemm_batched(F_OBs, M_a2_t, transpose_left=True,
                         left_cols=(s_local, S), out=bc_bot)
    session.launch("pack_factor_tails", _grid(2 * S * a), 256,
                   [q, ns, s0, S, a, Sas_t, Qas_t, bc])
    return fa, bc, bs_canon


def _factorized_device_dispersion(provider, spaces, corrections, energies, *,
                                  block_size, rotations, factor_cache_bytes, started):
    """Algebraic occupied-first 6-pair DFMO + zero-sync tiled contraction engine.

    Exploits D_A = Cocc_A Cocc_A^T and D_B = Cocc_B Cocc_B^T (verified on GPU)
    to reduce all 10 dispersion 3-index tensors to 6 occupied-first DFMO pairs
    computed in one call (or large memory-bounded slabs), with direct in-place
    strided-batched reconstruction, pre-localized X/W factors, zero per-block
    allocations/synchronizations, and coalesced warp reductions.
    """
    from .sapt_dispersion_cuda_setup import finish_components
    from .sapt_fisapt_cuda import localized_dispersion_block

    session = provider.session
    hf = spaces._hf_factorization
    CA, CB, S_ao = hf["Cocc_A"], hf["Cocc_B"], hf["S"]
    frozen_a, frozen_b = hf["frozen_a"], hf["frozen_b"]
    a, nr, b, ns = (energies[name].size for name in "arbs")
    OA, OB = CA.shape[1], CB.shape[1]
    OAB_n = OA + OB
    nao, q = provider.nao, provider.naux
    localized = rotations is not None
    ua, ub = rotations if localized else (None, None)

    t_setup = t_dfmo = t_pack = t_contract = 0.0
    count = dfmo_calls = r_batches = s_batches = cache_hits = 0

    with session.scope():
        t0 = time.perf_counter()
        a_t = session.gemm(spaces["a"], ua) if localized else spaces["a"]
        b_t = session.gemm(spaces["b"], ub) if localized else spaces["b"]
        OAB = session.empty((nao, OAB_n))
        session.launch("concat_columns", _grid(nao * OAB_n), 256,
                       [nao, OA, OB, CA, CB, OAB])
        S_r = session.gemm(S_ao, spaces["r"])
        S_s = session.gemm(S_ao, spaces["s"])
        S_CA = session.gemm(S_ao, CA)
        S_CB = session.gemm(S_ao, CB)
        S_a = session.gemm(S_ao, a_t)
        S_b = session.gemm(S_ao, b_t)
        SB_r = session.gemm_at_b(CB, S_r)
        SA_s = session.gemm_at_b(CA, S_s)
        SA_B = session.gemm_at_b(CA, S_CB)
        SB_A = session.gemm_at_b(CB, S_CA)
        SA_b = session.gemm_at_b(CA, S_b)
        SB_a = session.gemm_at_b(CB, S_a)
        M_a4_t = session.gemm(SA_B, SB_a, alpha=-2.0)
        M_b4_t = session.gemm(SB_A, SA_b, alpha=-2.0)
        if localized:
            M_a_t = session.empty((OA, a))
            session.launch("fill_active_rotation", _grid(OA * a), 256, [OA, frozen_a, a, ua, M_a_t])
            M_b_t = session.empty((OB, b))
            session.launch("fill_active_rotation", _grid(OB * b), 256, [OB, frozen_b, b, ub, M_b_t])
            Qar_t = session.gemm_at_b(ua, corrections["Qar"])
            SBar_t = session.gemm_at_b(ua, corrections["SBar"])
            Qbr_t = session.gemm_at_b(ub, corrections["Qbr"])
            Sbr_t = session.gemm_at_b(ub, corrections["Sbr"])
            SAbs_t = session.gemm_at_b(ub, corrections["SAbs"])
            Qbs_t = session.gemm_at_b(ub, corrections["Qbs"])
            Sas_t = session.gemm_at_b(ua, corrections["Sas"])
            Qas_t = session.gemm_at_b(ua, corrections["Qas"])
        else:
            M_a_t = M_b_t = None
            Qar_t, SBar_t, Qbr_t, Sbr_t = (corrections[k] for k in ("Qar", "SBar", "Qbr", "Sbr"))
            SAbs_t, Qbs_t, Sas_t, Qas_t = (corrections[k] for k in ("SAbs", "Qbs", "Sas", "Qas"))

        X_r = session.empty((OAB_n, nr))
        session.gemm(SA_B, SB_r, alpha=-2.0, out=X_r.view((OA, nr)))
        session.launch("linear_combination", _grid(OB * nr), 256,
                       [OB * nr, 2, 0, SB_r, SB_r, X_r.view((OB, nr), offset=8 * OA * nr)])
        X_s = session.empty((OAB_n, ns))
        session.launch("linear_combination", _grid(OA * ns), 256,
                       [OA * ns, 2, 0, SA_s, SA_s, X_s.view((OA, ns))])
        session.gemm(SB_A, SA_s, alpha=-2.0,
                     out=X_s.view((OB, ns), offset=8 * OA * ns))

        d, x = session.empty((a, b), zero=True), session.empty((a, b), zero=True)

        # Memory-aware tile & DFMO slab planner (planned before allocating buffers).
        avail = _available_device_bytes(session)
        R_blk, S_blk, tile_bytes = _choose_tile_extents(nr, ns, a, b, q, block_size, localized, avail)
        bytes_oo = 8 * q * (a + b) * OAB_n
        per_r = 8 * q * (OA + b)
        per_s = 8 * q * (OB + a)
        bytes_r_full = per_r * nr
        bytes_s_full = per_s * ns
        bytes_all_raw = bytes_oo + bytes_r_full + bytes_s_full

        headroom = max(0, avail - tile_bytes)
        base_var = getattr(provider, "variable_bytes", 256 * 1024**2)
        if headroom > bytes_all_raw + 4 * 1024**3:
            var_bytes = max(base_var, min(2 * 1024**3, headroom // 4))
        elif headroom < base_var * 2:
            var_bytes = min(base_var, max(16 * 1024**2, headroom // 4))
        else:
            var_bytes = base_var
        dfmo_reserve = min(var_bytes, headroom // 3) if headroom > 0 else 0
        max_raw_budget = max(0, headroom - dfmo_reserve)
        min_working_raw = bytes_oo + per_r * min(nr, block_size) + per_s * min(ns, block_size)
        if factor_cache_bytes > 0:
            requested = max(factor_cache_bytes, min_working_raw)
            raw_budget = min(requested, max_raw_budget) if max_raw_budget > 0 else requested
        else:
            raw_budget = min(8 * 1024**3, max_raw_budget)

        single_call = bytes_all_raw <= raw_budget
        if single_call:
            R_df_plan, S_df_plan = nr, ns
        else:
            slab_budget = max(0, raw_budget - bytes_oo)
            if bytes_s_full + per_r <= slab_budget:
                # Prefer retaining all S slabs in the inner loop to eliminate S recomputation.
                S_df_plan = ns
                R_df_plan = min(nr, max(1, (slab_budget - bytes_s_full) // per_r))
            else:
                R_df_plan = min(nr, max(1, (slab_budget // 2) // per_r))
                rem_s = max(0, slab_budget - R_df_plan * per_r)
                S_df_plan = min(ns, max(1, rem_s // per_s))
            R_blk = min(R_blk, R_df_plan)
            S_blk = min(S_blk, S_df_plan)

        # Pre-allocate reusable tile buffers ONCE outside all loops (zero per-block sync/alloc).
        af_buf = session.empty(((2 * q + 2) * R_blk * a,))
        cb_buf = session.empty(((2 * q + 2) * R_blk * b,))
        fa_buf = session.empty(((2 * q + 2) * S_blk * b,))
        bc_buf = session.empty(((2 * q + 2) * S_blk * a,))
        ar_canon_buf = session.empty((q * R_blk * a,)) if localized else None
        bs_canon_buf = session.empty((q * S_blk * b,)) if localized else None
        max_4idx = R_blk * a * S_blk * b
        v_buf = session.empty((max_4idx,))
        x_buf = session.empty((max_4idx,))
        w_buf = session.empty((max_4idx,))
        loc_ws = (session.empty((max_4idx,)), session.empty((max_4idx,))) if localized else None
        r_grid = _reduce_grid(a, b)
        t_setup += time.perf_counter() - t0

        def run_slab_pair(r_slab0, R_df, F_OAr, F_br, s_slab0, S_df, F_OBs, F_as, F_aO, F_bO):
            nonlocal count, t_pack, t_contract
            n_r_tiles = (R_df + R_blk - 1) // R_blk
            packed_s_bytes = 8 * ((2 * q + 2) * (a + b) + (q * b if localized else 0)) * S_df
            pack_s_limit = min(4 * 1024**3, _available_device_bytes(session) // 2)
            if factor_cache_bytes > 0:
                pack_s_limit = min(pack_s_limit, factor_cache_bytes)
            cache_packed_s = n_r_tiles > 1 and packed_s_bytes <= pack_s_limit
            with session.scope() if cache_packed_s else _nullcontext():
                packed_s_tiles = None
                if cache_packed_s:
                    tp_s0 = time.perf_counter()
                    fa_all = session.empty(((2 * q + 2) * S_df * b,))
                    bc_all = session.empty(((2 * q + 2) * S_df * a,))
                    bs_all = session.empty((q * S_df * b,)) if localized else None
                    packed_s_tiles = []
                    off_fa = off_bc = off_bs = 0
                    for s_local in range(0, S_df, S_blk):
                        S = min(S_blk, S_df - s_local)
                        s0 = s_slab0 + s_local
                        sz_fa = (2 * q + 2) * S * b
                        sz_bc = (2 * q + 2) * S * a
                        sz_bs = q * S * b
                        fa_v = fa_all.view((sz_fa,), offset=off_fa)
                        bc_v = bc_all.view((sz_bc,), offset=off_bc)
                        bs_v = bs_all.view((sz_bs,), offset=off_bs) if localized else None
                        packed_s_tiles.append(_pack_s_tile(
                            session, q, a, b, OA, OB, ns, frozen_b, s0, s_local, S,
                            F_OBs, F_as, F_aO, F_bO, M_b_t, M_b4_t, SB_a, X_s, SA_s,
                            SAbs_t, Qbs_t, Sas_t, Qas_t, fa_v, bc_v, bs_v, localized))
                        off_fa += 8 * sz_fa
                        off_bc += 8 * sz_bc
                        off_bs += 8 * sz_bs
                    t_pack += time.perf_counter() - tp_s0
                for r_local in range(0, R_df, R_blk):
                    R = min(R_blk, R_df - r_local)
                    r0 = r_slab0 + r_local
                    tp0 = time.perf_counter()
                    af, cb, ar_canon = _pack_r_tile(
                        session, q, a, b, OA, OB, nr, frozen_a, r0, r_local, R,
                        F_OAr, F_br, F_aO, F_bO, M_a_t, M_a4_t, SA_b, X_r, SB_r,
                        Qar_t, SBar_t, Qbr_t, Sbr_t, af_buf, cb_buf, ar_canon_buf, localized)
                    t_pack += time.perf_counter() - tp0
                    for s_idx, s_local in enumerate(range(0, S_df, S_blk)):
                        S = min(S_blk, S_df - s_local)
                        s0 = s_slab0 + s_local
                        if packed_s_tiles is not None:
                            fa, bc, bs_canon = packed_s_tiles[s_idx]
                        else:
                            tp1 = time.perf_counter()
                            fa, bc, bs_canon = _pack_s_tile(
                                session, q, a, b, OA, OB, ns, frozen_b, s0, s_local, S,
                                F_OBs, F_as, F_aO, F_bO, M_b_t, M_b4_t, SB_a, X_s, SA_s,
                                SAbs_t, Qbs_t, Sas_t, Qas_t, fa_buf, bc_buf, bs_canon_buf, localized)
                            t_pack += time.perf_counter() - tp1
                        tc0 = time.perf_counter()
                        v = session.gemm_at_b(ar_canon, bs_canon, out=v_buf.view((R * a, S * b)))
                        direct = session.gemm_at_b(af, fa, out=x_buf.view((R * a, S * b)))
                        crossed = session.gemm_at_b(bc, cb, out=w_buf.view((S * a, R * b)))
                        if not localized:
                            session.launch("reduce_dispersion", r_grid, 256,
                                           [a, b, R, S, r0, s0, energies["a"], energies["b"],
                                            energies["r"], energies["s"], v, direct, crossed, d, x])
                        else:
                            localized_dispersion_block(
                                session, v, direct, crossed, energies, ua, ub, d, x,
                                R=R, S=S, r0=r0, s0=s0, pre_localized_xw=True, workspace=loc_ws)
                        t_contract += time.perf_counter() - tc0
                        count += 1

        if single_call:
            allocated_raw = bytes_all_raw
            td0 = time.perf_counter()
            factors = _dfmo_call(session, provider, {
                "aO": (a_t, OAB), "bO": (b_t, OAB),
                "OAr": (CA, spaces["r"]), "br": (b_t, spaces["r"]),
                "OBs": (CB, spaces["s"]), "as": (a_t, spaces["s"]),
            }, variable_bytes=var_bytes)
            t_dfmo += time.perf_counter() - td0
            dfmo_calls = 1
            r_batches = s_batches = 1
            cached_slabs = 1
            cache_hits = max(0, ((nr + R_blk - 1) // R_blk - 1) * ((ns + S_blk - 1) // S_blk))
            run_slab_pair(0, nr, factors["OAr"], factors["br"],
                          0, ns, factors["OBs"], factors["as"],
                          factors["aO"], factors["bO"])
        else:
            # Memory-bounded tiled mode: compute tiny OO once, then large R/S DFMO slabs.
            td0 = time.perf_counter()
            oo_factors = _dfmo_call(session, provider, {
                "aO": (a_t, OAB), "bO": (b_t, OAB),
            }, variable_bytes=var_bytes)
            t_dfmo += time.perf_counter() - td0
            dfmo_calls += 1
            F_aO, F_bO = oo_factors["aO"], oo_factors["bO"]
            allocated_raw = bytes_oo + R_df_plan * per_r + S_df_plan * per_s
            from .sapt_dispersion_cuda_setup import DeviceMath
            m = DeviceMath(session)
            n_r_slabs = (nr + R_df_plan - 1) // R_df_plan
            # If all S fits in one slab alongside one R slab, reuse S across all R slabs.
            reuse_s = (S_df_plan == ns and n_r_slabs > 1
                       and bytes_oo + R_df_plan * per_r + bytes_s_full <= raw_budget)
            cached_s = None
            cached_slabs = 1 if reuse_s else 0
            if reuse_s:
                td1 = time.perf_counter()
                cached_s = _dfmo_call(session, provider, {
                    "OBs": (CB, spaces["s"]), "as": (a_t, spaces["s"]),
                }, variable_bytes=var_bytes)
                t_dfmo += time.perf_counter() - td1
                dfmo_calls += 1
                s_batches += 1
            for r_slab0 in range(0, nr, R_df_plan):
                R_df = min(R_df_plan, nr - r_slab0)
                with session.scope():
                    r_win = spaces["r"] if (r_slab0 == 0 and R_df == nr) else m.window(spaces["r"], r_slab0, R_df)
                    td1 = time.perf_counter()
                    rf = _dfmo_call(session, provider, {
                        "OAr": (CA, r_win), "br": (b_t, r_win),
                    }, variable_bytes=var_bytes)
                    t_dfmo += time.perf_counter() - td1
                    dfmo_calls += 1
                    r_batches += 1
                    if reuse_s:
                        if r_slab0 > 0:
                            cache_hits += 1
                        run_slab_pair(r_slab0, R_df, rf["OAr"], rf["br"],
                                      0, ns, cached_s["OBs"], cached_s["as"], F_aO, F_bO)
                    else:
                        for s_slab0 in range(0, ns, S_df_plan):
                            S_df = min(S_df_plan, ns - s_slab0)
                            with session.scope():
                                s_win = spaces["s"] if (s_slab0 == 0 and S_df == ns) else m.window(spaces["s"], s_slab0, S_df)
                                td2 = time.perf_counter()
                                sf = _dfmo_call(session, provider, {
                                    "OBs": (CB, s_win), "as": (a_t, s_win),
                                }, variable_bytes=var_bytes)
                                t_dfmo += time.perf_counter() - td2
                                dfmo_calls += 1
                                s_batches += 1
                                run_slab_pair(r_slab0, R_df, rf["OAr"], rf["br"],
                                              s_slab0, S_df, sf["OBs"], sf["as"], F_aO, F_bO)

        combined, components = finish_components(session, d, x)
        d_host, x_host = session.download(d), session.download(x)
        combined_host, scalar_host = session.download(combined), session.download(components)
    if any(not np.isfinite(value).all() for value in (d_host, x_host, combined_host, scalar_host)):
        raise FloatingPointError("Nonfinite signed CUDA dispersion components")
    total_wall = time.perf_counter() - started
    return {
        "Disp20,u": float(scalar_host[0]), "Exch-Disp20,u": float(scalar_host[1]),
        "Disp0,u": float(scalar_host[2]),
        "Disp20 occupied pairs": d_host, "Exch-Disp20 occupied pairs": x_host,
        "Disp0 occupied pairs": combined_host,
        "Disp0 execution": {
            "backend": ("CUDA_CUEST_LOCALIZED_DEVICE_EXPERIMENTAL" if localized
                        else "CUDA_CUEST_DEVICE_INPUT_EXPERIMENTAL"),
            "occupied_representation": "localized" if localized else "canonical",
            "compute_blocks": count, "block_size": block_size,
            "compute_tile_extents": [R_blk, S_blk],
            "dfmo_slab_extents": [R_df_plan, S_df_plan],
            "dfmo_calls": dfmo_calls,
            "dfmo_variable_bytes": var_bytes,
            "factor_cache": {
                "policy": "ALGEBRAIC_OCCUPIED_FIRST_FACTORIZED",
                "budget_bytes": factor_cache_bytes,
                "allocated_bytes": allocated_raw,
                "cached_slabs": cached_slabs,
                "hits": cache_hits,
                "r_factor_batches": r_batches,
                "s_factor_batches": s_batches,
            },
            "stage_seconds": {
                "setup": t_setup, "dfmo": t_dfmo,
                "pack": t_pack, "contraction_and_reduction": t_contract,
            },
            "blocked_execution_seconds": total_wall,
            "owned_peak_device_bytes_session_lifetime": session.peak_live_bytes,
            "declared_native_persistent_bytes": session.native_bytes,
            "cuda_source_sha256": hashlib.sha256(CUDA_SOURCE.encode()).hexdigest(),
            "actual_gpu_execution_verified": False,
            "scope": "Occupied-first factorized device DFMO and zero-sync GPU contractions/reductions; "
                     "excludes raw HF upload/setup and plan creation/cleanup",
            "producer": provider.report(),
        },
    }


def blocked_device_dispersion(provider, spaces, corrections, energies, *, block_size=24, rotations=None,
                              factor_cache_bytes=0, algorithm="auto"):
    """GPU-only numerical continuation of prepare_hf, including global sums.

    All inputs must be device outputs of validated raw-HF preparation on this
    session. The provider must prepare persistent DFMO parameters before the
    caller enters its enclosing setup/compute scope. No host coefficient upload,
    correction staging or CPU numerical fallback is used here.

    When ``algorithm="auto"`` (default) and ``spaces`` carries verified
    idempotent HF metadata from ``prepare_hf``, uses the algebraic occupied-first
    factorized path (collapsing all 10 tensors to 6 occupied-first DFMO pairs).
    Plain synthetic space dictionaries or ``algorithm="blocked"`` retain the
    general blocked packed-S path with pre-allocated zero-sync contraction buffers.
    """
    from .sapt_dispersion_cuda_setup import DeviceMath, coefficient_pairs as device_pairs
    from .sapt_dispersion_cuda_setup import finish_components
    if type(block_size) is not int or block_size < 1:
        raise ValueError("CUDA dispersion block_size must be a positive integer")
    if type(factor_cache_bytes) is not int or factor_cache_bytes < 0:
        raise ValueError("Dispersion factor cache needs a nonnegative integer byte budget")
    if algorithm not in ("auto", "factorized", "blocked"):
        raise ValueError("CUDA dispersion algorithm must be 'auto', 'factorized', or 'blocked'")
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
    hf = getattr(spaces, "_hf_factorization", None)
    if algorithm != "blocked":
        if hf is not None and _hf_idempotent(session, m, hf, spaces["a"], spaces["b"], provider.nao):
            return _factorized_device_dispersion(
                provider, spaces, corrections, energies,
                block_size=block_size, rotations=rotations,
                factor_cache_bytes=factor_cache_bytes, started=started)
        if algorithm == "factorized":
            raise ValueError("Factorized GPU dispersion requires verified idempotent HF inputs from prepare_hf")
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
    max_r = min(block_size, nr)
    max_s = min(block_size, ns)
    max_4idx = max_r * a * max_s * b
    r_grid = _reduce_grid(a, b)
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
        # Pre-allocate reusable contraction buffers ONCE outside R/S loops so
        # cache-hit blocks perform zero allocations and zero device syncs.
        v_buf = session.empty((max_4idx,))
        x_buf = session.empty((max_4idx,))
        w_buf = session.empty((max_4idx,))
        loc_ws = (tuple(session.empty((max_4idx,)) for _ in range(4))
                  if rotations is not None else None)
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
                    entry=cached.get(s0)
                    if entry is None or r0==0:
                        with session.scope():
                            sf = _factor_batch(session,provider,device_pairs(session,spaces,"s",s0,S),device=True)
                            st = {name:m.window(corrections[name],s0,S)
                                  for name in ("SAbs","Qbs","Sas","Qas")}
                            fa = _pack(session,q,sf,"Dbs","Abs","Ebs",None,st["SAbs"],st["Qbs"],
                                       out=entry[0] if entry is not None else None)
                            bc = _pack(session,q,sf,"Bas","Cas",None,None,st["Sas"],st["Qas"],
                                       out=entry[1] if entry is not None else None)
                            s_batches+=1
                            v = session.gemm_at_b(af.view((q,R*a)),fa.view((q,S*b),offset=q*S*b*8),
                                                  out=v_buf.view((R*a,S*b)))
                            direct = session.gemm_at_b(af,fa,out=x_buf.view((R*a,S*b)))
                            crossed = session.gemm_at_b(bc,cb,out=w_buf.view((S*a,R*b)))
                            if rotations is None:
                                session.launch("reduce_dispersion",r_grid,256,
                                               [a,b,R,S,r0,s0,energies["a"],energies["b"],
                                                energies["r"],energies["s"],v,direct,crossed,d,x])
                            else:
                                localized_dispersion_block(session,v,direct,crossed,energies,
                                    *rotations,d,x,R=R,S=S,r0=r0,s0=s0,workspace=loc_ws)
                    else:
                        fa,bc=entry
                        cache_hits+=1
                        v = session.gemm_at_b(af.view((q,R*a)),fa.view((q,S*b),offset=q*S*b*8),
                                              out=v_buf.view((R*a,S*b)))
                        direct = session.gemm_at_b(af,fa,out=x_buf.view((R*a,S*b)))
                        crossed = session.gemm_at_b(bc,cb,out=w_buf.view((S*a,R*b)))
                        if rotations is None:
                            session.launch("reduce_dispersion",r_grid,256,
                                           [a,b,R,S,r0,s0,energies["a"],energies["b"],
                                            energies["r"],energies["s"],v,direct,crossed,d,x])
                        else:
                            localized_dispersion_block(session,v,direct,crossed,energies,
                                *rotations,d,x,R=R,S=S,r0=r0,s0=s0,workspace=loc_ws)
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
