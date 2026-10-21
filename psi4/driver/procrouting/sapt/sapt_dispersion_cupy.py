"""Experimental CuPy SAPT0 blocks; no NumPy fallback or implicit dependency.

cuEST factors must already include the fitting metric. Scientific expressions
remain Python array operations. Native library calls are supplied by the DFMO
producer, never by a new native dispersion kernel.
"""
import time

import numpy as np


def require_cupy():
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError("The experimental CuPy backend requires optional CuPy "
                           "in the dedicated GPU environment") from exc
    if cp.cuda.runtime.is_hip or cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("CuPy SAPT0 requires an actual NVIDIA CUDA device")
    return cp


def _pack(xp, factors, first, second, tail1, tail2):
    """Q,left,right -> [Q,(left,right)] with right orbital fastest."""
    a, b = factors[first], factors[second]
    if a.shape != b.shape or a.ndim != 3:
        raise ValueError(f"Incompatible DF factors {first}/{second}")
    q, left, right = a.shape
    if tail1.shape != (right, left) or tail2.shape != (right, left):
        raise ValueError(f"Wrong rank-one correction shapes for {first}")
    return xp.concatenate((a.reshape(q, left * right),
                           b.reshape(q, left * right),
                           tail1.T.reshape(1, left * right),
                           tail2.T.reshape(1, left * right)), axis=0)


def _energy_block(xp, af, fa, bc, cb, ea, er, eb, es):
    """Pure block equations; xp is NumPy ONLY in independent CPU tests.

    All packed columns use a-fastest (r,a) / b-fastest (s,b).
    Returns signed occupied-pair matrices, never a four-index full-space array.
    """
    a, r, b, s = ea.size, er.size, eb.size, es.size
    if (af.shape[1] != r * a or fa.shape[1] != s * b
            or bc.shape[1] != s * a or cb.shape[1] != r * b
            or af.shape[0] != fa.shape[0] or af.shape[0] != bc.shape[0]
            or af.shape[0] != cb.shape[0] or af.shape[0] < 4
            or (af.shape[0] - 2) % 2):
        raise ValueError("Malformed blocked SAPT0 packed factors")
    q = (af.shape[0] - 2) // 2
    v = af[:q].T @ fa[q:2 * q]
    denominator = ((ea[None, :] - er[:, None]).reshape(-1, 1)
                   + (eb[None, :] - es[:, None]).reshape(1, -1))
    t = v / denominator
    disp = 4.0 * xp.sum((t * v).reshape(r, a, s, b), axis=(0, 2))
    x = af.T @ fa
    w = bc.T @ cb
    # W[(s,a),(r,b)] -> the same (r,a,s,b) index order as T.
    crossed = w.reshape(s, a, r, b).transpose(2, 1, 0, 3)
    exch = -2.0 * xp.sum(t.reshape(r, a, s, b)
                         * (x.reshape(r, a, s, b) + crossed), axis=(0, 2))
    return disp, exch


def _pairs(spaces, side, slab):
    if side == "r":
        return {
            "Aar": (spaces["r"][:, slab], spaces["a"]),
            "Bbr": (spaces["r1"][:, slab], spaces["b"]),
            "Cbr": (spaces["r"][:, slab], spaces["b2"]),
            "Dar": (spaces["r3"][:, slab], spaces["a"]),
            "Ear": (spaces["r"][:, slab], spaces["a4"]),
        }
    return {
        "Abs": (spaces["s"][:, slab], spaces["b"]),
        "Bas": (spaces["s1"][:, slab], spaces["a"]),
        "Cas": (spaces["s"][:, slab], spaces["a2"]),
        "Dbs": (spaces["s3"][:, slab], spaces["b"]),
        "Ebs": (spaces["s"][:, slab], spaces["b4"]),
    }


def _validate_inputs(spaces, corrections, energies):
    needed = ("a", "r", "b", "s", "r1", "s1", "a2", "b2", "r3", "s3", "a4", "b4")
    if set(spaces) != set(needed) or set(energies) != {"a", "r", "b", "s"}:
        raise ValueError("Incomplete canonical orbital spaces/energies")
    nao = spaces["a"].shape[0]
    for name in needed:
        matrix = spaces[name]
        reference = name[0] if name[0] in "abrs" else name
        if (matrix.ndim != 2 or matrix.shape != (nao, energies[reference].size)
                or matrix.dtype != np.float64 or not np.isfinite(matrix).all()):
            raise ValueError(f"Wrong FP64 orbital space {name}")
    for name, energy in energies.items():
        if energy.ndim != 1 or energy.dtype != np.float64 or not np.isfinite(energy).all():
            raise ValueError(f"Wrong FP64 energy vector {name}")
    if (any(energies[name].size == 0 for name in "arbs")
            or max(energies["a"]) >= min(energies["r"])
            or max(energies["b"]) >= min(energies["s"])):
        raise ValueError("Nonempty occupied/virtual spaces with positive HF gaps required")
    shapes = {"Qar": ("a", "r"), "SBar": ("a", "r"),
              "Qbr": ("b", "r"), "Sbr": ("b", "r"),
              "SAbs": ("b", "s"), "Qbs": ("b", "s"),
              "Sas": ("a", "s"), "Qas": ("a", "s")}
    if set(corrections) != set(shapes):
        raise ValueError("Incomplete exchange-dispersion rank-one corrections")
    for name, (occ, vir) in shapes.items():
        array = corrections[name]
        if (array.shape != (energies[occ].size, energies[vir].size)
                or array.dtype != np.float64 or not np.isfinite(array).all()):
            raise ValueError(f"Wrong FP64 correction {name}")


def _validate_factors(cp, factors, pairs, naux):
    if set(factors) != set(pairs):
        raise ValueError("DFMO producer returned an incomplete factor batch")
    device = cp.cuda.runtime.getDevice()
    for name, (left, right) in pairs.items():
        value = factors[name]
        if (not isinstance(value, cp.ndarray) or value.dtype != cp.float64
                or value.shape != (naux, left.shape[1], right.shape[1])
                or value.device.id != device):
            raise ValueError(f"DFMO factor {name} is not the expected owned FP64 CUDA array")


def blocked_dispersion(provider, spaces, corrections, energies, *, block_size=24):
    """Device factors, packing, contractions, division and reductions.

    Setup orbital matrices/energies are host inputs. Only final occupied-pair
    matrices and scalar reporting are downloaded. No fragment/S-inf algorithm
    is implemented. A provider must implement check_memory() and compute().
    """
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise ValueError("CuPy dispersion block_size must be a positive integer")
    _validate_inputs(spaces, corrections, energies)
    cp = require_cupy()
    if provider.cp is not cp or provider.nao != spaces["a"].shape[0]:
        raise ValueError("Foreign array backend or incompatible DFMO AO space")
    started = time.perf_counter()
    a, r, b, s = (energies[name].size for name in "arbs")
    q = provider.naux
    # Conservative live-block bound, including producer outputs, packed copies,
    # products, amplitude, denominator, reductions and crossed-layout arithmetic.
    rr, ss = min(r, block_size), min(s, block_size)
    estimate = 8 * (4 * (2 * q + 2) * (a + b) * (rr + ss)
                    + 16 * a * b * rr * ss + 8 * a * b
                    + sum(x.size for x in corrections.values()) + a + r + b + s)
    provider.check_memory(estimate)
    e = {name: cp.asarray(value) for name, value in energies.items()}
    corr = {name: cp.asarray(value) for name, value in corrections.items()}
    disp = cp.zeros((a, b), dtype=cp.float64)
    exch = cp.zeros_like(disp)
    blocks = 0
    for r0 in range(0, r, block_size):
        rs = slice(r0, min(r, r0 + block_size))
        pairs = _pairs(spaces, "r", rs)
        rf = provider.compute(pairs)
        _validate_factors(cp, rf, pairs, q)
        rf["Far"] = rf.pop("Dar") + rf.pop("Ear")
        af = _pack(cp, rf, "Aar", "Far", corr["Qar"][:, rs], corr["SBar"][:, rs])
        cb = _pack(cp, rf, "Bbr", "Cbr", corr["Qbr"][:, rs], corr["Sbr"][:, rs])
        del rf
        for s0 in range(0, s, block_size):
            sslice = slice(s0, min(s, s0 + block_size))
            pairs = _pairs(spaces, "s", sslice)
            sf = provider.compute(pairs)
            _validate_factors(cp, sf, pairs, q)
            sf["Fbs"] = sf.pop("Dbs") + sf.pop("Ebs")
            fa = _pack(cp, sf, "Fbs", "Abs", corr["SAbs"][:, sslice], corr["Qbs"][:, sslice])
            bc = _pack(cp, sf, "Bas", "Cas", corr["Sas"][:, sslice], corr["Qas"][:, sslice])
            del sf
            d, x = _energy_block(cp, af, fa, bc, cb,
                                 e["a"], e["r"][rs], e["b"], e["s"][sslice])
            disp += d
            exch += x
            blocks += 1
            del fa, bc, d, x
        del af, cb
    cp.cuda.get_current_stream().synchronize()
    # Explicit device->host final component matrices, not factors/products.
    d_host, x_host = cp.asnumpy(disp), cp.asnumpy(exch)
    if not np.isfinite(d_host).all() or not np.isfinite(x_host).all():
        raise FloatingPointError("Nonfinite signed CuPy dispersion components")
    return {
        "Disp20,u": float(d_host.sum()), "Exch-Disp20,u": float(x_host.sum()),
        "Disp20 occupied pairs": d_host, "Exch-Disp20 occupied pairs": x_host,
        "Disp0 execution": {
            "backend": "CUPY_CUEST_EXPERIMENTAL", "block_replays": blocks,
            "block_size": block_size, "estimated_block_bytes": estimate,
            "blocked_execution_seconds": time.perf_counter() - started,
            "scope": "Caller-supplied host coefficients; CuPy factors/packing/blocks/reductions; "
                     "excludes upstream orbital setup and native-plan creation/cleanup",
            "actual_gpu_execution_verified": False,
            "note": "Device API use is not independent CUDA/cuBLAS trace or physics proof",
            "producer": provider.report(),
        },
    }
