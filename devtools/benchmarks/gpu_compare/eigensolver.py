"""Explicit GPU-only canonical eigensolver control; no silent fallback.

The native GPU4PySCF generalized solver remains the default. This opt-in uses
CuPy's standard Hermitian solver twice, with the same absolute overlap cutoff.
No host matrix diagonalization or installed-package mutation is performed.
"""
import contextlib
import math
from unittest.mock import patch


def canonical_eigh(h, s, cutoff, xp):
    """Solve in the retained overlap subspace; xp is CuPy (NumPy in unit tests)."""
    if not math.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("Canonical solver requires a positive overlap cutoff")
    if h.ndim != 2 or h.shape != s.shape or h.shape[0] != h.shape[1]:
        raise ValueError("Incompatible Fock/overlap dimensions")
    values, vectors = xp.linalg.eigh(s)
    keep = values > cutoff
    if not int(xp.count_nonzero(keep)):
        raise ValueError("No retained overlap directions")
    x = vectors[:, keep] / xp.sqrt(values[keep])
    reduced = x.T.conj() @ h @ x
    energies, rotations = xp.linalg.eigh(reduced)
    return energies, x @ rotations


@contextlib.contextmanager
def eigensolver_policy(mf, config, record, profile=False):
    mode = config.get("pyscf_eigensolver", "native")
    if mode not in ("native", "cupy-canonical"):
        raise ValueError(f"Unknown explicit PySCF eigensolver: {mode}")
    if mode == "native":
        record["eigensolver"] = dict(mode=mode, modified=False)
        yield
        return
    cutoff = config.get("overlap_eigenvalue_cutoff")
    if cutoff is None or not math.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("Explicit canonical solver needs the audited overlap cutoff")
    import cupy as cp
    proof = record["eigensolver"] = dict(
        mode=mode, modified=True, host_eigensolver=False, calls=0,
        absolute_overlap_cutoff=cutoff, implementation="CuPy Hermitian eigh / canonical X",
        note="Explicit algorithm control, not unmodified GPU4PySCF eigensolver")

    def eig(h, s, overwrite=False, x=None, **kwargs):
        if kwargs:
            raise TypeError(f"Unrecognized eigensolver arguments: {sorted(kwargs)}")
        for matrix in (h, s):
            if (not isinstance(matrix, cp.ndarray) or matrix.dtype != cp.float64 or
                    int(matrix.device.id) != 0):
                raise RuntimeError("Canonical eigensolver requires visible-GPU FP64 matrices")
        # Native callers may provide a cached orthogonalizer. Recompute from
        # the explicit S and prescribed cutoff rather than changing subspaces.
        if x is not None:
            proof["precomputed_x_recomputed_from_s"] = True
        energies, coefficients = canonical_eigh(h, s, cutoff, cp)
        if not all(isinstance(a, cp.ndarray) and int(a.device.id) == 0
                   for a in (energies, coefficients)):
            raise RuntimeError("Canonical eigensolver returned non-GPU arrays")
        cp.cuda.runtime.deviceSynchronize()
        if not bool(cp.isfinite(energies).all()) or not bool(cp.isfinite(coefficients).all()):
            raise RuntimeError("Nonfinite canonical eigensolution")
        rank = int(energies.size)
        if proof.get("retained_rank", rank) != rank:
            raise RuntimeError("Overlap rank changed between eigensolver calls")
        proof.update(calls=proof["calls"] + 1, retained_rank=rank)
        if profile:
            # The full residual need not vanish along removed S directions.
            gram = coefficients.T.conj() @ s @ coefficients
            error = float(cp.max(cp.abs(gram - cp.eye(rank))).item())
            proof["maximum_s_orthogonality_error"] = max(
                proof.get("maximum_s_orthogonality_error", 0.), error)
            if error > 1e-7:
                raise RuntimeError("Canonical orbitals failed S-orthogonality")
        return energies, coefficients

    with patch.object(mf, "eig", eig):
        yield
        if proof["calls"] < 1:
            raise RuntimeError("Explicit eigensolver was not consumed")
