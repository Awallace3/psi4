# Copyright (c) 2007-2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""Bounded DF operator actions, not a CamCASP preset.

The factors are caller-supplied scientific operands. In plain Coulomb DF,
``oo`` and ``ov`` are MO three-centre integrals and ``dual_ov``/``dual_vv``
are solved against the Coulomb metric. Keeping left/right factors separate
does not assume that other fitting models are a single whitened Gram matrix.
No metric inversion, constrained fitting, kernel construction, or native admission
is performed here.
"""
from numbers import Real

import numpy as np


def mo_three_center_shell(coulomb, main, coefficients, shell_index, *,
                          max_bytes=512*1024**2):
    """Return one AUX shell's raw MO integrals, shaped (function,MO,MO).

    ``coefficients`` must already be in the explicit spherical MAIN order;
    no Psi4-to-MAIN conversion, metric solve, or state certification is inferred.
    The selected AUX shell retains its original Cartesian/spherical ordering.
    Only its AO integrals are generated, never the full AUX-by-AO-pair tensor.

    Admission conservatively reserves 15 rows (the largest admitted S--G
    shell), native spherical accumulation (at most nine rows), coefficient
    storage, output, validation buffers and sequential two-matrix-product
    temporaries. Caller-retained earlier outputs, basis
    descriptors and implementation-specific Libint/BLAS workspace are excluded:
    a collecting consumer must maintain its own aggregate allocation ledger.
    The returned array is independent of inputs. No symmetrization is applied.
    """
    from psi4 import core

    if not isinstance(coulomb, core.IsaAuxCoulomb) or not isinstance(main, core.IsaExplicitBasis):
        raise ValueError("explicit Coulomb provider and MAIN basis required")
    if main.role != core.IsaBasisRole.Orbital:
        raise ValueError("MAIN must have Orbital role")
    if type(shell_index) is not int or shell_index < 0:
        raise ValueError("shell_index must be a nonnegative integer")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    nao = main.nfunction
    if (not isinstance(coefficients, np.ndarray) or coefficients.dtype != np.float64
            or coefficients.ndim != 2 or coefficients.shape[0] != nao
            or coefficients.shape[1] < 1):
        raise ValueError("coefficients must be nonempty float64 (nmain,nmo)")
    nmo = coefficients.shape[1]
    # Native generation holds its result and spherical accumulation together.
    # Include room for noncontiguous coefficient packing and finite masks.
    needed = 8*(24*nao**2 + 18*nmo**2 + 4*nao*nmo)
    if needed > max_bytes:
        raise ValueError("MO shell transform byte resource limit exceeded")
    work = 15*(2*nao**2*nmo + 2*nao*nmo**2)
    if work > FactorizedDFOperators.MAX_WORK:
        raise ValueError("MO shell transform work resource limit exceeded")
    if not np.isfinite(coefficients).all():
        raise ValueError("coefficients must be finite")
    ao = np.asarray(coulomb.three_center_shell_block(
        main, shell_index, 1, 15*nao**2*8)).reshape(-1, nao, nao)
    result = np.empty((len(ao), nmo, nmo))
    with np.errstate(over="raise", invalid="raise"):
        for row in range(len(ao)):
            result[row] = (coefficients.T @ ao[row]) @ coefficients
    if not np.isfinite(result).all():
        raise ValueError("nonfinite MO shell integrals")
    return result


class FactorizedDFOperators:
    """Owned occupied-fast H1/H2 actions with no dense OV-square operators.

    Factor shapes are (naux,nocc,nocc), (naux,nocc,nvir),
    (naux,nocc,nvir), and (naux,nvir,nvir), respectively. Gaps have shape
    (nocc*nvir,), indexed by ``a*nocc+i``. Factors are used as supplied;
    no symmetry repair, regularization, or claim of constrained-DF parity.

    H1 = gap + 4 V - exchange*(X+Y) + 4*local_scale*D S D.T;
    H2 = gap - exchange*(X-Y). Optional supplied kernel operands D and S
    have shapes (nov,nkernel) and (nkernel,nkernel); no grid is inferred.
    RHS blocks are finite float64 arrays (nov,nrhs), at most 64 columns.
    Admission accounts owned snapshots, conversion/validation buffers and
    explicit action temporaries under the caller's max_bytes. Caller storage and
    implementation-specific BLAS workspace are not a process-RSS guarantee.
    Auxiliary terms and RHS columns are processed sequentially.
    Overflow fails closed as ``FloatingPointError`` for NumPy arithmetic or
    ``ValueError`` when detected by the final finite-output check.
    """
    MAX_BYTES = 512*1024**2
    MAX_WORK = 64_000_000_000

    def __init__(self, gaps, oo, ov, dual_ov, dual_vv, *, exact_exchange,
                 local_scale=0., kernel_legs=None, auxiliary_kernel=None,
                 max_bytes=MAX_BYTES):
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        # Require array inputs so dimensions can be admitted before conversion.
        operands = (gaps, oo, ov, dual_ov, dual_vv)
        if any(not isinstance(a, np.ndarray) or a.dtype != np.float64 for a in operands):
            raise ValueError("factors and gaps must be float64 arrays")
        if ov.ndim != 3 or min(ov.shape) < 1:
            raise ValueError("ov must have nonempty (naux,nocc,nvir) shape")
        naux, nocc, nvir = ov.shape
        nov = nocc*nvir
        expected = ((nov,), (naux, nocc, nocc), (naux, nocc, nvir),
                    (naux, nocc, nvir), (naux, nvir, nvir))
        if any(a.shape != shape for a, shape in zip(operands, expected)):
            raise ValueError("factor/gap dimensions mismatch")
        if (not isinstance(local_scale, Real) or not np.isfinite(local_scale)
                or not 0 <= local_scale <= 1):
            raise ValueError("local_scale must be finite in [0,1]")
        if (kernel_legs is None) != (auxiliary_kernel is None):
            raise ValueError("both kernel operands must be declared")
        if local_scale and kernel_legs is None:
            raise ValueError("nonzero local_scale requires kernel operands")
        nkernel = 0
        if kernel_legs is not None:
            if any(not isinstance(a, np.ndarray) or a.dtype != np.float64
                   for a in (kernel_legs, auxiliary_kernel)):
                raise ValueError("kernel operands must be float64 arrays")
            if (kernel_legs.ndim != 2 or kernel_legs.shape[0] != nov
                    or kernel_legs.shape[1] < 1):
                raise ValueError("kernel legs must have shape (nov,nkernel)")
            nkernel = kernel_legs.shape[1]
            if auxiliary_kernel.shape != (nkernel, nkernel):
                raise ValueError("auxiliary kernel dimensions mismatch")
            operands += (kernel_legs, auxiliary_kernel)
        storage = sum(a.size*8 for a in operands)
        if 3*storage > max_bytes:
            raise ValueError("factor snapshot byte resource limit exceeded")
        if any(not np.isfinite(a).all() for a in operands) or np.any(gaps <= 0):
            raise ValueError("finite factors and positive finite gaps required")
        if (not np.isscalar(exact_exchange) or not np.isfinite(exact_exchange)
                or not 0 <= exact_exchange <= 1):
            raise ValueError("exact_exchange must be finite in [0,1]")
        # Bytes backing prevents callers from re-enabling writes on snapshots.
        snapshots = tuple(
            np.frombuffer(a.tobytes(order="C"), dtype=np.float64).reshape(a.shape)
            for a in operands)
        self._gaps, self._oo, self._ov, self._dual_ov, self._dual_vv = snapshots[:5]
        self._legs, self._kernel = snapshots[5:] if nkernel else (None, None)
        self.naux, self.nocc, self.nvir, self.nov = naux, nocc, nvir, nov
        self._nkernel, self._local_scale = nkernel, float(local_scale)
        self._exchange = float(exact_exchange)
        self._max_bytes, self._storage = max_bytes, storage

    def _apply(self, rhs, first):
        if (not isinstance(rhs, np.ndarray) or rhs.dtype != np.float64
                or rhs.ndim != 2 or rhs.shape[0] != self.nov
                or not 1 <= rhs.shape[1] <= 64):
            raise ValueError("rhs must be float64 (nov,nrhs), 1 <= nrhs <= 64")
        nrhs = rhs.shape[1]
        workspace = 8*(4*self.nov*nrhs + 16*self.nov +
                       4*self.nocc**2 + 4*self.nvir**2 + 4*self._nkernel*nrhs)
        if self._storage + workspace > self._max_bytes:
            raise ValueError("action byte resource limit exceeded")
        # Count multiply-adds as two operations, including both products in
        # each exchange contraction and conservative elementwise overhead.
        work = 2*self.nov*nrhs
        work += self.naux*nrhs*(12*self.nov + 6*self.nocc**2*self.nvir +
                               2*self.nocc*self.nvir**2)
        if first and self._local_scale:
            work += nrhs*(4*self.nov*self._nkernel + 2*self._nkernel**2 + 2*self.nov)
        if work > self.MAX_WORK:
            raise ValueError("factorized action work resource limit exceeded")
        if not np.isfinite(rhs).all():
            raise ValueError("rhs must be finite")
        with np.errstate(over="raise", invalid="raise"):
            result = self._gaps[:, None]*rhs
            for col in range(nrhs):
                z = rhs[:, col].reshape(self.nvir, self.nocc).T
                accum = np.zeros((self.nocc, self.nvir))
                for p in range(self.naux):
                    if first:
                        accum += 4*self._ov[p]*np.sum(self._dual_ov[p]*z)
                    if self._exchange:
                        x = (self._oo[p] @ z) @ self._dual_vv[p].T
                        y = (self._ov[p] @ z.T) @ self._dual_ov[p]
                        accum -= self._exchange*(x+y if first else x-y)
                result[:, col] += accum.T.reshape(self.nov)
            if first and self._local_scale:
                result += (4*self._local_scale)*(
                    self._legs @ (self._kernel @ (self._legs.T @ rhs)))
        if not np.isfinite(result).all():
            raise ValueError("nonfinite factorized response action")
        return result

    def apply_h1(self, rhs):
        """Apply H1 to a bounded RHS block; return an owned array."""
        return self._apply(rhs, True)

    def apply_h2(self, rhs):
        """Apply H2 to a bounded RHS block; return an owned array."""
        return self._apply(rhs, False)
