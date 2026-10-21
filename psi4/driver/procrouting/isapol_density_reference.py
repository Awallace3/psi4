# Copyright (c) 2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""Explicit small high-accuracy finite-density diagnostic, never a default."""
from dataclasses import dataclass
from decimal import (Context, Decimal, DecimalException, DivisionByZero,
                     InvalidOperation, Overflow, ROUND_HALF_EVEN, localcontext)
import hashlib
import math
from numbers import Real
import struct

import numpy as np


_CONTEXT = Context(prec=80, rounding=ROUND_HALF_EVEN, Emin=-999999, Emax=999999,
                   traps=[DivisionByZero, InvalidOperation, Overflow])
_CANCELLATION_FLOOR = Decimal("1e-40")
_RESIDUAL_LIMIT = Decimal("1e-50")


@dataclass(frozen=True)
class FiniteDensityReference:
    """Immutable rounded coefficients and immutable diagnostic/provenance data.

    Residuals/charge use the returned binary64 coefficients unless prefixed
    ``reference_``. Scaled penalized residual uses the documented upper-bound
    scale, not an exact penalized-matrix backward-error denominator.
    """
    coefficients: np.ndarray
    charge_error: Decimal
    original_residual_inf: Decimal
    penalized_residual_inf: Decimal
    scaled_penalized_residual: Decimal
    reference_charge_error: Decimal
    reference_scaled_penalized_residual: Decimal
    base_scaled_residual: Decimal
    denominator: Decimal
    denominator_cancellation_ratio: Decimal
    input_sha256: str
    planned_buffer_bytes: int
    estimated_scalar_operations: int
    decimal_digits: int = 80
    factorizations: int = 1
    right_hand_sides: int = 2
    algorithm: str = "fixed80 Decimal partial-pivot J LU; two RHS; finite rank-one update"


def _dot(left, right):
    return sum((x*y for x, y in zip(left, right)), Decimal(0))


def _solve_two(matrix, rhs_vectors):
    # Extracted privately from the reviewed fixed-input Decimal audit. Swap
    # whole rows, including earlier multipliers, and permute both RHS equally.
    n = len(matrix)
    lu = [[+v for v in row] for row in matrix]
    permutation = list(range(n))
    for k in range(n):
        pivot = max(range(k, n), key=lambda i: abs(lu[i][k]))
        if lu[pivot][k] == 0:
            raise np.linalg.LinAlgError("nonsingular J required: zero Decimal LU pivot")
        if pivot != k:
            lu[pivot], lu[k] = lu[k], lu[pivot]
            permutation[pivot], permutation[k] = permutation[k], permutation[pivot]
        pivot_row = lu[k]
        for i in range(k+1, n):
            row = lu[i]
            factor = row[k]/pivot_row[k]
            row[k] = factor
            for j in range(k+1, n):
                row[j] -= factor*pivot_row[j]
    answers = []
    for rhs in rhs_vectors:
        y = [+rhs[i] for i in permutation]
        for i in range(n):
            y[i] -= _dot(lu[i][:i], y[:i])
        for i in range(n-1, -1, -1):
            y[i] = (y[i]-_dot(lu[i][i+1:], y[i+1:]))/lu[i][i]
        answers.append(y)
    return answers


def _nonnegative_scalar(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative binary64 real")
    try:
        value = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite nonnegative binary64 real") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative binary64 real")
    return value


def finite_density_reference(metric, density_rhs, charges, *, electron_count, charge_penalty):
    """Diagnose (J+lambda*q*q.T)c=b+lambda*N*q from supplied binary64 arrays.

    Opt-in only: no Psi4 objects, integrals, fields or caller/default selection.
    J must be nonsingular, even if the penalized matrix alone is nonsingular.
    J/b/q must be finite float64 NumPy arrays of shapes (n,n)/(n,)/(n,),
    1<=n<=256. N/lambda are finite nonnegative binary64 reals, not booleans.
    Asymmetric entries are retained. One fixed80-digit partial-pivot J LU,
    two RHS, accurate rank-one update; no penalized matrix, double fallback,
    symmetrization, shifts, normalization or hard-charge replacement.

    Before Decimal allocation, reserve 4*n*n*512+48*n*512+2MiB bytes and
    2*n**3+64*n*n+1e6 scalar operations (max143MB/39M; below256MiB/500M).
    These explicit-buffer/work bounds exclude interpreter/vendor overhead.
    No user precision, budget or tolerance tuning is exposed.

    Refuse zero denominator; also refuse abs(den)/(1+lambda*sum(abs(q_i*u_i)))
    <=1e-40, conservatively reserving40 of80 digits against cancellation.
    Base/implicit-reference scaled residuals must be <=1e-50. These fixed
    sanity guards are NOT universal forward-accuracy/interval certificates.
    Rounded coefficients must remain finite; no clipping/charge repair.

    Intended residual is evaluated implicitly as Jc-b+lambda*q*(qTc-N).
    Its scale is ||J||inf*||c||inf+||b||inf+
    lambda*||q||inf*(||q||1*||c||inf+N), a declared upper-bound scale.
    Diagnostics distinguish the80-digit vector and returned rounded vector.
    ValueError: invalid inputs; MemoryError: finite profile exceeded;
    LinAlgError: singular/cancelling/nonfinite/overflow or sanity-guard failure.
    """
    if (not isinstance(metric, np.ndarray) or metric.dtype != np.float64
            or metric.ndim != 2 or not metric.shape[0] or metric.shape[0] != metric.shape[1]):
        raise ValueError("metric must be nonempty float64 square (n,n)")
    n = metric.shape[0]
    buffers = 4*n*n*512+48*n*512+2*1024**2
    operations = 2*n**3+64*n*n+1000000
    if n > 256 or buffers > 256*1024**2 or operations > 500000000:
        raise MemoryError("finite diagnostic profile exceeded (at most256 functions)")
    for name, vector in (("density_rhs", density_rhs), ("charges", charges)):
        if (not isinstance(vector, np.ndarray) or vector.dtype != np.float64
                or vector.shape != (n,) or not np.isfinite(vector).all()):
            raise ValueError(f"{name} must be finite float64 (n,)")
    if not np.isfinite(metric).all():
        raise ValueError("metric must be finite")
    target = _nonnegative_scalar("electron_count", electron_count)
    penalty = _nonnegative_scalar("charge_penalty", charge_penalty)
    fingerprint = hashlib.sha256(b"finite-density-reference/v1\0"+struct.pack(">Idd", n, target, penalty))
    for array in (metric, density_rhs, charges):
        fingerprint.update(array.astype("<f8", copy=False).tobytes(order="C"))
    try:
        with localcontext(_CONTEXT):
            J = [[Decimal.from_float(float(v)) for v in row] for row in metric]
            b, q = ([Decimal.from_float(float(v)) for v in row] for row in (density_rhs, charges))
            N, lam = Decimal.from_float(target), Decimal.from_float(penalty)
            plain, u = _solve_two(J, (b, q))
            jnorm = max(sum((abs(v) for v in row), Decimal(0)) for row in J)
            base = max(
                max(abs(_dot(row, c)-rhs) for row, rhs in zip(J, vector)) /
                (jnorm*max(abs(v) for v in c)+max(abs(v) for v in vector))
                if any(c) or any(vector) else Decimal(0)
                for c, vector in ((plain, b), (u, q)))
            denominator = Decimal(1)+lam*_dot(q, u)
            if denominator == 0:
                raise np.linalg.LinAlgError("singular finite-penalty denominator")
            ratio = abs(denominator)/(Decimal(1)+lam*sum((abs(x*y) for x, y in zip(q, u)), Decimal(0)))
            if ratio <= _CANCELLATION_FLOOR:
                raise np.linalg.LinAlgError("near-cancelling finite-penalty denominator")
            alpha = lam*(N-_dot(q, plain))/denominator
            high = [v+alpha*w for v, w in zip(plain, u)]

            def diagnostics(c):
                charge = _dot(q, c)-N
                residual = [_dot(row, c)-rhs for row, rhs in zip(J, b)]
                original = max(abs(v) for v in residual)
                intended = max(abs(v+lam*qi*charge) for v, qi in zip(residual, q))
                cnorm = max(abs(v) for v in c)
                scale = (jnorm*cnorm+max(abs(v) for v in b)
                         +lam*max(abs(v) for v in q)*(sum((abs(v) for v in q), Decimal(0))*cnorm+N))
                return charge, original, intended, intended/scale if scale else Decimal(0)

            high_diag = diagnostics(high)
            if not all(v.is_finite() for v in (*high, base, ratio, *high_diag)):
                raise np.linalg.LinAlgError("nonfinite fixed80 diagnostic")
            if max(base, high_diag[3]) > _RESIDUAL_LIMIT:
                raise np.linalg.LinAlgError("fixed80 residual sanity guard failed; no accuracy certificate")
            rounded = [float(v) for v in high]
            if not all(math.isfinite(v) for v in rounded):
                raise np.linalg.LinAlgError("binary64 coefficient overflow in fixed80 diagnostic")
            rounded_diag = diagnostics([Decimal.from_float(v) for v in rounded])
            if not all(v.is_finite() for v in rounded_diag):
                raise np.linalg.LinAlgError("nonfinite rounded-coefficient diagnostic")
            coefficients = np.frombuffer(np.array(rounded, dtype=np.float64).tobytes(), dtype=np.float64)
            return FiniteDensityReference(
                coefficients, *rounded_diag, high_diag[0], high_diag[3], base,
                denominator, ratio, fingerprint.hexdigest(), buffers, operations)
    except DecimalException as exc:
        raise np.linalg.LinAlgError("nonfinite/overflow in fixed80 Decimal arithmetic") from exc
