# Copyright (c) 2007-2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""Bounded DF operator actions and projected solves, not a CamCASP preset.

The factors are caller-supplied scientific operands. In plain Coulomb DF,
``oo`` and ``ov`` are MO three-centre integrals and ``dual_ov``/``dual_vv``
are solved against the Coulomb metric. Keeping left/right factors separate
does not assume that other fitting models are a single whitened Gram matrix.
No metric inversion, constrained fitting, kernel construction, or native admission
is performed here. Existing exact-response backends and their limits remain
unchanged.
"""
from dataclasses import dataclass
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
    if type(max_bytes) is not int or not 0 < max_bytes <= 512*1024**2:
        raise ValueError("max_bytes must be a positive integer at most 512 MiB")
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
    explicit action temporaries under at most 512 MiB. Caller storage and
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
        if type(max_bytes) is not int or not 0 < max_bytes <= self.MAX_BYTES:
            raise ValueError("max_bytes must be a positive integer at most 512 MiB")
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


class KernelCorrectedDFOperators(FactorizedDFOperators):
    """Compose plain DF with an explicitly supplied auxiliary-space action.

    H1 gains ``4*local_scale*D0 @ action(D0.T @ rhs)``; H2 and the
    shared plain factors are unchanged. Kernel legs are frozen independently
    of the target/anchor projection legs supplied to the frequency solver.
    No kernel-density, screening, fitting or CamCASP policy is inferred.

    The callable must implement a fixed linear operator, return an owned
    float64 array with the input shape, and not mutate its input. Ownership
    and nonmutation are trusted obligations, not alias analysis. Its retained
    numeric storage, peak workspace (including returned arrays) and per-RHS
    work must be declared conservatively. These are trusted producer
    contracts, not introspection or a sandbox for arbitrary Python code.
    Keep the callable and its underlying operands unchanged for the lifetime
    of this wrapper. This interface does not admit any native kernel producer
    or relax its sampling, I/O or lifetime-work gates.

    Declared callback workspace must cover every admitted RHS block width
    (1 through 64), including output storage; it is not a per-RHS declaration.
    Callback workspace is reserved throughout the solve, in addition to
    ordinary action/solver buffers. External retained fits/state must be
    deducted from max_bytes by the orchestrating consumer. Factor storage
    is shared, not copied; implementation-specific runtime overhead is excluded.
    """

    def __init__(self, plain, kernel_legs, kernel_action, *, local_scale,
                 kernel_storage_bytes, kernel_workspace_bytes, kernel_work_per_rhs,
                 max_bytes=FactorizedDFOperators.MAX_BYTES):
        if (not isinstance(plain, FactorizedDFOperators)
                or isinstance(plain, KernelCorrectedDFOperators) or plain._nkernel):
            raise ValueError("plain factorized operators without a kernel required")
        if type(max_bytes) is not int or not 0 < max_bytes <= self.MAX_BYTES:
            raise ValueError("max_bytes must be a positive integer at most 512 MiB")
        for value in (kernel_storage_bytes, kernel_workspace_bytes, kernel_work_per_rhs):
            if type(value) is not int or value < 0:
                raise ValueError("kernel resource declarations must be nonnegative integers")
        if not callable(kernel_action):
            raise ValueError("kernel_action must be callable")
        if (not isinstance(local_scale, Real) or not np.isfinite(local_scale)
                or not 0 <= local_scale <= 1):
            raise ValueError("local_scale must be finite in [0,1]")
        if (not isinstance(kernel_legs, np.ndarray) or kernel_legs.dtype != np.float64
                or kernel_legs.ndim != 2 or kernel_legs.shape[0] != plain.nov
                or kernel_legs.shape[1] < 1):
            raise ValueError("kernel legs must be float64 (nov,nkernel)")
        limit = min(max_bytes, plain._max_bytes)
        storage = (plain._storage + kernel_legs.nbytes +
                   kernel_storage_bytes + kernel_workspace_bytes)
        if storage + 2*kernel_legs.nbytes > limit:
            raise ValueError("kernel composition byte resource limit exceeded")
        if not np.isfinite(kernel_legs).all():
            raise ValueError("kernel legs must be finite")
        self._legs = np.frombuffer(kernel_legs.tobytes(), dtype=np.float64).reshape(kernel_legs.shape)
        self._plain, self._kernel_action = plain, kernel_action
        self._kernel_work = kernel_work_per_rhs
        self._local_scale, self._nkernel = float(local_scale), kernel_legs.shape[1]
        self._storage, self._max_bytes = storage, limit
        self.naux, self.nocc, self.nvir, self.nov = plain.naux, plain.nocc, plain.nvir, plain.nov
        self._gaps = plain._gaps

    def _apply(self, rhs, first):
        if (not isinstance(rhs, np.ndarray) or rhs.dtype != np.float64
                or rhs.ndim != 2 or rhs.shape[0] != self.nov
                or not 1 <= rhs.shape[1] <= 64):
            raise ValueError("rhs must be float64 (nov,nrhs), 1 <= nrhs <= 64")
        nrhs = rhs.shape[1]
        space = 8*(4*self.nov*nrhs + 16*self.nov + 4*self.nocc**2 +
                   4*self.nvir**2 + 4*self._nkernel*nrhs)
        if self._storage + space > self._max_bytes:
            raise ValueError("action byte resource limit exceeded")
        work = (2*self.nov + self.naux*(12*self.nov +
                6*self.nocc**2*self.nvir + 2*self.nocc*self.nvir**2))*nrhs
        if first and self._local_scale:
            work += nrhs*(4*self.nov*self._nkernel + self._kernel_work + 2*self.nov)
        if work > self.MAX_WORK:
            raise ValueError("factorized action work resource limit exceeded")
        result = self._plain.apply_h1(rhs) if first else self._plain.apply_h2(rhs)
        if first and self._local_scale:
            with np.errstate(over="raise", invalid="raise"):
                projected = self._legs.T @ rhs
                if not np.isfinite(projected).all():
                    raise ValueError("nonfinite projected kernel input")
                projected.setflags(write=False)
                value = self._kernel_action(projected)
                if (not isinstance(value, np.ndarray) or value.dtype != np.float64
                        or value.shape != projected.shape or not np.isfinite(value).all()):
                    raise ValueError("kernel action must return finite float64 (nkernel,nrhs)")
                result += (4*self._local_scale)*(self._legs @ value)
        if not np.isfinite(result).all():
            raise ValueError("nonfinite factorized response action")
        return result


@dataclass(frozen=True)
class FactorizedFrequencyResponse:
    """Projected response and true residual evidence, not a stability certificate."""
    response: np.ndarray
    relative_residuals: tuple
    operator_actions: int
    frequency: float
    static_relative_residuals: tuple = ()


class FactorizedResponseBudget:
    """Shared H1/H2 call ledger for a sequence of projected solves.

    Pass the same instance to every frequency/leg solve in a campaign.
    Calls are charged before dispatch, including RHS construction and true
    residual checks; failed calls consume their charge and are never refunded.
    This single-threaded ledger has no reset operation. It bounds operator
    calls, not kernel sampling, disk I/O, wall time or process RSS. Those
    resources require separate ownership and admission by the producer.
    """

    def __init__(self, *, max_actions=1000):
        if type(max_actions) is not int or not 1 <= max_actions <= 10000:
            raise ValueError("max_actions must be an integer in [1,10000]")
        self._limit = max_actions
        self._used = 0

    @property
    def operator_actions(self):
        """Total charged calls, including any failed operator dispatch."""
        return self._used

    def _charge(self):
        if self._used >= self._limit:
            raise RuntimeError("shared frequency operator-action budget exhausted")
        self._used += 1


def factorized_projected_response(operators, legs, frequency, *, restart=20,
                                 max_actions=1000, max_bytes=512*1024**2, budget=None,
                                 static_h1=False):
    """Solve (H2 H1 + omega² I) X = -4 H2 D and return D.T X.

    Restarted GMRES consumes one RHS at a time with a positive-gap diagonal
    preconditioner. No dense OV matrices, pseudoinverse, symmetrization or
    fallback solve. A true relative equation residual <=1e-10 is mandatory
    for every nonzero RHS. Zero RHS is solved exactly. The fixed internal
    GMRES tolerance is 1e-12; neither tolerance is caller-adjustable.

    ``max_actions`` bounds all H1/H2 calls across the whole frequency,
    including RHS construction and final residual checks. Optional ``budget``
    is a shared ``FactorizedResponseBudget`` imposing an additional cumulative
    limit across frequency/leg calls, without resetting on failure. The result's
    action count remains local to this solve. Memory admission
    includes owned operator factors, action buffers, leg snapshots, projected
    output, and conservative Krylov/Hessenberg workspaces. Caller storage and
    implementation-specific BLAS/SciPy overhead are not an RSS guarantee.
    This solves the supplied equation; it does not certify electronic
    stability or implement the missing constrained-CamCASP propagator.

    Explicit ``static_h1=True`` is allowed only at exactly zero frequency.
    It solves H1 X = -4 D, preconditioned by the positive gaps, and certifies
    both this equation and H2 H1 X = -4 H2 D. No H2 inverse is assumed.
    For singular H2 this is a stronger equation and may select a different
    solution or refuse where the default equation is consistent. No fallback
    occurs. An exactly zero original RHS requires an exactly zero original
    defect (relative error is otherwise undefined). Zero D uses no calls on
    this opt-in path. All other calls, including both checks, share the same
    budgets. The returned static residual tuple is empty for the default path.
    """
    from scipy.sparse.linalg import LinearOperator, gmres

    if not isinstance(operators, FactorizedDFOperators):
        raise ValueError("explicit factorized operators required")
    if (not isinstance(legs, np.ndarray) or legs.dtype != np.float64
            or legs.ndim != 2 or legs.shape[0] != operators.nov
            or legs.shape[1] < 1):
        raise ValueError("legs must be a nonempty float64 (nov,nprojected) array")
    if type(restart) is not int or not 1 <= restart <= 64:
        raise ValueError("restart must be an integer in [1,64]")
    if type(max_actions) is not int or not 1 <= max_actions <= 10000:
        raise ValueError("max_actions must be an integer in [1,10000]")
    if budget is not None and not isinstance(budget, FactorizedResponseBudget):
        raise ValueError("budget must be a FactorizedResponseBudget")
    if type(max_bytes) is not int or not 0 < max_bytes <= operators.MAX_BYTES:
        raise ValueError("solver max_bytes must be positive and at most 512 MiB")
    if (not isinstance(frequency, Real) or not np.isfinite(frequency) or frequency < 0):
        raise ValueError("frequency must be finite and nonnegative")
    if type(static_h1) is not bool or (static_h1 and frequency != 0):
        raise ValueError("static_h1 must be bool and requires exactly zero frequency")
    n, projected = legs.shape
    restart = min(restart, n)
    action_space = 8*(20*n + 4*operators.nocc**2 + 4*operators.nvir**2 +
                      4*operators._nkernel)
    needed = (operators._storage + action_space +
              8*(2*n*projected + 2*projected**2 +
                 n*(restart+32+(8 if static_h1 else 0)) + 4*(restart+1)**2))
    if needed > min(max_bytes, operators._max_bytes):
        raise ValueError("frequency solver byte resource limit exceeded")
    if not np.isfinite(legs).all():
        raise ValueError("legs must be finite")
    d = np.frombuffer(legs.tobytes(order="C"), dtype=np.float64).reshape(legs.shape)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        omega2 = np.float64(frequency)*np.float64(frequency)
        diagonal = operators._gaps if static_h1 else operators._gaps*operators._gaps + omega2
        inverse = 1./diagonal
    if not np.isfinite(inverse).all():
        raise ValueError("nonfinite gap preconditioner")
    actions = 0

    def action(vector, first):
        nonlocal actions
        if actions >= max_actions:
            raise RuntimeError("frequency solver operator-action budget exhausted")
        if budget is not None:
            budget._charge()
        actions += 1
        method = operators.apply_h1 if first else operators.apply_h2
        return method(np.asarray(vector).reshape(n, 1))[:, 0]

    def multiply(vector):
        if static_h1:
            return action(vector, True)
        with np.errstate(over="raise", invalid="raise"):
            value = action(action(vector, True), False) + omega2*vector
        if not np.isfinite(value).all():
            raise ValueError("nonfinite frequency operator action")
        return value

    matrix = LinearOperator((n, n), matvec=multiply, dtype=np.float64)
    preconditioner = LinearOperator((n, n), matvec=lambda x: inverse*x, dtype=np.float64)
    response = np.empty((projected, projected))
    residuals = []
    static_residuals = []
    for column in range(projected):
        if static_h1:
            with np.errstate(over="raise", invalid="raise"):
                rhs = -4*d[:, column]
            if not np.isfinite(rhs).all():
                raise ValueError("nonfinite static RHS")
            if not np.any(rhs):
                response[:, column] = 0.
                residuals.append(0.)
                static_residuals.append(0.)
                continue
            scale = np.max(np.abs(rhs))
            normalized = rhs/scale
            original_rhs = action(normalized, False)
            solution, info = gmres(matrix, normalized, M=preconditioner,
                                   restart=restart, maxiter=max_actions, rtol=1e-12, atol=0.)
            h1_solution = action(solution, True)
            original_lhs = action(h1_solution, False)
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                static_relative = float(np.linalg.norm(h1_solution-normalized) /
                                        np.linalg.norm(normalized))
                original_defect = original_lhs-original_rhs
                if np.any(original_rhs):
                    reference_scale = np.max(np.abs(original_rhs))
                    relative = float(np.linalg.norm(original_defect/reference_scale) /
                                     np.linalg.norm(original_rhs/reference_scale))
                else:
                    relative = 0. if not np.any(original_defect) else np.inf
            if (info != 0 or not np.isfinite(static_relative) or static_relative > 1e-10
                    or not np.isfinite(relative) or relative > 1e-10):
                raise RuntimeError("static solver failed dual residual/convergence gate")
            with np.errstate(over="raise", invalid="raise"):
                response[:, column] = (d.T @ solution)*scale
            residuals.append(relative)
            static_residuals.append(static_relative)
            continue
        with np.errstate(over="raise", invalid="raise"):
            rhs = -4*action(d[:, column], False)
        if not np.isfinite(rhs).all():
            raise ValueError("nonfinite frequency RHS")
        if not np.any(rhs):
            response[:, column] = 0.
            residuals.append(0.)
            continue
        # Scale before norm evaluation to avoid overflow in sum-of-squares.
        scale = np.max(np.abs(rhs))
        normalized = rhs/scale
        solution, info = gmres(matrix, normalized, M=preconditioner,
                               restart=restart, maxiter=max_actions, rtol=1e-12, atol=0.)
        defect = multiply(solution)-normalized
        relative = float(np.linalg.norm(defect)/np.linalg.norm(normalized))
        if info != 0 or not np.isfinite(relative) or relative > 1e-10:
            raise RuntimeError("frequency solver failed true residual/convergence gate")
        with np.errstate(over="raise", invalid="raise"):
            response[:, column] = (d.T @ solution)*scale
        residuals.append(relative)
    if not np.isfinite(response).all():
        raise ValueError("nonfinite projected response")
    snapshot = np.frombuffer(response.tobytes(order="C"), dtype=np.float64).reshape(response.shape)
    return FactorizedFrequencyResponse(snapshot, tuple(residuals), actions, float(frequency),
                                       tuple(static_residuals))
