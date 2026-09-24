# Copyright (c) 2007-2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""Shell-streamed plain Coulomb DF operands, not a CamCASP response preset."""
import warnings
from dataclasses import dataclass

import numpy as np

from .isapol_factorized_response import FactorizedDFOperators, mo_three_center_shell


def _freeze(array):
    return np.frombuffer(array.tobytes(order="C"), dtype=np.float64).reshape(array.shape)


class _ColumnTiles:
    """Private immutable column tiles with bounded per-AUX gathering."""

    def __init__(self, tiles, nrow, ncol):
        self.tiles = tuple(tiles)
        self.nrow, self.ncol = nrow, ncol

    def __getitem__(self, auxiliary):
        result = np.empty(self.nrow*self.ncol)
        offset = 0
        for tile in self.tiles:
            width = tile.shape[1]
            result[offset:offset+width] = tile[auxiliary]
            offset += width
        return result.reshape(self.nrow, self.ncol)


def native_plain_df_operators(auxiliary, main, coefficients, energies, *,
                              nocc, shell_count, exact_exchange,
                              tile_columns=708, max_bytes=512*1024**2):
    """Build independent, owned plain-DF H1/H2 actions from native integrals.

    Explicit coefficients are in MAIN order, with occupied columns first.
    This low-level operation does not certify SCF convergence, orthonormality,
    orbital correction, constrained fitting, or CamCASP kernel equivalence.
    No local kernel is included. ``plain_density_coefficients`` is an immutable
    AUX vector solving J*c = 2*sum_i B[:,i,i], with no charge penalty, damping
    or density repair; it is not Drho-C. ``shell_count`` declares the complete AUX
    shell sequence; missing or excess shells fail closed.

    Raw VV storage comprises independently allocated column tiles. Each tile
    is solved with the unchanged Coulomb metric, checked against the original
    equation, frozen into immutable bytes, and released before the next solve.
    There is never a complete AO tensor or a second complete VV tensor.
    The existing arbitrary-array operator constructor is not weakened.

    ``max_bytes`` bounds explicitly planned numeric storage, including metric
    copies, factor payloads, freeze overlaps, validation/solve workspaces and
    the native shell transform. Basis descriptors, caller-owned state and
    implementation-specific BLAS/Libint overhead are excluded. A higher-level
    consumer must subtract other live allocations from its common budget;
    independent per-object budgets are not a whole-pipeline guarantee.
    """
    from psi4 import core
    from scipy.linalg import LinAlgWarning, lu_factor, lu_solve

    if (not isinstance(auxiliary, core.IsaExplicitBasis)
            or auxiliary.role != core.IsaBasisRole.MolecularAux
            or not isinstance(main, core.IsaExplicitBasis)
            or main.role != core.IsaBasisRole.Orbital):
        raise ValueError("explicit MolecularAux and Orbital bases required")
    if type(max_bytes) is not int or not 0 < max_bytes <= FactorizedDFOperators.MAX_BYTES:
        raise ValueError("max_bytes must be positive and at most 512 MiB")
    if type(tile_columns) is not int or not 1 <= tile_columns <= 708:
        raise ValueError("tile_columns must be an integer in [1,708]")
    nao, naux = main.nfunction, auxiliary.nfunction
    if type(shell_count) is not int or not 1 <= shell_count <= naux:
        raise ValueError("shell_count must be a positive integer at most naux")
    if (not isinstance(coefficients, np.ndarray) or coefficients.dtype != np.float64
            or coefficients.ndim != 2 or coefficients.shape[0] != nao):
        raise ValueError("coefficients must be float64 (nmain,nmo)")
    nmo = coefficients.shape[1]
    if type(nocc) is not int or not 0 < nocc < nmo:
        raise ValueError("nocc must leave nonempty occupied and virtual spaces")
    if (not isinstance(energies, np.ndarray) or energies.dtype != np.float64
            or energies.shape != (nmo,)):
        raise ValueError("energies must be float64 (nmo,)")
    if (not np.isscalar(exact_exchange) or not np.isfinite(exact_exchange)
            or not 0 <= exact_exchange <= 1):
        raise ValueError("exact_exchange must be finite in [0,1]")
    nv, no = nmo-nocc, nocc
    nov = no*nv
    width = min(tile_columns, max(nv*nv, nov))
    storage = 8*(naux*(no*no+2*nov+nv*nv+1)+nov)
    transform = 8*(24*nao**2+18*nmo**2+4*nao*nmo)
    # Eight tile buffers cover layout conversion, solved RHS, residuals and
    # freezing overlap even when LAPACK declines an overwrite request.
    # OO/OV can freeze whole: their largest duplication is explicitly charged.
    planned = (storage + 24*naux*naux + 64*naux*width
               + 8*naux*max(no*no, nov) + transform + 32*(nov+nmo+naux))
    if planned > max_bytes:
        raise ValueError("native factor construction byte resource limit exceeded")
    if (not np.isfinite(coefficients).all() or not np.isfinite(energies).all()):
        raise ValueError("finite coefficients and energies required")
    with np.errstate(over="raise", invalid="raise"):
        gaps = (energies[no:, None]-energies[None, :no]).ravel()
    if not np.isfinite(gaps).all() or np.any(gaps <= 0):
        raise ValueError("positive finite occupied-virtual gaps required")

    provider = core.IsaAuxCoulomb(auxiliary)
    oo = np.empty((naux, no, no))
    ov = np.empty((naux, no, nv))
    vv = [np.empty((naux, min(tile_columns, nv*nv-start)))
          for start in range(0, nv*nv, tile_columns)]
    offset = 0
    for shell in range(shell_count):
        block = mo_three_center_shell(provider, main, coefficients, shell,
                                       max_bytes=transform)
        stop = offset+len(block)
        if stop > naux:
            raise ValueError("AUX shell coverage exceeds naux")
        oo[offset:stop] = block[:, :no, :no]
        ov[offset:stop] = block[:, :no, no:]
        # Process a single AUX row to avoid packing a full noncontiguous
        # shell's VV slice while the complete MO shell remains live.
        for p in range(len(block)):
            row = block[p, no:, no:].reshape(-1)
            column = 0
            for tile in vv:
                tile[offset+p] = row[column:column+tile.shape[1]]
                column += tile.shape[1]
            del row
        offset = stop
        del block
    if offset != naux:
        raise ValueError("incomplete AUX shell coverage")
    del tile  # Do not retain the last mutable raw VV tile during replacement.

    metric = np.asarray(provider.metric())
    if not np.isfinite(metric).all():
        raise ValueError("nonfinite Coulomb metric")
    with warnings.catch_warnings():
        warnings.simplefilter("error", LinAlgWarning)
        lu, pivots = lu_factor(metric, overwrite_a=False, check_finite=False)
    if not np.isfinite(lu).all():
        raise ValueError("nonfinite Coulomb metric factorization")
    metric_norm = np.linalg.norm(metric, ord=np.inf)

    def solve(rhs):
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            solution = lu_solve((lu, pivots), rhs, check_finite=False)
            residual = np.linalg.norm(metric @ solution-rhs, ord=np.inf)
            scale = metric_norm*np.linalg.norm(solution, ord=np.inf) + np.linalg.norm(rhs, ord=np.inf)
        if (not np.isfinite(solution).all() or not np.isfinite(residual)
                or not np.isfinite(scale) or residual > 1.e-10*scale):
            raise ValueError("plain DF metric solve residual failed")
        return _freeze(solution)

    for i in range(len(vv)):
        # Replacing an independent allocation releases the original tile,
        # unlike a view into a retained full raw-VV backing allocation.
        vv[i] = solve(vv[i])
    dual_ov = []
    ov_flat = ov.reshape(naux, nov)
    for start in range(0, nov, tile_columns):
        dual_ov.append(solve(ov_flat[:, start:start+tile_columns]))
    # Linearity avoids a complete additional solved OO tensor. Temporary
    # trace/RHS/solution vectors are covered by the tile workspace allowance.
    density = solve((2*np.trace(oo, axis1=1, axis2=2))[:, None]).reshape(naux)
    del ov_flat, metric, lu, pivots, provider
    frozen_oo = _freeze(oo)
    del oo
    frozen_ov = _freeze(ov)
    del ov
    # Only internally generated immutable operands enter this private path.
    result = FactorizedDFOperators.__new__(FactorizedDFOperators)
    result._gaps = _freeze(gaps)
    result._oo, result._ov = frozen_oo, frozen_ov
    result._dual_ov = _ColumnTiles(dual_ov, no, nv)
    result._dual_vv = _ColumnTiles(vv, nv, nv)
    result._legs, result._kernel = None, None
    result.naux, result.nocc, result.nvir, result.nov = naux, no, nv, nov
    result._nkernel, result._local_scale = 0, 0.
    result._exchange = float(exact_exchange)
    result._max_bytes, result._storage = max_bytes, storage
    result.plain_density_coefficients = density
    result._native_auxiliary = auxiliary
    result.construction_planned_bytes = planned
    return result


@dataclass(frozen=True)
class NativeConstrainedOV:
    """Immutable occupied-fast OV coefficients; no response/kernel certificate."""
    coefficients: np.ndarray
    charge_penalty: float
    offsite_metric_damping: float
    relative_backward_residual: float
    planned_bytes: int


def native_constrained_ov(operators, *, charge_penalty, offsite_metric_damping,
                          tile_columns=64, max_bytes=512*1024**2):
    """Fit retained raw native OV integrals against the declared damped metric.

    A[P,Q] = (1-eta*(1-same_centre))*J[P,Q] + lambda*q[P]*q[Q].
    Zero target charge is the declared OV constraint, as in native ``fit_ov``.
    The returned matrix is (nov,naux), row a*nocc+i; the unchanged plain
    factors continue to serve the two-electron operators. This does not attach
    kernel legs or rebuild Hessians when fitting a different eta.

    The AUX and centre map come from the original native builder, never a
    separately supplied basis or site map. Tiled LU solves use the original
    equation's fixed backward-residual gate, with no regularization/fallback.
    Admission includes live operator storage, output/freezing overlap, metric/
    LU copies and tile work. Previously returned fits and other caller-retained
    objects must be subtracted from the shared budget by the consumer.
    """
    from psi4 import core
    from scipy.linalg import LinAlgWarning, lu_factor, lu_solve

    if (not isinstance(operators, FactorizedDFOperators)
            or not isinstance(getattr(operators, "_native_auxiliary", None), core.IsaExplicitBasis)):
        raise ValueError("native plain-DF operators required")
    if (not np.isscalar(charge_penalty) or not np.isfinite(charge_penalty)
            or charge_penalty < 0):
        raise ValueError("charge_penalty must be finite and nonnegative")
    if (not np.isscalar(offsite_metric_damping) or not np.isfinite(offsite_metric_damping)
            or not 0 <= offsite_metric_damping < 1):
        raise ValueError("offsite_metric_damping must be finite in [0,1)")
    if type(tile_columns) is not int or not 1 <= tile_columns <= 708:
        raise ValueError("tile_columns must be an integer in [1,708]")
    if type(max_bytes) is not int or not 0 < max_bytes <= FactorizedDFOperators.MAX_BYTES:
        raise ValueError("max_bytes must be positive and at most 512 MiB")
    naux, nov, no = operators.naux, operators.nov, operators.nocc
    width = min(tile_columns, nov)
    planned = (operators._storage + 16*naux*nov + 24*naux*naux
               + 64*naux*width + 64*naux + 32*nov)
    if planned > min(max_bytes, operators._max_bytes):
        raise ValueError("constrained OV byte resource limit exceeded")
    provider = core.IsaAuxCoulomb(operators._native_auxiliary)
    centres = np.empty(naux, dtype=np.int64)
    for offset, count, centre, angular in operators._native_auxiliary.shell_layout():
        centres[offset:offset+count] = centre
    charges = np.asarray(provider.charges())
    metric = np.asarray(provider.metric())
    with np.errstate(over="raise", invalid="raise"):
        for p in range(naux):
            metric[p] *= np.where(centres == centres[p], 1., 1.-offsite_metric_damping)
            metric[p] += (charge_penalty*charges[p])*charges
    if not np.isfinite(metric).all():
        raise ValueError("nonfinite constrained metric")
    with warnings.catch_warnings():
        warnings.simplefilter("error", LinAlgWarning)
        lu, pivots = lu_factor(metric, overwrite_a=False, check_finite=False)
    if not np.isfinite(lu).all():
        raise ValueError("nonfinite constrained metric factorization")
    metric_norm = np.linalg.norm(metric, ord=np.inf)
    coefficients = np.empty((nov, naux))
    worst = 0.
    for begin in range(0, nov, tile_columns):
        end = min(nov, begin+tile_columns)
        rhs = np.empty((naux, end-begin))
        for col, transition in enumerate(range(begin, end)):
            a, i = divmod(transition, no)
            rhs[:, col] = operators._ov[:, i, a]
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            solution = lu_solve((lu, pivots), rhs, check_finite=False)
            residual = np.linalg.norm(metric @ solution-rhs, ord=np.inf)
            scale = metric_norm*np.linalg.norm(solution, ord=np.inf) + np.linalg.norm(rhs, ord=np.inf)
        if (not np.isfinite(solution).all() or not np.isfinite(residual)
                or not np.isfinite(scale) or residual > 1.e-10*scale):
            raise ValueError("constrained OV solve residual failed")
        worst = max(worst, float(residual/scale) if scale else 0.)
        coefficients[begin:end] = solution.T
        del rhs, solution
    return NativeConstrainedOV(_freeze(coefficients), float(charge_penalty),
                               float(offsite_metric_damping), worst, planned)
