# Copyright (c) 2007-2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""Explicit fitted-density ALDA kernel with literal native shell screening."""
from dataclasses import dataclass

import numpy as np

from psi4 import core
from .isapol_native_propagator import KernelSmoothing, PROPAGATOR_WORK_LIMITS, _superfunctional


@dataclass(frozen=True)
class ScreenedAuxiliaryKernel:
    """Owned kernel and diagnostics.

    ``capped_rows`` counts raw |fxc| strictly exceeding F-Max, not every row
    changed by smoothing (FD can affect values below it; ZERO affects equality).
    """
    matrix: np.ndarray
    grid_rows: int
    floored_rows: int
    capped_rows: int
    retained_shell_pairs: int
    planned_bytes: int


def screened_auxiliary_kernel(auxiliary, density_coefficients, grid, *,
                              smoothing, cutoff, kernel="alda_slater_pw92",
                              block_rows=512, max_bytes=512*1024**2):
    """Integrate chi_P * fxc(chi@density) * chi_Q on the complete given grid.

    Density coefficients are explicit AUX operands, not an inferred Drho-C fit.
    Use the native plain-OO fit for the traced plain-density CamCASP protocol.
    Every row is retained, including negative/low density (floored according
    to the declared smoothing model). Weights are not renormalized.

    Skip shell pairs iff abs(native signed s-surrogate) < cutoff. Retained
    upper shell blocks are integrated directly; lower entries copy the upper
    triangle as part of construction, not a repair of a response matrix.
    No dense OV operator or complete grid-by-AUX tensor is formed.

    Sampling and retained-block multiplication obey the EXISTING propagator
    work ceilings, applied to total work, never reset per grid block.
    This bounded implementation does not promise admission of the full benzene
    quadrature. Its explicit numeric ledger excludes library/basis overhead
    and other caller-retained factors/fits; subtract those from a shared budget.
    """
    if (not isinstance(auxiliary, core.IsaExplicitBasis)
            or auxiliary.role != core.IsaBasisRole.MolecularAux):
        raise ValueError("explicit MolecularAux basis required")
    if not isinstance(smoothing, KernelSmoothing):
        raise ValueError("explicit KernelSmoothing required")
    if kernel not in ("alda_slater", "alda_slater_pw92", "alda_slater_vwn"):
        raise ValueError("unsupported ALDA kernel")
    if not np.isscalar(cutoff) or not np.isfinite(cutoff) or cutoff < 0:
        raise ValueError("cutoff must be finite and nonnegative")
    if type(block_rows) is not int or not 1 <= block_rows <= 512:
        raise ValueError("block_rows must be an integer in [1,512]")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    naux = auxiliary.nfunction
    if (not isinstance(density_coefficients, np.ndarray)
            or density_coefficients.dtype != np.float64
            or density_coefficients.shape != (naux,)):
        raise ValueError("density_coefficients must be float64 (naux,)")
    if (not isinstance(grid, np.ndarray) or grid.dtype != np.float64
            or grid.ndim != 2 or grid.shape[1] != 4 or not len(grid)):
        raise ValueError("grid must be nonempty float64 (rows,4), bohr and weights")
    rows = len(grid)
    block_rows = min(block_rows, rows)
    planned = 8*(4*naux*naux + 6*block_rows*naux + 64*block_rows + 8*naux) + 2*grid.nbytes
    if planned > max_bytes:
        raise ValueError("auxiliary kernel byte resource limit exceeded")
    if rows*naux > PROPAGATOR_WORK_LIMITS["kernel_sampling"]:
        raise ValueError("kernel_sampling work resource limit")
    if not np.isfinite(density_coefficients).all() or not np.isfinite(grid).all():
        raise ValueError("finite density coefficients and grid required")
    layout = auxiliary.shell_layout()
    screen = np.asarray(auxiliary.screening_s_overlap(max_bytes=naux*naux*8))
    retained = np.abs(screen) >= cutoff
    work_per_row = sum(left[1]*right[1] for i, left in enumerate(layout)
                       for j, right in enumerate(layout) if j >= i and retained[i, j])
    if rows*work_per_row > PROPAGATOR_WORK_LIMITS["auxiliary_metric_kernel"]:
        raise ValueError("auxiliary_metric_kernel work resource limit")
    pairs = sum(int(retained[i, j]) for i in range(len(layout)) for j in range(i, len(layout)))
    del screen
    functional = _superfunctional(kernel, smoothing.rho_epsilon, block_rows)
    matrix = np.zeros((naux, naux))
    floored = capped = 0
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for start in range(0, rows, block_rows):
            stop = min(rows, start+block_rows)
            chi = np.asarray(auxiliary.evaluate(grid[start:stop, :3].tolist()))
            density = chi @ density_coefficients
            if not np.isfinite(density).all():
                raise ValueError("nonfinite fitted kernel density")
            floored += int(np.count_nonzero(density < smoothing.rho_epsilon))
            values = functional.compute_functional(
                {"RHO_A": core.Vector.from_array(np.maximum(density, smoothing.rho_epsilon))},
                stop-start, True)
            fxc = np.asarray(values["V_RHO_A_RHO_A"])[:stop-start].copy()
            if not np.isfinite(fxc).all():
                raise ValueError("nonfinite ALDA functional")
            capped += int(np.count_nonzero(np.abs(fxc) > smoothing.f_max))
            factor = grid[start:stop, 3]*smoothing.limit(fxc)
            for i, (offset, count, centre, angular) in enumerate(layout):
                chunks = [np.arange(right[0], right[0]+right[1])
                          for j, right in enumerate(layout) if j >= i and retained[i, j]]
                if not chunks:
                    continue
                columns = np.concatenate(chunks)
                weighted = factor[:, None]*chi[:, columns]
                matrix[offset:offset+count, columns] += chi[:, offset:offset+count].T @ weighted
                del chunks, columns, weighted
            del chi
    for row in range(naux):
        matrix[row+1:, row] = matrix[row, row+1:]
    if not np.isfinite(matrix).all():
        raise ValueError("nonfinite auxiliary ALDA kernel")
    frozen = np.frombuffer(matrix.tobytes(), dtype=np.float64).reshape(matrix.shape)
    return ScreenedAuxiliaryKernel(frozen, rows, floored, capped, pairs, planned)
