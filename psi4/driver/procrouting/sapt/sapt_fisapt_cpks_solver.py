"""Bounded matrix-free CPU PCG for the source-resolved monomer response."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ResponseSolution:
    amplitudes: object
    iterations: int
    absolute_residual: float
    relative_residual: float


def solve_positive(apply, gap, rhs, *, rtol=1e-10, atol=1e-12, max_iterations=128):
    """Solve H Y = W, return physical kappa = -Y, and check the true residual."""
    if (not callable(apply) or not isinstance(gap, np.ndarray) or not isinstance(rhs, np.ndarray)
            or gap.dtype != np.float64 or rhs.dtype != np.float64 or gap.shape != rhs.shape
            or gap.ndim != 2 or not gap.size or not np.isfinite(gap).all()
            or not np.isfinite(rhs).all() or np.any(gap <= 0)
            or any(type(x) not in (int, float) or not np.isfinite(x) or x < 0 for x in (rtol, atol))
            or max(rtol, atol) == 0 or type(max_iterations) is not int
            or not 1 <= max_iterations <= 4096):
        raise ValueError("Finite FP64 response/gaps and explicit positive solver policy required")
    y = np.zeros_like(rhs)
    residual = rhs.copy()
    norm = float(np.linalg.norm(rhs))
    threshold = max(atol, rtol * norm)

    def metrics(value):
        absolute = float(np.linalg.norm(value))
        if not np.isfinite(absolute):
            raise RuntimeError("Nonfinite CPU response residual")
        return absolute, absolute / norm if norm else absolute

    def action(value):
        product = np.asarray(apply(value))
        if product.dtype != np.float64 or product.shape != rhs.shape or not np.isfinite(product).all():
            raise RuntimeError("Invalid CPU Hessian action")
        return product

    absolute, relative = metrics(residual)
    if absolute <= threshold:
        return ResponseSolution(-y, 0, absolute, relative)
    z = residual / gap
    direction = z.copy()
    rho = float(np.vdot(residual, z).real)
    for iteration in range(1, max_iterations + 1):
        product = action(direction)
        curvature = float(np.vdot(direction, product).real)
        if not np.isfinite(curvature) or curvature <= 0 or not np.isfinite(rho) or rho <= 0:
            raise RuntimeError("CPU response has nonpositive/nonfinite curvature")
        alpha = rho / curvature
        y += alpha * direction
        residual -= alpha * product
        absolute, relative = metrics(residual)
        if absolute <= threshold:
            absolute, relative = metrics(rhs - action(y))
            if absolute > threshold:
                raise RuntimeError("CPU response true residual exceeds tolerance")
            return ResponseSolution(-y, iteration, absolute, relative)
        z = residual / gap
        next_rho = float(np.vdot(residual, z).real)
        if not np.isfinite(next_rho) or next_rho <= 0:
            raise RuntimeError("CPU response preconditioner failed")
        direction = z + (next_rho / rho) * direction
        rho = next_rho
    raise RuntimeError("CPU response did not converge within its iteration limit")
