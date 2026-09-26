# Psi4: Copyright (c) 2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""The declared ALDA kernel smoothing and the kernel's LibXC functional.

CamCASP declares the kernel floor and cap in its own input:

    SET NEW-PROP ... KERNEL-INTEGRAL-PARAMETERS (INFINITY-CONTROL-METHOD FD,
        RHO-EPS=1e-8, F-MAX=1000.0, FD-DELTA=0.01, FD-ALPHA=1.0)

:class:`KernelSmoothing` is that declaration; it is a model, not a tolerance.
"""
from dataclasses import dataclass

import numpy as np

from psi4 import core

#: Named ``Kernel_Smooth_Method`` values of CamCASP ``prop_parser.F90:127-160``.
SMOOTHING_METHODS = {'ZERO': 1, 'CONSTANT': 2, 'FD': 3}

#: Exponent above which ``1/(1+exp(z))`` is replaced by ``exp(-z)``; CamCASP
#: ``dft_Sx_PW92c.F90`` ``FD_z``. Part of the declared functional form.
FD_EXPONENT_CUTOFF = 40.0


@dataclass(frozen=True)
class KernelSmoothing:
    """CamCASP's declared ALDA kernel floor and cap: a model, not a tolerance.

    ``rho_epsilon`` is ``Kernel_Rho_Epsilon``: the density handed to the
    functional is raised to it (``rs_and_fix_n_simple``), and no row is dropped
    for being below it. ``f_max``/``fd_delta``/``fd_alpha``/``method`` are
    ``Kernel_F_Max``, ``Kernel_Smooth_FD_Delta``, ``Kernel_Smooth_FD_Alpha`` and
    ``Kernel_Smooth_Method`` of ``subroutine limit``:

        ZERO      |Q| >= Qmax          -> 0
        CONSTANT  Q clamped to +-Qmax
        FD        z = (|Q|/Qmax - 1)/Delta,  FD = (1/(1+exp z))**Alpha
                  Q <= Qmax ? Q*FD : Qmax*FD

    Changing any field names a different model. There is no default instance and
    no inference from the functional, the grid or the basis.
    """
    rho_epsilon: float
    f_max: float
    fd_delta: float
    fd_alpha: float
    method: str

    def __post_init__(self):
        for name in ('rho_epsilon', 'f_max', 'fd_delta', 'fd_alpha'):
            value = getattr(self, name)
            if type(value) is not float or not np.isfinite(value) or value <= 0.:
                raise ValueError(f'KernelSmoothing.{name} must be an explicit finite positive float')
        if self.method not in SMOOTHING_METHODS:
            raise ValueError('KernelSmoothing.method must be an explicit ZERO, CONSTANT or FD')

    @property
    def method_code(self):
        """The reference's integer ``Kernel_Smooth_Method``, for the record only."""
        return SMOOTHING_METHODS[self.method]

    @property
    def declaration(self):
        return (f'RHO-EPS={self.rho_epsilon!r}; F-MAX={self.f_max!r}; '
                f'FD-DELTA={self.fd_delta!r}; FD-ALPHA={self.fd_alpha!r}; '
                f'METHOD {self.method}')

    def floor(self, density):
        """Raise the density to ``rho_epsilon``; never renormalize the grid."""
        return np.maximum(np.asarray(density, dtype=float), self.rho_epsilon)

    def limit(self, values):
        """Apply ``subroutine limit`` elementwise to a real kernel array."""
        q = np.asarray(values, dtype=float)
        if self.method == 'ZERO':
            return np.where(np.abs(q) >= self.f_max, 0., q)
        if self.method == 'CONSTANT':
            return np.clip(q, -self.f_max, self.f_max)
        z = (np.abs(q)/self.f_max - 1.)/self.fd_delta
        # Both branches of the reference's FD_z split, evaluated without overflow.
        low = (1./(1. + np.exp(np.minimum(z, FD_EXPONENT_CUTOFF))))**self.fd_alpha
        high = np.exp(-self.fd_alpha*np.maximum(z, FD_EXPONENT_CUTOFF))
        fd = np.where(z <= FD_EXPONENT_CUTOFF, low, high)
        return np.where(q <= self.f_max, q*fd, self.f_max*fd)


def _superfunctional(kernel, density_cutoff, block_rows):
    """Blank Slater (+ optional LDA correlation) functional for the ALDA kernel.

    Each component's own density cutoff sits just below half the declared
    floor, so a floored unpolarized density is never cut by LibXC.
    """
    f = core.SuperFunctional.blank()
    f.set_density_tolerance(float(density_cutoff))
    x = core.LibXCFunctional('XC_LDA_X', True)
    x.set_alpha(1.)
    x.set_density_cutoff(float(np.nextafter(.5*density_cutoff, 0.)))
    f.add_x_functional(x)
    name = {'alda_slater_pw92': 'XC_LDA_C_PW', 'alda_slater_vwn': 'XC_LDA_C_VWN'}.get(kernel)
    if name is not None:
        c = core.LibXCFunctional(name, True)
        c.set_alpha(1.)
        c.set_density_cutoff(float(np.nextafter(.5*density_cutoff, 0.)))
        f.add_c_functional(c)
    f.set_max_points(int(block_rows))
    f.set_deriv(2)
    f.allocate()
    return f
