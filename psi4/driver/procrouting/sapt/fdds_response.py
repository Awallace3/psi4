#
# @BEGIN LICENSE
#
# Psi4: an open-source quantum chemistry software package
#
# Copyright (c) 2007-2026 The Psi4 Developers.
#
# The copyrights for code used from other parties are included in
# the corresponding files.
#
# This file is part of Psi4.
#
# Psi4 is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3.
#
# Psi4 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with Psi4; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# @END LICENSE
#

"""Shared single-system FDDS coupling and explicit-orbital construction.

The legacy numerical kernel accepts supplied auxiliary intermediates; its optional
monomer provider constructs Psi4 intermediates but never performs SCF. Those
responses are Coulomb-auxiliary, not CamCASP C_DF coefficient responses. The separate
FDDSFullOVResponse route accepts supplied operators and explicitly declared
transition-leg coordinates, including fitted-density coefficients.
No quadrature weight or dispersion prefactor is included.
"""

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np

from psi4 import core


@dataclass(frozen=True)
class FDDSFrequencyResponse:
    """Owned, independently mutable arrays; field bindings cannot be reassigned.

    Raw arrays retain reciprocity defects before the historical symmetrization.
    All arrays are independent of the caller's inputs.
    """

    uncoupled: np.ndarray
    coupled: np.ndarray
    raw_uncoupled: np.ndarray
    raw_coupled: np.ndarray
    representation: str = "fdds_coulomb_auxiliary"


def _square(value, name, shape=None, *, finite=True):
    value = np.asarray(value)
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    value = np.asarray(value, dtype=float)
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[0] != value.shape[1]:
        raise ValueError(f"{name} must be a nonempty square matrix")
    if shape is not None and value.shape != shape:
        raise ValueError(f"{name} has inconsistent dimensions")
    if finite and not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


def _symmetrize(mat):
    return 0.5 * (mat + mat.transpose())


def prepare_fdds_hybrid_transform(R):
    """Return R pseudoinverse-transpose under the existing SAPT policy.

    Preserve nan_to_num sanitation and rcond=1.e-13, without mutating R. These
    are inherited SAPT policies, not newly introduced CamCASP stabilization.
    """
    R = _square(R, "R", finite=False)
    R = np.nan_to_num(R)
    return np.linalg.pinv(R, rcond=1.e-13).transpose()


def solve_fdds_response(*, metric, metric_inv, W, uncoupled, x_alpha=0.0,
                        hybrid: Optional[Mapping[str, np.ndarray]] = None):
    """Couple one system at one supplied imaginary-axis frequency.

    ``uncoupled`` is already SIGNED (negative) auxiliary response: negate the
    nonhybrid C++ form_unc_amplitude output, but not hybrid ``amp``. Frequency
    dependence is in the supplied intermediates. W includes J and the explicit
    kernel; this function does not infer kernel policy from an SCF functional.

    Hybrid mode requires K1LD, K2LD, K2L, K21L and Rtinv. x_alpha multiplies those
    exchange terms only. Preserve the existing pseudoinverse threshold and dot
    ordering, including the hybrid one-quarter factor. A small residual or
    symmetrized output alone does not establish physical or reference parity.
    """
    metric = _square(metric, "metric")
    shape = metric.shape
    metric_inv = _square(metric_inv, "metric_inv", shape)
    W = _square(W, "W", shape)
    U = _square(uncoupled, "uncoupled", shape).copy()
    if not np.isscalar(x_alpha) or not np.isreal(x_alpha) or not np.isfinite(x_alpha):
        raise ValueError("x_alpha must be a finite real scalar")

    if hybrid is not None:
        names = ("K1LD", "K2LD", "K2L", "K21L", "Rtinv")
        missing = set(names).difference(hybrid)
        if missing:
            raise ValueError(f"Missing hybrid intermediates: {sorted(missing)}")
        h = {name: _square(hybrid[name], name, shape) for name in names}
        X = U - x_alpha * h["K2L"]
        K = -x_alpha * h["K1LD"] - x_alpha * h["K2LD"] + x_alpha * x_alpha * h["K21L"]
        KRS = K.dot(h["Rtinv"]).dot(metric)
    else:
        X = U

    XSW = X.dot(metric_inv).dot(W)
    if hybrid is not None:
        XSW += 0.25 * KRS
    amplitude = np.linalg.pinv(metric - XSW, rcond=1.e-13)
    coupled = X + XSW.dot(amplitude).dot(X)
    return FDDSFrequencyResponse(_symmetrize(U), _symmetrize(coupled), U, coupled)


@dataclass(frozen=True)
class FDDSFullOVFrequencyResponse:
    """Owned raw results in explicitly declared transition-leg coordinates.

    The baseline already includes the supplied Coulomb/exchange operators. It is
    not ordinary uncoupled FDDS. No historical symmetrization is applied here.
    """

    omega: float
    raw_baseline: np.ndarray
    raw_coupled: np.ndarray
    representation: str
    method: str = "full_ov_effective_baseline"


class FDDSFullOVResponse:
    """Explicit supplied-operator route; never selected by legacy SAPT callers.

    For transition legs D (OV rows, declared-coordinate columns), define
    H1 = H1_baseline + 4 D coupling D^T. The effective baseline is
    U = D^T solve(H2 H1_baseline + omega^2 I, -4 H2 D).
    The existing auxiliary coupling helper adds the remaining interaction.

    Full-OV solves use unregularized numpy.linalg.solve; singular baselines raise
    LinAlgError without fallback. Auxiliary coupling retains the helper's 1e-13
    pseudoinverse cutoff. Exact full-coupled equivalence requires its denominator
    to be nonsingular and untruncated. No SCF, integrals, fit, kernel, coordinate
    transformation or orbital-order inference is performed by this class.

    Inputs must already use one consistent OV order. Allowed representations are
    'fitted_density_coefficients', 'fdds_coulomb_auxiliary', and
    'supplied_transition_leg_coordinates'; the caller must choose explicitly.
    This route does not change the inherited projected-hybrid FDDS algorithm.
    """

    def __init__(self, *, h1_baseline, h2, transition_legs, coupling, representation):
        allowed = ("fitted_density_coefficients", "fdds_coulomb_auxiliary",
                   "supplied_transition_leg_coordinates")
        if not isinstance(representation, str) or representation not in allowed:
            raise ValueError("representation must explicitly identify transition-leg coordinates")
        h1 = _square(h1_baseline, "h1_baseline").copy()
        h2 = _square(h2, "h2", h1.shape).copy()
        legs = np.asarray(transition_legs)
        if np.iscomplexobj(legs):
            raise ValueError("transition_legs must be real")
        legs = np.asarray(legs, dtype=float)
        if (legs.ndim != 2 or legs.shape[0] != h1.shape[0] or legs.shape[1] == 0
                or not np.isfinite(legs).all()):
            raise ValueError("transition_legs must be finite with shape (nov, ncoordinates)")
        self._legs = legs.copy()
        n = legs.shape[1]
        self._coupling = _square(coupling, "coupling", (n, n)).copy()
        self._representation = representation
        self._identity_ov = np.eye(h1.shape[0])
        self._identity_coordinates = np.eye(n)
        # Owned products only. Do not retain caller arrays or assume symmetry.
        with np.errstate(over="raise", invalid="raise"):
            self._baseline_product = h2.dot(h1)
            self._rhs = -4.0 * h2.dot(self._legs)
        if not np.isfinite(self._baseline_product).all() or not np.isfinite(self._rhs).all():
            raise ValueError("nonfinite full-OV baseline products")

    @property
    def representation(self):
        return self._representation

    def at_frequency(self, omega):
        """Return independent raw responses at finite imaginary magnitude omega."""
        if (not np.isscalar(omega) or np.asarray(omega).dtype.kind not in "iuf"
                or not np.isfinite(omega) or omega < 0):
            raise ValueError("frequency must be finite, real and nonnegative")
        omega = float(omega)
        omega2 = omega * omega
        if not np.isfinite(omega2):
            raise ValueError("squared frequency must be finite")
        with np.errstate(over="raise", invalid="raise"):
            operator = self._baseline_product + omega2 * self._identity_ov
            solution = np.linalg.solve(operator, self._rhs)
            baseline = self._legs.T.dot(solution)
        response = solve_fdds_response(
            metric=self._identity_coordinates, metric_inv=self._identity_coordinates,
            W=self._coupling, uncoupled=baseline)
        if not np.isfinite(response.raw_coupled).all():
            raise ValueError("nonfinite full-OV coupled response")
        # Do not expose the auxiliary helper's fixed label for artificial identity
        # coordinates, or call this exchange-containing baseline 'uncoupled'.
        return FDDSFullOVFrequencyResponse(
            omega, baseline.copy(), response.raw_coupled.copy(), self._representation)


def _compute_fxc(PQrho, half_Saux, halfp_Saux, x_alpha, rho_thresh=1.e-8):
    """Shared, unchanged SAPT gridless (P|fxc|Q) ALDA construction."""
    naux = PQrho.shape[0]
    PQrho_lvl = core.triplet(half_Saux, PQrho, half_Saux, False, False, False)
    rho = core.Vector("rho eigenvalues", naux)
    U = core.Matrix("rho eigenvectors", naux, naux)
    PQrho_lvl.diagonalize(U, rho, core.DiagonalizeOrder.Ascending)
    mask = rho.np < rho_thresh
    rho.np[mask] = rho_thresh
    dft_size = rho.shape[0]
    inp = {"RHO_A": rho}
    out = {"V": core.Vector(dft_size), "V_RHO_A": core.Vector(dft_size), "V_RHO_A_RHO_A": core.Vector(dft_size)}
    func_x = core.LibXCFunctional('XC_LDA_X', True)
    func_x.compute_functional(inp, out, dft_size, 2)
    out["V_RHO_A_RHO_A"].scale(1.0 - x_alpha)
    func_c = core.LibXCFunctional('XC_LDA_C_VWN', True)
    func_c.compute_functional(inp, out, dft_size, 2)
    out["V_RHO_A_RHO_A"].np[mask] = 0
    Ul = U.clone()
    Ul.np[:] *= out["V_RHO_A_RHO_A"].np
    tmp = core.doublet(Ul, U, False, True)
    return core.triplet(halfp_Saux, tmp, halfp_Saux, False, False, False)


class FDDSMonomerResponse:
    """Psi4 FDDS response from ONE explicit system's bases and orbital data.

    No SCF, basis selection, dummy partner or dispersion integration is performed.
    C1 closed-shell inputs with positive gaps are required. D_alpha is the one-spin
    AO density. x_alpha is the explicit response exchange fraction, not inferred
    from an SCF functional. Nonhybrid construction requires x_alpha=0. Gridless
    ALDA and inherited metric/QR thresholds are Psi4 policies, not an assertion of
    CamCASP constrained-DF/kernel equivalence. Hybrid QR requires nov >= naux.
    """

    __slots__ = ("_native", "_metric", "_metric_inv", "_W", "_Rtinv", "_is_hybrid", "_x_alpha", "_rho_thresh")

    def __init__(self, primary, auxiliary, occupied, virtuals, occupied_energies,
                 virtual_energies, D_alpha, *, is_hybrid, x_alpha, rho_thresh=1.e-8):
        if not isinstance(is_hybrid, (bool, np.bool_)):
            raise ValueError("is_hybrid must be boolean")
        for name, value in (("x_alpha", x_alpha), ("rho_thresh", rho_thresh)):
            if not np.isscalar(value) or not np.isreal(value) or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite real scalar")
        if rho_thresh <= 0:
            raise ValueError("rho_thresh must be positive")
        if not is_hybrid and x_alpha != 0:
            raise ValueError("Nonhybrid response requires x_alpha=0")
        self._is_hybrid, self._x_alpha, self._rho_thresh = bool(is_hybrid), float(x_alpha), float(rho_thresh)
        self._native = core.FDDS_Monomer(primary, auxiliary, occupied, virtuals,
                                         occupied_energies, virtual_energies, self._is_hybrid)
        density = self._native.project_density(D_alpha)
        half_Saux = self._native.aux_overlap()
        half_Saux.power(-0.5, 1.e-12)
        halfp_Saux = self._native.aux_overlap()
        halfp_Saux.power(0.5, 1.e-12)
        W = self._native.metric()
        W.axpy(1.0, _compute_fxc(density, half_Saux, halfp_Saux, self._x_alpha, self._rho_thresh))
        self._W = W.to_array()
        self._metric = self._native.metric().to_array()
        self._metric_inv = self._native.metric_inv().to_array()
        self._Rtinv = prepare_fdds_hybrid_transform(self._native.R().to_array()) if self._is_hybrid else None

    @property
    def is_hybrid(self):
        return self._is_hybrid

    @property
    def x_alpha(self):
        return self._x_alpha

    @property
    def rho_thresh(self):
        return self._rho_thresh

    def at_frequency(self, omega):
        """Return owned auxiliary responses at imaginary frequency magnitude omega."""
        if not np.isscalar(omega) or not np.isreal(omega) or not np.isfinite(omega) or omega < 0:
            raise ValueError("frequency must be finite, real and nonnegative")
        if self._is_hybrid:
            h = {k: v.to_array() for k, v in self._native.form_aux_matrices(float(omega)).items()}
            u = h.pop("amp")
            h["Rtinv"] = self._Rtinv
        else:
            u = self._native.form_unc_amplitude(float(omega))
            u.scale(-1.0)
            u = u.to_array()
            h = None
        return solve_fdds_response(metric=self._metric, metric_inv=self._metric_inv, W=self._W,
                                   uncoupled=u, x_alpha=self._x_alpha, hybrid=h)
