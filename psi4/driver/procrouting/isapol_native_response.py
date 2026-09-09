# Psi4: Copyright (c) 2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""Native restricted-C1 operators with shared full-OV frequency solves.

No SCF, GRAC inference, auxiliary metric inference, legacy SAPT options, or
atomic-property registration. This producer is not an atomic alpha/Cn endpoint.
"""
from dataclasses import dataclass, field

import numpy as np

from psi4 import core
from .sapt.fdds_response import FDDSFullOVResponse
from .isapol_native_correction import validate_correction


@dataclass(frozen=True)
class NativeWavefunctionResponse:
    """Owned native operators and one reusable shared solver.

    ``provider`` getters return independent matrices/vectors. ``caller_converged``
    is a declaration, not an independently observed SCF convergence status.
    Orbital context uses Psi4 AO order and OV index a*nocc+i, all in atomic units.
    Fitted legs, when supplied, must already use precisely that context/order.
    """
    provider: object
    _response: FDDSFullOVResponse = field(repr=False)
    coordinate_declaration: str
    correction_provenance: object
    caller_converged: bool = True
    convergence_evidence: str = "caller declaration only; restricted metadata, density and orthonormality checked"

    @property
    def representation(self):
        return self._response.representation

    def at_frequency(self, omega):
        """Return shared FDDS full-OV raw signed response at imaginary omega."""
        return self._response.at_frequency(omega)


def native_response_from_wavefunction(wavefunction, *, caller_converged, kernel,
                                     exact_exchange, local_scale, grid=None,
                                     density_cutoff=1.e-10, max_bytes=512*1024**2,
                                     max_nov=512, transition_legs=None,
                                     representation=None, scf_correction='NONE',
                                     expected_grac_shift=None):
    """Construct native operators ONCE; return a reusable frequency provider.

    Explicit kernels:
      * ``no_local``: direct RPA at a=0, TDHF at a=1, scaled global exchange
        otherwise; requires local_scale=0, grid=None.
      * ``alda_slater``, ``alda_slater_pw92``, ``alda_slater_vwn``: numerical
        real-space unpolarized LibXC second derivative, with exchange XC_LDA_X
        and (if named) XC_LDA_C_PW or XC_LDA_C_VWN. No gridless FDDS substitution.
        Requires explicit finite [x,y,z,w] rows (bohr, bohr^3), nonnegative w.

    H1=Delta+4V-a(X+Y)+4*b*L, H2=Delta-a(X-Y). Both a and b are explicit;
    b is NOT inferred as 1-a. The local primitive L has no internal spin factor.
    Total rho=2 sum_i phi_i^2; rho<density_cutoff is skipped without renormalizing
    the grid. The grid is caller quadrature, NOT a verified SCF-grid snapshot.
    No assertion of CamCASP parity is made for any named policy.

    Default coordinates are identity OV, NOT fitted AUX. To contract distributed
    Q, supply an explicit fitted D with shape (nov,naux), occupied-fast rows and
    representation='fitted_density_coefficients'. Then the shared solver returns
    D.T @ R_OV @ D; all interactions are already in H1, so remaining coupling=0.
    No metric factors, fit regularization, or normalization are inferred here.

    scf_correction defaults to NONE and rejects attached GRAC. FIXED_GRAC
    requires an explicit positive expected_grac_shift, canonical PBE0 and the
    current successful SCF seal. Only the canonical .5/40 LB*.75/VWN*1 profile
    is supported. This accepts SCF inputs, not a GRAC kernel derivative.

    C++ holds private snapshots of basis/molecule, orbitals, energies, density,
    grid and policy. Current support excludes open shell, non-C1, fractional or
    non-Aufbau occupations, complex state, ECP, and dense resources above
    hard caps. Caller must not concurrently mutate inputs during construction.
    """
    correction = validate_correction(wavefunction, scf_correction=scf_correction,
                                     expected_grac_shift=expected_grac_shift)
    if correction.policy == 'FIXED_GRAC' and kernel == 'no_local':
        raise ValueError('FIXED_GRAC admission requires an explicit ALDA response policy; no GRAC kernel derivative')
    if not isinstance(caller_converged, (bool, np.bool_)) or not caller_converged:
        raise ValueError("caller_converged must explicitly be True (declaration, not a verified seal)")
    for name, value in (("exact_exchange", exact_exchange), ("local_scale", local_scale),
                        ("density_cutoff", density_cutoff)):
        if (not np.isscalar(value) or np.asarray(value).dtype.kind not in "iuf"
                or not np.isfinite(value)):
            raise ValueError(f"{name} must be a finite real scalar")
    for name, value in (("max_bytes", max_bytes), ("max_nov", max_nov)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if kernel not in ("no_local", "alda_slater", "alda_slater_pw92", "alda_slater_vwn"):
        raise ValueError("unsupported explicit native kernel")
    if not 0 <= exact_exchange <= 1 or not 0 <= local_scale <= 1 or density_cutoff <= 0:
        raise ValueError("exchange/local scales must be in [0,1] and density cutoff positive")
    if kernel == "no_local" and (local_scale != 0 or grid is not None):
        raise ValueError("no_local requires local_scale=0 and grid=None")
    if kernel != "no_local" and grid is None:
        raise ValueError("numerical ALDA requires an explicit quadrature grid")
    # Check coordinate declaration before expensive native construction.
    if transition_legs is None:
        if representation is not None:
            raise ValueError("representation requires explicit transition_legs; default is direct OV")
        legs = None
    else:
        if representation not in ("fitted_density_coefficients", "supplied_transition_leg_coordinates"):
            raise ValueError("explicit legs require fitted_density_coefficients or supplied_transition_leg_coordinates")
        legs = np.asarray(transition_legs)
        if np.iscomplexobj(legs) or legs.ndim != 2 or not legs.size or not np.isfinite(legs).all():
            raise ValueError("transition_legs must be a finite real nonempty matrix")
        if legs.shape[1] > min(max_nov, 512) or legs.size * 8 > max_bytes:
            raise ValueError("transition coordinate resource limit")
        legs = np.array(legs, dtype=float, copy=True)
    native_grid = None
    if grid is not None:
        points = np.asarray(grid)
        if (np.iscomplexobj(points) or points.ndim != 2 or points.shape[1] != 4
                or not points.shape[0] or points.shape[0] > 1000000
                or not np.isfinite(points).all() or np.any(points[:, 3] < 0)):
            raise ValueError("grid must be finite real nonempty [x,y,z,nonnegative weight] rows")
        if points.size * 8 > max_bytes:
            raise ValueError("grid snapshot resource limit")
        native_grid = core.Matrix.from_array(np.array(points, dtype=float, copy=True))
    provider = core.NativeResponseProvider(
        wavefunction, True, kernel, float(exact_exchange), float(local_scale),
        native_grid, float(density_cutoff), int(max_bytes), int(max_nov))
    nov = provider.nocc * provider.nvir
    if legs is None:
        legs = np.eye(nov)
        representation = "supplied_transition_leg_coordinates"
        declaration = "identity direct OV; t=a*nocc+i; not fitted AUX"
    else:
        if legs.shape[0] != nov:
            raise ValueError("transition_legs rows must match occupied-fast native OV context")
        declaration = "caller-supplied D; t=a*nocc+i; no implicit auxiliary metric or fit"
    response = FDDSFullOVResponse(
        h1_baseline=provider.h1().to_array(), h2=provider.h2().to_array(),
        transition_legs=legs, coupling=np.zeros((legs.shape[1], legs.shape[1])),
        representation=representation)
    if correction.policy == 'FIXED_GRAC':
        if validate_correction(wavefunction, scf_correction=scf_correction,
                               expected_grac_shift=expected_grac_shift) != correction:
            raise ValueError('SCF correction changed during native response construction')
    return NativeWavefunctionResponse(provider, response, declaration, correction,
        convergence_evidence=('verified current SCF seal; ' + correction.response_description
                              if correction.policy == 'FIXED_GRAC' else
                              'caller declaration only; restricted metadata, density and orthonormality checked'))
