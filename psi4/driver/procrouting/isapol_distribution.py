# Copyright (c) 2007-2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""Owned distributed moments, independent of the response solver and LW.

Q has site-major rows, real Racah components 00,10,11c,11s,... (z,x,y
for dipoles), global Cartesian axes and bohr origins. Columns are the exact
ordered effective functions of a declared response BasisRecipe, not necessarily
ISA's density-fit AUX. Q maps fitted *density* coefficients to moments; it
contains neither an electronic minus sign nor a response/neutrality correction.

Producers: the analytic DF-centre rule and an explicitly declared, experimental
ISA-A partition (``isa_moments``). Neither is inferred from the other. The ISA
iteration, SCF seal and state modules load only when an ISA function is called,
so the DF-centre contract stays independent of them.
"""
from dataclasses import dataclass
import hashlib

import numpy as np
from psi4 import core

from .isapol_basis import BasisRecipe, _owned
from .isapol_df_multipoles import analytic_df_centre_multipoles

CONVENTION = 'site-major real Racah 00,10,11c,11s,...; global axes; bohr; density moments'


def basis_identity(recipe):
    """Conservative identity including representation, ordered shells and origins."""
    if not isinstance(recipe, BasisRecipe):
        raise TypeError('explicit response AUX BasisRecipe required')
    return hashlib.sha256(repr(recipe).encode()).hexdigest()


def _width(recipe):
    return sum((s.l+1)*(s.l+2)//2 if recipe.representation == 'Cartesian' else 2*s.l+1
               for s in recipe.shells)


@dataclass(frozen=True)
class DistributedMoments:
    """Immutable numeric snapshot with explicit axes and successful model status.

    ``state_sha256`` is None only for a density-independent producer (DF-centre).
    Diagnostics are immutable scalar/tuple pairs, not live native controller
    records. Convergence is a provider declaration, not an accuracy certificate.
    """
    values: np.ndarray
    labels: tuple
    origins_bohr: tuple
    rank: int
    auxiliary: BasisRecipe
    model: str
    provenance: str
    converged: bool
    state_sha256: str | None = None
    diagnostics: tuple = ()
    convention: str = CONVENTION

    def __post_init__(self):
        labels = tuple(self.labels)
        origins = tuple(tuple(float(x) for x in p) for p in self.origins_bohr)
        if (not labels or any(not isinstance(s, str) or not s.strip() for s in labels)
                or len(set(labels)) != len(labels)):
            raise ValueError('unique nonempty site labels required')
        if np.asarray(origins).shape != (len(labels), 3) or not np.isfinite(origins).all():
            raise ValueError('finite bohr origins in site order required')
        if type(self.rank) is not int or not 0 <= self.rank <= 4:
            raise ValueError('distributed moment rank must be 0..4')
        basis_identity(self.auxiliary)
        if origins != self.auxiliary.centres:
            raise ValueError('site order/origins must match response AUX centres')
        if self.convention != CONVENTION:
            raise ValueError('unsupported distributed moment convention')
        if self.converged is not True:
            raise ValueError('distributed moments require a converged model')
        if any(not isinstance(s, str) or not s.strip() for s in (self.model, self.provenance)):
            raise ValueError('model and provenance required')
        if self.state_sha256 is not None and (not isinstance(self.state_sha256, str) or not self.state_sha256):
            raise ValueError('invalid density state identity')
        if np.asarray(self.values).dtype.kind not in 'fiu':
            raise ValueError('Q must contain real numeric values')
        values = _owned(self.values)
        if values.shape != (len(labels)*(self.rank+1)**2, _width(self.auxiliary)):
            raise ValueError('Q shape must match site/rank rows and response AUX columns')
        def immutable(value):
            if isinstance(value, (tuple, list)):
                return tuple(immutable(v) for v in value)
            if isinstance(value, (str, bool, int, float, type(None))):
                return value
            raise TypeError('diagnostics must contain scalar/tuple values')
        diagnostics = tuple((str(k), immutable(v)) for k, v in self.diagnostics)
        object.__setattr__(self, 'values', values)
        object.__setattr__(self, 'labels', labels)
        object.__setattr__(self, 'origins_bohr', origins)
        object.__setattr__(self, 'diagnostics', diagnostics)

    @property
    def auxiliary_sha256(self):
        return basis_identity(self.auxiliary)

    def validate_for(self, auxiliary, sites, rank, state_sha256):
        if (auxiliary != self.auxiliary or tuple(s.label for s in sites) != self.labels
                or tuple(tuple(s.origin) for s in sites) != self.origins_bohr or rank != self.rank):
            raise ValueError('distributed moment site/rank/response AUX identity mismatch')
        if self.state_sha256 is not None and state_sha256 != self.state_sha256:
            raise ValueError('distributed moment density state identity mismatch')

    def anchor_legs(self, coefficients):
        """The sole provider-independent replacement for fit.coefficients @ Q.T."""
        if np.asarray(coefficients).dtype.kind not in 'fiu':
            raise ValueError('real numeric fitted response AUX coefficients required')
        coefficients = np.asarray(coefficients, dtype=float)
        if (coefficients.ndim != 2 or coefficients.shape[1] != self.values.shape[1]
                or not np.isfinite(coefficients).all()):
            raise ValueError('finite fitted response AUX coefficient matrix required')
        legs = coefficients @ self.values.T
        if not np.isfinite(legs).all():
            raise ValueError('nonfinite distributed anchor legs')
        return legs


def analytic_df_moments(auxiliary, sites, rank):
    """Wrap the existing analytic producer without altering a single Q element."""
    result = analytic_df_centre_multipoles(auxiliary, sites, rank)
    return DistributedMoments(result.values, result.labels, result.origins, rank, auxiliary,
                              'df_centre_analytic', result.provenance, True,
                              diagnostics=tuple(result.diagnostics.items()))


def _validate_rank_and_grid(model, rank, integration_grid):
    if type(rank) is not int or not 0 <= rank <= 4:
        raise ValueError('distributed moment rank must be 0..4')
    if (not isinstance(integration_grid, np.ndarray) or integration_grid.dtype != np.float64
            or integration_grid.ndim != 2 or integration_grid.shape[1] != 4
            or not len(integration_grid) or not np.isfinite(integration_grid).all()):
        raise ValueError(f'{model} requires an explicit finite float64 full integration grid (rows,4)')


def validate_isa_inputs(wfn, recipe, auxiliary, sites, rank, integration_grid):
    from . import isapol_native_partition as isa
    if not isinstance(recipe, isa.PartitionRecipe):
        raise TypeError('ISA requires an explicit PartitionRecipe')
    geometry = tuple(map(tuple, np.asarray(wfn.molecule().geometry())))
    if (tuple(s.label for s in recipe.sites) != tuple(s.label for s in sites)
            or tuple(s.origin for s in recipe.sites) != geometry
            or tuple(tuple(s.origin) for s in sites) != geometry
            or auxiliary.centres != geometry or recipe.auxiliary.centres != geometry):
        raise ValueError('ISA recipe/site/response AUX/wavefunction identity mismatch')
    _validate_rank_and_grid('ISA', rank, integration_grid)


def isa_resource_plan(wfn, recipe, auxiliary, integration_grid, rank):
    """Conservative numeric/work estimates, not RSS or integral-engine CPU caps.

    Charge full-grid atomic collocation even when native cache admission chooses
    the uncached path, all max_iterations history metrics, copied grids and Q,
    MAIN/Drho buffers, plus 256 MiB scratch. Lebedev rounding is bounded by
    2*requested+6 (the native grid clamps at 5294). No per-iteration refund.
    Python object/container overhead and vendor integral/BLAS workspace are not
    numerically capped. Large recipes can be refused even if they would converge
    early; this adapter does not claim an exact work bound for arbitrary shells.
    """
    ns = len(recipe.sites)
    g = ns*(recipe.grid.radial_points-1)*min(5294, 2*recipe.grid.spherical_points+6)
    h, p, d = len(integration_grid), _width(auxiliary), _width(recipe.auxiliary)
    a = [_width(s.atomic) for s in recipe.sites]
    s = sum(_width(site.shape) for site in recipe.sites)
    b, it = wfn.basisset().nbf(), recipe.controller.max_iterations
    a2 = sum(k*k for k in a)
    q = ns*(rank+1)**2
    numeric = 8*(g*(sum(a)+40*ns+40)+(it+12)*(a2+16*sum(a)+16*s)
                 +h*(32*ns+16)+8*p*q+16*d*d+8*d*b*b+64*b*b)+256*1024**2
    work = (it*(8*g*(a2+ns*s)+8*sum(k**3 for k in a))
            +16*d*b**3+8*d**3+4*h*q*p)
    return int(numeric), int(work)


def _stockholder_q(auxiliary, sites, rank, points, weights, sampled, provenance, cutoff):
    """Integrate w_A = shape_A/sum_B shape_B against the response AUX at every point.

    The native integrator forms the ratio and excludes points whose denominator
    is at most ``cutoff``. The charge-row error compares the site-summed charge
    rows with the analytic AUX charges; it measures Q quadrature, not the model.
    """
    total = np.sum(sampled, axis=0).tolist()
    qsites = []
    for site, shape in zip(sites, sampled):
        samples = core.IsaMultipoleSamples()
        samples.points, samples.weights = points, weights
        samples.shape, samples.shape_sum = shape.tolist(), total
        samples.auxiliary_sites = list(range(len(sites)))
        item = core.IsaMultipoleSite()
        item.label, item.origin, item.rank, item.samples = site.label, site.origin, rank, samples
        qsites.append(item)
    response_aux = auxiliary.build('MolecularAux')
    q = core.IsaPartitionedMultipoles(response_aux, qsites, provenance, cutoff)
    values = np.asarray(q.values)
    charge_error = float(np.max(np.abs(values[::(rank+1)**2].sum(axis=0)
                                      - np.asarray(core.IsaAuxCoulomb(response_aux).charges()))))
    return q, values, charge_error


# The ISA-A iteration is checked for convergence; Q-grid convergence of the site
# properties is not (site C6 moves ~1e-4 between tested grids at the Fit-3 switch).
ISA_EXPERIMENTAL = ('EXPERIMENTAL: converged ISA-A iteration; site-property Q-grid convergence '
                    'not established (Fit-3 tail switch)')


def isa_moments(wfn, recipe, auxiliary, sites, rank, *, caller_converged, integration_grid, ledger):
    """Fresh native ISA iteration -> final-tail shapes -> response-AUX Q.

    The recipe's grid controls iteration; integration_grid independently declares
    the full molecular Q quadrature. All sites contribute at every point. Native
    signed-tail, Gaussian clipping and denominator cutoff semantics are retained.
    The integration grid is snapshotted on entry, so the sampled points and the
    recorded grid identity cannot diverge from a later caller mutation.
    """
    from . import isapol_native_partition as isa
    from .isapol_native import _context
    from .isapol_native_correction import require_scf_seal
    validate_isa_inputs(wfn, recipe, auxiliary, sites, rank, integration_grid)
    if caller_converged is not True:
        raise ValueError('ISA requires caller_converged=True')
    require_scf_seal(wfn)
    state_id = _context(wfn)
    ledger.admit('ISA iteration and response-AUX Q',
                 *isa_resource_plan(wfn, recipe, auxiliary, integration_grid, rank))
    integration_grid = _owned(integration_grid)
    result = isa.native_partition(wfn, recipe, caller_converged=True, build_multipoles=False)
    if not result.converged:
        raise RuntimeError(f'ISA-A did not converge: {result.trajectory.termination}; Q unavailable')
    points, weights = integration_grid[:, :3].tolist(), integration_grid[:, 3].tolist()
    shapes = [s.shape.build('Shape') for s in recipe.sites]
    sampled = isa.final_shape_samples(shapes, result.trajectory.state, recipe.sites, points,
                                  tails=result.trajectory.final_tails)
    provenance = (result.provenance + '; frozen ground-state shapes integrated on explicit response-AUX grid; '
                  + ISA_EXPERIMENTAL)
    q, values, charge_error = _stockholder_q(auxiliary, sites, rank, points, weights, sampled,
                                             provenance, recipe.controller.density_cutoff)
    diagnostics = dict(status='experimental', iteration_converged=True,
        property_grid_convergence='not_established',
        q_shape=values.shape, q_charge_row_error=charge_error,
        iterations=result.trajectory.state.iteration, max_delta=result.trajectory.state.max_delta,
        termination=result.trajectory.termination, drho_metric_condition=result.drho_metric_condition,
        drho_relative_residual=result.drho.relative_residual,
        grid_charge_error=float(result.grid_weights @ result.density_samples - result.drho.fitted_electrons),
        iteration_grid_points=len(result.grid_points), integration_grid_points=len(integration_grid),
        recipe_sha256=hashlib.sha256(repr(recipe).encode()).hexdigest(),
        density_auxiliary_sha256=basis_identity(recipe.auxiliary),
        integration_grid_sha256=hashlib.sha256(integration_grid.tobytes()).hexdigest(),
        denominator_cutoff=recipe.controller.density_cutoff,
        excluded_denominators=tuple(q.excluded_denominators), negative_ratios=tuple(q.negative_ratios),
        final_tails=tuple((t.defined, t.amplitude, t.exponent, t.cutoff) for t in result.trajectory.final_tails))
    return DistributedMoments(values, tuple(s.label for s in sites),
        tuple(tuple(s.origin) for s in sites), rank, auxiliary, 'isa', provenance, True,
        state_id, tuple(diagnostics.items()))
