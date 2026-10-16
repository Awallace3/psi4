# Copyright (c) 2007-2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""Owned distributed moments, independent of the response solver and LW.

Q has site-major rows, real Racah components 00,10,11c,11s,... (z,x,y
for dipoles), global Cartesian axes and bohr origins. Columns are the exact
ordered effective functions of a declared response BasisRecipe, not necessarily
ISA's density-fit AUX. Q maps fitted *density* coefficients to moments; it
contains neither an electronic minus sign nor a response/neutrality correction.

Producers: the analytic DF-centre rule, an explicitly declared, experimental
ISA-A partition (``isa_moments``) and the native all-shell MBIS partition
(``mbis_moments``). None is inferred from another. The ISA iteration, SCF seal
and state modules load only when an ISA or MBIS function is called, so the
DF-centre contract stays independent of them.
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
    Drho-C refinement is charged explicitly, not taken from the scratch term or
    the d**3 solve term. Bytes: 4d doubles and the 134-digit accumulator when
    refinement is enabled, held once for all iterations: the solver owns three
    d-vectors (x, the plain LU x, and one correction into which each residual is
    written) and a stack accumulator; the fourth vector is margin.
    Work, in source-level scalar operations (each arithmetic, bitwise, shift,
    compare, select, branch, cast, element load or store and loop step is one
    unit), for every iteration up to the cap whatever the observed count: 256 per
    residual product term (d*(d+1) per residual, about 190 counted), 16384 per
    row for rounding and the update (about 8500 counted, worst case), and 16d**2
    plus 64d for the triangular correction solves; plus 16d**2+64d once for the
    finiteness scans and copies (about 8d**2+40d counted). These are operation counts, not a time bound.
    Python object/container overhead and vendor integral/BLAS workspace are not
    numerically capped. Large recipes can be refused even if they would converge
    early; this adapter does not claim an exact work bound for arbitrary shells.
    """
    from .isapol_native_partition import DRHO_REFINEMENT_ITERATIONS as refinement
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
    if refinement:
        numeric += 8*(4*d+134)
    work += 16*d*d+64*d+refinement*(256*d*(d+1)+16384*d+16*d*d+64*d)
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


MBIS_OPTIONS = ('MBIS_RADIAL_POINTS', 'MBIS_SPHERICAL_POINTS', 'MBIS_PRUNING_SCHEME',
                'MBIS_MAXITER', 'MBIS_D_CONVERGENCE')
#: Global DFT grid options the native MBIS DFTGrid also reads (cubature.cc
#: DFTGrid::buildGridFromOptions); recorded, never changed here.
MBIS_GRID_OPTIONS = ('DFT_RADIAL_SCHEME', 'DFT_NUCLEAR_SCHEME', 'DFT_GRID_NAME', 'DFT_BLOCK_SCHEME',
                     'DFT_BLOCK_MAX_POINTS', 'DFT_BLOCK_MIN_POINTS', 'DFT_BS_RADIUS_ALPHA',
                     'DFT_PRUNING_ALPHA', 'DFT_WEIGHTS_TOLERANCE', 'DFT_BLOCK_MAX_RADIUS',
                     'DFT_BASIS_TOLERANCE', 'MAX_RADIAL_MOMENT')
MBIS_SNAPSHOT = ('MBIS SHELL COUNTS', 'MBIS SHELL POPULATIONS', 'MBIS SHELL WIDTHS')
MBIS_MAX_SHELLS = 7
MBIS_SCRATCH_BYTES = 256*1024**2
_MBIS_REGION_PRUNING = ('ROBUST', 'TREUTLER')
_MBIS_FUNCTION_PRUNING = ('NONE', 'FLAT', 'P_GAUSSIAN', 'D_GAUSSIAN', 'P_SLATER', 'D_SLATER',
                          'LOG_GAUSSIAN', 'LOG_SLATER')


def _mbis_settings():
    """Validated native MBIS settings; refused before any grid or SCF-state work.

    Fewer than two MBIS_MAXITER performs no shell update and can never converge.
    The plan needs a point-count bound: region pruning uses at most
    max(MBIS_SPHERICAL_POINTS, 50) points per radial shell (its fixed inner
    regions are Lebedev orders 7 and 11), and function pruning at most the
    requested sphere when DFT_PRUNING_ALPHA >= 0. A named DFT grid replaces the
    MBIS grid entirely, so it has no such bound and is refused.
    """
    values = {name: core.get_global_option(name) for name in MBIS_OPTIONS+MBIS_GRID_OPTIONS}
    for name in ('MBIS_RADIAL_POINTS', 'MBIS_SPHERICAL_POINTS'):
        if type(values[name]) is not int or values[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    if type(values['MBIS_MAXITER']) is not int or values['MBIS_MAXITER'] < 2:
        raise ValueError('MBIS_MAXITER must allow at least one shell update (>= 2)')
    threshold = values['MBIS_D_CONVERGENCE']
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('MBIS_D_CONVERGENCE must be finite and positive')
    scheme = values['MBIS_PRUNING_SCHEME']
    if values['DFT_GRID_NAME']:
        raise ValueError('MBIS point-count bound unavailable: DFT_GRID_NAME replaces the MBIS grid')
    if scheme in _MBIS_REGION_PRUNING:
        sphere = max(values['MBIS_SPHERICAL_POINTS'], 50)
    elif scheme in _MBIS_FUNCTION_PRUNING and values['DFT_PRUNING_ALPHA'] >= 0:
        sphere = values['MBIS_SPHERICAL_POINTS']
    else:
        raise ValueError(f'MBIS point-count bound unavailable for pruning {scheme} '
                         f"with DFT_PRUNING_ALPHA={values['DFT_PRUNING_ALPHA']}")
    return values, sphere


def validate_mbis_inputs(wfn, auxiliary, sites, rank, integration_grid):
    """Sites, response AUX, Q grid and native MBIS settings; no native work."""
    basis_identity(auxiliary)
    geometry = tuple(map(tuple, np.asarray(wfn.molecule().geometry())))
    if tuple(tuple(s.origin) for s in sites) != geometry or auxiliary.centres != geometry:
        raise ValueError('MBIS site/response AUX/wavefunction identity mismatch')
    _validate_rank_and_grid('MBIS', rank, integration_grid)
    _mbis_settings()


def mbis_resource_plan(wfn, auxiliary, integration_grid, rank):
    """Conservative numeric/work plans for native MBIS and for Q, not RSS or time caps.

    Both stages hold the owned copy of the integration grid. Native: every
    point of the bounded grid (``_mbis_settings``) carries 16 doubles of grid
    storage and the function's own 7 per-point and 7 per-atom-point vectors
    (coordinates, weights, density, distances, displacements, proatom and
    promolecule densities with their next iterates, partitioned density); 8
    basis-squared density copies; every MBIS_MAXITER-1 update over seven shells.
    Q: log-domain proatoms and their temporaries, stockholder shapes, CPython
    list conversions at 64-bit object sizes, the native sample copies for
    every site, and Q with its owned copies. Each stage adds a fixed 256 MiB
    allowance for DFTGrid, point-function, OpenMP, integrator and OEProp
    internals that are not individually known. Work counts source-level scalar
    operations with exp/log/pow charged 16 each; it is not a time bound.
    Interpreter allocator overhead and vendor workspaces are excluded.
    """
    values, sphere = _mbis_settings()
    ns, b = wfn.molecule().natom(), wfn.basisset().nbf()
    g = ns*values['MBIS_RADIAL_POINTS']*sphere
    updates = values['MBIS_MAXITER']-1
    h, p, q = len(integration_grid), _width(auxiliary), ns*(rank+1)**2
    grid_copy = 32*h
    native = 8*(g*(24+8*ns)+8*b*b)+grid_copy+MBIS_SCRATCH_BYTES
    moments = max(4, values['MAX_RADIAL_MOMENT'])-1
    native_work = (2*g*b*b+g*ns*(MBIS_MAX_SHELLS*64+256+32*moments)
                   +updates*g*ns*(MBIS_MAX_SHELLS*128+16))
    sampling = 8*h*(64+16*ns)+32*p*q+grid_copy+MBIS_SCRATCH_BYTES
    sampling_work = h*ns*(MBIS_MAX_SHELLS*64+64)+4*h*q*p
    return (int(native), int(native_work)), (int(sampling), int(sampling_work))


def run_native_mbis(wfn):
    """Fresh native MBIS_CHARGES attempt; returns the validated, owned all-shell snapshot.

    Deliberately not psi4.oeprop: MBIS_VOLUME_RATIOS runs free-atom SCFs in
    Python that can fail before native entry and leave an older snapshot live.
    The snapshot is invalidated here and again at native entry, and native MBIS
    sets MBIS CONVERGED last, after all postprocessing, so only this attempt's
    success can be read. Unsupported ECP/ghost inputs are native refusals.
    """
    wfn.set_scalar_variable('MBIS CONVERGED', 0.)
    for name in MBIS_SNAPSHOT:
        if wfn.has_array_variable(name):
            wfn.del_array_variable(name)
    oe = core.OEProp(wfn)
    oe.add('MBIS_CHARGES')
    try:
        oe.compute()
    except Exception as exc:
        raise RuntimeError(f'native MBIS failed ({exc}); Q unavailable') from exc
    nat = wfn.molecule().natom()
    if (not wfn.has_scalar_variable('MBIS CONVERGED') or wfn.scalar_variable('MBIS CONVERGED') != 1
            or not all(wfn.has_array_variable(n) for n in MBIS_SNAPSHOT)):
        raise RuntimeError('native MBIS did not publish a converged all-shell snapshot; Q unavailable')
    counts, populations, widths = (np.array(wfn.array_variable(n).np, dtype=float) for n in MBIS_SNAPSHOT)
    if (counts.shape != (nat, 1) or populations.shape != (nat, MBIS_MAX_SHELLS)
            or widths.shape != (nat, MBIS_MAX_SHELLS)):
        raise RuntimeError('native MBIS snapshot has inconsistent dimensions')
    counts = counts[:, 0]
    if not np.all((counts == np.round(counts)) & (counts >= 1) & (counts <= MBIS_MAX_SHELLS)):
        raise RuntimeError('native MBIS shell counts must be integers 1..7')
    counts = counts.astype(int)
    active = np.arange(MBIS_MAX_SHELLS)[None, :] < counts[:, None]
    for array in (populations, widths):
        if (not np.isfinite(array).all() or not np.all(array[active] > 0)
                or np.any(array[~active] != 0)):
            raise RuntimeError('native MBIS shells must be finite and positive, with zero padding')
    residual = wfn.scalar_variable('MBIS DENSITY RESIDUAL')
    iterations = wfn.scalar_variable('MBIS ITERATIONS')
    electrons = (wfn.scalar_variable('MBIS GRID ELECTRONS')
                 if wfn.has_scalar_variable('MBIS GRID ELECTRONS') else float('nan'))
    threshold = core.get_global_option('MBIS_D_CONVERGENCE')
    if not (np.isfinite(residual) and 0 <= residual < threshold and np.isfinite(electrons) and electrons > 0
            and iterations == int(iterations) and 1 <= iterations < core.get_global_option('MBIS_MAXITER')):
        raise RuntimeError('native MBIS convergence diagnostics are inconsistent with its snapshot')
    return dict(counts=tuple(int(c) for c in counts),
                populations=tuple(map(tuple, populations.tolist())),
                widths_bohr=tuple(map(tuple, widths.tolist())),
                iterations=int(iterations), density_residual=float(residual), threshold=float(threshold),
                grid_electrons=float(electrons), population_sum=float(populations.sum()))


def mbis_log_proatoms(snapshot, origins, points):
    """ln rho_A^0(r), rho_A^0 = sum_s N_s exp(-|r-R_A|/sigma_s)/(8 pi sigma_s^3), all shells.

    Summed as a log-sum-exp, so no proatom underflows to zero at any distance;
    a nonfinite logarithm (an absurdly distant point) is refused.
    """
    points = np.asarray(points, dtype=float)
    result = np.empty((len(origins), len(points)))
    for a, (count, origin) in enumerate(zip(snapshot['counts'], origins)):
        n = np.asarray(snapshot['populations'][a][:count], dtype=float)
        sigma = np.asarray(snapshot['widths_bohr'][a][:count], dtype=float)
        if not (len(n) == count >= 1 and np.isfinite(n).all() and np.isfinite(sigma).all()
                and np.all(n > 0) and np.all(sigma > 0)):
            raise ValueError('MBIS shells must be finite and positive')
        distance = np.linalg.norm(points-np.asarray(origin), axis=1)
        terms = np.log(n/(8*np.pi*sigma**3))[:, None]-distance[None, :]/sigma[:, None]
        result[a] = np.logaddexp.reduce(terms, axis=0)
    if not np.isfinite(result).all():
        raise ValueError('nonfinite MBIS proatom logarithm; integration grid too distant')
    return result


def mbis_moments(wfn, auxiliary, sites, rank, *, caller_converged, integration_grid, ledger):
    """Fresh native MBIS -> frozen all-shell proatoms -> stockholder response-AUX Q.

    No ISA recipe or iteration is involved. Weights are
    w_A = exp(l_A - max_B l_B)/sum_C exp(l_C - max_B l_B) with l = ln rho^0, so
    the integrator's denominator lies in [1, nsites] at every grid point: there
    are no denominator exclusions (cutoff 0 is declared), no clipping and no
    neutrality projection. A weight below ~1e-308 relative to the dominant
    proatom underflows to exactly zero; those counts are reported. Far from the
    molecule the site with the most diffuse shell takes the whole weight. The
    integration grid and the snapshot are copied on entry, so later caller or
    wavefunction changes cannot alter Q or its recorded identities.
    """
    from .isapol_native import _context
    from .isapol_native_correction import require_scf_seal
    validate_mbis_inputs(wfn, auxiliary, sites, rank, integration_grid)
    if caller_converged is not True:
        raise ValueError('MBIS requires caller_converged=True')
    require_scf_seal(wfn)
    state_id = _context(wfn)
    native_plan, sampling_plan = mbis_resource_plan(wfn, auxiliary, integration_grid, rank)
    ledger.admit('native MBIS partition', *native_plan)
    ledger.admit('MBIS stockholder response-AUX Q', *sampling_plan)
    integration_grid = _owned(integration_grid)
    settings = _mbis_settings()[0]
    options = tuple((name, settings[name]) for name in MBIS_OPTIONS)
    grid_options = tuple((name, settings[name]) for name in MBIS_GRID_OPTIONS)
    snapshot = run_native_mbis(wfn)
    if _context(wfn) != state_id:
        raise RuntimeError('native MBIS changed the wavefunction state')
    points, weights = integration_grid[:, :3].tolist(), integration_grid[:, 3].tolist()
    log_proatoms = mbis_log_proatoms(snapshot, [s.origin for s in sites], integration_grid[:, :3])
    sampled = np.exp(log_proatoms-log_proatoms.max(axis=0))
    del log_proatoms
    weight_sum_error = float(np.max(np.abs((sampled/sampled.sum(axis=0)).sum(axis=0)-1)))
    provenance = ('native Psi4 MBIS (Verstraelen et al. JCTC 2016), all converged shells; '
                  + ', '.join(f'{k}={v}' for k, v in options)
                  + '; frozen ground-state log-domain stockholder weights on explicit response-AUX grid')
    q, values, charge_error = _stockholder_q(auxiliary, sites, rank, points, weights, sampled,
                                             provenance, 0.)
    if any(q.excluded_denominators) or any(q.negative_ratios):
        raise RuntimeError('MBIS weights produced excluded or negative stockholder ratios')
    diagnostics = dict(q_shape=values.shape, q_charge_row_error=charge_error,
        weight_sum_error=weight_sum_error, underflowed_weights=tuple(int(n) for n in (sampled == 0).sum(axis=1)),
        iterations=snapshot['iterations'], density_residual=snapshot['density_residual'],
        density_threshold=snapshot['threshold'], grid_electrons=snapshot['grid_electrons'],
        population_sum=snapshot['population_sum'], shell_counts=snapshot['counts'],
        shell_populations=snapshot['populations'], shell_widths_bohr=snapshot['widths_bohr'],
        native_options=options, native_grid_options=grid_options,
        integration_grid_points=len(integration_grid),
        integration_grid_sha256=hashlib.sha256(integration_grid.tobytes()).hexdigest(),
        denominator_cutoff=0., excluded_denominators=tuple(q.excluded_denominators),
        negative_ratios=tuple(q.negative_ratios))
    return DistributedMoments(values, tuple(s.label for s in sites),
        tuple(tuple(s.origin) for s in sites), rank, auxiliary, 'mbis', provenance, True,
        state_id, tuple(diagnostics.items()))
