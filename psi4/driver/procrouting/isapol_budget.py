# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Property-anchored precision budget for the native ISA intermediates.

Answers one question only: *how accurately must a named intermediate be known
so that a named property of interest is correct to a stated tolerance?* It does
so by perturbing that intermediate and rebuilding every downstream stage through
the same owned objects the shipped chain used, then dividing the property defect
by the input defect. The property defect always uses the recorded-error metric

    max(abs(actual-reference))/max(1,max(abs(reference)))

because a property group shares one scale and is compared against a tolerance
stated in that same scale. The input defect uses it too by default, so that a
required precision is directly comparable to the intermediate errors recorded in
``PROVISIONAL_ACCEPTANCE.md``. That metric divides by the largest
element, which is the right question for an intermediate whose elements share a
scale and the wrong one for an intermediate spanning many decades: a max-scaled
probe of tabulated shape samples is dominated by the tail, where the shipped
values underflow the probe no matter how small the probe is, and the measured
amplification then grows without bound as the probe shrinks instead of
converging. For such an intermediate the property-relevant error model is
elementwise relative accuracy, so ``geometry='relative'`` probes

    max(abs(actual-reference)/abs(reference)) over the reference support

instead. The two metrics are NOT interchangeable and a requirement derived in
one is never compared against an error recorded in the other.

One intermediate is probed as raw parameters rather than as a sampled array: the
ISA-A exponential tail. Its recorded error is a joint (amplitude, exponent)
error per site, and the site carrying the largest error also carries the largest
parameter, so the trajectory comparator's per-site denominator and this module's
single denominator coincide bit-for-bit on that reference. ``raw_tail_parameters``
is therefore compared against that recorded number in exactly the metric it was
recorded in, while the sampled shape array -- whose elements span many decades --
is not.

The sampled shape array is instead anchored one step upstream, at
``shape_coefficients``: the reference comparison records a per-site error for the
ISA-A shape coefficients W, and because every site but the largest has a clamped
denominator there, the concatenated-array error in THIS module's metric is
recovered exactly from those per-site records rather than merely bounded. The
shipped (lagged) tails are held fixed across that probe, as the algorithm's own
final sampling boundary does.

This module is DIAGNOSTIC ONLY. It is not on the public property path, it
produces no property values of its own, and it waives no gate: every rebuild
runs the unrelaxed production LW policy, and a rebuild that the gate rejects is
reported as rejected rather than retried with a looser policy. An amplification
is a *measured lower bound over sampled directions*, so a precision derived from
it is necessary, not sufficient. Amplifications are first-order: each is quoted
only when halving the probe reproduces it, and the linearity defect is reported.

The model (sites, bonds, frames, rank, grid, recipe) is the caller's, read back
from the shipped result records; nothing here infers a model from a property.
"""
from dataclasses import dataclass, field
import hashlib
import numpy as np
from psi4 import core
from .isapol_native_partition import one_gto_initialization, final_shape_samples
from .sapt.fdds_response import FDDSFullOVResponse
from . import isapol_lw as lw
from .isapol_native import NativeProperties

#: Named intermediates, in chain order. Each is perturbed in place and every
#: later stage is rebuilt; no earlier stage is touched.
STAGES = ('drho_c_coefficients', 'shape_coefficients', 'raw_tail_parameters',
          'partition_shape_samples', 'ov_transition_legs', 'coefficient_responses',
          'distributed_site_tensors')

#: The default metric: always the property defect, and the input defect of an
#: ``absolute``-geometry probe.
METRIC = 'max(abs(actual-reference))/max(1,max(abs(reference)))'

STATUS = 'diagnostic_only_first_order_directional_lower_bound_not_a_gate'


#: The elementwise-relative metric, for intermediates spanning many decades.
RELATIVE_METRIC = 'max(abs(actual-reference)/abs(reference)) over the reference support'

#: Probe geometries. ``absolute`` moves every element by the same amount, scaled
#: by the largest; ``relative`` moves every element by the same fraction OF
#: ITSELF and so leaves an exact zero exactly zero.
GEOMETRIES = ('absolute', 'relative')


def _finite_pair(actual, reference):
    a, r = np.asarray(actual, dtype=float), np.asarray(reference, dtype=float)
    if a.shape != r.shape:
        raise ValueError('defect requires identical shapes')
    if not (np.isfinite(a).all() and np.isfinite(r).all()):
        raise ValueError('defect requires finite arrays')
    return a, r


def scaled_max(actual, reference):
    """The recorded-error metric, identical to the provisional oracle's."""
    a, r = _finite_pair(actual, reference)
    return float(np.max(np.abs(a - r)) / max(1., float(np.max(np.abs(r)))))


def relative_max(actual, reference):
    """Largest elementwise relative change on the reference support.

    Elements that are exactly zero in the reference carry no relative scale, so
    they are excluded from the maximum and are required to stay exactly zero:
    a probe that populates them is not a relative error of this intermediate.
    """
    a, r = _finite_pair(actual, reference)
    support = r != 0.
    if not support.any():
        raise ValueError('relative defect requires a nonzero reference')
    if np.any(a[~support] != 0.):
        raise ValueError('a relative probe must leave exact zeros exactly zero')
    return float(np.max(np.abs((a[support] - r[support]) / r[support])))


def defect(actual, reference, geometry='absolute'):
    """The input/property defect in the named probe geometry."""
    if geometry == 'absolute':
        return scaled_max(actual, reference)
    if geometry == 'relative':
        return relative_max(actual, reference)
    raise ValueError(f'unknown probe geometry {geometry!r}')


def sign_direction(seed_text, size):
    """Deterministic +-1 pattern from SHA256 bytes, not a NumPy stream.

    Max-norm perturbation balls are polytopes whose worst case for a linear map
    sits at a vertex, so every probed component moves by the full budget.
    Direction 0 is the all-positive vertex; the rest are hash-derived vertices.
    """
    if type(size) is not int or size <= 0:
        raise ValueError('direction size must be a positive integer')
    out = np.empty(size)
    filled, counter = 0, 0
    while filled < size:
        digest = hashlib.sha256(f'{seed_text}|{counter}'.encode()).digest()
        bits = np.unpackbits(np.frombuffer(digest, dtype=np.uint8))
        take = min(len(bits), size - filled)
        out[filled:filled + take] = np.where(bits[:take] > 0, 1., -1.)
        filled += take
        counter += 1
    return out


#: Structural invariants that the unrelaxed production LW gate enforces on its
#: supplied data. A probe of a stage must stay on the manifold that preserves
#: them, because a probe that violates them measures the gate, not the physics.
#: The restriction is part of the reported result: an amplification is a lower
#: bound over the sampled directions *within* this manifold.
RESTRICTIONS = {'drho_c_coefficients': 'regenerated_downstream_no_restriction',
                'shape_coefficients': 'supplied_lagged_tails_held_fixed',
                'raw_tail_parameters': 'positive_exponent_supplied_cutoff_held_fixed',
                'partition_shape_samples': 'regenerated_downstream_no_restriction',
                'ov_transition_legs': 'charge_neutral_transition_legs',
                'coefficient_responses': 'symmetric_charge_null_preserving',
                'distributed_site_tensors': 'reciprocity_and_charge_sum_preserving'}


def _charge_vector(chain):
    """The auxiliary charge vector the fitted transition densities annihilate.

    The direct-OV route has no fitted auxiliary charges: its OV charges vanish
    analytically by MO orthonormality, so no projection is needed or applied.
    """
    if chain.response_basis == 'direct_ov':
        return None
    charges = np.asarray(chain.properties.ov_fit.charges, dtype=float)
    return charges if float(charges @ charges) > 0. else None


def _null_projector(charges):
    return np.eye(len(charges)) - np.outer(charges, charges) / float(charges @ charges)


def _reciprocal(raw):
    """Reciprocity-preserving part of a pair-tensor perturbation."""
    a = np.asarray(raw, dtype=float)
    return .5 * (a + a.transpose(0, 2, 1, 4, 3))


def _charge_sum_defect(raw):
    a = np.asarray(raw, dtype=float)
    return max(float(np.max(np.abs(a[:, :, :, 0, :].sum(axis=1)))),
               float(np.max(np.abs(a[:, :, :, :, 0].sum(axis=2)))))


def _pair_tensor_manifold(raw, sweeps=8):
    """Alternating projection onto reciprocal, site-charge-sum-free tensors.

    Both constraints are linear subspaces, so alternating projection converges
    to the projection onto their intersection; the caller verifies the residual
    and refuses the probe rather than accepting an off-manifold direction.
    """
    a = _reciprocal(raw)
    for _ in range(sweeps):
        a[:, :, :, 0, :] -= a[:, :, :, 0, :].mean(axis=1, keepdims=True)
        a[:, :, :, :, 0] -= a[:, :, :, :, 0].mean(axis=2, keepdims=True)
        a = _reciprocal(a)
    return a


#: Only a stage whose whole downstream is regenerated from scratch can be probed
#: relatively: the restricted stages carry linear invariants that an elementwise
#: multiplicative probe does not respect.
RELATIVE_ELIGIBLE = tuple(stage for stage, restriction in RESTRICTIONS.items()
                          if restriction == 'regenerated_downstream_no_restriction')


def restricted_direction(chain, stage, reference, index):
    """A deterministic probe direction that preserves the stage's invariants."""
    shape = np.asarray(reference).shape
    size = int(np.prod(shape))
    unit = (np.ones(size) if index == 0
            else sign_direction(f'{chain.provenance}|{stage}|direction{index}', size))
    delta = unit.reshape(shape)
    restriction = RESTRICTIONS[stage]
    if restriction == 'charge_neutral_transition_legs':
        charges = _charge_vector(chain)
        if charges is not None:
            delta = delta @ _null_projector(charges)
    elif restriction == 'symmetric_charge_null_preserving':
        delta = .5 * (delta + delta.transpose(0, 2, 1))
        charges = _charge_vector(chain)
        if charges is not None:
            projector = _null_projector(charges)
            delta = np.array([projector @ block @ projector for block in delta])
    elif restriction == 'reciprocity_and_charge_sum_preserving':
        delta = _pair_tensor_manifold(delta)
        scale = float(np.max(np.abs(delta)))
        if scale <= 0. or _charge_sum_defect(delta) > 1.e-12 * scale:
            raise RuntimeError('probe direction did not converge onto the '
                               'reciprocal charge-sum-free manifold')
    largest = float(np.max(np.abs(delta)))
    if largest <= 0.:
        raise RuntimeError('probe direction vanished under the stage restriction')
    return delta / largest


def perturb(chain, stage, reference, epsilon, index, geometry='absolute'):
    """Return an array whose defect against ``reference`` is exactly epsilon."""
    r = np.asarray(reference, dtype=float)
    if geometry not in GEOMETRIES:
        raise ValueError(f'unknown probe geometry {geometry!r}')
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError('epsilon must be finite and nonnegative')
    if type(index) is not int or index < 0:
        raise ValueError('direction index must be a nonnegative integer')
    if epsilon == 0.:
        return r.copy()
    if geometry == 'relative':
        if stage not in RELATIVE_ELIGIBLE:
            raise ValueError(f'{stage} carries linear invariants that an elementwise '
                             'relative probe does not preserve')
        unit = (np.ones(r.size) if index == 0
                else sign_direction(f'{chain.provenance}|{stage}|relative{index}', r.size))
        return r * (1. + epsilon * unit.reshape(r.shape))
    scale = epsilon * max(1., float(np.max(np.abs(r))))
    return r + scale * restricted_direction(chain, stage, r, index)


# ---------------------------------------------------------------------------
# The chain: everything needed to rebuild a shipped native result downstream
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BudgetChain:
    """Owned rebuild handles for one accepted ``NativeProperties`` result.

    The shipped result is retained only as the reference and as the source of
    the caller's own declarations; no field of it is mutated.
    """
    properties: NativeProperties = field(repr=False)
    response_basis: str
    nocc: int
    adapted: np.ndarray = field(repr=False)
    rank: int
    pair_self: bool
    partner: object = field(repr=False)
    max_order: int
    provenance: str

    @property
    def frequencies(self):
        return self.properties.frequencies

    @property
    def local(self):
        return self.properties.require_local()


def budget_chain(properties, *, max_order=12):
    """Validate that a shipped native result is rebuildable, and record how."""
    if not isinstance(properties, NativeProperties):
        raise TypeError('a factory-produced NativeProperties result is required')
    local = properties.require_local()
    if properties.context is None or properties.partition.q is None:
        raise ValueError('budget requires an accepted partition and native response context')
    basis = properties.diagnostics.get('response_basis')
    if basis not in ('fitted_auxiliary', 'direct_ov'):
        raise ValueError('shipped result does not declare a supported response basis')
    if basis == 'fitted_auxiliary' and properties.ov_fit is None:
        raise ValueError('fitted route without a recorded OV fit is not rebuildable')
    if type(max_order) is not int or not 2 <= max_order <= 12:
        raise ValueError('max_order must be an explicit integer in [2, 12]')
    dispersion = properties.dispersion
    pair_self = dispersion is not None and dispersion.model_b is local
    partner = None if dispersion is None or pair_self else dispersion.model_b
    if dispersion is not None:
        max_order = max(c.order for p in dispersion.pairs for c in p.coefficients)
    return BudgetChain(properties, basis, int(properties.partition.main.occupied.shape[1]),
                       np.array(properties.full_adapted_orbitals.array, dtype=float),
                       int(properties.partition.recipe.sites[0].rank), pair_self, partner,
                       int(max_order),
                       'rebuilt from ' + properties.partition.provenance + '; ' + properties.model)


# ---------------------------------------------------------------------------
# Rebuild stages. Each consumes ONE perturbed intermediate and the shipped rest.
# ---------------------------------------------------------------------------
def _rebuild_provenance(chain, stage):
    return lw.Provenance(f'precision-budget rebuild perturbing {stage}',
                         hashlib.sha256(f'{stage}|{chain.provenance}'.encode()).hexdigest(),
                         'Psi4 isapol_budget downstream rebuild',
                         chain.provenance + '; diagnostic perturbation, not a shipped property')


def _partitioned_multipoles(chain, shape_samples, stage):
    """Rebuild Q from shape samples exactly as the shipped chain built it."""
    partition = chain.properties.partition
    recipe = partition.recipe
    samples = np.asarray(shape_samples, dtype=float)
    total = np.sum(samples, axis=0)
    points, weights = partition.grid_points.tolist(), partition.grid_weights.tolist()
    sites = []
    for index, site in enumerate(recipe.sites):
        record = core.IsaMultipoleSamples()
        record.points, record.weights = points, weights
        record.shape, record.shape_sum = samples[index].tolist(), total.tolist()
        record.auxiliary_sites = list(range(len(recipe.sites)))
        item = core.IsaMultipoleSite()
        item.label, item.origin, item.rank, item.samples = site.label, site.origin, site.rank, record
        sites.append(item)
    provenance = partition.provenance + f'; precision-budget rebuild perturbing {stage}'
    if chain.response_basis == 'direct_ov':
        return core.IsaPartitionedMultipoles(partition.main.basis, sites, provenance,
                                             recipe.controller.density_cutoff,
                                             core.Matrix.from_array(chain.adapted), chain.nocc)
    return core.IsaPartitionedMultipoles(partition.auxiliary, sites, provenance,
                                         recipe.controller.density_cutoff)


def _from_drho(chain, coefficients, stage):
    """Perturbed Drho-C coefficients -> fixed density -> full ISA-A -> shapes."""
    partition = chain.properties.partition
    recipe = partition.recipe
    density = core.IsaFixedDensity(partition.auxiliary,
                                   np.asarray(coefficients, dtype=float).tolist())
    atomic = [s.atomic.build('AtomAux') for s in recipe.sites]
    shapes = [s.shape.build('Shape') for s in recipe.sites]
    points, weights = partition.grid_points.tolist(), partition.grid_weights.tolist()
    grids = []
    for _ in recipe.sites:
        grid = core.IsaNoTailGrid()
        grid.points, grid.weights = points, weights
        grid.density_sites = list(range(len(recipe.auxiliary.centres)))
        grid.shape_sites = list(range(len(recipe.sites)))
        grids.append(grid)
    controller = core.IsaAController(atomic, shapes, [s.shell_map for s in recipe.sites],
                                     density, grids, recipe.controller.build(recipe.sites))
    trajectory = controller.run(controller.initialize(one_gto_initialization(recipe.sites)))
    if not trajectory.state.converged:
        raise RuntimeError(f'perturbed ISA-A did not converge: {trajectory.termination}')
    return _from_shapes(chain, final_shape_samples(shapes, trajectory.state, recipe.sites, points), stage)


def _applied_tails(state, sites):
    """Indices of the tails the shipped ``final_shape_samples`` actually applies."""
    return tuple(i for i, (tail, site) in enumerate(zip(state.tails, sites))
                 if bool(state.apply_tails and site.tail_allowed and tail.defined))


def _from_tails(chain, parameters, stage):
    """Perturbed joint (amplitude, exponent) tails -> resampled final shapes.

    The cutoff is supplied configuration rather than a fitted intermediate -- the
    trajectory comparator holds it fixed at the captured endpoint radius -- so it
    is excluded from the probed array and carried through unchanged. The shipped
    controller state is never mutated: a surrogate state carries a copy of the
    shipped shape coefficients and tail switch alongside fresh tail objects.
    """
    partition = chain.properties.partition
    recipe = partition.recipe
    state = partition.trajectory.state
    applied = _applied_tails(state, recipe.sites)
    values = np.asarray(parameters, dtype=float)
    if values.shape != (len(applied), 2):
        raise ValueError('tail parameters need one (amplitude, exponent) row per applied tail')
    if not np.all(values[:, 1] > 0.):
        raise ValueError('a defined ISA-A tail requires a positive exponent')
    surrogate = core.IsaAControllerState()
    surrogate.coefficients, surrogate.apply_tails = state.coefficients, state.apply_tails
    tails = []
    for index, shipped in enumerate(state.tails):
        tail = core.IsaExponentialTail()
        tail.defined, tail.cutoff = shipped.defined, shipped.cutoff
        pair = (values[applied.index(index)] if index in applied
                else (shipped.amplitude, shipped.exponent))
        tail.amplitude, tail.exponent = float(pair[0]), float(pair[1])
        tails.append(tail)
    surrogate.tails = tails
    shapes = [s.shape.build('Shape') for s in recipe.sites]
    return _from_shapes(chain, final_shape_samples(shapes, surrogate, recipe.sites,
                                                   partition.grid_points.tolist()), stage)


def _shape_coefficient_lengths(state):
    """Per-site shape-coefficient counts of the shipped controller state."""
    return tuple(len(v) for v in state.coefficients.shape_coefficients)


def _from_shape_coefficients(chain, flat, stage):
    """Perturbed ISA-A shape coefficients -> resampled final shapes.

    The probed array is the per-site coefficient vectors concatenated in site
    order, which is also the array the reference comparison's per-site records
    reconstruct exactly, so one denominator covers the whole probe.

    The shipped tails are carried through unchanged rather than refitted. That
    is a documented property of the algorithm, not a convenience:
    ``IsaAController::step`` fits iteration n+1's tails from iteration n's shape
    coefficients (the source lag), and ``final_shape_samples`` samples the stored
    final tails without refitting at that boundary. Refitting a tail from a
    perturbed *final* W would therefore model a different algorithm rather than
    perturb this one.
    """
    partition = chain.properties.partition
    recipe = partition.recipe
    state = partition.trajectory.state
    lengths = _shape_coefficient_lengths(state)
    values = np.asarray(flat, dtype=float)
    if values.shape != (sum(lengths),):
        raise ValueError('shape coefficients need one flat entry per shipped per-site '
                         'coefficient, concatenated in site order')
    surrogate = core.IsaAControllerState()
    surrogate.tails, surrogate.apply_tails = state.tails, state.apply_tails
    coefficients = core.IsaSweepState()
    coefficients.atomic_coefficients = state.coefficients.atomic_coefficients
    edges = np.cumsum((0,) + lengths)
    coefficients.shape_coefficients = [values[a:b].tolist()
                                       for a, b in zip(edges[:-1], edges[1:])]
    surrogate.coefficients = coefficients
    shapes = [s.shape.build('Shape') for s in recipe.sites]
    return _from_shapes(chain, final_shape_samples(shapes, surrogate, recipe.sites,
                                                   partition.grid_points.tolist()), stage)


def _from_shapes(chain, shape_samples, stage):
    return _from_coupled(chain, [np.asarray(r.raw_coupled, dtype=float)
                                 for r in chain.properties.coefficient_responses], stage,
                         partition=_partitioned_multipoles(chain, shape_samples, stage))


def _from_legs(chain, legs, stage):
    """Perturbed transition legs -> re-solved full-OV response at every node."""
    provider = chain.properties.context.response.provider
    d = np.asarray(legs, dtype=float)
    solver = FDDSFullOVResponse(h1_baseline=np.asarray(provider.h1()), h2=np.asarray(provider.h2()),
                                transition_legs=d, coupling=np.zeros((d.shape[1], d.shape[1])),
                                representation=('supplied_transition_leg_coordinates'
                                                if chain.response_basis == 'direct_ov'
                                                else 'fitted_density_coefficients'))
    return _from_coupled(chain, [np.asarray(solver.at_frequency(w).raw_coupled, dtype=float)
                                 for w in chain.frequencies], stage)


def _from_coupled(chain, coupled, stage, partition=None):
    """Perturbed coefficient responses -> distributed site/component tensors."""
    if partition is None:
        partition = _partitioned_multipoles(chain, chain.properties.partition.shape_samples, stage)
    distributed = core.IsaDistributedResponse(
        partition, list(chain.frequencies),
        [core.Matrix.from_array(np.ascontiguousarray(c)) for c in coupled],
        'direct_ov' if chain.response_basis == 'direct_ov' else 'fitted_density_coefficients',
        f'precision-budget rebuild perturbing {stage}')
    n, m = len(chain.properties.partition.recipe.sites), (chain.rank + 1)**2
    raw = np.array([np.asarray(distributed.at_index(k)).reshape(n, m, n, m).transpose(0, 2, 1, 3)
                    for k in range(len(chain.frequencies))])
    return _from_raw(chain, raw, stage)


def _from_raw(chain, raw, stage):
    """Perturbed distributed tensors -> unrelaxed production LW -> dispersion."""
    local = chain.local
    result = lw.supplied_nonlocal_properties(
        labels=local.labels, origins=np.asarray(local.origins.array),
        frames=np.asarray(local.frames.array), bonds=local.bonds,
        frequencies=chain.frequencies, tensors=np.asarray(raw, dtype=float),
        input_rank=chain.rank, truncation=lw.TRUNCATE_RANK4 if chain.rank == 4 else None,
        provenance=_rebuild_provenance(chain, stage), residual_policy='production')
    dispersion = None
    if chain.pair_self or chain.partner is not None:
        quadrature = chain.properties.quadrature
        dispersion = lw.isotropic_dispersion(result, result if chain.pair_self else chain.partner,
                                             cp_weights=quadrature.cp_weights,
                                             quadrature_provenance=quadrature.provenance,
                                             max_order=chain.max_order)
    return result, dispersion


def reference_value(chain, stage):
    """The shipped value of one named intermediate, as an owned array."""
    properties = chain.properties
    if stage == 'drho_c_coefficients':
        return np.array(properties.partition.drho.coefficients, dtype=float)
    if stage == 'shape_coefficients':
        state = properties.partition.trajectory.state
        return np.concatenate([np.asarray(v, dtype=float)
                               for v in state.coefficients.shape_coefficients])
    if stage == 'raw_tail_parameters':
        state = properties.partition.trajectory.state
        applied = _applied_tails(state, properties.partition.recipe.sites)
        if not applied:
            raise ValueError('no applied ISA-A tail: the shipped shape samples do not '
                             'depend on any tail parameter here, so this probe would '
                             'measure an identity, not a sensitivity')
        return np.array([[state.tails[i].amplitude, state.tails[i].exponent]
                         for i in applied], dtype=float)
    if stage == 'partition_shape_samples':
        return np.array(properties.partition.shape_samples, dtype=float)
    if stage == 'ov_transition_legs':
        if chain.response_basis != 'fitted_auxiliary':
            raise ValueError('direct_ov legs are the declared OV coordinate identity, '
                             'not a fitted intermediate; perturbing them probes the '
                             'coordinate declaration, not a fit error')
        return np.array(properties.ov_fit.coefficients, dtype=float)
    if stage == 'coefficient_responses':
        return np.array([np.asarray(r.raw_coupled, dtype=float)
                         for r in properties.coefficient_responses])
    if stage == 'distributed_site_tensors':
        return np.array(properties.pair_tensors.array, dtype=float)
    raise ValueError(f'unknown intermediate {stage!r}')


def rebuild(chain, stage, value):
    """Rebuild every stage downstream of ``stage`` from a supplied value."""
    if stage == 'drho_c_coefficients':
        return _from_drho(chain, value, stage)
    if stage == 'shape_coefficients':
        return _from_shape_coefficients(chain, value, stage)
    if stage == 'raw_tail_parameters':
        return _from_tails(chain, value, stage)
    if stage == 'partition_shape_samples':
        return _from_shapes(chain, value, stage)
    if stage == 'ov_transition_legs':
        return _from_legs(chain, value, stage)
    if stage == 'coefficient_responses':
        return _from_coupled(chain, list(np.asarray(value, dtype=float)), stage)
    if stage == 'distributed_site_tensors':
        return _from_raw(chain, value, stage)
    raise ValueError(f'unknown intermediate {stage!r}')


# ---------------------------------------------------------------------------
# Properties of interest
# ---------------------------------------------------------------------------
def property_groups(local, dispersion=None):
    """Named property vectors: isotropic site polarizabilities and C_n.

    Grouping keeps the defect metric identical to the recorded-error metric:
    one max over the group against the group's own scale.
    """
    groups = {}
    scalars = np.asarray(local.atomic_scalars.array, dtype=float)
    for rank in range(scalars.shape[2]):
        groups[f'alpha_iso_rank{rank + 1}'] = scalars[:, :, rank].reshape(-1).copy()
    if dispersion is not None:
        orders = sorted({c.order for pair in dispersion.pairs for c in pair.coefficients})
        for order in orders:
            groups[f'C{order}'] = np.array([c.value for pair in dispersion.pairs
                                            for c in pair.coefficients if c.order == order])
    return groups


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StageProbe:
    stage: str
    direction: int
    epsilon: float
    input_defect: float
    property_defects: dict
    status: str
    message: str = ''
    restriction: str = 'none'
    geometry: str = 'absolute'


@dataclass(frozen=True)
class StageAmplification:
    stage: str
    property_name: str
    epsilon: float
    amplification: float
    amplification_half: float
    linearity_defect: float
    quoted: bool
    direction: int
    restriction: str = 'none'
    geometry: str = 'absolute'


@dataclass(frozen=True)
class PrecisionRequirement:
    stage: str
    property_name: str
    property_tolerance: float
    amplification: float
    required_precision: float
    quoted: bool
    recorded_error: object = None
    satisfied: object = None
    note: str = 'measured directional lower bound; necessary, not sufficient'
    restriction: str = 'none'
    geometry: str = 'absolute'


@dataclass(frozen=True)
class PrecisionBudget:
    metric: str
    epsilon: float
    directions: int
    linearity_tolerance: float
    baseline_scale: dict
    self_consistency: dict
    probes: tuple
    amplifications: tuple
    requirements: tuple
    provenance: str
    geometry: str = 'absolute'
    status: str = STATUS


def _defects(baseline, local, dispersion):
    rebuilt = property_groups(local, dispersion)
    if set(rebuilt) != set(baseline):
        raise ValueError('rebuilt property groups do not match the shipped groups')
    return {name: scaled_max(rebuilt[name], baseline[name]) for name in baseline}


def _probe(chain, stage, baseline, epsilon, direction, reference, geometry='absolute'):
    """One rebuild at one probe size, in one restricted direction.

    The INPUT defect is measured in the probe geometry; the PROPERTY defect is
    always the absolute metric, because a property group shares one scale and is
    compared against a tolerance stated in that same scale.
    """
    restriction = (RESTRICTIONS[stage] if epsilon > 0. else 'unperturbed') \
        if geometry == 'absolute' else ('elementwise_relative' if epsilon > 0. else 'unperturbed')
    try:
        value = perturb(chain, stage, reference, epsilon, direction, geometry)
    except Exception as exc:
        return StageProbe(stage, direction, epsilon, 0., {},
                          'direction_rejected:' + type(exc).__name__, str(exc),
                          restriction, geometry)
    input_defect = defect(value, reference, geometry)
    try:
        local, dispersion = rebuild(chain, stage, value)
    except Exception as exc:
        return StageProbe(stage, direction, epsilon, input_defect, {},
                          'rebuild_rejected:' + type(exc).__name__, str(exc),
                          restriction, geometry)
    return StageProbe(stage, direction, epsilon, input_defect,
                      _defects(baseline, local, dispersion), 'measured', '',
                      restriction, geometry)


def precision_budget(chain, *, property_tolerances, stages=STAGES, epsilon=1.e-6,
                     directions=2, linearity_tolerance=0.1, recorded_errors=None,
                     geometry='absolute', recorded_error_metric='absolute'):
    """Measure how precisely each named intermediate must be known.

    ``property_tolerances`` is a single float applied to every property group or
    a mapping from group name to tolerance. ``recorded_errors`` optionally maps a
    stage to its measured error so the report states whether the recorded error
    already meets the derived requirement. Every rebuild runs the production LW
    policy; a rejected rebuild is recorded as rejected and quotes nothing.

    ``geometry`` selects the probe geometry, and with it the metric the derived
    requirement is stated in. ``recorded_error_metric`` names the metric the
    supplied ``recorded_errors`` were measured in; supplying errors measured in
    one metric while probing in the other is refused rather than silently
    compared, because the two numbers are not commensurable.
    """
    if not isinstance(chain, BudgetChain):
        raise TypeError('an isapol_budget.budget_chain result is required')
    if geometry not in GEOMETRIES:
        raise ValueError(f'unknown probe geometry {geometry!r}')
    if recorded_error_metric not in GEOMETRIES:
        raise ValueError(f'unknown recorded-error metric {recorded_error_metric!r}')
    stages = tuple(stages)
    if not stages or any(s not in STAGES for s in stages) or len(set(stages)) != len(stages):
        raise ValueError('stages must be distinct names drawn from STAGES')
    if not np.isfinite(epsilon) or not 0. < epsilon < 1.:
        raise ValueError('epsilon must be a finite probe size in (0, 1)')
    if type(directions) is not int or not 1 <= directions <= 32:
        raise ValueError('directions must be an integer in [1, 32]')
    if not np.isfinite(linearity_tolerance) or linearity_tolerance <= 0:
        raise ValueError('linearity_tolerance must be positive')
    baseline = property_groups(chain.local, chain.properties.dispersion)
    if isinstance(property_tolerances, dict):
        tolerances = dict(property_tolerances)
        if set(tolerances) - set(baseline):
            raise ValueError('tolerance named for an unknown property group')
    else:
        tolerances = {name: float(property_tolerances) for name in baseline}
    for name, value in tolerances.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'tolerance for {name} must be positive and finite')
    recorded = dict(recorded_errors or {})
    if set(recorded) - set(STAGES):
        raise ValueError('recorded error named for an unknown intermediate')
    if recorded and recorded_error_metric != geometry:
        raise ValueError(f'recorded errors measured in the {recorded_error_metric} metric '
                         f'cannot be compared against a {geometry}-geometry requirement')

    probes, amplifications, self_consistency = [], [], {}
    for stage in stages:
        reference = reference_value(chain, stage)
        zero = _probe(chain, stage, baseline, 0., 0, reference, geometry)
        self_consistency[stage] = (dict(zero.property_defects) if zero.status == 'measured'
                                   else {'status': zero.status, 'message': zero.message})
        probes.append(zero)
        if zero.status != 'measured':
            continue
        for direction in range(directions):
            full = _probe(chain, stage, baseline, epsilon, direction, reference, geometry)
            half = _probe(chain, stage, baseline, .5 * epsilon, direction, reference, geometry)
            probes.extend((full, half))
            if full.status != 'measured' or half.status != 'measured':
                continue
            for name in baseline:
                a = full.property_defects[name] / full.input_defect
                b = half.property_defects[name] / half.input_defect
                scale = max(a, b)
                linearity = 0. if scale == 0. else abs(a - b) / scale
                amplifications.append(StageAmplification(stage, name, epsilon, a, b, linearity,
                                                         linearity <= linearity_tolerance,
                                                         direction, full.restriction, geometry))

    requirements = []
    insensitive = ('no property change in any probed direction: this property is '
                   'structurally insensitive to this intermediate in this molecule')
    for stage in stages:
        for name in baseline:
            candidates = [a for a in amplifications if a.stage == stage and a.property_name == name]
            quotable = [a for a in candidates if a.quoted]
            if not quotable:
                continue
            best = max(quotable, key=lambda a: a.amplification)
            required = (float('inf') if best.amplification == 0.
                        else tolerances[name] / best.amplification)
            error = recorded.get(stage)
            requirements.append(PrecisionRequirement(
                stage, name, tolerances[name], best.amplification, required, True,
                None if error is None else float(error),
                None if error is None else bool(float(error) <= required),
                insensitive if best.amplification == 0. else PrecisionRequirement.note,
                best.restriction, geometry))
    scale = {name: float(np.max(np.abs(v))) for name, v in baseline.items()}
    return PrecisionBudget(METRIC if geometry == 'absolute' else RELATIVE_METRIC,
                           float(epsilon), int(directions), float(linearity_tolerance),
                           scale, self_consistency, tuple(probes), tuple(amplifications),
                           tuple(requirements), chain.provenance, geometry)
