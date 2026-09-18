# Psi4: Copyright (c) 2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""End-to-end PFIT refinement of native localized tensors, from one wavefunction.

``isapol_native.native_properties`` produces localized per-site polarizabilities
that *anchor* a refinement; ``isapol_refine`` owns the model and the solver.
Neither of them generates the data being fitted, chooses a point cloud or walks
the frequency grid, and that orchestration is what this module is: the stage
that turns an accepted :class:`isapol_native.NativeProperties` into one refined
model per Casimir-Polder node, and optionally into the refined dispersion
coefficients.

The chain is CamCASP's own, in CamCASP's order:

1. the fit-point cloud, ``core.FitPoints`` over the certified Maclaren draw
   (``SET Lattice / LoLim / HiLim / Random / Seed / END``),
2. the point-to-point response on that cloud,
   ``isapol_native_point_response.native_point_charge_response``, which is what
   CamCASP's ``localize`` writes to a ``.p2p`` file,
3. one ``pfit`` solve per frequency against those targets, anchored on the
   localized tensors, and
4. ``casimir`` over the refined tensors.

What this module deliberately does not do.  It does not run an SCF, does not
localize, does not build a partition, does not choose rank limits or site types
(the caller declares both, per site type, exactly as a ``.pdef`` does), does not
invent a quadrature, and does not decide that a node whose solver status is not
``Solved`` should be dropped -- every node is returned with its status, because
a rank-deficient refinement is a fact about the model.  It also does not
recompute anything ``native_properties`` already produced: the response handle,
the localized tensors, the frames and the origins are read off the result.

The one numerical subtlety it owns is the anchor offset.  ``LocalProperties.raw_local``
is indexed ``frequency, site, response_component, potential_component`` over
ranks ``1..limit`` -- rank 0 is *absent*, because a localized polarizability has
no charge-flow block -- while a refinement site's tensor is indexed over the
full Racah packing ``00, 10, 11c, ...``.  The anchor is therefore the localized
block placed at ``[1:, 1:]`` with a zero rank-0 row and column: the charge-flow
variables start from zero and are found by the fit, which is what CamCASP's
``.pdef`` declares when it lists ``00_00`` among the variables with no anchor.
"""

from dataclasses import dataclass, field
import hashlib

import numpy as np

from psi4 import core

from . import isapol_logging as _lg
from . import isapol_refine as _refine
from .isapol_native_point_response import MAXIMUM_POINTS, native_point_charge_response


#: CamCASP's declared lattice, as ``H2O_aTZ.cks`` writes it.  The C++ defaults
#: are the 2000-point production cloud, which ``MAXIMUM_POINTS`` refuses; the
#: properties protocol's ``Random 500`` is what fits, so it is what is declared
#: here rather than a silently truncated 2000.
DEFAULT_POINTS = 500
DEFAULT_SEED = 1
DEFAULT_LOWER_LIMIT = 2.0
DEFAULT_UPPER_LIMIT = 4.0

#: ``FitPointsOptions`` cutoffs are multiples of the van der Waals radius, not
#: bohr.  Named here because 2.0 read as bohr is a different shell entirely.
LIMIT_UNITS = 'multiples of the van der Waals radius'

GENERATOR = ('core.FitPoints; Maclaren 1992 lagged-Fibonacci draw, bitwise certified '
             'against CamCASP sdprnd/dprand')


@dataclass(frozen=True)
class RefinementLattice:
    """The accepted fit-point cloud, identified by its own hash.

    The points are bulk and the hash is the reproducibility statement: a
    refinement is only comparable point-for-point against the same cloud, and
    the same ``seed`` with the same cutoffs is the only thing that produces one.
    """
    points_bohr: np.ndarray = field(repr=False)
    npoints: int
    ncandidates: int
    dmax: float
    centre_bohr: tuple
    sha256: str
    seed: int
    lower_limit: float
    upper_limit: float
    limit_units: str = LIMIT_UNITS
    generator: str = GENERATOR
    declared_by: str = 'SET Lattice / Charge 1.0 / LoLim / HiLim / Random / Seed / END'


@dataclass(frozen=True)
class NativeRefinement:
    """One refined model per frequency node, with the data it was fitted to."""
    lattice: RefinementLattice
    targets: object = field(repr=False)
    refinements: tuple = field(repr=False)
    frequencies: tuple
    site_types: tuple
    rank_limits: tuple
    dispersion: object = None
    source_id: str = ''
    solver_status: tuple = ()

    def __post_init__(self):
        for name in ('refinements', 'frequencies', 'site_types', 'rank_limits',
                     'solver_status'):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    @property
    def solved(self):
        """True only if every node's solver reported ``Solved``."""
        return bool(self.solver_status) and all(s == 'Solved' for s in self.solver_status)

    @property
    def refined_tensors(self):
        """``(node, site)`` refined local tensors, in the caller's node order."""
        return tuple(r.refined_tensors for r in self.refinements)


def refinement_lattice(molecule, *, npoints=DEFAULT_POINTS, seed=DEFAULT_SEED,
                       lower_limit=DEFAULT_LOWER_LIMIT, upper_limit=DEFAULT_UPPER_LIMIT,
                       log=None, wfn=None):
    """Draw the declared fit-point cloud around ``molecule``.

    ``lower_limit`` rejects a candidate closer than ``lower_limit*R_vdW(k)`` to
    any atom ``k``; ``upper_limit`` rejects one farther than
    ``upper_limit*R_vdW(k)`` from every atom ``k``.  Both are in
    :data:`LIMIT_UNITS`.  ``npoints`` is bounded by the point-response cap:
    the refinement consumes an ``npoint x npoint`` target matrix, so a cloud
    larger than ``MAXIMUM_POINTS`` is refused here rather than silently
    truncated downstream.
    """
    if not isinstance(npoints, (int, np.integer)) or isinstance(npoints, (bool, np.bool_)) \
            or not 1 <= int(npoints) <= MAXIMUM_POINTS:
        raise ValueError(f'npoints must be an integer in 1..{MAXIMUM_POINTS}; the '
                         'point-to-point target matrix is npoint x npoint')
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)) \
            or int(seed) <= 0:
        raise ValueError('seed must be a positive integer; the cloud is a pseudorandom '
                         'sample and the seed is part of the declared model')
    lower, upper = float(lower_limit), float(upper_limit)
    if not np.isfinite([lower, upper]).all() or not 0. < lower < upper:
        raise ValueError(f'0 < lower_limit < upper_limit is required, in {LIMIT_UNITS}')
    log = _lg.silent() if log is None else log
    log.stage('refinement fit-point lattice', _lg.refinement_lattice_parameters(
        npoints=int(npoints), seed=int(seed), lower_limit=lower, upper_limit=upper,
        maximum_points=MAXIMUM_POINTS, generator=GENERATOR,
        declared_by=RefinementLattice.declared_by))
    options = core.FitPointsOptions()
    options.npoints, options.seed = int(npoints), int(seed)
    options.lolim, options.hilim = lower, upper
    cloud = core.FitPoints(molecule.clone(), options)
    points = np.ascontiguousarray(
        np.column_stack([cloud.x(), cloud.y(), cloud.z()]), dtype=float)
    points.flags.writeable = False
    lattice = RefinementLattice(
        points_bohr=points, npoints=int(points.shape[0]),
        ncandidates=int(cloud.ncandidates()), dmax=float(cloud.dmax()),
        centre_bohr=tuple(map(float, cloud.centre())),
        sha256=hashlib.sha256(np.ascontiguousarray(points, dtype='<f8').tobytes()).hexdigest(),
        seed=int(seed), lower_limit=lower, upper_limit=upper)
    _lg.report_refinement_lattice(log, wfn, lattice)
    log.stage_end()
    return lattice


def refinement_sites(local, *, site_types, rank_limits):
    """Declared refinement sites from an accepted localization.

    ``site_types`` is the caller's declaration, one per site in site order, and
    is what drives CamCASP's COPY equivalence: two sites of one type share a
    single variable set, read off the first of them.  It is not inferred from
    the site labels here -- ``O1``/``H2`` happen to start with their element
    symbol in the generated recipe, but that is a property of that recipe, not a
    rule -- and ``rank_limits`` maps each declared type to its rank limit,
    exactly as a ``.pdef``'s ``Limit rank to 2`` plus ``Limit rank to 1 for
    sites H1 H2`` does.
    """
    labels = tuple(local.labels)
    types = tuple(site_types)
    if len(types) != len(labels):
        raise ValueError('site_types must declare one type per localized site, in site order')
    if any(not isinstance(t, str) or not t.strip() for t in types):
        raise ValueError('each site type must be a nonblank string')
    missing = sorted(set(types) - set(rank_limits))
    if missing:
        raise ValueError('every declared site type needs a declared rank limit; '
                         f'missing {missing}')
    origins = np.asarray(local.origins.array, dtype=float)
    frames = np.asarray(local.frames.array, dtype=float)
    return tuple(_refine.RefinementSite(
        label=label, site_type=types[index], origin_bohr=tuple(origins[index]),
        frame=tuple(map(tuple, frames[index])), rank_limit=int(rank_limits[types[index]]))
        for index, label in enumerate(labels))


def anchor_tensors(local, sites, node):
    """Localized tensors of one frequency node, placed as refinement anchors.

    ``raw_local`` runs over ranks ``1..limit`` with no rank-0 block, so the
    anchor is that block at ``[1:, 1:]`` and zero on the rank-0 row and column:
    the charge-flow variables are found by the fit, not carried in.  A site
    limited below the localization is truncated here rather than zero-padded,
    which is the same statement -- those components are absent by declaration.
    """
    raw = np.asarray(local.raw_local.array, dtype=float)
    if not 0 <= int(node) < raw.shape[0]:
        raise ValueError('node index outside the localized frequency grid')
    anchors = []
    for index, site in enumerate(sites):
        ncomp = site.component_count
        block = raw[int(node)][index]
        if block.shape[0] < ncomp - 1:
            raise ValueError(f'site {site.label} declares rank limit {site.rank_limit}, '
                             'above what the localization produced')
        anchor = np.zeros((ncomp, ncomp))
        anchor[1:, 1:] = block[:ncomp - 1, :ncomp - 1]
        anchors.append(anchor)
    return tuple(anchors)


def native_refinement(properties, wfn, *, site_types, rank_limits,
                      npoints=DEFAULT_POINTS, seed=DEFAULT_SEED,
                      lower_limit=DEFAULT_LOWER_LIMIT, upper_limit=DEFAULT_UPPER_LIMIT,
                      weight_type=4, weight_coefficient=1.0e-3, cutoff=1.0e-4,
                      damping=0.0, dispersion=False, max_order=12, site_ranks=None,
                      declared_variables=None, options=None, log=None, source_id=None):
    """Refine an accepted native result on its own frequency grid.

    Every node is refined against the *same* cloud and the same T-function
    fields, which are built once: the fields depend only on the geometry of the
    cloud relative to the sites, so recomputing them per node would be the same
    matrix at a cost of one solve each.  The anchors, the targets and the
    penalty weights are what change with frequency.

    ``dispersion`` requires ``properties.quadrature``: the CP weights belong to
    the quadrature that produced the frequency grid, and this module will not
    invent a set for a grid it did not choose.  The refined coefficients come
    back on the result under their own names and are never mixed with the
    unrefined ones in ``properties.dispersion`` -- the two are different models.
    """
    local = properties.require_local()
    frequencies = tuple(float(x) for x in properties.frequencies)
    if tuple(float(x) for x in local.frequencies) != frequencies:
        raise ValueError('the localized grid and the produced grid disagree; '
                         'a refinement cannot straddle two grids')
    response = properties.context.response
    sites = refinement_sites(local, site_types=site_types, rank_limits=rank_limits)
    log = _lg.silent() if log is None else log

    lattice = refinement_lattice(wfn.molecule(), npoints=npoints, seed=seed,
                                 lower_limit=lower_limit, upper_limit=upper_limit,
                                 log=log, wfn=wfn)
    points = lattice.points_bohr
    log.stage('point-to-point response targets', (
        ('points', lattice.npoints), ('points sha256', lattice.sha256),
        ('frequency nodes', len(frequencies)),
        ('producer', 'isapol_native_point_response.native_point_charge_response'),
        ('reused context', 'the accepted native response; H1/H2 are not recomputed'),
        ('maximum points', MAXIMUM_POINTS)))
    targets = native_point_charge_response(response, wfn, points, frequencies=frequencies)
    _lg.report_refinement_targets(log, targets)
    log.stage_end()

    declared_source = source_id if source_id is not None else (
        f'native {properties.distributed.partition.representation} actual point-charge '
        f'response on a Random {lattice.npoints}/Seed {lattice.seed} lattice '
        f'(sha256={lattice.sha256}); anchors from the accepted native localization '
        f'at rank limit {local.metadata.localization_rank_limit}')
    fields, refinements = None, []
    for node, frequency in enumerate(frequencies):
        model = _refine.refinement_model(
            sites, anchor_tensors(local, sites, node), frequency_au=frequency,
            cutoff=cutoff, weight_type=weight_type, weight_coefficient=weight_coefficient,
            declared_variables=declared_variables,
            provenance=(f'declared {"/".join(f"{t}{rank_limits[t]}" for t in dict.fromkeys(site_types))} '
                        f'variable set on native localized tensors at omega={frequency:.12g} au; '
                        f'weight {weight_type}/{weight_coefficient!r}'))
        if fields is None:
            fields = _refine.channel_fields(points, model, damping=damping)
        refinements.append(_refine.refine(
            model, points, targets.packed_targets[node], damping=damping, fields=fields,
            target_origin=core.IsaPfitTargetOrigin.NativeDirectActualPointResponse,
            response_representation=targets.representation, source_id=declared_source,
            generation_record=targets.generation_record, options=options, log=log, wfn=wfn))

    coefficients = None
    if dispersion:
        quadrature = properties.quadrature
        if quadrature is None:
            raise ValueError('refined dispersion needs the quadrature that produced the '
                             'frequency grid; no CP weights are invented here')
        coefficients = _refine.refined_isotropic_dispersion(
            refinements, cp_weights=quadrature.cp_weights,
            quadrature_provenance=quadrature.provenance, max_order=max_order,
            site_ranks=site_ranks, log=log, wfn=wfn)
    return NativeRefinement(
        lattice=lattice, targets=targets, refinements=tuple(refinements),
        frequencies=frequencies, site_types=tuple(site_types),
        rank_limits=tuple(int(rank_limits[t]) for t in site_types),
        dispersion=coefficients, source_id=declared_source,
        solver_status=tuple(str(r.status).rsplit('.', 1)[-1] for r in refinements))
