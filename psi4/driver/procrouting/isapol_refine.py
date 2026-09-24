# Psi4: Copyright (c) 2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""Refinement of local polarizabilities against a point-to-point response target.

This is the stage that stands between raw per-site distributed polarizabilities
and a refined model: the raw tensors only *anchor* the fit, and the data being
fitted is the point-to-point response ``v(i,j)`` between unit source charges at
a point cloud.  The algebra, the model construction and the penalty convention
are transcribed from CamCASP 6.0 by Alston J. Misquitta and Anthony J. Stone
(http://gitlab.com/anthonyjstone/camcasp), MIT licensed, and are referenced
inline by file and line:

* ``src/tools/process_data.F90::write_pfit_local_symm`` (1894-2391) builds the
  variable list, the ``.pdef`` COPY equivalences and the ``Penalties`` block.
* ``src/tools/process_data.F90::weights`` (1810-1876) is ``penalty_weight``.
* ``src/tools/process_data.F90::index_to_rank`` (1878-1892) is
  ``component_rank``, shifted to a zero-based component index.
* ``src/pfit/process.F90::setup``/``solve`` is the least-squares problem that
  :func:`refinement_problem` hands to the owned C++ solver; the fields are
  ``core.isa_t_functions``, which is bitwise certified against CamCASP's own
  compiled ``solidh``.

No CamCASP source is executed, linked or vendored here.

What this module deliberately does not do: it does not generate the target
response (see ``isapol_native_point_response``), does not localize, does not
choose rank limits or site types for the caller, does not symmetrize or repair
the anchor tensors, and does not iterate the fit.  Components excluded by the
cutoff are absent from the model and therefore exactly zero in the refined
tensors, exactly as they are absent from a CamCASP ``.pdef``; they are not
carried through from the anchors.

Conventions.  Multipole components are real Racah ``00,10,11c,11s,20,...``
with no Condon-Shortley phase, the ordering of
``core.isa_irregular_solid_harmonics``.  Site frames are proper local-to-global
Cartesian columns, the convention of ``core.isa_multipole_rotation`` and of
CamCASP's ``sm(i,j,s)``; local polarizability components are in those local
axes.  Targets follow the PFIT sign convention
``v_pq = -d(phi_induced at R_p)/d(q at R_q)`` in Eh/e^2 with no energy 1/2
factor, and are packed at ``i*(i+1)//2 + j`` for every ``j <= i``.  Frequencies
are nonnegative imaginary-axis magnitudes in atomic units.
"""
from dataclasses import dataclass, field
import hashlib
from typing import Optional, Sequence

import numpy as np

from psi4 import core

from . import isapol_logging as _lg
from . import isapol_lw as _lw

#: Racah component names, CamCASP ``comp_name`` (process_data.F90:1938-1942).
COMPONENT_NAMES = ('00', '10', '11c', '11s',
                   '20', '21c', '21s', '22c', '22s',
                   '30', '31c', '31s', '32c', '32s', '33c', '33s',
                   '40', '41c', '41s', '42c', '42s', '43c', '43s', '44c', '44s')

#: Highest rank the certified T functions cover.
MAX_RANK = 4
#: Guard rails; not physics.  Rows grow as npoint*(npoint+1)/2.
MAX_SITES = 64
MAX_POINTS = 512
MAX_PARAMETERS = 4096

#: CamCASP's weight index (the ``WEIGHT`` key of a ``.prss`` file).
WEIGHT_TYPES = (0, 1, 2, 3, 4, 5, 6)

TARGET_CONVENTION = 'v_pq = -d(phi_induced at R_p)/d(q at R_q); atomic units Eh/e^2'


def component_count(rank):
    """Number of Racah components at ranks 0..``rank`` inclusive."""
    if not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank <= MAX_RANK:
        raise ValueError(f'rank must be an int in [0,{MAX_RANK}]')
    return (rank + 1) ** 2


def component_rank(index):
    """Rank owning zero-based component ``index``.

    CamCASP ``index_to_rank`` (process_data.F90:1878-1892) takes a one-based
    index and returns the smallest ``l`` with ``(l+1)**2 >= indx``; this is the
    same function on ``index+1``, so index 0 is rank 0 and indices 1..3 rank 1.
    """
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError('component index must be a nonnegative int')
    for l in range(MAX_RANK + 1):
        if (l + 1) ** 2 >= index + 1:
            return l
    raise ValueError(f'component index {index} exceeds rank {MAX_RANK}')


def penalty_weight(*, weight_type, weight_coefficient, alpha, frequency, rank1, rank2):
    """CamCASP ``weights`` (process_data.F90:1810-1876), transcribed exactly.

    ``alpha`` is the anchor value of the parameter, ``rank1``/``rank2`` the
    ranks of its two components.  The literals ``10.0e-3`` and ``10.0e-2`` of
    cases 5 and 6 are upstream's and are reproduced as written, including the
    fact that they are ten times the values the upstream comments claim.  An
    illegal weight index stops CamCASP; here it raises.
    """
    if weight_type not in WEIGHT_TYPES:
        raise ValueError(f'weight_type must be one of {WEIGHT_TYPES}')
    for name, value in (('weight_coefficient', weight_coefficient), ('alpha', alpha),
                        ('frequency', frequency)):
        if not np.isfinite(value):
            raise ValueError(f'{name} must be finite')
    if weight_coefficient < 0.0:
        raise ValueError('weight_coefficient must be nonnegative for a PSD penalty')
    if frequency < 0.0:
        raise ValueError('frequency must be a nonnegative imaginary-axis magnitude')
    low = rank1 <= 1 and rank2 <= 1
    if weight_type == 0:
        weight = 0.0
    elif weight_type == 1:
        weight = weight_coefficient
    elif weight_type == 2:
        weight = weight_coefficient / (abs(alpha) + 1)
    elif weight_type == 3:
        weight = weight_coefficient / (alpha * alpha + 1)
    elif weight_type == 4:
        weight = weight_coefficient if low else 0.0
    elif weight_type == 5:
        weight = weight_coefficient if low else weight_coefficient * 10.0e-3
    else:
        weight = weight_coefficient if low else weight_coefficient * 10.0e-2
    if frequency != 0.0:
        weight = weight / (1.0 + frequency * frequency)
    if not np.isfinite(weight):
        raise ValueError('nonfinite penalty weight')
    return float(weight)


def _label(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 64:
        raise ValueError(f'{name} must be a short nonempty string')
    return value


def _vector(value, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f'{name} must be three finite floats')
    return tuple(map(float, array))


def _frame(value, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (3, 3) or not np.all(np.isfinite(array)):
        raise ValueError(f'{name} must be a finite 3x3 matrix')
    if np.max(np.abs(array.T @ array - np.eye(3))) > 1e-12 or abs(np.linalg.det(array) - 1.0) > 1e-12:
        raise ValueError(f'{name} must be proper orthogonal to 1e-12 (local-to-global columns)')
    return tuple(tuple(map(float, row)) for row in array)


@dataclass(frozen=True)
class RefinementSite:
    """One expansion site: where it is, how it is oriented, and how far it goes.

    ``site_type`` drives the COPY equivalence exactly as CamCASP's
    ``sitetype`` does: all sites sharing a type share one set of variables,
    expressed in each site's own local axes, and the type's parameters are read
    off the *first* site of that type in input order
    (``find_sites_with_type`` fills ``indices`` in ascending site order,
    molecule_operations_cluster.F90:423-431, and ``RefSiteIndx=indices(1)``).
    ``rank_limit`` is CamCASP's per-site ``lim``; a limit of zero contributes
    no variables at all (process_data.F90:2105).
    """
    label: str
    site_type: str
    origin_bohr: tuple
    frame: tuple
    rank_limit: int

    def __post_init__(self):
        object.__setattr__(self, 'label', _label(self.label, 'label'))
        object.__setattr__(self, 'site_type', _label(self.site_type, 'site_type'))
        object.__setattr__(self, 'origin_bohr', _vector(self.origin_bohr, 'origin_bohr'))
        object.__setattr__(self, 'frame', _frame(self.frame, 'frame'))
        if not isinstance(self.rank_limit, int) or isinstance(self.rank_limit, bool) \
                or not 0 <= self.rank_limit <= MAX_RANK:
            raise ValueError(f'rank_limit must be an int in [0,{MAX_RANK}]')

    @property
    def component_count(self):
        return (self.rank_limit + 1) ** 2


@dataclass(frozen=True)
class RefinementModel:
    """Variables, COPY equivalences, anchors and penalty strengths.

    ``parameter_entries[k]`` lists every ``(site, row, col)`` the variable
    occupies, one triple per site of the owning type, with ``row <= col``.  The
    channel vector concatenates each site's components in site order, so site
    ``s`` owns ``channel_offsets[s] : channel_offsets[s]+component_count``.
    ``anchors[k]`` is the reference site's raw value, and ``strengths[k]`` the
    weight of the ``strengths[k]*(z-anchor)**2`` penalty term; both are the two
    numbers CamCASP prints on a ``Penalties`` line (process_data.F90:2273-2275).

    ``copy_anchor_discrepancy`` is how far the COPY declaration is from the
    caller's own anchors: the largest ``abs(anchor_tensors[s][row,col] -
    anchors[k])`` over every variable and every *equivalent* site it occupies.
    It is zero exactly when the caller's frames really do make each type's
    sites equivalent.  A nonzero value means the reference site's value is
    being imposed on sites whose own tensors disagree -- which is what CamCASP
    does too, so it is measured and reported here rather than repaired or
    refused.  A refinement whose penalty pins the parameters to the anchors
    cannot then reproduce the equivalent sites' anchors, and misses them by up
    to this much.

    ``declared_variables`` is empty when the variable list was derived from the
    cutoff, and is the caller's own ``.pdef`` variable list when one was
    supplied.  ``unpenalized_variables`` names the declared variables the
    cutoff screens out, which are free parameters carrying anchor 0 and
    strength 0 -- CamCASP's own split, where ``.pdef`` sets the variable list
    and the ``Penalties`` block ``process`` writes covers only the above-cutoff
    subset.  A model with a nonempty ``declared_variables`` is a DIFFERENT
    DECLARED MODEL from the cutoff-derived one, even where the two share every
    penalized variable.
    """
    sites: tuple
    site_types: tuple
    reference_sites: tuple
    equivalent_sites: tuple
    channel_offsets: tuple
    channel_count: int
    channel_labels: tuple
    parameter_labels: tuple
    parameter_entries: tuple
    anchors: tuple
    strengths: tuple
    cutoff: float
    weight_type: int
    weight_coefficient: float
    frequency_au: float
    nonsymmetric_parameter_count: int
    anchor_sha256: str
    copy_anchor_discrepancy: float
    provenance: str
    declared_variables: tuple = ()
    unpenalized_variables: tuple = ()

    @property
    def parameter_count(self):
        return len(self.parameter_labels)


def _validated_sites(sites):
    sites = tuple(sites)
    if not 1 <= len(sites) <= MAX_SITES:
        raise ValueError(f'between 1 and {MAX_SITES} sites are required')
    if any(not isinstance(s, RefinementSite) for s in sites):
        raise ValueError('every site must be a RefinementSite')
    if len({s.label for s in sites}) != len(sites):
        raise ValueError('site labels must be unique')
    for name in ('rank_limit',):
        limits = {}
        for s in sites:
            other = limits.setdefault(s.site_type, getattr(s, name))
            if other != getattr(s, name):
                raise ValueError(f'sites of type {s.site_type} disagree on {name}; '
                                 'a COPY equivalence cannot span different rank limits')
    return sites


def _validated_anchor_tensors(sites, tensors):
    tensors = tuple(np.array(t, dtype=float, copy=True) for t in tensors)
    if len(tensors) != len(sites):
        raise ValueError('one anchor tensor per site is required')
    for site, tensor in zip(sites, tensors):
        n = site.component_count
        if tensor.shape != (n, n):
            raise ValueError(f'site {site.label} needs a {n}x{n} anchor tensor')
        if not np.all(np.isfinite(tensor)):
            raise ValueError(f'site {site.label} anchor tensor must be finite')
        tensor.flags.writeable = False
    return tensors


def _anchor_hash(tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(np.ascontiguousarray(tensor, dtype='<f8').tobytes())
    return digest.hexdigest()


def _validated_declared_variables(declared, sites, ordered_types, first):
    """Resolve a ``.pdef`` variable list to ``{(site_type, row, col): label}``.

    Each label is ``<reference site>_<component>_<component>_A``, the name
    CamCASP's ``.pdef`` and ``pfit`` both print, and names a component of that
    type's reference site with ``row <= col``.  A label for a non-reference
    site is refused: the ``COPY`` declaration means the equivalent sites do not
    carry variables of their own.
    """
    if declared is None:
        return None
    declared = tuple(declared)
    if not declared:
        raise ValueError('declared_variables must name at least one variable')
    if len(set(declared)) != len(declared):
        raise ValueError('declared_variables must not repeat a variable')
    reference_labels = {sites[first[t]].label: t for t in ordered_types}
    index_of = {name: i for i, name in enumerate(COMPONENT_NAMES)}
    resolved = {}
    for label in declared:
        if not isinstance(label, str) or not label.endswith('_A'):
            raise ValueError(f'declared variable {label!r} is not a <site>_<i>_<j>_A name')
        owners = [(name, t) for name, t in reference_labels.items()
                  if label.startswith(name + '_')]
        if len(owners) != 1:
            raise ValueError(f'declared variable {label!r} names no single reference '
                             f'site; the reference sites are '
                             f'{sorted(reference_labels)}')
        name, site_type = owners[0]
        parts = label[len(name) + 1:-2].split('_')
        if len(parts) != 2 or any(q not in index_of for q in parts):
            raise ValueError(f'declared variable {label!r} does not name two '
                             'multipole components')
        row, col = index_of[parts[0]], index_of[parts[1]]
        if row > col:
            raise ValueError(f'declared variable {label!r} is below the diagonal; '
                             'CamCASP names the upper triangle')
        site = sites[first[site_type]]
        if site.rank_limit == 0 or col >= site.component_count:
            raise ValueError(f'declared variable {label!r} is outside site {name}\'s '
                             f'rank limit {site.rank_limit}')
        resolved[(site_type, row, col)] = label
    return resolved


def refinement_model(sites, anchor_tensors, *, frequency_au=0.0, cutoff=1e-4,
                     weight_type=3, weight_coefficient=1e-3, provenance='',
                     declared_variables=None):
    """Build the variable list CamCASP's ``.pdef`` plus ``Penalties`` block encodes.

    ``anchor_tensors[s]`` is site ``s``'s raw local polarizability at
    ``frequency_au``, shaped ``(ncomp, ncomp)`` for that site's rank limit, in
    the site's local axes.  A variable is created for each upper-triangle
    ``(row, col)`` of each unique site type whose *reference site* value exceeds
    ``cutoff`` in magnitude (process_data.F90:2109-2118); the test is on the
    reference site alone, so a large value at an equivalent site does not
    rescue a component the reference site screens out.  Type order, and hence
    parameter order, is first appearance in ``sites``
    (``list_unique_types_mol``, molecule_operations_cluster.F90:964-983).

    ``declared_variables`` supplies the variable list explicitly, as a
    ``.pdef`` does, instead of deriving it from the cutoff.  It must name every
    variable the cutoff would have produced -- a declared list that omits a
    penalized variable would misrepresent the ``Penalties`` block as covering
    the fit -- and may name more; the extra ones are free parameters with
    anchor 0.0 and strength 0.0, exactly the state a ``.pdef`` variable with no
    ``Penalties`` line is in.  Supplying it is a different declared model, not
    a relaxation of the cutoff: the cutoff still decides which variables are
    anchored and penalized, and is recorded unchanged on the model.
    """
    sites = _validated_sites(sites)
    tensors = _validated_anchor_tensors(sites, anchor_tensors)
    if not np.isfinite(cutoff) or cutoff < 0.0:
        raise ValueError('cutoff must be finite and nonnegative')
    if not np.isfinite(frequency_au) or frequency_au < 0.0:
        raise ValueError('frequency_au must be a nonnegative imaginary-axis magnitude')

    offsets, total = [], 0
    labels = []
    for site in sites:
        offsets.append(total)
        for index in range(site.component_count):
            labels.append(f'{site.label}_{COMPONENT_NAMES[index]}')
        total += site.component_count

    ordered_types, first = [], {}
    for index, site in enumerate(sites):
        if site.site_type not in first:
            first[site.site_type] = index
            ordered_types.append(site.site_type)
    members = {t: tuple(i for i, s in enumerate(sites) if s.site_type == t) for t in ordered_types}
    declared = _validated_declared_variables(declared_variables, sites, ordered_types, first)

    parameter_labels, parameter_entries, anchors, strengths = [], [], [], []
    nonsymmetric, copy_discrepancy = 0, 0.0
    unpenalized, missing = [], []
    for site_type in ordered_types:
        reference = first[site_type]
        site = sites[reference]
        if site.rank_limit == 0:
            continue
        tensor = tensors[reference]
        for row in range(site.component_count):
            rank1 = component_rank(row)
            for col in range(row, site.component_count):
                rank2 = component_rank(col)
                alpha = float(tensor[row, col])
                penalized = abs(alpha) > cutoff
                label = (f'{site.label}_{COMPONENT_NAMES[row]}'
                         f'_{COMPONENT_NAMES[col]}_A')
                if declared is not None and (site_type, row, col) not in declared:
                    # A variable the cutoff keeps but the declared list omits
                    # would leave a penalized anchor out of the fit entirely.
                    if penalized:
                        missing.append(label)
                    continue
                if declared is None and not penalized:
                    continue
                parameter_labels.append(label)
                parameter_entries.append(tuple((s, row, col) for s in members[site_type]))
                anchors.append(alpha if penalized else 0.0)
                strengths.append(penalty_weight(weight_type=weight_type,
                                                weight_coefficient=weight_coefficient,
                                                alpha=alpha, frequency=frequency_au,
                                                rank1=rank1, rank2=rank2)
                                 if penalized else 0.0)
                nonsymmetric += len(members[site_type])
                if penalized:
                    for other in members[site_type]:
                        copy_discrepancy = max(copy_discrepancy,
                                               abs(float(tensors[other][row, col]) - alpha))
                else:
                    unpenalized.append(label)
    if missing:
        raise ValueError('declared_variables omits variables the cutoff keeps, whose '
                         f'anchors would then go unfitted: {missing}')
    if declared is not None and len(parameter_labels) != len(declared):
        raise ValueError('declared_variables names a variable the component scan does '
                         'not reach')
    if not parameter_labels:
        raise ValueError('no component of any reference site survives the cutoff')
    if len(parameter_labels) > MAX_PARAMETERS:
        raise ValueError(f'refinement is limited to {MAX_PARAMETERS} parameters')

    return RefinementModel(
        sites=sites, site_types=tuple(ordered_types),
        reference_sites=tuple(first[t] for t in ordered_types),
        equivalent_sites=tuple(members[t] for t in ordered_types),
        channel_offsets=tuple(offsets), channel_count=total, channel_labels=tuple(labels),
        parameter_labels=tuple(parameter_labels), parameter_entries=tuple(parameter_entries),
        anchors=tuple(anchors), strengths=tuple(strengths),
        cutoff=float(cutoff), weight_type=int(weight_type),
        weight_coefficient=float(weight_coefficient), frequency_au=float(frequency_au),
        nonsymmetric_parameter_count=nonsymmetric,
        anchor_sha256=_anchor_hash(tensors),
        copy_anchor_discrepancy=float(copy_discrepancy),
        provenance=provenance or 'caller-supplied raw local polarizabilities; '
                                 'CamCASP write_pfit_local_symm variable construction',
        declared_variables=() if declared is None else tuple(declared_variables),
        unpenalized_variables=tuple(unpenalized))


def _validated_points(points_bohr):
    points = np.array(points_bohr, dtype=float, copy=True)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points_bohr must be npoint x 3')
    if not 1 <= points.shape[0] <= MAX_POINTS:
        raise ValueError(f'between 1 and {MAX_POINTS} points are required')
    if not np.all(np.isfinite(points)):
        raise ValueError('points_bohr must be finite')
    return points


def channel_fields(points_bohr, model, *, damping=0.0):
    """The ``npoint x channel_count`` matrix of T functions.

    Row ``i`` concatenates, per site and in site order, the interaction
    functions between a unit charge at ``points_bohr[i]`` and that site's
    multipole components in the site's local axes -- CamCASP's
    ``tfn(i,:,s)`` (process.F90:451-505) via the certified
    ``core.isa_t_functions``.
    """
    if not isinstance(model, RefinementModel):
        raise ValueError('model must be a RefinementModel')
    points = _validated_points(points_bohr)
    fields = np.zeros((points.shape[0], model.channel_count))
    for index, site in enumerate(model.sites):
        start = model.channel_offsets[index]
        stop = start + site.component_count
        for row, point in enumerate(points):
            fields[row, start:stop] = core.isa_t_functions(
                site.rank_limit, list(point), list(site.origin_bohr),
                [list(axis) for axis in site.frame], damping)
    if not np.all(np.isfinite(fields)):
        raise ValueError('nonfinite T function; a fit point coincides with a site')
    return fields


def block_diagonal_tensor(model, per_site_tensors):
    """Assemble per-site local tensors into one ``channel_count`` square block matrix."""
    tensors = _validated_anchor_tensors(model.sites, per_site_tensors)
    matrix = np.zeros((model.channel_count, model.channel_count))
    for index, site in enumerate(model.sites):
        start = model.channel_offsets[index]
        stop = start + site.component_count
        matrix[start:stop, start:stop] = tensors[index]
    return matrix


def isotropic_scalars(result):
    """``alpha_l = trace(alpha_ll)/(2l+1)`` per site of a refinement, l = 1..limit.

    The dispersion kernel wants one scalar per site and rank, not a tensor, and
    the reduction is a trace over the ``2l+1`` components of one rank in the
    Racah packing this module uses everywhere (00 at index 0, then 10, 11c,
    11s, 20, ...).  Doing it here keeps the convention in one place: the same
    reduction that :func:`isapol_lw.supplied_nonlocal_properties` applies to the
    unrefined localized tensors, applied to the refined ones.

    Rank 0 is deliberately absent.  The charge-flow block is a refinement
    variable but not a polarizability of a rank the Casimir-Polder sum runs
    over, and the isotropic dispersion contract starts at rank 1.

    Returns ``(ranks_per_site, scalars_per_site)``: for site ``s``,
    ``ranks_per_site[s]`` is ``(1, ..., rank_limit)`` and ``scalars_per_site[s]``
    the matching ``alpha_l``.  A site with ``rank_limit == 0`` yields two empty
    tuples and cannot enter a dispersion model; that is reported, not repaired.
    """
    if not isinstance(result, RefinementResult):
        raise ValueError('isotropic_scalars requires a RefinementResult')
    ranks, scalars = [], []
    for site, tensor in zip(result.model.sites, result.refined_tensors):
        array = np.asarray(tensor, dtype=float)
        if array.shape != (site.component_count,) * 2:
            raise ValueError('refined tensor shape disagrees with the site rank limit')
        site_ranks = tuple(range(1, site.rank_limit + 1))
        ranks.append(site_ranks)
        scalars.append(tuple(
            float(np.trace(array[l * l:(l + 1) ** 2, l * l:(l + 1) ** 2]) / (2 * l + 1))
            for l in site_ranks))
    return tuple(ranks), tuple(scalars)


def point_to_point_response(fields, model, per_site_tensors):
    """Forward map ``v(i,j) = sum_s T(i,s) . alpha_s . T(j,s)``.

    This is what CamCASP's ``process`` evaluates when it turns a distributed
    model back into a point-to-point response.  It is the model prediction, not
    a target: a refinement target must come from an actual response calculation.
    Returns the full signed matrix; use :func:`pack_lower_triangle` for the
    packing the solver consumes.
    """
    matrix = block_diagonal_tensor(model, per_site_tensors)
    fields = np.asarray(fields, dtype=float)
    if fields.ndim != 2 or fields.shape[1] != model.channel_count:
        raise ValueError('fields must be npoint x channel_count')
    return fields @ matrix @ fields.T


def pack_lower_triangle(matrix):
    """Pack ``v[i][j]`` at ``i*(i+1)//2 + j`` for every ``j <= i``, taking the lower triangle."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError('a square matrix is required')
    rows, cols = np.tril_indices(matrix.shape[0])
    return matrix[rows, cols].copy()


def _parameter_tensor(model, entries):
    tensor = np.zeros((model.channel_count, model.channel_count))
    for site, row, col in entries:
        start = model.channel_offsets[site]
        # process.F90 setup: aij(k) accumulates tfn(i,t,a)*cp*tfn(j,u,b) and,
        # when (t,a) and (u,b) differ, the transposed product as well.  The
        # symmetric channel tensor of the owned solver expresses exactly that.
        tensor[start + row, start + col] += 1.0
        if row != col:
            tensor[start + col, start + row] += 1.0
    return tensor


#: Representation each declared target origin must carry, mirroring the
#: solver's own admission rules (``pfit.cc`` 135-152) so that a caller gets the
#: message in Python rather than a bare ValueError out of C++.
ORIGIN_REPRESENTATIONS = {
    'NativeDirectActualPointResponse': ('native_point_charge_ov_operators', False),
    'NativeFittedPointResponse': ('fitted_density_coefficients', True),
    'SuppliedFittedPropagatorPointResponse': ('fitted_density_coefficients', True),
}


def _validated_origin(target_origin, response_representation, auxiliary_basis_id):
    if target_origin is None:
        raise ValueError('target_origin is required; declare where the response came from')
    name = str(target_origin).rsplit('.', 1)[-1]
    if name in ORIGIN_REPRESENTATIONS:
        expected, wants_auxiliary = ORIGIN_REPRESENTATIONS[name]
        if response_representation != expected:
            raise ValueError(f'origin {name} requires response_representation {expected!r}')
        if wants_auxiliary and not auxiliary_basis_id.strip():
            raise ValueError(f'origin {name} requires an auxiliary_basis_id')
        if not wants_auxiliary and auxiliary_basis_id:
            raise ValueError(f'origin {name} carries no auxiliary fit; declaring '
                             'auxiliary_basis_id would be a false provenance claim')
    return name


def refinement_problem(model, points_bohr, packed_targets, *, target_origin=None,
                       source_id, generation_record, fields=None, damping=0.0,
                       label='refinement', response_representation='',
                       auxiliary_basis_id=''):
    """Assemble the ``core.IsaPfitProblem`` the owned solver consumes.

    ``packed_targets`` must already be in the solver's packing, one value per
    ``j <= i``.  ``target_origin``, ``source_id`` and ``generation_record`` are
    required and are not defaulted: the whole point of the origin tag is that a
    native direct-OV response and a supplied fitted-propagator response are not
    interchangeable targets, and the solver refuses a blank record anyway.
    """
    if not isinstance(model, RefinementModel):
        raise ValueError('model must be a RefinementModel')
    _validated_origin(target_origin, response_representation, auxiliary_basis_id)
    for name, value in (('source_id', source_id), ('generation_record', generation_record)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{name} must be a nonblank string')
    points = _validated_points(points_bohr)
    npoint = points.shape[0]
    targets = np.asarray(packed_targets, dtype=float)
    expected = npoint * (npoint + 1) // 2
    if targets.shape != (expected,):
        raise ValueError(f'{expected} packed targets are required for {npoint} points')
    if not np.all(np.isfinite(targets)):
        raise ValueError('packed_targets must be finite')
    if fields is None:
        fields = channel_fields(points, model, damping=damping)
    fields = np.asarray(fields, dtype=float)
    if fields.shape != (npoint, model.channel_count):
        raise ValueError('fields must be npoint x channel_count')

    problem = core.IsaPfitProblem()
    problem.frequency_au = model.frequency_au
    provenance = core.IsaPfitTargetProvenance()
    provenance.origin = target_origin
    provenance.convention = \
        core.IsaPfitTargetConvention.NegativeInducedPotentialPerUnitSourceChargeAtomicUnits
    provenance.source_id = source_id
    provenance.response_representation = response_representation
    provenance.auxiliary_basis_id = auxiliary_basis_id
    provenance.generation_record = generation_record
    problem.target_provenance = provenance

    pfit_model = core.IsaPfitModel()
    pfit_model.channel_labels = list(model.channel_labels)
    pfit_model.parameter_labels = list(model.parameter_labels)
    pfit_model.parameter_units = [
        f'bohr^{component_rank(row) + component_rank(col) + 1}'
        for entries in model.parameter_entries for (_, row, col) in entries[:1]]
    tensors = []
    for entries in model.parameter_entries:
        matrix = core.IsaPfitMatrix()
        matrix.rows = model.channel_count
        matrix.cols = model.channel_count
        matrix.values = _parameter_tensor(model, entries).ravel().tolist()
        tensors.append(matrix)
    pfit_model.parameter_tensors = tensors
    pfit_model.fixed = [False] * model.parameter_count
    pfit_model.fixed_values = [0.0] * model.parameter_count
    pfit_model.provenance = model.provenance
    problem.model = pfit_model

    batch = core.IsaPfitBatch()
    batch.label = label
    batch.points_bohr = [list(point) for point in points]
    field_matrix = core.IsaPfitMatrix()
    field_matrix.rows = npoint
    field_matrix.cols = model.channel_count
    field_matrix.values = fields.ravel().tolist()
    batch.fields = field_matrix
    batch.targets = targets.tolist()
    problem.batches = [batch]

    penalty = core.IsaPfitMatrixPenalty()
    strength = core.IsaPfitMatrix()
    strength.rows = model.parameter_count
    strength.cols = model.parameter_count
    strength.values = np.diag(np.asarray(model.strengths, dtype=float)).ravel().tolist()
    penalty.matrix = strength
    penalty.anchor = list(model.anchors)
    problem.penalty = penalty
    problem.linear_penalties = []
    return problem


@dataclass(frozen=True)
class RefinementResult:
    """Refined per-site local tensors and the solver's own record of the fit."""
    model: RefinementModel
    parameters: tuple
    refined_tensors: tuple = field(repr=False)
    anchor_shift_maxabs: float
    status: object
    diagnostics: object = field(repr=False)
    result: object = field(repr=False)
    refinement_status: str
    penalty_convention: str = 'strengths[k]*(z[k]-anchors[k])**2, CamCASP read_penalties'


def refine(model, points_bohr, packed_targets, *, target_origin=None, source_id,
           generation_record, fields=None, damping=0.0, options=None, log=None, wfn=None,
           **provenance):
    """Solve the refinement and return the refined per-site local tensors.

    The solver is the owned ``core.isa_pfit_solve``.  The default here is
    ``NormalEquationsDSYSV`` because that is what CamCASP's ``pfit`` uses
    (``process.F90::solve`` calls ``DSYSV('L',...)`` on the normal equations);
    pass ``options`` to use the streaming QR path instead, which is more
    accurate and not what upstream does.  A non-``Solved`` status is returned
    rather than raised: a rank-deficient or ill-conditioned refinement is a
    result about the model, and is reported, not repaired.

    ``wfn`` is optional and is written to only by the reporting module, which
    publishes this stage's machine-readable variables on it.  Passing one does
    not make the refinement a property of that wavefunction: the caller owns
    the points, the target response and the model, and nothing here reads the
    wavefunction back or checks it against the targets it was given.
    """
    log = _lg.silent() if log is None else log
    problem = refinement_problem(model, points_bohr, packed_targets, fields=fields,
                                 damping=damping, target_origin=target_origin,
                                 source_id=source_id, generation_record=generation_record,
                                 **provenance)
    if options is None:
        options = core.IsaPfitOptions()
        options.solver = core.IsaPfitSolver.NormalEquationsDSYSV
    log.stage('point-to-point refinement (PFIT)', _lg.refine_parameters(
        model=model, points=points_bohr, fields=fields, damping=damping,
        options=options, source_id=source_id, generation_record=generation_record,
        target_origin=target_origin, target_convention=TARGET_CONVENTION,
        label=provenance.get('label', 'refinement'),
        response_representation=provenance.get('response_representation', ''),
        auxiliary_basis_id=provenance.get('auxiliary_basis_id', '')))
    _lg.report_refinement_model(log, model)
    result = core.isa_pfit_solve(problem, options)
    parameters = tuple(map(float, result.parameters))

    refined = [np.zeros((site.component_count,) * 2) for site in model.sites]
    for value, entries in zip(parameters, model.parameter_entries):
        for site, row, col in entries:
            refined[site][row, col] = value
            refined[site][col, row] = value
    for tensor in refined:
        tensor.flags.writeable = False
    shift = max((abs(z - a) for z, a in zip(parameters, model.anchors)), default=0.0)
    refinement = RefinementResult(
        model=model, parameters=parameters, refined_tensors=tuple(refined),
        anchor_shift_maxabs=float(shift), status=result.status,
        diagnostics=result.diagnostics, result=result,
        refinement_status='PFIT_refined_against_point_to_point_response')
    _lg.report_refinement(log, wfn, refinement, frequency=model.frequency_au)
    log.stage_end()
    return refinement


# ------------------------------------------ dispersion from refined tensors ----

#: Orders the isotropic Casimir-Polder kernel admits.  Odd orders need the
#: lower-J normalization that is not certified on this branch, and are refused
#: here rather than computed and labelled provisional.
DISPERSION_ORDERS = (6, 8, 10, 12)


@dataclass(frozen=True)
class RefinedIsotropicDispersion:
    """Isotropic C_n contracted from PFIT-refined tensors.

    Deliberately NOT an ``isapol_lw.IsotropicDispersion``: that record is typed
    to two ``LocalProperties``, which are certified factory results of the
    localization stage and may not be fabricated from refined tensors.  The
    refinement is a different model of the same molecule -- it moves the
    polarizabilities off the localized anchors to fit an actual point-to-point
    response -- so it gets its own record, its own provenance and its own
    QCVariable names, and the two C_n are never interchangeable.

    ``site_ranks`` is the resolved per-site rank tuple, not the caller's
    argument: which ranks entered decides whether an order is complete, so what
    was used is stored rather than what was requested.
    """
    labels: tuple
    origins_bohr: tuple = field(repr=False)
    frequencies: tuple
    cp_weights: tuple = field(repr=False)
    quadrature_provenance: object
    site_ranks: tuple
    pairs: tuple = field(repr=False)
    anchor_shift_maxabs: float
    anchor_sha256: str
    weight_type: int
    weight_coefficient: float
    refinement_status: str
    solver_status: tuple
    provenance: str
    units: str = 'C_n: Eh bohr^n; cp_weights includes the Jacobian and 1/(2*pi) exactly once'
    origin: str = 'Psi4_isotropic_dispersion_from_PFIT_refined_tensors'
    model_status: str = ('refined against a point-to-point response; NOT comparable with '
                         'the unrefined LW C_n, which is a different model')

    def __post_init__(self):
        for name in ('labels', 'frequencies', 'cp_weights', 'site_ranks', 'pairs',
                     'solver_status'):
            object.__setattr__(self, name, tuple(getattr(self, name)))


def _dispersion_nodes(refinements):
    """Validate a per-frequency refinement sequence as one model on one grid."""
    nodes = tuple(refinements)
    if not nodes or not all(isinstance(r, RefinementResult) for r in nodes):
        raise ValueError('a non-empty sequence of RefinementResult is required, '
                         'one per frequency node')
    first = nodes[0].model
    for other in nodes[1:]:
        if other.model.sites != first.sites:
            raise ValueError('every node must refine the identical declared site set; '
                             'a C_n contracted across two site sets is not one model')
        if (other.model.site_types != first.site_types
                or other.model.parameter_labels != first.parameter_labels):
            raise ValueError('every node must share one declared variable set')
        for name in ('weight_type', 'weight_coefficient', 'cutoff'):
            if getattr(other.model, name) != getattr(first, name):
                raise ValueError(f'every node must share one penalty scheme; {name} differs')
    frequencies = tuple(float(r.model.frequency_au) for r in nodes)
    if not all(np.isfinite(frequencies)) or any(x < 0. for x in frequencies):
        raise ValueError('frequency nodes must be finite and non-negative')
    if len(set(frequencies)) != len(frequencies):
        raise ValueError('frequency nodes must be distinct; a repeated node is not a grid')
    return nodes, first, frequencies


def _dispersion_site_ranks(first, declared, available):
    """Resolve the per-site rank tuple actually contracted.

    ``None`` means every rank the site was refined at, read off that site's own
    declared ``rank_limit`` -- not a fixed ``(1, 2)`` or ``(1, 2, 3)``.  A rank
    above the limit is refused rather than zero-filled: those components are
    absent by declaration, and reading their zeros would drop terms from the
    C_n sum while still reporting the order complete.
    """
    if declared is None:
        resolved = tuple(tuple(site) for site in available)
    else:
        resolved = tuple(tuple(site) for site in declared)
        if len(resolved) != len(first.sites):
            raise ValueError('site_ranks must declare one rank tuple per site, in site order')
        for site, ranks in zip(first.sites, resolved):
            if not ranks or any(type(r) is not int for r in ranks):
                raise ValueError('each site must declare a non-empty tuple of integer ranks')
            if any(r < 1 or r > MAX_RANK for r in ranks):
                raise ValueError(f'declarable dispersion ranks are 1..{MAX_RANK}')
            if any(b <= a for a, b in zip(ranks, ranks[1:])):
                raise ValueError('site ranks must be strictly increasing')
            if any(r > site.rank_limit for r in ranks):
                raise ValueError(
                    f'declared rank exceeds the refinement rank limit of site {site.label} '
                    f'({site.rank_limit}); those components are absent by declaration, '
                    'not zero-valued physics')
    for site, ranks in zip(first.sites, resolved):
        if not ranks:
            raise ValueError(f'site {site.label} was refined at rank limit 0 and has no '
                             'polarizability of any dispersion rank; it cannot enter a '
                             'dispersion model')
    return resolved


def refined_isotropic_dispersion(refinements, *, cp_weights, quadrature_provenance,
                                 max_order=12, site_ranks=None, log=None, wfn=None):
    """Isotropic C_n from one refinement per Casimir-Polder frequency node.

    The refinement solves one frequency at a time, so a dispersion coefficient
    needs the whole grid: ``refinements`` is the per-node sequence, all of them
    refining the identical declared site set under one penalty scheme, and their
    ``frequency_au`` values are the grid that is contracted.  They are taken in
    the order given and are NOT sorted here -- the CP weights belong to the
    caller's node order, and reordering one without the other would silently
    misweight the sum.

    The contraction itself is the same owned ``core.isa_isotropic_dispersion``
    the unrefined path uses, with the model paired against itself.  What differs
    is the input: ``trace(alpha_ll)/(2l+1)`` of the refined tensors rather than
    of the localized ones.  That is a different model, so the result is a
    :class:`RefinedIsotropicDispersion` and its variables are published under
    ``ATOMIC REFINED DISPERSION ...``; nothing here may be compared against an
    unrefined C_n as though it were the same quantity.

    A node whose solver status is not ``Solved`` still returned numbers, and is
    contracted rather than dropped: the status is recorded on the result and
    narrated per node, because a rank-deficient refinement is a fact about the
    model and is reported, not repaired.

    ``log`` and ``wfn`` are reporting surfaces only.  ``wfn`` is written to
    solely by the reporting module; passing one does not make the C_n a property
    of that wavefunction, and nothing here reads it back.
    """
    nodes, first, frequencies = _dispersion_nodes(refinements)
    if not isinstance(quadrature_provenance, _lw.Provenance):
        raise ValueError('explicit quadrature Provenance required')
    if type(max_order) is not int or max_order not in DISPERSION_ORDERS:
        raise ValueError('max_order must be 6, 8, 10 or 12')
    weights = np.asarray(cp_weights, dtype=float).ravel()
    if weights.shape != (len(nodes),):
        raise ValueError('one CP weight per refinement node is required')
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.) or not np.any(weights > 0.):
        raise ValueError('CP weights must be finite, non-negative and not all zero')
    if any(x == 0. and w != 0. for x, w in zip(frequencies, weights)):
        raise ValueError('the static node carries zero CP weight by construction; '
                         'a nonzero one is refused')

    available, scalars = [], []
    for node in nodes:
        node_ranks, node_scalars = isotropic_scalars(node)
        available.append(node_ranks)
        scalars.append(node_scalars)
    if any(node_ranks != available[0] for node_ranks in available[1:]):
        raise ValueError('every node must expose the identical per-site rank inventory')
    resolved = _dispersion_site_ranks(first, site_ranks, available[0])

    digest = hashlib.sha256()
    for node in nodes:
        digest.update(node.model.anchor_sha256.encode())
    anchor_sha256 = digest.hexdigest()
    provenance = (f'Psi4_PFIT_refined; nodes={len(nodes)}; variables={first.parameter_count}; '
                  f'weight={first.weight_type}/{first.weight_coefficient!r}; '
                  f'cutoff={first.cutoff!r}; anchors_sha256={anchor_sha256}; '
                  f'{first.provenance}')
    log = _lg.silent() if log is None else log
    log.stage('isotropic dispersion from refined tensors (Casimir-Polder)',
              _lg.refined_dispersion_parameters(
                  refinements=nodes, max_order=max_order, cp_weights=weights,
                  quadrature_provenance=quadrature_provenance, site_ranks=site_ranks,
                  resolved_site_ranks=resolved, anchor_sha256=anchor_sha256))

    model_sites = []
    for index, site in enumerate(first.sites):
        column = [available[0][index].index(rank) for rank in resolved[index]]
        table = np.array([[scalars[node][index][j] for j in column]
                          for node in range(len(nodes))], dtype=float)
        isosite = core.IsaIsotropicSite()
        isosite.label, isosite.origin = site.label, list(map(float, site.origin_bohr))
        isosite.ranks = list(resolved[index])
        isosite.polarizabilities = core.Matrix.from_array(np.ascontiguousarray(table))
        model_sites.append(isosite)
    isomodel = core.IsaIsotropicModel(list(frequencies), model_sites, provenance)
    result = core.isa_isotropic_dispersion(isomodel, isomodel, weights.tolist(), max_order)
    labels = tuple(site.label for site in first.sites)
    pairs = tuple(_lw.DispersionPair(
        p.site_a, p.site_b, labels[p.site_a], labels[p.site_b],
        tuple(_lw.Coefficient(c.order, c.value, tuple(map(tuple, c.included_rank_pairs)),
                              tuple(map(tuple, c.missing_rank_pairs)), c.complete)
              for c in p.coefficients)) for p in result.pairs)
    record = RefinedIsotropicDispersion(
        labels=labels,
        origins_bohr=tuple(tuple(map(float, site.origin_bohr)) for site in first.sites),
        frequencies=frequencies, cp_weights=tuple(map(float, weights)),
        quadrature_provenance=quadrature_provenance, site_ranks=resolved, pairs=pairs,
        anchor_shift_maxabs=max(float(r.anchor_shift_maxabs) for r in nodes),
        anchor_sha256=anchor_sha256, weight_type=first.weight_type,
        weight_coefficient=first.weight_coefficient,
        refinement_status=nodes[0].refinement_status,
        solver_status=tuple(str(r.status).rsplit('.', 1)[-1] for r in nodes),
        provenance=provenance)
    _lg.report_refined_dispersion(log, wfn, record)
    log.stage_end()
    return record
