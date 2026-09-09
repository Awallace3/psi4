# Psi4: Copyright (c) 2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""Native direct-OV point-charge response targets for the owned PFIT solver.

This is a PFIT *prerequisite*: it produces the induced-potential response of an
actual native wavefunction to unit source charges at caller-owned points. It is
not the historical CamCASP target, which is a constrained-NN/distributed
fitted-propagator quantity at a different point lattice, and it does not
perform refinement, anchoring, frame conversion or any charge/multipole model
inference. No SCF, ISA partition, auxiliary fit or atomic-property
registration happens here.
"""
import hashlib
from dataclasses import dataclass, field

import numpy as np

from psi4 import core
from .sapt.fdds_response import FDDSFullOVResponse

#: Maximum number of source points; matches the shared transition-leg limit.
MAXIMUM_POINTS = 512

CONVENTION = 'v_pq = -d(phi_induced at R_p)/d(q at R_q); atomic units Eh/e^2'
REPRESENTATION = 'native_point_charge_ov_operators'


@dataclass(frozen=True)
class NativePointChargeResponse:
    """Owned per-frequency point-charge response targets and diagnostics.

    ``responses[k]`` is the raw signed npoint x npoint response at
    ``frequencies_au[k]``, in the PFIT sign convention
    ``v_pq = -d(phi_induced at R_p)/d(q at R_q)`` with no energy 1/2 factor, no
    bare electrostatics and no nuclear contribution. ``packed_targets[k]``
    stores ``v[i][j]`` at ``i*(i+1)//2 + j`` for every ``j <= i`` exactly once,
    taken from the computed lower triangle: nothing is symmetrized. The
    residual asymmetry is reported in ``reciprocity_defects``.

    ``origin`` is deliberately the native direct-OV origin. It must not be
    relabelled as a supplied actual or fitted-propagator target.
    """
    operators: object
    points_bohr: np.ndarray = field(repr=False)
    frequencies_au: tuple
    responses: tuple = field(repr=False)
    packed_targets: tuple = field(repr=False)
    reciprocity_defects: tuple
    minimum_diagonals: tuple
    maximum_absolute_values: tuple
    convention: str
    representation: str
    generation_record: str
    context_sha256: str
    correction_provenance: object
    caller_converged: bool
    convergence_evidence: str

    @property
    def npoint(self):
        return int(self.points_bohr.shape[0])

    def target_provenance(self, source_id):
        """Return a PFIT provenance value object for these native targets."""
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError('explicit nonempty source_id required')
        provenance = core.IsaPfitTargetProvenance()
        provenance.origin = core.IsaPfitTargetOrigin.NativeDirectActualPointResponse
        provenance.convention = (
            core.IsaPfitTargetConvention.NegativeInducedPotentialPerUnitSourceChargeAtomicUnits)
        provenance.source_id = source_id
        provenance.response_representation = self.representation
        provenance.auxiliary_basis_id = ''
        provenance.generation_record = self.generation_record
        return provenance

    def batch(self, label, frequency_index, fields):
        """Assemble a PFIT batch from these targets and CALLER-OWNED fields.

        ``fields`` is the caller's model design matrix (npoint x nchannel). No
        channel, site, parameter count or model convention is inferred here,
        and none is taken from a final Cn or another track.
        """
        if not isinstance(label, str) or not label.strip():
            raise ValueError('explicit nonempty batch label required')
        if (isinstance(frequency_index, (bool, np.bool_))
                or not isinstance(frequency_index, (int, np.integer))
                or not 0 <= frequency_index < len(self.frequencies_au)):
            raise ValueError('frequency_index must select a computed frequency')
        design = np.asarray(fields)
        if (np.iscomplexobj(design) or design.ndim != 2 or design.shape[0] != self.npoint
                or not design.shape[1] or not np.isfinite(design).all()):
            raise ValueError('fields must be a finite real (npoint, nchannel) design matrix')
        batch = core.IsaPfitBatch()
        batch.label = label
        batch.points_bohr = [[float(x) for x in row] for row in self.points_bohr]
        matrix = core.IsaPfitMatrix()
        matrix.rows = int(design.shape[0])
        matrix.cols = int(design.shape[1])
        matrix.values = [float(v) for v in np.ascontiguousarray(design, dtype=float).reshape(-1)]
        batch.fields = matrix
        batch.targets = [float(v) for v in self.packed_targets[frequency_index]]
        return batch


def _validated_points(points_bohr, max_points):
    points = np.asarray(points_bohr)
    if (np.iscomplexobj(points) or points.ndim != 2 or points.shape[1] != 3
            or not points.shape[0] or not np.isfinite(points).all()):
        raise ValueError('points_bohr must be finite real [x,y,z] rows in bohr')
    if points.shape[0] > min(max_points, MAXIMUM_POINTS):
        raise ValueError('source point resource limit')
    points = np.array(points, dtype=float, copy=True)
    unique = np.unique(points, axis=0)
    if unique.shape[0] != points.shape[0]:
        raise ValueError('duplicate (exactly coincident) source point')
    return points


def _validated_frequencies(frequencies):
    values = np.asarray(frequencies)
    if (np.iscomplexobj(values) or values.ndim != 1 or not values.size or values.size > 64
            or not np.isfinite(values).all() or np.any(values < 0)):
        raise ValueError('frequencies must be 1-64 finite nonnegative imaginary-axis magnitudes')
    values = np.array(values, dtype=float, copy=True)
    if np.unique(values).size != values.size:
        raise ValueError('duplicate frequency requested')
    return values


def _require_same_context(response, wavefunction):
    """Reject a response built from different orbital/energy/occupation state.

    Exact equality only: a re-converged or edited wavefunction invalidates the
    owned native operators, and no tolerance can make the two contexts the same
    physical state.
    """
    provider = response.provider
    orbitals = provider.orbitals().to_array()
    energies = np.asarray(provider.energies())
    if (wavefunction is None or wavefunction.nirrep() != 1
            or not wavefunction.same_a_b_orbs() or not wavefunction.same_a_b_dens()
            or wavefunction.nalpha() != wavefunction.nbeta()
            or wavefunction.soccpi()[0] != 0):
        raise ValueError('only restricted closed-shell C1 wavefunctions are supported')
    if provider.nocc != wavefunction.nalpha():
        raise ValueError('native response context occupation differs from the wavefunction')
    if provider.nocc + provider.nvir != wavefunction.nmo():
        raise ValueError('native response context dimension differs from the wavefunction')
    if not np.array_equal(orbitals, np.asarray(wavefunction.Ca())):
        raise ValueError('native response context orbitals differ from the wavefunction')
    if not np.array_equal(energies, np.asarray(wavefunction.epsilon_a())):
        raise ValueError('native response context energies differ from the wavefunction')
    return provider


def native_point_charge_response(response, wavefunction, points_bohr, *, frequencies=(0.,),
                                 max_bytes=512 * 1024**2, max_points=MAXIMUM_POINTS):
    """Return raw signed point-charge response targets for one native context.

    ``response`` is an existing :class:`NativeWavefunctionResponse` whose owned
    H1/H2 are reused. Its own transition-leg coordinates are irrelevant here and
    are not mutated: a second shared full-OV solver is constructed over the same
    operators with the point-charge legs

        W[t][p] = + integral phi_i(r) phi_a(r) / |r - R_p| dr,  t = a*nocc+i

    which is the positive Coulomb kernel, i.e. minus the charge-inclusive
    electron ESP operator oeprop accumulates. W enters the contraction twice, so
    the target sign below is invariant to that choice; the two conventions must
    never be mixed inside one leg, hence only this one is published.

    With C(iw) the raw signed full-OV response and coupling already contained in
    H1, the returned per-frequency target is

        v(iw)_pq = -(W^T C(iw) W)_pq

    in atomic units Eh/e^2, equal to -d(phi_induced at R_p)/d(q at R_q). There
    is no energy 1/2 factor, no bare (uninduced) electrostatics and no nuclear
    term. ``frequencies`` are imaginary-axis magnitudes in hartree, w >= 0, and
    w = 0 is the static response; no quadrature weight is applied.

    Cost, stated exactly, because it is easy to overstate: the npoint
    right-hand sides avoid only the nov x nov right-hand-side and solution
    blocks of the nov-RHS route. The dominant term is unchanged — one
    O(nov^3) LU factorization of the same nov x nov operator, plus the shared
    solver's own O(nov^3) H2*H1 product, which is recomputed here rather than
    reused, and ~O(nov^2) dense workspace. The saving is O(nov^2*npoint)
    instead of O(nov^3) in the solve/contraction tail, and it vanishes in the
    corner npoint == nov. What is true unconditionally is the second half: this
    does not relax, bypass or re-tune the native response work guard, which the
    supplied ``response`` already passed, and ``max_bytes`` bounds the C++
    dense envelope only. The numpy arrays above are bounded by the nov cap, not
    by ``max_bytes``.

    C++ holds the authoritative admission and resource guards and computes W
    with a local serial Libint engine over all ordered shell pairs and no global
    option mutation. The only integrals dropped are shell pairs libint2 itself
    reports as precision zero; nothing screens on distance, magnitude or
    geometry. Diagnostics are recorded only: no point screening, distance
    repair, symmetrization or conditioning policy.

    ``correction_provenance``, ``caller_converged`` and ``convergence_evidence``
    are **forwarded** from the supplied response. Nothing here can re-verify
    them: `_require_same_context` proves only that the orbitals, energies,
    occupation and dimension are the wavefunction's own. They are hashed into
    ``context_sha256`` so a different declaration is a different context, which
    is not the same as validating the declaration.
    """
    for name, value in (('max_bytes', max_bytes), ('max_points', max_points)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if not hasattr(response, 'provider') or not hasattr(response, 'at_frequency'):
        raise ValueError('response must be a native wavefunction response producer')
    if not getattr(response, 'caller_converged', False):
        raise ValueError('response must carry an explicit caller convergence declaration')
    points = _validated_points(points_bohr, max_points)
    omegas = _validated_frequencies(frequencies)
    provider = _require_same_context(response, wavefunction)
    h1 = provider.h1().to_array()
    h2 = provider.h2().to_array()
    nov = provider.nocc * provider.nvir
    if points.shape[0] * nov * 8 > max_bytes:
        raise ValueError('point-charge operator resource limit')
    operators = core.IsaPointChargeOperators(
        wavefunction, True, core.Matrix.from_array(points), int(max_bytes), int(max_points))
    if operators.nocc != provider.nocc or operators.nvir != provider.nvir:
        raise ValueError('point-charge operator context differs from the native response context')
    # Declaration hygiene, not verification: both sides are fixed literals, so
    # this can only catch a future provider that changes its declared order. The
    # actual t=a*nocc+i packing is verified against an independent MO transform
    # in tests/pytests/test_isapol_native_point_response.py.
    if operators.representation != REPRESENTATION or operators.ov_order != provider.ov_order:
        raise ValueError('point-charge operator representation/order declaration mismatch')
    legs = operators.operators().to_array()
    if legs.shape != (nov, points.shape[0]):
        raise ValueError('point-charge operator shape does not match the native OV context')
    solver = FDDSFullOVResponse(h1_baseline=h1, h2=h2, transition_legs=legs,
                                coupling=np.zeros((points.shape[0], points.shape[0])),
                                representation='supplied_transition_leg_coordinates')
    responses, targets, defects, diagonals, magnitudes = [], [], [], [], []
    for omega in omegas:
        raw = np.asarray(solver.at_frequency(float(omega)).raw_coupled)
        if raw.shape != (points.shape[0], points.shape[0]) or not np.isfinite(raw).all():
            raise ValueError('nonfinite or misshaped point-charge response')
        value = -raw
        value.flags.writeable = False
        responses.append(value)
        # Packed lower triangle of the COMPUTED matrix; the residual asymmetry is
        # reported, never averaged away.
        packed = np.array([value[i, j] for i in range(value.shape[0]) for j in range(i + 1)])
        packed.flags.writeable = False
        targets.append(packed)
        defects.append(float(np.max(np.abs(value - value.T))))
        diagonals.append(float(np.min(np.diag(value))))
        magnitudes.append(float(np.max(np.abs(value))))
    points.flags.writeable = False
    digest = hashlib.sha256()
    for block in (points, np.ascontiguousarray(omegas), np.ascontiguousarray(legs),
                  np.ascontiguousarray(h1), np.ascontiguousarray(h2),
                  np.ascontiguousarray(provider.orbitals().to_array()),
                  np.ascontiguousarray(np.asarray(provider.energies()))):
        digest.update(np.ascontiguousarray(block, dtype=float).tobytes())
    for text in (CONVENTION, REPRESENTATION, operators.convention, provider.kernel,
                 solver.representation, provider.ov_order,
                 str(response.correction_provenance), str(response.convergence_evidence),
                 repr(bool(response.caller_converged))):
        digest.update(text.encode())
    for number in (provider.exact_exchange, provider.local_scale, provider.density_cutoff):
        digest.update(np.float64(number).tobytes())
    record = (
        'native direct-OV point-charge response; v=-W^T C W; '
        f'kernel={provider.kernel}; a={provider.exact_exchange!r}; b={provider.local_scale!r}; '
        f'nocc={provider.nocc}; nvir={provider.nvir}; npoint={points.shape[0]}; '
        f'frequencies_au={[float(w) for w in omegas]!r}; '
        f'legs={operators.convention}; not fitted, not constrained-NN, not refined')
    return NativePointChargeResponse(
        operators=operators, points_bohr=points,
        frequencies_au=tuple(float(w) for w in omegas), responses=tuple(responses),
        packed_targets=tuple(targets), reciprocity_defects=tuple(defects),
        minimum_diagonals=tuple(diagonals), maximum_absolute_values=tuple(magnitudes),
        convention=CONVENTION, representation=REPRESENTATION, generation_record=record,
        context_sha256=digest.hexdigest(),
        correction_provenance=response.correction_provenance,
        caller_converged=bool(response.caller_converged),
        convergence_evidence=response.convergence_evidence)
