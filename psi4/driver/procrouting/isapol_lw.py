# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Expert supplied-nonlocal -> C++ LW -> local properties, not oeprop/SCF.

No input IO, graph inference, PFIT, tensor repair, or imported-ORIENT model adapter.
Multipole tensors use real Racah 00,10,11c,11s,... order (dipoles z,x,y).
The global_dipoles property instead uses Cartesian x,y,z on both component axes.
The explicit array axes are frequency,response_site,potential_site,response_component,
potential_component; both site axes follow labels. Origins are bohr; frequencies
are nonnegative imaginary-axis magnitudes in atomic units. Frames are proper
local-to-global Cartesian columns; frames=None explicitly defaults every site to
identity. Rank4 requires truncation='discard_rank4_rows_and_columns': exactly the
first16 rows/columns enter LW, including rank0. Output contains ranks1..3 only.

Owned arrays are immutable byte snapshots; .array returns a defensive copy.
Core imports are lazy and package-relative, with no dependency on isapol_supplied.
The only relaxed diagnostic is pinned to the complete historical water snapshot;
its success does not imply production acceptance, external parity, or native prediction.
"""
from dataclasses import dataclass, field
import hashlib
import math
import re
from typing import Optional, Sequence, Union

import numpy as np

NumericArray = Union[np.ndarray, list, tuple]

TRUNCATE_RANK4 = 'discard_rank4_rows_and_columns'
PRODUCTION_TOLERANCE = 1e-6
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_FREQUENCIES = 4096
HISTORICAL_FIXTURE_SHA256 = 'b86d411e5fd81fc20358370ee211ba5b9bd997c57f8918c32ce0334c93ad7f49'
HISTORICAL_SOURCE_SHA256 = '9b6130f42fc50b50b80d5860f13cc6b5b198002c4c74a1b3500893481502c166'
# Canonical little-endian float64, C order, INCLUDING discarded rank4 and signed zeros.
_HISTORICAL_ARRAY_HASHES = (
    'd756966bfcff85ba02b05c279bba6a88eb5e177a224e6e3742d14953e826cec1',
    '7693a87f99b542e76639c9c9d53420c2ffceec8638181dd568eb1aa1852e006c',
    '99ffae29f099ca7f7fb81f914e8dc3f525e6bddd3ddcc87429c9acef9ab0930b',
    'af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc',
)


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 4096:
        raise ValueError(name + ' must be nonempty text of at most4096 characters')
    return value


def _length(value, name, maximum):
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(name + ' must be a bounded list, tuple, or ndarray (not an iterator)')
    try:
        n = len(value)
    except TypeError as exc:
        raise ValueError(name + ' must be a sequence') from exc
    if n > maximum:
        raise ValueError(name + ' resource limit exceeded')
    return n


def _shape_check(value, shape, name):
    # Preflight nested lists BEFORE np.asarray can allocate an oversized/ragged array.
    if isinstance(value, np.ndarray):
        if value.shape != shape or value.dtype.kind not in 'fiu':
            raise ValueError(name + ' has invalid shape or non-real numeric dtype')
    elif shape:
        if _length(value, name, shape[0]) != shape[0]:
            raise ValueError(name + ' has invalid shape')
        for row in value:
            _shape_check(row, shape[1:], name)
    elif isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(name + ' requires real numeric scalars')


def _array(value, shape, name):
    _shape_check(value, shape, name)
    try:
        a = np.array(value, dtype='<f8', order='C', copy=True)
    except (ValueError, OverflowError, TypeError) as exc:
        raise ValueError(name + ' cannot be represented as float64') from exc
    if not np.isfinite(a).all():
        raise ValueError(name + ' must be finite')
    return a


def _hash(a):
    return hashlib.sha256(a.tobytes(order='C')).hexdigest()


@dataclass(frozen=True)
class ArraySnapshot:
    """Portable float64 bytes; no writable array is held by a result."""
    shape: tuple[int, ...]
    data: bytes

    def __post_init__(self):
        shape = tuple(self.shape)
        if any(type(n) is not int or n < 0 for n in shape) or math.prod(shape)*8 != len(self.data):
            raise ValueError('invalid snapshot dimensions')
        object.__setattr__(self, 'shape', shape)
        object.__setattr__(self, 'data', bytes(self.data))

    @classmethod
    def of(cls, a):
        a = np.asarray(a, dtype='<f8', order='C')
        if not np.isfinite(a).all():
            raise ValueError('nonfinite computed property')
        return cls(a.shape, a.tobytes())

    @property
    def array(self) -> np.ndarray:
        return np.frombuffer(self.data, dtype='<f8').reshape(self.shape).copy()

    @property
    def canonical_array_sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


@dataclass(frozen=True)
class Provenance:
    """Caller declarations, not verified producer claims. source_sha256 hashes the
    original source artifact, NOT the canonical float64 array (separately recorded).
    """
    source_name: str
    source_sha256: str
    producer: str
    description: str

    def __post_init__(self):
        for key in ('source_name', 'producer', 'description'):
            _text(getattr(self, key), key)
        if not isinstance(self.source_sha256, str) or not re.fullmatch('[0-9a-f]{64}', self.source_sha256):
            raise ValueError('source_sha256 must be a lowercase SHA256 declaration')


@dataclass(frozen=True)
class Residuals:
    off_site: float
    charge_sum: float
    reciprocity: float
    molecular_sum: float
    local_charge: float
    input_sum_rule: float
    charge_sum_transport: float

    @property
    def maximum(self) -> float:
        """Every residual, algorithm-controlled and supplied-input alike."""
        return max(self.off_site, self.charge_sum, self.reciprocity, self.molecular_sum,
                   self.local_charge, self.input_sum_rule, self.charge_sum_transport)

    @property
    def algorithm_maximum(self) -> float:
        """Only what LW is responsible for; excludes the supplied input's sum-rule defect.

        charge_sum and local_charge are omitted because they reproduce
        input_sum_rule: charge_sum_transport measures the part LW owns.
        """
        return max(self.off_site, self.reciprocity, self.molecular_sum, self.charge_sum_transport)


@dataclass(frozen=True)
class FrequencyDiagnostics:
    frequency: float
    residuals: Residuals
    transfer_count: int
    omitted_component_pairs: tuple[tuple[int, int], ...]
    omitted_transfer_count: int
    input_max_reciprocity_error: float
    production_postcondition_passed: bool
    # True when only the supplied input's sum-rule defect keeps the line above False.
    algorithm_postcondition_passed: bool = True

    def __post_init__(self):
        object.__setattr__(self, 'omitted_component_pairs', tuple(map(tuple, self.omitted_component_pairs)))


@dataclass(frozen=True)
class TensorDiagnostic:
    frequency_index: int
    label: str
    axes: str
    max_asymmetry: float
    minimum_symmetric_eigenvalue: float


@dataclass(frozen=True)
class Metadata:
    input_rank: int
    truncation: Optional[str]
    discarded_rank4_entry_count: int
    canonical_input_array_sha256: str
    residual_policy: str
    residual_tolerance: float
    production_postcondition_passed: bool
    historical_fixture_sha256: Optional[str]
    mode: str = 'supplied_nonlocal'
    tensor_origin: str = 'Psi4_LW'
    wavefunction_status: str = 'no_native_wavefunction'
    refinement_status: str = 'no_PFIT'
    native_verified: bool = False
    numerical_agreement: Optional[bool] = None
    anisotropic_status: str = 'separate_explicit_placement_adapter_available_not_computed'
    input_axes: str = 'frequency,response_site,potential_site,response_component,potential_component; global'
    output_axes: str = 'raw_global/raw_local: frequency,site,response_component,potential_component; ranks1..3'
    scalar_axes: str = 'frequency,site,rank; trace(alpha_ll)/(2*l+1), ranks1,2,3'
    dipole_axes: str = 'frequency,site,response_xyz,potential_xyz; global Cartesian x,y,z'
    frame_convention: str = 'local_to_global_columns; local=D(F).T @ global @ D(F)'
    input_components: tuple[str, ...] = field(init=False)
    local_components: tuple[str, ...] = field(init=False)
    output_ranks: tuple[int, ...] = (1, 2, 3)
    coverage: str = 'rank0..3 working; rank1..3 local; rank4 input does not restore missing C12 (1,4)/(4,1)'
    units: str = 'origins: bohr; xi: Eh; alpha_llprime: bohr^(l+lprime+1); dipoles: bohr^3'
    hash_convention: str = 'canonical_input_array_sha256: little-endian float64 C-order full input; source_sha256: caller-declared original source bytes'

    def __post_init__(self):
        components = tuple(c for l in range(self.input_rank+1) for c in [f'{l}0'] +
                           [f'{l}{m}{s}' for m in range(1,l+1) for s in 'cs'])
        object.__setattr__(self, 'input_components', components)
        object.__setattr__(self, 'local_components', components[1:16])
        object.__setattr__(self, 'output_ranks', tuple(self.output_ranks))


@dataclass(frozen=True)
class LocalProperties:
    """Trusted factory-produced result, not a deserialization/validation boundary.

    Consumers require supplied_nonlocal_properties output. Manually constructed or
    dataclasses.replace-altered scientific records are not certified factory results.
    """
    labels: tuple[str, ...]
    origins: ArraySnapshot
    frames: ArraySnapshot
    frequencies: tuple[float, ...]
    bonds: tuple[tuple[int, int], ...]
    provenance: Provenance
    raw_input: ArraySnapshot
    raw_global: ArraySnapshot
    raw_local: ArraySnapshot
    atomic_scalars: ArraySnapshot
    global_dipoles: ArraySnapshot
    frequency_diagnostics: tuple[FrequencyDiagnostics, ...]
    tensor_diagnostics: tuple[TensorDiagnostic, ...]
    warnings: tuple[str, ...]
    metadata: Metadata

    def __post_init__(self):
        for name in ('labels', 'frequencies', 'frequency_diagnostics', 'tensor_diagnostics', 'warnings'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, 'bonds', tuple(map(tuple, self.bonds)))
        for name in ('origins', 'frames', 'raw_input', 'raw_global', 'raw_local', 'atomic_scalars', 'global_dipoles'):
            if not isinstance(getattr(self, name), ArraySnapshot):
                raise ValueError(name + ' requires an immutable ArraySnapshot')
        if not isinstance(self.provenance, Provenance) or not isinstance(self.metadata, Metadata):
            raise ValueError('typed provenance and metadata required')


def supplied_nonlocal_properties(*, labels: Sequence[str], origins: NumericArray,
                                bonds: Sequence[Sequence[int]], frequencies: Sequence[float],
                                tensors: NumericArray, input_rank: int, provenance: Provenance,
                                frames: Optional[NumericArray] = None,
                                truncation: Optional[str] = None,
                                residual_policy: str = 'production') -> LocalProperties:
    """Localize each supplied frequency with core.isa_localize_lw only.

    Static-only needs neither CP weights nor a partner. No generic tolerance knob.
    Three residual policies, none of which loosens the1e-6 algorithm gate:
      production                  - single combined1e-6 gate over every residual.
      reported_input_sum_rule     - 1e-6 on everything LW controls (off_site,
            reciprocity, molecular_sum, charge_sum_transport); the supplied data's
            charge-flow sum-rule defect is measured, warned about and reported. LW
            transports that defect exactly, so gating it would gate the producer of
            the supplied pair data rather than this routine.
      historical_water_diagnostic - the one genuinely relaxed path (1e-3), restricted
            by pinned full input/geometry/frame/frequency bytes and exact labels/bond
            order, and recording failed production acceptance independently of any
            comparison.
    Input storage is capped at64MiB, frequencies at4096, sites at256. Conservative
    native workspace admission mirrors the current768MiB core budget, which core
    also enforces. These bounds are NOT a total process RSS guarantee. Counting
    transfers uses the existing binding's copied list (bounded by the core's
    one-million-transfer cap); no new count-only binding is introduced here.
    """
    n = _length(labels, 'labels', 256)
    if not n:
        raise ValueError('at least one site required')
    labels = tuple(_text(s, 'label') for s in labels)
    if len(set(labels)) != n:
        raise ValueError('labels must be unique')
    if type(input_rank) is not int or input_rank not in (3, 4):
        raise ValueError('input_rank must be explicit integer3 or4')
    if truncation != (TRUNCATE_RANK4 if input_rank == 4 else None):
        raise ValueError('exact rank4 truncation declaration required only for rank4')
    if not isinstance(provenance, Provenance):
        raise ValueError('explicit typed Provenance required')
    if residual_policy not in ('production', 'historical_water_diagnostic',
                              'reported_input_sum_rule'):
        raise ValueError('unsupported residual policy')
    edges = _length(bonds, 'bonds', n*(n-1)//2)
    graph, seen = [], set()
    for edge in bonds:
        if _length(edge, 'bond', 2) != 2 or any(type(i) is not int for i in edge):
            raise ValueError('bond requires two zero-based Python integer indices')
        a, b = edge
        if not (0 <= a < n and 0 <= b < n) or a == b or (min(a,b), max(a,b)) in seen:
            raise ValueError('invalid self/duplicate/out-of-range bond')
        seen.add((min(a,b), max(a,b)))
        graph.append((a,b))
    graph = tuple(graph)
    # IsaLwWorkingMatrix=2048, local=1800, position=24, transfer<=48 bytes.
    native_bytes = 3*n*n*2048 + (2*edges+n)*2048 + n*(1800+24) + 16*n*n*8 + 4*1000000*48
    if native_bytes > 768*1024*1024:
        raise ValueError('native workspace exceeds768MiB resource limit')
    nf = _length(frequencies, 'frequencies', MAX_FREQUENCIES)
    m = (input_rank+1)**2
    if not nf or nf*n*n*m*m*8 > MAX_INPUT_BYTES:
        raise ValueError('empty frequency grid or input exceeds64MiB resource limit')
    freq = _array(frequencies, (nf,), 'frequencies')
    if np.any(freq < 0) or np.any(freq[1:] <= freq[:-1]):
        raise ValueError('frequencies must be strictly increasing nonnegative')
    pos = _array(origins, (n,3), 'origins')
    frame = _array(np.tile(np.eye(3), (n,1,1)) if frames is None else frames, (n,3,3), 'frames')
    for f in frame:
        with np.errstate(over='raise', invalid='raise'):
            if not np.allclose(f.T @ f, np.eye(3), atol=1e-12, rtol=0) or abs(np.linalg.det(f)-1) > 1e-12:
                raise ValueError('frames must be proper local-to-global rotations')
    raw = _array(tensors, (nf,n,n,m,m), 'tensors')
    raw_hash = _hash(raw)
    historical = residual_policy == 'historical_water_diagnostic'
    reported_sum_rule = residual_policy == 'reported_input_sum_rule'
    if historical and not (
        input_rank == 4 and labels == ('O','H1','H2') and graph == ((0,1),(0,2))
        and raw.shape == (1,3,3,25,25)
        and (raw_hash, _hash(pos), _hash(frame), _hash(freq)) == _HISTORICAL_ARRAY_HASHES
        and provenance.source_sha256 == HISTORICAL_SOURCE_SHA256
    ):
        raise ValueError('historical_water_diagnostic requires the exact approved water identity (full tensors, geometry, frames, frequency, ordered bonds, source hash)')
    from psi4 import core
    rotations = [np.asarray(core.isa_multipole_rotation(3, f.tolist()))[1:16,1:16].copy() for f in frame]
    globals_, locals_, scalars, dipoles, fd, td, warnings = [], [], [], [], [], [], []
    for k, xi in enumerate(freq):
        blocks = [core.Matrix.from_array(raw[k,a,b,:16,:16].copy()) for a in range(n) for b in range(n)]
        args = (core.Matrix.from_array(pos), blocks, float(xi), graph)
        # Never catch/retry a failed production request with relaxed tolerance.
        if historical:
            result = core.isa_localize_lw(*args, 1e-3)
        elif reported_sum_rule:
            # Production 1e-6 on everything LW controls; the supplied data's own
            # charge-flow sum-rule defect is measured and reported, not gated.
            result = core.isa_localize_lw(*args, PRODUCTION_TOLERANCE, math.inf)
        else:
            result = core.isa_localize_lw(*args)
        residuals = Residuals(*(float(getattr(result.residuals, name)) for name in Residuals.__dataclass_fields__))
        g = np.array([np.asarray(a) for a in result.local])
        with np.errstate(over='raise', invalid='raise'):
            local = np.array([d.T @ a @ d for d,a in zip(rotations,g)])
            scalar = np.array([[np.trace(a[l*l-1:(l+1)**2-1,l*l-1:(l+1)**2-1])/(2*l+1) for l in (1,2,3)] for a in local])
            xyz = g[:,:3,:3][:,[1,2,0],:][:,:,[1,2,0]].copy()
            input_error = float(np.max(np.abs(raw[k]-raw[k].transpose(1,0,3,2))))
            for axes, arrays in (('global',g), ('site_local',local)):
                for label,a in zip(labels,arrays):
                    asym = float(np.max(np.abs(a-a.T)))
                    eig = float(np.linalg.eigvalsh(a/2+a.T/2)[0])
                    if not np.isfinite(eig):
                        raise ValueError('nonfinite passivity diagnostic')
                    td.append(TensorDiagnostic(k,label,axes,asym,eig))
                    if asym:
                        warnings.append(f'{label}[{k}] {axes}: asymmetric raw tensor ({asym:g}); no symmetrization; strict anisotropic conversion unsupported')
                    if eig < 0:
                        warnings.append(f'{label}[{k}] {axes}: indefinite symmetric part ({eig:g}); no clipping')
        if input_error:
            warnings.append(f'input[{k}] raw reciprocity error {input_error:g}; retained without repair')
        fd.append(FrequencyDiagnostics(float(xi), residuals, len(result.transfers),
                  tuple(tuple(p) for p in result.omitted_component_pairs), result.omitted_transfer_count,
                  input_error, residuals.maximum <= PRODUCTION_TOLERANCE,
                  residuals.algorithm_maximum <= PRODUCTION_TOLERANCE))
        if reported_sum_rule and residuals.input_sum_rule > PRODUCTION_TOLERANCE:
            warnings.append(
                f'input[{k}] supplied charge-flow sum-rule defect {residuals.input_sum_rule:g} exceeds '
                f'{PRODUCTION_TOLERANCE:g}; reported not gated, transported by LW at '
                f'{residuals.charge_sum_transport:g} and not repaired')
        globals_.append(g); locals_.append(local); scalars.append(scalar); dipoles.append(xyz)
    if historical:
        warnings.append('Historical diagnostic only: production postcondition failed; no native prediction or external parity claim.')
    if reported_sum_rule:
        warnings.append('Supplied charge-flow sum-rule defect reported, not gated: the algorithm-controlled '
                        'postconditions held at the production tolerance, the input sum rule is a property of '
                        'the supplied producer, and no native prediction or external parity is claimed.')
    metadata = Metadata(input_rank, truncation, nf*n*n*(m*m-256), raw_hash, residual_policy,
                        1e-3 if historical else PRODUCTION_TOLERANCE,
                        False if historical else all(d.production_postcondition_passed for d in fd),
                        HISTORICAL_FIXTURE_SHA256 if historical else None)
    return LocalProperties(labels, ArraySnapshot.of(pos), ArraySnapshot.of(frame), tuple(map(float,freq)), graph,
                           provenance, ArraySnapshot.of(raw), ArraySnapshot.of(globals_), ArraySnapshot.of(locals_),
                           ArraySnapshot.of(scalars), ArraySnapshot.of(dipoles), tuple(fd), tuple(td), tuple(warnings), metadata)


@dataclass(frozen=True)
class Coefficient:
    order: int
    value: float
    included_rank_pairs: tuple[tuple[int, int], ...]
    missing_rank_pairs: tuple[tuple[int, int], ...]
    unrestricted_complete: bool

    def __post_init__(self):
        object.__setattr__(self, 'included_rank_pairs', tuple(map(tuple, self.included_rank_pairs)))
        object.__setattr__(self, 'missing_rank_pairs', tuple(map(tuple, self.missing_rank_pairs)))


@dataclass(frozen=True)
class DispersionPair:
    site_a: int
    site_b: int
    label_a: str
    label_b: str
    coefficients: tuple[Coefficient, ...]

    def __post_init__(self):
        object.__setattr__(self, 'coefficients', tuple(self.coefficients))


@dataclass(frozen=True)
class IsotropicDispersion:
    model_a: LocalProperties
    model_b: LocalProperties
    cp_weights: tuple[float, ...]
    quadrature_provenance: Provenance
    pairs: tuple[DispersionPair, ...]
    units: str = 'C_n: Eh bohr^n; cp_weights includes Jacobian and1/(2*pi)'
    origin: str = 'Psi4_isotropic_dispersion_from_Psi4_LW'
    anisotropic_status: str = 'separate_explicit_placement_adapter_available_not_computed'

    def __post_init__(self):
        object.__setattr__(self, 'cp_weights', tuple(self.cp_weights))
        object.__setattr__(self, 'pairs', tuple(self.pairs))


def isotropic_dispersion(model_a: LocalProperties, model_b: LocalProperties, *,
                         cp_weights: Sequence[float], quadrature_provenance: Provenance,
                         max_order: int = 12) -> IsotropicDispersion:
    """Explicit A/B local results and CP weights, no invented static quadrature.

    Site sets may differ; frequency grids must match exactly, as in the C++ contract.
    Weights include the Jacobian and1/(2*pi) exactly once. No interpolation, no
    anisotropic conversion, and no Python duplication of the C_n formula.
    """
    if not isinstance(model_a, LocalProperties) or not isinstance(model_b, LocalProperties):
        raise ValueError('dispersion requires typed LW local results')
    if not isinstance(quadrature_provenance, Provenance):
        raise ValueError('explicit quadrature Provenance required')
    if model_a.frequencies != model_b.frequencies:
        raise ValueError('A/B frequency grids must match exactly; no interpolation')
    if type(max_order) is not int or max_order not in (6,8,10,12):
        raise ValueError('max_order must be6,8,10,12')
    weights = _array(cp_weights, (len(model_a.frequencies),), 'cp_weights')
    if np.any(weights < 0) or not np.any(weights > 0) or any(x == 0 and w != 0 for x,w in zip(model_a.frequencies, weights)):
        raise ValueError('CP weights require positive dynamic weight and zero static weight')
    from psi4 import core
    def convert(model):
        sites = []
        origins, scalars = model.origins.array, model.atomic_scalars.array
        for j,label in enumerate(model.labels):
            site = core.IsaIsotropicSite()
            site.label, site.origin, site.ranks = label, origins[j].tolist(), [1,2,3]
            site.polarizabilities = core.Matrix.from_array(scalars[:,j,:])
            sites.append(site)
        provenance = f'Psi4_LW; source_sha256={model.provenance.source_sha256}; canonical_input_array_sha256={model.metadata.canonical_input_array_sha256}; policy={model.metadata.residual_policy}'
        return core.IsaIsotropicModel(list(model.frequencies), sites, provenance)
    result = core.isa_isotropic_dispersion(convert(model_a), convert(model_b), weights.tolist(), max_order)
    pairs = tuple(DispersionPair(p.site_a, p.site_b, model_a.labels[p.site_a], model_b.labels[p.site_b],
                  tuple(Coefficient(c.order, c.value, tuple(map(tuple,c.included_rank_pairs)),
                                    tuple(map(tuple,c.missing_rank_pairs)), c.complete) for c in p.coefficients)) for p in result.pairs)
    return IsotropicDispersion(model_a, model_b, tuple(map(float,weights)), quadrature_provenance, pairs)


@dataclass(frozen=True)
class Placement:
    """Whole-model active Cartesian placement: r_placed = rotation @ r + translation.

    Both arrays are explicit, bounded immutable snapshots. Translation is in bohr;
    rotation is proper and is NOT a multipole tensor transformation in Python.
    """
    rotation: ArraySnapshot
    translation: ArraySnapshot

    def __post_init__(self):
        for value, shape in ((self.rotation, (3,3)), (self.translation, (3,))):
            if isinstance(value, ArraySnapshot) and value.shape != shape:
                raise ValueError('placement snapshot has invalid bounded shape')
        r = _array(self.rotation.array if isinstance(self.rotation, ArraySnapshot) else self.rotation,
                   (3,3), 'placement rotation')
        t = _array(self.translation.array if isinstance(self.translation, ArraySnapshot) else self.translation,
                   (3,), 'placement translation')
        with np.errstate(over='ignore', invalid='ignore'):
            if (not np.allclose(r.T @ r, np.eye(3), atol=1e-12, rtol=0)
                    or not np.isfinite(np.linalg.det(r)) or abs(np.linalg.det(r)-1) > 1e-12):
                raise ValueError('placement rotation must be proper Cartesian rotation')
        object.__setattr__(self, 'rotation', ArraySnapshot.of(r))
        object.__setattr__(self, 'translation', ArraySnapshot.of(t))


@dataclass(frozen=True)
class PlacedGeometry:
    origins: ArraySnapshot
    source_frames: ArraySnapshot
    component_frames: ArraySnapshot
    core_provenance: str
    ranks: tuple[int, ...] = (1, 2, 3)
    declaration: str = 'supplied_local_response'
    tensor_axes: str = 'raw_global; real Racah ranks1..3; component frame is placement rotation'


@dataclass(frozen=True)
class AnisotropicCoefficient:
    order: int
    value: float
    energy: float
    included_rank_quadruples: tuple[tuple[int, int, int, int], ...]
    missing_rank_quadruples: tuple[tuple[int, int, int, int], ...]
    declared_model_complete: bool
    unrestricted_complete: bool

    def __post_init__(self):
        for name in ('included_rank_quadruples', 'missing_rank_quadruples'):
            object.__setattr__(self, name, tuple(map(tuple, getattr(self, name))))


@dataclass(frozen=True)
class AnisotropicPair:
    site_a: int
    site_b: int
    label_a: str
    label_b: str
    distance: float
    displacement: tuple[float, float, float]
    direction: tuple[float, float, float]
    coefficients: tuple[AnisotropicCoefficient, ...]
    truncated_energy: float

    def __post_init__(self):
        for name in ('displacement', 'direction', 'coefficients'):
            object.__setattr__(self, name, tuple(getattr(self, name)))


@dataclass(frozen=True)
class AnisotropicDispersion:
    model_a: LocalProperties
    model_b: LocalProperties
    placement_a: Placement
    placement_b: Placement
    placed_a: PlacedGeometry
    placed_b: PlacedGeometry
    frequencies: tuple[float, ...]
    cp_weights: tuple[float, ...]
    quadrature_provenance: Provenance
    max_order: int
    pairs: tuple[AnisotropicPair, ...]
    truncated_energy: float
    origin: str = 'Psi4_anisotropic_dispersion_from_Psi4_LW'
    kind: str = 'orientation_resolved_scalars_not_recoupled_components'
    units: str = 'origins/distance: bohr; C_n: Eh bohr^n; energy: Eh; cp_weights includes Jacobian and1/(2*pi)'

    def __post_init__(self):
        for name in ('frequencies', 'cp_weights', 'pairs'):
            object.__setattr__(self, name, tuple(getattr(self, name)))


def anisotropic_dispersion(model_a: LocalProperties, model_b: LocalProperties, *,
                           placement_a: Placement, placement_b: Placement,
                           cp_weights: Sequence[float], quadrature_provenance: Provenance,
                           max_order: int = 12) -> AnisotropicDispersion:
    """Expert adapter for trusted factory-produced LocalProperties (see that type).

    Both entire models require explicit placements, including an explicit identity
    for an unmoved model. Keep source results/input/provenance intact. raw_global
    is already expressed in identity component frames: core receives those exact
    bytes with frame=placement.rotation. Core alone rotates multipoles and contracts
    responses; raw_local is never roundtripped, symmetrized, or used as a fallback.
    Placed source frames are metadata only. Python performs Cartesian geometry
    placement, not multipole translation/localization or dispersion numerics.

    Exact reciprocity is mandatory, including roundoff-size defects. No tolerance
    policy override, tensor repair, isotropic fallback, interpolation, or invented
    quadrature is available. The historical static example cannot provide dynamic
    weights and remains unsupported. This is not molecular-frequency acceptance.
    Existing C++ model/pair budgets remain authoritative for core resources.
    """
    if not isinstance(model_a, LocalProperties) or not isinstance(model_b, LocalProperties):
        raise ValueError('dispersion requires typed LW local results')
    if not isinstance(placement_a, Placement) or not isinstance(placement_b, Placement):
        raise ValueError('explicit typed A/B Placement required')
    if not isinstance(quadrature_provenance, Provenance):
        raise ValueError('explicit quadrature Provenance required')
    if type(max_order) is not int or not 6 <= max_order <= 12:
        raise ValueError('max_order must be a Python integer6..12, including odd orders')
    if model_a.frequencies != model_b.frequencies:
        raise ValueError('A/B frequency grids must match exactly; no interpolation')
    nf = _length(model_a.frequencies, 'frequencies', MAX_FREQUENCIES)
    weights = _array(cp_weights, (nf,), 'cp_weights')
    if (not nf or np.any(weights < 0) or not np.any(weights > 0)
            or any(x == 0 and w != 0 for x,w in zip(model_a.frequencies, weights))):
        raise ValueError('CP weights require positive dynamic weight and zero static weight')
    from psi4 import core

    def convert(model, placement):
        raw = model.raw_global.array
        if not np.array_equal(raw, raw.transpose(0,1,3,2)):
            raise ValueError('strict anisotropic conversion requires exact reciprocity; no repair or fallback')
        rotation, translation = placement.rotation.array, placement.translation.array
        with np.errstate(over='raise', invalid='raise'):
            origins = model.origins.array @ rotation.T + translation
            frames = rotation @ model.frames.array
        placed_origins, placed_frames = ArraySnapshot.of(origins), ArraySnapshot.of(frames)
        sites = []
        for j,label in enumerate(model.labels):
            site = core.IsaAnisotropicSite()
            site.label, site.origin, site.ranks = label, origins[j].tolist(), [1,2,3]
            site.frame = rotation.tolist()
            site.responses = [core.Matrix.from_array(a) for a in raw[:,j]]
            sites.append(site)
        provenance = (f'Psi4_LW; source_sha256={model.provenance.source_sha256}; '
                      f'canonical_input_array_sha256={model.metadata.canonical_input_array_sha256}; '
                      f'policy={model.metadata.residual_policy}; raw_global; '
                      f'placement_rotation_sha256={placement.rotation.canonical_array_sha256}; '
                      f'placement_translation_sha256={placement.translation.canonical_array_sha256}')
        native = core.IsaAnisotropicModel(list(model.frequencies), sites, 'supplied_local_response', provenance)
        geometry = PlacedGeometry(placed_origins, placed_frames,
                                 ArraySnapshot.of(np.tile(rotation, (len(sites),1,1))), native.provenance)
        return native, geometry

    a, placed_a = convert(model_a, placement_a)
    b, placed_b = convert(model_b, placement_b)
    result = core.isa_anisotropic_dispersion(a, b, weights.tolist(), max_order)
    pairs = tuple(AnisotropicPair(
        p.site_a, p.site_b, model_a.labels[p.site_a], model_b.labels[p.site_b],
        p.distance, tuple(p.displacement), tuple(p.direction),
        tuple(AnisotropicCoefficient(c.order, c.value, c.energy,
              tuple(map(tuple,c.included_rank_quadruples)), tuple(map(tuple,c.missing_rank_quadruples)),
              c.declared_model_complete, c.unrestricted_complete) for c in p.coefficients),
        p.truncated_energy) for p in result.pairs)
    return AnisotropicDispersion(model_a, model_b, placement_a, placement_b, placed_a, placed_b,
                                 tuple(result.frequencies), tuple(result.cp_weights), quadrature_provenance,
                                 result.max_order, pairs, result.truncated_energy)
