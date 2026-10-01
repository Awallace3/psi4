# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Nonlocal response -> C++ LW -> owned local tensors.

No input IO, graph inference, PFIT or tensor repair.
Multipole tensors use real Racah 00,10,11c,11s,... order (dipoles z,x,y);
global_dipoles uses Cartesian x,y,z on both component axes.
Array axes are frequency, response_site, potential_site, response_component,
potential_component; both site axes follow labels. Origins are bohr; frequencies
are nonnegative imaginary-axis magnitudes in atomic units. Frames are proper
local-to-global Cartesian columns; frames=None means identity for every site.
Rank-4 input needs an explicit truncation, and the choices are different models:
'discard_rank4_rows_and_columns' sends the first 16 rows/columns (including rank
0) and outputs ranks 1..3; 'retain_rank4_rows_and_columns' sends all 25 and
outputs ranks 1..4, the only route to the C12 (1,4)/(4,1) rank pairs.

Owned arrays are immutable byte snapshots; .array returns a defensive copy.
Core imports are lazy and package-relative.

LW is the Lillestolen--Wheatley localization (Lillestolen and Wheatley,
J. Phys. Chem. A 111, 11141 (2007); CamCASP user's guide section 8.2.1), as
implemented in ORIENT, whose output this reference-model implementation
reproduces. It is NOT rotationally covariant: rotating the whole molecule
(origins and tensors together) can change the local LW tensors and the rank>=2
isotropic scalars that later feed C8/C10. Rank-1 scalars were unchanged in the
sampled cases, but no orientation independence of C6 is guaranteed. The
``production`` residual policy and the ``production_postcondition_passed`` flags
gate input and residual admission only; they are not a guarantee of physical
accuracy or of rotational invariance.
"""
from dataclasses import dataclass, field
import hashlib
import math
import re
from typing import Optional, Sequence, Union

import numpy as np

NumericArray = Union[np.ndarray, list, tuple]

TRUNCATE_RANK4 = 'discard_rank4_rows_and_columns'
RETAIN_RANK4 = 'retain_rank4_rows_and_columns'
# Working/local widths of the core LW matrices, by declared localization rank limit.
_LOCAL_WIDTH = {1: 15, 2: 15, 3: 15, 4: 24}
_INPUT_WIDTH = {TRUNCATE_RANK4: 16, RETAIN_RANK4: 25, None: 16}
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
    localization_rank_limit: int = 3
    # Largest supplied magnitude discarded above the declared limit, both at this
    # boundary (rank-4 rows) and inside core; core reports only the latter.
    localization_truncated_input_maxabs: float = 0.0
    mode: str = 'supplied_nonlocal'
    tensor_origin: str = 'Psi4_LW'
    wavefunction_status: str = 'no_native_wavefunction'
    refinement_status: str = 'no_PFIT'
    native_verified: bool = False
    numerical_agreement: Optional[bool] = None
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
        # Limits 1..3 keep the 15-wide output (ranks above the limit are zeros);
        # limit 4 is 24 wide.
        width = _LOCAL_WIDTH[self.localization_rank_limit]
        object.__setattr__(self, 'local_components', components[1:width+1])
        if self.localization_rank_limit == 4:
            object.__setattr__(self, 'output_ranks', (1, 2, 3, 4))
            object.__setattr__(self, 'output_axes',
                'raw_global/raw_local: frequency,site,response_component,potential_component; ranks1..4')
            object.__setattr__(self, 'scalar_axes',
                'frequency,site,rank; trace(alpha_ll)/(2*l+1), ranks1,2,3,4')
            object.__setattr__(self, 'coverage',
                'rank0..4 working; rank1..4 local; C12 (1,4)/(4,1) available -- a DIFFERENT model '
                'from the rank1..3 localization, not a more accurate one')
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
                                residual_policy: str = 'production',
                                localization_rank_limit: int = 3) -> LocalProperties:
    """Localize each supplied frequency with core.isa_localize_lw only.

    The result depends on the supplied molecular orientation (see the module
    docstring): LW localization output is not rotationally covariant.

    Static-only needs neither CP weights nor a partner. Residual policies:
      production                  - one combined 1e-6 gate over every residual.
      reported_input_sum_rule     - 1e-6 on everything LW controls (off_site,
            reciprocity, molecular_sum, charge_sum_transport); the supplied
            charge-flow sum-rule defect, which LW transports exactly, is measured,
            warned about and reported.
      historical_water_diagnostic - the only relaxed gate (1e-3), admitted only
            for the pinned water input/geometry/frame/frequency bytes and exact
            labels/bond order; it records failed production acceptance.
    ``localization_rank_limit`` (1..4, default 3) is the protocol's single
    ``Limit`` applied to the whole localization (per-site ``WSM-Limit``/``H-Limit``
    belong to PFIT). Limit 4 requires ``truncation=RETAIN_RANK4``. Translation is
    rank-raising, so the result at L equals the result at L' > L restricted to
    ranks 1..L, bitwise; ranks above L are zeros, not computed response.

    Caps: 64 MiB input, 4096 frequencies, 256 sites, and the core's 768 MiB
    workspace and one-million-transfer limits. These do not bound process RSS.
    """
    n = _length(labels, 'labels', 256)
    if not n:
        raise ValueError('at least one site required')
    labels = tuple(_text(s, 'label') for s in labels)
    if len(set(labels)) != n:
        raise ValueError('labels must be unique')
    if type(input_rank) is not int or input_rank not in (3, 4):
        raise ValueError('input_rank must be explicit integer3 or4')
    if input_rank == 4:
        if truncation not in (TRUNCATE_RANK4, RETAIN_RANK4):
            raise ValueError('rank4 input requires an explicit discard/retain rank4 declaration')
    elif truncation is not None:
        raise ValueError('exact rank4 truncation declaration required only for rank4')
    if not isinstance(provenance, Provenance):
        raise ValueError('explicit typed Provenance required')
    if residual_policy not in ('production', 'historical_water_diagnostic',
                              'reported_input_sum_rule'):
        raise ValueError('unsupported residual policy')
    if type(localization_rank_limit) is not int or localization_rank_limit not in (1,2,3,4):
        raise ValueError('localization_rank_limit must be explicit integer1,2,3 or4')
    # Retaining rank-4 input and localizing at rank 4 are one declaration.
    if (truncation == RETAIN_RANK4) != (localization_rank_limit == 4):
        raise ValueError('localization_rank_limit4 requires retained rank4 input, and retained '
                         'rank4 input requires localization_rank_limit4')
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
    # IsaLwWorkingMatrix=25*25*8=5000, local=24*24*8=4608, position=24, transfer<=48 bytes.
    # The core budget is unchanged by the rank4 widening, so the admissible site count
    # falls from256 to about174; that cost is mirrored here, not relaxed.
    native_bytes = 3*n*n*5000 + (2*edges+n)*5000 + n*(4608+24) + 16*n*n*8 + 4*1000000*48
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
    # The frame rotation is built at the rank the localization runs at, never wider:
    # a rotation block for a rank the model does not carry would have nothing to act on.
    rotation_rank = 4 if localization_rank_limit == 4 else 3
    width = _LOCAL_WIDTH[localization_rank_limit]
    input_width = _INPUT_WIDTH[truncation]
    output_ranks = tuple(range(1, rotation_rank+1))
    rotations = [np.asarray(core.isa_multipole_rotation(rotation_rank, f.tolist()))[1:width+1,1:width+1].copy()
                 for f in frame]
    globals_, locals_, scalars, dipoles, fd, td, warnings = [], [], [], [], [], [], []
    # Discarded rank-4 rows never reach core, so their magnitude is measured here.
    truncated_maxabs = 0.0
    if input_width < m:
        truncated_maxabs = max(float(np.max(np.abs(raw[:,:,:,input_width:,:]))),
                               float(np.max(np.abs(raw[:,:,:,:input_width,input_width:]))))
    if localization_rank_limit < rotation_rank:
        # A frame rotation must not mix a retained rank with a discarded one.
        # isa_multipole_rotation is block diagonal in rank; check, do not assume.
        keep = (localization_rank_limit+1)**2 - 1
        for d in rotations:
            if np.any(d[:keep,keep:]) or np.any(d[keep:,:keep]):
                raise ValueError('frame rotation mixes ranks across the declared localization limit')
    for k, xi in enumerate(freq):
        blocks = [core.Matrix.from_array(raw[k,a,b,:input_width,:input_width].copy())
                  for a in range(n) for b in range(n)]
        args = (core.Matrix.from_array(pos), blocks, float(xi), graph)
        # Never catch/retry a failed production request with relaxed tolerance.
        if historical:
            result = core.isa_localize_lw(*args, 1e-3, -1.0, localization_rank_limit)
        elif reported_sum_rule:
            # Production 1e-6 on everything LW controls; the supplied data's own
            # charge-flow sum-rule defect is measured and reported, not gated.
            result = core.isa_localize_lw(*args, PRODUCTION_TOLERANCE, math.inf,
                                          localization_rank_limit)
        else:
            result = core.isa_localize_lw(*args, PRODUCTION_TOLERANCE, -1.0,
                                          localization_rank_limit)
        if result.localization_rank_limit != localization_rank_limit:
            raise ValueError('core reported a different localization rank limit')
        truncated_maxabs = max(truncated_maxabs, float(result.truncated_input_maxabs))
        residuals = Residuals(*(float(getattr(result.residuals, name)) for name in Residuals.__dataclass_fields__))
        # Core always returns the full 24-wide local block; slicing to the declared
        # width keeps a rank<=3 model's arrays byte-identical to the rank-3 pipeline.
        g = np.array([np.asarray(a)[:width,:width] for a in result.local])
        with np.errstate(over='raise', invalid='raise'):
            local = np.array([d.T @ a @ d for d,a in zip(rotations,g)])
            scalar = np.array([[np.trace(a[l*l-1:(l+1)**2-1,l*l-1:(l+1)**2-1])/(2*l+1)
                                for l in output_ranks] for a in local])
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
                        warnings.append(f'{label}[{k}] {axes}: asymmetric raw tensor ({asym:g}); no symmetrization')
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
    if localization_rank_limit == 4:
        warnings.append(
            'localization declared at rank4: the local tensors carry ranks1..4 and the C12 (1,4)/(4,1) '
            'rank pairs become available. This is a DIFFERENT model from the rank1..3 localization, not '
            'a more accurate one, and the two must never be quoted as agreeing.')
    # Warn on input rank, not rotation_rank, so rank-4 input localized at rank 3
    # is still announced.
    if localization_rank_limit < input_rank:
        warnings.append(
            f'localization declared at rank {localization_rank_limit}: ranks '
            f'{localization_rank_limit+1}..{input_rank} are absent by declaration, not computed and small; '
            f'{truncated_maxabs:g} of supplied magnitude was truncated before localizing. The '
            f'restriction is exact (translation is rank-raising), so this equals the '
            f'rank-{input_rank} localization restricted to ranks 1..{localization_rank_limit} and '
            f'cannot change any number at those ranks.')
    metadata = Metadata(input_rank, truncation, nf*n*n*(m*m-input_width*input_width), raw_hash, residual_policy,
                        1e-3 if historical else PRODUCTION_TOLERANCE,
                        False if historical else all(d.production_postcondition_passed for d in fd),
                        HISTORICAL_FIXTURE_SHA256 if historical else None,
                        localization_rank_limit, truncated_maxabs)
    return LocalProperties(labels, ArraySnapshot.of(pos), ArraySnapshot.of(frame), tuple(map(float,freq)), graph,
                           provenance, ArraySnapshot.of(raw), ArraySnapshot.of(globals_), ArraySnapshot.of(locals_),
                           ArraySnapshot.of(scalars), ArraySnapshot.of(dipoles), tuple(fd), tuple(td), tuple(warnings), metadata)
