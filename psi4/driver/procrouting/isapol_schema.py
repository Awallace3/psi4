# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Strict serializable native monomer inputs and artifact transport.

This seam carries only declared native calculation inputs. Benchmark labels
and SAPT references belong on QCArchive entry attributes, never here.
Installed implementation pins identify code; they do not certify molecular
accuracy, convergence, or an OS-level independence audit.
"""
from dataclasses import asdict, dataclass
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile
import zipfile

import numpy as np

from psi4 import core

from .isapol_bounded import BoundedResources
from .isapol_molecular import MolecularDeclaration, independent_sites, plan_molecular_refinement

REQUEST_SCHEMA = 'native-monomer-request/1'
MAX_TENSOR_BYTES = 16*1024**2
MAX_ENVELOPE_BYTES = 8*1024**2
SCHEMA_KEYWORDS = frozenset((
    'function_kwargs', 'reference', 'puream', 'scf_type', 'guess',
    'e_convergence', 'd_convergence', 'maxiter', 'print',
))


@dataclass(frozen=True)
class GeometrySnapshot:
    symbols: tuple
    geometry_bohr: tuple


@dataclass(frozen=True)
class NativeMonomerRequest:
    declaration: MolecularDeclaration
    resources: BoundedResources
    max_replay_work: int
    implementation_sha256: tuple

    def __post_init__(self):
        if not isinstance(self.declaration, MolecularDeclaration):
            raise TypeError('explicit molecular declaration required')
        plan_molecular_refinement(self.declaration, self.resources,
                                  max_replay_work=self.max_replay_work)
        identity = dict(self.implementation_sha256)
        if (set(identity) != {'core', 'isapol_python'}
                or any(type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None
                       for value in identity.values())):
            raise ValueError('exact installed implementation SHA256 pins required')
        object.__setattr__(self, 'implementation_sha256', tuple(sorted(identity.items())))

    @classmethod
    def from_dict(cls, value):
        if (type(value) is not dict
                or set(value) != {'schema', 'declaration', 'resources', 'max_replay_work',
                                  'implementation_sha256'}
                or value['schema'] != REQUEST_SCHEMA):
            raise ValueError('exact versioned native request required')
        if (type(value['resources']) is not dict
                or set(value['resources']) != {'max_bytes', 'max_work', 'max_io_bytes', 'capacity'}
                or type(value['implementation_sha256']) is not dict):
            raise ValueError('explicit native resources and implementation pins required')
        return cls(MolecularDeclaration.from_dict(value['declaration']),
                   BoundedResources(**value['resources']), value['max_replay_work'],
                   tuple(value['implementation_sha256'].items()))

    def as_dict(self):
        return dict(schema=REQUEST_SCHEMA, declaration=self.declaration.as_dict(),
                    resources=asdict(self.resources), max_replay_work=self.max_replay_work,
                    implementation_sha256=dict(self.implementation_sha256))


def _file_digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def implementation_identity():
    """Hash this installation, not a caller-supplied filesystem location.

    All shipped isapol Python files and the actual imported core binary are
    bound. Dependency/runtime/trace qualification remains a separate gate.
    Do not cache identity across requests or accept an alternate input path.
    """
    routing = Path(__file__).resolve().parent
    driver = routing.parent
    paths = list(routing.glob('isapol_*.py')) + [
        driver/'schema_wrapper.py', driver/'driver.py', driver/'task_planner.py',
        driver/'p4util/util.py', routing/'proc.py',
    ]
    modules = {str(path.relative_to(driver)): _file_digest(path) for path in sorted(paths)}
    if not modules or 'procrouting/'+Path(__file__).name not in modules:
        raise RuntimeError('installed native Python implementation is incomplete')
    python_digest = hashlib.sha256(
        json.dumps(modules, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(
        schema='native-monomer-implementation/1', python_files=modules,
        sha256=dict(core=_file_digest(Path(core.__file__).resolve()),
                    isapol_python=python_digest))


def validate_native_request(value, *, method, basis, molecule):
    """Fail before SCF on unsupported model, moved frames or unpinned code."""
    request = NativeMonomerRequest.from_dict(value)
    if (type(method) is not str or method.lower() != 'pbe0'
            or type(basis) is not str or basis.lower() != 'aug-cc-pvtz'):
        raise ValueError('native monomer request requires PBE0/aug-cc-pVTZ')
    if (not core.get_global_option('PUREAM')
            or core.get_option('SCF', 'REFERENCE').upper() != 'RHF'):
        raise ValueError('native monomer request requires spherical restricted orbital options')
    if not isinstance(molecule, core.Molecule):
        raise TypeError('live native Molecule required')
    if molecule.molecular_charge() != 0 or molecule.multiplicity() != 1:
        raise ValueError('native monomer request requires a neutral singlet')
    if (molecule.schoenflies_symbol().lower() != 'c1'
            or not molecule.com_fixed() or not molecule.orientation_fixed()):
        raise ValueError('native monomer request requires fixed COM/orientation and explicit C1')
    independent_sites(molecule, request.declaration)
    if request.resources.max_bytes > int(core.get_memory()):
        raise ValueError('molecular numeric budget exceeds the Psi4 memory setting')
    identity = implementation_identity()
    if identity['sha256'] != dict(request.implementation_sha256):
        raise ValueError('native monomer implementation identity differs from immutable request pins')
    return request, identity


def admit_native_schema(specification, molecule_schema):
    """Strict internal-QCSchema-v2 admission, before molecule construction/SCF.

    The first supported surface is deliberately small: ordinary PK PBE0
    properties with a fresh SAD guess, no embedding, corrections, functional
    override, alternative basis or caller-controlled extra files.
    """
    model = specification.get('model', {})
    method, basis = model.get('method'), model.get('basis')
    if (specification.get('driver') != 'properties'
            or type(method) is not str or method.lower() != 'pbe0'
            or type(basis) is not str or basis.lower() != 'aug-cc-pvtz'):
        raise ValueError('native schema requires properties at PBE0/aug-cc-pVTZ')
    keywords = specification.get('keywords')
    if type(keywords) is not dict or set(keywords) - SCHEMA_KEYWORDS:
        raise ValueError('unsupported native schema scientific keywords')
    kwargs = keywords.get('function_kwargs')
    if (type(kwargs) is not dict or 'native_monomer' not in kwargs
            or set(kwargs) - {'native_monomer', 'properties'}
            or kwargs.get('properties', ['dipole']) != ['dipole']):
        raise ValueError('native function_kwargs permit only native_monomer and lowercase dipole')
    if (any(type(keywords.get(key)) is not str for key in ('reference', 'scf_type', 'guess'))
            or keywords['reference'].lower() != 'rhf'
            or keywords.get('puream') is not True
            or keywords['scf_type'].lower() != 'pk'
            or keywords['guess'].lower() != 'sad'):
        raise ValueError('native schema requires explicit spherical RHF/PK and fresh SAD')
    for name in ('e_convergence', 'd_convergence'):
        value = keywords.get(name)
        if type(value) not in (int, float) or not np.isfinite(value) or not 0 < value <= 1e-10:
            raise ValueError('native schema requires explicit tight finite SCF stopping thresholds')
    for name, default, upper in (('maxiter', 100, 300), ('print', 1, 3)):
        value = keywords.get(name, default)
        if type(value) is not int or not (1 if name == 'maxiter' else 0) <= value <= upper:
            raise ValueError('invalid native schema iteration/print limit')
    extras = specification.get('extras', {})
    if type(extras) is not dict or extras.get('extra_infiles'):
        raise ValueError('native schema forbids extra input files')
    request = NativeMonomerRequest.from_dict(kwargs['native_monomer'])
    symmetry = molecule_schema.get('fix_symmetry')
    if (molecule_schema.get('fix_com') is not True
            or molecule_schema.get('fix_orientation') is not True
            or type(symmetry) is not str or symmetry.lower() != 'c1'):
        raise ValueError('native schema requires explicit fixed COM/orientation and C1')
    symbols = tuple(molecule_schema.get('symbols', ()))
    count = len(symbols)
    geometry = np.asarray(molecule_schema.get('geometry'), dtype=float)
    if (count != len(request.declaration.frames) or any(s not in ('C', 'H', 'N', 'O') for s in symbols)
            or geometry.size != count*3 or not np.isfinite(geometry).all()):
        raise ValueError('native schema geometry must match the declared CHNO atoms')
    if (molecule_schema.get('molecular_charge', 0.) != 0
            or molecule_schema.get('molecular_multiplicity', 1) != 1):
        raise ValueError('native schema requires a neutral singlet')
    real = np.asarray(molecule_schema.get('real', [True]*count))
    if real.shape != (count,) or real.dtype != np.dtype('bool') or not np.all(real):
        raise ValueError('native schema forbids ghost atoms')
    numbers = molecule_schema.get('atomic_numbers')
    if numbers is not None and not np.array_equal(
            numbers, [dict(H=1, C=6, N=7, O=8)[s] for s in symbols]):
        raise ValueError('native schema atomic numbers differ from CHNO symbols')
    fragments = molecule_schema.get('fragments')
    if fragments is not None and (len(fragments) != 1
            or not np.array_equal(fragments[0], np.arange(count))):
        raise ValueError('native schema requires one complete ordered monomer fragment')
    if molecule_schema.get('connectivity') is not None:
        raise ValueError('native schema uses only the explicit declaration bonds')
    return request, GeometrySnapshot(
        symbols, tuple(tuple(float(v) for v in xyz) for xyz in geometry.reshape(count, 3)))


def check_native_geometry(molecule, snapshot):
    """Compare against admitted coordinates, never an overwritten schema."""
    if (not isinstance(snapshot, GeometrySnapshot)
            or molecule.natom() != len(snapshot.symbols)
            or tuple(molecule.symbol(i) for i in range(molecule.natom())) != snapshot.symbols
            or not np.allclose(molecule.geometry().np, snapshot.geometry_bohr, rtol=0, atol=1e-10)
            or molecule.molecular_charge() != 0 or molecule.multiplicity() != 1):
        raise ValueError('native molecule geometry/order/state differs from admitted input')


def tensor_artifact(data):
    """JSON-safe transport, not a scientific or origin certificate."""
    if type(data) is not bytes or not 0 < len(data) <= MAX_TENSOR_BYTES:
        raise ValueError('bounded tensor artifact bytes required')
    artifact = dict(encoding='base64', media_type='application/x-npz',
                    sha256=hashlib.sha256(data).hexdigest(), size=len(data),
                    data=base64.b64encode(data).decode('ascii'))
    decode_tensor_artifact(artifact)
    return artifact


def decode_tensor_artifact(artifact):
    """Validate digest and finite native layouts before allocating any array.

    Parse bounded NPY headers explicitly. Merely setting allow_pickle=False
    in np.load would still allow a forged huge shape to request an allocation.
    """
    if (type(artifact) is not dict
            or set(artifact) != {'encoding', 'media_type', 'sha256', 'size', 'data'}
            or artifact['encoding'] != 'base64'
            or artifact['media_type'] != 'application/x-npz'
            or type(artifact['size']) is not int
            or not 0 < artifact['size'] <= MAX_TENSOR_BYTES
            or type(artifact['data']) is not str
            or len(artifact['data']) > 4*((MAX_TENSOR_BYTES+2)//3)):
        raise ValueError('invalid bounded tensor artifact envelope')
    data = base64.b64decode(artifact['data'], validate=True)
    if (len(data) != artifact['size']
            or hashlib.sha256(data).hexdigest() != artifact['sha256']):
        raise ValueError('tensor artifact digest/size mismatch')
    arrays = {}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        if sorted(archive.namelist()) != ['local.npy', 'refined.npy']:
            raise ValueError('exact local/refined tensor members required')
        if sum(item.file_size for item in archive.infolist()) > MAX_TENSOR_BYTES:
            raise ValueError('expanded tensor artifact exceeds byte limit')
        for name, width in (('local', 15), ('refined', 9)):
            raw = archive.read(name+'.npy')
            stream = io.BytesIO(raw)
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError('unsupported tensor NPY version')
            if (len(shape) != 4 or shape[0] != 11 or not 3 <= shape[1] <= 17
                    or shape[2:] != (width, width) or dtype != np.dtype('float64') or fortran):
                raise ValueError('tensor artifact requires native finite float64 layouts')
            expected_size = stream.tell() + int(np.prod(shape))*8
            if len(raw) != expected_size:
                raise ValueError('tensor payload length differs from bounded shape')
            array = np.frombuffer(raw, dtype=dtype, offset=stream.tell()).reshape(shape)
            if not np.isfinite(array).all():
                raise ValueError('tensor artifact contains nonfinite values')
            arrays[name] = array
    if arrays['local'].shape[:2] != arrays['refined'].shape[:2]:
        raise ValueError('local/refined frequency-site layouts differ')
    return arrays


def execute_native_request(wfn, request):
    """Run the owned native producer and package its checked exports in memory.

    Scratch is rooted only in Psi4's configured IO directory. No filenames,
    executable paths, saved states or reference products are accepted as
    request data. Temporary numeric files are removed on success or failure.
    """
    from .isapol_bounded_oeprop import invalidate
    from .isapol_molecular import run_molecular_properties, export_molecular_result
    if not isinstance(wfn, core.Wavefunction):
        raise TypeError('live native wavefunction required')
    invalidate(wfn)
    if not isinstance(request, NativeMonomerRequest):
        raise TypeError('validated NativeMonomerRequest required')
    # This second check binds the post-SCF execution to the same installed
    # implementation pins checked before SCF by the properties entry point.
    _, identity = validate_native_request(request.as_dict(), method='pbe0',
                                         basis=wfn.basisset().name(), molecule=wfn.molecule())
    try:
        root = Path(core.IOManager.shared_object().get_default_path())
        with tempfile.TemporaryDirectory(prefix='psi4-native-monomer-', dir=root) as temporary:
            work = Path(temporary)
            scratch = work/'scratch'
            scratch.mkdir()
            result = run_molecular_properties(
                wfn, request.declaration, request.resources,
                max_replay_work=request.max_replay_work, scratch_directory=scratch)
            record = export_molecular_result(
                wfn, result, request.declaration, output=work/'export')
            artifact = tensor_artifact((work/'export/tensors.npz').read_bytes())
            if record['tensors_sha256'] != artifact['sha256']:
                raise RuntimeError('owned tensor export and transport digest differ')
            arrays = decode_tensor_artifact(artifact)
            for name in ('local', 'refined'):
                if tuple(record['tensor_layout'][name+'_shape']) != arrays[name].shape:
                    raise RuntimeError('owned tensor export and transport layout differ')
            if implementation_identity()['sha256'] != identity['sha256']:
                raise RuntimeError('native implementation changed during calculation')
            envelope = dict(
                schema='native-monomer-artifacts/1', request=request.as_dict(),
                implementation=identity, result=record,
                artifacts={'tensors.npz': artifact},
                qualification='Native numeric export; installed/runtime trace and QCArchive roundtrip qualification are separate. Experimental rank2 model, C10 partial(2,2).')
            if len(json.dumps(envelope, allow_nan=False).encode()) > MAX_ENVELOPE_BYTES:
                raise ValueError('native monomer result exceeds bounded JSON transport size')
            return envelope
    except Exception:
        invalidate(wfn)
        raise
