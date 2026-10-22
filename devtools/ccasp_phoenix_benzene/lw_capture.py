# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Standalone owned LW bytes. No native/scientific import at module import time."""
from dataclasses import dataclass
import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import socket
import subprocess
import sys

MAX_BYTES = 64 * 1024**2
MAX_SITES = 64
MAX_METADATA_BYTES = 1024**2
SCHEMA = "ccasp-lw-owned-f64/1"
TRUNCATION = "discard_rank4_rows_and_columns"
TOLERANCE = 1e-6  # unchanged production LW gate
COMPONENTS = tuple(f"{l}0" if m == 0 else f"{l}{m}{s}"
                   for l in range(5) for m in range(l + 1) for s in (("",) if m == 0 else ("c", "s")))
CONVENTIONS = dict(
    input_rank=4, truncation=TRUNCATION, residual_policy="production",
    residual_tolerance_hex=TOLERANCE.hex(), localization_rank_limit=2,
    tensor_axes="frequency,response_site,potential_site,response_component,potential_component",
    frame_convention="proper local-to-global Cartesian columns",
    coordinate_frame="absolute supplied bohr frame; no recentering or reorientation",
    components=list(COMPONENTS), charge_component_index=0, dipoles=["z", "x", "y"],
    units=dict(origins="bohr", frequency="Eh, nonnegative imaginary magnitude",
               response="bohr^(l+lprime+1)"),
    python_input_width=25, actual_native_input_width=16, core_working_width=9)
SUM_NAMES = ("sum_potential_site", "sum_response_site", "fsum_potential_site", "fsum_response_site")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _keys(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError("inconsistent metadata keys")


def _pairs(items):
    out = {}
    for key, value in items:
        if key in out:
            raise ValueError("duplicate JSON key")
        out[key] = value
    return out


def _json(data):
    def invalid(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(data, object_pairs_hook=_pairs, parse_constant=invalid)


def _hash_text(value, width=64):
    if type(value) is not str or not re.fullmatch("[0-9a-f]{" + str(width) + "}", value):
        raise ValueError("invalid identity hash")


def _path_text(value):
    if (type(value) is not str or not 1 <= len(value) <= 1024 or "\x00" in value
            or "\n" in value or not Path(value).is_absolute()):
        raise ValueError("absolute bounded identity path required")


def _identities(identity):
    _keys(identity, ("native_core", "python_boundary", "driver", "config_sha256",
                     "basis_manifest_sha256", "fixture_sha256"))
    common = ("source_root", "source_commit", "source_tree")
    for name, extra in (("native_core", ("path", "sha256", "build_node")),
                        ("python_boundary", ("path", "sha256", "package_root")),
                        ("driver", ())):
        item = identity[name]
        _keys(item, (*common, *extra))
        _path_text(item["source_root"])
        for key in ("source_commit", "source_tree"):
            _hash_text(item[key], 40)
        if extra:
            _path_text(item["path"])
            _hash_text(item["sha256"])
    _path_text(identity["python_boundary"]["package_root"])
    node = identity["native_core"]["build_node"]
    if type(node) is not str or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", node):
        raise ValueError("invalid build-node identity")
    for key in ("config_sha256", "basis_manifest_sha256", "fixture_sha256"):
        _hash_text(identity[key])


@dataclass(frozen=True)
class Float64Array:
    shape: tuple
    data: bytes
    dtype: str = "<f8"

    def __post_init__(self):
        if (type(self.shape) is not tuple or not self.shape
                or len(self.shape) > 5
                or any(type(n) is not int or not 1 <= n <= 25 * MAX_SITES for n in self.shape)
                or self.dtype != "<f8" or type(self.data) is not bytes):
            raise ValueError("owned little-endian float64 bytes and positive integer shape required")
        if math.prod(self.shape) * 8 != len(self.data) or len(self.data) > MAX_BYTES:
            raise ValueError("array shape/size limit")
        if any(not math.isfinite(x[0]) for x in struct.iter_unpack("<d", self.data)):
            raise ValueError("nonfinite array")


def site_major_to_pairs(raw):
    """Convert site-component-major matrix to frequency/site-pair-major bytes."""
    if (len(raw.shape) != 2 or raw.shape[0] != raw.shape[1]
            or raw.shape[0] % 25 or not 1 <= raw.shape[0] // 25 <= MAX_SITES):
        raise ValueError("site-major matrix must have shape (25*nsite,25*nsite)")
    n = raw.shape[0] // 25
    data = bytearray(len(raw.data))
    for a in range(n):
        for b in range(n):
            for t in range(25):
                source = ((a * 25 + t) * n * 25 + b * 25) * 8
                target = ((a * n + b) * 25 + t) * 25 * 8
                data[target:target + 200] = raw.data[source:source + 200]
    return Float64Array((1, n, n, 25, 25), bytes(data))


@dataclass(frozen=True)
class CaptureReceipt:
    path: Path
    manifest_sha256: str


@dataclass(frozen=True)
class Capture:
    declaration_bytes: bytes
    arrays: tuple
    manifest_sha256: str
    diagnostics_bytes: bytes
    manifest_bytes: bytes

    @property
    def declaration(self):
        return _json(self.declaration_bytes)  # owned defensive metadata copy

    def array(self, name):
        return dict(self.arrays)[name]

    @property
    def diagnostics(self):
        return _json(self.diagnostics_bytes)


def charge_diagnostics(tensors, labels):
    """Signed GLOBAL input sums; ordered += mirrors core, fsum is report-only."""
    n = len(labels)
    if tensors.shape != (1, n, n, 25, 25) or not 1 <= n <= MAX_SITES:
        raise ValueError("charge diagnostics require the complete pair-major rank4 tensor")
    def value(a, b, t, u):
        return struct.unpack_from("<d", tensors.data, (((a*n+b)*25+t)*25+u)*8)[0]
    values = {name: [] for name in SUM_NAMES}
    for a in range(n):
        for t in range(25):
            for axis, terms in (("potential_site", [value(a,b,t,0) for b in range(n)]),
                                ("response_site", [value(b,a,0,t) for b in range(n)])):
                ordered = 0.
                for x in terms:
                    ordered += x
                    if not math.isfinite(ordered):
                        raise ValueError("nonfinite ordered charge reduction")
                values["sum_" + axis].append(ordered)
                values["fsum_" + axis].append(math.fsum(terms))
    arrays = {key: Float64Array((n,25), b"".join(struct.pack("<d", x) for x in row))
              for key, row in values.items()}
    def maximum(names, width):
        best = None
        for site in range(n):
            for t in range(width):
                for name in names:
                    x = values[name][site*25+t]
                    if best is None or abs(x) > best["magnitude"]:
                        best = dict(axis=name, site=site, label=labels[site], component=t,
                            component_name=COMPONENTS[t], signed_value_hex=x.hex(),
                            magnitude_hex=abs(x).hex(), magnitude=abs(x))
        best.pop("magnitude")
        return best
    metadata = dict(
        axis_definitions=dict(sum_potential_site="sum_b T[a,b,t,0]",
                              sum_response_site="sum_b T[b,a,0,t]"),
        tie_break="first site, then component, then potential-site sum before response-site sum",
        convention="global real Racah/no-CS; charge index0, dipoles z,x,y; signed, no repair",
        native_ordered=dict(working_maximum=maximum(SUM_NAMES[:2],9),
            full_rank4_maximum=maximum(SUM_NAMES[:2],25),
            potential_site_maximum=maximum(SUM_NAMES[:1],9),
            response_site_maximum=maximum(SUM_NAMES[1:2],9)),
        compensated=dict(working_maximum=maximum(SUM_NAMES[2:],9),
                         full_rank4_maximum=maximum(SUM_NAMES[2:],25)),
        compensated_role="math.fsum diagnostic only, never substituted into LW input or gate")
    return arrays, metadata

def _validate_declaration(declaration):
    _keys(declaration, (*CONVENTIONS, "kind", "labels", "bonds", "frequency_au_hex", "identities"))
    for name, value in CONVENTIONS.items():
        if declaration[name] != value or type(declaration[name]) is not type(value):
            raise ValueError("inconsistent strict declaration: " + name)
    if declaration["kind"] not in ("synthetic", "native_owned"):
        raise ValueError("capture kind must be explicit synthetic or native_owned")
    labels = declaration["labels"]
    if (type(labels) is not list or not 1 <= len(labels) <= MAX_SITES
            or any(type(s) is not str or not s.strip() or len(s.encode()) > 256 for s in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError("invalid ordered labels")
    n = len(labels)
    bonds = declaration["bonds"]
    if type(bonds) is not list or len(bonds) > n * (n - 1) // 2:
        raise ValueError("invalid ordered bonds")
    seen = set()
    for edge in bonds:
        if (type(edge) is not list or len(edge) != 2
                or any(type(i) is not int or not 0 <= i < n for i in edge) or edge[0] == edge[1]):
            raise ValueError("invalid oriented bond")
        key = tuple(sorted(edge))  # duplicate detection ONLY; stored orientation is unchanged
        if key in seen:
            raise ValueError("duplicate undirected bond")
        seen.add(key)
    f = declaration["frequency_au_hex"]
    if type(f) is not str:
        raise ValueError("frequency must be exact float.hex")
    frequency = float.fromhex(f)
    if not math.isfinite(frequency) or frequency < 0 or frequency.hex() != f:
        raise ValueError("invalid exact frequency")
    _identities(declaration["identities"])
    return n


def _validate(declaration, arrays, *, signed=True):
    n = _validate_declaration(declaration)
    required = dict(tensors=(1, n, n, 25, 25), origins=(n, 3), frames=(n, 3, 3))
    if signed:
        required.update({name: (n,25) for name in SUM_NAMES})
    if set(arrays) not in (set(required), set(required) | {"raw_site_major"}):
        raise ValueError("unexpected/missing array")
    if "raw_site_major" in arrays:
        required["raw_site_major"] = (25 * n, 25 * n)
    for name, shape in required.items():
        if not isinstance(arrays[name], Float64Array) or arrays[name].shape != shape:
            raise ValueError("array shape/representation mismatch: " + name)
    if sum(len(a.data) for a in arrays.values()) > MAX_BYTES:
        raise ValueError("capture exceeds total 64MiB bound")
    if "raw_site_major" in arrays and site_major_to_pairs(arrays["raw_site_major"]) != arrays["tensors"]:
        raise ValueError("raw site-major and pair-major bytes disagree")
    # Mirror the existing public frame validity check, without rotating/repairing.
    frames = struct.unpack("<" + "d" * (9 * n), arrays["frames"].data)
    for i in range(n):
        m = frames[9 * i:9 * i + 9]
        for a in range(3):
            for b in range(3):
                if abs(math.fsum(m[3 * k + a] * m[3 * k + b] for k in range(3)) - (a == b)) > 1e-12:
                    raise ValueError("improper frame")
        det = (m[0] * (m[4]*m[8] - m[5]*m[7]) - m[1] * (m[3]*m[8] - m[5]*m[6])
               + m[2] * (m[3]*m[7] - m[4]*m[6]))
        if abs(det - 1.) > 1e-12:
            raise ValueError("improper frame")


def _write(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _read(path, limit, expected_size=None):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_size > limit
                or (expected_size is not None and info.st_size != expected_size)):
            raise ValueError("file type/size limit")
        data = stream.read(info.st_size + 1)
    if len(data) != info.st_size:
        raise ValueError("file changed size during read")
    return data


def snapshot(path, *, tensors, origins, frames, labels, bonds, frequency,
             identities, kind, raw_site_major=None):
    """Explicit caller-invoked owned snapshot. Never calls/retries LW or a solver."""
    if (type(labels) not in (list, tuple) or not 1 <= len(labels) <= MAX_SITES
            or type(bonds) not in (list, tuple)
            or len(bonds) > len(labels) * (len(labels) - 1) // 2
            or any(type(b) not in (list, tuple) or len(b) != 2 for b in bonds)
            or type(frequency) is not float):
        raise ValueError("bounded explicit labels/bonds and binary64 frequency required")
    _identities(identities)
    arrays = dict(tensors=tensors, origins=origins, frames=frames)
    if raw_site_major is not None:
        arrays["raw_site_major"] = raw_site_major
    declaration = _json(_json_bytes(dict(CONVENTIONS, labels=list(labels), bonds=list(map(list, bonds)),
        frequency_au_hex=float(frequency).hex(), identities=identities, kind=kind)))
    _validate(declaration, arrays, signed=False)
    signed_arrays, diagnostics = charge_diagnostics(tensors, declaration["labels"])
    arrays.update(signed_arrays)
    _validate(declaration, arrays)
    path = Path(path)
    path.mkdir(mode=0o700, exist_ok=False)
    records = {}
    for name, a in arrays.items():
        filename = name + ".f64"
        _write(path / filename, a.data)
        records[name] = dict(file=filename, shape=list(a.shape), dtype=a.dtype,
                             nbytes=len(a.data), sha256=_sha(a.data))
    manifest = _json_bytes(dict(schema=SCHEMA, declaration=declaration, arrays=records,
                               diagnostics=diagnostics))
    if len(manifest) > MAX_METADATA_BYTES:
        raise ValueError("manifest size limit")
    _write(path / "manifest.json", manifest)  # last; partial failures are not cleaned/reused
    return CaptureReceipt(path, _sha(manifest))


def load_capture(path, manifest_sha256):
    """Validate bounded metadata BEFORE opening binary data. Pickle/NPY is never used."""
    _hash_text(manifest_sha256)
    path = Path(path)
    if path.is_symlink():
        raise ValueError("capture directory may not be a symlink")
    encoded = _read(path / "manifest.json", MAX_METADATA_BYTES)
    if _sha(encoded) != manifest_sha256:
        raise ValueError("manifest checksum mismatch")
    manifest = _json(encoded)
    _keys(manifest, ("schema", "declaration", "arrays", "diagnostics"))
    if manifest["schema"] != SCHEMA or type(manifest["arrays"]) is not dict:
        raise ValueError("capture schema mismatch")
    _validate_declaration(manifest["declaration"])
    records = manifest["arrays"]
    # Preflight shapes/declarations/total size without loading any binary file.
    shapes = {}
    for name, rec in records.items():
        if name not in ("tensors", "origins", "frames", "raw_site_major", *SUM_NAMES):
            raise ValueError("unexpected array")
        _keys(rec, ("file", "shape", "dtype", "nbytes", "sha256"))
        shape = rec["shape"]
        if (rec["file"] != name + ".f64" or rec["dtype"] != "<f8" or type(shape) is not list
                or not 1 <= len(shape) <= 5
                or any(type(x) is not int or not 1 <= x <= 25*MAX_SITES for x in shape)
                or type(rec["nbytes"]) is not int or math.prod(shape)*8 != rec["nbytes"]):
            raise ValueError("array declaration/type/size mismatch")
        _hash_text(rec["sha256"])
        shapes[name] = tuple(shape)
    labels = manifest["declaration"].get("labels") if type(manifest["declaration"]) is dict else None
    if type(labels) is not list or not 1 <= len(labels) <= MAX_SITES:
        raise ValueError("site limit")
    n = len(labels)
    expected = dict(tensors=(1,n,n,25,25), origins=(n,3), frames=(n,3,3))
    expected.update({name: (n,25) for name in SUM_NAMES})
    if "raw_site_major" in shapes:
        expected["raw_site_major"] = (25*n,25*n)
    if shapes != expected or sum(r["nbytes"] for r in records.values()) > MAX_BYTES:
        raise ValueError("capture shape/size limit")
    arrays = {}
    for name, rec in records.items():
        data = _read(path / rec["file"], MAX_BYTES, rec["nbytes"])
        if _sha(data) != rec["sha256"]:
            raise ValueError("array checksum mismatch: " + name)
        arrays[name] = Float64Array(shapes[name], data, rec["dtype"])
    _validate(manifest["declaration"], arrays)
    expected_sums, diagnostics = charge_diagnostics(arrays["tensors"], labels)
    if any(arrays[name] != a for name, a in expected_sums.items()) or manifest["diagnostics"] != diagnostics:
        raise ValueError("signed diagnostic arrays/metadata inconsistent with raw input")
    return Capture(_json_bytes(manifest["declaration"]), tuple(arrays.items()), manifest_sha256,
                   _json_bytes(diagnostics), encoded)


_NUMBER = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
_RESIDUAL_NAMES = ("off-site", "charge-sum", "reciprocity", "molecular-sum",
                   "local-charge", "input-sum-rule", "charge-sum-transport")
_REFUSAL = re.compile(re.escape("localize_lw: postcondition exceeds residual tolerance (")
    + ", ".join(re.escape(name + "=") + _NUMBER for name in _RESIDUAL_NAMES) + r"\)")


def _expected_refusal(error, maximum):
    matches = list(_REFUSAL.finditer(str(error)))
    if len(matches) != 1 or maximum <= TOLERANCE:
        return None
    residuals = dict(zip(_RESIDUAL_NAMES, map(float, matches[0].groups())))
    if any(not math.isfinite(v) or v < 0 for v in residuals.values()):
        return None
    if any(residuals[k] >= TOLERANCE for k in
           ("off-site", "reciprocity", "molecular-sum", "charge-sum-transport")):
        # Six-digit rounding at exactly the gate could hide a real violation.
        # Conservatively reject that ambiguous display, even if exact==gate.
        # With this fixed, exactly displayed 1e-6 production gate, a strictly
        # smaller rounded token cannot hide an above-gate value.
        return None
    # The native supplied gate uses a MAXIMUM, not an all-three-fail rule.
    # The exact captured input maximum already establishes input refusal.
    # Default C++ ostream uses six significant digits, not a tensor fixture.
    # This is a display-token corroboration, NEVER a new numerical tolerance.
    if residuals["input-sum-rule"] != float(format(maximum, ".6g")):
        return None
    return residuals


def _invocation_hash(kwargs):
    return _sha(_json_bytes({k: dict(shape=v.shape, dtype=v.dtype, sha256=_sha(v.data))
                            if isinstance(v, Float64Array) else v for k, v in kwargs.items()}))


def replay(capture, *, strict_boundary):
    """Call one supplied strict seam once. No solve, retry, PFIT or publication.

    Boundary adapters receive immutable Float64Array copies, never original
    mutable NumPy views. Native adapter below delegates to the public LW factory.
    """
    if not isinstance(capture, Capture) or _sha(capture.manifest_bytes) != capture.manifest_sha256:
        raise ValueError("invalid capture receipt")
    manifest = _json(capture.manifest_bytes)
    declaration, arrays = capture.declaration, dict(capture.arrays)
    _validate(declaration, arrays)
    if manifest["declaration"] != declaration or manifest["diagnostics"] != capture.diagnostics:
        raise ValueError("capture metadata changed")
    for name, a in arrays.items():
        if _sha(a.data) != manifest["arrays"][name]["sha256"]:
            raise ValueError("capture bytes changed")
    expected_sums, diagnostics = charge_diagnostics(arrays["tensors"], declaration["labels"])
    if diagnostics != capture.diagnostics or any(arrays[k] != a for k, a in expected_sums.items()):
        raise ValueError("capture signed diagnostics changed")
    def owned(name):
        a = arrays[name]
        return Float64Array(a.shape, bytes(memoryview(a.data)), a.dtype)
    kwargs = dict(labels=tuple(declaration["labels"]),
        origins=owned("origins"), frames=owned("frames"),
        bonds=tuple(tuple(b) for b in declaration["bonds"]),
        frequencies=(float.fromhex(declaration["frequency_au_hex"]),),
        tensors=owned("tensors"), input_rank=4, truncation=TRUNCATION,
        residual_policy="production", localization_rank_limit=2,
        provenance=dict(capture_manifest_sha256=capture.manifest_sha256,
                        identities=declaration["identities"]))
    before = _invocation_hash(kwargs)
    maximum = float.fromhex(diagnostics["native_ordered"]["working_maximum"]["magnitude_hex"])
    outcome = dict(capture_manifest_sha256=capture.manifest_sha256,
        scientific_properties_accepted=False, boundary_calls=1,
        execution_kind="caller-supplied strict boundary; no native verification claim")
    try:
        strict_boundary(**kwargs)  # ONE call, no fallback or second solve
    except BaseException as error:
        if _invocation_hash(kwargs) != before:
            raise ValueError("boundary mutated replay operands") from error
        residuals = _expected_refusal(error, maximum) if isinstance(error, RuntimeError) else None
        if residuals is None:
            raise  # preserve the very same unrelated exception, never label it expected
        return dict(outcome, status="EXPECTED_INPUT_SUM_RULE_REFUSAL", diagnostic_replay_passed=True,
                    exception=str(error), printed_residuals=residuals,
                    signed_input_maximum=diagnostics["native_ordered"]["working_maximum"])
    if _invocation_hash(kwargs) != before:
        raise ValueError("boundary mutated replay operands")
    return dict(outcome, status="UNEXPECTED_STRICT_SUCCESS", diagnostic_replay_passed=False)


def _file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def native_replay(capture):
    """Allocation-only public LW adapter, AFTER a separately approved runtime bootstrap.

    Never initialize/import a fallback scientific runtime. CLI callers must run
    in the SAME interpreter as their reviewed runtime initializer/CDLL handle.
    """
    host = socket.gethostname().split(".")[0]
    identity = capture.declaration["identities"]
    if (not os.environ.get("SLURM_JOB_ID") or os.environ.get("USER") != "awallace43"
            or not os.environ.get("HOME", "").startswith("/storage/home/")
            or host.startswith("login") or host != os.environ.get("SLURMD_NODENAME", "").split(".")[0]
            or host != identity["native_core"]["build_node"]):
        raise ValueError("Phoenix compute allocation on captured build node required")
    if capture.declaration["kind"] != "native_owned":
        raise ValueError("native replay requires an actual native_owned capture")
    if "psi4" not in sys.modules or "numpy" not in sys.modules:
        raise ValueError("preinitialized reviewed runtime required; no automatic scientific import/loader fallback")
    for item in (identity["native_core"], identity["python_boundary"], identity["driver"]):
        def git(*args):
            return subprocess.check_output(["git", "-C", item["source_root"], *args], text=True).strip()
        if (git("rev-parse", "HEAD") != item["source_commit"]
                or git("rev-parse", "HEAD^{tree}") != item["source_tree"]
                or git("status", "--porcelain")):
            raise ValueError("captured source identity/clean state differs")
    psi4, np = sys.modules["psi4"], sys.modules["numpy"]
    core_id, python_id = identity["native_core"], identity["python_boundary"]
    if (Path(psi4.core.__file__).resolve() != Path(core_id["path"]).resolve()
            or _file_sha(psi4.core.__file__) != core_id["sha256"]
            or Path(psi4.__file__).resolve().parent != Path(python_id["package_root"]).resolve()
            or _file_sha(python_id["path"]) != python_id["sha256"]):
        raise ValueError("captured native core/Python package identity differs")
    lw = importlib.import_module("psi4.driver.procrouting.isapol_lw")
    if (Path(lw.__file__).resolve() != Path(python_id["path"]).resolve()
            or _file_sha(lw.__file__) != python_id["sha256"]
            or lw.PRODUCTION_TOLERANCE != TOLERANCE or lw.TRUNCATE_RANK4 != TRUNCATION):
        raise ValueError("strict public LW seam identity/default differs")
    def boundary(**kwargs):
        kwargs = dict(kwargs)
        for key in ("tensors", "origins", "frames"):
            a = kwargs[key]
            kwargs[key] = np.frombuffer(a.data, dtype="<f8").reshape(a.shape)  # read-only owned bytes
        kwargs["provenance"] = lw.Provenance(
            "owned pre-LW capture", capture.manifest_sha256, "explicit strict public LW replay",
            "caller-pinned identities; diagnostic replay, not accepted molecular properties")
        return lw.supplied_nonlocal_properties(**kwargs)  # one frequency -> one native call
    outcome = replay(capture, strict_boundary=boundary)
    return dict(outcome, execution_kind="public native LW in preinitialized approved process",
                binary_source_attestation=False)


def synthetic_example(path):
    """Literal two-site toy, NOT a molecular tensor or inference from rounded P003 residuals."""
    raw = bytearray(50 * 50 * 8)
    for a,b,t,u,value in ((0,0,1,0,2.), (0,1,1,0,-5.), (0,0,0,2,-1.), (1,0,0,2,7.),
                          (1,1,24,24,-0.)):
        struct.pack_into("<d", raw, ((a*25+t)*50+b*25+u)*8, value)
    raw = Float64Array((50,50), bytes(raw))
    def array(shape, values):
        return Float64Array(shape, b"".join(struct.pack("<d", x) for x in values))
    source = dict(source_root="/synthetic/source", source_commit="1"*40, source_tree="2"*40)
    ids = dict(native_core=dict(source, path="/synthetic/stage/lib/psi4/core.so",
                               sha256="3"*64, build_node="synthetic-node"),
        python_boundary=dict(source, path="/synthetic/stage/lib/psi4/driver/procrouting/isapol_lw.py",
                             package_root="/synthetic/stage/lib/psi4", sha256="4"*64),
        driver=source.copy(), config_sha256="5"*64, basis_manifest_sha256="6"*64, fixture_sha256="7"*64)
    return snapshot(path, tensors=site_major_to_pairs(raw), raw_site_major=raw,
        origins=array((2,3), [0.,0.,0.,1.,2.,3.]),
        frames=array((2,3,3), [1.,0.,0.,0.,1.,0.,0.,0.,1.]*2),
        labels=["Left","Right"], bonds=[(1,0)], frequency=0., identities=ids, kind="synthetic")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    example = commands.add_parser("example", help="write only a literal synthetic example")
    example.add_argument("path")
    for command in ("inspect", "replay-native"):
        sub = commands.add_parser(command)
        sub.add_argument("path")
        sub.add_argument("--manifest-sha256", required=True)
        if command == "replay-native":
            sub.add_argument("--out", required=True, help="new exclusive JSON result file; parent exists")
    args = parser.parse_args(argv)
    if args.command == "example":
        receipt = synthetic_example(args.path)
        print(json.dumps(dict(path=str(receipt.path), manifest_sha256=receipt.manifest_sha256,
                              kind="synthetic")))
        return 0
    capture = load_capture(args.path, args.manifest_sha256)
    if args.command == "inspect":
        print(json.dumps(dict(manifest_sha256=capture.manifest_sha256,
            declaration=capture.declaration, diagnostics=capture.diagnostics,
            shapes={name: a.shape for name, a in capture.arrays}), indent=2))
        return 0
    # Claim the output BEFORE the one possible native call; never overwrite.
    fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        try:
            outcome = native_replay(capture)
        except BaseException as error:
            json.dump(dict(status="REPLAY_FAILED", error=f"{type(error).__name__}: {error}",
                diagnostic_replay_passed=False, scientific_properties_accepted=False), stream)
            raise
        json.dump(outcome, stream, indent=2, allow_nan=False)
    return 0 if outcome["diagnostic_replay_passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
