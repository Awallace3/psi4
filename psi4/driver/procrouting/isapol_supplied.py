"""Supplied ORIENT NEW local-response bridge (not a wavefunction prediction).

Only numerical output is imported; no ORIENT localization algorithm is implemented.
Manifest schema 1 is illustrated by data_isapol/orient_local/manifest.json. A manifest
is a hash-pinned declaration, not proof of the external producer's correctness.
Arrays are stored as immutable tuples and all array properties return copies.
Core imports are lazy so parse/geometry checks need no compiled Psi4 runtime.
"""
from dataclasses import dataclass, fields, is_dataclass, replace
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
from typing import Optional

import numpy as np

MODES = ("imported_ORIENT_localized", "imported_ORIENT_localized_and_external_PFIT_refined")
TODOS = (
    "TODO: implement and independently validate internal localization and refinement.",
    "TODO: correct native density/transition-fit/response discrepancies before wavefunction-first claims.",
    "TODO: tighten numerical reference comparisons independently; never suppress valid supplied C_n.",
    "TODO: supply missing ranks for unrestricted completeness; rank-3 isotropic C12 is partial.",
    "TODO: audit external sum rules/passivity and high-rank reference conventions; no clipping or repair.",
)
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?"
HEADER = re.compile(r"ALPHA\s+(\S+)\s+SITE-NAMES\s+(\S+)\s+(\S+)\s+RANK\s+(\d+)\s+TO\s+(\d+)\s+INDEX\s+(\d+)\s+FREQSQ\s+(" + NUMBER + r")")


def _array(value, shape=None):
    a = np.array(value, dtype=float, copy=True)
    if (shape is not None and a.shape != shape) or not np.isfinite(a).all():
        raise ValueError("invalid array shape or nonfinite values")
    return a


def _freeze(value):
    a = _array(value)
    return tuple(_freeze(x) if np.ndim(x) else float(x) for x in a)


def _frame(value):
    f = _array(value, (3, 3))
    if not np.allclose(f.T @ f, np.eye(3), rtol=0, atol=1e-12) or abs(np.linalg.det(f)-1) > 1e-12:
        raise ValueError("frame must be proper local-to-global Cartesian rotation")
    return f


def _number(token):
    if not re.fullmatch(NUMBER, token):
        raise ValueError("unknown numeric syntax or nonfinite value: " + token)
    x = float(token.replace("D", "E").replace("d", "e"))
    if not np.isfinite(x):
        raise ValueError("nonfinite numeric value")
    return x


def printed_tolerance(token):
    """Half one last printed decimal place (including E/D exponent)."""
    d = Decimal(token.replace("D", "E").replace("d", "e"))
    return float(Decimal(5).scaleb(d.as_tuple().exponent - 1))


def printed_matches(value, token):
    return bool(abs(value - _number(token)) <= printed_tolerance(token) + 8*np.finfo(float).eps*max(1., abs(value)))


@dataclass(frozen=True)
class Source:
    name: str
    sha256: str
    text: str
    role: str
    path: str = ""


@dataclass(frozen=True)
class RawTensor:
    molecule: str
    site: str
    lo: int
    hi: int
    index: int
    freqsq: str
    section: Optional[int]
    source: str
    line: int
    header: str
    _values: tuple

    def __post_init__(self):
        if any(type(x) is not int for x in (self.lo, self.hi, self.index)) or self.index < 0:
            raise ValueError("ranks/index must be nonnegative integers")
        if self.section is not None and (type(self.section) is not int or self.section < 0):
            raise ValueError("invalid section index")
        _number(self.freqsq)
        n = (self.hi+1)**2-self.lo**2
        if not 1 <= self.lo <= self.hi <= 4:
            raise ValueError("only contiguous ranks 1..4, no rank zero")
        object.__setattr__(self, "_values", _freeze(_array(self._values, (n, n))))

    @property
    def values(self):
        return np.array(self._values)

    @property
    def components(self):
        return tuple(c for l in range(self.lo, self.hi+1)
                     for c in [f"{l}0"] + [f"{l}{m}{s}" for m in range(1, l+1) for s in "cs"])


def parse_orient_new(text, source="supplied"):
    """Strict dense NEW grammar. No frequency inference, localization or repair.

    Enclosing # INDEX comments remain distinct from the raw tensor INDEX. Duplicate
    sections, including empty sections, fail. Site/grid completeness is checked by
    read_orient_local_response after explicit manifest mapping.
    """
    records, section, seen_sections = [], None, set()
    pending, rows = None, []

    def finish():
        nonlocal pending, rows
        if pending is None:
            return
        mol, a, b, lo, hi, idx, freq, lineno, header = pending
        n = (hi+1)**2-lo**2
        if len(rows) != n:
            raise ValueError(f"{source}:{lineno}: truncated matrix: expected {n} rows")
        records.append(RawTensor(mol, a, lo, hi, idx, freq, section, source, lineno, header, rows))
        pending, rows = None, []

    section_count = 0
    terminated = False
    for lineno, raw in enumerate(text.splitlines(), 1):
        stripped = raw.strip()
        marker = re.fullmatch(r"#\s*INDEX\s+(\d+)\s*", stripped)
        if marker:
            if (section is not None or records or pending is not None) and not terminated:
                raise ValueError(f"{source}:{lineno}: missing ENDFILE before next section")
            finish()
            if section is not None and len(records) == section_count:
                raise ValueError("empty section")
            section = int(marker[1])
            if section in seen_sections:
                raise ValueError("duplicate section")
            seen_sections.add(section)
            section_count = len(records)
            terminated = False
            continue
        if stripped.startswith("#"):
            if re.match(r"#\s*INDEX\b", stripped):
                raise ValueError("malformed section marker")
            continue
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line == "ENDFILE":
            if terminated or (pending is None and len(records) == section_count):
                raise ValueError(f"{source}:{lineno}: unexpected or repeated ENDFILE")
            finish()
            terminated = True
            continue
        if terminated:
            raise ValueError(f"{source}:{lineno}: data after ENDFILE requires a new INDEX section")
        if line.startswith("ALPHA"):
            finish()
            m = HEADER.fullmatch(line)
            if not m:
                raise ValueError(f"{source}:{lineno}: unsupported ALPHA grammar (distributed/old forbidden)")
            mol, a, b, lo, hi, idx, freq = m.groups()
            lo, hi, idx = int(lo), int(hi), int(idx)
            if a != b:
                raise ValueError("offsite/distributed response is not a local model")
            if not 1 <= lo <= hi <= 4:
                raise ValueError("unsupported rank (rank zero forbidden)")
            _number(freq)
            pending = mol, a, b, lo, hi, idx, freq, lineno, raw
            continue
        if pending is None:
            raise ValueError(f"{source}:{lineno}: unknown syntax")
        n = (pending[4]+1)**2-pending[3]**2
        row = [_number(t) for t in line.split()]
        if len(row) != n or len(rows) >= n:
            raise ValueError(f"{source}:{lineno}: wrong dense matrix size")
        rows.append(row)
    finish()
    if not records or (section is not None and len(records) == section_count):
        raise ValueError("empty response or section")
    if not terminated:
        raise ValueError(f"{source}: missing final ENDFILE")
    return tuple(records)


@dataclass(frozen=True)
class Site:
    label: str
    origin: tuple
    _frame: tuple
    ranks: tuple

    def __post_init__(self):
        object.__setattr__(self, "origin", tuple(_array(self.origin, (3,))))
        object.__setattr__(self, "_frame", _freeze(_frame(self._frame)))
        ranks = tuple(self.ranks)
        if not self.label or not ranks or any(type(l) is not int for l in ranks) or ranks != tuple(range(ranks[0], ranks[-1]+1)) or not 1 <= ranks[0] <= ranks[-1] <= 4:
            raise ValueError("invalid site/ranks")
        object.__setattr__(self, "ranks", ranks)

    @property
    def frame(self):
        return np.array(self._frame)


@dataclass(frozen=True)
class Diagnostic:
    site: str
    frequency_index: int
    max_asymmetry: float
    minimum_symmetric_eigenvalue: float


@dataclass(frozen=True)
class LocalResponse:
    """Immutable imported model; tensors ordered frequency-major then site-major."""
    molecule: str
    track: str
    tensor_origin: str
    sites: tuple
    frequencies: tuple
    cp_weights: tuple
    tensors: tuple
    sources: tuple
    manifest_sha256: str
    manifest_text: str
    authority: str
    warnings: tuple
    diagnostics: tuple

    def __post_init__(self):
        for key in ("sites", "frequencies", "cp_weights", "tensors", "sources", "warnings", "diagnostics"):
            object.__setattr__(self, key, tuple(getattr(self, key)))
        f, w = _array(self.frequencies), _array(self.cp_weights)
        if f.ndim != 1 or len(f) < 2 or w.shape != f.shape or f[0] != 0 or np.any(np.diff(f) <= 0) or w[0] != 0 or np.any(w < 0) or not np.any(w > 0):
            raise ValueError("invalid static-plus-dynamic CP grid")
        if self.tensor_origin not in MODES or not self.track or not self.sites:
            raise ValueError("explicit local provenance required")
        if len({s.label for s in self.sites}) != len(self.sites) or len(self.tensors) != len(f)*len(self.sites):
            raise ValueError("duplicate sites or incomplete tensor grid")
        diagnostics, warnings = [], list(self.warnings)
        for k in range(len(f)):
            for j, s in enumerate(self.sites):
                t = self.tensors[k*len(self.sites)+j]
                if t.site != s.label or t.molecule != self.molecule or (t.lo, t.hi) != (s.ranks[0], s.ranks[-1]):
                    raise ValueError("tensor/site/rank mismatch")
                a = t.values
                with np.errstate(over="raise", invalid="raise"):
                    error = float(np.max(np.abs(a-a.T)))
                    eig = float(np.linalg.eigvalsh(a/2+a.T/2)[0])
                if not np.isfinite(eig):
                    raise ValueError("nonfinite reciprocity/passivity diagnostic")
                diagnostics.append(Diagnostic(s.label, k, error, eig))
                if error:
                    warnings.append(f"{s.label}[{k}] asymmetric raw tensor ({error:g}); trace isotropization allowed, strict anisotropic conversion forbidden")
                if eig < 0:
                    warnings.append(f"{s.label}[{k}] indefinite symmetric part ({eig:g}); retained without clipping")
        object.__setattr__(self, "diagnostics", tuple(diagnostics))
        object.__setattr__(self, "warnings", tuple(dict.fromkeys(warnings)))

    @property
    def raw_tensors(self):
        """One owned (frequency, component, component) array per site."""
        return tuple(np.array([self.tensors[k*len(self.sites)+j].values for k in range(len(self.frequencies))])
                     for j in range(len(self.sites)))

    @property
    def atomic_scalars(self):
        out = []
        for site, a in zip(self.sites, self.raw_tensors):
            start, cols = 0, []
            for l in site.ranks:
                stop = start+2*l+1
                cols.append(np.trace(a[:, start:stop, start:stop], axis1=1, axis2=2)/(2*l+1))
                start = stop
            out.append(np.array(cols).T)
        return tuple(out)

    @property
    def global_dipoles(self):
        out = []
        for site, a in zip(self.sites, self.raw_tensors):
            if site.ranks[0] != 1:
                out.append(None)  # Explicitly unavailable, not a zero-padded dipole.
                continue
            p = [1, 2, 0]
            out.append(site.frame @ a[:, :3, :3][:, p, :][:, :, p] @ site.frame.T)
        return tuple(out)


@dataclass(frozen=True)
class Placement:
    """Rigid placement: r'=R r+t, F'=R F. No implicit B placement."""
    translation: tuple
    _rotation: tuple

    def __post_init__(self):
        object.__setattr__(self, "translation", tuple(_array(self.translation, (3,))))
        object.__setattr__(self, "_rotation", _freeze(_frame(self._rotation)))

    @property
    def rotation(self):
        return np.array(self._rotation)


def place_model(model, placement):
    return replace(model, sites=tuple(replace(s, origin=placement.rotation @ s.origin + placement.translation,
                                             _frame=placement.rotation @ s.frame) for s in model.sites))


def _json(text):
    def pairs(items):
        d = {}
        for k, v in items:
            if k in d:
                raise ValueError("duplicate manifest key: " + k)
            d[k] = v
        return d
    return json.loads(text, object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def _hashed(path, expected, role):
    b = Path(path).read_bytes()
    actual = hashlib.sha256(b).hexdigest()
    if actual != expected:
        raise ValueError(f"SHA256 mismatch: {path}")
    return Source(Path(path).name, actual, b.decode("utf-8"), role, str(Path(path).resolve()))


def _geometry(manifest, sources):
    """Narrow configuration check, not an ORIENT interpreter.

    Unrotated Molecule at 0 0 0 establishes global O when explicitly declared;
    axes supports only the observed global-Z/from-to recipe. Unknown axes fail.
    """
    byrole = {s.role: s.text for s in sources}
    st, recipe, axes = byrole["sites"], byrole["recipe"], byrole["axes"]
    if not re.search(r"Units\s+BOHR", st, re.I) or not re.search(r"UNITS\s+BOHR", recipe, re.I):
        raise ValueError("geometry source must declare BOHR")
    coord = re.compile(r"^\s*(\S+)\s+("+NUMBER+r")\s+("+NUMBER+r")\s+("+NUMBER+r")\s+Type\s+\S+\s*$", re.M)
    def coords(text):
        rows = coord.findall(text)
        if len({x[0] for x in rows}) != len(rows):
            raise ValueError("duplicate geometry site")
        return {x[0]: tuple(_number(v) for v in x[1:]) for x in rows}
    expected = {s["label"]: tuple(s["origin"]) for s in manifest["sites"]}
    if coords(st) != expected or coords(recipe) != expected:
        raise ValueError("manifest geometry mismatch with sites/recipe")
    body = re.findall(r"^\s*Molecule\s+[^\n]+\n(.*?)^\s*End\s*$", recipe, re.M | re.S | re.I)
    if len(body) != 1 or any(not coord.fullmatch(line) for line in body[0].splitlines()
                             if line.strip() and not line.lstrip().startswith("!")):
        raise ValueError("unsupported molecule geometry/orientation directives")
    mol = re.findall(r"^\s*Molecule\s+(\S+)\s+at\s+(.+)$", recipe, re.M | re.I)
    if len(mol) != 1 or mol[0][0] != manifest["molecule"] or [_number(x) for x in mol[0][1].split()] != [0., 0., 0.]:
        raise ValueError("only declared unrotated molecule at global origin supported")
    if not re.search(r"^\s*Localise\b", recipe, re.M | re.I) or not re.search(r"Write all local ranks", recipe, re.I):
        raise ValueError("recipe does not establish localized output")
    edits = re.findall(r"^\s*Edit\s+(\S+)\s*\n(.*?)^\s*End\s*$", recipe, re.M | re.S | re.I)
    axes_name = next(s.name for s in sources if s.role == "axes")
    if len(edits) != 1 or edits[0][0] != manifest["molecule"] or edits[0][1].strip() not in ("#include {AXES}", "#include "+axes_name):
        raise ValueError("only the declared axes include is supported in the frame-edit recipe")
    if re.search(r"^\s*(?:Rotate|Orient|Translate|Move)\b", recipe, re.M | re.I):
        raise ValueError("unsupported recipe placement directive")
    derived = {}
    lines = [x.split("#", 1)[0].strip() for x in axes.splitlines() if x.strip() and not x.lstrip().startswith("!")]
    if lines[0:1] != ["Axes"] or lines[-1:] != ["End"]:
        raise ValueError("unsupported axes grammar")
    for line in lines[1:-1]:
        m = re.fullmatch(r"(\S+)\s+z global Z x from (\S+) to (\S+)", line)
        if not m or any(x not in expected for x in m.groups()) or m[1] in derived:
            raise ValueError("unsupported/duplicate axes directive")
        x = np.array(expected[m[3]])-expected[m[2]]
        x[2] = 0
        if np.linalg.norm(x) == 0:
            raise ValueError("degenerate axes")
        x /= np.linalg.norm(x)
        z = np.array([0., 0., 1.])
        derived[m[1]] = np.column_stack((x, np.cross(z, x), z))
    global_sites = manifest["global_frame_sites"]
    if len(set(global_sites)) != len(global_sites) or set(global_sites) != set(expected)-set(derived):
        raise ValueError("explicit global frame declarations required exactly for sites absent from axes")
    for s in manifest["sites"]:
        wanted = derived[s["label"]] if s["label"] in derived else np.eye(3)
        if not np.allclose(_frame(s["frame"]), wanted, atol=1e-12, rtol=0):
            raise ValueError("manifest frame mismatch with explicit axes/global declaration")


def read_orient_local_response(files, manifest, *, manifest_sha256=None, core_module=None):
    """Read hash-verified local files using a hash-pinned schema-1 manifest.

    files is a directory (manifest filenames resolved there) or an explicit list.
    A required adjacent manifest.json.sha256 may supply the manifest hash. Frequency
    sections map via files[].sections; an unsectioned individual file uses key 'file'.
    authority='external_manifest' explicitly overrides defective raw INDEX/FREQSQ;
    authority='headers' requires both to agree at printing precision. Index base is
    explicit. Canonical frequencies/weights ALWAYS come from core.CasimirGrid.
    """
    path = Path(manifest)
    if manifest_sha256 is None:
        manifest_sha256 = Path(str(path)+".sha256").read_text().split()[0]
    ms = _hashed(path, manifest_sha256, "manifest")
    m = _json(ms.text)
    if m["schema"] != 1 or m["units"] != "atomic" or m["geometry_units"] != "bohr" or m["components"] != "real_Racah_10_11c_11s" or m["frame_convention"] != "local_to_global_columns":
        raise ValueError("unsupported manifest schema/units/conventions")
    if m["tensor_origin"] not in MODES or m["authority"] not in ("headers", "external_manifest"):
        raise ValueError("explicit local provenance and frequency authority required")
    if m["grid"] != {"n": 10, "beta": 0.5}:
        raise ValueError("this bridge requires canonical CasimirGrid(10,0.5)")
    if type(m["header_index_base"]) is not int:
        raise ValueError("explicit integer header index base required")
    if isinstance(files, (str, Path)):
        root = Path(files)
        paths = {x["name"]: root/x["name"] for x in m["files"]}
    else:
        files = list(map(Path, files))
        paths = {p.name: p for p in files}
        if len(paths) != len(files):
            raise ValueError("duplicate input filenames")
    if len({x["name"] for x in m["files"]}) != len(m["files"]) or set(paths) != {x["name"] for x in m["files"]}:
        raise ValueError("input files mismatch manifest")
    sources = []
    for x in m["provenance_sources"]:
        if Path(x["name"]).name != x["name"]:
            raise ValueError("provenance filenames must be local basenames")
        sources.append(_hashed(path.parent/x["name"], x["sha256"], x["role"]))
    roles = [s.role for s in sources]
    if len(set(roles)) != len(roles) or not {"recipe", "axes", "sites"}.issubset(roles) or (m["tensor_origin"] == MODES[1] and "pdef" not in roles):
        raise ValueError("missing/duplicate geometry and producer provenance sources")
    _geometry(m, sources)
    sites = tuple(Site(s["label"], s["origin"], s["frame"], s["ranks"]) for s in m["sites"])
    if len({s.label for s in sites}) != len(sites):
        raise ValueError("duplicate manifest sites")
    if core_module is None:
        from psi4 import core as core_module
    grid = core_module.CasimirGrid(10, 0.5)
    freq = tuple(grid.omega(k) for k in range(11))
    weights = tuple(grid.cp_weight(k) for k in range(11))
    ordered, warnings = {}, list(m.get("warnings", []))
    for spec in m["files"]:
        if Path(spec["name"]).name != spec["name"]:
            raise ValueError("input filenames must be basenames")
        src = _hashed(paths[spec["name"]], spec["sha256"], "response")
        sources.append(src)
        records = parse_orient_new(src.text, src.name)
        mapping = spec["sections"]
        keys = {"file" if r.section is None else str(r.section) for r in records}
        if keys != set(mapping) or any(type(k) is not int or not 0 <= k <= 10 for k in mapping.values()):
            raise ValueError("manifest section/grid mapping mismatch")
        if len(set(mapping.values())) != len(mapping):
            raise ValueError("duplicate canonical section mapping")
        for r in records:
            k = mapping["file" if r.section is None else str(r.section)]
            if r.site not in {s.label for s in sites} or r.molecule != m["molecule"]:
                raise ValueError("unknown site/molecule")
            key = k, r.site
            if key in ordered:
                raise ValueError("duplicate site/frequency")
            conflict = r.index != k+m["header_index_base"] or not printed_matches(-freq[k]**2, r.freqsq)
            if conflict:
                if m["authority"] != "external_manifest":
                    raise ValueError("header frequency contradiction requires explicit external_manifest authority")
                warnings.append(f"{r.source}:{r.line}: raw INDEX/FREQSQ conflict; external manifest maps section to canonical index {k}")
            ordered[key] = r
    if set(ordered) != {(k, s.label) for k in range(11) for s in sites}:
        raise ValueError("missing site/frequency")
    tensors = tuple(ordered[k, s.label] for k in range(11) for s in sites)
    return LocalResponse(m["molecule"], m["track"], m["tensor_origin"], sites, freq, weights,
                         tensors, tuple(sources), ms.sha256, ms.text, m["authority"], tuple(warnings), ())


@dataclass(frozen=True)
class Coefficient:
    order: int
    value: float
    included: tuple
    missing: tuple
    unrestricted_complete: bool
    declared_model_complete: bool = True
    energy: Optional[float] = None

    def __post_init__(self):
        object.__setattr__(self, "included", tuple(map(tuple, self.included)))
        object.__setattr__(self, "missing", tuple(map(tuple, self.missing)))


@dataclass(frozen=True)
class Pair:
    site_a: int
    site_b: int
    coefficients: tuple
    distance: Optional[float] = None
    direction: tuple = ()
    truncated_energy: Optional[float] = None

    def __post_init__(self):
        object.__setattr__(self, "coefficients", tuple(self.coefficients))
        object.__setattr__(self, "direction", tuple(self.direction))


@dataclass(frozen=True)
class Comparison:
    """Numerical comparison only; failure never invalidates a computed coefficient."""
    name: str
    max_absolute_error: Optional[float]
    tolerance: Optional[float]
    agreement: Optional[bool]
    convention: str
    checked_points: int = 0
    max_printed_units: Optional[float] = None
    availability: str = "available"
    reason: Optional[str] = None


def compare_frequency_headers(model):
    """Independent archived NL4/unrefined numerical headers, never local tensors.

    Re-serialization may print more places than upstream precision supported; such
    discrepancies remain failed comparisons at the literal displayed precision.
    No quadrature node is replaced by a rounded header's square root.
    """
    sources = [s for s in model.sources if s.role == "independent_frequency_headers"]
    if not sources:
        return ()
    source, = sources
    rows = _json(source.text)["rows"]
    out = []
    for row in rows:
        k = row["canonical_index"]
        if type(k) is not int or not 0 <= k < len(model.frequencies):
            raise ValueError("invalid comparison frequency index")
        token = row["freqsq_token"]
        if not re.search(r"FREQ(?:SQ|2)\s+"+re.escape(token)+r"(?:\s|$)", row["header"]):
            raise ValueError("comparison token/header mismatch")
        expected = -model.frequencies[k]**2
        error, tolerance = abs(expected-_number(token)), printed_tolerance(token)
        out.append(Comparison("frequency_header:"+Path(row["source"]).name, error, tolerance,
                              printed_matches(expected, token), "FREQSQ/FREQ2=-xi^2, literal printed precision", 1,
                              error/tolerance))
    return tuple(out)


def compare_casimir_data(model, *, input_rounding=5e-13):
    """Compare the separately serialized, ten-dynamic-frequency Casimir input.

    Narrow Print nonzero format only; omitted entries are explicit serialized zeros.
    Upper triangle is mirrored in the comparison oracle, NEVER in imported tensors.
    Allow half a last printed place plus the supplied .pol input rounding allowance
    (default twelve-decimal archive). Failures return records, not suppressed C_n.
    """
    source, = [s for s in model.sources if s.role == "independent_casimir_data"]
    if not np.isfinite(input_rounding) or input_rounding < 0:
        raise ValueError("invalid input rounding allowance")
    lines = [x.strip() for x in source.text.splitlines() if x.strip() and not x.lstrip().startswith("!")]
    if "Frequencies   0.5    10" not in lines or "Print nonzero" not in lines or "Skip  0" not in lines:
        raise ValueError("unsupported independent Casimir frequency/printing convention")
    entries, site, i = {}, None, 0
    components = {s.label: model.tensors[j].components for j, s in enumerate(model.sites)}
    while i < len(lines):
        line = lines[i]
        m = re.fullmatch(r"Site\s+(\S+)\s+type\s+\S+", line)
        if m:
            site = m[1]
            if site not in components or site in entries:
                raise ValueError("unknown/duplicate Casimir site")
            entries[site] = {}
        elif line == "End":
            site = None
        elif site is not None:
            pair = line.split()
            if len(pair) != 2 or any(c not in components[site] for c in pair):
                raise ValueError("unknown Casimir component syntax")
            key = tuple(components[site].index(c) for c in pair)
            if key in entries[site] or key[0] > key[1]:
                raise ValueError("duplicate/non-upper Casimir component")
            tokens = []
            while len(tokens) < 10:
                i += 1
                if i >= len(lines):
                    raise ValueError("truncated Casimir dynamic series")
                tokens.extend(lines[i].split())
            if len(tokens) != 10:
                raise ValueError("invalid Casimir dynamic series size")
            for t in tokens:
                _number(t)
            entries[site][key] = tokens
        elif not (line.startswith(("Title ", "Frequencies ", "Skip ", "Molecule ", "CGdir ", "Dispersion ")) or line in ("Print nonzero", "Finish")):
            raise ValueError("unknown Casimir syntax")
        i += 1
    if set(entries) != set(components):
        raise ValueError("missing Casimir site")
    comparisons = []
    for s, raw in zip(model.sites, model.raw_tensors):
        actual = raw[1:]
        expected, tolerance = np.zeros_like(actual), np.full_like(actual, input_rounding)
        for (a, c), tokens in entries[s.label].items():
            vals = [_number(t) for t in tokens]
            tols = [printed_tolerance(t)+input_rounding for t in tokens]
            expected[:, a, c] = expected[:, c, a] = vals
            tolerance[:, a, c] = tolerance[:, c, a] = tols
        error = np.abs(actual-expected)
        # Include floating conversion ulps, not a physics tolerance.
        tolerance += 8*np.finfo(float).eps*np.maximum(1, np.abs(expected))
        comparisons.append(Comparison("serialized_casimir_data:"+s.label, float(error.max()),
                                      float(tolerance.max()), bool(np.all(error <= tolerance)),
                                      "local real Racah; ten dynamic nodes; per-token half-place plus input rounding",
                                      int(error.size), float(np.max(error/tolerance))))
    return tuple(comparisons)


@dataclass(frozen=True)
class SuppliedProperties:
    model_a: LocalResponse
    model_b: LocalResponse
    isotropic: tuple
    anisotropic: tuple
    anisotropic_status: str
    core_version: str
    core_path: str
    core_sha256: str
    driver_sha256: str
    warnings: tuple
    placement_b: Optional[Placement] = None
    placement_source: Optional[Source] = None
    comparisons: tuple = ()
    mode: str = "supplied_external_local_response"
    structural_parse_success: bool = True
    computed_status: str = "available"
    native_verified: bool = False
    wavefunction_first: bool = False
    scalar_alpha_origin: str = "Psi4_trace_of_imported_tensor"
    dispersion_origin: str = "Psi4_computed_from_imported_local_response"
    anisotropic_kind: str = "orientation_resolved_scalars_not_recoupled_CamCASP_components"
    units: str = "alpha_llprime: bohr^(l+lprime+1); C_n: Eh bohr^n; energy: Eh"
    todos: tuple = TODOS

    def __post_init__(self):
        for name in ("isotropic", "anisotropic", "warnings", "comparisons", "todos"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    @property
    def numerical_agreement(self):
        return None if not self.comparisons else all(c.agreement for c in self.comparisons)

    def to_dict(self):
        def encode(x):
            if is_dataclass(x):
                d = {f.name.lstrip("_"): encode(getattr(x, f.name)) for f in fields(x)}
                if isinstance(x, LocalResponse):
                    d["atomic_scalars"] = encode(x.atomic_scalars)
                    d["global_cartesian_dipoles"] = encode(x.global_dipoles)
                    d["components_by_site"] = [list(x.tensors[j].components) for j in range(len(x.sites))]
                return d
            if isinstance(x, np.ndarray):
                return x.tolist()
            if isinstance(x, (tuple, list)):
                return [encode(y) for y in x]
            return x
        d = encode(self)
        d["numerical_agreement"] = self.numerical_agreement
        # Validate finiteness even if callers construct/replace a comparison record.
        json.dumps(d, allow_nan=False)
        return d

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2, allow_nan=False)+"\n"


def _core_model(model, core, anisotropic=False):
    sites = []
    for s, raw, scalars in zip(model.sites, model.raw_tensors, model.atomic_scalars):
        out = core.IsaAnisotropicSite() if anisotropic else core.IsaIsotropicSite()
        out.label, out.origin, out.ranks = s.label, list(s.origin), list(s.ranks)
        if anisotropic:
            if not np.array_equal(raw, raw.transpose(0, 2, 1)):
                raise ValueError("strict anisotropic conversion requires exact reciprocity; raw retained; no projection implemented")
            out.frame = s.frame.tolist()
            out.responses = [core.Matrix.from_array(a) for a in raw]
        else:
            out.polarizabilities = core.Matrix.from_array(scalars)
        sites.append(out)
    provenance = model.tensor_origin+"; track="+model.track+"; manifest_sha256="+model.manifest_sha256
    if anisotropic:
        return core.IsaAnisotropicModel(list(model.frequencies), sites, "supplied_local_response", provenance)
    return core.IsaIsotropicModel(list(model.frequencies), sites, provenance)


def _pairs(result, anisotropic=False):
    out = []
    for p in result.pairs:
        coeffs = []
        for c in p.coefficients:
            inc = c.included_rank_quadruples if anisotropic else c.included_rank_pairs
            miss = c.missing_rank_quadruples if anisotropic else c.missing_rank_pairs
            coeffs.append(Coefficient(c.order, c.value, tuple(map(tuple, inc)), tuple(map(tuple, miss)),
                                      c.unrestricted_complete if anisotropic else c.complete,
                                      c.declared_model_complete if anisotropic else True,
                                      c.energy if anisotropic else None))
        out.append(Pair(p.site_a, p.site_b, tuple(coeffs), p.distance if anisotropic else None,
                        tuple(p.direction) if anisotropic else (), p.truncated_energy if anisotropic else None))
    return tuple(out)


def supplied_local_properties(model_a, model_b=None, *, placement_b=None, anisotropic=False):
    """Compute all-site-pair tables with existing kernels, no implicit projection.

    Isotropic defaults to A versus A. Anisotropic requires explicit B AND placement;
    coincident geometry fails structurally. An asymmetric requested conversion is
    reported as rejected while valid isotropic outputs remain available. Reference
    comparisons are separate Comparison records and never gate calculation output.
    """
    from psi4 import core, __version__
    if anisotropic and (model_b is None or placement_b is None):
        raise ValueError("anisotropic requires explicit model_b and placement_b")
    b = model_a if model_b is None else model_b
    if placement_b is not None:
        b = place_model(b, placement_b)
    if model_a.frequencies != b.frequencies or model_a.cp_weights != b.cp_weights:
        raise ValueError("A/B grids and CP weights must match exactly")
    if anisotropic and any(np.array_equal(a.origin, s.origin) for a in model_a.sites for s in b.sites):
        raise ValueError("coincident A/B sites: provide noncoincident placement")
    iso = core.isa_isotropic_dispersion(_core_model(model_a, core), _core_model(b, core), list(model_a.cp_weights), 12)
    aniso, status = (), "not_requested"
    warnings = list(dict.fromkeys(model_a.warnings+b.warnings))
    if anisotropic:
        if any(not np.array_equal(a, a.transpose(0, 2, 1)) for m in (model_a, b) for a in m.raw_tensors):
            status = "rejected_nonreciprocal_raw_input"
            warnings.append("Requested anisotropic conversion failed exact symmetry; isotropic C_n remain available. No projection/repair.")
        else:
            result = core.isa_anisotropic_dispersion(_core_model(model_a, core, True), _core_model(b, core, True), list(model_a.cp_weights), 12)
            aniso, status = _pairs(result, True), "available"
    corepath = Path(core.__file__).resolve()
    return SuppliedProperties(model_a, b, _pairs(iso), aniso, status, __version__, str(corepath),
                              hashlib.sha256(corepath.read_bytes()).hexdigest(),
                              hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), tuple(warnings), placement_b=placement_b)
