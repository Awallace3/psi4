"""Portable pure-parser tests. No private archive or compiled Psi4 dependency.

The injected independent NumPy grid here tests mapping, not core grid parity;
production CasimirGrid/dispersion tests live in test_isapol_supplied_driver.py.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[2]/"psi4/driver/procrouting/isapol_supplied.py"
spec = importlib.util.spec_from_file_location("orient_bridge_pure", MODULE)
b = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = b
spec.loader.exec_module(b)
DATA = Path(__file__).parent/"data_isapol/orient_local"


class IndependentGrid:
    def __init__(self, n, beta):
        t, w = np.polynomial.legendre.leggauss(n)
        self.f = np.r_[0., beta*(1+t)/(1-t)]
        self.w = np.r_[0., w*2*beta/(1-t)**2/(2*np.pi)]

    def omega(self, k):
        return self.f[k]

    def cp_weight(self, k):
        return self.w[k]


CORE = SimpleNamespace(CasimirGrid=IndependentGrid)
HAND = """# INDEX 000
ALPHA H2O SITE-NAMES H1 H1 RANK 1 TO 1 INDEX 0 FREQSQ 0.0000000
2D0 0.25d0 0
0.25D0 -1D0 0
0 0 3D0
ENDFILE
"""


def test_handwritten_dense_d_exponents_and_copies():
    t, = b.parse_orient_new(HAND)
    assert t.components == ("10", "11c", "11s")
    np.testing.assert_array_equal(t.values, [[2, .25, 0], [.25, -1, 0], [0, 0, 3]])
    assert t.index == 0 and t.section == 0 and t.freqsq == "0.0000000"
    a = t.values; a[:] = 99
    assert t.values[0, 1] == .25
    assert t.header.startswith("ALPHA H2O")


@pytest.mark.parametrize("text", [
    HAND.replace("H1 H1", "H1 H2"), HAND.replace("SITE-NAMES", "SITE-LABELS"),
    HAND.replace("RANK 1", "RANK 0"), HAND.replace("TO 1", "TO 5"),
    HAND.replace("TO 1", "TO 2"), HAND.replace("2D0", "NaN"),
    HAND.replace("2D0", "inf"), HAND.replace("2D0", "1e999"),
    HAND.replace("0 0 3D0\n", ""), HAND.replace("0 0 3D0", "0 0 3D0 4"),
    HAND.replace("ENDFILE", "FINISH"), HAND.replace("FREQSQ", "FREQ2"),
    "O O\n1 2 3\n", HAND + "# INDEX 000\n" + HAND,
    HAND + "# INDEX 001\n", HAND.replace("# INDEX 000", "# INDEX bad"),
    HAND.replace("ENDFILE", "0 0 0\nENDFILE"),
])
def test_structural_invalidity(text):
    with pytest.raises(ValueError):
        b.parse_orient_new(text)


def test_sections_not_deduplicated_and_reordered():
    t = b.parse_orient_new(HAND.replace("000", "010", 1)+HAND)
    assert [x.section for x in t] == [10, 0]
    assert [x.index for x in t] == [0, 0]


def prepare(tmp_path, *, text=None, change=None):
    m = json.loads((DATA/"manifest.json").read_text())
    # All source files here are portable bounded snapshots.
    for s in m["provenance_sources"]:
        (tmp_path/s["name"]).write_bytes((DATA/s["name"]).read_bytes())
    name = m["files"][0]["name"]
    content = (DATA/name).read_text() if text is None else text
    (tmp_path/name).write_text(content)
    m["files"][0]["sha256"] = hashlib.sha256(content.encode()).hexdigest()
    if change:
        change(m)
    raw = json.dumps(m).encode()
    path = tmp_path/"manifest.json"; path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def load(tmp_path, **kw):
    manifest, sha = prepare(tmp_path, **kw)
    return b.read_orient_local_response(tmp_path, manifest, manifest_sha256=sha, core_module=CORE)


@pytest.mark.parametrize('defect', ['premature', 'missing', 'repeated', 'leading'])
def test_termination_enforced_by_parser_and_manifest_reader(tmp_path, defect):
    text = (DATA/'H2O_ref_wt4_L3_0f10.pol').read_text()
    if defect == 'premature':
        lines = text.splitlines()
        index = next(i for i, line in enumerate(lines)
                     if line.split()[:5] == ['ALPHA', 'H2O', 'SITE-NAMES', 'H1', 'H1'])
        lines.insert(index, 'ENDFILE')
        text = '\n'.join(lines)+'\n'
    elif defect == 'missing':
        text = text.replace('ENDFILE', '')
    elif defect == 'repeated':
        text = text.replace('ENDFILE', 'ENDFILE\nENDFILE', 1)
    else:
        text = 'ENDFILE\n'+text
    with pytest.raises(ValueError, match='ENDFILE'):
        b.parse_orient_new(text)
    with pytest.raises(ValueError, match='ENDFILE'):
        load(tmp_path, text=text)


def test_individual_unsectioned_termination():
    text = HAND.replace('# INDEX 000\n', '')
    record, = b.parse_orient_new(text)
    assert record.section is None
    with pytest.raises(ValueError, match='ENDFILE'):
        b.parse_orient_new(text.replace('ENDFILE', ''))
    with pytest.raises(ValueError, match='ENDFILE'):
        b.parse_orient_new(text+text)


def test_actual_refined_33_full_tensors():
    m = b.read_orient_local_response(DATA, DATA/"manifest.json", core_module=CORE)
    assert len(m.tensors) == 33
    assert all(a.shape == (11, 15, 15) for a in m.raw_tensors)
    assert all(t.index == 0 and float(t.freqsq) == 0 for t in m.tensors)
    assert len({t.section for t in m.tensors}) == 11
    assert all(d.max_asymmetry == 0 for d in m.diagnostics)
    assert sum(d.minimum_symmetric_eigenvalue < 0 for d in m.diagnostics) == 22
    assert sum("raw INDEX/FREQSQ conflict" in w for w in m.warnings) == 30
    assert m.tensors[0].values[0, 0] == 5.583657081749  # literal archive numerical anchor
    assert m.tensors[1].values[0, 1] == -.005761700031
    assert m.global_dipoles[1][0, 0, 2] == .005761700031
    assert m.global_dipoles[2][0, 0, 2] == -.005761700031
    a = m.raw_tensors[0]; a[:] = 0
    assert m.raw_tensors[0][0, 0, 0] == 5.583657081749
    m.sites[1].frame[:] = 0
    assert m.sites[1].frame[0, 0] == -1


def test_actual_independent_serialized_casimir():
    m = b.read_orient_local_response(DATA, DATA/"manifest.json", core_module=CORE)
    comparisons = b.compare_casimir_data(m)
    assert len(comparisons) == 3
    assert all(c.agreement for c in comparisons)
    assert sum(c.checked_points for c in comparisons) == 3*10*225


def test_bad_headers_require_authority(tmp_path):
    with pytest.raises(ValueError, match="contradiction"):
        load(tmp_path, change=lambda m: m.update(authority="headers"))


@pytest.mark.parametrize("mutation", [
    lambda m: m["sites"][1]["origin"].__setitem__(0, 1),
    lambda m: m["sites"][1].update(frame=np.eye(3).tolist()),
    lambda m: m.update(global_frame_sites=[]),
    lambda m: m.update(geometry_units="angstrom"),
    lambda m: m.update(tensor_origin="distributed"),
    lambda m: m["sites"][0].update(ranks=[1, 2]),
    lambda m: m["files"][0]["sections"].pop("10"),
    lambda m: m["files"][0]["sections"].update({"10": 0}),
    lambda m: m.update(grid={"n": 10, "beta": .3}),
])
def test_manifest_mismatches(tmp_path, mutation):
    with pytest.raises(ValueError):
        load(tmp_path, change=mutation)


def test_missing_duplicate_unknown_site_and_reordering(tmp_path):
    text = (DATA/"H2O_ref_wt4_L3_0f10.pol").read_text()
    with pytest.raises(ValueError, match="duplicate"):
        load(tmp_path, text=text.replace("SITE-NAMES  H2  H2", "SITE-NAMES  H1  H1"))
    with pytest.raises(ValueError, match="unknown site"):
        load(tmp_path, text=text.replace("SITE-NAMES  H2  H2", "SITE-NAMES  X  X"))
    # Split independently on headers/section markers, permute all sections and sites.
    sections = text.split("# INDEX ")[1:]
    reordered = []
    for sec in reversed(sections):
        marker, rest = sec.split("\n", 1)
        blocks = rest.replace("ENDFILE", "").split("ALPHA")[1:]
        reordered.append("# INDEX "+marker+"\n"+"".join("ALPHA"+x for x in reversed(blocks))+"ENDFILE\n")
    model = load(tmp_path, text="".join(reordered))
    assert model.tensors[0].site == "O" and model.tensors[-1].section == 10
    with pytest.raises(ValueError, match="section/grid mapping"):
        load(tmp_path, text="".join(reordered[:-1]))
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if "SITE-NAMES  H2  H2" in line)
    del lines[start:start+16]
    with pytest.raises(ValueError, match="missing site"):
        load(tmp_path, text="\n".join(lines))


def test_source_hash_and_manifest_mutation(tmp_path):
    path, sha = prepare(tmp_path)
    name = "H2O_ref_wt4_L3_0f10.pol"
    (tmp_path/name).write_text((tmp_path/name).read_text()+"\n")
    with pytest.raises(ValueError, match="SHA256"):
        b.read_orient_local_response(tmp_path, path, manifest_sha256=sha, core_module=CORE)
    path.write_text(path.read_text()+"\n")
    with pytest.raises(ValueError, match="SHA256"):
        b.read_orient_local_response(tmp_path, path, manifest_sha256=sha, core_module=CORE)


def test_tiny_asymmetry_and_indefinite_raw_retained(tmp_path):
    text = (DATA/"H2O_ref_wt4_L3_0f10.pol").read_text()
    text = text.replace("-0.676364753503", "-0.676364753502", 1)
    m = load(tmp_path, text=text)
    assert m.tensors[0].values[0, 3] != m.tensors[0].values[3, 0]
    assert m.diagnostics[0].max_asymmetry > 0
    assert any("asymmetric" in w for w in m.warnings)
    assert any("indefinite" in w for w in m.warnings)


def test_printed_frequency_precision_not_exact_float():
    grid = IndependentGrid(10, .5)
    assert b.printed_matches(-grid.omega(1)**2, "-0.0000437")
    assert b.printed_matches(-grid.omega(1)**2, "-0.4368683D-04")
    assert not b.printed_matches(-grid.omega(1)**2, "0.0000000")
    assert b.printed_tolerance("-0.4368683D-04") == 5e-12


def test_placement_owns_inputs_and_rotates_origins_and_frames():
    m = b.read_orient_local_response(DATA, DATA/"manifest.json", core_module=CORE)
    r = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.]])
    t = np.array([0., 0., 10.]); p = b.Placement(t, r)
    q = b.place_model(m, p)
    np.testing.assert_allclose(q.sites[1].origin, r@np.array(m.sites[1].origin)+t)
    np.testing.assert_allclose(q.sites[1].frame, r@m.sites[1].frame)
    t[:] = 0; r[:] = 0
    assert p.translation[2] == 10 and p.rotation[1, 0] == 1


def test_all_frequency_header_crosschecks_retained():
    m = b.read_orient_local_response(DATA, DATA/"manifest.json", core_module=CORE)
    checks = b.compare_frequency_headers(m)
    assert len(checks) == 22
    failed = {c.name for c in checks if not c.agreement}
    assert failed == {f"frequency_header:H2O_L3_{k:03}.pol" for k in range(7, 11)}
    assert all(c.agreement for c in checks if "NL4" in c.name)


def test_explicit_individual_file_mapping_and_input_reorder(tmp_path):
    path, _ = prepare(tmp_path)
    m = json.loads(path.read_text())
    text = (DATA/"H2O_ref_wt4_L3_0f10.pol").read_text()
    paths, specs = [], []
    for section in text.split("# INDEX ")[1:]:
        label, contents = section.split("\n", 1)
        k = int(label)
        filename = f"individual_{k:03}.pol"
        dest = tmp_path/filename
        dest.write_text(contents)
        paths.append(dest)
        specs.append(dict(name=filename, sha256=hashlib.sha256(contents.encode()).hexdigest(), sections={"file": k}))
    m["files"] = specs
    raw = json.dumps(m).encode(); path.write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    model = b.read_orient_local_response(list(reversed(paths)), path, manifest_sha256=sha, core_module=CORE)
    assert len(model.tensors) == 33 and all(t.section is None for t in model.tensors)
    assert model.tensors[0].values[0, 0] == 5.583657081749
    with pytest.raises(ValueError, match="files mismatch"):
        b.read_orient_local_response(paths[:-1], path, manifest_sha256=sha, core_module=CORE)
    with pytest.raises(ValueError, match="duplicate input"):
        b.read_orient_local_response(paths+paths[:1], path, manifest_sha256=sha, core_module=CORE)


def test_manifest_is_mandatory_and_hash_pinned(tmp_path):
    path, sha = prepare(tmp_path)
    with pytest.raises(FileNotFoundError):
        b.read_orient_local_response(tmp_path, path, core_module=CORE)
    path.write_text(path.read_text().replace('"schema": 1', '"schema": 1, "schema": 1'))
    changed_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="duplicate manifest key"):
        b.read_orient_local_response(tmp_path, path, manifest_sha256=changed_sha, core_module=CORE)


def test_separately_serialized_comparison_failure_is_data_not_exception():
    from dataclasses import replace
    m = b.read_orient_local_response(DATA, DATA/"manifest.json", core_module=CORE)
    src = next(s for s in m.sources if s.role == "independent_casimir_data")
    changed = replace(src, text=src.text.replace("5.582772756", "6.582772756", 1))
    model = replace(m, sources=tuple(changed if s is src else s for s in m.sources))
    comparisons = b.compare_casimir_data(model)
    assert not comparisons[0].agreement
    assert model.raw_tensors[0][1, 0, 0] == m.raw_tensors[0][1, 0, 0]
