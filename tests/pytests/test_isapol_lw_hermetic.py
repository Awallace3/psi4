# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Historical supplied NL4 -> native LW -> external unrefined L3 diagnostic.

No reference-tree reads or executables, no PFIT fixture, no missing-binding skips.
The production 1e-6 postcondition rejects this archived input, tested separately.
User-authorized historical-only 1e-3 enables the distinct 1e-11/rtol=0 comparison
of all 675 entries. Diagnostic success is NOT production postcondition acceptance.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import psi4

pytestmark = [pytest.mark.psi, pytest.mark.api]
DATA = Path(__file__).parent / "data_isapol" / "orient_local"
FIXTURE = DATA / "lw-hermetic-water.json"
FIXTURE_HASH = "b86d411e5fd81fc20358370ee211ba5b9bd997c57f8918c32ce0334c93ad7f49"
REFERENCE_HASHES = {
    "H2O_NL4_000.pol": "9b6130f42fc50b50b80d5860f13cc6b5b198002c4c74a1b3500893481502c166",
    "H2O_L3_000.pol": "9c027937ffe3eb8c80c67015dc927983ca70464bd7e265e0f417eb59a5114172",
}
RESIDUAL_NAMES = ("off_site", "charge_sum", "reciprocity", "molecular_sum", "local_charge",
                  "input_sum_rule", "charge_sum_transport")
LABELS = ("O", "H1", "H2")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def fixture():
    assert _sha(FIXTURE) == FIXTURE_HASH
    assert FIXTURE.with_suffix(".json.sha256").read_text().split() == [FIXTURE_HASH, FIXTURE.name]
    return json.loads(FIXTURE.read_text())


@pytest.fixture(scope="module")
def extractor():
    # Import our pure parser only; the opt-in CLI never runs during pytest.
    path = DATA.parent / "oracle" / "extract_lw_hermetic.py"
    spec = importlib.util.spec_from_file_location("lw_hermetic_extractor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _portable_pol(record, distributed):
    lines = []
    for section in record["sections"]:
        lines.append(section["header"])
        lines.extend(" ".join(row) for row in section["values"])
        if distributed:
            lines.append("END")
    return "\n".join(lines + ["ENDFILE"]) + "\n"


def _arrays(record):
    return np.array([s["values"] for s in record["sections"]], dtype=float)


def _truncate(raw):
    assert raw.shape == (9, 25, 25)
    return raw[:, :16, :16].copy()


def _matrix(array):
    return psi4.core.Matrix.from_array(np.array(array, dtype=float))


def _rotation(frame):
    # D(F) maps local moments to global moments. Therefore global response ->
    # local response is D(F)^T A_global D(F). Drop ONLY the rank-0 rotation row/col.
    return np.asarray(psi4.core.isa_multipole_rotation(3, frame))[1:16, 1:16]


def test_lw_fixture_identity_schema_and_authority(fixture, extractor):
    f = fixture
    assert f["schema_version"] == 1
    assert (f["molecule"], f["frequency"], f["units"], f["geometry_units"]) == ("H2O", 0.0, "atomic", "bohr")
    assert (f["input_frame"], f["expected_frame"], f["frame_convention"]) == (
        "global", "site_local", "local_to_global_columns")
    assert f["bonds_zero_based"] == [[0, 1], [0, 2]]
    assert f["component_order"].split(",") == [
        "00", "10", "11c", "11s", "20", "21c", "21s", "22c", "22s",
        "30", "31c", "31s", "32c", "32s", "33c", "33s",
        "40", "41c", "41s", "42c", "42s", "43c", "43s", "44c", "44s"]
    assert set(f["authority_sha256"]) == {"manifest.json", "H2O.sites", "H2O.axes"}
    for name, digest in f["authority_sha256"].items():
        assert _sha(DATA / name) == digest
    assert (DATA / "manifest.json.sha256").read_text().split()[0] == f["authority_sha256"]["manifest.json"]
    manifest = json.loads((DATA / "manifest.json").read_text())
    assert f["sites"] == manifest["sites"]
    assert [s["label"] for s in f["sites"]] == list(LABELS)
    site_lines = (DATA / "H2O.sites").read_text().splitlines()
    assert site_lines[0] == "! Units  BOHR"
    assert site_lines[1] == "Sites" and site_lines[5] == "End"
    for site, line in zip(f["sites"], site_lines[2:5]):
        tokens = line.split()
        assert tokens[0] == site["label"]
        assert [float(v) for v in tokens[1:4]] == site["origin"]
    assert (DATA / "H2O.axes").read_text().split() == (
        "Axes H1 z global Z x from H2 to H1 H2 z global Z x from H1 to H2 End".split())
    np.testing.assert_array_equal(f["sites"][0]["frame"], np.eye(3))
    np.testing.assert_array_equal(f["sites"][1]["frame"], np.diag([-1, -1, 1]))
    np.testing.assert_array_equal(f["sites"][2]["frame"], np.eye(3))
    assert f["provenance"]["extractor_sha256"] == _sha(DATA.parent / "oracle" / "extract_lw_hermetic.py")
    headers = json.loads((DATA / "frequency_header_excerpt.json").read_text())["rows"]
    for key, distributed, size in [("distributed", True, 25), ("expected_local", False, 15)]:
        record = f[key]
        assert record["sha256"] == REFERENCE_HASHES[record["filename"]]
        assert record["raw_frequency_index"] == 1 and float(record["frequency_squared"]) == 0.0
        assert (record["rank_min"], record["rank_max"]) == ((0, 4) if distributed else (1, 3))
        assert record["representation"] == "real_Racah_spherical"
        # Cross-check previously imported static header identities, without following source paths.
        old = next(row for row in headers if row["canonical_index"] == 0
                   and Path(row["source"]).name == record["filename"])
        assert old["sha256"] == record["sha256"]
        assert old["header"] == record["sections"][0]["header"]
        assert extractor.parse_pol(_portable_pol(record, distributed), distributed) == record["sections"]
        arrays = _arrays(record)
        assert arrays.shape == ((9 if distributed else 3), size, size)
        assert np.isfinite(arrays).all()


@pytest.mark.parametrize("distributed", [True, False])
@pytest.mark.parametrize("fault", ["label", "index", "rank", "frequency", "representation",
                                   "short_row", "extra_row", "number", "end", "trailing", "missing_section"])
def test_lw_extraction_rejects_malformed_sections(fixture, extractor, distributed, fault):
    record = fixture["distributed" if distributed else "expected_local"]
    lines = _portable_pol(record, distributed).splitlines()
    if fault == "label":
        lines[0] = lines[0].replace("  O  O", "  O  H1")
    elif fault == "index":
        lines[0] = lines[0].replace("SITE-INDICES     1     1", "SITE-INDICES     2     1") if distributed else lines[0].replace("INDEX   1", "INDEX   2")
    elif fault == "rank":
        lines[0] = lines[0].replace("RANK  0", "RANK  1") if distributed else lines[0].replace("TO 3", "TO 4")
    elif fault == "frequency":
        lines[0] = lines[0].replace("0.0000000", "0.1000000")
    elif fault == "representation":
        lines[0] = lines[0].replace("CARTSPHER S", "CARTSPHER C") if distributed else lines[0] + " CARTSPHER C"
    elif fault == "short_row":
        lines[1] = " ".join(lines[1].split()[:-1])
    elif fault == "extra_row":
        lines.insert(2, lines[1])
    elif fault == "number":
        lines[1] = "NaN " + " ".join(lines[1].split()[1:])
    elif fault == "end":
        lines[26 if distributed else -1] = "BADEND"
    elif fault == "trailing":
        lines.append("ENDFILE")
    elif fault == "missing_section":
        del lines[:27 if distributed else 16]
    with pytest.raises(ValueError):
        extractor.parse_pol("\n".join(lines) + "\n", distributed)


def test_lw_rank4_truncation_and_raw_asymmetry(fixture):
    raw = _arrays(fixture["distributed"])
    working = _truncate(raw)
    assert raw.size == 5625 and working.size == 2304
    assert raw.size - working.size == 3321
    for pair in range(9):
        for row in range(16):
            for col in range(16):
                assert working[pair, row, col] == float(fixture["distributed"]["sections"][pair]["values"][row][col])
    # Every discarded rank-4 row/column must be irrelevant to the selected input.
    poisoned = raw.copy()
    poisoned[:, 16:, :] = np.nan
    poisoned[:, :, 16:] = np.nan
    np.testing.assert_array_equal(_truncate(poisoned), working)
    assert np.count_nonzero(raw[:, 16:, :]) > 0
    assert np.count_nonzero(raw[:, :, 16:]) > 0
    reciprocal = raw.reshape(3, 3, 25, 25).transpose(1, 0, 3, 2).reshape(9, 25, 25)
    assert 0.0 < np.max(np.abs(raw - reciprocal)) < 1e-6
    # Literal unequal transpose entries guard against accidental symmetrization.
    assert raw[0, 0, 2] == 0.5720608e-8
    assert raw[0, 2, 0] == 0.5720595e-8


def test_lw_reference_frame_negative_control(fixture):
    expected = _arrays(fixture["expected_local"])
    for site in (0, 2):
        np.testing.assert_allclose(_rotation(fixture["sites"][site]["frame"]), np.eye(15), atol=1e-14, rtol=0)
    d = _rotation(fixture["sites"][1]["frame"])
    global_reference = d @ expected[1] @ d.T
    # This fixture-only control is NOT a measurement of candidate LW output.
    assert np.max(np.abs(global_reference - expected[1])) > 15.0
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(global_reference, expected[1], atol=1e-11, rtol=0)
    np.testing.assert_allclose(d.T @ global_reference @ d, expected[1], atol=1e-11, rtol=0)


def test_lw_historical_input_rejected_by_production_default(fixture):
    # Retain the strict failure explicitly; never silently retry a production call.
    working = _truncate(_arrays(fixture["distributed"]))
    from decimal import Decimal
    terms = [fixture["distributed"]["sections"][i]["values"][12][0] for i in (3, 4, 5)]
    assert sum(map(Decimal, terms)) == Decimal("-0.0007011")
    with pytest.raises(RuntimeError, match=r"postcondition.*charge-sum=.*local-charge="):
        psi4.core.isa_localize_lw(
            _matrix([s["origin"] for s in fixture["sites"]]),
            [_matrix(block) for block in working], fixture["frequency"], fixture["bonds_zero_based"])


def test_lw_historical_diagnostic_all_675_entries(fixture, record_property):
    raw = _arrays(fixture["distributed"])
    working = _truncate(raw)
    expected = _arrays(fixture["expected_local"])
    assert expected.size == 675
    # Explicit user authorization applies ONLY to this historical fixture.
    # Production default and all 675 expected numbers/output tolerances are unchanged.
    record_property("production_postcondition_accepted", False)
    record_property("historical_diagnostic_postcondition", 1e-3)
    result = psi4.core.isa_localize_lw(
        _matrix([s["origin"] for s in fixture["sites"]]),
        [_matrix(block) for block in working], fixture["frequency"], fixture["bonds_zero_based"], 1e-3)
    residuals = {name: getattr(result.residuals, name) for name in RESIDUAL_NAMES}
    record_property("lw_residuals", json.dumps(residuals, sort_keys=True))
    assert 1e-6 < max(residuals.values()) <= 1e-3
    assert result.frequency == 0.0
    np.testing.assert_array_equal(result.positions, [s["origin"] for s in fixture["sites"]])
    actual_global = np.array([np.asarray(block) for block in result.local])
    assert actual_global.shape == (3, 15, 15)
    actual = np.array([_rotation(s["frame"]).T @ a @ _rotation(s["frame"])
                       for s, a in zip(fixture["sites"], actual_global)])
    errors = np.abs(actual - expected)
    maxima = dict(zip(LABELS, errors.max(axis=(1, 2)).tolist()))
    record_property("lw_site_maxabs", json.dumps(maxima))
    record_property("lw_compared_entries", int(errors.size))
    record_property("lw_transfers", len(result.transfers))
    record_property("lw_omitted_component_pairs", len(result.omitted_component_pairs))
    record_property("lw_omitted_transfers", result.omitted_transfer_count)
    print("LW historical diagnostic (production postcondition FAIL)", json.dumps({"site_maxabs": maxima, "residuals": residuals,
                                      "entries": int(errors.size)}))
    assert np.isfinite(errors).all() and errors.size == 675
    np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=0)
    # Candidate negative frame control, reachable only after the postcondition.
    assert np.max(np.abs(actual_global[1] - expected[1])) > 15.0
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(actual_global[1], expected[1], atol=1e-11, rtol=0)
