"""Publication guards must fail closed before a timing enters the report."""
import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import publish_protein157_grids as publication


@pytest.fixture
def matched():
    geometry = "synthetic geometry for guard tests"
    manifest = dict(selected=dict(nbf=2171, auxiliary="def2-universal-jkfit"),
                    expected_package="/pinned/lib/psi4", threads=8)
    row = dict(ok=True, job_id=publication.JOB, scientific_commit=publication.SCIENCE,
               system="protein157", arm="coarse75", backend="cuda", nbf=2171,
               auxiliary="def2-universal-jkfit", geometry=geometry,
               geometry_sha256=hashlib.sha256(geometry.encode()).hexdigest(),
               package="/pinned/lib/psi4/__init__.py", threads=8, memory="112 GiB",
               diagnostic_only=False, eligible_for_speedup_bars=True,
               timers={"cuEST XC: CUDA LibXC Functional": {}},
               mapped_libxc_sha256={"/software/libxc-7.1.2-cuda/lib64/libxc.so.15": "hash"},
               options=dict(BASIS="heavy-aug-cc-pvdz", SCF_TYPE="DF",
                   SAPT_DFT_FUNCTIONAL="PBE0", SAPT_DFT_INDUCTION_TYPE="NONE",
                   SAPT_DFT_GRAC_COMPUTE="ITERATIVE", SAPT_DFT_GRAC_SEED_CATION=False,
                   SAPT_DFT_DO_DHF=True, USE_CUEST=True, CUEST_XC=True, CUEST_SAD=True,
                   CUEST_MIXED_PRECISION=False, CUEST_XC_TIMING=False,
                   INTS_TOLERANCE=1.e-12, DFT_GRID_SWITCH_FACTOR=10.,
                   DFT_RADIAL_POINTS=75, DFT_SPHERICAL_POINTS=302,
                   DFT_RADIAL_POINTS_INITIAL=35, DFT_SPHERICAL_POINTS_INITIAL=194))
    return row, manifest


def check(row, manifest, monkeypatch):
    monkeypatch.setattr(publication, "GEOMETRY", hashlib.sha256(
        "synthetic geometry for guard tests".encode()).hexdigest())
    publication.validate_row(row, manifest, "coarse75", diagnostic=False)


def test_matching_timing_accepted(matched, monkeypatch):
    check(*matched, monkeypatch)


@pytest.mark.parametrize("key,value", [
    ("backend", "cpu"), ("nbf", 922), ("scientific_commit", "wrong"),
    ("diagnostic_only", True), ("eligible_for_speedup_bars", False),
    ("package", "/wrong/lib/psi4/__init__.py"), ("geometry", "different geometry"),
    ("mapped_libxc_sha256", {"/software/libxc-host/libxc.so": "hash"}),
])
def test_mismatched_row_rejected(matched, monkeypatch, key, value):
    row, manifest = matched
    row[key] = value
    with pytest.raises(AssertionError):
        check(row, manifest, monkeypatch)


@pytest.mark.parametrize("key,value", [
    ("CUEST_MIXED_PRECISION", True), ("CUEST_XC_TIMING", True),
    ("SCF_TYPE", "DIRECT"), ("DFT_RADIAL_POINTS", 99),
    ("DFT_RADIAL_POINTS_INITIAL", 50), ("SAPT_DFT_INDUCTION_TYPE", "CPKS"),
    ("SAPT_DFT_GRAC_SEED_CATION", True),
])
def test_changed_science_rejected(matched, monkeypatch, key, value):
    row, manifest = matched
    row["options"][key] = value
    with pytest.raises(AssertionError):
        check(row, manifest, monkeypatch)


def test_microprofiled_timing_rejected(matched, monkeypatch):
    row, manifest = matched
    row["timers"]["XC micro: density"] = {}
    with pytest.raises(AssertionError):
        check(row, manifest, monkeypatch)
