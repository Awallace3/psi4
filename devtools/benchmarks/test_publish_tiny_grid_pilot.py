from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import publish_tiny_grid_pilot as publication


def test_grac_attempts_and_sad_accumulate():
    text = """GRAC (Monomer A)
Computing GRAC shift for Monomer A using iterative
==> GRAC Monomer A Given Molecule: first <==
@RHF iter SAD:  0.0 XC-INITIAL
@RHF iter 1: 0.0 XC-FINAL
==> GRAC Monomer A Electron Removed Molecule: first <==
@UHF iter 1: 0.0 XC-INITIAL
==> GRAC Monomer A Given Molecule: retry <==
@RHF iter 1: 0.0 XC-FINAL
GRAC shift Monomer A: 0.1
SAPT(DFT): delta HF Dimer
@RHF iter 1: 0.0
SAPT(DFT): delta HF Segment
SAPT(DFT): DFT Monomer A
@RHF iter 1: 0.0 XC-FINAL
"""
    sections = publication.iteration_sections(text)
    assert sections["gracA"]["total"] == 4
    assert sections["gracA"]["initial"] == sections["gracA"]["final"] == 2
    assert sections["hfD"]["untagged"] == 1
    assert sum(s["total"] for s in sections.values()) == 6


def test_unassigned_iterations_are_an_accounting_error():
    with pytest.raises(AssertionError, match="Unassigned"):
        publication.iteration_sections("@RHF iter 1: 0.0")


def test_new_banners_end_previous_section():
    with pytest.raises(AssertionError, match="Unassigned"):
        publication.iteration_sections(
            "SAPT(DFT): DFT Monomer B\n@RHF iter 1: 0.0\n"
            "SAPT(DFT): Localization\n@RHF iter 1: 0.0\n")


def test_fixed_grid_iterations_stay_untagged():
    sections = publication.iteration_sections(
        "SAPT(DFT): DFT Monomer B\n@RHF iter SAD: 0.0\n@RHF iter 1: 0.0\n")
    assert sections["dftB"]["total"] == sections["dftB"]["untagged"] == 2
    assert sections["dftB"]["final"] == 0


def test_timing_boundaries_remain_inclusive():
    sections = publication.iteration_sections("")
    publication.attach_times(sections, {
        "SAPT(DFT) Energy;SAPT(DFT):GRAC Shift Monomer A": {"wall_time": 3.},
        "SAPT(DFT) Energy;SAPT(DFT):GRAC Shift Monomer A;SCF:Driver;RV: Form V": {"wall_time": 1.},
        "SAPT(DFT) Energy;SAPT(DFT):GRAC Shift Monomer A;SCF:Driver;RV: Form V;Functional": {"wall_time": .2},
    })
    assert sections["gracA"]["wall_s"] == 3.
    assert sections["gracA"]["xc_s"] == 1.


def test_missing_failed_result_keeps_planned_grid_and_unavailable_counts():
    data = publication.prepare(dict(
        systems=["water"], arms=["tiny05"], job_id="14097678", threads=16),
        [dict(system="water", arm="tiny05", ok=False, error="Killed before output",
              sections=[], iterations_observed=False)])
    row = data["groups"][0]["rows"][0]
    assert row["label"] == "15×74 → 75×302"
    assert row["total_iterations"] is None
    assert "wall_s" not in row and "speedup" not in row
