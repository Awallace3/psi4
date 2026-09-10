import json

import iterative_accuracy as mod
import pytest

PSI4_OUT = """  ==> GRAC Monomer A Given Molecule: charge=0 mult=1 <==
  ==> GRAC Monomer A Electron Removed Molecule: charge=1 mult=2 <==
 GRAC shift Monomer A: {shift_a:.8f}
 E_given = {neutral_a:.8f}, E_cation = {cation_a:.8f}, HOMO = -0.26368518
 GRAC shift Monomer B: {shift_b:.8f}
 E_given = {neutral_b:.8f}, E_cation = {cation_b:.8f}, HOMO = -0.26368518
"""


def write_case(root, stem, mode, repeat, total, *, shift=0.075, cation=-231.6426, ok=True):
    directory = root / f"{stem}-{mode}-{repeat}"
    directory.mkdir(parents=True)
    (directory / "result.json").write_text(json.dumps({
        "ok": ok, "grac_compute": "ITERATIVE",
        "grac_shifts_hartree": {"A": shift, "B": shift},
        "components_hartree": {key: total for key in mod.COMPONENTS}}))
    (directory / "psi4.out").write_text(PSI4_OUT.format(
        shift_a=shift, shift_b=shift, neutral_a=-231.9814962, neutral_b=-231.9814962,
        cation_a=cation, cation_b=cation))
    return directory


def test_a_different_cation_solution_is_named_rather_than_called_gpu_error(tmp_path):
    # Neutrals agree to 2.5e-7; cations differ by 1.2e-4 -- the benzene signature.
    write_case(tmp_path, "benzene-cc-pvdz", "cpu", 1, -0.004962451,
               shift=0.07513675, cation=-231.64267427)
    # ELST is the term that breaches the gate at 1.503e-6; TOTAL alone stays under it.
    write_case(tmp_path, "benzene-cc-pvdz", "gpu", 1, -0.004962451 + 1.503e-6,
               shift=0.07501733, cation=-231.64279399)

    row = mod.analyze(tmp_path)["benzene-cc-pvdz"]
    assert not row["within_tolerance"]
    assert row["solution_selection_differs"]
    # The GPU cation is the lower of the two, so the GPU has the better solution.
    assert row["better_arm"] == ["gpu"]
    assert "different cation SCF solutions" in row["verdict"]
    assert row["monomers"]["A"]["different_cation_solution"]


def test_matching_solutions_within_tolerance_are_reported_as_agreement(tmp_path):
    write_case(tmp_path, "water-cc-pvdz", "cpu", 1, -0.001, cation=-75.89067660)
    write_case(tmp_path, "water-cc-pvdz", "gpu", 1, -0.001 + 5e-8, cation=-75.89067658)

    row = mod.analyze(tmp_path)["water-cc-pvdz"]
    assert row["within_tolerance"] and not row["solution_selection_differs"]
    assert row["verdict"] == "agrees within tolerance"


def test_same_solution_but_out_of_tolerance_is_flagged_for_investigation(tmp_path):
    # Both SCFs agree tightly, so a large component delta is not explained away.
    write_case(tmp_path, "peptide-6-31+gss", "cpu", 1, -0.010, cation=-247.90888033)
    write_case(tmp_path, "peptide-6-31+gss", "gpu", 1, -0.010 + 5e-5, cation=-247.90888037)

    row = mod.analyze(tmp_path)["peptide-6-31+gss"]
    assert not row["within_tolerance"] and not row["solution_selection_differs"]
    assert row["delta_exceeds_scatter"]
    assert "investigate arithmetic" in row["verdict"]


def test_repeats_are_reduced_by_median_and_their_scatter_recorded(tmp_path):
    for repeat, total in enumerate([-0.001, -0.0015, -0.002], start=1):
        write_case(tmp_path, "water-cc-pvdz", "cpu", repeat, total)
    write_case(tmp_path, "water-cc-pvdz", "gpu", 1, -0.0015)

    row = mod.analyze(tmp_path)["water-cc-pvdz"]
    # Median of the three CPU repeats matches the single GPU run exactly.
    assert row["max_component_delta_eh"] == pytest.approx(0.0, abs=1e-15)
    assert row["run_to_run_scatter_eh"] == pytest.approx(0.001)
    assert row["repeats"] == {"cpu": 3, "gpu": 1}


def test_a_delta_buried_in_run_to_run_scatter_is_not_called_a_disagreement(tmp_path):
    # 5e-6 apart, but each arm wanders by 4e-5 between repeats on its own.
    for repeat, total in enumerate([-0.001, -0.00104], start=1):
        write_case(tmp_path, "water-cc-pvdz", "cpu", repeat, total)
    for repeat, total in enumerate([-0.001005, -0.001045], start=1):
        write_case(tmp_path, "water-cc-pvdz", "gpu", repeat, total)

    row = mod.analyze(tmp_path)["water-cc-pvdz"]
    assert not row["within_tolerance"]
    assert not row["delta_exceeds_scatter"]
    assert "not distinguishable" in row["verdict"]


def test_unpaired_and_failed_cases_are_dropped(tmp_path):
    write_case(tmp_path, "water-cc-pvdz", "cpu", 1, -0.001)
    write_case(tmp_path, "benzene-cc-pvdz", "cpu", 1, -0.001)
    write_case(tmp_path, "benzene-cc-pvdz", "gpu", 1, -0.001, ok=False)
    assert mod.analyze(tmp_path) == {}


def test_grac_scf_reads_both_monomers(tmp_path):
    directory = write_case(tmp_path, "benzene-cc-pvdz", "cpu", 1, -0.001)
    monomers = mod.grac_scf(directory / "psi4.out")
    assert sorted(monomers) == ["A", "B"]
    assert monomers["A"]["cation"] == pytest.approx(-231.6426)
    assert monomers["A"]["homo"] == pytest.approx(-0.26368518)
