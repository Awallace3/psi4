import json

import grac_cost as mod
import pytest

TIMER = """
Timers:

SAPT(DFT) Energy                    : 100.0u   1.0s   {total}w      1 calls
{grac_a}{grac_b}JK: JK                              :  10.0u   0.0s     5.0w     50 calls

Call tree:

| | SAPT(DFT):GRAC Shift Monomer A  :   1.0u   0.0s  9999.0w      1 calls
"""


def write_case(root, system, basis, mode, threads, repeat, wall_s,
               grac_a=None, grac_b=None, grac_compute="ITERATIVE", timer=True):
    directory = root / f"{system}-{basis}-{mode}-{repeat}"
    directory.mkdir(parents=True)
    (directory / "result.json").write_text(json.dumps({
        "system": system, "basis": basis, "mode": mode, "threads": threads,
        "wall_s": wall_s, "ok": True, "grac_compute": grac_compute}))
    if timer:
        line = lambda name, value: (
            "" if value is None else
            f"{name:<36}:   1.0u   0.0s   {value}w      1 calls\n")
        (directory / "timer.dat").write_text(TIMER.format(
            total=wall_s,
            grac_a=line("SAPT(DFT):GRAC Shift Monomer A", grac_a),
            grac_b=line("SAPT(DFT):GRAC Shift Monomer B", grac_b)))
    return directory


def test_grac_phase_is_reported_as_a_fraction_of_the_whole_calculation(tmp_path):
    write_case(tmp_path, "protein157", "6-31+g**", "cpu", 24, 1, 1000.0,
               grac_a=400.0, grac_b=20.0)
    row = mod.summarize([mod.case_cost(next(tmp_path.iterdir()))])[0]
    assert row["grac_a_s"] == 400.0 and row["grac_b_s"] == 20.0
    assert row["grac_s"] == 420.0
    assert row["grac_fraction"] == pytest.approx(0.42)


def test_the_two_monomers_stay_separate_because_one_can_dominate(tmp_path):
    # 1344 vs 442 basis functions: pooling these would hide which one costs.
    directory = write_case(tmp_path, "protein157", "6-31+g**", "cpu", 24, 1, 1000.0,
                           grac_a=400.0, grac_b=20.0)
    record = mod.case_cost(directory)
    assert record["grac_a_s"] > 10 * record["grac_b_s"]


def test_iterative_without_a_grac_timer_is_an_error_not_zero_percent(tmp_path):
    directory = write_case(tmp_path, "water", "cc-pvdz", "cpu", 8, 1, 10.0)
    with pytest.raises(ValueError, match="no GRAC phase timer"):
        mod.case_cost(directory)


def test_a_fixed_shift_run_that_somehow_timed_grac_is_an_error(tmp_path):
    directory = write_case(tmp_path, "water", "cc-pvdz", "cpu", 8, 1, 10.0,
                           grac_a=1.0, grac_compute="NONE")
    with pytest.raises(ValueError, match="GRAC phase timer is present"):
        mod.case_cost(directory)


def test_a_fixed_shift_run_costs_no_grac_time(tmp_path):
    directory = write_case(tmp_path, "water", "cc-pvdz", "cpu", 8, 1, 10.0,
                           grac_compute="NONE")
    assert mod.case_cost(directory)["grac_fraction"] == 0.0


def test_the_call_tree_copy_of_the_phase_timer_is_ignored(tmp_path):
    # The tree section repeats the name at 9999w; taking it would exceed the total.
    directory = write_case(tmp_path, "peptide", "6-31+g**", "cpu", 8, 1, 100.0,
                           grac_a=30.0, grac_b=10.0)
    assert mod.case_cost(directory)["grac_s"] == 40.0


def test_repeats_of_one_arm_collapse_to_a_median(tmp_path):
    for repeat, (wall, a) in enumerate([(100.0, 40.0), (110.0, 50.0), (90.0, 30.0)], start=1):
        write_case(tmp_path, "benzene", "cc-pvdz", "cpu", 8, repeat, wall, grac_a=a, grac_b=1.0)
    rows = mod.summarize([mod.case_cost(d) for d in sorted(tmp_path.iterdir())])
    assert len(rows) == 1 and rows[0]["repeats"] == 3
    assert rows[0]["total_wall_s"] == 100.0 and rows[0]["grac_a_s"] == 40.0


def test_grac_speedup_is_reported_beside_the_whole_calculation_speedup(tmp_path):
    write_case(tmp_path, "benzene", "cc-pvdz", "cpu", 8, 1, 400.0, grac_a=200.0, grac_b=0.0)
    write_case(tmp_path, "benzene", "cc-pvdz", "gpu", 8, 1, 100.0, grac_a=50.0, grac_b=0.0)
    rows = mod.summarize([mod.case_cost(d) for d in sorted(tmp_path.iterdir())])
    paired = mod.speedups(rows)
    assert len(paired) == 1
    assert paired[0]["grac_speedup"] == pytest.approx(4.0)
    assert paired[0]["total_speedup"] == pytest.approx(4.0)


def test_arms_at_different_widths_are_not_paired_into_a_speedup(tmp_path):
    # A 24-thread CPU against an 8-thread GPU is a baseline-width ratio, not an
    # accelerator speedup, so it must not appear in the paired table at all.
    write_case(tmp_path, "protein157", "6-31+g**", "cpu", 24, 1, 8000.0, grac_a=3400.0, grac_b=110.0)
    write_case(tmp_path, "protein157", "6-31+g**", "gpu", 8, 1, 900.0, grac_a=300.0, grac_b=20.0)
    rows = mod.summarize([mod.case_cost(d) for d in sorted(tmp_path.iterdir())])
    assert len(rows) == 2
    assert mod.speedups(rows) == []


def test_a_lone_cpu_arm_still_reports_its_cost(tmp_path):
    write_case(tmp_path, "protein157", "6-31+g**", "cpu", 24, 1, 8000.0, grac_a=3400.0, grac_b=110.0)
    rows = mod.summarize([mod.case_cost(d) for d in sorted(tmp_path.iterdir())])
    payload = {"cases": [], "summary": rows, "paired": mod.speedups(rows)}
    text = mod.markdown(payload)
    assert "protein157" in text and "43.9%" in text


def test_a_case_that_failed_to_measure_is_named_rather_than_dropped(tmp_path):
    payload = {"cases": [{"case": "nanotube-cpu-2", "error": "ValueError: no GRAC phase timer"}],
               "summary": [], "paired": []}
    assert "nanotube-cpu-2" in mod.markdown(payload)
