import json

import pytest
import thread_scaling as mod


def write_case(root, system, basis, threads, repeat, wall_s, ok=True, recorded_threads=None):
    directory = root / f"{system}-{basis}-cpu{threads}-{repeat}"
    directory.mkdir(parents=True)
    (directory / "result.json").write_text(json.dumps({
        "system": system, "basis": basis, "mode": "cpu",
        "threads": threads if recorded_threads is None else recorded_threads,
        "wall_s": wall_s, "ok": ok}))


def test_amdahl_recovers_a_known_serial_and_parallel_split():
    # T(n) = 10 + 80/n  ->  T(8) = 20, T(24) = 13.333...
    serial, parallel = mod.amdahl(8, 20.0, 24, 10.0 + 80.0 / 24)
    assert serial == pytest.approx(10.0)
    assert parallel == pytest.approx(80.0)


def test_amdahl_refuses_to_extrapolate_when_more_threads_did_not_help():
    assert mod.amdahl(8, 100.0, 24, 100.0) is None
    assert mod.amdahl(8, 100.0, 24, 120.0) is None


def test_amdahl_refuses_a_superlinear_pair_that_implies_negative_serial_time():
    # Better than linear (100 -> 20 on 3x the threads) cannot fit T = s + p/n.
    assert mod.amdahl(8, 100.0, 24, 20.0) is None


def test_measured_speedup_and_efficiency_use_medians(tmp_path):
    for repeat, wall in enumerate([100.0, 90.0, 110.0], start=1):
        write_case(tmp_path, "benzene", "cc-pvdz", 8, repeat, wall)
    for repeat, wall in enumerate([50.0, 60.0, 40.0], start=1):
        write_case(tmp_path, "benzene", "cc-pvdz", 24, repeat, wall)

    row = mod.analyze(tmp_path)["rows"][0]
    assert row["narrow_median_s"] == 100.0 and row["wide_median_s"] == 50.0
    assert row["measured_speedup"] == pytest.approx(2.0)
    # 2x speedup from 3x the threads.
    assert row["parallel_efficiency"] == pytest.approx(2.0 / 3.0)
    assert row["repeats"] == {"8": 3, "24": 3}


def test_projection_is_bounded_by_the_asymptote(tmp_path):
    write_case(tmp_path, "nanotube", "6-31+g**", 8, 1, 500.0)
    write_case(tmp_path, "nanotube", "6-31+g**", 24, 1, 257.0)

    row = mod.analyze(tmp_path, target_threads=56)["rows"][0]
    assert row["measured_speedup"] < row["projected_speedup_vs_narrow"]
    assert row["projected_speedup_vs_narrow"] < row["asymptotic_speedup_vs_narrow"]
    assert row["projected_target_s"] > row["amdahl_serial_s"]


def test_single_width_case_is_skipped_rather_than_projected(tmp_path):
    write_case(tmp_path, "water", "cc-pvdz", 24, 1, 9.0)
    assert mod.analyze(tmp_path)["rows"] == []


def test_failed_runs_are_excluded(tmp_path):
    write_case(tmp_path, "water", "cc-pvdz", 8, 1, 12.0)
    write_case(tmp_path, "water", "cc-pvdz", 24, 1, 10.0, ok=False)
    assert mod.analyze(tmp_path)["rows"] == []


def test_thread_count_disagreement_between_name_and_record_is_fatal(tmp_path):
    write_case(tmp_path, "water", "cc-pvdz", 24, 1, 9.0, recorded_threads=8)
    with pytest.raises(ValueError, match="directory says 24 threads"):
        mod.analyze(tmp_path)
