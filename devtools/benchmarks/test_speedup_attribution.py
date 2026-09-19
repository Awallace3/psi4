import json

import pytest
import speedup_attribution as mod

TIMER = """
  Module           User     System    Wall    Calls

  JK: JK        : {jk}u {jk}s {jk}w {calls} calls
  RV: Form V    : {rv}u {rv}s {rv}w {calls} calls
  UV: Form V    : {uv}u {uv}s {uv}w {calls} calls
--------------------------------------------------
"""


def write_case(root, name, total, jk, rv, uv, ok=True):
    directory = root / name
    directory.mkdir(parents=True)
    mode = "gpu" if "-gpu-" in name else "cpu"
    (directory / "result.json").write_text(json.dumps(
        {"wall_s": total, "ok": ok, "mode": mode, "threads": 8}))
    (directory / "timer.dat").write_text(
        TIMER.format(jk=jk, rv=rv, uv=uv, calls=10))
    return directory


def test_saving_shares_sum_to_one_and_match_the_measured_speedup(tmp_path):
    # CPU 1000 = 200 jk + 600 xc + 200 other; GPU 100 = 5 jk + 55 xc + 40 other.
    write_case(tmp_path, "nanotube-cpu-1", 1000.0, 200.0, 400.0, 200.0)
    write_case(tmp_path, "nanotube-gpu-1", 100.0, 5.0, 30.0, 25.0)
    row = mod.analyze(tmp_path)["nanotube"]

    assert row["speedup"] == pytest.approx(10.0)
    shares = [row[f"{p}_share_of_saving"] for p in ("jk", "xc", "other")]
    assert sum(shares) == pytest.approx(1.0)
    assert row["jk_share_of_saving"] == pytest.approx(195.0 / 900.0)
    assert row["xc_share_of_saving"] == pytest.approx(545.0 / 900.0)
    assert row["other_share_of_saving"] == pytest.approx(160.0 / 900.0)


def test_dfk_alone_cannot_explain_a_large_end_to_end_speedup(tmp_path):
    # DF-K is 20% of CPU time, so removing it entirely gives at most 1.25x,
    # even though the measured end-to-end speedup is 10x.
    write_case(tmp_path, "nanotube-cpu-1", 1000.0, 200.0, 400.0, 200.0)
    write_case(tmp_path, "nanotube-gpu-1", 100.0, 5.0, 30.0, 25.0)
    row = mod.analyze(tmp_path)["nanotube"]

    assert row["amdahl_bound_from_dfk_alone"] == pytest.approx(1.25)
    assert row["amdahl_bound_from_dfk_alone"] < row["speedup"]
    assert row["dfk_share_of_cpu_time"] == pytest.approx(0.2)


def test_components_exceeding_total_wall_are_rejected(tmp_path):
    write_case(tmp_path, "bogus-cpu-1", 100.0, 60.0, 40.0, 30.0)
    with pytest.raises(ValueError, match="not disjoint"):
        mod.analyze(tmp_path)


def test_medians_are_used_across_repeats(tmp_path):
    for repeat, total in enumerate([1000.0, 1200.0, 800.0], start=1):
        write_case(tmp_path, f"benzene-cpu-{repeat}", total, 200.0, 400.0, 200.0)
    for repeat, total in enumerate([100.0, 90.0, 110.0], start=1):
        write_case(tmp_path, f"benzene-gpu-{repeat}", total, 5.0, 30.0, 25.0)
    row = mod.analyze(tmp_path)["benzene"]

    assert row["cpu"]["total_s"] == 1000.0 and row["gpu"]["total_s"] == 100.0
    assert row["repeats"] == {"cpu": 3, "gpu": 3}


def test_unpaired_and_failed_cases_are_dropped(tmp_path):
    write_case(tmp_path, "lonely-cpu-1", 100.0, 10.0, 10.0, 10.0)
    write_case(tmp_path, "failed-cpu-1", 100.0, 10.0, 10.0, 10.0)
    write_case(tmp_path, "failed-gpu-1", 10.0, 1.0, 1.0, 1.0, ok=False)
    assert mod.analyze(tmp_path) == {}
