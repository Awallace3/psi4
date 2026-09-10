import json

import pytest

import host_speed as mod


def canary(tree, phase="start", threads=8, dgemm=75.0, triad=9.2, scalar=24.6,
           mhz=2800.0, node="atl1-1-02-014-9-0"):
    metadata = tree / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / f"canary-{phase}-t{threads}.json").write_text(json.dumps({
        "node": node, "job": "1", "threads": threads,
        "dgemm_per_core_gflops": dgemm,
        "stream": {"gb_s": triad}, "scalar": {"miter_s": scalar},
        "cpu": {"live_mhz_mine": [mhz] * threads},
    }))
    (tree / "results").mkdir(exist_ok=True)
    return tree


def test_a_tree_reports_the_throughput_its_own_allocation_measured(tmp_path):
    canary(tmp_path / "A")
    row = mod.profile(tmp_path / "A")
    assert row["status"] == "measured"
    assert row["dgemm_gflops_per_core"] == pytest.approx(75.0)
    assert row["live_mhz"] == pytest.approx(2800.0)


def test_the_results_directory_finds_the_metadata_beside_it(tmp_path):
    canary(tmp_path / "A")
    assert mod.profile(tmp_path / "A" / "results")["status"] == "measured"


def test_a_tree_without_a_canary_is_uncertified_rather_than_agreeing(tmp_path):
    (tmp_path / "A" / "results").mkdir(parents=True)
    assert mod.profile(tmp_path / "A")["status"] == mod.UNCERTIFIED


def test_hosts_within_tolerance_are_poolable(tmp_path):
    canary(tmp_path / "A", dgemm=75.0)
    canary(tmp_path / "B", dgemm=73.0)
    assert mod.compare([tmp_path / "A", tmp_path / "B"])["verdict"] == "matched"


def test_the_three_times_deficit_that_motivated_this_is_caught(tmp_path):
    canary(tmp_path / "fast", dgemm=75.0, scalar=24.6, mhz=2800.0)
    canary(tmp_path / "slow", dgemm=23.0, scalar=7.7, mhz=1900.0,
           node="atl1-1-02-012-23-0")
    payload = mod.compare([tmp_path / "fast", tmp_path / "slow"])
    assert payload["verdict"] == "mismatched"
    assert payload["worst_ratio"] == pytest.approx(75.0 / 23.0, rel=1e-3)


def test_an_uncertified_tree_is_not_reported_as_matched_by_its_neighbour(tmp_path):
    canary(tmp_path / "A")
    (tmp_path / "B" / "results").mkdir(parents=True)
    payload = mod.compare([tmp_path / "A", tmp_path / "B"])
    assert payload["verdict"] == mod.UNCERTIFIED
    assert payload["uncertified"] == [str(tmp_path / "B")]


def test_a_host_that_changed_speed_mid_run_is_flagged_not_averaged(tmp_path):
    canary(tmp_path / "A", phase="start", dgemm=75.0)
    canary(tmp_path / "A", phase="end", dgemm=25.0)
    row = mod.profile(tmp_path / "A")
    assert row["drifted_during_run"]["dgemm_gflops_per_core"] == pytest.approx(3.0)


def test_a_steady_host_is_not_flagged_for_drift(tmp_path):
    canary(tmp_path / "A", phase="start", dgemm=75.0)
    canary(tmp_path / "A", phase="end", dgemm=74.0)
    assert "drifted_during_run" not in mod.profile(tmp_path / "A")


def test_the_probe_matching_the_campaign_width_is_the_one_compared(tmp_path):
    canary(tmp_path / "A", threads=1, dgemm=99.0)
    canary(tmp_path / "A", threads=8, dgemm=75.0)
    row = mod.profile(tmp_path / "A")
    assert row["threads"] == 8 and row["dgemm_gflops_per_core"] == pytest.approx(75.0)


def test_a_truncated_canary_is_skipped_rather_than_crashing_the_report(tmp_path):
    tree = canary(tmp_path / "A")
    (tree / "metadata" / "canary-end-t8.json").write_text("{trunc")
    assert mod.profile(tree)["status"] == "measured"


def test_metrics_missing_from_one_tree_do_not_silently_pass_the_others(tmp_path):
    canary(tmp_path / "A", dgemm=75.0)
    tree = canary(tmp_path / "B", dgemm=75.0)
    record = json.loads((tree / "metadata" / "canary-start-t8.json").read_text())
    record["scalar"] = {}
    (tree / "metadata" / "canary-start-t8.json").write_text(json.dumps(record))
    payload = mod.compare([tmp_path / "A", tmp_path / "B"])
    assert "scalar_miter_s" not in payload["spread"]
    assert payload["verdict"] == "matched"


def test_require_match_exits_nonzero_on_a_degraded_host(tmp_path, monkeypatch, capsys):
    canary(tmp_path / "fast", dgemm=75.0)
    canary(tmp_path / "slow", dgemm=23.0)
    monkeypatch.setattr("sys.argv", ["host_speed", str(tmp_path / "fast"),
                                     str(tmp_path / "slow"), "--require-match"])
    with pytest.raises(SystemExit, match="mismatched"):
        mod.main()


def test_the_table_names_uncertified_trees_instead_of_dropping_them(tmp_path):
    canary(tmp_path / "A")
    (tmp_path / "B" / "results").mkdir(parents=True)
    text = mod.markdown(mod.compare([tmp_path / "A", tmp_path / "B"]))
    assert "Uncertified" in text and "`B`" in text


def test_a_tree_is_named_by_its_job_directory_not_the_results_leaf(tmp_path):
    canary(tmp_path / "A-core6-h200-job13060539")
    assert mod.label(tmp_path / "A-core6-h200-job13060539" / "results") \
        == "A-core6-h200-job13060539"


def test_two_uncertified_trees_are_distinguishable_in_the_table(tmp_path):
    for name in ("A-job1", "A2-job2"):
        (tmp_path / name / "results").mkdir(parents=True)
    text = mod.markdown(mod.compare([tmp_path / "A-job1" / "results",
                                     tmp_path / "A2-job2" / "results"]))
    assert "`A-job1`" in text and "`A2-job2`" in text
