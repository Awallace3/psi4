import pytest
import merge_case_trees as mod


def write(root, tree, name, result=True):
    directory = root / tree / name
    directory.mkdir(parents=True)
    (directory / "psi4.out").write_text("out")
    if result:
        (directory / "result.json").write_text("{}")
    return directory


def test_cases_from_several_trees_are_gathered_under_one_name(tmp_path):
    write(tmp_path, "A", "water-cpu-1")
    write(tmp_path, "B", "water-cpu-2")
    chosen = mod.resolve([tmp_path / "A", tmp_path / "B"])
    assert sorted(chosen) == ["water-cpu-1", "water-cpu-2"]


def test_a_completed_rerun_beats_the_stub_the_killed_run_left(tmp_path):
    write(tmp_path, "A", "nanotube-cpu-2", result=False)
    rerun = write(tmp_path, "B", "nanotube-cpu-2")
    assert mod.resolve([tmp_path / "A", tmp_path / "B"])["nanotube-cpu-2"] == rerun


def test_the_stub_loses_regardless_of_tree_order(tmp_path):
    good = write(tmp_path, "A", "nanotube-cpu-2")
    write(tmp_path, "B", "nanotube-cpu-2", result=False)
    assert mod.resolve([tmp_path / "A", tmp_path / "B"])["nanotube-cpu-2"] == good


def test_two_completed_copies_of_a_case_is_an_error_not_a_silent_choice(tmp_path):
    write(tmp_path, "A", "water-cpu-1")
    write(tmp_path, "B", "water-cpu-1")
    with pytest.raises(SystemExit, match="completed in both"):
        mod.resolve([tmp_path / "A", tmp_path / "B"])


def test_links_point_at_the_job_tree_rather_than_copying_it(tmp_path):
    source = write(tmp_path, "A", "water-cpu-1")
    merged = mod.link(mod.resolve([tmp_path / "A"]), tmp_path / "merged")
    entry = merged / "water-cpu-1"
    assert entry.is_symlink() and entry.resolve() == source.resolve()


def test_rerunning_the_merge_replaces_stale_links(tmp_path):
    write(tmp_path, "A", "water-cpu-1")
    mod.link(mod.resolve([tmp_path / "A"]), tmp_path / "merged")
    write(tmp_path, "B", "water-cpu-2")
    merged = mod.link(mod.resolve([tmp_path / "B"]), tmp_path / "merged")
    assert [p.name for p in merged.iterdir()] == ["water-cpu-2"]


def test_a_directory_holding_real_case_trees_is_not_overwritten(tmp_path):
    write(tmp_path, "notmerged", "water-cpu-1")
    write(tmp_path, "A", "water-cpu-1")
    with pytest.raises(SystemExit, match="not a merged tree"):
        mod.link(mod.resolve([tmp_path / "A"]), tmp_path / "notmerged")


def test_generated_manifests_in_the_merged_tree_survive_a_rerun(tmp_path):
    # case_dirs_to_campaign.py writes campaign.json here and overwrites it itself.
    write(tmp_path, "A", "water-cpu-1")
    merged = mod.link(mod.resolve([tmp_path / "A"]), tmp_path / "merged")
    (merged / "campaign.json").write_text("{}")
    mod.link(mod.resolve([tmp_path / "A"]), merged)
    assert (merged / "campaign.json").exists()


def canary(tree, dgemm=75.0):
    import json
    metadata = tree / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / "canary-start-t8.json").write_text(json.dumps({
        "node": "n", "threads": 8, "dgemm_per_core_gflops": dgemm,
        "stream": {"gb_s": 9.2}, "scalar": {"miter_s": 24.6}}))
    return tree


def test_trees_from_hosts_that_ran_at_different_speeds_are_not_pooled(tmp_path):
    write(tmp_path, "A/results", "water-cpu-1")
    write(tmp_path, "B/results", "water-cpu-2")
    canary(tmp_path / "A", dgemm=75.0)
    canary(tmp_path / "B", dgemm=23.0)
    with pytest.raises(SystemExit, match="mismatched"):
        mod.check_hosts([tmp_path / "A/results", tmp_path / "B/results"], allow=None)


def test_trees_with_no_canary_are_refused_rather_than_assumed_equal(tmp_path):
    write(tmp_path, "A/results", "water-cpu-1")
    write(tmp_path, "B/results", "water-cpu-2")
    with pytest.raises(SystemExit, match="uncertified"):
        mod.check_hosts([tmp_path / "A/results", tmp_path / "B/results"], allow=None)


def test_matched_hosts_pool_without_a_flag(tmp_path):
    write(tmp_path, "A/results", "water-cpu-1")
    write(tmp_path, "B/results", "water-cpu-2")
    canary(tmp_path / "A", dgemm=75.0)
    canary(tmp_path / "B", dgemm=74.0)
    payload = mod.check_hosts([tmp_path / "A/results", tmp_path / "B/results"], allow=None)
    assert payload["verdict"] == "matched"


def test_an_override_records_its_reason_inside_the_merged_tree(tmp_path, monkeypatch, capsys):
    write(tmp_path, "A/results", "water-cpu-1")
    write(tmp_path, "B/results", "water-cpu-2")
    target = tmp_path / "merged"
    monkeypatch.setattr("sys.argv", [
        "merge", str(tmp_path / "A/results"), str(tmp_path / "B/results"),
        "--output", str(target), "--allow-host-mismatch", "pre-canary trees"])
    assert mod.main() == 0
    note = (target / mod.MISMATCH_NOTE).read_text()
    assert "pre-canary trees" in note and "uncertified" in note


def test_a_stale_mismatch_note_does_not_survive_a_clean_remerge(tmp_path):
    target = tmp_path / "merged"
    target.mkdir()
    (target / mod.MISMATCH_NOTE).write_text("old caveat")
    good = write(tmp_path, "A", "water-cpu-1")
    mod.link({"water-cpu-1": good}, target)
    assert not (target / mod.MISMATCH_NOTE).exists()
