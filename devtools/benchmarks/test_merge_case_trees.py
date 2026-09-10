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
