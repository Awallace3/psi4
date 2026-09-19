import json

import case_dirs_to_campaign as mod
import pytest


def write_case(root, name, ok=True, wall_s=1.5, system="water", basis="cc-pvdz"):
    directory = root / name
    directory.mkdir(parents=True)
    record = {"system": system, "basis": basis, "ok": ok, "wall_s": wall_s}
    if not ok:
        record["error"] = "RuntimeError: boom"
    (directory / "result.json").write_text(json.dumps(record))
    return directory


def test_requested_but_absent_case_is_a_failure_not_a_silent_drop(tmp_path):
    write_case(tmp_path, "water-cc-pvdz-cpu-1")
    write_case(tmp_path, "water-cc-pvdz-gpu-1")
    manifest, complete, missing = mod.build(tmp_path, [("water", "cc-pvdz", 2)])

    assert len(manifest["records"]) == 4
    assert missing == ["water-cc-pvdz-cpu-2", "water-cc-pvdz-gpu-2"]
    assert complete == {"ok": False, "count": 4}


def test_run_reporting_not_ok_gets_a_nonzero_returncode(tmp_path):
    write_case(tmp_path, "water-cc-pvdz-cpu-1")
    write_case(tmp_path, "water-cc-pvdz-gpu-1", ok=False)
    manifest, complete, _ = mod.build(tmp_path, [("water", "cc-pvdz", 1)])

    codes = {r["name"]: r["returncode"] for r in manifest["records"]}
    assert codes == {"water-cc-pvdz-cpu-1": 0, "water-cc-pvdz-gpu-1": 1}
    assert complete["ok"] is False
    failed = next(r for r in manifest["records"] if r["returncode"])
    assert failed["error"] == "RuntimeError: boom"


def test_complete_campaign_matches_the_summarizer_expected_count(tmp_path):
    for basis in ("cc-pvdz", "aug-cc-pvdz"):
        for repeat in (1, 2):
            for mode in ("cpu", "gpu"):
                write_case(tmp_path, f"water-{basis}-{mode}-{repeat}", basis=basis)
    manifest, complete, missing = mod.build(
        tmp_path, [("water", "cc-pvdz", 2), ("water", "aug-cc-pvdz", 2)])

    assert not missing and complete == {"ok": True, "count": 8}
    # The summarizer recomputes this independently and compares it to the record count.
    expected = sum(len(bases) for _, bases in manifest["cases"]) * 2 * manifest["repeats"]
    assert expected == len(manifest["records"]) == 8


def test_mixed_repeat_counts_are_rejected_rather_than_silently_flattened(tmp_path):
    write_case(tmp_path, "water-cc-pvdz-cpu-1")
    with pytest.raises(ValueError, match="one repeat count"):
        mod.build(tmp_path, [("water", "cc-pvdz", 1), ("benzene", "cc-pvdz", 3)])


def test_discover_ignores_unpaired_thread_scaling_directories(tmp_path):
    write_case(tmp_path, "water-cc-pvdz-cpu-1")
    write_case(tmp_path, "water-cc-pvdz-gpu-1")
    scaling = tmp_path / "water-cc-pvdz-cpu24-1"
    scaling.mkdir()
    (scaling / "result.json").write_text(json.dumps({"system": "water", "basis": "cc-pvdz", "ok": True}))

    assert mod.discover(tmp_path) == {("water", "cc-pvdz"): 1}


def test_expect_parsing_rejects_malformed_specs():
    assert mod.parse_expect("benzene:aug-cc-pvdz:3") == ("benzene", "aug-cc-pvdz", 3)
    for bad in ("benzene:aug-cc-pvdz", "benzene::3", "benzene:cc-pvdz:x", "benzene:cc-pvdz:0"):
        with pytest.raises(Exception):
            mod.parse_expect(bad)
