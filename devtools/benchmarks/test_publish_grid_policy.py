from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from publish_grid_policy import validate_policy


@pytest.fixture
def row():
    return dict(ok=True, arm="tiny10_full2",
        grid_policy=dict(name="full2", switch_factor=1., minimum_final_iterations=2),
        options=dict(DFT_GRID_SWITCH_FACTOR=1., DFT_RADIAL_POINTS_INITIAL=20),
        iterator_overlay_sha256="checked-source",
        per_scf_grid_iteration_proofs=[dict(converged=True, minimum_final_iterations=2,
                                           initial=10, final=2)],
        initial_iterations=10, final_iterations=2)


def test_full_threshold_minimum_two_accepted(row):
    validate_policy(row, "checked-source")


def test_converged_single_final_iteration_rejected(row):
    row["per_scf_grid_iteration_proofs"][0]["final"] = 1
    row["final_iterations"] = 1
    with pytest.raises(AssertionError):
        validate_policy(row, "checked-source")


def test_wrong_threshold_rejected(row):
    row["options"]["DFT_GRID_SWITCH_FACTOR"] = 10.
    with pytest.raises(AssertionError):
        validate_policy(row, "checked-source")


def test_wrong_driver_overlay_rejected(row):
    with pytest.raises(AssertionError):
        validate_policy(row, "other-source")


def test_missing_proofs_rejected(row):
    row["per_scf_grid_iteration_proofs"] = []
    with pytest.raises(AssertionError, match="No per-SCF"):
        validate_policy(row, "checked-source")


def test_retry_counts_are_retained_without_requiring_failed_attempt_to_converge(row):
    row["per_scf_grid_iteration_proofs"].append(dict(
        converged=False, minimum_final_iterations=2, initial=20, final=0))
    row["initial_iterations"] += 20
    validate_policy(row, "checked-source")


def test_counts_must_match_every_printed_grid_iteration(row):
    row["final_iterations"] += 1
    with pytest.raises(AssertionError):
        validate_policy(row, "checked-source")
