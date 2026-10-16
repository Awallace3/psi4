import importlib.util
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tiny_grid_pilot as pilot


def row(arm, wall=10., shift=0., ok=True):
    return dict(system="water", arm=arm, ok=ok, wall_s=wall,
                components={k: shift for k in pilot.grid.COMPONENTS},
                initial_iterations=8, final_iterations=3, all_iterations=11,
                xc_s=3., scf_s=8., switch_s=.1, error="Not converged")


def test_initial_ten_percent_and_default_final():
    final, initial = pilot.CONFIGS["tiny10"]
    assert final == (75, 302)
    assert .09 < initial[0]*initial[1]/(final[0]*final[1]) < .11
    assert all(final == (75, 302) for key, (final, _) in pilot.CONFIGS.items() if key != "fixed99")


def test_stability_and_speed_are_distinct():
    values = [row("fixed75", 10), row("fixed99", 20),
              row("tiny10", 12), row("coarse75", 7)]
    group = pilot.summarize(values)[0]
    assert group["smallest_stable"] == "tiny10"
    assert group["fastest_stable"] == "coarse75"


def test_failed_tiny_arm_not_ranked_or_dropped():
    group = pilot.summarize([row("fixed75"), row("fixed99"), row("tiny10", ok=False),
                              row("coarse75", 7)])[0]
    assert len(group["rows"]) == 4
    assert group["smallest_stable"] == "coarse75"


def test_switching_accuracy_and_final_grid_accuracy_both_gate():
    group = pilot.summarize([row("fixed75", shift=2.e-5), row("fixed99"),
                              row("tiny10", shift=2.e-5)])[0]
    assert group["fastest_stable"] is None
    assert group["rows"][-1]["switching_error_eh"] == 0
    assert group["rows"][-1]["reference_error_eh"] > 1.e-5


def test_no_ranking_until_both_controls_complete():
    group = pilot.summarize([row("fixed75"), row("tiny10")])[0]
    assert group["smallest_stable"] is None
