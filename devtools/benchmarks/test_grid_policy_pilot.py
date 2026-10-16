from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grid_policy_pilot as policy


def test_policy_parameters_are_independent():
    assert policy.POLICIES["early1"] == (10., 1)
    assert policy.POLICIES["full2"] == (1., 2)
    assert policy.POLICIES["full1"] == (1., 1)
    assert policy.POLICIES["early2"] == (10., 2)


def test_each_policy_uses_identical_initial_and_final_grid():
    for base, config in policy.BASES.items():
        assert len({policy.pilot.CONFIGS[base+"_"+name] for name in policy.POLICIES}) == 1
        assert config[0] == (75, 302)
