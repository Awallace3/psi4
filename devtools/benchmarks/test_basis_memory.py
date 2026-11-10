import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("basis_memory", Path(__file__).with_name("basis_memory.py"))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_scan_separates_primary_and_auxiliary_dimensions():
    control = probe.CONFIGS["control922"]
    for key, value in probe.CONFIGS.items():
        if key.startswith("primary"):
            assert value[1] == control[1] and value[3] == control[3]
        elif key.startswith("aux"):
            assert value[0] == control[0] and value[2] == control[2]
    assert max(probe.structural_plan_gib(v[2], v[3]) for v in probe.CONFIGS.values()) < 45


def test_workspace_parser_does_not_call_free_memory_an_allocation():
    text = """ Device memory, AO pair list:  0.002 [GiB]
    Device memory, DF int plan:  10.111 [GiB]
    Device memory, free/total:  88.000 / 141.000 [GiB]
    Device memory, J/K scratch:  2.345 [GiB]
    """
    assert probe.queried_workspaces(text) == [
        dict(allocation="AO pair list", queried_gib_rounded=0.002),
        dict(allocation="DF int plan", queried_gib_rounded=10.111),
        dict(allocation="J/K scratch", queried_gib_rounded=2.345),
    ]


def test_empty_workspace_log_is_not_zero_allocations():
    assert probe.queried_workspaces("unavailable") == []
