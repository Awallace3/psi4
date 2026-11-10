import importlib.util
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("protein157_grids", HERE / "protein157_grids.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def row(name, persistent, temporary, free=140):
    return dict(ok=True, basis=name, query=dict(
        df_persistent_bytes=int(persistent * 2**30),
        df_construction_temporary_bytes=int(temporary * 2**30),
        device_free_bytes=int(free * 2**30), df_plan_allocated=False))


def test_select_largest_that_fits_with_headroom():
    choices = [row("small", 43, 27), row("medium", 73, 45), row("too-large", 101, 62)]
    assert probe.select_candidate(choices)["basis"] == "medium"


def test_persistent_fit_is_not_sufficient():
    assert probe.select_candidate([row("too-large", 101, 62)]) is None


def test_query_must_not_allocate_df_plan():
    r = row("bad", 43, 27)
    r["query"]["df_plan_allocated"] = True
    with pytest.raises(ValueError, match="allocated"):
        probe.select_candidate([r])


def test_failed_queries_never_selected():
    assert probe.select_candidate([dict(ok=False, basis="failed")]) is None


def test_geometry_is_original_protein157():
    assert probe.geometry().count("\n") > 157
    assert "direct" not in probe.ARMS


def test_source_query_never_creates_df_plan():
    source = (HERE / "cuest_df_workspace_query.cc").read_text()
    assert "cuestDFIntPlanCreateWorkspaceQuery(" in source
    assert "cuestDFIntPlanCreate(" not in source
