"""Read-only pytest plugin: exact collection and all phase outcomes outside source."""
import json
import os
from pathlib import Path

items = []
phases = {}


def pytest_collection_finish(session):
    items.extend(item.nodeid for item in session.items)
    if len(items) != int(os.environ["MAJOR2_EXPECTED_COUNT"]):
        import pytest
        raise pytest.UsageError("required bounded test matrix changed; stop before chemistry")


def pytest_runtest_logreport(report):
    phases.setdefault(report.nodeid, {})[report.when] = report.outcome


def pytest_sessionfinish(session, exitstatus):
    p = Path(os.environ["MAJOR2_PYTEST_EVIDENCE"])
    with p.open("x") as f:
        json.dump(dict(collected=items, phases=phases, exitstatus=int(exitstatus)), f, indent=2)
