"""Bootstrap only our evidence plugin, then pytest against the newly staged core."""
from pathlib import Path
import os
import sys

plugin_dir, delimiter, *args = sys.argv[1:]
if delimiter != "--":
    raise RuntimeError("pytest argv delimiter")
sys.path.insert(0, plugin_dir)
import pytest
import pytest_evidence
from probe import main as probe
old_argv = sys.argv
sys.argv = [__file__, os.environ["MAJOR2_PROCESS_PROBE"]]
probe()
sys.argv = old_argv
raise SystemExit(pytest.main(args, plugins=[pytest_evidence]))
