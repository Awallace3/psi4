"""Bootstrap only our evidence plugin, then pytest against the newly staged core."""
from pathlib import Path
import faulthandler
import os
import sys

plugin_dir, delimiter, *args = sys.argv[1:]
if delimiter != "--":
    raise RuntimeError("pytest argv delimiter")
# Keep a shutdown traceback even after pytest returns and native finalization hangs.
# Diagnostic only: no forced exit, timer reset, or changed pytest verdict.
faulthandler.enable()
faulthandler.dump_traceback_later(540, repeat=False, exit=False)
sys.path.insert(0, plugin_dir)
import pytest
import pytest_evidence
from probe import main as probe
old_argv = sys.argv
sys.argv = [__file__, os.environ["MAJOR2_PROCESS_PROBE"]]
probe()
sys.argv = old_argv
status = pytest.main(args, plugins=[pytest_evidence])
# Re-arm after pytest restores its own faulthandler configuration.
faulthandler.enable()
faulthandler.dump_traceback_later(30, repeat=False, exit=False)
raise SystemExit(status)
