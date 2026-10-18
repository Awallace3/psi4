"""Fresh process, unchanged committed runner; mappings taken AFTER its energy timer."""
from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from policy import require, write_json
from probe import main as probe, maps

if __name__ == "__main__":
    # probe output, post-run mappings output, runner path, runner argv...
    before, after, runner, *args = sys.argv[1:]
    sys.argv = [__file__, before]
    probe()  # import/interface only; no energy/JK/XC construction or warmup
    sys.argv = [runner, *args]
    try:
        runpy.run_path(runner, run_name="__main__")
    finally:
        write_json(after, maps())
