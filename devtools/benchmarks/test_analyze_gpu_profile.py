"""Guard against duplicated timer-tree accounting in performance diagnosis."""
from pathlib import Path
import tempfile
import unittest

from analyze_gpu_profile import flat_timers


BLOCK = """********************************
Module User System Wall Calls
JK: JK : 4.000u 0.100s 1.250w 3 calls
RV: Form V : 5.000u 0.000s 2.500w 2 calls
--------------------------------
| JK: JK : 4.000u 0.100s 1.250w 3 calls
| RV: Form V : 5.000u 0.000s 2.500w 2 calls
********************************
"""


class TimerTests(unittest.TestCase):
    def test_flat_section_counted_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timer.dat"
            path.write_text(BLOCK)
            timers = flat_timers(path)
            self.assertEqual(timers["JK: JK"], {"wall_s": 1.25, "calls": 3})
            self.assertEqual(timers["RV: Form V"]["wall_s"], 2.5)
            self.assertEqual(len(timers), 2)

    def test_appended_runs_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timer.dat"
            path.write_text(BLOCK + BLOCK)
            with self.assertRaises(ValueError):
                flat_timers(path)


if __name__ == "__main__":
    unittest.main()
