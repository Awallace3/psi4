"""Prove the projection attribution build changes no preexisting C++ code."""
from pathlib import Path
import re
import subprocess
import unittest


class TimerOnly(unittest.TestCase):
    def test_removing_instrumentation_recovers_exact_accepted_source(self):
        root = Path(__file__).resolve().parents[2]
        file = "psi4/src/psi4/libsapt_solver/fdds_disp.cc"
        accepted = subprocess.check_output(
            ["git", "-C", str(root), "show",
             "dcc8ac971310530688920efd0b2ba587a6e5cfd7:" + file], text=True)
        current = (root / file).read_text()
        current, classes = re.subn(
            r"// Coarse attribution only: outside OpenMP regions, no changed contraction\n"
            r".*?\n};\n\n", "", current, count=1, flags=re.S)
        self.assertEqual(classes, 1)
        current, starts = re.subn(r'^    DensityPhaseTimer (ao|metric|overlap)_phase\("FDDS density: [^"]+"\);\n',
                                 "", current, flags=re.M)
        current, stops = re.subn(r"^    (ao|metric|overlap)_phase\.stop\(\);\n",
                                "", current, flags=re.M)
        self.assertEqual((starts, stops), (3, 3))
        self.assertEqual(current, accepted)


if __name__ == "__main__":
    unittest.main()
