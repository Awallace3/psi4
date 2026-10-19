"""Compile the EXACT extracted helper on host; no Psi4/CUDA/math proof."""
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
FILE = "psi4/src/psi4/libfock/cuESTJK.cc"
OLD = "c9c266e5a383c962f9a4c8449dfbc54aeafaf6f2"
NEW = "21a7b4d9c922f293d6220cafd8143d70f7d85e18"
PATTERN = r"bool initial_shell_coordinate_matches\(double actual, double expected\) \{.*?\n\}"
MAIN = r"""
#include <cmath>
#include <limits>
HELPER
int main() {
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    // ACTUAL captured negative pair, and its positive counterpart (hex is exact).
    for (double sign : {-1.0, 1.0}) {
        double a = sign * 0x1.8b568ade94da3p-55;
        double b = sign * 0x1.8b568ade94da2p-55;
        if (!initial_shell_coordinate_matches(a, b)) return 1;
        if (!initial_shell_coordinate_matches(b, a)) return 2;
        double c = std::nextafter(b, 0.0);  // TWO ULP from a
        if (initial_shell_coordinate_matches(a, c)) return 3;
        if (initial_shell_coordinate_matches(c, a)) return 4;
    }
    for (double a : {-1.0, 0.0, 1.0}) {
        if (!initial_shell_coordinate_matches(a, a)) return 5;
        for (double dir : {-inf, inf}) {
            double b = std::nextafter(a, dir);
            if (!initial_shell_coordinate_matches(a, b)) return 6;
            if (!initial_shell_coordinate_matches(b, a)) return 7;
            double c = std::nextafter(b, dir);
            if (initial_shell_coordinate_matches(a, c)) return 8;
        }
        if (initial_shell_coordinate_matches(a, a+0.125)) return 9;
    }
    if (!initial_shell_coordinate_matches(0.0, -0.0) ||
        !initial_shell_coordinate_matches(-0.0, 0.0)) return 10;
    for (double bad : {nan, inf, -inf}) {
        if (initial_shell_coordinate_matches(bad, bad) ||
            initial_shell_coordinate_matches(bad, 0.0) ||
            initial_shell_coordinate_matches(0.0, bad)) return 11;
    }
    double max = std::numeric_limits<double>::max();
    if (initial_shell_coordinate_matches(max, inf) ||
        initial_shell_coordinate_matches(inf, max)) return 12;
    return 0;
}
""".replace("#include <limits>", "#include <limits>\n#include <initializer_list>")


def source(pin):
    return subprocess.check_output(["git", "-C", str(ROOT), "show", pin+":"+FILE], text=True)


class NativeConstructorTests(unittest.TestCase):
    def test_actual_extracted_cpp_red_green(self):
        text = source(NEW)
        helper, = re.findall(PATTERN, text, re.S)
        self.assertIn("std::nextafter(actual, expected) == expected", helper)
        old = source(OLD)
        self.assertIn("basis->shell(s).coord(k) != xyz.get(basis->shell_to_center(s), k)", old)
        # Red uses the exact old C++ comparison, not a Python floating mock.
        red = "bool initial_shell_coordinate_matches(double actual, double expected) { return !(actual != expected); }"
        with tempfile.TemporaryDirectory(dir=os.environ["MAJOR2_TEST_TMP"]) as td:
            for label, body, expected in [("red", red, 1), ("green", helper, 0)]:
                cpp, exe = Path(td)/(label+".cc"), Path(td)/label
                cpp.write_text(MAIN.replace("HELPER", body))
                proc = subprocess.run(["c++", "-std=c++17", "-O0", "-Wall", "-Wextra",
                                       str(cpp), "-o", str(exe)], capture_output=True, text=True, timeout=30)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertEqual(subprocess.run([str(exe)], timeout=10).returncode, expected, label)

    def test_only_initial_guard_changed_exact_later_keys_preserved(self):
        old, new = source(OLD), source(NEW)
        start, end = old.index("void key_double("), old.index("void basis_key(")
        self.assertIn(old[start:end], new)
        start = old.index("std::string cuESTJK::current_plan_key() const")
        self.assertEqual(old[start:], new[new.index("std::string cuESTJK::current_plan_key() const"):])
        self.assertEqual(new.count("initial_shell_coordinate_matches("), 2)
