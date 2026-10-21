# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
from decimal import Decimal, localcontext
from pathlib import Path
import tempfile
import unittest
from grid import read_printed_grid
from rounding import e15_8_interval


class GridTests(unittest.TestCase):
    def test_order_tokens_and_print_intervals_are_retained(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"grid.dat"
            p.write_text("# Grid name toy\n# SITE 1 NAME A\n"
                         "0.10000000E+01 0.00000000E+00 -0.10000000D+01 0.20000000E+01\n")
            out = read_printed_grid(p)
        self.assertEqual(out.sites, (("A", 1),))
        self.assertEqual(out.tokens[0][2], "-0.10000000D+01")
        self.assertEqual(out.intervals[0][0], (Decimal(".999999995"), Decimal("1.00000005")))
        self.assertEqual(out.intervals[0][1], (Decimal(0), Decimal(0)))

    def test_strict_print_tokens_ignore_caller_decimal_context(self):
        with localcontext() as c:
            c.prec = 2
            self.assertEqual(e15_8_interval("-0.10000000E+01"),
                             (Decimal("-1.00000005"), Decimal("-.999999995")))
        for token in ("NaN", "1.0", "0.01000000E+01", "0.00000000E-99"):
            with self.subTest(token=token), self.assertRaises(ValueError):
                e15_8_interval(token)

    def test_unlabelled_empty_duplicate_and_malformed_blocks_refused(self):
        for text in ("0.10000000E+01 "*4, "# SITE 1 NAME A\n",
                     "# SITE 2 NAME A\n", "# SITE broken\n"):
            with tempfile.TemporaryDirectory() as d:
                p = Path(d)/"bad.dat"
                p.write_text(text)
                with self.subTest(text=text), self.assertRaises(ValueError):
                    read_printed_grid(p)


if __name__ == "__main__":
    unittest.main(verbosity=2)
