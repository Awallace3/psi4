"""Guard the DF-K flop model against silent dimension and spin-factor drift."""
from pathlib import Path
import tempfile
import unittest

from dfk_effective_tflops import case_rate, fill_aux, flops, scf_blocks

RESTRICTED = """
  Nalpha       = 2
  Nbeta        = 2

  ==> Primary Basis <==

    Number of basis functions: 10

   => Auxiliary Basis Set <=

    Number of basis functions: 40

  ==> Iterations <==

    cuESTJK compute_JK: total=  2.00ms | alloc= 0.00ms ws= 0.00ms J=  1.00ms K=  4.00ms memcpy(H2D)= 0.00ms memcpy(D2H)= 0.00ms transpose= 0.00ms free= 0.00ms
   @DF-RKS iter   1:  -1.0   -1.0e+00   1.0e-03    0.10s
"""

UNRESTRICTED = """
  Nalpha       = 2
  Nbeta        = 1

  ==> Primary Basis <==

    Number of basis functions: 10

   => Auxiliary Basis Set <=

    Number of basis functions: 40

  ==> Iterations <==

   @DF-UKS iter   1:  -1.0   -1.0e+00   1.0e-03    0.10s
   @DF-UKS iter   2:  -1.0   -1.0e-06   1.0e-06    0.10s
"""

TIMER = """********************************
Module User System Wall Calls
JK: JK : 1.000u 0.000s 0.500w 6 calls
--------------------------------
"""


class FlopModelTests(unittest.TestCase):
    def blocks(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "psi4.out"
            path.write_text(text)
            return scf_blocks(path)

    def test_restricted_uses_alpha_occupation_once(self):
        block, = self.blocks(RESTRICTED)
        self.assertEqual((block["nbf"], block["naux"], block["jk_calls"]), (10, 40, 1))
        counted = flops(block)
        self.assertEqual(counted["n_occ"], 2)
        self.assertEqual(counted["n_dens"], 1)
        self.assertEqual(counted["k_flops"], 4 * 40 * 100 * 2)
        self.assertEqual(counted["j_flops"], 4 * 40 * 100 * 1)

    def test_unrestricted_counts_both_spins(self):
        block, = self.blocks(UNRESTRICTED)
        counted = flops(block)
        self.assertEqual(counted["n_occ"], 3)
        self.assertEqual(counted["n_dens"], 2)
        # No cuEST lines: the SCF iteration count stands in for the call count.
        self.assertEqual(counted["calls"], 2)
        self.assertEqual(counted["k_flops"], 4 * 40 * 100 * 3 * 2)

    def test_unprinted_fitting_basis_inherited_at_equal_nbf(self):
        blocks = fill_aux([{"nbf": 10, "naux": 40}, {"nbf": 10, "naux": None},
                           {"nbf": 20, "naux": None}])
        self.assertEqual(blocks[1]["naux"], 40)
        self.assertTrue(blocks[1]["naux_inherited"])
        self.assertIsNone(blocks[2]["naux"], "must not borrow across a different nbf")

    def test_rate_and_coverage_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory)
            (case / "psi4.out").write_text(RESTRICTED)
            (case / "timer.dat").write_text(TIMER)
            record = case_rate(case)
            self.assertEqual(record["modeled_jk_calls"], 1)
            self.assertEqual(record["timer_jk_calls"], 6)
            self.assertAlmostEqual(record["call_coverage"], 1 / 6)
            # 4 * naux * nbf^2 * nocc = 32000 flops over a 4 ms K kernel.
            self.assertAlmostEqual(record["k_kernel_tflops"], 32000 / 0.004 / 1e12)
            self.assertAlmostEqual(record["jk_timer_tflops"], 48000 / 0.5 / 1e12)


if __name__ == "__main__":
    unittest.main()
