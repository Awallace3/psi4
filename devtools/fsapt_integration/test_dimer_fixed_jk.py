import unittest
import numpy as np

from dimer_fixed_jk import delta_summary


class FixedDimerJKContracts(unittest.TestCase):
    def test_closed_shell_factors_and_sign(self):
        d = np.diag([0.25, 0.75])
        z = np.zeros((2, 2))
        result = delta_summary(z, z, np.diag([2., 4.]), np.diag([3., 5.]), d)
        self.assertEqual(result["two_Tr_D_deltaJ_Eh"], 7.)
        self.assertEqual(result["minus_Tr_D_deltaK_Eh"], -4.5)
        self.assertEqual(result["closed_shell_JK_delta_Eh"], 2.5)

    def test_identical_inputs_have_zero_deltas(self):
        j = np.array([[1., 2.], [2., 3.]])
        result = delta_summary(j, j, j.copy(), j.copy(), np.eye(2))
        self.assertTrue(result["J_bitwise_equal"])
        self.assertTrue(result["K_bitwise_equal"])
        self.assertEqual(result["closed_shell_JK_delta_Eh"], 0.)
        self.assertEqual(result["J_rms"], 0.)

    def test_off_diagonal_trace_and_direction(self):
        d = np.array([[1., 2.], [3., 4.]])
        z = np.zeros((2, 2))
        j = np.array([[0., 5.], [7., 0.]])
        forward = delta_summary(z, z, j, z, d)
        reverse = delta_summary(j, z, z, z, d)
        self.assertEqual(forward["two_Tr_D_deltaJ_Eh"], 58.)
        self.assertEqual(reverse["two_Tr_D_deltaJ_Eh"], -58.)

    def test_bitwise_equality_distinguishes_signed_zero(self):
        plus, minus = np.zeros((1, 1)), np.full((1, 1), -0.)
        result = delta_summary(plus, plus, minus, plus, np.ones((1, 1)))
        self.assertFalse(result["J_bitwise_equal"])
        self.assertTrue(result["K_bitwise_equal"])
        self.assertEqual(result["J_max_abs"], 0.)


if __name__ == "__main__":
    unittest.main()
