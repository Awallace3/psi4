import unittest
import numpy as np

from mgga_grid_probe import restricted_ingredients


class GridProbeTests(unittest.TestCase):
    def test_restricted_spin_factors(self):
        raw = np.array([[1., 2., 3., 4., 5.], [.5, 0., 1., 0., .25]])
        ingredients = restricted_ingredients(raw)
        np.testing.assert_array_equal(ingredients["RHO_A"], [2., 1.])
        np.testing.assert_array_equal(ingredients["GAMMA_AA"], [116., 4.])
        np.testing.assert_array_equal(ingredients["TAU_A"], [10., .5])


if __name__ == "__main__":
    unittest.main()
