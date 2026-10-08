import unittest
import numpy as np

from mgga_grid_probe import preload_paths, restricted_ingredients


class GridProbeTests(unittest.TestCase):
    def test_restricted_spin_factors(self):
        raw = np.array([[1., 2., 3., 4., 5.], [.5, 0., 1., 0., .25]])
        ingredients = restricted_ingredients(raw)
        np.testing.assert_array_equal(ingredients["RHO_A"], [2., 1.])
        np.testing.assert_array_equal(ingredients["GAMMA_AA"], [116., 4.])
        np.testing.assert_array_equal(ingredients["TAU_A"], [10., .5])

    def test_conda_runtimes_precede_capture_object(self):
        self.assertEqual(list(map(str, preload_paths("/env", "/run/capture.so"))),
                         ["/env/lib/libgcc_s.so.1", "/env/lib/libstdc++.so.6",
                          "/run/capture.so"])


if __name__ == "__main__":
    unittest.main()
