import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("grid_evidence", Path(__file__).with_name("grid_evidence.py"))
grid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grid)


class EvidenceTests(unittest.TestCase):
    def rows(self):
        return [dict(system="water", arm=arm, ok=True, wall_s=10 - i + repeat * .01,
                     options={"CUEST_XC_TIMING": False},
                     components={key: (2.e-6 if "75" in arm else 0.) for key in grid.COMPONENTS},
                     grac_shifts={"A": .1, "B": .1}, xc_s=1)
                for i, arm in enumerate(grid.ARMS) for repeat in range(3)]

    def test_complete_matched_groups(self):
        data = grid.summarize(self.rows())
        self.assertEqual(len(data["groups"]), 1)
        arms = {a["arm"]: a for a in data["groups"][0]["arms"]}
        self.assertEqual(arms["fixed99"]["median_s"], 10.01)
        self.assertAlmostEqual(arms["fixed75"]["max_component_error_eh"], 2.e-6)
        self.assertTrue(arms["fixed75"]["agreement_pass"])
        self.assertEqual(max(arms["coarse75"]["switching_error_eh"].values()), 0.)

    def test_incomplete_group_never_published(self):
        self.assertEqual(grid.summarize(self.rows()[:-1])["groups"], [])

    def test_failed_group_never_published(self):
        rows = self.rows()
        rows[0]["ok"] = False
        data = grid.summarize(rows)
        self.assertEqual(data["groups"], [])
        self.assertEqual(len(data["failures"]), 1)

    def test_profiling_is_rejected(self):
        rows = self.rows()
        rows[0]["options"]["CUEST_XC_TIMING"] = True
        with self.assertRaises(AssertionError):
            grid.summarize(rows)

    def test_accuracy_failure_remains_visible(self):
        rows = self.rows()
        for row in rows:
            if row["arm"] == "coarse75":
                row["components"][grid.COMPONENTS[0]] = 2.e-5
        arm = grid.summarize(rows)["groups"][0]["arms"][-1]
        self.assertFalse(arm["agreement_pass"])
        self.assertGreater(arm["speedup"], 1)

    def test_regions_use_leaf_labels_without_double_counting(self):
        d = {"root;RV: Form V": {"wall_time": 4},
             "root;RV: Form V;cuEST XC: Host Functional": {"wall_time": 2},
             "root;RV: Form V;cuEST XC: Host Functional;cuEST XC: GRAC Host Functional": {"wall_time": 1},
             "root;SCF: Driver": {"wall_time": 8}}
        self.assertEqual(grid.timer_regions(d)["xc_s"], 4)
        self.assertEqual(grid.timer_regions(d)["functional_s"], 2)


if __name__ == "__main__":
    unittest.main()
