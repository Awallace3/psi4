"""Deterministic host-clock tests; no native/GPU execution."""
import unittest
from unittest.mock import patch

from test_cuest_plan_owner import PACKAGE
import importlib

timing = importlib.import_module(PACKAGE+".sapt_cuest_timing")


class Timings(unittest.TestCase):
    def test_nested_inclusive_clocks_counts_failures_and_snapshot(self):
        times = timing.PhaseTimings()
        with patch.object(timing.time,"perf_counter",side_effect=[0.,1.,3.,5.,7.,11.]), \
             patch.object(timing.time,"process_time",side_effect=[0.,2.,6.,8.,9.,10.]):
            with times.measure("outer"):
                with times.measure("inner"):
                    pass
            snapshot = times.report()
            error = KeyboardInterrupt("original")
            with self.assertRaises(KeyboardInterrupt) as caught:
                with times.measure("outer"):
                    raise error
        self.assertIs(caught.exception,error)
        phases = times.report()["phases"]
        self.assertEqual(phases["inner"], {
            "calls":1,"failures":0,"wall_seconds":2.,"process_cpu_seconds":4.})
        self.assertEqual(phases["outer"], {
            "calls":2,"failures":1,"wall_seconds":9.,"process_cpu_seconds":9.})
        self.assertEqual(snapshot["phases"]["outer"]["calls"],1)
        snapshot["phases"]["outer"]["calls"] = 99
        self.assertEqual(times.report()["phases"]["outer"]["calls"],2)


if __name__ == "__main__":
    unittest.main()
