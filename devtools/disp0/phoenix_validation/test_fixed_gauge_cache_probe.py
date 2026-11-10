"""Offline work-count/parity contracts, not physical GPU execution evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from fixed_gauge_cache_probe import (cache_plan, compare_components, PAIR_FIELDS, SCALARS,
                                    SCOPE, CAPS, ORDER, verify, digest)


class FixedGaugeCacheTests(unittest.TestCase):
    def test_protein_budget_changes_work_not_contraction_count(self):
        small = cache_plan(2488,58,49,688,698,24,64*1024**2)
        full = cache_plan(2488,58,49,688,698,24,4*1024**3)
        self.assertEqual((small["cached_slabs"],small["allocated_bytes"],
                          small["native_batches"],small["compute_blocks"]),
                         (1,8522336,871,870))
        self.assertEqual((full["cached_slabs"],full["allocated_bytes"],
                          full["native_batches"],full["compute_blocks"]),
                         (30,2974295264,59,870))
        self.assertEqual(full["hits"],840)

    def test_zero_tail_full_and_single_r_selection(self):
        for cap, expected in ((0,(0,12)),(480,(1,10)),(3360,(3,6))):
            plan = cache_plan(5,2,3,9,7,3,cap)
            self.assertEqual((plan["cached_slabs"],plan["native_batches"]),expected)
            self.assertLessEqual(plan["allocated_bytes"],cap)
        self.assertEqual(cache_plan(5,2,3,2,7,3,100000)["cached_slabs"],0)
        with self.assertRaises(ValueError):
            cache_plan(5,2,3,9,7,True,100)
        with self.assertRaises(ValueError):
            cache_plan(5,2,3,9,7,3,-1)

    def test_pair_and_scalar_checks_are_complete_and_strict(self):
        left = {k:np.ones((2,3)) for k in PAIR_FIELDS}
        left.update({k:6. for k in SCALARS})
        self.assertEqual(set(compare_components(left,left,(2,3))),
                         set(PAIR_FIELDS+SCALARS))
        for key in PAIR_FIELDS+SCALARS:
            right = dict(left)
            right[key] = np.asarray(left[key])+1e-8
            with self.assertRaises(AssertionError):
                compare_components(left,right,(2,3))
        for value in (np.full((2,3),np.nan),np.ones((3,2))):
            right = dict(left)
            right[PAIR_FIELDS[0]] = value
            with self.assertRaises(ValueError):
                compare_components(left,right,(2,3))

    def test_parent_replays_all_sealed_results_and_requires_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            values = {f"r{i}_{k}":np.ones((2,3)) if k in PAIR_FIELDS else np.array(6.)
                      for i in range(4) for k in PAIR_FIELDS+SCALARS}
            doc = {"status":"PASS_FIXED_GAUGE_CACHE_DIAGNOSTIC", "scope":SCOPE,
                   "accepted_speedup_count":0, "science_pin":"science",
                   "reference_result_sha256":"reference", "cache_caps":list(CAPS),
                   "order":list(ORDER), "scf_executed":False, "fisapt_executed":False,
                   "device":{"name":"NVIDIA H200"},
                   "provider_cleanup":{"complete":True,"live_bytes":0,"native_owners":0},
                   "runs":[{"arm":arm,"cache_cap_bytes":CAPS[arm],"wall_seconds":1.,
                            "maximum_abs_errors_from_first":{k:0. for k in PAIR_FIELDS+SCALARS}}
                           for arm in ORDER]}
            manifest = {"science_pin":"science","reference_result_sha256":"reference"}
            def seal():
                np.savez(output/"dispersion-pairs.npz",**values)
                doc["matrices_sha256"] = digest(output/"dispersion-pairs.npz")
                (output/"probe-result.json").write_text(json.dumps(doc))
            seal()
            with patch("scf_repeat_probe.require_reference",
                       return_value={"orbital_dimensions":{"active":[2,3]}}):
                self.assertEqual(verify(output,manifest)["status"],doc["status"])
                values["r2_"+PAIR_FIELDS[0]] += 1e-5
                np.savez(output/"dispersion-pairs.npz",**values)
                with self.assertRaises(RuntimeError):
                    verify(output,manifest)
                seal()
                with self.assertRaises(AssertionError):
                    verify(output,manifest)
                values["r2_"+PAIR_FIELDS[0]] -= 1e-5
                doc["provider_cleanup"]["complete"] = False
                seal()
                with self.assertRaises(RuntimeError):
                    verify(output,manifest)


if __name__ == "__main__":
    unittest.main()
