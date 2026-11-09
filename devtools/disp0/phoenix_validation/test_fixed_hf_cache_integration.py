"""Offline full-fragment work contracts, not GPU execution evidence."""
import copy
import unittest

from fixed_hf_cache_integration import validate_work, validate_trace, CAPS, cache_plan


class FixedHFIntegrationTests(unittest.TestCase):
    def fixture(self):
        dims = {"active":[58,49]}
        plan = cache_plan(2488,58,49,688,698,24,CAPS[1])
        execution = {"same_session":True,"backend":"CUDA_CUEST_SHARED_IBO_EXPERIMENTAL",
            "jkfit_producer":{"calls":5},"rifit_producer":{"calls":59},
            "dispersion":{"compute_blocks":870,"factor_cache":{
                **{k:v for k,v in plan.items() if k not in ("native_batches","compute_blocks")},
                "budget_bytes":CAPS[1]}}}
        return execution,dims,plan

    def test_full_coexistence_requires_original_contractions_and_both_native_plans(self):
        execution,dims,plan = self.fixture()
        self.assertEqual(validate_work(execution,dims,2488,688,698),plan)
        for group,key,bad in (
            (None,"same_session",False),
            ("jkfit_producer","calls",4),
            ("rifit_producer","calls",871),
            ("dispersion","compute_blocks",869),
        ):
            changed = copy.deepcopy(execution)
            (changed[group] if group else changed)[key] = bad
            with self.assertRaises(RuntimeError):
                validate_work(changed,dims,2488,688,698)

    def test_cache_cap_occupancy_and_reuse_must_match_measured_plan(self):
        execution,dims,_ = self.fixture()
        for key,bad in (("budget_bytes",64*1024**2),("allocated_bytes",0),
                        ("cached_slabs",1),("hits",28),("s_factor_batches",842)):
            changed = copy.deepcopy(execution)
            changed["dispersion"]["factor_cache"][key] = bad
            with self.assertRaises(RuntimeError):
                validate_work(changed,dims,2488,688,698)

    def test_only_complete_nonoverlapping_term_windows_are_reportable(self):
        trace = {"workflow":{"status":"COMPLETED","wall_seconds":7.,
            "workflow_other_seconds":2.,
            "windows":{k:{"wall_seconds":1.} for k in
                       ("localization","elst","exch","ind","disp")}}}
        validate_trace(trace)
        changed = copy.deepcopy(trace)
        changed["workflow"]["windows"].pop("ind")
        with self.assertRaises(RuntimeError):
            validate_trace(changed)
        changed = copy.deepcopy(trace)
        changed["workflow"]["workflow_other_seconds"] = 3.
        with self.assertRaises(RuntimeError):
            validate_trace(changed)


if __name__ == "__main__":
    unittest.main()
