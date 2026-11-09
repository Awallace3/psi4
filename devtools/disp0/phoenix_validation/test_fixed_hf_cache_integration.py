"""Offline full-fragment work contracts, not GPU execution evidence."""
import copy
from pathlib import Path
import tempfile
import unittest

from fixed_hf_cache_integration import (validate_work, validate_trace, CAPS, cache_plan,
                                      require_minimal_basis)
from fixed_hf_cache_integration import expected_minimal_basis


class FixedHFIntegrationTests(unittest.TestCase):
    def test_registered_minimal_basis_is_not_an_assumed_alias(self):
        require_minimal_basis("CC-PVTZ-MINAO","CC-PVTZ-MINAO")
        require_minimal_basis("cc-pvtz-minao","CC-PVTZ-MINAO")
        with self.assertRaises(ValueError):
            require_minimal_basis("minao","CC-PVTZ-MINAO")

    def test_original_explicit_or_pinned_registered_basis_controls_selection(self):
        source = Path(__file__).resolve().parents[3]
        self.assertEqual(expected_minimal_basis(source,{}),"CC-PVTZ-MINAO")
        self.assertEqual(expected_minimal_basis(source,{"minao_basis":"other"}),"other")
        self.assertEqual(expected_minimal_basis(source,{"MINAO_BASIS":"other"}),"other")
        with self.assertRaises(ValueError):
            expected_minimal_basis(source,{"minao_basis":"one","MINAO_BASIS":"two"})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/"psi4/src/read_options.cc"
            path.parent.mkdir(parents=True)
            for text in ("", 'options.add_str("MINAO_BASIS","a");\n'
                             'options.add_str("MINAO_BASIS","b");\n'):
                path.write_text(text)
                with self.assertRaises(RuntimeError):
                    expected_minimal_basis(root,{})

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
