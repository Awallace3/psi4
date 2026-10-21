# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Bounded/property orchestration contract tests with no scientific engine."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from isapol_source_test_support import load_source

B = load_source("isapol_bounded")
R = load_source("isapol_refine")
R.core.print_out = lambda text:None


class BoundedWaterSourceTests(unittest.TestCase):
    def test_default_builder_is_published_static_mask_not_cleanup_current_default(self):
        site = R.RefinementSite("A", "A", (0.,0.,0.), np.eye(3), 1)
        static, current = np.diag([0.,2.,0.,0.]), np.diag([0.,8e-5,5.,0.])
        models = B.refinement_models([site], [(0.,[static]),(.7,[current])],
            cutoff=1e-4, weight_type=3, weight_coefficient=1e-3,
            declared_variables=("A_10_10_A","A_11c_11c_A"))
        self.assertEqual(models[1].anchors, (8e-5,0.))
        self.assertGreater(models[1].strengths[0], 0.)
        self.assertEqual(models[1].strengths[1], 0.)
        self.assertEqual(models[1].unpenalized_variables, ("A_11c_11c_A",))
        self.assertEqual(models[0].static_reference_sha256, models[1].static_reference_sha256)
        self.assertNotEqual(models[0].anchor_sha256, models[1].anchor_sha256)
        self.assertEqual(models[0].penalty_selection, "static_reference")
        saved = models[0].static_reference_sha256
        static[:] = 99.
        self.assertEqual(models[0].static_reference_sha256, saved)

    def test_builder_retains_published_undeclared_dynamic_component_refusal(self):
        site = R.RefinementSite("A", "A", (0.,0.,0.), np.eye(3), 1)
        static, current = np.diag([0.,2.,0.,0.]), np.diag([0.,8e-5,5.,0.])
        with self.assertRaisesRegex(ValueError, "declared_variables omits"):
            B.refinement_models([site], [(0.,[static]),(.7,[current])],
                cutoff=1e-4, weight_type=3, weight_coefficient=1e-3)

    def test_published_all_dynamic_grid_keeps_first_node_subset_without_static_claim(self):
        site = R.RefinementSite("A","A",(0.,0.,0.),np.eye(3),1)
        models = B.refinement_models([site],[(.1,[np.diag([0.,2.,0.,0.])]),
                                             (.7,[np.diag([0.,8e-5,0.,0.])])],
            cutoff=1e-4,weight_type=3,weight_coefficient=1e-3)
        self.assertEqual(models[1].anchors,(8e-5,))
        self.assertEqual(models[0].penalty_selection,"explicit_subset")
        self.assertEqual(models[0].static_reference_sha256,"")
        self.assertEqual(models[0].selection_sha256,models[1].selection_sha256)

    def test_optional_angular_stage_failure_never_publishes_isotropic_variables(self):
        A = load_source("isapol_native_recoupled")
        q = SimpleNamespace(cp_weights=(0.,.4), provenance="fake")
        observed = []
        def fail(*a, **kw):
            observed.append("angular")
            raise RuntimeError("fake angular failure")
        def iso(*a, **kw):
            observed.append(("isotropic",kw.get("wfn")))
            return SimpleNamespace(pairs=())
        log = load_source("isapol_logging").StageLog(0)
        with patch.object(A,"refined_recoupled_dispersion",fail), \
             patch.object(R,"refined_isotropic_dispersion",iso):
            with self.assertRaisesRegex(RuntimeError,"fake angular"):
                B._refined_dispersions((), q, 6, object(), recoupled=True, log=log, wfn=object())
        self.assertEqual(observed, ["angular"])

    def test_no_opt_in_never_enters_angular_consumer(self):
        A = load_source("isapol_native_recoupled")
        q = SimpleNamespace(cp_weights=(0.,.4), provenance="fake")
        iso = SimpleNamespace(pairs=())
        with patch.object(A,"refined_recoupled_dispersion",side_effect=AssertionError("angular entered")), \
             patch.object(R,"refined_isotropic_dispersion",return_value=iso):
            result = B._refined_dispersions((),q,6,object(),log=load_source("isapol_logging").StageLog(0))
        self.assertEqual(result,(iso,None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
