# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Independent public-consumer preallocation guards; deliberately no core."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from isapol_source_test_support import load_source

R = load_source("isapol_refine")
R.core.IsaPfitStatus = SimpleNamespace(Solved="Solved")


def nodes():
    site = R.RefinementSite("A", "A", (0., 0., 0.), np.eye(3), 1)
    a = np.diag([0., 2., 3., 4.])
    result = []
    for f in (0., .7):
        m = R.refinement_model([site], [a], frequency_au=f,
            penalty_selection="static_reference", static_reference_tensors=[a])
        result.append(R.RefinementResult(m, m.anchors, (a,), 0., "Solved", None, None, "fake; no fit"))
    return result


class SchemeSourceTests(unittest.TestCase):
    def test_matching_schemes_reach_each_consumer_with_full_declared_tensors(self):
        A = load_source("isapol_native_recoupled")
        B = load_source("isapol_bounded_response")
        core = R.core
        def local(frequencies, sites, declaration, provenance):
            self.assertEqual(declaration, "supplied_local_response")
            self.assertEqual(sites[0].ranks, [1])
            np.testing.assert_array_equal(sites[0].responses[0], np.diag([2., 3., 4.]))
            return SimpleNamespace(frequencies=frequencies, sites=sites, provenance=provenance)
        def angular(a,b,w,n):
            return SimpleNamespace(model_a=a, pairs=(), frequencies=a.frequencies, cp_weights=w, max_order=n)
        replacements = dict(IsaAnisotropicSite=SimpleNamespace, IsaAnisotropicModel=local,
            print_out=lambda text:None,
            IsaRecoupledModel=lambda m:m, isa_recoupled_dispersion=angular,
            Matrix=SimpleNamespace(from_array=lambda a:a.copy()),
            isapol_realcg_defined=lambda a,b:True, isapol_realcg_terms=lambda a,b:[None],
            isapol_recoupling_blocks=lambda:[],
            IsaIsotropicSite=SimpleNamespace, IsaIsotropicModel=lambda *a:None,
            isa_isotropic_dispersion=lambda *a:SimpleNamespace(pairs=()))
        with patch.dict(core.__dict__, replacements):
            for mode in ("current", "static_reference", "explicit_subset"):
                rr = nodes()
                if mode != "static_reference":
                    models = [R.refinement_model(r.model.sites, r.refined_tensors,
                        frequency_au=r.model.frequency_au,
                        **(dict(declared_variables=r.model.parameter_labels,
                                penalized_variables=r.model.parameter_labels)
                           if mode=="explicit_subset" else {})) for r in rr]
                    rr = [replace(r,model=m) for r,m in zip(rr,models)]
                provenance = R._lw.Provenance("fake", "a"*64, "analytic", "no science")
                iso = R.refined_isotropic_dispersion(rr, cp_weights=(0., .4),
                                                     quadrature_provenance=provenance)
                ledger = B._Ledger(B.BoundedResources(512*1024**2, 1000000000000, 1024**3))
                ang = A.refined_recoupled_dispersion(rr, cp_weights=(0., .4),
                    quadrature_provenance=provenance, max_order=6, ledger=ledger)
                self.assertEqual(iso.frequencies, (0., .7))
                self.assertEqual(tuple(ang.cp_weights), (0., .4))
                self.assertGreater(ledger.peak, 0)
                if mode == "static_reference":
                    for r in rr:
                        self.assertIn(r.model.anchor_sha256, ang.provenance)

    def test_isotropic_refuses_policy_and_reference_before_native_allocation(self):
        for field, value in (("penalty_selection", "current"),
                             ("static_reference_sha256", "b"*64),
                             ("selection_sha256", "c"*64)):
            rr = nodes()
            rr[1] = replace(rr[1], model=replace(rr[1].model, **{field: value}))
            # No Matrix/IsaIsotropicSite exists in the fake core: validation
            # must refuse before trying to construct one.
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "penalty scheme.*"+field):
                R.refined_isotropic_dispersion(rr, cp_weights=(0., .4),
                    quadrature_provenance=R._lw.Provenance("fake", "a"*64, "analytic", "no science"))

    def test_angular_refuses_scheme_before_table_queries_or_ledger_charge(self):
        A = load_source("isapol_native_recoupled")
        B = load_source("isapol_bounded_response")
        for field, value in (("penalty_selection", "current"),
                             ("static_reference_sha256", "b"*64),
                             ("selection_sha256", "c"*64),
                             ("weight_type", 1), ("weight_coefficient", .002), ("cutoff", .0002)):
            rr = nodes()
            rr[1] = replace(rr[1], model=replace(rr[1].model, **{field: value}))
            ledger = B._Ledger(B.BoundedResources(512*1024**2, 1000000000000, 1024**3))
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "penalty scheme.*"+field):
                A.refined_recoupled_dispersion(rr, cp_weights=(0., .4),
                    quadrature_provenance=R._lw.Provenance("fake", "a"*64, "analytic", "no science"),
                    max_order=6, ledger=ledger)
            self.assertEqual((ledger.work, ledger.peak), (0, 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
