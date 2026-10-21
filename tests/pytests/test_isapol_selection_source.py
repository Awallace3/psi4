# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Pure constructor contract tests, independent of dispersion/ISA/MBIS/core."""
import hashlib
import unittest
import numpy as np
from isapol_source_test_support import load_source

R = load_source("isapol_refine")


class SelectionSourceTests(unittest.TestCase):
    def setUp(self):
        self.sites = (R.RefinementSite("H1", "H", (0., 0., 0.), np.eye(3), 2),
                      R.RefinementSite("H2", "H", (1., 0., 0.), np.eye(3), 2))
        self.static = np.zeros((9, 9))
        self.static[5, 7] = self.static[7, 5] = .02
        self.current = np.zeros((9, 9))
        self.current[5, 7] = self.current[7, 5] = 8.409218082086611e-5
        self.name = "H1_21c_22c_A"
        self.extra = "H1_10_11c_A"

    def model(self, **kwargs):
        return R.refinement_model(self.sites, [self.current]*2,
            frequency_au=6.91088595, declared_variables=(self.name, self.extra), **kwargs)

    def test_legacy_direct_and_explicit_subset_keep_published_numbers(self):
        direct = self.model()
        self.assertEqual(direct.anchors, (0., 0.))
        self.assertEqual(direct.strengths, (0., 0.))
        selected = self.model(penalized_variables=(self.name,))
        self.assertEqual(selected.anchors, (0., 8.409218082086611e-5))
        self.assertAlmostEqual(selected.strengths[1], 2.050846881665693e-5, places=12)
        self.assertEqual(selected.unpenalized_variables, (self.extra,))
        empty = self.model(penalized_variables=())
        self.assertEqual(empty.anchors, (0., 0.))

    def test_static_reference_keeps_current_anchor_and_free_user_extra(self):
        other = self.static.copy()
        other[1, 2] = other[2, 1] = 12.  # non-reference COPY site cannot select
        m = self.model(penalty_selection="static_reference",
                       static_reference_tensors=[self.static, other])
        self.assertEqual(m.anchors, (0., 8.409218082086611e-5))
        self.assertEqual(m.unpenalized_variables, (self.extra,))
        self.assertEqual(m.penalty_selection, "static_reference")
        expected = hashlib.sha256(self.static.astype("<f8").tobytes()
                                  + other.astype("<f8").tobytes()).hexdigest()
        self.assertEqual(m.static_reference_sha256, expected)
        self.assertNotEqual(expected, m.anchor_sha256)
        self.assertIn(expected, m.provenance)
        self.static[:] = 99.
        self.assertEqual(m.static_reference_sha256, expected)

    def test_ambiguous_and_incomplete_policy_declarations_fail_closed(self):
        reference = [self.static]*2
        for kw in (
            dict(penalty_selection="unknown"),
            dict(penalty_selection="static_reference"),
            dict(static_reference_tensors=reference),
            dict(penalty_selection="current", penalized_variables=()),
            dict(penalty_selection="static_reference", static_reference_tensors=reference,
                 penalized_variables=(self.name,)),
            dict(penalty_selection="explicit_subset"),
            dict(penalty_selection="explicit_subset", penalized_variables=(),
                 static_reference_tensors=reference),
            dict(penalty_selection="static_reference", static_reference_tensors=[]),
            dict(penalty_selection="static_reference", static_reference_tensors=[np.eye(4)]*2),
            dict(penalty_selection="static_reference",
                 static_reference_tensors=[np.full((9, 9), np.nan)]*2),
        ):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                self.model(**kw)

    def test_explicit_subset_identity_is_order_independent_and_empty_is_distinct(self):
        a = self.model(penalized_variables=(self.name, self.extra))
        b = self.model(penalized_variables=(self.extra, self.name))
        self.assertEqual(a.selection_sha256, b.selection_sha256)
        self.assertEqual(a.penalty_selection, "explicit_subset")
        self.assertEqual(len(a.selection_sha256), 64)
        self.assertNotEqual(self.model(penalized_variables=()).selection_sha256,
                            self.model().selection_sha256)

    def test_static_strict_cutoff_and_zero_current_anchor(self):
        self.static[5, 7] = self.static[7, 5] = 1e-4
        at = self.model(penalty_selection="static_reference",
                        static_reference_tensors=[self.static]*2)
        self.assertEqual(at.strengths, (0., 0.))
        self.static[5, 7] = self.static[7, 5] = np.nextafter(1e-4, np.inf)
        self.current[:] = 0.
        above = self.model(penalty_selection="static_reference",
                           static_reference_tensors=[self.static]*2)
        self.assertEqual(above.anchors, (0., 0.))
        self.assertGreater(above.strengths[1], 0.)
        self.assertEqual(above.unpenalized_variables, (self.extra,))

    def test_published_subset_completeness_check_is_unchanged(self):
        self.current[2, 2] = 5.
        with self.assertRaisesRegex(ValueError, "declared_variables omits"):
            self.model(penalized_variables=(self.name,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
