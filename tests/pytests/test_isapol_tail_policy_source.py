# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Inherited ISA recipe policy serialization/mapping, no native controller."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from isapol_source_test_support import load_source

P = load_source("isapol_native_partition")
core = P.core


class TailPolicySourceTests(unittest.TestCase):
    def test_published_always_default_and_when_enabled_mapping_are_unchanged(self):
        recipe = P.ControllerRecipe(convergence=1e-8,max_iterations=2,w_eps=1e-8,
            positive_lambda=1.,positive_max_alpha=1.,positive_auto=False,damping=0.,
            s_block_only=True,density_cutoff=0.,w_eps_activation=1.,positive_activation=1.,
            tail_activation=1.,mixing=1.,mixing_skip=0,tail_iteration_limit=1,fix_tails=False)
        self.assertEqual(recipe.tail_fit_policy,"always")
        self.assertIn("tail_fit_policy='always'",repr(recipe))
        alternatives = SimpleNamespace(always=object(),when_enabled=object())
        with patch.dict(core.__dict__,dict(IsaAControllerOptions=SimpleNamespace,
                IsaAFitOptions=SimpleNamespace,IsaTailFitPolicy=alternatives)):
            for policy in ("always","when_enabled"):
                built = replace(recipe,tail_fit_policy=policy).build(
                    [SimpleNamespace(tail_cutoff=2.,tail_allowed=False)])
                self.assertIs(built.tail_fit_policy,getattr(alternatives,policy))
                self.assertEqual(built.tail_allowed,[False])
                self.assertFalse(built.fix_tails)
        with self.assertRaisesRegex(ValueError,"tail_fit_policy"):
            replace(recipe,tail_fit_policy="automatic")
        self.assertEqual(P.DRHO_REFINEMENT_ITERATIONS,10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
