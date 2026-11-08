"""Offline native parameter-wiring contracts, not native/GPU numerical evidence."""
from pathlib import Path
import re
import unittest

SOURCE = Path(__file__).resolve().parents[2]/"psi4/src/psi4/libfock/cuESTJK.cc"


def initialization():
    source = SOURCE.read_text()
    source = re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.S)
    return source.split("void cuESTJK::preiterations()", 1)[1].split(
        "void cuESTJK::destroy_cuest_objects()", 1)[0]


class NativeFittingPolicyTests(unittest.TestCase):
    def test_requested_cutoff_is_passed_before_plan_workspace_query(self):
        source = initialization()
        match = re.search(
            r"cuestParametersConfigure\(\s*CUEST_DFINTPLAN_PARAMETERS,\s*dfint_params,\s*"
            r"CUEST_DFINTPLAN_PARAMETERS_FITTING_CUTOFF,\s*&condition_,\s*sizeof\(double\)\s*\)",
            source)
        self.assertIsNotNone(match, "DF_FITTING_CONDITION is only printed, not applied")
        self.assertLess(match.start(), source.index("cuestDFIntPlanCreateWorkspaceQuery"))

    def test_metric_inverse_is_explicit_matrix_power_and_relative(self):
        source = initialization()
        for prop in ("FITTING_ALGORITHM", "FITTING_RELATIVE_CONDITIONING"):
            self.assertIn("CUEST_DFINTPLAN_PARAMETERS_"+prop, source)
            self.assertLess(source.index("CUEST_DFINTPLAN_PARAMETERS_"+prop),
                            source.index("cuestDFIntPlanCreateWorkspaceQuery"))
        self.assertRegex(source, r"fitting_algorithm\s*=\s*"
            r"CUEST_DFINTPLAN_PARAMETERS_FITTING_ALGORITHM_MATRIXPOWER")
        self.assertRegex(source, r"int32_t\s+relative_conditioning\s*=\s*1")
        self.assertRegex(source, r"CUEST_DFINTPLAN_PARAMETERS_FITTING_ALGORITHM,\s*"
                         r"&fitting_algorithm,\s*sizeof\(fitting_algorithm\)")
        self.assertRegex(source, r"CUEST_DFINTPLAN_PARAMETERS_FITTING_RELATIVE_CONDITIONING,\s*"
                         r"&relative_conditioning,\s*sizeof\(relative_conditioning\)")

    def test_requested_fp64_precedes_all_numerical_plan_construction(self):
        source = initialization()
        mode = source.index("cuestSetMathMode")
        self.assertLess(mode, source.index("cuestAOPairListCreateWorkspaceQuery"))
        self.assertLess(mode, source.index("cuestDFIntPlanCreateWorkspaceQuery"))
        self.assertIn('!options_.get_bool("CUEST_MIXED_PRECISION")', source)
        self.assertIn("CUEST_NATIVE_FP64_MATH_MODE", source)


if __name__ == "__main__":
    unittest.main()
