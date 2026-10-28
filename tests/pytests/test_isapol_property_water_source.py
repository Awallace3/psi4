# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Property settings/forwarding/owned-publication cleanup, fake wavefunction."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from isapol_source_test_support import load_source

P = load_source("isapol_bounded_oeprop")
B = load_source("isapol_bounded")
core = load_source("isapol_refine").core
core.print_out = lambda text:None


class Wfn:
    def __init__(self):
        self._native_atomic_property_result = object()
        self.variables = {"ATOMIC REFINED DISPERSION OLD":1., "UNRELATED":2.}
    def molecule(self):
        raise AssertionError("fake preparation must be overridden")
    def scalar_variables(self):
        return self.variables.copy()
    def array_variables(self):
        return {}
    def del_variable(self,key):
        del self.variables[key]


class PropertyWaterSourceTests(unittest.TestCase):
    def test_defaults_64_allowance_72_and_angular_opt_in_are_explicit(self):
        with patch.dict(core.__dict__,dict(has_global_option_changed=lambda name:False)):
            v = P.settings("water",{})
            self.assertEqual(v["max_refinement_parameters"],64)
            self.assertFalse(v["recoupled_dispersion"])
            self.assertEqual(v["distribution"],"df_centre_analytic")
            self.assertEqual(v["response"],"reference_h2h1")
            self.assertEqual(P.settings("water",{"recoupled_dispersion":True})["recoupled_dispersion"],True)
        self.assertEqual(B.check_max_parameters(72),72)
        for bad in (True,64.,np.int64(64),0,73,"72"):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                B.check_max_parameters(bad)

    def test_bad_opt_in_refuses_before_preparation_and_cleans_only_owned_names(self):
        wfn = Wfn()
        with patch.dict(core.__dict__,dict(Wavefunction=Wfn,has_global_option_changed=lambda name:False)):
            for bad in ("true",1,None):
                with self.subTest(bad=bad),self.assertRaisesRegex(TypeError,"explicit bool"):
                    P.run(wfn,["ATOMIC_REFINED_DISPERSION"],preset="water",recoupled_dispersion=bad)
        self.assertIsNone(wfn._native_atomic_property_result)
        self.assertEqual(wfn.variables,{"UNRELATED":2.})

    def test_angular_keyword_supplied_response_and_finite_allowance_forward_unchanged(self):
        wfn = Wfn()
        seen = {}
        output = object()
        def bounded(wfn,recipe,**kwargs):
            seen.update(kwargs)
            return output
        overrides = dict(auxiliary_recipe=object(),response_grid=object(),lattice_options=object(),
            quadrature=object(),resources=SimpleNamespace(max_bytes=512*1024**2),
            sites=(),bonds=(),scf_correction="NONE",
            distribution="supplied",distributed_moments=object(),response="native_fdds",fdds=object(),
            recoupled_dispersion=True,max_refinement_parameters=72,log=load_source("isapol_logging").StageLog(
                0,writer=lambda text:None))
        with patch.dict(core.__dict__,dict(Wavefunction=Wfn,has_global_option_changed=lambda name:False)), \
             patch.object(Wfn,"molecule",return_value=object()), \
             patch.object(P,"preset_sites",return_value=((),())), \
             patch.object(P,"bounded_properties",bounded):
            P.run(wfn,["ATOMIC_REFINED_DISPERSION"],preset="water",**overrides)
        self.assertIs(wfn._native_atomic_property_result,output)
        for name in ("distribution","distributed_moments","response","fdds","recoupled_dispersion"):
            self.assertEqual(seen[name],overrides[name])
        self.assertEqual(seen["max_parameters"],72)
        self.assertTrue(seen["publish_qcvariables"])
        overrides.pop("recoupled_dispersion")
        with patch.dict(core.__dict__,dict(Wavefunction=Wfn,has_global_option_changed=lambda name:False)), \
             patch.object(Wfn,"molecule",return_value=object()), \
             patch.object(P,"preset_sites",return_value=((),())), \
             patch.object(P,"bounded_properties",bounded):
            P.run(wfn,[P.RECOUPLED_TASK],preset="water",**overrides)
        self.assertTrue(seen["recoupled_dispersion"])
        self.assertFalse(seen["publish_qcvariables"])
        with patch.dict(core.__dict__,dict(Wavefunction=Wfn)), self.assertRaisesRegex(TypeError,"duplicate"):
            P.run(wfn,[P.RECOUPLED_TASK],preset="water",recoupled_dispersion=True)
        self.assertIsNone(wfn._native_atomic_property_result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
