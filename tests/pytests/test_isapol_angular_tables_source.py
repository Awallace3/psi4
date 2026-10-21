# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Source table identities and independent factorial checks, not compiled tests."""
import hashlib
import json
import math
from pathlib import Path
import re
import unittest
import numpy as np
from isapol_factorial_oracle import independent_g

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "psi4/src/psi4/libisapol"


class AngularTableSourceTests(unittest.TestCase):
    def test_all_real_cg_records_against_independent_factorial_oracle(self):
        text = (NATIVE / "realcg_data.inc").read_text()
        rows = [tuple(map(int, row.split(","))) for row in re.findall(r"\{([^{}]+)\}", text)]
        self.assertEqual(len(rows), 1887)
        self.assertEqual(len(set((a, b, v, k, q) for a,b,k,q,v,*_ in rows)), 1887)
        self.assertEqual({(a,b) for a,b,*_ in rows},
                         {(a,b) for a in range(1,5) for b in range(1,5) if a+b<=6})
        for a in range(1,5):
            for b in range(1,5):
                if a+b>6:
                    continue
                expected = independent_g(a,b)
                actual = np.zeros_like(expected)
                for la,lb,k,q,v,p,d,r,s in rows:
                    if (la,lb)==(a,b):
                        actual[v,k,q] = p/d*math.sqrt(abs(r)/s)*(1 if r>0 else 1j)
                np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=3e-15)
                square = actual[(a-b)**2:].reshape((2*a+1)*(2*b+1),-1)
                np.testing.assert_allclose(square@square.conj().T, np.eye(len(square)),
                                           rtol=0, atol=3e-15)

    def test_license_manifest_and_referencing_build_closure(self):
        manifest = json.loads((NATIVE / "realcg_manifest.txt").read_text())
        self.assertEqual(manifest["records"], 1887)
        self.assertEqual(hashlib.sha256((NATIVE/"RECOUPLED_CAMCASP_LICENSE").read_bytes()).hexdigest(),
                         manifest["license_sha256"])
        cmake = (NATIVE/"CMakeLists.txt").read_text()
        export = (ROOT/"psi4/src/export_isapol.cc").read_text()
        for source in ("anisotropic_dispersion", "recoupled_dispersion", "realcg_tables", "recoupling_tables"):
            self.assertIn(source+".cc", cmake)
            self.assertIn("psi4/libisapol/"+source+".h", export)
            for include in re.findall(r'#include "([^"]+)"', (NATIVE/(source+".cc")).read_text()):
                path = (ROOT/"psi4/src"/include) if include.startswith("psi4/") else NATIVE/include
                self.assertTrue(path.is_file(), str(path))
        for symbol in ("IsaAnisotropicModel", "IsaRecoupledModel", "isa_recoupled_dispersion",
                       "isapol_recoupling_blocks", "isapol_realcg_terms", "isapol_realcg_defined"):
            self.assertIn('"'+symbol+'"', export)
        for name in ("recoupled_h2o_isagrid_l3", "recoupled_rank4_casimir"):
            folder = ROOT/"tests/pytests/data_isapol"/name
            pins = json.loads((folder/"manifest.txt").read_text())
            self.assertEqual(hashlib.sha256((folder/"literal_numeric.json.gz").read_bytes()).hexdigest(),
                             pins["fixture_sha256"])
            self.assertEqual((folder/"LICENSE").read_bytes(),
                             (NATIVE/"RECOUPLED_CAMCASP_LICENSE").read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
