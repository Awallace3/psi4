import psi4
import pytest
import numpy as np

from pprint import pprint as pp
# pytestmark = [pytest.mark.psi, pytest.mark.api]


def test_water_xdm():
    """Test XDM on water dimer."""
    psi4.set_num_threads(12)
    psi4.set_memory("32 GB")
    mol = psi4.geometry("""
0 1
O    -1.55100700  -0.11452000   0.00000000
H    -1.93425900   0.76250300   0.00000000
H    -0.59967700   0.04071200   0.00000000
units angstrom
    """)
    psi4.set_options(
        {
            "basis": "aug-cc-pvtz",
            "DFT_SPHERICAL_POINTS": 590,
            "DFT_RADIAL_POINTS": 99,
        }
    )
    e, wfn = psi4.energy("b3lyp-xdm", molecule=mol, return_wfn=True)
    print(e)
    qcvars = psi4.core.variables()
    pp(qcvars)
    # set np print options to have commas, no truncation and 12 decimal places
    np.set_printoptions(precision=12, suppress=True, separator=", ")
    print(qcvars['XDM C6 COEFFICIENTS'].np)
    return


def test_water_water_xdm_IE():
    """Test XDM on water dimer."""
    psi4.set_num_threads(12)
    psi4.set_memory("32 GB")
    mol = psi4.geometry("""
0 1
O    -1.55100700  -0.11452000   0.00000000
H    -1.93425900   0.76250300   0.00000000
H    -0.59967700   0.04071200   0.00000000
--
0 1
O    1.35062500   0.11146900   0.00000000
H    1.68039800  -0.37374100  -0.75856100
H    1.68039800  -0.37374100   0.75856100
units angstrom
    """)
    psi4.set_options(
        {
            "basis": "aug-cc-pvtz",
            "DFT_SPHERICAL_POINTS": 590,
            "DFT_RADIAL_POINTS": 99,
        }
    )
    e, wfn = psi4.energy("b3lyp-xdm", molecule=mol, bsse_type="cp", return_wfn=True)
    print(e)
    qcvars = psi4.core.variables()
    pp(qcvars)
    pp(wfn.variables())
    return


if __name__ == "__main__":
    # pytest.main([__file__, "-x", "-v"])
    test_water_xdm()
    # test_water_water_xdm_IE()
