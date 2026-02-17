import psi4
import pytest
import numpy as np
from pprint import pprint as pp

from psi4 import core

from utils import *
from addons import using, uusing

pytestmark = [pytest.mark.psi, pytest.mark.api]


def test_water_water_xdm():
    """Test XDM on water dimer."""
    psi4.set_num_threads(12)
    psi4.set_memory("32 GB")
    mol = psi4.geometry("""
0 1
8   -0.702196054   -0.056060256   0.009942262
1   -1.022193224   0.846775782   -0.011488714
1   0.257521062   0.042121496   0.005218999
--
0 1
8   2.268880784   0.026340101   0.000508029
1   2.645502399   -0.412039965   0.766632411
1   2.641145101   -0.449872874   -0.744894473
units angstrom
    """)
    psi4.set_options(
        {
            "basis": "aug-cc-pvtz",
        }
    )
    psi4.energy("pbe0-xdm/aug-cc-pvdz", molecule=mol, bsse_type='cp')
    qcvars = psi4.core.variables()
    pp(qcvars)
    return


if __name__ == "__main__":
    # pytest.main([__file__, "-x", "-v"])
    test_water_water_xdm()
