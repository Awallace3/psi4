import pytest
from addons import uusing
import psi4

pytestmark = [pytest.mark.psi, pytest.mark.api]

# Two rigid methanes, the second rotated (zyx Euler 37, 21, 63 deg), 12.5 A apart: no dipole or quadrupole, so the
# true B3LYP interaction energy is far below the grid noise being tested.
_ch4_pair = """
0 1
C   0.000000   0.000000   0.000000
H   0.629118   0.629118   0.629118
H  -0.629118  -0.629118   0.629118
H  -0.629118   0.629118  -0.629118
H   0.629118  -0.629118  -0.629118
--
0 1
C   12.000000    3.000000   -2.000000
H   12.341055    2.916209   -0.968482
H   12.109857    2.037158   -2.498231
H   10.952015    3.298205   -2.012973
H   12.597074    3.748428   -2.520313
units angstrom
no_com
no_reorient
"""


@uusing("qcmanybody")
@pytest.mark.nbody
@pytest.mark.quick
def test_nocp_dft_grid_orientation():
    """No-CP DFT subsystems integrate on input-frame grids so quadrature error cancels.

    In the standard orientation each monomer's angular grid is rotated into its own frame,
    which differs from the dimer's, leaving a spurious interaction energy.
    """
    mol = psi4.geometry(_ch4_pair)
    psi4.set_options({"basis": "cc-pvdz", "scf_type": "df", "e_convergence": 1e-10, "d_convergence": 1e-10})

    from psi4.driver.task_planner import task_planner
    plan = task_planner("energy", "b3lyp", mol, bsse_type="nocp")
    assert all(task.keywords["DFT_GRID_ORIENTATION"] == "INPUT" for task in plan.task_list.values())
    plan = task_planner("energy", "b3lyp", mol, bsse_type="cp")
    assert all("DFT_GRID_ORIENTATION" not in task.keywords for task in plan.task_list.values())

    e_input = psi4.energy("b3lyp", molecule=mol, bsse_type="nocp")
    assert abs(e_input) < 5.0e-7, f"input-frame no-CP IE {e_input:.3e} Eh"

    # An explicit user setting is kept.
    psi4.set_options({"dft_grid_orientation": "standard"})
    e_standard = psi4.energy("b3lyp", molecule=mol, bsse_type="nocp")
    assert abs(e_standard - e_input) > 1.5e-6, f"standard {e_standard:.3e} vs input {e_input:.3e} Eh"
