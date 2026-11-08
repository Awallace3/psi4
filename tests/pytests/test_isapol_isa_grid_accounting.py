"""Native grid row contract only: no SCF or change to grid construction.

Run in a fresh qualified compute allocation, NOT the local pure test suite.
These cases must join the native qualification union for this new identity.
"""
import pytest
from psi4 import core

from psi4.driver.procrouting.isapol_isa_grid_plan import isa_grid_rows

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]


@pytest.mark.parametrize("sites", [1, 3, 17])
def test_closed_isa_grid_rows_and_atom_offsets(sites):
    text = "0 1\n" + "\n".join(f"He {3*a} 0 0" for a in range(sites))
    molecule = core.Molecule.from_string(text+"\nunits bohr\nno_reorient\nno_com\n")
    options = core.IsaGridOptions()
    options.radial_points, options.spherical_points = 100, 200
    options.becke_smoothing, options.radius_scaling = 3, 1.
    grid = core.IsaGrid(molecule, options)
    expected = isa_grid_rows(
        sites, radial_points=100, spherical_points=200, becke_smoothing=3,
        radius_scaling=1., radius_policy="native_tabulated_bragg_slater",
        neighbour_policy="all_sites_unscreened_full_molecular_grid")
    assert grid.radial_points() == 100
    assert grid.spherical_points() == 230
    assert grid.natom() == sites
    assert grid.npoints() == expected == sites*99*230
    assert [grid.atom_start(a) for a in range(sites+1)] == [a*99*230 for a in range(sites+1)]
    assert [grid.atom_npoints(a) for a in range(sites)] == [99*230]*sites


@pytest.mark.parametrize("radial,angular_request,angular", [
    (2, 1, 6), (7, 7, 14), (101, 194, 194), (100, 200, 230),
    (3, 5294, 5294), (3, 9999, 5294)])
def test_general_recipe_keeps_conservative_accounting(radial, angular_request, angular):
    molecule = core.Molecule.from_string("0 1\nHe 0 0 0\nunits bohr\n")
    options = core.IsaGridOptions()
    options.radial_points, options.spherical_points = radial, angular_request
    options.becke_smoothing, options.radius_scaling = 2, 1.1
    grid = core.IsaGrid(molecule, options)
    envelope = isa_grid_rows(
        1, radial_points=radial, spherical_points=angular_request, becke_smoothing=2,
        radius_scaling=1.1, radius_policy="native_tabulated_bragg_slater",
        neighbour_policy="all_sites_unscreened_full_molecular_grid")
    assert grid.spherical_points() == angular
    assert grid.npoints() == (radial-1)*angular <= envelope
    assert envelope == (radial-1)*min(5294, 2*angular_request+6)
