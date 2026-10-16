"""External potentials with USE_CUEST.

Point charges go through cuEST's one-electron potential plan (the same call that builds the
nuclear attraction), so their matrix must match Libint's PotentialInt. Diffuse (Gaussian)
external charges have no cuEST path yet: they keep running on the CPU and print a warning.
"""

import numpy as np
import pytest
import psi4

from addons import uusing

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick, pytest.mark.cuest]

_POINTS = [[-0.5, 1.0, 1.0, 0.0], [0.5, 0.0, 1.0, 3.0], [7.7, 0.0, 4.0, 1.0], [-7.7, 3.0, 1.0, -4.0]]
_DIFFUSE = [[-0.5, 1.0, 1.0, 0.0, 0.8], [0.5, 0.0, 4.0, 0.0, 0.7]]
_WARNING = "diffuse (Gaussian)"


def _water():
    psi4.set_num_threads(2)
    mol = psi4.geometry("""
        O
        H 1 0.96
        H 1 0.96 2 104.5
        symmetry c1
    """)
    psi4.set_options({"basis": "aug-cc-pvdz", "df_basis_scf": "aug-cc-pvdz-jkfit", "scf_type": "df",
                      "e_convergence": 1.e-10, "d_convergence": 1.e-9,
                      "use_cuest": False, "cuest_xc": False, "cuest_mixed_precision": False})
    return mol


def _potential_matrix(basis, charges, use_cuest):
    extern = psi4.core.ExternalPotential()
    for q, x, y, z in charges:
        extern.addCharge(q, x, y, z)
    psi4.set_options({"use_cuest": use_cuest})
    try:
        return extern.computePotentialMatrix(basis).np.copy()
    finally:
        psi4.set_options({"use_cuest": False})


@uusing("cuest")
@pytest.mark.parametrize("ncharge", [1, 4, 500])
def test_cuest_point_charge_matrix(ncharge):
    mol = _water()
    basis = psi4.core.BasisSet.build(mol, "ORBITAL", "aug-cc-pvdz")
    if ncharge > len(_POINTS):
        rng = np.random.default_rng(7)
        charges = np.column_stack([rng.normal(size=ncharge), rng.uniform(-15., 15., size=(ncharge, 3))]).tolist()
    else:
        charges = _POINTS[:ncharge]
    cpu = _potential_matrix(basis, charges, False)
    gpu = _potential_matrix(basis, charges, True)
    assert np.abs(cpu).max() > 1.e-3
    np.testing.assert_allclose(gpu, gpu.T, atol=1.e-13, rtol=0)
    np.testing.assert_allclose(gpu, cpu, atol=1.e-10 * max(1., np.abs(cpu).max()), rtol=0)


@uusing("cuest")
@pytest.mark.parametrize("spec", ["points", "points+diffuse"])
def test_cuest_external_potential_scf(spec, tmp_path):
    mol = _water()
    external = {"points": _POINTS}
    if spec == "points+diffuse":
        external["diffuse"] = _DIFFUSE
    cpu = psi4.energy("hf", molecule=mol, external_potentials=external)
    out = tmp_path / "cuest.out"
    psi4.core.set_output_file(str(out), False)
    psi4.set_options({"use_cuest": True})
    try:
        gpu = psi4.energy("hf", molecule=mol, external_potentials=external)
    finally:
        psi4.set_options({"use_cuest": False})
        psi4.core.close_outfile()
    assert abs(gpu - cpu) < 1.e-6
    assert (_WARNING in out.read_text()) == (spec == "points+diffuse")
