"""Initial XC grids must never become the reported final SCF solution."""

import numpy as np
import pytest

import psi4
from addons import uusing

pytestmark = [pytest.mark.psi, pytest.mark.api]

WATER = "O 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.24\nsymmetry c1"


def options(**extra):
    psi4.set_options({
        "basis": "sto-3g", "scf_type": "pk", "guess": "core",
        "e_convergence": 10, "d_convergence": 8,
        "dft_radial_points": 75, "dft_spherical_points": 302,
        **extra,
    })


@pytest.mark.parametrize("method,reference,geometry", [
    ("pbe0", "rks", WATER),
    ("svwn", "uks", "0 2\nH 0 0 0\nsymmetry c1"),
    ("tpss", "rks", WATER),
])
def test_switch_matches_final_grid(method, reference, geometry):
    mol = psi4.geometry(geometry)
    options(reference=reference)
    expected, ref = psi4.energy(method, molecule=mol, return_wfn=True)
    options(dft_radial_points_initial=35, dft_spherical_points_initial=110)
    actual, wfn = psi4.energy(method, molecule=mol, return_wfn=True)
    assert actual == pytest.approx(expected, abs=2.e-9)
    assert not wfn.V_potential().initial_grid()
    assert wfn.V_potential().grid().npoints() == ref.V_potential().grid().npoints()
    assert wfn.variable("SCF DFT INITIAL GRID ITERATIONS") >= 1
    assert wfn.variable("SCF DFT FINAL GRID ITERATIONS") >= 1
    assert np.max(np.abs(wfn.Da().to_array() - ref.Da().to_array())) < 2.e-6
    assert psi4.core.get_option("SCF", "DFT_RADIAL_POINTS") == 75


def test_already_converged_still_evaluates_final_grid():
    # A one-orbital atom can converge before the grid transition; it must not stop there.
    options(dft_radial_points_initial=35, dft_spherical_points_initial=110)
    _, wfn = psi4.energy("pbe", molecule=psi4.geometry("He 0 0 0\nsymmetry c1"), return_wfn=True)
    assert wfn.variable("SCF DFT FINAL GRID ITERATIONS") >= 1
    assert not wfn.V_potential().initial_grid()


@pytest.mark.parametrize("extra,message", [
    ({"dft_radial_points_initial": 35}, "Set both"),
    ({"dft_radial_points_initial": -1, "dft_spherical_points_initial": 110}, "Set both"),
    ({"dft_radial_points_initial": 35, "dft_spherical_points_initial": 111}, "Lebedev"),
    ({"dft_radial_points_initial": 75, "dft_spherical_points_initial": 302}, "smaller"),
    ({"dft_radial_points_initial": 35, "dft_spherical_points_initial": 110,
      "dft_grid_switch_factor": 0.0}, "DFT_GRID_SWITCH_FACTOR"),
])
def test_invalid_initial_grid(extra, message):
    options(**extra)
    with pytest.raises(Exception, match=message):
        psi4.energy("pbe", molecule=psi4.geometry(WATER))


@pytest.mark.parametrize("maxiter", [1, 2])
def test_maxiter_cannot_finalize_coarse_grid(maxiter):
    options(dft_radial_points_initial=35, dft_spherical_points_initial=110,
            maxiter=maxiter, fail_on_maxiter=False)
    with pytest.raises(psi4.driver.p4util.exceptions.SCFConvergenceError):
        psi4.energy("pbe", molecule=psi4.geometry("He 0 0 0\nsymmetry c1"))


def test_response_potential_uses_final_grid():
    options(dft_radial_points_initial=35, dft_spherical_points_initial=110)
    mol = psi4.geometry(WATER)
    basis = psi4.core.BasisSet.build(mol, "ORBITAL", "sto-3g")
    sf, _ = psi4.driver.dft.build_superfunctional("pbe", True)
    potential = psi4.core.VBase.build(basis, sf, "RV")
    potential.initialize()
    assert not potential.initial_grid()
    initial_points = potential.grid().npoints()
    potential.set_grid("Initial")
    assert potential.grid().npoints() < initial_points
    potential.set_grid("Final")
    assert potential.grid().npoints() == initial_points
    assert len(potential.properties()) == psi4.core.get_num_threads()


def test_grac_survives_switch():
    options(dft_grac_shift=0.1)
    mol = psi4.geometry(WATER)
    expected, _ = psi4.energy("pbe0", molecule=mol, return_wfn=True)
    options(dft_radial_points_initial=35, dft_spherical_points_initial=110)
    actual, wfn = psi4.energy("pbe0", molecule=mol, return_wfn=True)
    assert actual == pytest.approx(expected, abs=2.e-9)
    assert wfn.functional().needs_grac()
    assert wfn.V_potential().functional().grac_shift() == pytest.approx(0.1)


@pytest.mark.parametrize("postscf", [False, True])
def test_vv10_survives_switch(postscf):
    # The post-SCF correction is not stationary in the semilocal SCF density:
    # tighten the orbital residual for a sub-nanohartree differential comparison.
    options(dft_vv10_postscf=postscf, e_convergence=12, d_convergence=10)
    mol = psi4.geometry(WATER)
    expected, _ = psi4.energy("b97m-v", molecule=mol, return_wfn=True)
    options(dft_radial_points_initial=35, dft_spherical_points_initial=110,
            e_convergence=12, d_convergence=10)
    actual, wfn = psi4.energy("b97m-v", molecule=mol, return_wfn=True)
    assert actual == pytest.approx(expected, abs=2.e-9)
    assert not wfn.V_potential().initial_grid()
    assert wfn.V_potential().quadrature_values()["VV10"] != 0.0


@pytest.mark.parametrize("reference", ["rks", "uks"])
def test_c1_copy_preserves_final_grid(reference):
    options(reference=reference, dft_radial_points_initial=35,
            dft_spherical_points_initial=110)
    mol = psi4.geometry("O\nH 1 0.96\nH 1 0.96 2 104.5\nsymmetry c2v")
    _, wfn = psi4.energy("pbe", molecule=mol, return_wfn=True)
    assert wfn.nirrep() > 1
    c1 = mol.clone()
    c1.reset_point_group("c1")
    c1.fix_orientation(True)
    c1.fix_com(True)
    c1.update_geometry()
    basis = psi4.core.BasisSet.build(c1, "ORBITAL", "sto-3g")
    copy = wfn.c1_deep_copy(basis)
    assert copy.nirrep() == 1
    assert not copy.V_potential().initial_grid()
    assert copy.V_potential().grid().npoints() == wfn.V_potential().grid().npoints()
    # Evaluate a real potential on the copied density rather than checking the flag alone.
    potential = copy.V_potential()
    potential.set_D([copy.Da()] if reference == "rks" else [copy.Da(), copy.Db()])
    matrices = [psi4.core.Matrix("V", basis.nbf(), basis.nbf()) for _ in
                range(1 if reference == "rks" else 2)]
    potential.compute_V(matrices)
    assert np.linalg.norm(matrices[0].to_array()) > 0.0


def test_saptdft_components_match_final_grid():
    """Exercise scf_helper through both monomer GRAC SCFs and the HF delta."""
    mol = psi4.geometry("""
    0 1
    O 0 0 0
    H 0 0 0.96
    H 0.93 0 -0.24
    --
    0 1
    O 0 0 3.0
    H 0 0 3.96
    H 0.93 0 2.76
    symmetry c1
    """)
    options(basis="cc-pvdz", scf_type="df", sapt_dft_functional="pbe0",
            sapt_dft_grac_shift_a=0.1, sapt_dft_grac_shift_b=0.1,
            sapt_dft_induction_type="NONE", sapt_dft_do_disp=False,
            sapt_dft_do_dhf=True)
    keys = ["SAPT ELST ENERGY", "SAPT EXCH ENERGY", "SAPT IND ENERGY", "SAPT TOTAL ENERGY"]
    psi4.energy("sapt(dft)", molecule=mol)
    expected = {key: psi4.variable(key) for key in keys}
    options(basis="cc-pvdz", scf_type="df",
            dft_radial_points_initial=35, dft_spherical_points_initial=110)
    psi4.energy("sapt(dft)", molecule=mol)
    for key in keys:
        assert psi4.variable(key) == pytest.approx(expected[key], abs=1.e-7), key


@pytest.mark.parametrize("enabled", [False, True])
def test_libxc_microtiming_opt_in(enabled):
    options(cuest_xc_timing=enabled, dft_grac_shift=0.1)
    psi4.energy("pbe0", molecule=psi4.geometry(WATER))
    paths = psi4.core.get_timer_dict()
    micro = [path for path in paths if "XC micro:" in path]
    assert bool(micro) == enabled
    if enabled:
        assert any("xc_gga_exc_vxc" in path for path in micro)
        assert any("xc_lda_exc_vxc" in path for path in micro)  # GRAC correlation
        assert any("wrapper" in path for path in micro)


@pytest.mark.parametrize("reference", ["rks", "uks"])
def test_potential_validation_keeps_parent_timers_balanced(reference):
    options(reference=reference, dft_radial_points=25, dft_spherical_points=110)
    mol = psi4.geometry(WATER)
    basis = psi4.core.BasisSet.build(mol, "ORBITAL", "sto-3g")
    restricted = reference == "rks"
    sf, _ = psi4.driver.dft.build_superfunctional("pbe", restricted)
    potential = psi4.core.VBase.build(basis, sf, "RV" if restricted else "UV")
    potential.initialize()
    count = 1 if restricted else 2
    potential.set_D([psi4.core.Matrix("D", basis.nbf(), basis.nbf()) for _ in range(count)])
    psi4.core.clean_timers()
    with pytest.raises(Exception, match="Matrices|Matrix"):
        potential.compute_V([])
    potential.compute_V([psi4.core.Matrix("V", basis.nbf(), basis.nbf()) for _ in range(count)])
    label = ("RV" if restricted else "UV") + ": Form V"
    timers = psi4.core.get_timer_dict()
    assert timers[label]["n_calls"] == 2
    assert not any(label + ";" + label in key for key in timers)


@pytest.mark.cuest
@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.parametrize("grac", [0.0, 0.1])
def test_cuest_switching_and_microtiming(grac):
    # Compare against a fixed GPU grid: backend quadrature differences are
    # independent of the switching accuracy being tested.
    options(scf_type="df", dft_grac_shift=grac, use_cuest=True,
            cuest_xc=True, cuest_mixed_precision=False)
    mol = psi4.geometry(WATER)
    expected = psi4.energy("pbe0", molecule=mol)
    psi4.core.clean_timers()
    options(scf_type="df", use_cuest=True, cuest_xc=True, cuest_mixed_precision=False,
            cuest_xc_timing=True, dft_radial_points_initial=35,
            dft_spherical_points_initial=110)
    actual, wfn = psi4.energy("pbe0", molecule=mol, return_wfn=True)
    assert actual == pytest.approx(expected, abs=2.e-9)
    assert not wfn.V_potential().initial_grid()


@pytest.mark.cuest
@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.parametrize("postscf", [False, True])
def test_cuest_vv10_survives_switch(postscf):
    options(scf_type="df", use_cuest=True, cuest_xc=True, cuest_mixed_precision=False,
            dft_vv10_postscf=postscf, e_convergence=12, d_convergence=10,
            cuest_xc_timing=True)
    mol = psi4.geometry(WATER)
    expected = psi4.energy("b97m-v", molecule=mol)
    # options() defaults to PK: explicitly preserve the same GPU DF builder.
    options(scf_type="df", dft_radial_points_initial=35, dft_spherical_points_initial=110,
            e_convergence=12, d_convergence=10, cuest_xc_timing=True)
    actual, wfn = psi4.energy("b97m-v", molecule=mol, return_wfn=True)
    assert actual == pytest.approx(expected, abs=2.e-9)
    assert not wfn.V_potential().initial_grid()
    assert psi4.core.get_option("SCF", "SCF_TYPE") == "DF"
    assert wfn.V_potential().quadrature_values()["VV10"] != 0.0
    assert wfn.variable("SCF DFT FINAL GRID ITERATIONS") >= 1
    paths = psi4.core.get_timer_dict()
    for api in ["cuestXCIntPlanCreate", "cuestXCIntegrationWeightCompute",
                "cuestXCDensityCompute", "cuestXCPotentialCompute", "cudaMemcpy"]:
        assert any("XC micro:" in path and api in path for path in paths), api


@pytest.mark.cuest
@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.parametrize("method,reference,geometry", [
    ("svwn", "uks", "0 2\nH 0 0 0\nsymmetry c1"),
    ("tpss", "rks", WATER),
])
def test_cuest_other_rungs_match_fixed_gpu_grid(method, reference, geometry):
    """Isolate switching from the known backend-quadrature differences."""
    options(reference=reference, scf_type="df", use_cuest=True, cuest_xc=True,
            cuest_mixed_precision=False)
    mol = psi4.geometry(geometry)
    expected = psi4.energy(method, molecule=mol)
    options(dft_radial_points_initial=35, dft_spherical_points_initial=110,
            reference=reference, scf_type="df", cuest_xc_timing=True)
    actual, wfn = psi4.energy(method, molecule=mol, return_wfn=True)
    assert actual == pytest.approx(expected, abs=2.e-9)
    assert wfn.variable("SCF DFT FINAL GRID ITERATIONS") >= 1
    assert not wfn.V_potential().initial_grid()


@pytest.mark.cuest
@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.parametrize("method", ["pbe", "tpss"])
def test_cuest_empty_spin_density_all_components(method):
    """Zero the empty spin's gradient/tau as well as its LDA density."""
    options(reference="uks", scf_type="df", use_cuest=True, cuest_xc=True,
            cuest_mixed_precision=False, cuest_xc_timing=True)
    mol = psi4.geometry("0 2\nH 0 0 0\nsymmetry c1")
    expected = psi4.energy(method, molecule=mol)
    options(reference="uks", scf_type="df",
            dft_radial_points_initial=35, dft_spherical_points_initial=110)
    actual, wfn = psi4.energy(method, molecule=mol, return_wfn=True)
    assert actual == pytest.approx(expected, abs=2.e-9)
    quadrature = wfn.V_potential().quadrature_values()
    assert quadrature["RHO_A"] > 0.0
    assert quadrature["RHO_B"] == pytest.approx(0.0, abs=1.e-14)
    assert wfn.variable("SCF DFT FINAL GRID ITERATIONS") >= 1


# => Minimum final-grid iterations (scf_iterate(..., xc_final_min_iterations=N)) <= #
# A Python-only policy knob: no registered option yet, so the tests reach it the way the
# Protein157 policy runs do, by wrapping HF.iterations.

import functools


@pytest.fixture
def final_minimum():
    from psi4.driver.procrouting.scf_proc import scf_iterator
    original = psi4.core.HF.iterations
    proofs = []

    def install(minimum):
        @functools.wraps(scf_iterator.scf_iterate)
        def iterate(wfn, *args, **kwargs):
            switched = bool(wfn.V_potential() and wfn.V_potential().initial_grid())
            result = scf_iterator.scf_iterate(wfn, *args, xc_final_min_iterations=minimum, **kwargs)
            if switched:
                proofs.append((int(wfn.variable("SCF DFT INITIAL GRID ITERATIONS")),
                               int(wfn.variable("SCF DFT FINAL GRID ITERATIONS"))))
            return result
        psi4.core.HF.iterations = iterate
        return proofs

    yield install
    psi4.core.HF.iterations = original


@pytest.mark.parametrize("reference,geometry", [("rks", WATER), ("uks", "0 2\nH 0 0 0\nsymmetry c1")])
def test_full_threshold_minimum_two_matches_final_grid(reference, geometry, final_minimum):
    mol = psi4.geometry(geometry)
    options(reference=reference)
    expected = psi4.energy("pbe", molecule=mol)
    proofs = final_minimum(2)
    options(reference=reference, dft_radial_points_initial=35, dft_spherical_points_initial=110,
            dft_grid_switch_factor=1.0)
    assert psi4.energy("pbe", molecule=mol) == pytest.approx(expected, abs=2.e-9)
    assert proofs and all(initial > 0 and final >= 2 for initial, final in proofs)


@pytest.mark.parametrize("minimum", [1, 2, 3])
def test_minimum_final_iterations_exact_when_already_converged(minimum, final_minimum):
    """One-orbital He converges at once on each grid: the minimum alone sets the final count."""
    mol = psi4.geometry("He 0 0 0\nsymmetry c1")
    proofs = final_minimum(minimum)
    options(reference="rks", e_convergence=3, d_convergence=3, dft_radial_points_initial=35,
            dft_spherical_points_initial=110, dft_grid_switch_factor=1.0)
    psi4.energy("pbe", molecule=mol)
    assert proofs[-1][1] == minimum


def test_maxiter_cannot_bypass_final_minimum(final_minimum):
    mol = psi4.geometry("He 0 0 0\nsymmetry c1")
    final_minimum(100)
    options(reference="rks", maxiter=8, fail_on_maxiter=False, e_convergence=3, d_convergence=3,
            dft_radial_points_initial=35, dft_spherical_points_initial=110, dft_grid_switch_factor=1.0)
    with pytest.raises(psi4.driver.p4util.exceptions.SCFConvergenceError):
        psi4.energy("pbe", molecule=mol)


@pytest.mark.parametrize("minimum", [0, -1, 1.5, True])
def test_invalid_final_minimum_rejected(minimum, final_minimum):
    mol = psi4.geometry("He 0 0 0\nsymmetry c1")
    final_minimum(minimum)
    options(reference="rks", dft_radial_points_initial=35, dft_spherical_points_initial=110)
    with pytest.raises(psi4.driver.p4util.exceptions.ValidationError, match="positive integer"):
        psi4.energy("pbe", molecule=mol)
