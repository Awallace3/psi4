import psi4
import pytest
import numpy as np
import re
from psi4 import compare_values

pytestmark = [pytest.mark.psi, pytest.mark.api]


@pytest.mark.xdm
def test_water_xdm():
    """Verify XDM energy bookkeeping for a single water molecule.

    Checks that the reported dispersion correction equals the difference between
    `b3lyp-xdm` and plain `b3lyp`, and that wavefunction variables are
    internally consistent.
    """
    mol = psi4.geometry("""
0 1
O    -1.55100700  -0.11452000   0.00000000
H    -1.93425900   0.76250300   0.00000000
H    -0.59967700   0.04071200   0.00000000
units angstrom
    """)
    psi4.set_options({
        "basis": "aug-cc-pvtz",
        "DFT_SPHERICAL_POINTS": 590,
        "DFT_RADIAL_POINTS": 99,
    })
    e_reg, wfn = psi4.energy("b3lyp", molecule=mol, return_wfn=True)
    psi4.set_options({"BASIS_GUESS": "sto-3g"})
    e, wfn = psi4.energy("b3lyp-xdm", molecule=mol, return_wfn=True)
    psi4.set_options({"BASIS_GUESS": False})
    wfn_vars = wfn.variables()
    assert np.isclose(wfn.energy(), e, atol=1.0e-12)
    direct_xdm = psi4.core.XDMDispersion.build("b3lyp", 0.5, 1.0)
    for hf_fraction in (-0.1, 1.1):
        with pytest.raises(ValueError, match="exact-exchange fraction must be between 0 and 1"):
            direct_xdm.compute_energy(wfn, hf_fraction)
    for functional_name, hf_fraction in (("lc-wpbe", 0.20), ("cam-b3lyp", 0.20), ("hse06", 0.20)):
        range_xdm = psi4.core.XDMDispersion.build(functional_name, 0.5, 1.0)
        with pytest.raises(ValueError, match="modified exact exchange is unsupported for range-separated functional"):
            range_xdm.compute_energy(wfn, hf_fraction)
    assert "DISPERSION CORRECTION ENERGY" in wfn_vars
    assert np.isclose(wfn_vars["DISPERSION CORRECTION ENERGY"], e - e_reg, atol=1e-6)
    assert np.isclose(
        wfn_vars["DFT TOTAL ENERGY"] - wfn_vars["DISPERSION CORRECTION ENERGY"],
        e_reg,
        atol=1e-6,
    )

    psi4.set_options({"REFERENCE": "UKS"})
    e_uks = psi4.energy("b3lyp-xdm", molecule=mol)
    assert compare_values(e, e_uks, 8, "RKS and UKS XDM energies")
    psi4.set_options({"REFERENCE": "RKS"})


@pytest.mark.xdm
def test_h2o_ghosts():
    """Ensure XDM pairwise outputs size correctly with and without ghosts.

    Runs a ghost-containing fragment calculation and a normal water monomer, then
    confirms the XDM C6 matrix shape reflects only real atoms. Also confirms
    the XDM energy matches the reference
    """

    m = psi4.geometry("""
0 1
Gh(O)    -1.55100700  -0.11452000   0.00000000
Gh(H)    -1.93425900   0.76250300   0.00000000
Gh(H)    -0.59967700   0.04071200   0.00000000
--
0 1
O    1.35062500   0.11146900   0.00000000
H    1.68039800  -0.37374100  -0.75856100
H    1.68039800  -0.37374100   0.75856100
units angstrom
    """)
    psi4.set_options({
        "basis": "sto-3g",
        "DFT_SPHERICAL_POINTS": 590,
        "DFT_RADIAL_POINTS": 99,
        "XDM_DISPERSION_PARAMETERS": [0.5, 1.0],
    })
    e_m, wfn_m = psi4.energy("b3lyp-xdm", molecule=m, return_wfn=True)
    disp_corr = wfn_m.variables()["DISPERSION CORRECTION ENERGY"]
    ref_disp_corr = -0.00474203021866958
    assert np.isclose(disp_corr, ref_disp_corr,
                      atol=1e-6), (f"Expected dispersion correction {ref_disp_corr}, got {disp_corr}")
    assert wfn_m.variables()["XDM C6 COEFFICIENTS"].shape == (3, 3)
    for variable in (
            "XDM C6 COEFFICIENTS",
            "XDM C8 COEFFICIENTS",
            "XDM C10 COEFFICIENTS",
            "XDM RC COEFFICIENTS",
    ):
        coefficients = wfn_m.variables()[variable].np
        assert np.all(np.isfinite(coefficients))
        assert np.all(np.diag(coefficients) > 0.0)
    pairwise_energy = wfn_m.variables()["XDM PAIRWISE ENERGY"].np
    assert np.allclose(np.diag(pairwise_energy), 0.0, atol=0.0)
    assert np.isclose(pairwise_energy.sum(), disp_corr, atol=1.0e-12)
    m = psi4.geometry("""
0 1
O    1.35062500   0.11146900   0.00000000
H    1.68039800  -0.37374100  -0.75856100
H    1.68039800  -0.37374100   0.75856100

units angstrom
    """)
    psi4.set_options({
        "basis": "sto-3g",
        "DFT_SPHERICAL_POINTS": 590,
        "DFT_RADIAL_POINTS": 99,
        "XDM_DISPERSION_PARAMETERS": [0.5, 1.0],
    })
    e_m, wfn_m = psi4.energy("b3lyp-xdm", molecule=m, return_wfn=True)
    assert wfn_m.variables()["XDM C6 COEFFICIENTS"].shape == (3, 3)
    disp_corr = wfn_m.variables()["DISPERSION CORRECTION ENERGY"]
    ref_disp_corr = -0.004982082759239244
    assert np.isclose(disp_corr, ref_disp_corr,
                      atol=1e-6), (f"Expected dispersion correction {ref_disp_corr}, got {disp_corr}")


@pytest.mark.xdm
def test_h2o_nh3_xdm_nocp():
    """Validate no-CP XDM interaction energies and reject CP."""

    dimer = psi4.geometry("""0 1
N -1.578718 -0.046611 0.000000
H -2.158621 0.136396 -0.809565
H -2.158621 0.136396 0.809565
H -0.849471 0.658193 0.000000
--
0 1
O    2.35062500   0.11146900   0.00000000
H    2.68039800  -0.37374100  -0.75856100
H    2.68039800  -0.37374100   0.75856100

units angstrom
    """)
    psi4.set_options({"basis": "aug-cc-pvdz"})

    for bsse_type in ["cp", "vmfc", ["nocp", "vmfc"]]:
        with pytest.raises(
                NotImplementedError,
                match="Counterpoise-based XDM energies are not implemented",
        ):
            psi4.energy("b3lyp-xdm", molecule=dimer, bsse_type=bsse_type)

    for bsse_type in ["cp", "vmfc", ["nocp", "vmfc"]]:
        with pytest.raises(
                NotImplementedError,
                match="Counterpoise-based XDM energies are not implemented",
        ):
            psi4.gradient("b3lyp-xdm", molecule=dimer, bsse_type=bsse_type)

    from psi4.driver.task_planner import task_planner

    findif_kwargs = dict(findif_verbose=1, findif_stencil_size=3, findif_step_size=0.005)
    with pytest.raises(NotImplementedError, match="Counterpoise-based XDM energies are not implemented"):
        task_planner("energy", "b3lyp-xdm/cc-pv[d,t]z", dimer, bsse_type="cp")
    with pytest.raises(NotImplementedError, match="Counterpoise-based XDM energies are not implemented"):
        task_planner("gradient", "b3lyp-xdm/cc-pv[d,t]z", dimer, bsse_type="cp", **findif_kwargs)
    with pytest.raises(NotImplementedError, match="Counterpoise-based XDM energies are not implemented"):
        task_planner("gradient", "b3lyp", dimer, bsse_type="cp", levels={1: "b3lyp-xdm"}, **findif_kwargs)

    e_los_ii = psi4.energy("b3lyp-xdm(los-ii)", molecule=dimer, bsse_type="nocp")
    assert compare_values(e_los_ii, -0.0010595757, 8, "No-CP XDM(LoS-II) energy")

    psi4.set_options({
        "basis": "sto-3g",
        "DFT_SPHERICAL_POINTS": 590,
        "DFT_RADIAL_POINTS": 99,
        "XDM_DISPERSION_PARAMETERS": [0.5, 1.0],
    })
    e_nocp = psi4.energy("b3lyp-xdm", molecule=dimer, bsse_type="nocp")
    assert compare_values(e_nocp, -0.0006344572, 8, "No-CP XDM energy")


@pytest.mark.xdm
def test_xdm_models_and_alias():
    """Check XDM model aliasing and explicit LOS-II selection.

    Verifies that ``-xdm`` and ``-xdm(kb49)`` are equivalent, and that
    ``-xdm(los-ii)`` selects a different parameterization.
    """

    mol = psi4.geometry("""
0 1
O    -1.55100700  -0.11452000   0.00000000
H    -1.93425900   0.76250300   0.00000000
H    -0.59967700   0.04071200   0.00000000
units angstrom
    """)
    psi4.set_options({
        "basis": "aug-cc-pvdz",
        "DFT_SPHERICAL_POINTS": 590,
        "DFT_RADIAL_POINTS": 99,
    })

    e_alias = psi4.energy("b3lyp-xdm", molecule=mol)
    e_kb49 = psi4.energy("b3lyp-xdm(kb49)", molecule=mol)
    e_los_ii = psi4.energy("b3lyp-xdm(los-ii)", molecule=mol)

    assert compare_values(e_alias, e_kb49, 10, "-XDM alias equals -XDM(KB49)")
    assert not np.isclose(e_los_ii, e_kb49, rtol=0.0, atol=1.0e-8)


@pytest.mark.xdm
def test_xdm_unsupported_functional():
    """Ensure unsupported functionals fail unless parameters are provided."""

    mol = psi4.geometry("""
0 1
O    -1.55100700  -0.11452000   0.00000000
H    -1.93425900   0.76250300   0.00000000
H    -0.59967700   0.04071200   0.00000000
units angstrom
    """)
    psi4.set_options({
        "basis": "sto-3g",
        "DFT_SPHERICAL_POINTS": 590,
        "DFT_RADIAL_POINTS": 99,
    })
    err_msg = ("XDMDispersion: No fitted BJ parameters for hf/sto-3g with model kb49. "
               "Provide [a1, a2] through XDM_DISPERSION_PARAMETERS.")
    with pytest.raises(
            psi4.p4util.ValidationError,
            match=re.escape(err_msg),
    ):
        psi4.energy("hf-xdm", molecule=mol)
    psi4.set_options({
        "basis": "sto-3g",
        "DFT_SPHERICAL_POINTS": 590,
        "DFT_RADIAL_POINTS": 99,
        "XDM_DISPERSION_PARAMETERS": [0.5, 1.0],
    })
    e_ref = -74.9690725681
    e = psi4.energy("hf-xdm", molecule=mol)
    assert compare_values(e, e_ref, 8, "HF-XDM energy with custom parameters")


@pytest.mark.xdm
def test_xdm_rejects_unknown_parameter_keys():
    from psi4.driver.procrouting import dft

    bad_functional = {
        "name": "TYPO-XDM",
        "xc_functionals": {
            "HYB_GGA_XC_B3LYP": {}
        },
        "dispersion": {
            "type": "xdm",
            "params": {
                "xdm_modle": "los-ii"
            }
        },
    }
    with pytest.raises(psi4.p4util.ValidationError, match="Unsupported XDM dispersion params.*xdm_modle"):
        dft.build_superfunctional(bad_functional, True)


@pytest.mark.xdm
def test_xdm_callable_basis_guess_skips_dispersion():
    mol = psi4.geometry("0 1\nH 0 0 0\nH 0 0 1.5\nunits angstrom")
    psi4.set_options({
        "basis": "aug-cc-pvdz",
        "BASIS_GUESS": "sto-3g",
        "DFT_SPHERICAL_POINTS": 110,
        "DFT_RADIAL_POINTS": 50,
    })

    def callable_xdm(name, npoints, deriv, restricted):
        superfunctional = psi4.core.SuperFunctional.XC_build("XC_HYB_GGA_XC_B3LYP", restricted)
        superfunctional.set_name("B3LYP-XDM")
        return superfunctional, {"type": "xdm", "params": {"xdm_model": "kb49"}}

    energy = psi4.energy("scf", molecule=mol, dft_functional=callable_xdm)
    psi4.set_options({"BASIS_GUESS": False})
    assert np.isfinite(energy)


@pytest.mark.xdm
def test_xdm_gradient_findif():
    """XDM gradients are finite differences of the full XDM-corrected energy.

    Compares ``psi4.gradient`` against an independently assembled central
    difference of ``psi4.energy`` over every Cartesian coordinate, and confirms
    that the XDM-minus-B3LYP gradient difference reproduces the finite-difference
    derivative of :psivar:`DISPERSION CORRECTION ENERGY`.
    """

    geom = """
0 1
O   -1.55100700  -0.11452000   0.00000000
H   -1.93425900   0.76250300   0.00000000
H   -0.59967700   0.04071200   0.10000000
units angstrom
symmetry c1
no_reorient
no_com
"""
    # Pin the grid and tighten SCF so the differences are not grid/convergence noise.
    psi4.set_options({
        "basis": "sto-3g",
        "DFT_SPHERICAL_POINTS": 302,
        "DFT_RADIAL_POINTS": 75,
        "XDM_DISPERSION_PARAMETERS": [0.5, 1.0],
        "scf__e_convergence": 1e-11,
        "scf__d_convergence": 1e-10,
    })

    mol = psi4.geometry(geom)
    grad_xdm = np.asarray(psi4.gradient("b3lyp-xdm", molecule=mol))
    assert grad_xdm.shape == (3, 3)

    def displaced(method, xyz):
        m = psi4.geometry(geom)
        m.set_geometry(psi4.core.Matrix.from_array(xyz))
        m.update_geometry()
        ene = psi4.energy(method, molecule=m)
        return ene, psi4.variable("DISPERSION CORRECTION ENERGY")

    xyz0 = np.asarray(mol.geometry())
    step = 0.005
    fd_total = np.zeros((3, 3))
    fd_disp = np.zeros((3, 3))
    for atom in range(3):
        for cart in range(3):
            plus, minus = xyz0.copy(), xyz0.copy()
            plus[atom, cart] += step
            minus[atom, cart] -= step
            e_p, d_p = displaced("b3lyp-xdm", plus)
            e_m, d_m = displaced("b3lyp-xdm", minus)
            fd_total[atom, cart] = (e_p - e_m) / (2 * step)
            fd_disp[atom, cart] = (d_p - d_m) / (2 * step)

    assert np.allclose(grad_xdm, fd_total,
                       atol=1e-5), (f"XDM gradient does not match central difference:\n{grad_xdm - fd_total}")

    # The XDM contribution must actually be present in the gradient.
    grad_b3lyp = np.asarray(psi4.gradient("b3lyp", molecule=mol, dertype=0))
    assert np.abs(fd_disp).max() > 1e-5, "dispersion gradient too small to be a meaningful test"
    assert np.allclose(grad_xdm - grad_b3lyp, fd_disp,
                       atol=1e-5), (f"XDM-minus-B3LYP gradient does not match d(DISPERSION CORRECTION ENERGY):\n"
                                    f"{(grad_xdm - grad_b3lyp) - fd_disp}")


@pytest.mark.xdm
def test_xdm_gradient_uks_parity():
    """A closed-shell UKS XDM gradient must match the RKS one."""

    mol = psi4.geometry("""
0 1
O   -1.55100700  -0.11452000   0.00000000
H   -1.93425900   0.76250300   0.00000000
H   -0.59967700   0.04071200   0.00000000
units angstrom
    """)
    psi4.set_options({
        "basis": "sto-3g",
        "DFT_SPHERICAL_POINTS": 302,
        "DFT_RADIAL_POINTS": 75,
        "XDM_DISPERSION_PARAMETERS": [0.5, 1.0],
        "scf__e_convergence": 1e-11,
        "scf__d_convergence": 1e-10,
        "REFERENCE": "RKS",
    })
    grad_rks = np.asarray(psi4.gradient("b3lyp-xdm", molecule=mol))

    psi4.set_options({"REFERENCE": "UKS"})
    grad_uks = np.asarray(psi4.gradient("b3lyp-xdm", molecule=mol))
    psi4.set_options({"REFERENCE": "RKS"})

    assert np.allclose(grad_rks, grad_uks, atol=1e-7), (f"RKS and UKS XDM gradients differ:\n{grad_rks - grad_uks}")


@pytest.mark.xdm
def test_xdm_gradient_open_shell():
    """XDM gradients work for an open-shell (UKS) system."""

    mol = psi4.geometry("""
0 2
C  0.000000  0.000000  0.000000
H  0.000000  1.078000  0.000000
H  0.933000 -0.539000  0.000000
H -0.933000 -0.539000  0.000000
units angstrom
    """)
    psi4.set_options({
        "basis": "sto-3g",
        "reference": "uks",
        "DFT_SPHERICAL_POINTS": 302,
        "DFT_RADIAL_POINTS": 75,
        "XDM_DISPERSION_PARAMETERS": [0.5, 1.0],
        "scf__e_convergence": 1e-11,
        "scf__d_convergence": 1e-10,
    })
    grad = np.asarray(psi4.gradient("b3lyp-xdm", molecule=mol))
    psi4.set_options({"reference": "rks"})
    assert grad.shape == (4, 3)
    assert np.isfinite(grad).all()


@pytest.mark.xdm
def test_xdm_gradient_nbody_levels():
    """XDM gradients compose with no-CP many-body ``levels`` and with a basis slash spec."""

    dimer = psi4.geometry("""0 1
N -1.578718 -0.046611 0.000000
H -2.158621 0.136396 -0.809565
H -2.158621 0.136396 0.809565
H -0.849471 0.658193 0.000000
--
0 1
O    2.35062500   0.11146900   0.00000000
H    2.68039800  -0.37374100  -0.75856100
H    2.68039800  -0.37374100   0.75856100
units angstrom
    """)
    psi4.set_options({
        "basis": "sto-3g",
        "DFT_SPHERICAL_POINTS": 302,
        "DFT_RADIAL_POINTS": 75,
        "XDM_DISPERSION_PARAMETERS": [0.5, 1.0],
    })

    from psi4.driver.task_planner import task_planner
    from psi4.driver.driver_findif import FiniteDifferenceComputer
    from psi4.driver.driver_nbody import ManyBodyComputer
    from psi4.driver.task_base import AtomicComputer

    findif_kwargs = dict(findif_verbose=1, findif_stencil_size=3, findif_step_size=0.005)

    # Every XDM gradient route must be finite difference, never an analytic SCF gradient.
    plan = task_planner("gradient", "b3lyp-xdm", dimer, **findif_kwargs)
    assert isinstance(plan, FiniteDifferenceComputer)

    plan = task_planner(
        "gradient",
        "scf",
        dimer,
        dft_functional={
            "name": "custom-xdm",
            "dispersion": {
                "type": "xdm"
            }
        },
        **findif_kwargs,
    )
    assert isinstance(plan, FiniteDifferenceComputer)

    plan = task_planner(
        "gradient",
        "scf",
        dimer,
        dft_functional={
            "name": "metadata-noise",
            "description": "Comparison to -XDM methods"
        },
        **findif_kwargs,
    )
    assert isinstance(plan, AtomicComputer)

    def callable_xdm(name, npoints, deriv, restricted):
        superfunctional = psi4.core.SuperFunctional.blank()
        superfunctional.set_name("CALLABLE-XDM")
        return superfunctional, {"type": "xdm", "params": {"xdm_model": "kb49"}}

    plan = task_planner("gradient", "scf", dimer, dft_functional=callable_xdm, **findif_kwargs)
    assert isinstance(plan, FiniteDifferenceComputer)

    with pytest.raises(NotImplementedError, match="Counterpoise-based XDM energies are not implemented"):
        task_planner("energy", "scf", dimer, dft_functional=callable_xdm, bsse_type="cp")

    plan = task_planner("gradient", "b3lyp-xdm/sto-3g", dimer, **findif_kwargs)
    assert isinstance(plan, FiniteDifferenceComputer)

    plan = task_planner("gradient",
                        "b3lyp",
                        dimer,
                        bsse_type="nocp",
                        levels={
                            1: "b3lyp-xdm",
                            2: "b3lyp"
                        },
                        **findif_kwargs)
    assert isinstance(plan, ManyBodyComputer)

    grad = np.asarray(psi4.gradient("b3lyp-xdm", molecule=dimer, bsse_type="nocp"))
    assert grad.shape == (7, 3)
    assert np.isfinite(grad).all()


@pytest.mark.xdm
def test_xdm_mixed_levels_preserve_analytic_gradients():
    """Only the ``levels`` entries that actually use XDM are planned as finite differences."""

    dimer = psi4.geometry("""0 1
N -1.578718 -0.046611 0.000000
H -2.158621 0.136396 -0.809565
H -2.158621 0.136396 0.809565
H -0.849471 0.658193 0.000000
--
0 1
O    2.35062500   0.11146900   0.00000000
H    2.68039800  -0.37374100  -0.75856100
H    2.68039800  -0.37374100   0.75856100
units angstrom
    """)
    psi4.set_options({
        "basis": "sto-3g",
        "DFT_SPHERICAL_POINTS": 302,
        "DFT_RADIAL_POINTS": 75,
        "XDM_DISPERSION_PARAMETERS": [0.5, 1.0],
    })

    from psi4.driver.task_planner import task_planner
    from psi4.driver.driver_cbs import CompositeComputer
    from psi4.driver.driver_findif import FiniteDifferenceComputer
    from psi4.driver.driver_nbody import ManyBodyComputer
    from psi4.driver.task_base import AtomicComputer

    findif_kwargs = dict(findif_verbose=1, findif_stencil_size=3, findif_step_size=0.005)

    plan = task_planner("gradient",
                        "b3lyp",
                        dimer,
                        bsse_type="nocp",
                        levels={
                            1: "b3lyp-xdm",
                            2: "b3lyp"
                        },
                        **findif_kwargs)
    assert isinstance(plan, ManyBodyComputer)

    planned = {}
    for task in plan.task_list.values():
        planned.setdefault(task.method.lower(), set()).add(type(task))

    assert planned["b3lyp-xdm"] == {FiniteDifferenceComputer}
    assert planned["b3lyp"] == {AtomicComputer}

    cbs_plan = task_planner("gradient",
                            "b3lyp",
                            dimer,
                            bsse_type="nocp",
                            levels={
                                1: "b3lyp-xdm/cc-pv[d,t]z",
                                2: "b3lyp"
                            },
                            **findif_kwargs)
    assert isinstance(cbs_plan, ManyBodyComputer)
    assert {type(task) for task in cbs_plan.task_list.values()} == {CompositeComputer, AtomicComputer}
    assert all(task.basis == "sto-3g" for task in cbs_plan.task_list.values() if isinstance(task, AtomicComputer))
    for task in cbs_plan.task_list.values():
        if isinstance(task, CompositeComputer):
            component_types = {}
            for component in task.task_list:
                component_types.setdefault(component.method.lower(), set()).add(type(component))
            assert component_types["b3lyp-xdm"] == {FiniteDifferenceComputer}
            assert component_types["hf"] == {AtomicComputer}

    mixed_cbs_plan = task_planner(
        "gradient",
        "cbs",
        dimer,
        scf_wfn="b3lyp-xdm",
        scf_basis="sto-3g",
        corl_wfn="mp2",
        corl_basis="sto-3g",
        **findif_kwargs,
    )
    mixed_tasks = {}
    for task in mixed_cbs_plan.task_list:
        mixed_tasks.setdefault(task.method.lower(), set()).add(type(task))
    assert mixed_tasks["b3lyp-xdm"] == {FiniteDifferenceComputer}
    assert mixed_tasks["mp2"] == {AtomicComputer}

    # A user-requested analytic gradient is still refused because one level needs XDM.
    with pytest.raises(NotImplementedError, match="Analytic XDM gradients are not implemented"):
        task_planner("gradient",
                     "b3lyp",
                     dimer,
                     bsse_type="nocp",
                     dertype=1,
                     levels={
                         1: "b3lyp-xdm",
                         2: "b3lyp"
                     },
                     **findif_kwargs)


@pytest.mark.xdm
def test_xdm_cbs_components_use_findif_convergence():
    """CBS XDM components use the same tightened convergence criteria as direct finite differences."""

    mol = psi4.geometry("""
0 1
O
H 1 0.96
H 1 0.96 2 104.5
symmetry c1
    """)
    psi4.set_options({
        "basis": "sto-3g",
        "e_convergence": 1.0e-10,
        "XDM_DISPERSION_PARAMETERS": [0.5, 1.0],
    })

    from psi4.driver.driver_findif import FiniteDifferenceComputer
    from psi4.driver.task_base import AtomicComputer
    from psi4.driver.task_planner import task_planner

    energy_plan = task_planner("energy", "scf", mol)
    assert energy_plan.keywords["E_CONVERGENCE"] == pytest.approx(1.0e-10)
    assert "SCF__E_CONVERGENCE" not in energy_plan.keywords

    psi4.set_options({"e_convergence": 1.0e-6})
    findif_kwargs = dict(findif_verbose=1, findif_stencil_size=3, findif_step_size=0.005)
    direct_plan = task_planner("gradient", "b3lyp-xdm", mol, **findif_kwargs)
    cbs_plan = task_planner(
        "gradient",
        "cbs",
        mol,
        scf_wfn="b3lyp-xdm",
        scf_basis="sto-3g",
        corl_wfn="mp2",
        corl_basis="sto-3g",
        **findif_kwargs,
    )

    xdm_tasks = [task for task in cbs_plan.task_list if task.method.lower() == "b3lyp-xdm"]
    assert xdm_tasks and all(isinstance(task, FiniteDifferenceComputer) for task in xdm_tasks)
    assert any(task.method.lower() == "mp2" and isinstance(task, AtomicComputer) for task in cbs_plan.task_list)
    assert direct_plan.keywords["E_CONVERGENCE"] == pytest.approx(1.0e-6)
    assert direct_plan.keywords["SCF__E_CONVERGENCE"] == pytest.approx(1.0e-8)
    assert direct_plan.keywords["SCF__D_CONVERGENCE"] == pytest.approx(1.0e-8)
    for task in xdm_tasks:
        assert task.keywords["E_CONVERGENCE"] == pytest.approx(1.0e-6)
        assert task.keywords["SCF__E_CONVERGENCE"] == direct_plan.keywords["SCF__E_CONVERGENCE"]
        assert task.keywords["SCF__D_CONVERGENCE"] == direct_plan.keywords["SCF__D_CONVERGENCE"]


@pytest.mark.xdm
def test_xdm_cbs_component_convergence_respects_stage_options():
    """CBS stage-local convergence options survive the finite-difference defaults."""

    mol = psi4.geometry("""
0 1
O
H 1 0.96
H 1 0.96 2 104.5
symmetry c1
    """)
    psi4.set_options({"basis": "sto-3g", "XDM_DISPERSION_PARAMETERS": [0.5, 1.0]})

    from psi4.driver.driver_findif import FiniteDifferenceComputer
    from psi4.driver.task_planner import task_planner

    findif_kwargs = dict(findif_verbose=1, findif_stencil_size=3, findif_step_size=0.005)

    def xdm_component_keywords(stage_options):
        """Split the XDM components into those carrying the stage options and those without."""
        plan = task_planner(
            "gradient",
            "cbs",
            mol,
            cbs_metadata=[
                {
                    "wfn": "b3lyp-xdm",
                    "basis": "sto-3g",
                    "options": stage_options
                },
                {
                    "wfn": "mp2",
                    "basis": "sto-3g"
                },
            ],
            **findif_kwargs,
        )
        staged, defaulted = [], []
        for job, task in zip(plan.compute_list, plan.task_list):
            if task.method.lower() != "b3lyp-xdm":
                continue
            assert isinstance(task, FiniteDifferenceComputer)
            if job["f_options"] == stage_options:
                staged.append(task.keywords)
            else:
                assert job["f_options"] is False
                defaulted.append(task.keywords)
        # The delta stage's implicit lower method repeats b3lyp-xdm without the stage options.
        assert staged and defaulted
        return staged, defaulted

    # A module-local stage override is not shadowed by a looser negotiated duplicate.
    staged, defaulted = xdm_component_keywords({"scf__e_convergence": 1.0e-10})
    for keywords in staged:
        assert [value for key, value in keywords.items() if key.upper() == "SCF__E_CONVERGENCE"] == [
            pytest.approx(1.0e-10)
        ]
        # Criteria the stage did not set still pick up the finite-difference default.
        assert keywords["SCF__D_CONVERGENCE"] == pytest.approx(1.0e-8)
    for keywords in defaulted:
        assert keywords["SCF__E_CONVERGENCE"] == pytest.approx(1.0e-8)
        assert keywords["SCF__D_CONVERGENCE"] == pytest.approx(1.0e-8)

    # An unqualified stage criterion does not suppress the SCF-local finite-difference default.
    staged, defaulted = xdm_component_keywords({"e_convergence": 1.0e-10})
    for keywords in staged:
        assert keywords["SCF__E_CONVERGENCE"] == pytest.approx(1.0e-8)
        assert [value for key, value in keywords.items() if key.upper() == "E_CONVERGENCE"] == [pytest.approx(1.0e-10)]
        assert keywords["SCF__D_CONVERGENCE"] == pytest.approx(1.0e-8)
    for keywords in defaulted:
        assert keywords["SCF__E_CONVERGENCE"] == pytest.approx(1.0e-8)
        assert keywords["SCF__D_CONVERGENCE"] == pytest.approx(1.0e-8)


@pytest.mark.xdm
def test_xdm_cbs_unavailable_method_raises_missing_method():
    """An unavailable -XDM CBS stage still reports Psi4's method-availability error."""

    mol = psi4.geometry("""
0 1
O
H 1 0.96
H 1 0.96 2 104.5
symmetry c1
    """)
    psi4.set_options({"basis": "sto-3g", "XDM_DISPERSION_PARAMETERS": [0.5, 1.0]})

    from psi4.driver.task_planner import task_planner

    with pytest.raises(psi4.MissingMethodError, match="b3lypp-xdm"):
        task_planner(
            "gradient",
            "cbs",
            mol,
            cbs_metadata=[
                {
                    "wfn": "b3lypp-xdm",
                    "basis": "sto-3g"
                },
                {
                    "wfn": "mp2",
                    "basis": "sto-3g"
                },
            ],
            findif_verbose=1,
            findif_stencil_size=3,
            findif_step_size=0.005,
        )


def test_empirical_dispersion_compatibility_import():
    from psi4.driver.procrouting.empirical_dispersion import EmpiricalDispersion as compatibility_class
    from psi4.driver.procrouting.empirical_disp.empirical_dispersion import EmpiricalDispersion

    assert compatibility_class is EmpiricalDispersion


@pytest.mark.xdm
def test_xdm_uses_wavefunction_exchange_fraction():
    from psi4.driver.procrouting.empirical_disp.empirical_dispersion import XDMDispersionFunctor

    class Functional:

        def x_alpha(self):
            return 0.54

        def is_x_lrc(self):
            return False

    class Wavefunction:

        def functional(self):
            return Functional()

        def set_array_variable(self, name, value):
            pass

    class Recorder:

        def compute_energy(self, wfn, hf_fraction):
            self.hf_fraction = hf_fraction
            return -0.01

    functor = XDMDispersionFunctor(functional_name="m06-2x", a1=0.5, a2_ang=1.0)
    recorder = Recorder()
    functor.xdm = recorder

    assert functor.compute_energy(None, Wavefunction()) == -0.01
    assert recorder.hf_fraction == pytest.approx(0.54)


@pytest.mark.xdm
def test_xdm_runtime_exchange_changes_free_volume():
    mol = psi4.geometry("0 1\nH 0 0 0\nH 0 0 1.5\nunits angstrom")
    psi4.set_options({
        "basis": "sto-3g",
        "DFT_SPHERICAL_POINTS": 110,
        "DFT_RADIAL_POINTS": 50,
    })
    _, wfn = psi4.energy("b3lyp", molecule=mol, return_wfn=True)
    xdm = psi4.core.XDMDispersion.build("b3lyp", 0.5, 1.0)

    xdm.compute_energy(wfn, 0.20)
    standard_c6 = psi4.core.array_variable("XDM C6 COEFFICIENTS").np[0, 1]
    xdm.compute_energy(wfn, 0.50)
    modified_c6 = psi4.core.array_variable("XDM C6 COEFFICIENTS").np[0, 1]

    assert not np.isclose(standard_c6, modified_c6, rtol=1.0e-10, atol=1.0e-12)


@pytest.mark.xdm
def test_xdm_modified_exchange_rejects_heavy_elements():
    mol = psi4.geometry("0 1\nCl 0 0 0\nCl 0 0 2.0\nunits angstrom")
    psi4.set_options({
        "basis": "sto-3g",
        "DFT_SPHERICAL_POINTS": 110,
        "DFT_RADIAL_POINTS": 50,
    })
    _, wfn = psi4.energy("b3lyp", molecule=mol, return_wfn=True)
    xdm = psi4.core.XDMDispersion.build("b3lyp", 0.5, 1.0)

    with pytest.raises(ValueError, match="modified HF exchange is unsupported for Z > 10"):
        xdm.compute_energy(wfn, 0.50)

    unknown_xdm = psi4.core.XDMDispersion.build("m06-2x", 0.5, 1.0)
    with pytest.raises(ValueError, match="modified HF exchange is unsupported for Z > 10"):
        unknown_xdm.compute_energy(wfn, 0.54)


@pytest.mark.xdm
def test_xdm_rejects_nonlocal_double_dispersion():
    from psi4.driver.p4util.exceptions import ValidationError
    from psi4.driver.procrouting import dft
    from psi4.driver.procrouting.dft import dft_builder

    assert "b3lyp-xdm" in dft_builder.functionals
    assert "wb97m-v-xdm" not in dft_builder.functionals
    with pytest.raises(ValidationError, match="XDM cannot be combined.*VV10"):
        dft.build_superfunctional("wb97m-v-xdm", True)


@pytest.mark.xdm
def test_xdm_rejects_empirical_double_dispersion():
    from psi4.driver.p4util.exceptions import ValidationError
    from psi4.driver.procrouting import dft, proc
    from psi4.driver.procrouting.dft import dft_builder

    assert dft_builder.functionals["b3lyp-d3bj"]["dispersion"]["type"] == "d3bj"
    assert "b3lyp-d3bj-xdm" not in dft_builder.functionals

    with pytest.raises(ValidationError, match="XDM cannot be combined with the 'd3bj' dispersion"):
        dft.build_superfunctional("b3lyp-d3bj-xdm", True)

    psi4.set_options({"basis": "aug-cc-pvdz", "XDM_DISPERSION_PARAMETERS": [0.5, 1.0]})
    with pytest.raises(ValidationError, match="XDM cannot be combined with the 'd3bj' dispersion"):
        proc.build_functional_and_disp("b3lyp-d3bj-xdm", True)


@pytest.mark.xdm
def test_xdm_kb49_aliases_and_lcwpbe_reference():
    from psi4.driver.procrouting import proc
    from psi4.driver.procrouting.dft import dft_builder

    assert "hse06-xdm" in dft_builder.functionals
    assert "hse06-xdm(kb49)" in dft_builder.functionals

    psi4.set_options({"basis": "aug-cc-pvdz"})
    reference, _ = proc.build_functional_and_disp("lc-wpbe", True)
    for alias in ("lc-wpbe-xdm", "lcwpbe-xdm", "wpbe-xdm"):
        superfunctional, functor = proc.build_functional_and_disp(alias, True)
        assert superfunctional.name().lower().startswith("lc-wpbe-xdm")
        assert superfunctional.x_alpha() == reference.x_alpha()
        assert superfunctional.x_beta() == reference.x_beta()
        assert superfunctional.x_omega() == reference.x_omega()
        assert functor.engine == "xdm"
        assert functor.xdm.functional_name() == "lc-wpbe"


@pytest.mark.xdm
def test_xdm_rejects_nonfinite_damping_parameters():
    from psi4.driver.procrouting import proc

    psi4.set_options({"XDM_DISPERSION_PARAMETERS": [float("nan"), 1.0]})
    with pytest.raises(psi4.p4util.ValidationError, match="values must be finite numbers"):
        proc.build_functional_and_disp("b3lyp-xdm", True)


@pytest.mark.xdm
def test_xdm_public_api_validation():
    direct = psi4.core.XDMDispersion(0.5, 1.0, "B3LYP")
    factory = psi4.core.XDMDispersion.build("b3lyp", 0.5, 0.52917720859)

    assert direct.functional_name() == "b3lyp"
    assert direct.functional_name() == factory.functional_name()
    with pytest.raises(ValueError, match="damping parameters must be finite"):
        psi4.core.XDMDispersion(float("inf"), 1.0, "b3lyp")
    with pytest.raises(ValueError, match="non-null wavefunction"):
        direct.compute_energy(None, 0.20)
    with pytest.raises(TypeError):
        direct.compute_energy(None)


@pytest.mark.xdm
def test_xdm_rejects_partial_explicit_parameters():
    from psi4.driver.procrouting.empirical_disp.empirical_dispersion import XDMDispersionFunctor

    with pytest.raises(psi4.p4util.ValidationError, match="a1 and a2_ang must be provided together"):
        XDMDispersionFunctor("b3lyp", basis_name="aug-cc-pvdz", a1=9.0)
    with pytest.raises(psi4.p4util.ValidationError, match="a1 and a2_ang must be provided together"):
        XDMDispersionFunctor("b3lyp", basis_name="aug-cc-pvdz", a2_ang=9.0)


@pytest.mark.xdm
def test_xdm_automatic_damping_rejects_modified_exchange():
    from psi4.driver.procrouting import proc

    psi4.set_options({"basis": "aug-cc-pvdz", "DFT_ALPHA": 0.50})
    with pytest.raises(
            psi4.p4util.ValidationError,
            match="Automatic XDM damping parameters require.*exact-exchange fraction",
    ):
        proc.build_functional_and_disp("b3lyp-xdm", True)


@pytest.mark.xdm
def test_xdm_rejects_range_separation_override():
    from psi4.driver.procrouting import proc

    psi4.set_options({"basis": "sto-3g", "XDM_DISPERSION_PARAMETERS": [0.5, 1.0]})

    def callable_xdm(name, npoints, deriv, restricted):
        superfunctional = psi4.core.SuperFunctional.XC_build("XC_HYB_GGA_XC_LRC_WPBE", restricted)
        superfunctional.set_lock(False)
        superfunctional.set_name("LC-wPBE-XDM")
        superfunctional.set_x_omega(0.8)
        return superfunctional, {"type": "xdm", "params": {"xdm_model": "kb49"}}

    superfunctional, functor = proc.build_functional_and_disp(callable_xdm, True)

    def unknown_callable_xdm(name, npoints, deriv, restricted):
        unknown = psi4.core.SuperFunctional.XC_build("XC_HYB_GGA_XC_LRC_WPBE", restricted)
        unknown.set_lock(False)
        unknown.set_name("MY-LRC-XDM")
        return unknown, {"type": "xdm", "params": {"xdm_model": "kb49"}}

    unknown_superfunctional, unknown_functor = proc.build_functional_and_disp(unknown_callable_xdm, True)

    class Wavefunction:

        def __init__(self, functional):
            self._functional = functional

        def functional(self):
            return self._functional

    with pytest.raises(psi4.p4util.ValidationError, match="modified range-separation parameters"):
        functor.compute_energy(None, Wavefunction(superfunctional))
    with pytest.raises(psi4.p4util.ValidationError, match="unknown range-separated functionals"):
        unknown_functor.compute_energy(None, Wavefunction(unknown_superfunctional))


@pytest.mark.xdm
def test_xdm_rejects_ecp_density():
    mol = psi4.geometry("0 2\nBr 0 0 0\nunits angstrom")
    psi4.set_options({
        "basis": "lanl2dz",
        "reference": "uks",
        "XDM_DISPERSION_PARAMETERS": [0.5, 1.0],
    })

    with pytest.raises(RuntimeError, match="effective-core potentials are not supported"):
        psi4.energy("pbe-xdm", molecule=mol)

    psi4.set_options({"reference": "rks"})


@pytest.mark.xdm
def test_xdm_hessian_and_properties_blocked():
    """XDM Hessians and analytic properties remain clearly rejected."""

    mol = psi4.geometry("""
0 1
O   -1.55100700  -0.11452000   0.00000000
H   -1.93425900   0.76250300   0.00000000
H   -0.59967700   0.04071200   0.00000000
units angstrom
    """)
    psi4.set_options({"basis": "sto-3g", "XDM_DISPERSION_PARAMETERS": [0.5, 1.0]})

    with pytest.raises(NotImplementedError, match="XDM hessian is not implemented"):
        psi4.hessian("b3lyp-xdm", molecule=mol)
    with pytest.raises(NotImplementedError, match="XDM properties is not implemented"):
        psi4.properties("b3lyp-xdm", properties=["dipole"], molecule=mol)

    # No internal route may use an incomplete analytic XDM derivative.
    from psi4.driver.procrouting.empirical_disp.empirical_dispersion import XDMDispersionFunctor

    functor = XDMDispersionFunctor(functional_name="b3lyp", a1=0.5, a2_ang=1.0)
    with pytest.raises(NotImplementedError, match="Analytic XDM gradients are not implemented"):
        functor.compute_gradient(mol)


@pytest.mark.xdm
def test_xdm_gradient_produces_attraction():
    """The XDM gradient must add real intermolecular attraction.

    A broken or silently omitted XDM gradient leaves the dispersion-free DFT gradient,
    which does not pull a stretched closed-shell dimer together. At a stretched Ne2
    separation the XDM gradient must point inward (dE/dR > 0, so the energy falls as the
    atoms approach) and must exceed the plain-B3LYP gradient by the finite-difference
    derivative of :psivar:`DISPERSION CORRECTION ENERGY`.
    """
    BOHR = 0.52917721067
    R, step = 4.0, 0.01

    psi4.set_options({
        "basis": "aug-cc-pvdz",
        "DFT_SPHERICAL_POINTS": 302,
        "DFT_RADIAL_POINTS": 75,
        "scf__e_convergence": 1e-11,
        "scf__d_convergence": 1e-10,
    })

    def dimer(r):
        return psi4.geometry(f"0 1\nNe 0 0 0\nNe 0 0 {r}\nunits angstrom")

    def dEdR(method, r):
        # z-gradient of the second atom is dE/dR for this collinear dimer, in Eh/bohr
        return np.asarray(psi4.gradient(method, molecule=dimer(r)))[1, 2]

    def edisp(r):
        psi4.energy("b3lyp-xdm", molecule=dimer(r))
        return psi4.variable("DISPERSION CORRECTION ENERGY")

    d_xdm = dEdR("b3lyp-xdm", R)
    d_b3lyp = dEdR("b3lyp", R)

    assert d_xdm > 0.0, f"XDM gradient is not attractive at {R} A: dE/dR = {d_xdm:.3e}"
    assert d_xdm > d_b3lyp, (f"XDM gradient ({d_xdm:.3e}) is not more attractive than B3LYP ({d_b3lyp:.3e})")

    fd_disp = (edisp(R + step) - edisp(R - step)) / (2 * step) * BOHR
    assert np.isclose(d_xdm - d_b3lyp, fd_disp,
                      atol=5e-6), (f"XDM-minus-B3LYP gradient {d_xdm - d_b3lyp:.4e} does not match "
                                   f"d(DISPERSION CORRECTION ENERGY)/dR {fd_disp:.4e}")
