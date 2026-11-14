"""Iteration-level safeguards for opt-in SCF acceleration."""
from types import SimpleNamespace

import numpy as np
import pytest
import psi4

from psi4.driver.procrouting.diis import DIIS, StoragePolicy
from psi4.driver.procrouting.scf_proc import scf_iterator, subclass_methods
from psi4.driver.p4util.exceptions import SCFConvergenceError

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]


def test_predicted_energy_contracts_density_difference():
    f = psi4.core.Matrix.from_array(np.diag([1e16, 1.0]))
    old = psi4.core.Matrix.from_array(np.eye(2))
    new = old.clone()
    new.np[1, 1] += 1e-8
    expected = new.np[1, 1] - old.np[1, 1]
    assert abs((f.vector_dot(new) - f.vector_dot(old)) - expected) > 0.5 * expected
    assert scf_iterator._density_energy_change([f], [new], [old], False) == expected
    assert scf_iterator._density_energy_change([f], [new], [old], True) == 2 * expected


@pytest.mark.parametrize("accelerator", ["NONE", "ADIIS", "EDIIS"])
@pytest.mark.parametrize("available", [1000, 100000])
def test_diis_reservation_precedes_jk_partition(monkeypatch, accelerator, available):
    psi4.set_options({"scf_initial_accelerator": accelerator, "diis_max_vecs": 8})
    get_global_option = psi4.core.get_global_option
    monkeypatch.setattr(psi4.core, "get_memory", lambda: available * 8)
    monkeypatch.setattr(
        psi4.core, "get_global_option",
        lambda key: 1.0 if key == "SCF_MEM_SAFETY_FACTOR" else get_global_option(key),
    )
    jk = SimpleNamespace(memory_estimate=lambda: 900)
    monkeypatch.setattr(scf_iterator, "_build_jk", lambda *args: jk)
    wfn = SimpleNamespace(
        diag_scratch_size=lambda: 0, nsopi=lambda: psi4.core.Dimension([1]),
        V_potential=lambda: None, jk=lambda: None, get_print=lambda: False,
        molecule=lambda: SimpleNamespace(), attempt_number_=2,
        form_D=lambda: None, compute_initial_E=lambda: 0.0, set_energies=lambda *args: None,
    )
    scf_iterator.scf_initialize(wfn)
    required = subclass_methods._scf_diis_memory_required(wfn, 8)
    assert wfn.memory_jk_ + wfn.memory_collocation_ + wfn._diis_memory_budget == available
    assert wfn._diis_memory_budget == (
        required if available == 100000 else subclass_methods._scf_diis_disk_scratch(wfn)
    )
    sample = [psi4.core.Matrix(1, 1)] * 6
    policy = subclass_methods._scf_diis_storage_policy(wfn, 8, sample)
    assert policy == (StoragePolicy.InCore if available == 100000 else StoragePolicy.OnDisk)


def test_diis_reservation_reselects_tight_memory_backend(monkeypatch):
    get_option = psi4.core.get_global_option
    monkeypatch.setattr(psi4.core, "get_memory", lambda: 910 * 8)
    monkeypatch.setattr(psi4.core, "get_global_option",
                        lambda key: 1.0 if key == "SCF_MEM_SAFETY_FACTOR" else get_option(key))
    calls = []
    def build(wfn, memory):
        calls.append(memory)
        return SimpleNamespace(memory_estimate=lambda: 900 if memory >= 900 else 500)
    monkeypatch.setattr(scf_iterator, "_build_jk", build)
    wfn = SimpleNamespace(
        diag_scratch_size=lambda: 0, nsopi=lambda: psi4.core.Dimension([1]),
        V_potential=lambda: None, jk=lambda: None, get_print=lambda: False,
        molecule=lambda: SimpleNamespace(), attempt_number_=2,
        form_D=lambda: None, compute_initial_E=lambda: 0.0, set_energies=lambda *args: None,
    )
    scf_iterator.scf_initialize(wfn)
    assert calls == [910, 894]
    assert wfn.memory_jk_ + wfn._diis_memory_budget == 910


@pytest.mark.parametrize("storage", [StoragePolicy.InCore, StoragePolicy.OnDisk])
@pytest.mark.parametrize("ediis", [False, True])
def test_grid_shift_rebases_each_density(monkeypatch, storage, ediis):
    manager = DIIS(4, "grid shift regression", storage_policy=storage,
                   engines={"diis", "ediis"} if ediis else {"diis"})
    f = psi4.core.Matrix.from_array(np.diag([1.0, 2.0]))
    dv = psi4.core.Matrix.from_array(np.array([[0.3, 0.7], [0.7, -0.2]]))
    densities = [
        psi4.core.Matrix.from_array(np.diag([1.0, 0.0])),
        psi4.core.Matrix.from_array(np.array([[0.4, 0.2], [0.2, 0.6]])),
    ]
    def commutator(fock, density):
        return psi4.core.Matrix.from_array(fock.np @ density.np - density.np @ fock.np)
    for density in densities:
        entry = {"target": [f], "error": [commutator(f, density)], "densities": [density]}
        if ediis:
            entry["energy"] = [0.5]
        manager.add_entry(entry)
    manager.get_dot_product(0, 1)
    monkeypatch.setattr(scf_iterator, "_alpha_beta", lambda wfn, name: [dv])
    wfn = SimpleNamespace(diis_manager_=manager, form_FDSmSDF=commutator)
    scf_iterator._xc_grid_shift_diis(wfn, [psi4.core.Matrix(2, 2)])
    if ediis:
        assert manager.stored_vectors == []
    else:
        shifted = f.clone()
        shifted.add(dv)
        expected_errors = []
        for i, density in enumerate(densities):
            expected = commutator(shifted, density)
            expected_errors.append(expected)
            np.testing.assert_allclose(manager.load_quantity("error", i, 0).np, expected.np, atol=1e-14)
            np.testing.assert_allclose(manager.load_quantity("target", i, 0).np, shifted.np, atol=1e-14)
        assert manager.get_dot_product(0, 1) == pytest.approx(expected_errors[0].vector_dot(expected_errors[1]))
    manager.delete_diis_file()


def test_streamed_adiis_matches_incore():
    rng = np.random.default_rng(898)
    managers = [
        DIIS(5, "streamed ADIIS", storage_policy=storage, engines={"diis", "adiis"})
        for storage in (StoragePolicy.InCore, StoragePolicy.OnDisk)
    ]
    try:
        for _ in range(5):
            entry = {key: [psi4.core.Matrix.from_array(rng.normal(size=(3, 3)))]
                     for key in ("target", "error", "densities")}
            for manager in managers:
                manager.add_entry(entry)
        for manager in managers:
            manager.adiis_populate()
        np.testing.assert_allclose(managers[0].adiis_linear, managers[1].adiis_linear, atol=1e-13)
        np.testing.assert_allclose(managers[0].adiis_quadratic, managers[1].adiis_quadratic, atol=1e-13)
    finally:
        for manager in managers:
            manager.delete_diis_file()


def test_vv10_restored_after_iteration_exception(monkeypatch):
    state = {"vv10": True}
    functional = SimpleNamespace(
        needs_vv10=lambda: state["vv10"], set_lock=lambda value: None,
        set_do_vv10=lambda value: state.update(vv10=value),
    )
    wfn = SimpleNamespace(
        V_potential=lambda: SimpleNamespace(initial_grid=lambda: True),
        functional=lambda: functional,
    )
    def failing(*args, **kwargs):
        state["vv10"] = False
        raise RuntimeError("injected iteration failure")
    monkeypatch.setattr(scf_iterator, "_scf_iterate_impl", failing)
    with pytest.raises(RuntimeError, match="injected iteration failure"):
        scf_iterator.scf_iterate(wfn)
    assert state["vv10"]


@pytest.mark.parametrize("reference", ["rhf", "uhf"])
def test_ooo_fp32_falls_back_to_internal(reference, tmp_path, monkeypatch):
    psi4.geometry("H 0 0 0\nH 0 0 0.9\nsymmetry c1\n")
    psi4.set_options({
        "reference": reference, "basis": "cc-pvdz", "scf_type": "mem_df",
        "e_convergence": 1e-10, "d_convergence": 1e-10, "save_jk": True,
    })
    energy, exact = psi4.energy("hf", return_wfn=True)
    density = exact.Da().np.copy()
    psi4.core.clean()
    psi4.set_options({"scf_k_fp32_convergence": 1e-3})
    # The compatibility policy must be testable even in builds where the
    # optional OOO library (and therefore its option value) is unavailable.
    original = psi4.core.get_option
    def get_option(module, option):
        if module.upper() == "SCF" and option.upper() == "ORBITAL_OPTIMIZER_PACKAGE":
            return "OOO"
        return original(module, option)
    monkeypatch.setattr(psi4.core, "get_option", get_option)
    out = tmp_path / "ooo-fallback.out"
    psi4.set_output_file(str(out), False)
    try:
        actual, wfn = psi4.energy("hf", return_wfn=True)
    finally:
        psi4.set_output_file("pytest_output.dat", True)
    assert "k_fp32=True" in out.read_text()
    assert not wfn.jk().dfh().K_fp32()
    np.testing.assert_allclose(actual, energy, atol=1e-9, rtol=0)
    np.testing.assert_allclose(wfn.Da().np, density, atol=1e-8, rtol=0)


@pytest.mark.parametrize("grid", [False, True])
def test_fast_paths_require_another_build_at_maxiter(grid):
    psi4.geometry("H 0 0 0\nH 0 0 0.9\nsymmetry c1\n")
    options = {
        "basis": "cc-pvdz", "scf_type": "mem_df", "guess": "core", "maxiter": 1,
        "e_convergence": 1000.0, "d_convergence": 1000.0,
        "save_jk": True, "fail_on_maxiter": True,
    }
    if grid:
        options.update(dft_radial_points_initial=20, dft_spherical_points_initial=50,
                       dft_radial_points=50, dft_spherical_points=110)
    else:
        options["scf_k_fp32_convergence"] = 1000.0
    psi4.set_options(options)
    with pytest.raises(SCFConvergenceError) as error:
        psi4.energy("wb97m-v" if grid else "hf")
    wfn = error.value.wfn
    assert not wfn.jk().dfh().K_fp32()
    if grid:
        assert wfn.functional().needs_vv10()
        assert not wfn._xc_grid_ready_for_finalize


@pytest.mark.parametrize("policy", ["SHIFT", "RESET"])
@pytest.mark.parametrize("accelerator", ["NONE", "ADIIS", "EDIIS"])
def test_grid_switch_isolated_observables(policy, accelerator):
    psi4.geometry("O 0 0 0\nH 0 0 1\nH 1 0 0\nsymmetry c1\n")
    psi4.set_options({
        "basis": "def2-svp", "scf_type": "mem_df",
        "e_convergence": 1e-10, "d_convergence": 1e-10,
        "scf_initial_accelerator": accelerator,
        "dft_radial_points": 50, "dft_spherical_points": 110,
    })
    exact, reference = psi4.energy("pbe", return_wfn=True)
    density, fock = reference.Da().np.copy(), reference.Fa().np.copy()
    psi4.core.clean()
    psi4.set_options({
        "dft_radial_points_initial": 20, "dft_spherical_points_initial": 50,
        "dft_grid_switch_diis": policy,
    })
    actual, wfn = psi4.energy("pbe", return_wfn=True)
    assert wfn._xc_grid_ready_for_finalize
    assert wfn.variable("SCF DFT INITIAL GRID ITERATIONS") > 0
    np.testing.assert_allclose(actual, exact, atol=1e-9, rtol=0)
    np.testing.assert_allclose(wfn.Da().np, density, atol=2e-8, rtol=0)
    np.testing.assert_allclose(wfn.Fa().np, fock, atol=2e-8, rtol=0)


@pytest.mark.parametrize("fp32", [False, True])
def test_final_grid_and_precision_before_accepting_convergence(monkeypatch, fp32):
    psi4.geometry("H 0 0 0\nH 0 0 0.9\nsymmetry c1\n")
    psi4.set_options({
        "basis": "cc-pvdz", "scf_type": "mem_df", "save_jk": True,
        "e_convergence": 1e-9, "d_convergence": 1e-9,
        "dft_radial_points": 50, "dft_spherical_points": 110,
        "dft_radial_points_initial": 20, "dft_spherical_points_initial": 50,
        "scf_k_fp32_convergence": 1000.0 if fp32 else 0.0,
    })
    exact, _ = psi4.energy("b3lyp", return_wfn=True)
    psi4.core.clean()
    original = scf_iterator._xc_grid_switch
    switches = []
    def switch(wfn, shift_diis, restore_vv10=False):
        switches.append(wfn.iteration_)
        return original(wfn, shift_diis, restore_vv10=restore_vv10)
    monkeypatch.setattr(scf_iterator, "_xc_grid_switch", switch)
    actual, wfn = psi4.energy("b3lyp", return_wfn=True)
    assert len(switches) == 1
    assert wfn.iteration_ > switches[0]
    assert wfn._xc_grid_ready_for_finalize
    assert not wfn.jk().dfh().K_fp32()
    np.testing.assert_allclose(actual, exact, atol=1e-8, rtol=0)


@pytest.mark.parametrize("reference", ["rhf", "uhf"])
@pytest.mark.parametrize("acceleration", ["predicted", "fp32"])
def test_precision_and_prediction_isolated(reference, acceleration):
    psi4.geometry("H 0 0 0\nH 0 0 0.9\nsymmetry c1\n")
    psi4.set_options({
        "reference": reference, "basis": "cc-pvdz", "scf_type": "mem_df", "save_jk": True,
        "e_convergence": 1e-10, "d_convergence": 1e-10,
    })
    exact, ref = psi4.energy("hf", return_wfn=True)
    density = ref.Da().np.copy()
    psi4.core.clean()
    psi4.set_options({"scf_e_conv_predicted": acceleration == "predicted",
                      "scf_k_fp32_convergence": 1e-3 if acceleration == "fp32" else 0.0})
    actual, wfn = psi4.energy("hf", return_wfn=True)
    assert not wfn.jk().dfh().K_fp32()
    np.testing.assert_allclose(actual, exact, atol=1e-9, rtol=0)
    np.testing.assert_allclose(wfn.Da().np, density, atol=1e-8, rtol=0)
