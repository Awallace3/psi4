# Copyright (c) 2007-2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""State-only native admission; no response operators or large fixtures."""
import numpy as np
import psi4
import pytest


@pytest.fixture
def water_state():
    psi4.core.be_quiet()
    molecule = psi4.geometry("0 1\nO 0 0 0\nH .757 0 .586\nH -.757 0 .586\n"
                             "symmetry c1\nno_com\nno_reorient")
    psi4.set_options(dict(basis="sto-3g", reference="rhf", scf_type="pk",
                          e_convergence=1e-11, d_convergence=1e-11))
    _, wfn = psi4.energy("hf", molecule=molecule, return_wfn=True)
    return wfn


def test_restricted_state_snapshot_matches_native_wavefunction(water_state):
    from psi4.driver.procrouting.isapol_native_response import native_restricted_state_from_wavefunction
    wfn = water_state
    state = native_restricted_state_from_wavefunction(wfn, caller_converged=True)
    assert (state.nbf, state.nmo, state.nocc, state.nvir, state.nov) == (7, 7, 5, 2, 10)
    np.testing.assert_array_equal(state.orbitals(), wfn.Ca())
    np.testing.assert_array_equal(state.energies(), wfn.epsilon_a())


def test_state_basis_snapshot_excludes_unbounded_molecule_metadata(water_state):
    from psi4.driver.procrouting.isapol_native_response import native_restricted_state_from_wavefunction
    water_state.basisset().molecule().set_comment("irrelevant metadata "*1024)
    state = native_restricted_state_from_wavefunction(water_state, caller_converged=True)
    basis = state.basis_snapshot()
    assert basis.molecule().comment() == ""
    np.testing.assert_array_equal(basis.molecule().geometry(), water_state.molecule().geometry())
    # Returned basis/geometry is a copy, not the context's retained object.
    displaced = np.asarray(basis.molecule().geometry()).copy()
    displaced[0, 0] += 1.
    basis.molecule().set_geometry(psi4.core.Matrix.from_array(displaced))
    np.testing.assert_array_equal(state.basis_snapshot().molecule().geometry(),
                                   water_state.molecule().geometry())


def test_state_snapshot_excludes_caller_matrix_and_vector_names(water_state):
    from psi4.driver.procrouting.isapol_native_response import native_restricted_state_from_wavefunction
    marker = "irrelevant caller array metadata "*1024
    for value in (water_state.Ca(), water_state.epsilon_a()):
        value.name = marker
    state = native_restricted_state_from_wavefunction(water_state, caller_converged=True)
    for value in (state.orbitals(), state.energies()):
        assert len(value.name) < 128


def test_snapshot_budget_boundary_and_owned_arrays(water_state):
    from psi4.driver.procrouting.isapol_native_response import native_restricted_state_from_wavefunction as make
    state = make(water_state, caller_converged=True)
    make(water_state, caller_converged=True, max_bytes=state.planned_bytes)
    with pytest.raises(ValueError, match="byte resource"):
        make(water_state, caller_converged=True, max_bytes=state.planned_bytes-1)
    original = [np.asarray(getter()).copy() for getter in
                (state.orbitals, state.energies)]
    for getter in (state.orbitals, state.energies):
        np.asarray(getter())[:] = 0.
    np.asarray(water_state.Ca())[:] = 0.
    np.asarray(water_state.epsilon_a())[:] = 0.
    for getter, expected in zip((state.orbitals, state.energies), original):
        np.testing.assert_array_equal(getter(), expected)


@pytest.mark.parametrize("kwargs", [
    {"caller_converged": False}, {"caller_converged": 1},
    {"caller_converged": True, "max_bytes": True},
    {"caller_converged": True, "max_bytes": 1.5},
    {"caller_converged": True, "max_bytes": 0},
])
def test_snapshot_python_declaration_refusals(water_state, kwargs):
    from psi4.driver.procrouting.isapol_native_response import native_restricted_state_from_wavefunction
    with pytest.raises(ValueError):
        native_restricted_state_from_wavefunction(water_state, **kwargs)


@pytest.mark.parametrize("fault, message", [
    ("energy", "wavefunction energy"), ("density", "density inconsistent"),
    ("gap", "gaps"), ("overlap", "orthonormal"),
])
def test_snapshot_retains_native_scientific_checks(water_state, fault, message):
    from psi4.driver.procrouting.isapol_native_response import native_restricted_state_from_wavefunction
    if fault == "energy":
        water_state.set_energy(float("nan"))
    elif fault == "density":
        np.asarray(water_state.Da())[0, 0] += .1
    elif fault == "gap":
        for eps in (water_state.epsilon_a(), water_state.epsilon_b()):
            np.asarray(eps)[water_state.nalpha()] = np.asarray(eps)[0]-1.
    elif fault == "overlap":
        # Virtual-column scaling leaves occupied density exactly unchanged.
        # Alpha/beta storage can alias: assign one computed value to both.
        changed = np.asarray(water_state.Ca()).copy()
        changed[:, water_state.nalpha()] *= 1.01
        np.asarray(water_state.Ca())[:] = changed
        np.asarray(water_state.Cb())[:] = changed
    with pytest.raises(ValueError, match=message):
        native_restricted_state_from_wavefunction(water_state, caller_converged=True)


@pytest.mark.parametrize("scale, accepted", [(1e-9, True), (1e-8, False)])
def test_snapshot_overlap_gate_brackets_its_tolerance(water_state, scale, accepted):
    """The native AO overlap must be accurate well inside the 1e-8 orthonormality gate.

    Scaling one virtual column by 1+s shifts its Gram diagonal by about 2s, so a
    2e-9 shift passes and a 2e-8 shift fails only if S is exact to ~1e-9.
    """
    from psi4.driver.procrouting.isapol_native_response import native_restricted_state_from_wavefunction
    changed = np.asarray(water_state.Ca()).copy()
    changed[:, water_state.nalpha()] *= 1+scale
    np.asarray(water_state.Ca())[:] = changed
    np.asarray(water_state.Cb())[:] = changed
    if accepted:
        native_restricted_state_from_wavefunction(water_state, caller_converged=True)
    else:
        with pytest.raises(ValueError, match="orthonormal"):
            native_restricted_state_from_wavefunction(water_state, caller_converged=True)
