"""Real cuEST plan lifecycle: count actual creations as well as checking J/K."""
import re

import numpy as np
import pytest
import psi4

from addons import using

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick, pytest.mark.cuest,
              *using("cuest"), *using("cuda_cc8")]


def _setup(tmp_path):
    psi4.core.clean()
    psi4.core.clean_options()
    path = tmp_path / "plan-trace.out"
    psi4.core.set_output_file(str(path), False)
    mol = psi4.geometry("""
        O 0 0 0
        H 0 1.4 1.0
        H 0 -1.4 1.0
        units bohr
        symmetry c1
        no_reorient
        no_com
    """)
    psi4.set_options({"scf_type": "df", "use_cuest": True,
                      "cuest_mixed_precision": False})
    return mol, path


def _builder(mol, basis="cc-pvdz", aux_basis="cc-pvdz-jkfit"):
    primary = psi4.core.BasisSet.build(mol, "ORBITAL", basis)
    aux = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", aux_basis, "JKFIT", basis)
    jk = psi4.core.JK.build_JK(primary, aux)
    assert jk.name() == "cuESTJK"
    jk.set_print(3)
    jk.set_do_J(True)
    jk.set_do_K(True)
    jk.set_omega_alpha(1.0)
    jk.set_omega_beta(0.0)
    return jk, primary, aux


def _creations(path):
    psi4.core.flush_outfile()
    text = path.read_text()
    traced = re.findall(r"cuESTJK plan created #\d+", text)
    # Baseline has no counter yet, but prints its workspace banner only after
    # construction. This lets the unchanged-reinitialize test go red on the
    # baseline's *second* creation, not just on absent new telemetry.
    return len(traced) if traced else text.count("Device memory, DF int plan:")


def _compute(jk, nbf, ranks=(1, 4, 0, 2)):
    rng = np.random.default_rng(1012)
    jk.C_clear()
    for rank in ranks:
        # Signed, unequal factors produce nonsymmetric, indefinite densities.
        jk.C_left_add(psi4.core.Matrix.from_array(rng.normal(size=(nbf, rank)) / np.sqrt(nbf)))
        jk.C_right_add(psi4.core.Matrix.from_array(-rng.normal(size=(nbf, rank)) / np.sqrt(nbf)))
    jk.compute()
    return [m.np.copy() for m in jk.J()], [m.np.copy() for m in jk.K()]


def _equal(a, b, atol=1.e-8):
    assert len(a[0]) == len(b[0])
    assert len(a[1]) == len(b[1])
    for aa, bb in zip(a[0] + a[1], b[0] + b[1]):
        np.testing.assert_allclose(aa, bb, atol=atol, rtol=0)


def test_cuest_unchanged_initialize_reuses_plan(tmp_path):
    mol, trace = _setup(tmp_path)
    jk, primary, aux = _builder(mol)
    try:
        jk.initialize()
        assert _creations(trace) == 1
        reference = _compute(jk, primary.nbf())
        for ranks in ((2,), (1, 4, 0, 2), (0,), (3, 1)):
            jk.initialize()
            assert _creations(trace) == 1  # matrix equality alone misses always-rebuild
            got = _compute(jk, primary.nbf(), ranks)
            if ranks == (1, 4, 0, 2):
                _equal(reference, got)
                np.testing.assert_array_equal(got[1][2], 0)
                assert np.max(np.abs(got[1][0] - got[1][0].T)) > 1.e-3
        psi4.set_options({"use_cuest": False})
        cpu = psi4.core.JK.build_JK(primary, aux)
        try:
            cpu.initialize()
            _equal(reference, _compute(cpu, primary.nbf()))
        finally:
            cpu.finalize()
    finally:
        jk.finalize()


@pytest.mark.parametrize("change", ["alpha", "beta", "omega", "precision", "slices", "moduli",
                                   "cutoff", "threshold_option", "fitting_condition"])
def test_cuest_changed_dependencies_rebuild(tmp_path, change):
    mol, trace = _setup(tmp_path)
    jk, primary, _ = _builder(mol)
    try:
        jk.initialize()
        reference = _compute(jk, primary.nbf())
        def change_fitting_condition():
            before = psi4.core.get_option("SCF", "DF_FITTING_CONDITION")
            after = before * 10.0
            assert np.isfinite(before) and before > 0
            assert np.isfinite(after) and after > 0 and after != before
            psi4.set_options({"df_fitting_condition": after})
            assert psi4.core.get_option("SCF", "DF_FITTING_CONDITION") == after
            assert psi4.core.get_option("SCF", "DF_FITTING_CONDITION") != before

        actions = {
            "alpha": lambda: jk.set_omega_alpha(0.25),
            "beta": lambda: jk.set_omega_beta(0.2),
            "omega": lambda: jk.set_omega(0.3),
            "precision": lambda: psi4.set_options({"cuest_mixed_precision": True}),
            "slices": lambda: psi4.set_options({"cuest_dfk_slices": 6}),
            "moduli": lambda: psi4.set_options({"cuest_dfk_moduli": 7}),
            "cutoff": lambda: jk.set_cutoff(1.e-11),
            "threshold_option": lambda: psi4.set_options({"ints_tolerance": 1.e-11}),
            "fitting_condition": change_fitting_condition,
        }
        actions[change]()
        # Operator and cutoff setters must still rebuild immediately.
        if change in ("alpha", "beta", "omega", "cutoff"):
            assert _creations(trace) == 2
        jk.initialize()
        assert _creations(trace) == 2
        got = _compute(jk, primary.nbf())
        if change == "alpha":
            for ref, k in zip(reference[1], got[1]):
                np.testing.assert_allclose(k, 0.25 * ref, atol=1.e-8, rtol=0)
            # The real PBE0 -> SAPT bare-K transition must NOT reuse weighted K.
            jk.set_omega_alpha(1.0)
            assert _creations(trace) == 3
            _equal(reference, _compute(jk, primary.nbf()))
        else:
            # Fresh object under identical settings is an independent plan.
            fresh, p2, _ = _builder(mol)
            try:
                if change == "beta":
                    fresh.set_omega_beta(0.2)
                if change == "omega":
                    fresh.set_omega(0.3)
                if change == "cutoff":
                    fresh.set_cutoff(1.e-11)
                fresh.initialize()
                _equal(got, _compute(fresh, p2.nbf()), atol=1.e-7)
            finally:
                fresh.finalize()
    finally:
        jk.finalize()


def test_cuest_same_fitting_condition_reuses_plan(tmp_path):
    mol, trace = _setup(tmp_path)
    jk, primary, _ = _builder(mol)
    try:
        jk.initialize()
        assert _creations(trace) == 1
        reference = _compute(jk, primary.nbf())
        before = psi4.core.get_option("SCF", "DF_FITTING_CONDITION")
        assert np.isfinite(before) and before > 0
        psi4.set_options({"df_fitting_condition": before})
        assert psi4.core.get_option("SCF", "DF_FITTING_CONDITION") == before
        jk.initialize()
        assert _creations(trace) == 1
        _equal(reference, _compute(jk, primary.nbf()))
        assert _creations(trace) == 1
    finally:
        jk.finalize()


def test_cuest_failed_rebuild_finalize_and_interleaved_objects(tmp_path):
    mol, trace = _setup(tmp_path)
    a, primary, _ = _builder(mol)
    b, p2, _ = _builder(mol)
    try:
        a.initialize()
        ref = _compute(a, primary.nbf())
        b.initialize()
        assert _creations(trace) == 2
        psi4.set_options({"cuest_mixed_precision": True})
        b.initialize()
        psi4.set_options({"cuest_mixed_precision": False})
        # Shared handle mode restored even when A's plan is reused.
        a.initialize()
        assert _creations(trace) == 3
        _equal(ref, _compute(a, primary.nbf()))
        # Deterministic, safe failure, not an OOM/driver fault on a shared GPU.
        a.set_omega(float("nan"))
    except RuntimeError as exc:
        assert "finite" in str(exc)
        with pytest.raises(RuntimeError, match="not initialized"):
            _compute(a, primary.nbf())
        a.set_omega(0.0)
        a.initialize()
        assert _creations(trace) == 4
        _equal(ref, _compute(a, primary.nbf()))
        a.finalize()
        a.finalize()
        with pytest.raises(RuntimeError, match="not initialized"):
            _compute(a, primary.nbf())
        a.initialize()
        assert _creations(trace) == 5
        _equal(ref, _compute(a, primary.nbf()))
    else:
        pytest.fail("nonfinite rebuild should fail closed")
    finally:
        a.finalize()
        b.finalize()


def test_cuest_geometry_and_basis_lifecycle(tmp_path):
    mol, trace = _setup(tmp_path)
    jk, primary, aux = _builder(mol)
    try:
        jk.initialize()
        xyz = mol.geometry()
        xyz.np[1, 1] += 0.15
        mol.set_geometry(xyz)
        mol.update_geometry()
        jk.initialize()
        assert _creations(trace) == 2
        got = _compute(jk, primary.nbf())
        fresh, p2, _ = _builder(mol)
        try:
            fresh.initialize()
            _equal(got, _compute(fresh, p2.nbf()))
        finally:
            fresh.finalize()
        # No global cache: new primary/auxiliary lifetimes each own a new plan.
        for basis, fit in (("cc-pvtz", "cc-pvtz-jkfit"), ("cc-pvdz", "def2-universal-jkfit")):
            other, p3, _ = _builder(mol, basis, fit)
            try:
                n = _creations(trace)
                other.initialize()
                other.initialize()
                assert _creations(trace) == n + 1
                _compute(other, p3.nbf(), (1,))
            finally:
                other.finalize()
        # Ghost state is read from the live molecule, never pointer identity.
        mol.set_nuclear_charge(0, 0.0)
        jk.initialize()
        assert _creations(trace) == 6
        _equal(got, _compute(jk, primary.nbf()))
    finally:
        jk.finalize()


def test_cuest_independent_basis_displacement_fails_closed(tmp_path):
    mol, trace = _setup(tmp_path)
    jk, primary, _ = _builder(mol)
    try:
        jk.initialize()
        ref = _compute(jk, primary.nbf())
        primary.move_atom(0, psi4.core.Vector3(0.125, 0.0, 0.0))
        with pytest.raises(RuntimeError, match="independent BasisSet::move_atom"):
            jk.initialize()
        assert _creations(trace) == 1
        with pytest.raises(RuntimeError, match="not initialized"):
            _compute(jk, primary.nbf())
        primary.move_atom(0, psi4.core.Vector3(-0.125, 0.0, 0.0))
        jk.initialize()
        assert _creations(trace) == 2
        _equal(ref, _compute(jk, primary.nbf()))
    finally:
        jk.finalize()


def test_cuest_compute_detects_live_options_without_initialize(tmp_path):
    mol, trace = _setup(tmp_path)
    jk, primary, _ = _builder(mol)
    try:
        jk.initialize()
        _compute(jk, primary.nbf())
        psi4.set_options({"cuest_dfk_slices": 6})
        _compute(jk, primary.nbf())
        assert _creations(trace) == 2
        # The same explicit value is a hit, even after an initialize.
        jk.initialize()
        assert _creations(trace) == 2
    finally:
        jk.finalize()


def test_cuest_explicit_cutoff_takes_precedence_over_options(tmp_path):
    mol, trace = _setup(tmp_path)
    jk, _, _ = _builder(mol)
    try:
        jk.initialize()
        jk.set_cutoff(1.e-11)
        assert _creations(trace) == 2
        psi4.set_options({"ints_tolerance": 1.e-9})
        jk.initialize()
        assert _creations(trace) == 2
    finally:
        jk.finalize()


def test_cuest_non_c1_frame_change_requires_new_basis_lifecycle(tmp_path):
    mol, trace = _setup(tmp_path)
    mol.reset_point_group("c2v")
    mol.update_geometry()
    jk, _, _ = _builder(mol)
    try:
        jk.initialize()
        jk.initialize()
        assert _creations(trace) == 1
        xyz = mol.geometry()
        xyz.np[0, 2] += 0.125  # retain C2v but change the fixed SO frame
        mol.set_geometry(xyz)
        mol.update_geometry()
        with pytest.raises(RuntimeError, match="non-C1 geometry changed"):
            jk.initialize()
        assert _creations(trace) == 1
    finally:
        jk.finalize()
