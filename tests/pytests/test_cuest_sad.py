"""Atomic SAD GPU routing, fractional occupations, and explicit CPU controls."""

import numpy as np
import pytest

import psi4
from addons import uusing

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]


def _sad(tmp_path, use_cuest, cuest_sad=True, fractional=True, sad_type="DF",
         geometry="O 0 0 0\nH 0 0 1\nH 0 1 0", puream=True, extra_options=None):
    psi4.core.clean_options()
    psi4.set_options({
        "basis": "cc-pvdz", "use_cuest": use_cuest, "cuest_sad": cuest_sad,
        "cuest_mixed_precision": False, "sad_frac_occ": fractional,
        "sad_spin_average": False,
        "sad_scf_type": sad_type, "sad_e_convergence": 1.e-10,
        "sad_d_convergence": 1.e-8, "puream": puream,
    })
    if extra_options:
        psi4.set_options(extra_options)
    psi4.core.prepare_options_for_module("SCF")
    mol = psi4.geometry(geometry + "\nsymmetry c1\nno_reorient\nno_com")
    primary = psi4.core.BasisSet.build(mol, "ORBITAL", "cc-pvdz", puream=puream)
    atoms = psi4.core.BasisSet.build(mol, "ORBITAL", "cc-pvdz",
                                    puream=puream, return_atomlist=True)
    # The production factory always forces spherical SAD fitting functions.
    # Explicit global PUREAM overrides BasisSet.build's keyword. Mirror
    # proc._set_sad_basissets even when the orbital basis is Cartesian.
    from psi4.driver import p4util
    with p4util.OptionsStateCM(["PUREAM"]):
        psi4.core.set_global_option("PUREAM", True)
        fits = psi4.core.BasisSet.build(mol, "DF_BASIS_SAD", "SAD-FIT",
                                       puream=True, return_atomlist=True)
    assert all(b.has_puream() for b in fits)
    assert bool(psi4.core.get_global_option("PUREAM")) == puream
    sad = psi4.core.SADGuess.build_SAD(primary, atoms)
    sad.set_atomic_fit_bases(fits)
    sad.set_print(1)
    output = tmp_path / f"sad-{use_cuest}-{cuest_sad}-{sad_type}.out"
    psi4.core.set_output_file(str(output), False)
    before = psi4.core.get_option("SCF", "INTS_TOLERANCE")
    changed = psi4.core.has_option_changed("SCF", "INTS_TOLERANCE")
    local_before = psi4.core.get_local_option("SCF", "INTS_TOLERANCE")
    local_changed = psi4.core.has_local_option_changed("SCF", "INTS_TOLERANCE")
    sad.compute_guess()
    assert psi4.core.get_option("SCF", "INTS_TOLERANCE") == before
    assert psi4.core.has_option_changed("SCF", "INTS_TOLERANCE") == changed
    assert psi4.core.get_local_option("SCF", "INTS_TOLERANCE") == local_before
    assert psi4.core.has_local_option_changed("SCF", "INTS_TOLERANCE") == local_changed
    densities = [np.array(sad.Da()), np.array(sad.Db())]
    psi4.core.flush_outfile()
    text = output.read_text()
    assert np.isfinite(densities).all()
    return densities, text


@pytest.mark.parametrize("sad_type", ["DF", "DIRECT"])
@pytest.mark.parametrize("puream", [True, False])
def test_sad_cpu_control(tmp_path, sad_type, puream):
    """CUEST_SAD has no effect unless USE_CUEST is enabled."""
    reference, _ = _sad(tmp_path, False, False, sad_type=sad_type, puream=puream)
    actual, text = _sad(tmp_path, False, True, sad_type=sad_type, puream=puream)
    for ref, got in zip(reference, actual):
        np.testing.assert_allclose(got, ref, atol=1.e-12, rtol=0)
    assert "SAD J/K backend:" in text
    assert "SAD J/K backend: cuESTJK" not in text


def test_sad_preserves_shadowed_local_tolerance(tmp_path):
    _sad(tmp_path, False, extra_options={"ints_tolerance": 1.e-9})


def test_failed_sad_restores_native_timers():
    """An atomic error must not leave the process-wide skip flag set."""
    psi4.core.prepare_options_for_module("SCF")
    mol = psi4.geometry("0 3\nO 0 0 0\nsymmetry c1")
    primary = psi4.core.BasisSet.build(mol, "ORBITAL", "cc-pvdz")
    # Deliberately too few atomic functions: hits the explicit electron-count
    # check before any integral/JK work, inside native SAD timer suppression.
    sad = psi4.core.SADGuess.build_SAD(primary, [psi4.core.BasisSet.zero_ao_basis_set()])
    with pytest.raises(RuntimeError, match="more electrons than basis functions"):
        sad.compute_guess()
    psi4.core.timer_on("after failed SAD")
    psi4.core.timer_off("after failed SAD")
    records = list(psi4.core.get_timer_records().values())
    sentinel = [rec for rec in records if rec["timer_name"] == "after failed SAD"]
    assert len(sentinel) == 1
    assert sentinel[0]["n_calls"] == 1
    assert sentinel[0]["parent_id"] is None


@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.cuest
@pytest.mark.parametrize("fractional", [True, False])
@pytest.mark.parametrize("puream", [True, False])
def test_sad_gpu_density_matches_cpu(tmp_path, fractional, puream):
    """O covers fractional p shells; H covers the empty-beta integer case."""
    reference, _ = _sad(tmp_path, False, fractional=fractional, puream=puream)
    actual, text = _sad(tmp_path, True, fractional=fractional, puream=puream)
    for ref, got in zip(reference, actual):
        np.testing.assert_allclose(got, ref, atol=2.e-7, rtol=0)
    assert "SAD J/K backend: cuESTJK" in text


@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.cuest
def test_sad_gpu_opt_out_and_direct(tmp_path):
    for sad_type in ("DF", "DIRECT"):
        reference, _ = _sad(tmp_path, False, sad_type=sad_type)
        actual, text = _sad(tmp_path, True, cuest_sad=(sad_type == "DIRECT"), sad_type=sad_type)
        for ref, got in zip(reference, actual):
            np.testing.assert_allclose(got, ref, atol=2.e-7, rtol=0)
        assert "SAD J/K backend: cuESTJK" not in text


@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.cuest
def test_sad_gpu_ghost_basis(tmp_path):
    geometry = "0 1\nO 0 0 0\nH 0 0 1\nH 0 1 0\n@O 0 0 4"
    reference, _ = _sad(tmp_path, False, geometry=geometry)
    actual, text = _sad(tmp_path, True, geometry=geometry)
    for ref, got in zip(reference, actual):
        np.testing.assert_allclose(got, ref, atol=2.e-7, rtol=0)
    # Only O and H are unique occupied atoms; the ghost needs no atomic SCF.
    assert text.count("SAD J/K backend: cuESTJK") == 2


@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.cuest
@pytest.mark.parametrize("options", [
    {"screening": "CSAM"}, {"screening": "NONE"},
    {"ints_tolerance": 0.0}, {"df_fitting_condition": 1.e-8},
])
def test_sad_explicit_cpu_contract(tmp_path, options):
    reference, _ = _sad(tmp_path, False, extra_options=options)
    actual, text = _sad(tmp_path, True, extra_options=options)
    for ref, got in zip(reference, actual):
        np.testing.assert_allclose(got, ref, atol=2.e-7, rtol=0)
    assert "SAD cuEST ineligible" in text
    assert "SAD J/K backend: cuESTJK" not in text


@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.cuest
def test_cuest_empty_occupied_rebuild(tmp_path):
    """Reusing a K output with zero columns must erase the previous answer."""
    psi4.set_options({"use_cuest": True, "cuest_mixed_precision": False, "scf_type": "df"})
    mol = psi4.geometry("He 0 0 0\nsymmetry c1")
    primary = psi4.core.BasisSet.build(mol, "ORBITAL", "cc-pvdz")
    aux = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", "", "JKFIT", "cc-pvdz")
    jk = psi4.core.JK.build_JK(primary, aux)
    jk.initialize()
    try:
        answers = []
        for ncol in (1, 0, 1):
            jk.C_clear()
            coeffs = np.full((primary.nbf(), ncol), 0.1)
            jk.C_left_add(psi4.core.Matrix.from_array(coeffs))
            jk.compute()
            answers.append(np.array(jk.K()[0]))
        assert np.max(np.abs(answers[0])) > 1.e-6
        np.testing.assert_array_equal(answers[1], 0.0)
        np.testing.assert_allclose(answers[2], answers[0], atol=1.e-12, rtol=0)
    finally:
        jk.finalize()
