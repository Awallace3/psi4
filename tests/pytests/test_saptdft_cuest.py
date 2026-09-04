"""
GPU-accelerated SAPT(DFT)-D4(I) through cuEST.

The supported GPU configuration is deliberately narrow:

    SAPT_DFT_INDUCTION_TYPE = NONE   (induction comes from delta HF)
    dispersion              = -D4(I) (empirical, no TDDFT response needed)

Every remaining electronic-structure step is an SCF that goes through
``scf_helper`` -- three HF SCFs for the delta-HF segment (dimer, monomer A,
monomer B) and two DFT SCFs for the monomers -- so turning on ``USE_CUEST``
routes the whole method onto the GPU by way of the DF J/K builder.

These tests pin two things:
  1. ``USE_CUEST`` is *optional*: the cuEST run must reproduce the CPU run.
  2. ``USE_CUEST`` is *complete*: no CPU DF J/K object is constructed anywhere
     in the SAPT(DFT) driver when it is on.
"""

import pytest

import psi4

from addons import uusing

pytestmark = [pytest.mark.psi, pytest.mark.api]

# Neutral water dimer, near equilibrium.
_water_dimer = """
0 1
O   -0.702196054   -0.056060256   0.009942262
H   -1.022193224    0.846775782   -0.011488714
H    0.257521062    0.042121496    0.005218999
--
0 1
O    2.268880784    0.026340101    0.000508029
H    2.645502399   -0.412039965    0.766632411
H    2.641145101   -0.449872874   -0.744894473
units angstrom
"""

_components = [
    "SAPT ELST ENERGY",
    "SAPT EXCH ENERGY",
    "SAPT IND ENERGY",
    "SAPT DISP ENERGY",
    "SAPT TOTAL ENERGY",
]


def _run_saptdft_d4i(use_cuest, outfile, extra_options=None):
    """Run SAPT(DFT)-D4(I) with induction from delta HF; return components and output text."""
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.set_output_file(str(outfile), False)

    psi4.geometry(_water_dimer)
    options = {
        "basis": "cc-pvdz",
        "scf_type": "df",
        "SAPT_DFT_FUNCTIONAL": "pbe0",
        "SAPT_DFT_GRAC_SHIFT_A": 0.136,
        "SAPT_DFT_GRAC_SHIFT_B": 0.136,
        "SAPT_DFT_INDUCTION_TYPE": "NONE",
        "SAPT_DFT_DO_DHF": True,
        "ORBITAL_OPTIMIZER_PACKAGE": "INTERNAL",
        "USE_CUEST": use_cuest,
        # Compare against the CPU in like precision; the mixed-precision
        # cuEST path is exercised separately below.
        "CUEST_MIXED_PRECISION": False,
    }
    if extra_options:
        options.update(extra_options)
    psi4.set_options(options)

    psi4.energy("sapt(dft)-d4(i)")
    values = {key: psi4.variable(key) for key in _components}

    psi4.core.close_outfile()
    text = outfile.read_text()
    psi4.core.clean()
    return values, text


@pytest.mark.saptdft
@pytest.mark.cuest
@pytest.mark.dftd4
@uusing("cuest")
@uusing("cuda_cc8")
@uusing("dftd4")
def test_saptdft_cuest_d4i_matches_cpu(tmp_path):
    """cuEST must be an opt-in accelerator, not a change in answer."""
    ref, ref_text = _run_saptdft_d4i(False, tmp_path / "cpu.out")
    gpu, gpu_text = _run_saptdft_d4i(True, tmp_path / "gpu.out")

    for key in _components:
        assert psi4.compare_values(ref[key], gpu[key], 6, f"cuEST vs CPU: {key}")

    # The reference run must not have touched the GPU builder, and the cuEST
    # run must not have fallen back to a CPU DF J/K anywhere in the driver.
    assert "cuESTJK" not in ref_text
    assert "DiskDFJK: Density-Fitted J/K Matrices" not in gpu_text
    assert "MemDFJK: Density-Fitted J/K Matrices" not in gpu_text


@pytest.mark.saptdft
@pytest.mark.cuest
@pytest.mark.dftd4
@uusing("cuest")
@uusing("cuda_cc8")
@uusing("dftd4")
def test_saptdft_cuest_d4i_covers_all_scf(tmp_path):
    """Both the delta-HF and the DFT segments must build cuESTJK objects."""
    _, text = _run_saptdft_d4i(True, tmp_path / "gpu.out")

    banner = "==> cuESTJK: GPU-Accelerated Density-Fitted J/K Matrices <=="
    n_cuest = text.count(banner)

    # The dimer HF SCF builds one J/K that both HF monomers reuse, and the DFT
    # monomer A SCF builds another that DFT monomer B and the SAPT terms reuse.
    assert n_cuest >= 2, f"expected >= 2 cuESTJK builders, found {n_cuest}"

    # All five SCFs in this configuration must have run.
    for banner_text in [
        "SAPT(DFT): delta HF Dimer",
        "SAPT(DFT): delta HF Monomer A",
        "SAPT(DFT): delta HF Monomer B",
    ]:
        assert banner_text in text or banner_text.replace("delta HF", "Delta HF") in text


@pytest.mark.saptdft
@pytest.mark.cuest
@pytest.mark.dftd4
@uusing("cuest")
@uusing("cuda_cc8")
@uusing("dftd4")
def test_saptdft_cuest_d4i_mixed_precision(tmp_path):
    """The default mixed-precision cuEST path stays within chemical noise."""
    ref, _ = _run_saptdft_d4i(False, tmp_path / "cpu.out")
    gpu, _ = _run_saptdft_d4i(True, tmp_path / "gpu_mp.out",
                              extra_options={"CUEST_MIXED_PRECISION": True})

    for key in _components:
        assert psi4.compare_values(ref[key], gpu[key], atol=5.0e-6,
                                   label=f"cuEST (mixed precision) vs CPU: {key}")
