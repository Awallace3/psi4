import pytest
import psi4
from pprint import pprint as pp


def test_sapt_e20disp():
    mol = psi4.geometry(
        """
0 1
O  -1.551007  -0.114520   0.000000
H  -1.934259   0.762503   0.000000
H  -0.599677   0.040712   0.000000
--
0 1
O   1.350625   0.111469   0.000000
H   1.680398  -0.373741  -0.758561
H   1.680398  -0.373741   0.758561

units angstrom
no_reorient
symmetry c1
    """
    )
    psi4.set_options(
        {
            "basis": "cc-pvtz",
            "SAPT0_E20DISP": True,
        }
    )
    psi4.energy("sapt0", molecule=mol)
    disp20 = psi4.core.variable("SAPT DISP20 ENERGY")
    exch_disp20 = psi4.core.variable("SAPT EXCH-DISP20 ENERGY")
    assert psi4.compare_values(-0.0035461568684353913, disp20, 12, "Disp20 energy")
    assert psi4.compare_values(
        0.000685344786411173, exch_disp20, 12, "Exch-Disp20 energy"
    )


def test_sapt_dispersion_correction():
    mol = psi4.geometry(
        """
0 1
O  -1.551007  -0.114520   0.000000
H  -1.934259   0.762503   0.000000
H  -0.599677   0.040712   0.000000
--
0 1
O   1.350625   0.111469   0.000000
H   1.680398  -0.373741  -0.758561
H   1.680398  -0.373741   0.758561

units angstrom
no_reorient
symmetry c1
    """
    )
    psi4.set_options(
        {
            "basis": "cc-pvdz",
            "SAPT0_E20DISP": True,
        }
    )
    psi4.energy("sapt0", molecule=mol)
    adz_disp20 = psi4.core.variable("SAPT DISP20 ENERGY")
    adz_exch_disp20 = psi4.core.variable("SAPT EXCH-DISP20 ENERGY")
    psi4.set_options(
        {
            "basis": "cc-pvtz",
            "SAPT0_E20DISP": True,
        }
    )
    psi4.energy("sapt0", molecule=mol)
    atz_disp20 = psi4.core.variable("SAPT DISP20 ENERGY")
    atz_exch_disp20 = psi4.core.variable("SAPT EXCH-DISP20 ENERGY")
    delta_dispersion_correction = (adz_disp20 + adz_exch_disp20) - (
        atz_disp20 + atz_exch_disp20
    )
    print(delta_dispersion_correction)
    assert psi4.compare_values(
        0.001029396729803937, delta_dispersion_correction, 12, "Dispersion correction"
    )

def test_sapt_dispersion_correction_hlsapt():
    mol = psi4.geometry(
        """
0 1
O  -1.551007  -0.114520   0.000000
H  -1.934259   0.762503   0.000000
H  -0.599677   0.040712   0.000000
--
0 1
O   1.350625   0.111469   0.000000
H   1.680398  -0.373741  -0.758561
H   1.680398  -0.373741   0.758561

units angstrom
no_reorient
symmetry c1
    """
    )
    psi4.set_options(
        {
            "basis": "cc-pvdz",
            "SAPT0_E20DISP": True,
        }
    )
    psi4.energy("sapt0", molecule=mol)
    adz_disp20 = psi4.core.variable("SAPT DISP20 ENERGY")
    adz_exch_disp20 = psi4.core.variable("SAPT EXCH-DISP20 ENERGY")
    psi4.set_options(
        {
            "basis": "cc-pvtz",
            "SAPT0_E20DISP": True,
        }
    )
    psi4.energy("sapt0", molecule=mol)
    atz_disp20 = psi4.core.variable("SAPT DISP20 ENERGY")
    atz_exch_disp20 = psi4.core.variable("SAPT EXCH-DISP20 ENERGY")
    delta_dispersion_correction = (adz_disp20 + adz_exch_disp20) - (
        atz_disp20 + atz_exch_disp20
    )
    print(delta_dispersion_correction)
    assert psi4.compare_values(
        0.001029396729803937, delta_dispersion_correction, 12, "Dispersion correction"
    )


if __name__ == "__main__":
    test_sapt_dispersion_correction_hlsapt()
