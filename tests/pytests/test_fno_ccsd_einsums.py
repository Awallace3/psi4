import pytest
import psi4


def test_fno_ccsd_einsums():
    mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
    """)
    psi4.set_options(
        {
            "basis": "cc-pvdz",
            "df_basis_scf": "cc-pvdz-jkfit",
            "df_basis_cc": "cc-pvdz-ri",
            "scf_type": "df",
            "guess": "gwh",
            "freeze_core": True,
            "cc_type": "df",
            "nat_orbs": False,
            "e_convergence": 1e-10,
            "d_convergence": 1e-10,
            "r_convergence": 1e-10,
        }
    )
    energy = psi4.energy("fno-ccsd(t)-ein", molecule=mol)
    expected = -76.21578814042408
    psi4.compare_values(expected, energy, 8, "FNO-CCSD(T) Energy")


if __name__ == "__main__":
    test_fno_ccsd_einsums()
