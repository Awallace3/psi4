import time

import pytest
import psi4
from psi4 import compare_values


pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.fsapt, pytest.mark.long]


def _run_fisapt0(use_einsums: bool):
    psi4.core.clean()

    mol = psi4.geometry(
        """
0 1
C   11.54100       27.68600       13.69600
H   12.45900       27.15000       13.44600
C   10.79000       27.96500       12.40600
H   10.55700       27.01400       11.92400
H   9.879000       28.51400       12.64300
H   11.44300       28.56800       11.76200
H   10.90337       27.06487       14.34224
H   11.78789       28.62476       14.21347
--
0 1
C   10.60200       24.81800       6.466000
O   10.95600       23.84000       7.103000
N   10.17800       25.94300       7.070000
C   10.09100       26.25600       8.476000
C   9.372000       27.59000       8.640000
C   11.44600       26.35600       9.091000
C   9.333000       25.25000       9.282000
H   9.874000       26.68900       6.497000
H   9.908000       28.37100       8.093000
H   8.364000       27.46400       8.233000
H   9.317000       27.84600       9.706000
H   9.807000       24.28200       9.160000
H   9.371000       25.57400       10.32900
H   8.328000       25.26700       8.900000
H   11.28800       26.57600       10.14400
H   11.97000       27.14900       8.585000
H   11.93200       25.39300       8.957000
H   10.61998       24.85900       5.366911
units angstrom

symmetry c1
no_reorient
no_com
"""
    )

    psi4.set_options(
        {
            "basis": "jun-cc-pvdz",
            "scf_type": "df",
            "guess": "sad",
            "freeze_core": "true",
            "FISAPT_USE_EINSUMS": use_einsums,
            "FISAPT_FSAPT_FILEPATH": "none",
        }
    )

    t0 = time.perf_counter()
    e_total = psi4.energy("fisapt0", molecule=mol)
    dt = time.perf_counter() - t0

    return e_total, dt


def test_fisapt0_runtime_einsums_vs_baseline():
    e_ref, t_ref = _run_fisapt0(False)
    e_ein, t_ein = _run_fisapt0(True)

    compare_values(e_ref, e_ein, 7, "fisapt0 total energy")

    assert t_ref > 0.0
    assert t_ein > 0.0

    ratio = t_ein / t_ref
    psi4.core.print_out(
        f"\nFISAPT0 runtime comparison (s): baseline={t_ref:.6f}, einsums={t_ein:.6f}, ratio={ratio:.3f}\n"
    )


if __name__ == "__main__":
    test_fisapt0_runtime_einsums_vs_baseline()
