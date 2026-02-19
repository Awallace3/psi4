import time

import pytest
import psi4
from psi4 import compare_values


pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.fsapt, pytest.mark.long]


def _run_fisapt0(use_einsums: bool):
    psi4.core.clean()

    mol = psi4.geometry(
        """0 1
H 0.029000000000 -1.119900000000 -1.524300000000
O 0.948100000000 -1.399000000000 -1.358700000000
H 1.437100000000 -0.558800000000 -1.309900000000
--
0 1
C 6.44536662 -0.26509169 -0.00000000
H 7.53536662 -0.26509169 -0.00000000
H 6.08203329 0.57399070 0.59332085
H 6.08203329 -0.17080196 -1.02332709
H 6.08203329 -1.19846381 0.43000624
symmetry c1
no_reorient
units angstrom
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
