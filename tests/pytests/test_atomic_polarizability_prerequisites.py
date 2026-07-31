import math

import pytest

import psi4


pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.mints]


def _valid_prerequisites(**changes):
    values = {
        "chf_exchange": 0.25,
        "alda_kernel": 0.75,
        "neutral_energy": -75.0,
        "cation_energy": -74.4,
        "homo_energy": -0.3,
        "ionization_potential": 0.6,
        "grac_shift": 0.3,
        "method_fingerprint": "RKS",
        "functional_fingerprint": "PBE0-GRAC",
        "basis_fingerprint": "sto-3g:synthetic",
        "grid_fingerprint": "grid:synthetic",
        "point_count": 2,
        "grid_dimension": 3,
        "site_count": 2,
        "points": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "quadrature_weights": [0.4, 0.6],
        "partition_weights": [0.2, 0.8, 0.7, 0.3],
    }
    values.update(changes)
    return values


def _validate(**changes):
    return psi4.core._atomic_polarizability_validate_response_prerequisites(
        **_valid_prerequisites(**changes)
    )


def test_response_prerequisites_accept_exact_consistent_metadata():
    result = _validate()

    assert result["kernel"] == pytest.approx((0.25, 0.75))
    assert result["grac"] == pytest.approx((-75.0, -74.4, -0.3, 0.6, 0.3))
    assert result["fingerprints"] == (
        "RKS",
        "PBE0-GRAC",
        "sto-3g:synthetic",
        "grid:synthetic",
    )
    assert result["isa_dimensions"] == (2, 3, 2)
    assert result["isa_data_sizes"] == (6, 2, 4)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"chf_exchange": 0.20}, "CHF exchange coefficient.*0.25"),
        ({"alda_kernel": 0.80}, "ALDA coefficient.*0.75"),
        ({"neutral_energy": math.inf}, "GRAC.*finite"),
        ({"method_fingerprint": ""}, "GRAC.*fingerprints"),
        ({"ionization_potential": 0.7}, "ionization potential.*cation.*neutral"),
        ({"grac_shift": 0.2}, "GRAC shift.*ionization potential.*HOMO"),
        ({"points": [0.0] * 5}, "ISA.*point coordinates.*dimensions"),
        ({"quadrature_weights": [1.0]}, "ISA.*quadrature weights.*dimensions"),
        ({"partition_weights": [0.5] * 3}, "ISA.*partition weights.*dimensions"),
        ({"grid_dimension": 2, "points": [0.0] * 4}, "ISA.*grid dimension.*3"),
        ({"site_count": 0, "partition_weights": []}, "ISA.*site count"),
        ({"partition_weights": [0.2, 0.7, 0.7, 0.3]}, "ISA.*partition unity"),
        ({"points": [0.0, 0.0, math.nan, 1.0, 0.0, 0.0]}, "ISA.*finite"),
        ({"quadrature_weights": [0.4, math.inf]}, "ISA.*finite"),
        ({"partition_weights": [0.2, 0.8, math.nan, math.nan]}, "ISA.*finite"),
    ],
)
def test_response_prerequisites_reject_invalid_metadata(changes, message):
    with pytest.raises(RuntimeError, match=message):
        _validate(**changes)


def test_calculator_names_missing_grac_and_isa_without_publishing_arrays():
    molecule = psi4.geometry(
        """
        H 0.0 0.0 0.0
        H 0.0 0.0 0.7
        symmetry c1
        units angstrom
        """
    )
    psi4.set_options({"basis": "sto-3g", "scf_type": "pk"})
    _, wfn = psi4.energy("scf", return_wfn=True)

    calculator = psi4.core.AtomicPolarizabilityCalculator(wfn)
    with pytest.raises(RuntimeError, match=r"missing GRAC provenance.*ISA weights"):
        calculator.compute()

    public_arrays = (
        "ATOMIC POLARIZABILITIES",
        "ATOMIC DYNAMIC POLARIZABILITIES",
        "ATOMIC POLARIZABILITY FREQUENCIES",
        "ATOMIC C6",
        "ATOMIC C8",
        "ATOMIC C10",
        "ATOMIC C12",
    )
    assert all(not wfn.has_array_variable(name) for name in public_arrays)
    assert all(not psi4.core.has_array_variable(name) for name in public_arrays)
