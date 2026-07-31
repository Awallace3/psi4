import math

import pytest

import psi4


pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.mints, pytest.mark.scf]


@pytest.fixture(scope="module")
def response_wavefunctions():
    psi4.core.be_quiet()
    psi4.set_options({"basis": "sto-3g", "scf_type": "pk"})

    primary = psi4.geometry(
        """
        H 0.0 0.0 0.0
        H 0.0 0.0 0.7
        symmetry c1
        units angstrom
        """
    )
    _, primary_wfn = psi4.energy("pbe0", molecule=primary, return_wfn=True)

    other = psi4.geometry(
        """
        He 0.0 0.0 0.0
        symmetry c1
        """
    )
    _, other_wfn = psi4.energy("pbe0", molecule=other, return_wfn=True)
    return primary_wfn, other_wfn


def _valid_prerequisites(identity_wfn, **changes):
    values = {
        "identity_wfn": identity_wfn,
        "chf_exchange": 0.25,
        "alda_kernel": 0.75,
        "neutral_energy": -75.0,
        "cation_energy": -74.4,
        "homo_energy": -0.3,
        "ionization_potential": 0.6,
        "grac_shift": 0.3,
        "point_count": 2,
        "grid_dimension": 3,
        "site_count": 2,
        "points": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "quadrature_weights": [0.4, 0.6],
        "partition_weights": [0.2, 0.8, 0.7, 0.3],
        "blank_identity_field": "",
    }
    values.update(changes)
    return values


def _validate(identity_wfn, **changes):
    return psi4.core._atomic_polarizability_validate_response_prerequisites(
        **_valid_prerequisites(identity_wfn, **changes)
    )


def _provider(wfn, identity_wfn, isa_identity_wfn=None, **changes):
    args = _valid_prerequisites(identity_wfn, **changes)
    args.pop("identity_wfn")
    args.pop("blank_identity_field")
    return psi4.core._AtomicPolarizabilityTestResponseProvider(
        wfn,
        identity_wfn,
        identity_wfn if isa_identity_wfn is None else isa_identity_wfn,
        **args,
    )


def test_response_prerequisites_accept_exact_consistent_actual_identity(response_wavefunctions):
    wfn, _ = response_wavefunctions
    result = _validate(wfn)

    assert result["kernel"] == pytest.approx((0.25, 0.75))
    assert result["grac"] == pytest.approx((-75.0, -74.4, -0.3, 0.6, 0.3))
    assert result["molecule"] == (2, 0, 1)
    assert result["basis_dimensions"] == (2, 2, 2)
    assert result["electronic_identity"] == ("scf/RHF", "RHF", "PBE0")
    assert result["grid_fingerprint"]
    assert result["isa_dimensions"] == (2, 3, 2)
    assert result["isa_data_sizes"] == (6, 2, 4)


@pytest.mark.parametrize(
    "coefficient, value, message",
    [
        ("chf_exchange", math.nextafter(0.25, 0.0), "CHF exchange coefficient.*0.25"),
        ("chf_exchange", math.nextafter(0.25, 1.0), "CHF exchange coefficient.*0.25"),
        ("alda_kernel", math.nextafter(0.75, 0.0), "ALDA coefficient.*0.75"),
        ("alda_kernel", math.nextafter(0.75, 1.0), "ALDA coefficient.*0.75"),
    ],
)
def test_response_kernel_rejects_every_nonexact_nextafter(response_wavefunctions, coefficient, value, message):
    wfn, _ = response_wavefunctions
    with pytest.raises(RuntimeError, match=message):
        _validate(wfn, **{coefficient: value})


@pytest.mark.parametrize(
    "field",
    [
        "basis_name",
        "basis_fingerprint",
        "reference",
        "method",
        "functional",
        "functional_fingerprint",
        "grid_fingerprint",
        "grid_radial_scheme",
        "grid_spherical_scheme",
        "grid_nuclear_scheme",
        "grid_pruning_scheme",
        "grid_block_scheme",
    ],
)
def test_wavefunction_identity_rejects_every_blank_required_field(response_wavefunctions, field):
    wfn, _ = response_wavefunctions
    with pytest.raises(RuntimeError, match=r"WavefunctionIdentity.*required"):
        _validate(wfn, blank_identity_field=field)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"neutral_energy": math.inf}, "GRAC.*finite"),
        ({"cation_energy": math.nan}, "GRAC.*finite"),
        ({"homo_energy": -math.inf}, "GRAC.*finite"),
        ({"ionization_potential": math.inf}, "GRAC.*finite"),
        ({"grac_shift": math.nan}, "GRAC.*finite"),
        ({"cation_energy": -75.0, "ionization_potential": 0.0, "grac_shift": -0.3}, "cation energy.*neutral"),
        ({"ionization_potential": 0.0}, "ionization potential.*positive"),
        ({"homo_energy": 0.0, "grac_shift": 0.6}, "HOMO.*negative"),
        ({"homo_energy": -0.7, "grac_shift": -0.1}, "GRAC shift.*nonnegative"),
        ({"ionization_potential": 0.7, "grac_shift": 0.4}, "ionization potential.*cation.*neutral"),
        ({"grac_shift": 0.2}, "GRAC shift.*ionization potential.*HOMO"),
    ],
)
def test_grac_rejects_nonphysical_or_inconsistent_metadata(response_wavefunctions, changes, message):
    wfn, _ = response_wavefunctions
    with pytest.raises(RuntimeError, match=message):
        _validate(wfn, **changes)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"point_count": 0, "points": [], "quadrature_weights": [], "partition_weights": []}, "ISA.*point count"),
        ({"points": [0.0] * 5}, "ISA.*point coordinates.*dimensions"),
        ({"quadrature_weights": [1.0]}, "ISA.*quadrature weights.*dimensions"),
        ({"partition_weights": [0.5] * 3}, "ISA.*partition weights.*dimensions"),
        ({"grid_dimension": 0, "points": []}, "ISA.*grid dimension.*3"),
        ({"site_count": 0, "partition_weights": []}, "ISA.*site count"),
        ({"partition_weights": [0.2, 0.7, 0.7, 0.3]}, "ISA.*partition unity"),
        ({"points": [0.0, 0.0, math.nan, 1.0, 0.0, 0.0]}, "ISA.*finite"),
        ({"quadrature_weights": [0.4, math.inf]}, "ISA.*finite"),
        ({"quadrature_weights": [0.4, 0.0]}, "quadrature weights must be positive"),
        ({"quadrature_weights": [0.4, -0.6]}, "quadrature weights must be positive"),
        ({"partition_weights": [0.2, 0.8, -0.1, 1.1]}, "partition weights must be nonnegative"),
        ({"partition_weights": [0.2, 0.8, math.nan, math.nan]}, "ISA.*finite"),
    ],
)
def test_isa_rejects_invalid_dimensions_or_weights(response_wavefunctions, changes, message):
    wfn, _ = response_wavefunctions
    with pytest.raises(RuntimeError, match=message):
        _validate(wfn, **changes)


def test_isa_accepts_zero_partition_elements(response_wavefunctions):
    wfn, _ = response_wavefunctions
    result = _validate(wfn, partition_weights=[0.0, 1.0, 1.0, 0.0])
    assert result["isa_data_sizes"] == (6, 2, 4)


def test_provider_rejects_null_and_cross_identity(response_wavefunctions):
    wfn, other_wfn = response_wavefunctions
    with pytest.raises(RuntimeError, match=r"Provider.*wavefunction is null"):
        _provider(None, wfn)
    with pytest.raises(RuntimeError, match=r"GRAC.*wavefunction identity"):
        _provider(wfn, other_wfn, isa_identity_wfn=wfn)
    with pytest.raises(RuntimeError, match=r"ISA.*wavefunction identity"):
        _provider(wfn, wfn, isa_identity_wfn=other_wfn)


def test_provider_revalidates_snapshot_and_frequency_cardinality(response_wavefunctions):
    wfn, _ = response_wavefunctions
    provider = _provider(wfn, wfn)
    assert provider.expected_response_count([0.0, 0.5], [0.0, 1.0]) == 2
    with pytest.raises(RuntimeError, match=r"frequency.*dimensions"):
        provider.expected_response_count([0.0, 0.5], [0.0])
    with pytest.raises(RuntimeError, match=r"not implemented.*no response"):
        provider.compute([0.0, 0.5], [0.0, 1.0])

    old_name = wfn.name()
    wfn.set_name(old_name + "-MUTATED")
    try:
        with pytest.raises(RuntimeError, match=r"wavefunction identity changed.*provider construction"):
            provider.expected_response_count([0.0], [0.0])
        with pytest.raises(RuntimeError, match=r"wavefunction identity changed.*provider construction"):
            provider.compute([0.0], [0.0])
    finally:
        wfn.set_name(old_name)

    provider = _provider(wfn, wfn)
    functional = wfn.functional()
    old_functional_name = functional.name()
    functional.set_name(old_functional_name + "-MUTATED")
    try:
        with pytest.raises(RuntimeError, match=r"wavefunction identity changed.*provider construction"):
            provider.expected_response_count([0.0], [0.0])
    finally:
        functional.set_name(old_functional_name)


def test_calculator_names_missing_grac_and_isa_without_publishing_arrays(response_wavefunctions):
    wfn, _ = response_wavefunctions
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
