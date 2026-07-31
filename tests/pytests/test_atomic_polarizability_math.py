from pathlib import Path

import pytest

import psi4


pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.mints]

_PUBLIC_ARRAYS = (
    "ATOMIC POLARIZABILITIES",
    "ATOMIC DYNAMIC POLARIZABILITIES",
    "ATOMIC POLARIZABILITY FREQUENCIES",
    "ATOMIC C6",
    "ATOMIC C8",
    "ATOMIC C10",
    "ATOMIC C12",
)


def test_atomic_polarizabilities_api_is_registered():
    assert "ATOMIC_POLARIZABILITIES" in psi4.core.OEProp.valid_methods
    assert psi4.core.get_global_option("ATOMIC_POLARIZABILITY_N_FREQUENCIES") == 10
    assert psi4.core.get_global_option("ATOMIC_POLARIZABILITY_FREQUENCY_SCALE") == pytest.approx(0.5)


def test_atomic_polarizabilities_fail_closed_without_response_data():
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

    oeprop = psi4.core.OEProp(wfn)
    oeprop.add("MULTIPOLE(2)")
    oeprop.add("ATOMIC_POLARIZABILITIES")

    with pytest.raises(RuntimeError, match=r"AtomicPolarizabilityCalculator.*response data"):
        oeprop.compute()

    unpublished = ("DIPOLE", "QUADRUPOLE", *_PUBLIC_ARRAYS)
    assert all(not wfn.has_array_variable(name) for name in unpublished)
    assert all(not psi4.core.has_array_variable(name) for name in unpublished)


def test_atomic_polarizabilities_reject_incomplete_wavefunction_prerequisites():
    molecule = psi4.geometry(
        """
        He 0.0 0.0 0.0
        symmetry c1
        """
    )
    wfn = psi4.core.Wavefunction.build(molecule, "sto-3g")
    calculator = psi4.core.AtomicPolarizabilityCalculator(wfn)

    with pytest.raises(RuntimeError, match=r"unsupported wavefunction.*orbital response data"):
        calculator.compute()

    assert all(not wfn.has_array_variable(name) for name in _PUBLIC_ARRAYS)


def test_native_atomic_polarizability_source_guard():
    from test_native_atomic_polarizability_source_guard import source_violations

    repo_root = next(
        parent for parent in Path(__file__).resolve().parents
        if (parent / "psi4/src/psi4/libmints/atomic_polarizability.cc").is_file()
    )
    native_sources = (
        repo_root / "psi4/src/psi4/libmints/atomic_polarizability.cc",
        repo_root / "psi4/src/psi4/libmints/atomic_polarizability.h",
        repo_root / "psi4/src/psi4/libmints/oeprop.cc",
        repo_root / "psi4/src/psi4/libmints/oeprop.h",
        repo_root / "psi4/src/export_oeprop.cc",
    )

    violations = []
    for source in native_sources:
        violations.extend(f"{source.name}: {item}" for item in source_violations(source.read_text()))

    cmake_text = (repo_root / "psi4/src/psi4/libmints/CMakeLists.txt").read_text()
    assert "atomic_polarizability.cc" in cmake_text
    assert violations == []

    canary = 'void launch() { std::system("camcasp"); }'
    assert source_violations(canary) == ["forbidden process API: std::system(", "forbidden external term: camcasp"]
