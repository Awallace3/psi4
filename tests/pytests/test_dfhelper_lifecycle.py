import psi4
import pytest

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]


def test_release_uninitialized_ao():
    """Cleanup is safe before initialization, including after failed setup."""
    molecule = psi4.geometry("He 0 0 0\nsymmetry c1")
    primary = psi4.core.BasisSet.build(molecule, "ORBITAL", "sto-3g")
    auxiliary = psi4.core.BasisSet.build(molecule, "ORBITAL", "def2-universal-jkfit")
    helper = psi4.core.DFHelper(primary, auxiliary)
    held = psi4.core.memory_committed()
    helper.release_AO()
    helper.release_AO()
    assert psi4.core.memory_committed() == held

    helper.set_memory(0)
    with pytest.raises(RuntimeError, match="metric"):
        helper.initialize()
    helper.release_AO()
    assert psi4.core.memory_committed() == held

    psi4.core.clean_timers()  # initialize() leaves its timer running on failure.
    helper.set_memory(1_000_000)
    helper.initialize()
    helper.release_AO()
    helper.release_AO()
    assert psi4.core.memory_committed() == held
