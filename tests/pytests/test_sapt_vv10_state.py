"""Exception-safe restoration of functional state around SAPT response."""

import importlib

import numpy as np
import pytest

import psi4


pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.saptdft]


@pytest.mark.parametrize("backend", ["standard", "einsums"])
@pytest.mark.parametrize("initial", [(False, False), (True, False), (False, True), (True, True), "shared"])
@pytest.mark.parametrize("exit_path", ["success", "setup_error", "solver_error"])
def test_sapt_cpscf_restores_vv10(monkeypatch, backend, initial, exit_path):
    if backend == "einsums":
        pytest.importorskip("einsums")
    module = importlib.import_module(
        "psi4.driver.procrouting.sapt.sapt_jk_terms" + ("_ein" if backend == "einsums" else "")
    )

    class Functional:
        def __init__(self, enabled):
            self.enabled = enabled

        def needs_vv10(self):
            return self.enabled

        def set_do_vv10(self, enabled):
            self.enabled = enabled

    if initial == "shared":
        functionals = [Functional(True)] * 2
    else:
        functionals = [Functional(flag) for flag in initial]
    original = [f.needs_vv10() for f in functionals]

    class InjectedError(RuntimeError):
        pass

    class Wavefunction:
        def __init__(self, functional):
            self._functional = functional

        def functional(self):
            return self._functional

        def set_jk(self, jk):
            assert not any(f.needs_vv10() for f in functionals)
            if exit_path == "setup_error":
                raise InjectedError("setup")

    cache = {}
    for monomer, functional in zip(("A", "B"), functionals):
        cache[f"wfn_{monomer}"] = Wavefunction(functional)
        cache[f"eps_occ_{monomer}"] = psi4.core.Vector.from_array(np.array([-1.0]))
        cache[f"eps_vir_{monomer}"] = psi4.core.Vector.from_array(np.array([1.0]))

    rhs = np.ones((1, 1))
    if backend == "standard":
        rhs = psi4.core.Matrix.from_array(rhs)
    expected = [rhs, rhs]
    called = []

    def solver(*args, **kwargs):
        called.append(True)
        assert not any(f.needs_vv10() for f in functionals)
        if exit_path == "solver_error":
            raise InjectedError("solver")
        return expected, [None, None]

    monkeypatch.setattr(module.solvers, "cg_solver_ein" if backend == "einsums" else "cg_solver", solver)
    if exit_path == "success":
        assert module._sapt_cpscf_solve(cache, object(), rhs, rhs, 1, 1.e-8) is expected
    else:
        with pytest.raises(InjectedError, match="setup" if exit_path == "setup_error" else "solver"):
            module._sapt_cpscf_solve(cache, object(), rhs, rhs, 1, 1.e-8)
    assert bool(called) == (exit_path != "setup_error")
    assert [f.needs_vv10() for f in functionals] == original
