from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from cachy_dimer_screen import case_specs
from peptide_plan_probe import fixed_samples, fresh_plan, option_receipt, rebuilt_state


def test_rebuilt_fock_checks_stationarity_not_retained_diis_fock():
    d, h = np.diag([1., 0.]), np.diag([-2., 1.])
    s = x = np.eye(2)
    j, k = np.diag([.5, .1]), np.diag([.2, 0.])
    row = rebuilt_state(d, h, s, x, j, k, 1.)
    assert row["total_rebuilt_Eh"] == pytest.approx(-2.2)
    assert row["orthonormal_commutator_rms"] == 0
    j[0, 1] = j[1, 0] = .01
    assert rebuilt_state(d, h, s, x, j, k, 1.)["orthonormal_commutator_rms"] > 0


def test_option_receipt_restores_current_module_even_on_error():
    options = SimpleNamespace(module="SAPT")
    options.get_current_module = lambda: options.module
    options.set_current_module = lambda v: setattr(options, "module", v)
    def get(module, key):
        options.module = module
        return 1
    core = SimpleNamespace(get_options=lambda: options, get_option=get,
                           get_global_option=lambda key: 1,
                           has_global_option_changed=lambda key: False,
                           has_local_option_changed=lambda module, key: False)
    assert option_receipt(core)["current_module"] == "SAPT"
    assert options.module == "SAPT"
    def fail(key):
        raise ValueError("missing")
    core.get_global_option = fail
    with pytest.raises(ValueError):
        option_receipt(core)
    assert options.module == "SAPT"


def test_plan_campaign_orders_producer_before_pinned_replays():
    assert case_specs("plans") == [
        ("retained", "peptide", "gpu", "energy"),
        ("cpu-replay", "peptide", "cpu", "energy"),
        ("gpu-replay", "peptide", "gpu", "energy")]
    assert len(case_specs("screen")) == 16
    assert len({s[0] for s in case_specs("screen")}) == 16


def test_fresh_plan_preserves_exposed_scf_parameters_and_factory_memory():
    jk = Mock()
    jk.name.return_value = "cuESTJK"
    core = SimpleNamespace(JK=SimpleNamespace(build=Mock(return_value=jk)))
    state = dict(memory_doubles=123456, omega=0.0, alpha=1.0, beta=0.0, wcombine=True)
    primary, auxiliary = object(), object()
    assert fresh_plan(core, primary, auxiliary, state, "gpu") is jk
    core.JK.build.assert_called_once_with(primary, auxiliary, jk_type="DF",
                                         do_wK=False, memory=123456)
    jk.set_omega.assert_called_once_with(0.0)
    jk.set_omega_alpha.assert_called_once_with(1.0)
    jk.set_omega_beta.assert_called_once_with(0.0)
    jk.set_wcombine.assert_called_once_with(True)
    jk.initialize.assert_not_called()
    jk.name.return_value = "DiskDFJK"
    with pytest.raises(RuntimeError, match="Unexpected"):
        fresh_plan(core, primary, auxiliary, state, "cpu")


class FakeJK:
    def __init__(self, mutate=False):
        self.calls = 0
        self.mutate = mutate
    def C_clear(self):
        pass
    def C_left_add(self, co):
        self.co = co
    def compute(self):
        self.calls += 1
        if self.mutate:
            self.co.np[0, 0] += 1
        self.d = self.co.np @ self.co.np.T
    def J(self):
        return [SimpleNamespace(np=2 * self.d)]
    def K(self):
        return [SimpleNamespace(np=self.d)]
    def D(self):
        return [SimpleNamespace(np=self.d)]
    def name(self):
        return "cuESTJK"
    def memory(self):
        return 100
    def get_omega(self):
        return 0.0
    def get_omega_alpha(self):
        return 1.0
    def get_omega_beta(self):
        return 0.0
    def get_wcombine(self):
        return False


def test_fixed_samples_do_not_invoke_scf_and_compare_identical_density():
    core = SimpleNamespace(Matrix=SimpleNamespace(from_array=lambda x: SimpleNamespace(np=x)))
    co = np.array([[0.5], [0.25]])
    d = co @ co.T
    jk = FakeJK()
    arrays, rows = {}, []
    baseline = fixed_samples(core, jk, co, d, "retained", arrays, rows)
    assert jk.calls == 4
    assert len(arrays) == 8 and len(rows) == 4
    assert all(r["D_max_abs"] == 0 for r in rows)
    assert all(r["vs_same_plan_first"]["closed_shell_JK_delta_Eh"] == 0 for r in rows)
    shifted = (baseline[0] + 1e-6, baseline[1])
    fixed_samples(core, jk, co, d, "fresh", arrays, rows, shifted)
    assert rows[-1]["vs_retained_first"]["closed_shell_JK_delta_Eh"] != 0
    with pytest.raises(RuntimeError, match="modified"):
        fixed_samples(core, FakeJK(True), co, d, "bad", {}, [])
