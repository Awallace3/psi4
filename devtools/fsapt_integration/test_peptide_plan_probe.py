from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from cachy_dimer_screen import case_specs
from peptide_plan_probe import (
    capture_orthogonalizer, density_reconstruction, fixed_samples, fresh_plan,
    hf_beta_control, option_receipt, rebuilt_state, replay_variants)


def test_beta_control_sets_only_unused_coefficient_before_real_initialization():
    class JK:
        def __init__(self):
            self.beta = 1.
            self.seen = []
        def initialize(self):
            self.seen.append(self.beta)
        def name(self):
            return "cuESTJK"
        def memory(self):
            return 100
        def get_omega(self):
            return 0.
        def get_omega_alpha(self):
            return 1.
        def get_omega_beta(self):
            return self.beta
        def get_wcombine(self):
            return False
        def set_omega_beta(self, value):
            self.beta = value
    original = JK.initialize
    records, jk = [], JK()
    with hf_beta_control(SimpleNamespace(JK=JK), 0, records):
        jk.initialize()
    assert jk.seen == [0.]
    assert records[0]["before"]["beta"] == 1.
    assert records[0]["after"]["beta"] == 0.
    assert JK.initialize is original
    assert [s[0] for s in case_specs("beta-scf")] == ["beta1-0", "beta0-0", "beta0-1", "beta1-1"]


def test_beta_entrypoint_screen_balances_order_and_repeats():
    cases = case_specs("beta-entrypoints")
    assert [row[0] for row in cases] == [
        "beta1-energy-0", "beta1-sapt-0", "beta0-energy-0", "beta0-sapt-0",
        "beta0-sapt-1", "beta0-energy-1", "beta1-sapt-1", "beta1-energy-1"]
    assert all(system == "peptide" and backend == "gpu"
               and label.split("-")[1] == entrypoint
               for label, system, backend, entrypoint in cases)
    with pytest.raises(ValueError):
        case_specs("beta-entrypoints", replay_only=True)


def test_unused_beta_ab_changes_only_beta_and_interleaves_actual_setting():
    state = dict(alpha=1., beta=1., omega=0., memory_doubles=123, wcombine=False)
    variants = replay_variants(state, "gpu")
    assert [v["beta"] for v in variants] == [1., 0., 1., 0.]
    assert all({k for k in state if state[k] != v[k]} <= {"beta"} for v in variants)
    assert state["beta"] == 1.
    assert replay_variants(state, "cpu") == [state, state]
    with pytest.raises(ValueError):
        replay_variants({**state, "omega": .3}, "gpu")


def test_density_check_records_roundoff_but_rejects_changed_inputs():
    co = np.array([[.5], [.25]])
    d = co @ co.T
    d[0, 0] = np.nextafter(d[0, 0], np.inf)
    result = density_reconstruction(co, d)
    assert result["unequal_elements"] == 1
    assert 0 < result["max_abs"] < 1e-13
    d[0, 0] += 1e-6
    with pytest.raises(AssertionError):
        density_reconstruction(co, d)


def test_orthogonalizer_copied_before_native_finalize_and_method_restored():
    class HF:
        def __init__(self):
            self.x = np.eye(2)
        def _diis_orthogonalizer(self):
            return SimpleNamespace(np=self.x)
        def finalize_energy(self):
            self.x[:] = 0
            self.x = None
            return -1.0
    original = HF.finalize_energy
    captured = {}
    wfn = HF()
    with capture_orthogonalizer(SimpleNamespace(HF=HF), captured):
        assert wfn.finalize_energy() == -1.0
    np.testing.assert_array_equal(captured["X"], np.eye(2))
    assert wfn.x is None
    assert HF.finalize_energy is original


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
    assert case_specs("plans", replay_only=True) == case_specs("plans")[1:]
    with pytest.raises(ValueError):
        case_specs("screen", replay_only=True)


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
    assert len(arrays) == 12 and len(rows) == 4
    assert all(r["D_max_abs"] == 0 for r in rows)
    assert all(r["vs_same_plan_first"]["closed_shell_JK_delta_Eh"] == 0 for r in rows)
    shifted = (baseline[0] + 1e-6, baseline[1])
    fixed_samples(core, jk, co, d, "fresh", arrays, rows, shifted)
    assert rows[-1]["vs_retained_first"]["closed_shell_JK_delta_Eh"] != 0
    with pytest.raises(RuntimeError, match="modified"):
        fixed_samples(core, FakeJK(True), co, d, "bad", {}, [])
