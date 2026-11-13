from contextlib import contextmanager
import json
from types import SimpleNamespace

import pytest

import dimer_beta_probe as probe


@pytest.mark.parametrize("fails", (False, True))
def test_beta_receipts_survive_native_scf_failure(tmp_path, monkeypatch, fails):
    @contextmanager
    def control(core, beta, records):
        records.append({"before": {"beta": 1}, "after": {"beta": beta}})
        yield

    calls = []
    def energy(psi4, inputs, output, backend, threads):
        calls.append((backend, threads))
        (output / "result.json").write_text(json.dumps(
            {"status": "FAILED" if fails else "CONVERGED_DIAGNOSTIC_ONLY"}))
        if fails:
            raise RuntimeError("SCF did not converge")

    monkeypatch.setattr(probe, "hf_beta_control", control)
    monkeypatch.setattr(probe, "energy_run", energy)
    psi4 = SimpleNamespace(core=object())
    if fails:
        with pytest.raises(RuntimeError, match="did not converge"):
            probe.run(psi4, tmp_path / "inputs", tmp_path, "gpu", 8, 0)
    else:
        probe.run(psi4, tmp_path / "inputs", tmp_path, "gpu", 8, 0)
    report = json.loads((tmp_path / "result.json").read_text())
    assert report["beta_control_verified"]
    assert report["beta_control"] == 0
    assert report["status"] == ("FAILED" if fails else "CONVERGED_DIAGNOSTIC_ONLY")
    assert not report["per_iteration_observer"]
    assert calls == [("gpu", 8)]


def test_no_uncontrolled_beta_or_cpu_override(tmp_path):
    for backend, beta in (("cpu", 0), ("gpu", None), ("gpu", 2)):
        with pytest.raises(ValueError):
            probe.run(None, tmp_path, tmp_path, backend, 8, beta)
