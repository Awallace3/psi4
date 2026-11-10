"""Offline process/scheduler fixtures; no real SLURM, SDK or native build."""

import importlib.util
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "devtools/disp0"


def plugin():
    spec = importlib.util.spec_from_file_location("revised_physical_proof", TOOLS / "revised_physical_proof.py")
    module = importlib.util.module_from_spec(spec)
    before = list(sys.path)
    sys.path.insert(0, str(TOOLS))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = before
    return module


@pytest.mark.parametrize("corrupt", [None, "changed-process", "foreign-xml"])
def test_physical_plugin_proof_is_written_only_after_same_process_check(tmp_path, monkeypatch, corrupt):
    module = plugin()
    run = tmp_path / "new-run"
    xml = run / "psi4-build/reference-check-fixture/results.xml"
    xml.parent.mkdir(parents=True)
    xml.write_text("<testsuite/>")
    monkeypatch.setenv("DISP0_RUN", str(run))
    state = {"pid": os.getpid(), "job": "24680", "source_pin": "a" * 40,
             "run": str(run), "binary_sha256": {"fixture": "b" * 64},
             "mapped_library_sha256": {"fixture": "b" * 64}}
    monkeypatch.setattr(module, "observed", lambda: dict(state))
    session = SimpleNamespace(config=SimpleNamespace(option=SimpleNamespace(xmlpath=str(xml))))
    module.pytest_sessionstart(session)
    if corrupt == "changed-process":
        state["binary_sha256"] = {"different": "c" * 64}
    elif corrupt == "foreign-xml":
        session.config.option.xmlpath = str(tmp_path / "historical.xml")
    if corrupt:
        with pytest.raises(RuntimeError):
            module.pytest_sessionfinish(session, 0)
        assert not (run / "physical-process-proof.json").exists()
    else:
        module.pytest_sessionfinish(session, 0)
        proof = json.loads((run / "physical-process-proof.json").read_text())
        assert proof["pid"] == os.getpid() and proof["exit_status"] == 0
        assert proof["actual_gpu_correctness_verified"] is False
        assert proof["xml_sha256"] == module.digest(xml)


def batch_fixture(tmp_path, *, control=None, child="printf 'FAKE_PAYLOAD\\n'"):
    commands = tmp_path / "commands"
    commands.mkdir(parents=True)
    logs = tmp_path / "runs/saptdft-cuest-disp0/logs"
    logs.mkdir(parents=True)
    if control is None:
        control = "printf 'JobId=24680 Account=gts-cs207-chemx QOS=embers Requeue=0 NumCPUs=8 \\n'"
    for name, body in (("scontrol", control), ("srun", child)):
        p = commands / name
        p.write_text("#!/usr/bin/env bash\n" + body + "\n")
        p.chmod(0o700)
    script = (TOOLS / "revised_cpu_build.sbatch").read_text().replace(
        "P=/storage/project/r-cs207-0/awallace43", "P=" + shlex.quote(str(tmp_path)))
    env = {k: v for k, v in os.environ.items() if not k.startswith("DISP0_")}
    env.update(SLURM_JOB_ID="24680", DISP0_SOURCE=str(tmp_path / "unused-source"),
               DISP0_PIN="a" * 40, DISP0_NEW_CPU_BUILD_APPROVAL_REFERENCE="FIXTURE_ONLY_NOT_APPROVAL",
               PATH=str(commands) + ":" + env["PATH"])
    return script, env, logs / "24680.revised-status"


@pytest.mark.parametrize("fault", [None, "no-approval", "wrong-qos", "wrong-account", "requeue", "cpu-count"])
def test_revised_batch_guards_do_not_masquerade_as_preparation(tmp_path, fault):
    script, env, status = batch_fixture(tmp_path)
    if fault == "no-approval":
        del env["DISP0_NEW_CPU_BUILD_APPROVAL_REFERENCE"]
    elif fault == "wrong-qos":
        script, env, status = batch_fixture(
            tmp_path / "other", control="printf 'JobId=24680 Account=gts-cs207-chemx QOS=inferno Requeue=0 NumCPUs=8 \\n'")
    elif fault == "wrong-account":
        script, env, status = batch_fixture(
            tmp_path / "other", control="printf 'JobId=24680 Account=gts-cs207 QOS=embers Requeue=0 NumCPUs=8 \\n'")
    elif fault == "requeue":
        script, env, status = batch_fixture(
            tmp_path / "other", control="printf 'JobId=24680 Account=gts-cs207-chemx QOS=embers Requeue=1 NumCPUs=8 \\n'")
    elif fault == "cpu-count":
        script, env, status = batch_fixture(
            tmp_path / "other", control="printf 'JobId=24680 Account=gts-cs207-chemx QOS=embers Requeue=0 NumCPUs=4 \\n'")
    result = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=5)
    if fault:
        assert result.returncode != 0
        assert "FAILED_OR_SIGNALLED" in status.read_text()
        assert "FAKE_PAYLOAD" not in result.stdout
    else:
        assert result.returncode == 0 and "FAKE_PAYLOAD" in result.stdout
        # This fake scheduler exit still cannot satisfy revised_admission,
        # which separately requires actual artifacts/API/physical+accounting.
        assert "CPU_PREPARATION_EXIT_ZERO" in status.read_text()


def test_revised_batch_signal_is_never_success(tmp_path):
    script, env, status = batch_fixture(tmp_path, child="printf 'CHILD_READY\\n'\nexec sleep 30")
    process = subprocess.Popen(["bash", "-c", script], env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    try:
        assert "JobId=" in process.stdout.readline()
        assert process.stdout.readline().strip() == "CHILD_READY"
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=5)
        assert process.returncode == 143
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
    assert "term_received=1" in status.read_text() and "CPU_PREPARATION_EXIT_ZERO" not in status.read_text()
