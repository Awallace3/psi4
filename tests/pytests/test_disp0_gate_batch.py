"""CPU-only fake-scheduler control-flow tests; no SLURM/GPU execution."""

import os
from pathlib import Path
import shlex
import signal
import subprocess

import pytest

SOURCES = Path(__file__).resolve().parents[2] / "devtools/disp0"


@pytest.fixture(params=["gpu", "cpu"])
def batch_kind(request):
    return request.param


def fixture_script(tmp_path, scontrol, srun="exit 99", *, batch_kind):
    logs = tmp_path / "runs/saptdft-cuest-disp0/logs"
    logs.mkdir(parents=True)
    commands = tmp_path / "commands"
    commands.mkdir()
    for name, content in (("scontrol", scontrol), ("srun", srun)):
        p = commands / name
        p.write_text("#!/usr/bin/env bash\n" + content + "\n")
        p.chmod(0o755)
    filename = "gpu_gate.sbatch" if batch_kind == "gpu" else "cpu_build.sbatch"
    script = (SOURCES / filename).read_text().replace(
        "P=/storage/project/r-cs207-0/awallace43", "P=" + shlex.quote(str(tmp_path)))
    env = {k: v for k, v in os.environ.items() if not k.startswith("DISP0_")}
    env.update({"SLURM_JOB_ID": "123456", "DISP0_GATE_PIN": "a" * 40,
                "DISP0_GATE_SOURCE": str(tmp_path / "unused-source"),
                "DISP0_BUILD_ROOT": str(tmp_path / "unused-build"),
                "DISP0_GATE_PREP": str(tmp_path / "unused-prep"),
                "DISP0_BUILD_PIN": "b" * 40, "DISP0_BUILD_JOB_ID": "999",
                "DISP0_SCIENCE_SOURCE": str(tmp_path / "unused-science"),
                "DISP0_PIN": "c" * 40, "DISP0_SOURCE": str(tmp_path / "unused-source"),
                "PATH": str(commands) + ":" + env["PATH"]})
    suffix = "gpu-status" if batch_kind == "gpu" else "status"
    return script, env, logs / f"123456.{suffix}"


@pytest.mark.parametrize("failure,expected", [("missing-pin", 1), ("controller", 7), ("wrong-qos", 1)])
def test_batch_early_failure_leaves_failed_status(tmp_path, failure, expected, batch_kind):
    control = ("exit 7" if failure == "controller" else
               "printf 'JobId=123456 Account=gts-cs207-chemx QOS=inferno JobState=RUNNING\\n'")
    script, env, status = fixture_script(tmp_path, control, batch_kind=batch_kind)
    if failure == "missing-pin":
        del env["DISP0_GATE_PIN" if batch_kind == "gpu" else "DISP0_PIN"]
    result = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=5)
    assert result.returncode == expected, result.stderr
    text = status.read_text()
    assert f"exit_code={expected}\n" in text and "FAILED_OR_SIGNALLED" in text
    assert "EXIT_ZERO" not in text


def test_batch_signal_cannot_write_success(tmp_path, batch_kind):
    control = "printf 'JobId=123456 Account=gts-cs207-chemx QOS=embers JobState=RUNNING\\n'"
    script, env, status = fixture_script(tmp_path, control, "printf 'CHILD_READY\\n'\nexec sleep 30",
                                        batch_kind=batch_kind)
    process = subprocess.Popen(["bash", "-c", script], env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert "JobId=" in process.stdout.readline()
        assert process.stdout.readline().strip() == "CHILD_READY"
        process.send_signal(signal.SIGTERM)
        _, err = process.communicate(timeout=5)
        assert process.returncode == 143, err
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
    text = status.read_text()
    assert "exit_code=143\n" in text and "term_received=1\n" in text
    assert "EXIT_ZERO" not in text
