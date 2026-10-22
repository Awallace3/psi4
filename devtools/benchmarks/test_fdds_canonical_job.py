"""Offline controller contracts; not a vendor/device test."""
from pathlib import Path
import subprocess
import pytest
import json
import os
import sys

from fdds_canonical_submit import accepted_id, durable
from fdds_canonical_controller import gpu_empty, supervised, validate_job

ROOT = Path(__file__).resolve().parent


def test_canonical_job_syntax():
    for name in ("fdds_canonical_job.sh", "fdds_canonical_payload.sh"):
        subprocess.run(["bash", "-n", str(ROOT / name)], check=True)


def test_gate_order_and_no_old_stage():
    payload = (ROOT / "fdds_canonical_payload.sh").read_text()
    assert payload.index("-DENABLE_cuEST=OFF") < payload.index("python\" -m pytest")
    assert payload.index('mkdir "$out/native"') < payload.index("fdds_validate.sh")
    assert payload.index('pathlib.Path("verified.json")') < payload.index("fdds_validate.sh")
    assert "not cuest and not cuda_cc8" in payload
    assert '[[ ! -v "$variable" ]]' in payload
    assert '[[ ! -e "$source_root/build_fdds_validation" ]]' in payload
    assert "CUEST_FDDS_FULL_AFTER_FOCUSED=1" in payload
    assert "cuest-0.2.50-a06e8148" in payload
    assert "pip install" not in payload and "conda install" not in payload


def test_cleanup_precedes_completion():
    job = (ROOT / "fdds_canonical_controller.py").read_text()
    assert job.index("fcntl.flock(lease") < job.index('assert gpu_empty(uuid')
    assert job.index('gpu_empty(uuid, out / "gpu-after.csv")') < job.index('stamp(out / "COMPLETED")')
    assert "13500, out" in job
    assert 'out.mkdir()' in job


@pytest.mark.parametrize("response", ["", "123\n456\n", "Submitted batch job 123\n", "0\n", "123;bad space\n"])
def test_ambiguous_receipt_rejected(response):
    with pytest.raises(ValueError):
        accepted_id(response)


def test_durable_no_overwrite(tmp_path):
    assert accepted_id("123;phoenix-slurm\n") == "123"
    path = tmp_path / "SUBMISSION_INVOKED"
    durable(path, "first")
    with pytest.raises(FileExistsError):
        durable(path, "second")
    assert path.read_text() == "first"


def test_nvml_failure_does_not_mean_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a[0], 1, "", "NVML failed"))
    with pytest.raises(subprocess.CalledProcessError):
        gpu_empty("GPU-test", tmp_path / "gpu.csv")


def test_nvml_busy(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a[0], 0, "12, python, 1\n", ""))
    assert not gpu_empty("GPU-test", tmp_path / "gpu.csv")


JOB = ("QOS=embers Account=gts-cs207-chemx Requeue=0 TimeLimit=04:00:00 "
       "NumNodes=1 NumTasks=1 NumCPUs=8 MinMemoryNode=128G TresPerNode=gres/gpu:h200:1")


@pytest.mark.parametrize("replacement", ["TimeLimit=08:00:00", "NumTasks=2", "NumNodes=2", "TresPerNode=gres/gpu:h200:2"])
def test_resource_envelope(replacement):
    validate_job(JOB)
    key = replacement.split("=")[0]
    bad = " ".join(replacement if item.startswith(key + "=") else item for item in JOB.split())
    with pytest.raises(AssertionError):
        validate_job(bad)


@pytest.mark.parametrize("kind,expected", [("normal", 0), ("timeout", 124), ("escaped_child", 125)])
def test_real_descendant_containment(kind, expected, tmp_path):
    body = {"normal": "pass", "timeout": "import time; time.sleep(60)",
            "escaped_child": "import subprocess,sys; subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True)"}[kind]
    assert supervised([sys.executable, "-c", body], .3, tmp_path, grace=.2) == expected
    receipt = json.loads((tmp_path / "containment.json").read_text())
    assert receipt["cleanup_verified"] and not receipt["remaining"]


@pytest.mark.parametrize("inherited", [None, "/existing/python/path"])
def test_psiapi_nounset_initialization(inherited):
    initialization = 'export PYTHONPATH="${PYTHONPATH:-}"'
    for filename in ("fdds_canonical_payload.sh", "fdds_validate.sh"):
        text = (ROOT / filename).read_text()
        assert text.index(initialization) < text.index('eval "$(')
    env = {"PATH": os.environ["PATH"]}
    if inherited is not None:
        env["PYTHONPATH"] = inherited
    result = subprocess.run(
        ["bash", "-uc", initialization + "\n"
         "eval 'export PYTHONPATH=/new/stage/lib:$PYTHONPATH'\n"
         'printf "%s" "$PYTHONPATH"'],
        env=env, check=True, capture_output=True, text=True)
    assert result.stdout == "/new/stage/lib:" + (inherited or "")
