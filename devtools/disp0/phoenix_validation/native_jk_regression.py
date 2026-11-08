"""Sealed native JK pytest checks. No benchmark or accepted speedup."""
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

from stage_scaling import (configuration, digest, pinned_science,
                           require_cpu_runtime_admission, write)

SCOPE = "NATIVE_JK_REGRESSION_AND_SMALL_GATE_NOT_A_BENCHMARK"
REQUIRED = "test_cuest_jk_honors_fitting_condition_like_cpu"
EXPECTED = {
    "test_cuest_jk_asymmetric_exchange_matches_cpu",
    "test_cuest_jk_symmetric_exchange_matches_cpu",
    "test_cuest_jk_asymmetric_is_not_the_symmetric_answer",
    "test_cuest_jk_k_carries_the_exchange_fraction",
    "test_cuest_jk_exchange_fraction_is_resettable_after_initialize",
    REQUIRED,
}


def junit_summary(path):
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    names = [case.get("name") for case in cases]
    result = {"tests":len(cases),
              "skipped":sum(c.find("skipped") is not None for c in cases),
              "failures":sum(c.find("failure") is not None for c in cases),
              "errors":sum(c.find("error") is not None for c in cases),
              "required_test_executed":REQUIRED in names}
    if (len(names)!=len(EXPECTED) or set(names)!=EXPECTED
            or any(int(s.get(k, "0")) for s in root.iter("testsuite")
                   for k in ("skipped", "failures", "errors"))
            or any(result[k] for k in ("skipped","failures","errors"))):
        raise RuntimeError("Native GPU regressions incomplete/failed: "+json.dumps(result))
    return result


def run(runtime, manifest, output):
    """Caller bootstraps before this function imports pytest or scientific code."""
    from scf_repeat_probe import native_device_identity
    source = Path(manifest["source"])
    tests = source/"tests/pytests/test_cuest_jk.py"
    if digest(tests)!=manifest["native_test_sha256"]:
        raise RuntimeError("Frozen native regression source changed")
    sys.path.insert(0,str(tests.parent))
    os.environ.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTEST_ADDOPTS="")
    import pytest
    xml = output/"native-jk.xml"
    report = {"status":"STARTED", "scope":SCOPE, "accepted_speedup_count":0,
              "science_pin":manifest["science_pin"],"pid":os.getpid(),
              "native_test_sha256":manifest["native_test_sha256"]}
    try:
        code = runtime.guarded_call(pytest.main, [
            "-q","--disable-warnings","-p","no:cacheprovider",
            "--junitxml="+str(xml),str(tests)])
        report["pytest_returncode"]=int(code)
        if xml.exists():
            report["junit_sha256"]=digest(xml)
        if code:
            raise RuntimeError("Native JK regression failed")
        report["junit"]=junit_summary(xml)
        report["device"]=native_device_identity()
        runtime.verify_optional_modules()
        report["status"]="PASS_NATIVE_JK_GPU_REGRESSION_ONLY"
    except BaseException as error:
        report.update(status="FAILED",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write(output/"probe-result.json",report)


def submit(args, schedule):
    """Use the common pinned batch/TERM/scratch machinery, not a native overlay."""
    outer=Path(__file__).resolve().parents[3]
    pin=subprocess.check_output(["git","-C",str(outer),"rev-parse","HEAD"],text=True).strip()
    if subprocess.check_output(["git","-C",str(outer),"status","--porcelain"],text=True).strip():
        raise RuntimeError("Clean pinned native regression launcher required")
    if args.kind!="gpu":
        raise ValueError("Native GPU regression requires its own H200 allocation")
    base=json.loads(args.config.resolve(strict=True).read_text())
    source=Path(base["source"]).resolve(strict=True)
    science_pin=pinned_science(base["source_pin"])
    if (subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip()!=science_pin
            or subprocess.check_output(["git","-C",str(source),"status","--porcelain"],text=True).strip()):
        raise RuntimeError("Exact frozen native source required")
    sys.path.insert(0,str(source/"devtools/disp0/phoenix_validation"))
    admission=require_cpu_runtime_admission(base)
    proof=Path(base["build_proof"])
    build=json.loads(proof.read_text())
    if (digest(proof)!=base["build_proof_sha256"] or build["status"]!="PASS"
            or build["pin"]!=science_pin):
        raise RuntimeError("Exact completed native build proof required")
    tests=source/"tests/pytests/test_cuest_jk.py"
    if "def "+REQUIRED+"(" not in tests.read_text():
        raise RuntimeError("Candidate source lacks the required fitting-cutoff regression")
    root=args.root
    if not root.is_absolute():
        raise ValueError("Fresh absolute regression root required")
    root.mkdir(mode=0o700)
    (root/"logs").mkdir()
    aliases=root/"sealed-workers"
    aliases.mkdir(mode=0o700)
    config=configuration(base,aliases,"peptide","gpu")
    path=root/"runtime-config.json"
    write(path,config)
    manifest={"kind":"gpu","mode":"native-tests","source":str(source),
        "science_pin":science_pin,"outer_pin":pin,"scope":SCOPE,
        "accepted_speedup_count":0,"automatic_retry":False,
        "config":str(path),"config_sha256":digest(path),
        "cpu_runtime_admission":admission,"native_test_sha256":digest(tests),
        "account":"gts-cs207-chemx","qos":"embers","allocation_wall_seconds":1800}
    write(root/"manifest.json",manifest)
    schedule(root,manifest,config,outer)
