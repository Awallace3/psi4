import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("grid_workload_gate",
                                            Path(__file__).with_name("grid_workload_gate.py"))
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


@pytest.fixture
def artifacts(tmp_path):
    validation, science = tmp_path / "validation", tmp_path / "science"
    def put(relative, data):
        p = validation / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(data if isinstance(data, str) else json.dumps(data))
    put("metadata/runtime.env", f"scientific_commit={gate.PIN}\nexit_code=1\nterm_received=0\njob_id=14044088\n")
    for backend in ("host", "cuda"):
        p = science / f"build_grid_{backend}/stage/lib/psi4"
        p.mkdir(parents=True)
        core = p / "core.so"
        core.write_bytes(backend.encode())
        put(f"{backend}/commit.txt", gate.PIN)
        put(f"{backend}/import.json", dict(core=str(core), package=str(p), variant=backend,
             core_sha256=hashlib.sha256(core.read_bytes()).hexdigest()))
        put(f"{backend}/regressions.xml", '<testsuites><testsuite tests="36" failures="0" errors="0" skipped="0">' +
            '<testcase/>' * 36 + '</testsuite></testsuites>')
        put(f"{backend}/profile/timers.json", dict(
            timers={"cuEST XC: CUDA LibXC Functional": {}} if backend == "cuda" else {},
            microtimers={"cuestXCPotentialCompute": {}, "xc_gga_exc_vxc": {}}))
        for stage in ("water", "water-iterative"):
            rows = []
            for final, initial in gate.CONFIGS:
                rows.append(dict(final=final, initial=initial, status="ok", full_sapt=False,
                    final_grid_iterations=2, components=dict.fromkeys(gate.COMPONENTS, 0.1),
                    options=dict(use_cuest=True, scf_type="df", sapt_dft_induction_type="none",
                                 sapt_dft_grac_compute="iterative" if stage.endswith("iterative") else "none")))
            put(f"{backend}/{stage}/summary.json", rows)
        full = [dict(final=f, initial=i, full_sapt=True, status="failed", error=gate.UNSUPPORTED,
                     options=dict(sapt_dft_induction_type="cpks")) for f, i in
                (((99, 590), None), ((99, 590), (50, 302)), ((75, 302), None), ((75, 302), (35, 194)))]
        put(f"{backend}/water-full/summary.json", full)
    return validation, science


def test_scoped_receipt_does_not_certify_full_sapt(artifacts):
    report = gate.validate(*artifacts, gate.PIN)
    assert report["scoped_validation_passed"]
    assert report["original_job_status"] == "FAILED"
    assert not report["full_sapt_validation_passed"]
    assert len(report["input_sha256"]) == 15


@pytest.mark.parametrize("mutation", [
    "skipped", "changed_core", "unexpected_full_error", "no_final", "bad_grid",
    "bad_switch", "nan", "missing_case", "wrong_workload", "wrong_profile",
])
def test_gate_rejects_incomplete_or_unrelated_evidence(artifacts, mutation):
    v, science = artifacts
    if mutation == "skipped":
        p = v / "host/regressions.xml"
        p.write_text(p.read_text().replace('skipped="0"', 'skipped="1"'))
    elif mutation == "changed_core":
        (science / "build_grid_host/stage/lib/psi4/core.so").write_bytes(b"changed")
    elif mutation == "wrong_profile":
        p = v / "host/profile/timers.json"
        d = json.loads(p.read_text());d["timers"]["cuEST XC: CUDA LibXC Functional"] = {}
        p.write_text(json.dumps(d))
    else:
        p = v / ("host/water-full/summary.json" if mutation == "unexpected_full_error" else
                 "host/water/summary.json")
        rows = json.loads(p.read_text())
        if mutation == "unexpected_full_error":
            rows[0]["error"] = "unrelated failure"
        elif mutation == "missing_case":
            rows.pop()
        else:
            row = next(r for r in rows if r["final"] == [75, 302] and r["initial"])
            if mutation == "no_final":
                row["final_grid_iterations"] = 0
            elif mutation == "wrong_workload":
                row["options"]["sapt_dft_induction_type"] = "cpks"
            else:
                row["components"]["SAPT TOTAL ENERGY"] += (
                    1e-4 if mutation == "bad_grid" else 1e-8 if mutation == "bad_switch" else float("nan"))
        p.write_text(json.dumps(rows))
    with pytest.raises(ValueError):
        gate.validate(v, science, gate.PIN)
