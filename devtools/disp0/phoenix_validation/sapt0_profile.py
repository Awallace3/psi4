"""Host phase evidence and matched runtime-contract A/B; never kernel timing."""
import json
import math
import statistics
from pathlib import Path

from sapt0_timing import RAW, sha


def phase_summary(record):
    plans = record["runtime"]["plans"][record.get("runtime_plan_start", 0):]
    if len(plans) != 1:
        raise ValueError("Expected one measured canonical dispersion provider")
    plan = plans[0]
    phases = plan["owner_cleanup"]["host_phase_timings"]["phases"]
    wall = lambda key: phases[key]["wall_seconds"]
    producer = plan["producer"]["device_host_phase_timings"]["phases"]
    execution = record["dispersion_execution"]
    return {
        "scope": "Inclusive host wall windows; nested intervals overlap, not GPU event times",
        "dispersion_seconds": record["seconds"]["disp"],
        "plan_build_seconds": wall("build_total"),
        "shell_admission_seconds": wall("primary shells.admission") + wall("auxiliary shells.admission"),
        "shell_admission_calls": sum(phases[k]["calls"] for k in
                                     ("primary shells.admission", "auxiliary shells.admission")),
        "shell_native_create_seconds": wall("primary shells.native_create") + wall("auxiliary shells.native_create"),
        "producer_calls": plan["producer"]["calls"],
        "producer_identity_seconds": producer["identity_and_abi"]["wall_seconds"],
        "producer_total_seconds": producer["call_total"]["wall_seconds"],
        "blocked_execution_seconds": execution["blocked_execution_seconds"],
        "entrypoint_seconds": execution["entrypoint_seconds_including_factory_setup_cleanup"],
        "compute_blocks": execution["compute_blocks"],
        "factor_cache": execution["factor_cache"],
        "immutable_runtime": record["runtime"].get("immutable_runtime"),
    }


def compare_contracts(control, candidate):
    control, candidate = Path(control), Path(candidate)
    reports = [json.loads((p/"comparison.json").read_text()) for p in (control, candidate)]
    if any(r["status"] != "PASS_MATCHED_SAPT0" or r["repeats_per_arm"] != 3 for r in reports):
        raise ValueError("Both contracts need three qualified fresh CPU/GPU pairs")
    records = [[json.loads((p/f"gpu-{i}"/"result.json").read_text()) for i in range(3)]
               for p in (control, candidate)]
    reference = records[0][0]
    if any(row["runtime"]["identity_contract"] != "fresh-bytes-v1" for row in records[0]):
        raise ValueError("Control is not the explicit fresh-byte runtime")
    if any(row["runtime"]["identity_contract"] != "process-owned-kernel-sealed-snapshots-v1"
           for row in records[1]):
        raise ValueError("Candidate is not the explicit sealed runtime")
    for row in records[0] + records[1]:
        for key in ("source_pin", "input_sha256", "options", "threads", "host", "host_memory", "nbf", "dimer_nmo"):
            if row[key] != reference[key]:
                raise ValueError("Unmatched runtime A/B: " + key)
        for key in RAW:
            if not math.isclose(row["raw_Eh"][key], reference["raw_Eh"][key],
                                rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("Runtime policy changed signed energy: " + key)
        if not math.isclose(row["total_Eh"], reference["total_Eh"], rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("Runtime policy changed total energy")
        if row["runtime"]["plans"][0]["device"]["uuid"] != reference["runtime"]["plans"][0]["device"]["uuid"]:
            raise ValueError("Runtime A/B used different GPUs")
    timings = {}
    for key in reports[0]["timing"]:
        old = reports[0]["timing"][key]["gpu_median_seconds"]
        new = reports[1]["timing"][key]["gpu_median_seconds"]
        timings[key] = {"fresh_byte_gpu_seconds": old, "sealed_gpu_seconds": new,
                        "old_over_new": old/new if new else None}
    return {
        "status": "PASS_SAME_NODE_RUNTIME_AB", "timing": timings,
        "control_sha256": sha(control/"comparison.json"),
        "candidate_sha256": sha(candidate/"comparison.json"),
        "scope": "Three fresh samples per contract; diagnostic profiles/warm calls excluded",
        "whole_process_gpu_median_seconds": {
            "fresh_byte": statistics.median(reports[0]["whole_process_seconds"]["gpu"]),
            "sealed": statistics.median(reports[1]["whole_process_seconds"]["gpu"])},
        "control_phases": [phase_summary(row) for row in records[0]],
        "candidate_phases": [phase_summary(row) for row in records[1]],
    }
