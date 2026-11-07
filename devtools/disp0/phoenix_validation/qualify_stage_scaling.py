"""Aggregate finalized stage-campaign baselines using the original comparisons.

Does not execute science, change a producer, or include the extra traced sample
in a speedup denominator. Intended for a bounded lightweight artifact analysis.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

from stage_scaling import SCIENCE_PIN, SYSTEMS, digest, require_producer, write


def completed(root, expected_status):
    if (root/"job-status.env").read_text().splitlines() != ["exit_code=0", "term_received=0"]:
        raise RuntimeError("Uninterrupted exit0 allocation required: "+str(root))
    document = json.loads((root/"stage-result.json").read_text())
    if (document["status"] != expected_status or document["source_pin"] != SCIENCE_PIN
            or not document["scratch_cleanup"]["complete"]
            or any(p["returncode"] or not p["owned_group_reaped"]
                   or p.get("retirement_reason") is not None for p in document["processes"])):
        raise RuntimeError("Incomplete/failed scientific producer: "+str(root))
    return document


def worker(row, expected):
    if Path(row["path"]).resolve(strict=True) != expected.resolve(strict=True):
        raise RuntimeError("Scientific sample path changed")
    for name, key in (("result.json", "result_sha256"), ("timer.dat", "timer_sha256"),
                      ("psi4.out", "psi4_out_sha256")):
        if digest(expected/name) != row[key]:
            raise RuntimeError("Final scientific artifact changed: "+str(expected/name))
    doc = json.loads((expected/"result.json").read_text())
    for name, checksum in doc["artifact_sha256"].items():
        if digest(expected/name) != checksum:
            raise RuntimeError("Scientific array seal changed")
    for name, checksum in doc["output_file_sha256"].items():
        if digest(expected/"fsapt"/name) != checksum:
            raise RuntimeError("Scientific F-SAPT output seal changed")
    if (doc["status"] != "EXECUTED_REQUIRES_INDEPENDENT_COMPARISON"
            or doc["source_pin"] != SCIENCE_PIN or not doc["benchmark"]
            or doc["benchmark_whole_energy_seconds"] != row["energy_seconds"]):
        raise RuntimeError("Scientific worker provenance/timer changed")
    return doc


def qualify(args):
    gpu_root = args.gpu_root.resolve(strict=True)
    gpu_manifest_path = gpu_root.parent/"manifest.json"
    manifest = json.loads(gpu_manifest_path.read_text())
    system = manifest["system"]
    if system not in SYSTEMS or len(args.cpu_roots) != 3:
        raise ValueError("Exactly three selected-system CPU reference/oracle roots required")
    source = Path(manifest["source"])
    if (subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"],
                                text=True).strip() != SCIENCE_PIN
            or subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"],
                                       text=True).strip()):
        raise RuntimeError("Frozen original scientific/comparison helper source changed")
    sys.path.insert(0, str(source/"devtools/disp0/phoenix_validation"))
    import numpy as np
    from fragment_gate import FIELDS, require_oracle_config
    from full_inputs import load_point
    from repeat_gauge import compare_repeat
    from run_full import compare_energy
    from run_gate import compare

    args.output.mkdir(mode=0o700)  # Never rewrite any previous qualification.
    path = args.output/"qualification.json"
    report = {"status": "STARTED", "accepted_speedup_count": 0, "system": system,
              "science_pin": SCIENCE_PIN, "execution_helper_pin": manifest["outer_pin"],
              "analysis_recipe_sha256": digest(__file__), "checks": [],
              "timing_scope": "FRESH_PROCESS_WHOLE_ENERGY_BASELINES_NOT_EXTRA_TERM_TRACE"}
    write(path, report)
    try:
        gpu_hash = digest(gpu_root/"stage-result.json")
        gpu = completed(gpu_root, "PASS_GPU_REPEATS_AND_TRACE_AWAITING_CPU")
        require_producer(gpu_root, system, manifest["config_sha256"])
        config = json.loads(Path(manifest["config"]).read_text())
        if digest(manifest["config"]) != manifest["config_sha256"]:
            raise RuntimeError("GPU runtime configuration changed")
        _, dimensions = load_point(Path(manifest["inputs"]), manifest["inputs_sha256"], system)
        cpus, values, cpu_hashes, roots = [], [], [], []
        for i, given in enumerate(args.cpu_roots):
            root = given.resolve(strict=True)
            cp = json.loads((root.parent/"manifest.json").read_text())
            d = completed(root, "PASS_CPU_REFERENCE_AND_ORACLE_NEEDS_THREE_REPEAT_AGGREGATION")
            if (cp["kind"] != "cpu" or cp["repeat"] != i or cp["system"] != system
                    or cp["science_pin"] != SCIENCE_PIN or cp["outer_pin"] != manifest["outer_pin"]
                    or d["outer_pin"] != manifest["outer_pin"]
                    or digest(root.parent/"manifest.json") != d["manifest_sha256"]
                    or cp["gpu_producer_sha256"] != gpu_hash
                    or cp["gpu_manifest_sha256"] != digest(gpu_manifest_path)
                    or cp["inputs_sha256"] != manifest["inputs_sha256"]
                    or cp["gpu_config_sha256"] != manifest["config_sha256"]
                    or digest(cp["config"]) != cp["config_sha256"]):
                raise RuntimeError("CPU/GPU submission/producer seals or identities differ")
            if cpus and cp["config_sha256"] != cpus[0]:
                raise RuntimeError("CPU repeat runtime configurations differ")
            cpus.append(cp["config_sha256"])
            require_oracle_config(json.loads(Path(cp["config"]).read_text()), config)
            cpu_path = root/(system+"-cpu")
            cpu = worker(d["reference"], cpu_path)
            original = gpu["samples"][i]
            gpu_path = gpu_root/f"{system}-r{i}"
            peer = worker(original, gpu_path)
            oracle_row = next(p for p in d["processes"] if p["label"] == system+"-oracle")
            oracle_path = root/(system+"-oracle")
            oracle = worker(oracle_row, oracle_path)
            if (cpu["arm"] != "cpu" or peer["arm"] != "gpu" or oracle["arm"] != "cpu"
                    or any(doc["system"] != system or doc["orbital_dimensions"] != dimensions
                           for doc in (cpu, peer, oracle))
                    or oracle["oracle_input_result_sha256"] != original["result_sha256"]
                    or set(oracle["same_gauge_compared_fields"]) != set(FIELDS)):
                raise RuntimeError("Independent oracle or scientific sample identity differs")
            errors = {}
            with np.load(oracle_path/"fragment-matrices.npz", allow_pickle=False) as left, \
                    np.load(gpu_path/"fragment-matrices.npz", allow_pickle=False) as right:
                for key in FIELDS:
                    np.testing.assert_allclose(left[key], right[key], rtol=2e-5, atol=2e-7,
                                               err_msg="Original independent-oracle policy: "+key)
                    errors[key] = float(np.max(np.abs(left[key]-right[key])))
            check = {"repeat": i, "cpu_job": root.name, "cpu_root": str(root),
                     "cpu_manifest_sha256": digest(root.parent/"manifest.json"),
                     "cpu_report_sha256": digest(root/"stage-result.json"),
                     "energy": compare_energy(cpu, peer), "signed_components": compare(cpu, peer),
                     "oracle_maximum_absolute_field_errors": errors}
            if roots:
                check["cpu_repeat"] = compare_repeat(roots[0], cpu_path,
                                                     dimensions=dimensions, arm="cpu")
            roots.append(cpu_path)
            values.append(float(cpu["benchmark_whole_energy_seconds"]))
            cpu_hashes.append(check["cpu_report_sha256"])
            report["checks"].append(check)
            write(path, report)
        traced = worker(gpu["trace"], gpu_root/(system+"-traced"))
        if traced["arm"] != "gpu":
            raise RuntimeError("Wrong separately traced worker backend")
        if len(set(roots)) != 3 or any(not 0 < v < float("inf") for v in values):
            raise RuntimeError("Three distinct positive CPU baseline samples required")
        gpu_values = [s["energy_seconds"] for s in gpu["samples"]]
        report.update(
            status="PASS_MATCHED_INSTRUMENTED_BENCHMARK", accepted_speedup_count=1,
            cpu_seconds=values, gpu_seconds=gpu_values,
            cpu_median_seconds=statistics.median(values),
            gpu_median_seconds=statistics.median(gpu_values),
            cpu_over_gpu_ratio=statistics.median(values)/statistics.median(gpu_values),
            gpu_producer_sha256=gpu_hash, gpu_manifest_sha256=digest(gpu_manifest_path),
            cpu_report_sha256=cpu_hashes, extra_trace_excluded_from_speedup=True,
            qualification_scope="HF-reference F-SAPT/aug-cc-pVDZ; CPU24 vs H200+8 host cores")
    except BaseException as error:
        report.update(status="FAILED_QUALIFICATION", accepted_speedup_count=0,
                      error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write(path, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    print("QUALIFICATION="+str(path)+"\nSHA256="+digest(path), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-root", type=Path, required=True)
    parser.add_argument("--cpu-roots", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    qualify(parser.parse_args())


if __name__ == "__main__":
    main()
