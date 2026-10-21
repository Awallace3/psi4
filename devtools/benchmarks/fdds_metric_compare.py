"""Audit candidate/native/base controls; no dropping failed or missing samples."""
import json
import math
from pathlib import Path
import statistics
import sys


def audit(root, system, route="metric"):
    assert route in ("metric", "response")
    controls = ("base-gpu-0", "candidate-gpu-0", "candidate-cpu-0",
                "candidate-gpu-1", "base-gpu-1")
    records = {c: json.loads((root / f"{system}-{c}/result.json").read_text()) for c in controls}
    reference = records["candidate-cpu-0"]
    errors = {}
    for name, d in records.items():
        assert d["system"] == system, name
        assert d["ok"], name
        assert math.isfinite(d["fdds_seconds"]) and d["fdds_seconds"] > 0, name
        assert all(math.isfinite(v) for v in d["energies_Eh"].values()), name
        for key in ("cache_sha256", "metadata_sha256", "runner_sha256", "options",
                    "hostname", "job_id", "threads", "memory"):
            assert d[key] == reference[key], (name, key)
        # Different cores/import paths are intentional, dependencies are not.
        dependencies = lambda r: {p: h for p, h in r["libraries_sha256"].items()
                                  if not (Path(p).parent == Path(r["psi4_package"]).parent and
                                          Path(p).name.startswith("core.cpython"))}
        assert dependencies(d) == dependencies(reference), name
        if "-cpu-" in name:
            assert d["gpu_timer_calls"] == d["gpu_exchange_timer_calls"] == d["gpu_metric_timer_calls"] == 0
            assert d.get("gpu_response_timer_calls", 0) == 0
        else:
            assert d["gpu_timer_calls"] == 20 and d["gpu_exchange_timer_calls"] == 4
            if route == "metric":
                assert (d["gpu_metric_timer_calls"] > 0) == name.startswith("candidate")
            else:
                assert d["gpu_metric_timer_calls"] == 12
                assert d["gpu_response_timer_calls"] == (60 if name.startswith("candidate") else 0)
        errors[name] = {k: abs(d["energies_Eh"][k] - reference["energies_Eh"][k])
                        for k in reference["energies_Eh"]}
        assert max(errors[name].values()) <= 1.e-5, (name, errors[name])
    for prefix in ("base", "candidate"):
        a, b = (records[f"{prefix}-gpu-{i}"] for i in (0, 1))
        for key in ("binary_pin", "core_sha256", "psi4_package"):
            assert a[key] == b[key], (prefix, key)
    for key in ("binary_pin", "core_sha256", "psi4_package"):
        assert records["candidate-gpu-0"][key] == reference[key], key
    base_times = [records[f"base-gpu-{i}"]["fdds_seconds"] for i in (0, 1)]
    gpu_times = [records[f"candidate-gpu-{i}"]["fdds_seconds"] for i in (0, 1)]
    result = dict(system=system, route=route, seconds={k: d["fdds_seconds"] for k, d in records.items()},
                  base_gpu_over_candidate_gpu=statistics.median(base_times) / statistics.median(gpu_times),
                  candidate_cpu_over_gpu=reference["fdds_seconds"] / statistics.median(gpu_times),
                  errors_Eh=errors, settings_unchanged=True, controls=len(records))
    destination = root / f"{system}-{route.upper()}_CANDIDATE_VERIFIED.json"
    assert not destination.exists()
    temporary = destination.with_suffix(".tmp")
    with temporary.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    temporary.replace(destination)
    return result


if __name__ == "__main__":
    print(json.dumps(audit(Path(sys.argv[1]), sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "metric"), indent=2))
