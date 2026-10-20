#!/usr/bin/env python3
"""Audit completed matched physical FDDS replays, not cross-node baselines."""
import argparse
import json
import math
from pathlib import Path
from statistics import median


def compare(rows, repeats):
    assert len(rows) == 2 * repeats
    assert all(r["ok"] for r in rows)
    for key in ("system", "cache_sha256", "metadata_sha256", "options",
                "binary_pin", "core_sha256", "runner_sha256", "threads", "memory",
                "hostname", "job_id"):
        assert all(r[key] == rows[0][key] for r in rows), key
    expected = {"Disp20", "Disp20,FDDS (unc)"}
    assert all(set(r["energies_Eh"]) == expected for r in rows)
    assert all(math.isfinite(v) for r in rows for v in r["energies_Eh"].values())
    assert all(math.isfinite(r["fdds_seconds"]) and r["fdds_seconds"] > 0 for r in rows)
    times = {}
    for arm in ("cpu", "gpu"):
        arm_rows = [r for r in rows if r["arm"] == arm]
        assert len(arm_rows) == repeats
        assert all((r["gpu_timer_calls"] > 0) == (arm == "gpu") for r in arm_rows)
        assert all(r["gpu_exchange_timer_calls"] == (4 if arm == "gpu" else 0)
                   for r in arm_rows)
        times[arm] = [r["fdds_seconds"] for r in arm_rows]
    errors = {key: max(abs(r["energies_Eh"][key] - rows[0]["energies_Eh"][key])
                       for r in rows) for key in sorted(expected)}
    assert max(errors.values()) <= 1.e-5, errors
    return {"seconds": times, "cpu_over_gpu": median(times["cpu"]) / median(times["gpu"]),
            "max_errors_Eh": errors, "repeats_per_arm": repeats}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--first-pair", action="store_true")
    parser.add_argument("--system", choices=("peptide", "nanotube", "protein83"))
    args = parser.parse_args()
    repeats = 1 if args.first_pair else 2
    summary = {}
    for system in ([args.system] if args.system else ("peptide", "nanotube", "protein83")):
        rows = [json.loads((args.root / f"{system}-{arm}-{i}" / "result.json").read_text())
                for arm in ("cpu", "gpu") for i in range(repeats)]
        assert all(r["system"] == system for r in rows)
        summary[system] = compare(rows, repeats)
    name = "FIRST_PAIR_VERIFIED.json" if args.first_pair else "VERIFIED.json"
    if args.system:
        name = args.system + "-" + name
    path = args.root / name
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2) + "\n")
    tmp.replace(path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
