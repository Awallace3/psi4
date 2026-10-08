#!/usr/bin/env python3
"""Replay attributable worker results; keep failures/missing cases visible."""
import argparse
import hashlib
import json
from pathlib import Path

from mgga_ie import atomic_json, process_result, summarize


def replay(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    results = {}
    failures = []
    for record in manifest["records"]:
        file = directory / record["name"] / "result.json"
        if file.exists():
            result = process_result(json.loads(file.read_text()), record["returncode"])
            key = tuple(result[k] for k in ("system", "functional", "route", "fragment"))
            # Build labels are part of the campaign name, not the worker result.
            build = next(b for b in manifest["packages"] if
                         record["name"] == "-".join((*key[:2], b, *key[2:])))
            results[(build, *key)] = result
            if not result["ok"]:
                failures.append(dict(**record, error=result.get("error"),
                                     reported_ok=result.get("calculation_reported_ok", False)))
        else:
            failures.append(dict(**record, error="Missing result.json"))
    rows = [r for r in summarize(results)
            if r["system"] in {c[0] for c in manifest["cases"]}
            and r["functional"] in manifest["functionals"]]
    stored = json.loads((directory / "comparisons.json").read_text())
    assert rows == stored, f"Archived comparisons differ from worker replay: {directory}"
    return rows, failures


def consolidate(matrix, recovery):
    main, failures = replay(matrix)
    recovered, recovery_failures = replay(recovery)
    key = lambda r: tuple(r[k] for k in ("build", "system", "functional", "route"))
    combined = {key(r): r for r in main}
    superseded = []
    for row in recovered:
        previous = combined.get(key(row))
        assert row["complete"], "Recovery is incomplete"
        assert previous is not None and not previous["complete"], "Unexpected recovery overlap"
        superseded.append(previous)
        combined[key(row)] = row
    rows = list(combined.values())
    complete = [r for r in rows if r["complete"]]
    stats = {}
    for route in ("gpu-jk", "gpu-xc"):
        selected = [r for r in complete if r["route"] == route]
        stats[route] = dict(count=len(selected),
                           max_ie_error_hartree=max(abs(r["delta_ie_hartree"]) for r in selected),
                           max_total_error_hartree=max(r["max_total_error_hartree"] for r in selected))
    return dict(comparisons=rows, complete_count=len(complete),
                incomplete=[r for r in rows if not r["complete"]],
                ie_failures=[r for r in complete if not r["within_ie_1e_6_Eh"]],
                original_process_failures=failures, recovery_process_failures=recovery_failures,
                superseded_incomplete=superseded, route_statistics=stats,
                sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                         for d in (matrix, recovery)
                         for p in (d / "manifest.json", d / "comparisons.json")},
                note="Main matrix final provenance verification was interrupted; "
                     "recovery supersedes four incomplete rows, not the original failure evidence.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--assert-parity", action="store_true",
                   help="Exit 1 for any incomplete or >1e-6 Eh IE case (a replay regression signal).")
    a = p.parse_args()
    assert not a.output.exists(), "Do not overwrite evidence"
    report = consolidate(a.archive / "20261008-matrix/results-13884503",
                         a.archive / "20261008-r2scan-recovery/results-13885267")
    atomic_json(a.output, report)
    print(json.dumps({k: report[k] for k in ("complete_count", "route_statistics")}, indent=2))
    print(f'{len(report["ie_failures"])} failed IE comparisons; '
          f'{len(report["incomplete"])} incomplete comparisons')
    return int(a.assert_parity and bool(report["ie_failures"] or report["incomplete"]))


if __name__ == "__main__":
    raise SystemExit(main())
