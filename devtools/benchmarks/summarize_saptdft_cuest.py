#!/usr/bin/env python3
"""Summarize completed paired measurements without hiding failed/missing runs."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

HARTREE_TO_KCAL_MOL = 627.5094740631


def summarize(root, tolerance=1e-6):
    root = Path(root)
    campaign = json.loads((root / "campaign.json").read_text())
    groups = defaultdict(lambda: defaultdict(list))
    failures = []
    for entry in campaign["records"]:
        path = root / entry["name"] / "result.json"
        if entry["returncode"] != 0 or not path.exists():
            failures.append(entry)
            continue
        result = json.loads(path.read_text())
        if not result["ok"]:
            failures.append({**entry, "error": result.get("error")})
            continue
        if not math.isfinite(result["wall_s"]) or result["wall_s"] <= 0:
            raise ValueError(f"Invalid timing: {path}")
        if not all(math.isfinite(v) for v in result["components_hartree"].values()):
            raise ValueError(f"Nonfinite energy: {path}")
        result["repeat"] = int(entry["name"].rsplit("-", 1)[1])
        groups[(result["system"], result["basis"])][result["mode"]].append(result)
    rows = []
    for (system, basis), modes in groups.items():
        cpu, gpu = modes.get("cpu", []), modes.get("gpu", [])
        if not cpu or not gpu:
            failures.append({"system": system, "basis": basis, "error": "Missing paired backend"})
            continue
        by_repeat = {v["repeat"]: v for v in cpu}
        pairs = [(by_repeat[v["repeat"]], v) for v in gpu if v["repeat"] in by_repeat]
        if not pairs:
            failures.append({"system": system, "basis": basis, "error": "No matching repeats"})
            continue
        for c, g in pairs:
            assert c["nbf"] == g["nbf"]
            assert c["geometry"] == g["geometry"]
            assert c["threads"] == g["threads"]
            assert c["psi4_version"] == g["psi4_version"]
            assert {k: v for k, v in c["options"].items() if k != "USE_CUEST"} == {
                k: v for k, v in g["options"].items() if k != "USE_CUEST"}
        times = {mode: [v["wall_s"] for v in values] for mode, values in modes.items()}
        components = {}
        for key in cpu[0]["components_hartree"]:
            deltas = [g["components_hartree"][key] - c["components_hartree"][key] for c, g in pairs]
            components[key] = {
                "cpu_median_hartree": statistics.median(v["components_hartree"][key] for v in cpu),
                "gpu_median_hartree": statistics.median(v["components_hartree"][key] for v in gpu),
                "max_abs_delta_hartree": max(map(abs, deltas)),
                "paired_deltas_hartree": deltas,
            }
        worst = max(v["max_abs_delta_hartree"] for v in components.values())
        row = {"system": system, "basis": basis, "nbf": cpu[0]["nbf"],
               "repeats": {m: len(v) for m, v in modes.items()}, "paired_repeats": len(pairs),
               "wall_s": {m: {"median": statistics.median(v), "min": min(v), "max": max(v)}
                          for m, v in times.items()},
               "speedup": statistics.median(times["cpu"]) / statistics.median(times["gpu"]),
               "components": components, "max_abs_delta_hartree": worst,
               "max_abs_delta_kcal_mol": worst * HARTREE_TO_KCAL_MOL,
               "accuracy_pass": worst <= tolerance}
        rows.append(row)
    marker = root / "COMPLETE.json"
    completion = json.loads(marker.read_text()) if marker.exists() else {}
    expected_count = (sum(len(bases) for _, bases in campaign["cases"]) * 2 * campaign["repeats"]
                      if "cases" in campaign else len(campaign["records"]))
    complete = (completion.get("ok") is True and
                completion.get("count") == len(campaign["records"]) == expected_count)
    return {"complete": complete, "requested_repeats": campaign["repeats"],
            "accuracy_tolerance_hartree": tolerance, "rows": rows, "failures": failures,
            "all_pass": bool(rows) and complete and not failures and
            all(r["accuracy_pass"] and r["paired_repeats"] == campaign["repeats"] for r in rows)}


def markdown(summary):
    text = ["# Phoenix cuEST GRAC timing and accuracy", "",
            "Status: " + ("complete" if summary["complete"] else "partial") + ".",
            "Wall time is the fresh-process `energy()` call, including backend initialization.",
            "Speedup is median CPU time / median GPU time; values below 1 mean GPU slowdown.", "",
            "| System | Basis | nbf | CPU/GPU n | CPU median [range], s | GPU median [range], s | Speedup | Max component Δ, Eh | Accuracy |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in summary["rows"]:
        c, g = row["wall_s"]["cpu"], row["wall_s"]["gpu"]
        text.append(f"| {row['system']} | {row['basis']} | {row['nbf']} | "
                    f"{row['repeats']['cpu']}/{row['repeats']['gpu']} | "
                    f"{c['median']:.2f} [{c['min']:.2f}–{c['max']:.2f}] | "
                    f"{g['median']:.2f} [{g['min']:.2f}–{g['max']:.2f}] | "
                    f"{row['speedup']:.2f}× | {row['max_abs_delta_hartree']:.3e} | "
                    f"{'PASS' if row['accuracy_pass'] else 'FAIL'} |")
    text += ["", f"Accuracy threshold: {summary['accuracy_tolerance_hartree']:.1e} Eh for every component and paired repeat.",
             "", "## Component accuracy", "",
             "| System / basis | Component | CPU median, Eh | GPU median, Eh | Max paired absolute Δ, Eh |",
             "|---|---|---:|---:|---:|"]
    for row in summary["rows"]:
        for name, values in row["components"].items():
            text.append(f"| {row['system']} / {row['basis']} | {name} | "
                        f"{values['cpu_median_hartree']:.12f} | {values['gpu_median_hartree']:.12f} | "
                        f"{values['max_abs_delta_hartree']:.3e} |")
    text += ["", "## Failed or incomplete measurements", "",
             "```json", json.dumps(summary["failures"], indent=2), "```", ""]
    return "\n".join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.results)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "summary.md").write_text(markdown(summary))
    print(f"rows={len(summary['rows'])} complete={summary['complete']} all_pass={summary['all_pass']}")
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
