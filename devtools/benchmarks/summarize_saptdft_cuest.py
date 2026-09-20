#!/usr/bin/env python3
"""Summarize completed paired measurements without hiding failed/missing runs."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

HARTREE_TO_KCAL_MOL = 627.5094740631


def _stat(values, function):
    """Apply a statistic only where every repeat reported the quantity.

    A median over the subset that happens to carry memory would silently mix
    instrumented and uninstrumented repeats, so a partially reported quantity is
    reported as absent instead.
    """
    values = list(values)
    if not values or any(v is None for v in values):
        return None
    return function(values)


def _memory_block(cpu, gpu):
    """Host peak for both arms, device peak for the GPU arm, plus how each was obtained.

    Every field is None on results recorded before the campaign measured memory,
    which is what keeps this summarizer able to read the earlier runs.
    """
    def host(values):
        peaks = [v.get("host_memory", {}).get("peak_rss_mib") for v in values]
        timed_only = [v.get("host_memory", {}).get("peak_covers_timed_region_only") for v in values]
        return {"peak_rss_mib": {"median": _stat(peaks, statistics.median),
                                 "min": _stat(peaks, min), "max": _stat(peaks, max)},
                # A peak the kernel never reset covers the whole process, imports
                # included, and so is not comparable with one that was reset.
                "peak_covers_timed_region_only": all(timed_only) if timed_only and
                all(v is not None for v in timed_only) else None}

    device = [v.get("device_memory", {}) for v in gpu]
    device_peaks = [d.get("peak_mib") for d in device]
    sources = {d.get("source") for d in device if d.get("source")}
    return {
        "host_peak_rss_mib": {"cpu": host(cpu), "gpu": host(gpu)},
        "device_peak_mib": {"median": _stat(device_peaks, statistics.median),
                            "min": _stat(device_peaks, min), "max": _stat(device_peaks, max)},
        # Device-wide accounting includes any other tenant on the GPU; per-process
        # does not. Mixing the two across repeats would make the median meaningless.
        "device_source": sources.pop() if len(sources) == 1 else (sorted(sources) or None),
        "device_sample_interval_s": _stat([d.get("sample_interval_s") for d in device], min),
        "device_measurement": "sampled peak: a spike shorter than the sample interval is missed",
    }


def summarize(root, tolerance=1e-5):
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
        row["memory"] = _memory_block(cpu, gpu)
        row["nbf_monomer_a"] = cpu[0].get("nbf_monomer_a")
        row["nbf_monomer_b"] = cpu[0].get("nbf_monomer_b")
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


def _mib(block):
    """A median [min-max] MiB cell, or an em dash when the quantity was not recorded."""
    if not block or block.get("median") is None:
        return "—"
    return f"{block['median']:.0f} [{block['min']:.0f}–{block['max']:.0f}]"


def markdown(summary):
    text = ["# Phoenix cuEST GRAC timing and accuracy", "",
            "Status: " + ("complete" if summary["complete"] else "partial") + ".",
            "Wall time is the fresh-process `energy()` call, including backend initialization.",
            "Speedup is median CPU time / median GPU time; values below 1 mean GPU slowdown.", "",
            "| System | Basis | MonA own nbf | MonB own nbf | Dimer nbf | CPU/GPU n | CPU median [range], s | GPU median [range], s | Speedup | Max component Δ, Eh | Accuracy |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in summary["rows"]:
        c, g = row["wall_s"]["cpu"], row["wall_s"]["gpu"]
        text.append(f"| {row['system']} | {row['basis']} | {row.get('nbf_monomer_a') or '—'} | "
                    f"{row.get('nbf_monomer_b') or '—'} | {row['nbf']} | "
                    f"{row['repeats']['cpu']}/{row['repeats']['gpu']} | "
                    f"{c['median']:.2f} [{c['min']:.2f}–{c['max']:.2f}] | "
                    f"{g['median']:.2f} [{g['min']:.2f}–{g['max']:.2f}] | "
                    f"{row['speedup']:.2f}× | {row['max_abs_delta_hartree']:.3e} | "
                    f"{'PASS' if row['accuracy_pass'] else 'FAIL'} |")
    text += ["", "## Memory", "",
             "Host memory is the kernel's `VmHWM` high-water mark over the timed region, "
             "so it is exact rather than sampled. Device memory is polled and is a "
             "**sampled** peak: a spike shorter than the interval is missed.",
             "A blank cell means that quantity was not recorded for every repeat of that arm.", "",
             "| System | Basis | CPU host peak, MiB | GPU host peak, MiB | GPU device peak, MiB | Device accounting | Poll, s | Peak scope |",
             "|---|---|---:|---:|---:|---|---:|---|"]
    for row in summary["rows"]:
        memory = row.get("memory") or {}
        host = memory.get("host_peak_rss_mib") or {}
        cpu_host = (host.get("cpu") or {}).get("peak_rss_mib") or {}
        gpu_host = (host.get("gpu") or {}).get("peak_rss_mib") or {}
        device = memory.get("device_peak_mib") or {}
        timed_only = (host.get("gpu") or {}).get("peak_covers_timed_region_only")
        scope = {True: "timed region", False: "whole process", None: "—"}[timed_only]
        source = memory.get("device_source")
        interval = memory.get("device_sample_interval_s")
        text.append(f"| {row['system']} | {row['basis']} | {_mib(cpu_host)} | {_mib(gpu_host)} | "
                    f"{_mib(device)} | {source if isinstance(source, str) else '—'} | "
                    f"{interval if interval is not None else '—'} | {scope} |")
    text += ["", "`whole process` means the kernel did not honor the high-water-mark reset, "
             "so that figure also covers Python imports and basis construction and is not "
             "comparable with a `timed region` one.",
             "", "Monomer columns give each fragment's own-basis size. The SAPT monomer SCFs use the dimer basis (ghosted partner), so their actual SCF basis size is the dimer column.",
             "", f"Accuracy threshold: {summary['accuracy_tolerance_hartree']:.1e} Eh for every component and paired repeat.",
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
