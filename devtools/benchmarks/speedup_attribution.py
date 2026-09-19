#!/usr/bin/env python3
"""Attribute a SAPT(DFT) CPU->GPU speedup to the components that produced it.

A vendor DF-K speedup and this PR's end-to-end SAPT speedup are different
quantities, and quoting one against the other invites the reader to assume the
exchange build is what got faster. It mostly is not: in this workload the DFT
exchange-correlation potential build costs several times what DF J/K costs, so
most of the saved time comes from XC, not from DF-K.

This splits each arm's wall time into DF J/K, XC potential, and remainder using
Psi4's flat timers, then reports how much of the measured saving each part
contributed and what an infinitely fast DF-K alone could ever have bought
(Amdahl's bound). That bound is the honest way to compare against a DF-K-only
claim.

`JK: JK`, `RV: Form V`, and `UV: Form V` are disjoint: J/K and the potential
build are separate phases, and the restricted and unrestricted potentials are
separate objects (ITERATIVE GRAC adds unrestricted cation SCFs). They are not
strictly nested inside `HF: Form G`, since some potential evaluations occur
outside the Fock build, so the sum is checked against total wall rather than
against Form G.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import statistics

from analyze_gpu_profile import flat_timers

NAME = re.compile(r"^(?P<stem>.+)-(?P<mode>cpu|gpu)-(?P<repeat>\d+)$")
JK_TIMERS = ("JK: JK",)
XC_TIMERS = ("RV: Form V", "UV: Form V")


def component_times(directory):
    """DF J/K, XC, and total wall seconds for one case directory."""
    record = json.loads((directory / "result.json").read_text())
    if not record.get("ok"):
        return None
    timers = flat_timers(directory / "timer.dat")
    total = record["wall_s"]
    jk = sum(timers.get(name, {}).get("wall_s", 0.0) for name in JK_TIMERS)
    xc = sum(timers.get(name, {}).get("wall_s", 0.0) for name in XC_TIMERS)
    if jk + xc > total:
        raise ValueError(f"{directory.name}: J/K + XC ({jk + xc:.1f}s) exceeds "
                         f"total wall ({total:.1f}s); timers are not disjoint")
    return {"total_s": total, "jk_s": jk, "xc_s": xc, "other_s": total - jk - xc,
            "mode": record["mode"], "threads": record["threads"]}


def load(results):
    cases = defaultdict(lambda: defaultdict(list))
    for path in sorted(Path(results).glob("*/timer.dat")):
        match = NAME.match(path.parent.name)
        if not match:
            continue
        parts = component_times(path.parent)
        if parts:
            cases[match.group("stem")][match.group("mode")].append(parts)
    return cases


def attribute(cpu, gpu):
    """Split the saved seconds across components, plus the DF-K-only bound."""
    med = lambda rows, key: statistics.median(row[key] for row in rows)
    fields = ("total_s", "jk_s", "xc_s", "other_s")
    c = {key: med(cpu, key) for key in fields}
    g = {key: med(gpu, key) for key in fields}
    saved = c["total_s"] - g["total_s"]
    row = {"cpu": c, "gpu": g, "repeats": {"cpu": len(cpu), "gpu": len(gpu)},
           "speedup": c["total_s"] / g["total_s"],
           "saved_s": saved,
           # An infinitely fast DF-K removes cpu jk_s and nothing else.
           "amdahl_bound_from_dfk_alone": c["total_s"] / (c["total_s"] - c["jk_s"]),
           "dfk_share_of_cpu_time": c["jk_s"] / c["total_s"],
           "xc_share_of_cpu_time": c["xc_s"] / c["total_s"]}
    for part in ("jk", "xc", "other"):
        delta = c[f"{part}_s"] - g[f"{part}_s"]
        row[f"{part}_speedup"] = (c[f"{part}_s"] / g[f"{part}_s"]
                                  if g[f"{part}_s"] else None)
        row[f"{part}_saved_s"] = delta
        row[f"{part}_share_of_saving"] = delta / saved if saved else None
    return row


def analyze(results):
    rows = {}
    for stem, arms in sorted(load(results).items()):
        if arms.get("cpu") and arms.get("gpu"):
            rows[stem] = attribute(arms["cpu"], arms["gpu"])
    return rows


def markdown(rows):
    out = ["| Case | CPU s | GPU s | Speedup | DF-K speedup | XC speedup | DF-K share of saving | "
           "XC share of saving | Max speedup from DF-K alone |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    pct = lambda v: "—" if v is None else f"{v * 100:.0f}%"
    mul = lambda v: "—" if v is None else f"{v:.1f}×"
    for stem, r in rows.items():
        out.append(f"| {stem} | {r['cpu']['total_s']:.1f} | {r['gpu']['total_s']:.1f} | "
                   f"{r['speedup']:.2f}× | {mul(r['jk_speedup'])} | {mul(r['xc_speedup'])} | "
                   f"{pct(r['jk_share_of_saving'])} | {pct(r['xc_share_of_saving'])} | "
                   f"{r['amdahl_bound_from_dfk_alone']:.2f}× |")
    out += ["", "The last column is Amdahl's bound: the end-to-end speedup that would result if "
                "DF J/K took zero time and nothing else changed. A vendor DF-K speedup cannot "
                "produce more than this on this workload, whatever its magnitude."]
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = analyze(args.results)
    if not rows:
        raise SystemExit(f"no paired cpu/gpu case with timers under {args.results}")
    print(markdown(rows))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
