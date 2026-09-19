#!/usr/bin/env python3
"""How much of a SAPT(DFT) calculation is the automatic GRAC determination?

`SAPT_DFT_GRAC_COMPUTE=ITERATIVE` finds each monomer's shift from a neutral and
a doublet-cation SCF inside the timed `energy()` call. That is real work, and a
fixed-shift timing hides all of it, so a fixed-shift speedup describes a
calculation nobody can run without first paying a cost the timing omits. This
measures the cost directly from Psi4's own phase timers rather than by
differencing against a separate fixed-shift job, so it needs one run per arm and
carries no cross-job assumption.

The cation SCF is unrestricted, so it roughly doubles the J/K and XC work of the
neutral one; a monomer whose GRAC phase costs more than its own SAPT DFT phase
is expected, not a symptom.

The two shifts are reported separately because they are not interchangeable: on
an asymmetric dimer the larger monomer dominates, and a single pooled percentage
would hide which monomer sets the cost.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from thread_scaling import timer_wall

GRAC_A = "SAPT(DFT):GRAC Shift Monomer A"
GRAC_B = "SAPT(DFT):GRAC Shift Monomer B"


def case_cost(directory):
    """GRAC phase seconds for one case directory, against its total wall time.

    A run that declares ITERATIVE but has no GRAC timer is an error rather than
    a zero: the two mean opposite things, and silently reporting 0% would turn a
    broken measurement into the strongest possible claim.
    """
    record = json.loads((directory / "result.json").read_text())
    timers = directory / "timer.dat"
    a, b = timer_wall(timers, GRAC_A), timer_wall(timers, GRAC_B)
    iterative = record.get("grac_compute") == "ITERATIVE"
    if iterative and a is None and b is None:
        raise ValueError(f"{directory.name}: grac_compute=ITERATIVE but no GRAC phase timer")
    if not iterative and (a or b):
        raise ValueError(f"{directory.name}: grac_compute={record.get('grac_compute')!r} "
                         "but a GRAC phase timer is present")
    total = record["wall_s"]
    grac = (a or 0.0) + (b or 0.0)
    return {"case": directory.name, "system": record.get("system"),
            "basis": record.get("basis"), "mode": record.get("mode"),
            "threads": record.get("threads"), "grac_compute": record.get("grac_compute"),
            "total_wall_s": total, "grac_a_s": a, "grac_b_s": b, "grac_s": grac,
            "grac_fraction": grac / total if total else None}


def summarize(records):
    """Median per (system, basis, mode, threads); repeats of one arm collapse."""
    groups = defaultdict(list)
    for record in records:
        groups[(record["system"], record["basis"], record["mode"],
                record["threads"])].append(record)
    rows = []
    for (system, basis, mode, threads), entries in sorted(
            groups.items(), key=lambda item: [str(part) for part in item[0]]):
        def median(key):
            """A phase absent from every repeat stays absent, rather than becoming 0.0."""
            values = [e[key] for e in entries if e[key] is not None]
            return statistics.median(values) if values else None
        rows.append({"system": system, "basis": basis, "mode": mode, "threads": threads,
                     "repeats": len(entries),
                     "total_wall_s": median("total_wall_s"),
                     "grac_a_s": median("grac_a_s"), "grac_b_s": median("grac_b_s"),
                     "grac_s": median("grac_s"),
                     "grac_fraction": median("grac_fraction")})
    return rows


def speedups(rows):
    """CPU/GPU ratio of the GRAC phase beside the whole calculation's.

    The interesting question is not whether GRAC is expensive but whether the
    device removes it at the same rate as everything else. Only arms matched on
    thread count are paired; a cross-width ratio is not an accelerator speedup.
    """
    arms = {(r["system"], r["basis"], r["mode"], r["threads"]): r for r in rows}
    paired = []
    for key, cpu in sorted(arms.items(), key=lambda item: [str(p) for p in item[0]]):
        system, basis, mode, threads = key
        if mode != "cpu":
            continue
        gpu = arms.get((system, basis, "gpu", threads))
        if gpu is None:
            continue
        paired.append({"system": system, "basis": basis, "threads": threads,
                       "grac_speedup": cpu["grac_s"] / gpu["grac_s"] if gpu["grac_s"] else None,
                       "total_speedup": cpu["total_wall_s"] / gpu["total_wall_s"],
                       "cpu_grac_fraction": cpu["grac_fraction"],
                       "gpu_grac_fraction": gpu["grac_fraction"]})
    return paired


def markdown(payload):
    pct = lambda v: "—" if v is None else f"{100 * v:.1f}%"
    sec = lambda v: "—" if v is None else f"{v:.1f}"
    out = ["| System | Basis | Arm | Rep. | Total wall, s | GRAC A, s | GRAC B, s | GRAC total, s | % of wall |",
           "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in payload["summary"]:
        arm = f"{row['mode']} {row['threads']}T"
        out.append(f"| {row['system']} | {row['basis']} | {arm} | {row['repeats']} | "
                   f"{row['total_wall_s']:.2f} | {sec(row['grac_a_s'])} | {sec(row['grac_b_s'])} | "
                   f"{row['grac_s']:.2f} | {pct(row['grac_fraction'])} |")
    if payload["paired"]:
        out += ["", "| System | Basis | GRAC phase speedup | Whole-calculation speedup | GRAC % of CPU wall | GRAC % of GPU wall |",
                "|---|---|---:|---:|---:|---:|"]
        for row in payload["paired"]:
            rate = "—" if row["grac_speedup"] is None else f"{row['grac_speedup']:.2f}×"
            out.append(f"| {row['system']} | {row['basis']} | {rate} | "
                       f"{row['total_speedup']:.2f}× | {pct(row['cpu_grac_fraction'])} | "
                       f"{pct(row['gpu_grac_fraction'])} |")
    failed = [r for r in payload["cases"] if "error" in r]
    if failed:
        out += ["", "Not measured:", ""] + [f"- `{r['case']}`: {r['error']}" for r in failed]
    out += ["", "Medians over repeats, from Psi4's `SAPT(DFT):GRAC Shift Monomer A/B` phase "
                "timers. The second table pairs arms only at equal thread counts, so its "
                "ratios are accelerator speedups rather than baseline-width effects."]
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", type=Path,
                        help="directories of per-case subdirectories")
    parser.add_argument("--output", type=Path, help="write JSON here")
    args = parser.parse_args()
    records = []
    for root in args.results:
        for directory in sorted(Path(root).iterdir()):
            if not (directory / "result.json").exists():
                continue
            try:
                records.append(case_cost(directory))
            except (ValueError, KeyError) as error:
                records.append({"case": directory.name,
                                "error": f"{type(error).__name__}: {error}"})
    good = [r for r in records if "error" not in r]
    if not good:
        raise SystemExit(f"no case with a GRAC phase timer under {args.results}")
    summary = summarize(good)
    payload = {"cases": records, "summary": summary, "paired": speedups(summary)}
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(markdown(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
