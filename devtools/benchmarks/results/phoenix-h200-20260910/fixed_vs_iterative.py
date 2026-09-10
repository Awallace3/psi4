#!/usr/bin/env python3
"""What automatic GRAC costs, and what the fixed-shift protocol left out.

The fixed-shift side is frozen historical data from job 13024192 and is a
literal below; it will never be regenerated, because that protocol is
superseded. The ITERATIVE side is read from the current paired summary so this
table cannot drift from the rest of the report when the campaign is re-run.

**The two jobs did not run at the same host speed.** Job 13024192
(atl1-1-02-014-9-0) and job 13060539 (atl1-1-02-012-23-0) are both gpu-h200
allocations of the same Xeon Platinum 8562Y+, running the same `core.so`
(sha256 bc7b9620cd41...) on byte-identical geometries with identical SCF
iteration counts — and the second host was 3.2-3.5x slower on phases doing
provably identical work (`SAPT(DFT):Dimer SCF` and `Monomer A SCF`, one call
each, in both arms), while the cuEST kernels were unchanged to within 1%.

So the two speedup columns are each a valid same-host ratio for their own job,
and **their difference is not attributable to the protocol**: it mixes the
protocol change with a factor-of-three host change. An earlier version of this
script computed a "CPU work multiplier" by dividing the two jobs' CPU times and
called the result the cost of automatic GRAC. That quantity was mostly the host
deficit and has been removed.

Its replacement is measured inside a single job, from Psi4's own
`SAPT(DFT):GRAC Shift Monomer A/B` phase timers, so it carries no cross-job
assumption at all. See `grac_cost.py`.
"""
import argparse
import json
from pathlib import Path

# job 13024192, SAPT_DFT_GRAC_COMPUTE=NONE with --shift 0.136, on
# atl1-1-02-014-9-0: (system, basis) -> (cpu median s, gpu median s)
FIXED_SHIFT = {
    ("water", "cc-pvdz"): (4.53, 4.69),
    ("water", "aug-cc-pvdz"): (5.53, 4.82),
    ("benzene", "cc-pvdz"): (28.47, 9.82),
    ("benzene", "aug-cc-pvdz"): (72.08, 11.83),
    ("peptide", "6-31+g**"): (32.08, 11.49),
    ("nanotube", "6-31+g**"): (168.32, 22.17),
}


def grac_fractions(payload):
    """CPU-arm GRAC share of wall, per (system, basis), from the same job."""
    if payload is None:
        return {}
    return {(row["system"], row["basis"]): row["grac_fraction"]
            for row in payload["summary"] if row["mode"] == "cpu"}


def rows(summary, grac=None):
    fractions = grac_fractions(grac)
    out = []
    for row in summary["rows"]:
        key = (row["system"], row["basis"])
        if key not in FIXED_SHIFT:
            continue
        fixed_cpu, fixed_gpu = FIXED_SHIFT[key]
        out.append({
            "system": row["system"], "basis": row["basis"],
            "fixed_speedup": fixed_cpu / fixed_gpu,
            "iterative_speedup": row["speedup"],
            "grac_fraction": fractions.get(key),
        })
    return out


def markdown(rows):
    pct = lambda v: "—" if v is None else f"{100 * v:.0f}%"
    return "\n".join(
        f"| {r['system']} | {r['basis']} | {r['fixed_speedup']:.2f}× | "
        f"{r['iterative_speedup']:.2f}× | {pct(r['grac_fraction'])} |"
        for r in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summary", type=Path, help="paired/summary.json")
    parser.add_argument("--grac-cost", type=Path,
                        help="grac_cost.py JSON, for the within-job GRAC share")
    args = parser.parse_args()
    grac = json.loads(args.grac_cost.read_text()) if args.grac_cost else None
    table = rows(json.loads(args.summary.read_text()), grac)
    if len(table) != len(FIXED_SHIFT):
        have = {(r["system"], r["basis"]) for r in table}
        raise SystemExit(f"no ITERATIVE result for: {sorted(set(FIXED_SHIFT) - have)}")
    print(markdown(table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
