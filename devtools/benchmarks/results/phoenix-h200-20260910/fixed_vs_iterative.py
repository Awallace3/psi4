#!/usr/bin/env python3
"""Show what automatic GRAC does to the speedup, per case.

The fixed-shift side is frozen historical data from job 13024192 and is a
literal below; it will never be regenerated, because that protocol is
superseded. The ITERATIVE side is read from the current paired summary so this
table cannot drift from the rest of the report when the campaign is re-run.

Both jobs ran eight threads on a gpu-h200 node with the same CPU model and the
same binary, so the two protocols are compared like-for-like and the ratio of
their CPU times is the cost of automatic GRAC rather than a hardware difference.
"""
import argparse
import json
from pathlib import Path

# job 13024192, SAPT_DFT_GRAC_COMPUTE=NONE with --shift 0.136:
#   (system, basis) -> (cpu median s, gpu median s)
FIXED_SHIFT = {
    ("water", "cc-pvdz"): (4.53, 4.69),
    ("water", "aug-cc-pvdz"): (5.53, 4.82),
    ("benzene", "cc-pvdz"): (28.47, 9.82),
    ("benzene", "aug-cc-pvdz"): (72.08, 11.83),
    ("peptide", "6-31+g**"): (32.08, 11.49),
    ("nanotube", "6-31+g**"): (168.32, 22.17),
}


def rows(summary):
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
            "cpu_work_multiplier": row["wall_s"]["cpu"]["median"] / fixed_cpu,
        })
    return out


def markdown(rows):
    return "\n".join(
        f"| {r['system']} | {r['basis']} | {r['fixed_speedup']:.2f}× | "
        f"{r['iterative_speedup']:.2f}× | {r['cpu_work_multiplier']:.2f}× |"
        for r in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="paired/summary.json")
    args = parser.parse_args()
    table = rows(json.loads(args.summary.read_text()))
    if len(table) != len(FIXED_SHIFT):
        have = {(r["system"], r["basis"]) for r in table}
        raise SystemExit(f"no ITERATIVE result for: {sorted(set(FIXED_SHIFT) - have)}")
    print(markdown(table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
