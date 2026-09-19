#!/usr/bin/env python3
"""Was this job's host as fast as that job's host?

Two campaigns on the same partition, same CPU model, same `core.so` sha256 and
byte-identical geometries can still differ by a factor of three, because a
scheduler allocation is not a performance contract. Job 13060539 and job
13024192 did exactly that: phases doing provably identical work (`Dimer SCF`,
`Monomer A SCF`, both one call) burned 3.2-3.5x the CPU-seconds in the former,
while the cuEST kernels were unchanged to within 1%. A CPU arm is the
denominator of every speedup in a paired campaign, so a degraded host inflates
every number in the report and nothing in the tree says so.

`common.inc` now runs `cpu_probe.py` inside each allocation and writes
`metadata/canary-<phase>-t<threads>.json`. This reads those back, so a host
deficit is read off the tree rather than discovered by differencing two
campaigns after the fact.

The three probes fail differently and are all reported: `dgemm` is the BLAS
path DF-K lives in, `stream` is bandwidth per core, `scalar` is a serial
interpreter loop that tracks clock times IPC. A uniform ratio means clock; a
ratio confined to one probe means that probe's resource. `live_mhz` is the
observed clock of the allocated cores and settles it directly when present.
"""
import argparse
from pathlib import Path
import json
import statistics

# A tree measured before the canary existed. Named rather than treated as
# agreeing, because "no evidence of a deficit" and "evidence of no deficit"
# are the two readings this whole module exists to separate.
UNCERTIFIED = "uncertified"

# Run-to-run scatter on a quiet node is a couple of percent; the deficit that
# motivated this was 200%. Anything past this is worth a human looking, and
# nothing inside it would change a conclusion.
TOLERANCE = 1.25


def canaries(tree):
    """Canary records for one job tree, keyed by (phase, threads).

    Accepts either the job root or its `results/` directory, because the
    analysis tools are handed the latter and the metadata lives beside it.
    """
    tree = Path(tree)
    for candidate in (tree / "metadata", tree.parent / "metadata"):
        if candidate.is_dir():
            break
    else:
        return {}
    found = {}
    for path in sorted(candidate.glob("canary-*-t*.json")):
        phase, _, threads = path.stem[len("canary-"):].rpartition("-t")
        try:
            found[(phase, int(threads))] = json.loads(path.read_text())
        except (ValueError, json.JSONDecodeError):
            continue
    return found


def metrics(record):
    """The comparable throughputs from one probe run, higher being faster."""
    out = {}
    if isinstance(record.get("dgemm_per_core_gflops"), (int, float)):
        out["dgemm_gflops_per_core"] = record["dgemm_per_core_gflops"]
    for key, field, name in (("stream", "gb_s", "stream_gb_s"),
                             ("scalar", "miter_s", "scalar_miter_s")):
        value = (record.get(key) or {}).get(field)
        if isinstance(value, (int, float)):
            out[name] = value
    live = (record.get("cpu") or {}).get("live_mhz_mine")
    if live:
        out["live_mhz"] = statistics.median(live)
    return out


def profile(tree):
    """One tree's host speed: the metrics of its widest start-phase probe.

    The widest probe is the one whose shape matches the campaign, and the start
    phase is the one that was in effect when the first case ran. A tree whose
    start and end canaries disagree is flagged rather than averaged, since a
    host that changed speed mid-campaign makes its own cases incomparable.
    """
    found = canaries(tree)
    if not found:
        return {"tree": str(tree), "status": UNCERTIFIED}
    phases = {phase for phase, _ in found}
    threads = max(t for _, t in found)
    start = "start" if "start" in phases else sorted(phases)[0]
    record = found[(start, threads)] if (start, threads) in found else \
        found[max(found, key=lambda k: k[1])]
    result = {"tree": str(tree), "status": "measured", "threads": threads,
              "node": record.get("node"), "job": record.get("job"),
              "phases": sorted(phases), **metrics(record)}
    if "end" in phases and ("end", threads) in found:
        drift = ratios(metrics(record), metrics(found[("end", threads)]))
        worst = max(drift.values(), default=1.0)
        if worst > TOLERANCE:
            result["drifted_during_run"] = drift
    return result


def ratios(fast, slow):
    """Per-metric ratio of two profiles, always >= 1, on shared metrics only."""
    out = {}
    for key in sorted(set(fast) & set(slow)):
        a, b = fast[key], slow[key]
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a and b:
            out[key] = max(a / b, b / a)
    return out


def compare(trees):
    """Do these trees' hosts agree closely enough to pool their timings?

    Uncertified trees are reported as uncertified rather than as agreeing: the
    absence of a canary is exactly the state that let a 3x deficit through.
    """
    profiles = [profile(tree) for tree in trees]
    measured = [p for p in profiles if p["status"] == "measured"]
    uncertified = [p["tree"] for p in profiles if p["status"] == UNCERTIFIED]
    worst, spread = 1.0, {}
    for index, first in enumerate(measured):
        for second in measured[index + 1:]:
            for key, value in ratios(first, second).items():
                spread[key] = max(spread.get(key, 1.0), value)
                worst = max(worst, value)
    if uncertified:
        verdict = UNCERTIFIED
    elif worst > TOLERANCE:
        verdict = "mismatched"
    else:
        verdict = "matched"
    return {"profiles": profiles, "uncertified": uncertified,
            "worst_ratio": worst, "spread": spread, "verdict": verdict,
            "tolerance": TOLERANCE}


def label(tree):
    """Name a tree by its job directory, not by the `results` leaf.

    The analysis tools are handed `<job>/results`, and four rows all reading
    "results" identify nothing — which matters most here, where the whole point
    is telling one allocation apart from another.
    """
    path = Path(tree)
    if path.name in ("results", "") and path.parent.name:
        return path.parent.name
    return path.name or str(tree)


def markdown(payload):
    num = lambda v: "—" if not isinstance(v, (int, float)) else f"{v:.1f}"
    out = ["| Tree | Node | Threads | DGEMM GF/s per core | Triad GB/s | Scalar Miter/s | Live MHz |",
           "|---|---|---:|---:|---:|---:|---:|"]
    for row in payload["profiles"]:
        name = label(row["tree"])
        if row["status"] == UNCERTIFIED:
            out.append(f"| {name} | — | — | — | — | — | — |")
            continue
        out.append(f"| {name} | {row.get('node') or '—'} | {row['threads']} | "
                   f"{num(row.get('dgemm_gflops_per_core'))} | {num(row.get('stream_gb_s'))} | "
                   f"{num(row.get('scalar_miter_s'))} | {num(row.get('live_mhz'))} |")
    out += ["", f"Verdict: **{payload['verdict']}** "
                f"(worst pairwise ratio {payload['worst_ratio']:.2f}×, "
                f"tolerance {payload['tolerance']}×)."]
    if payload["uncertified"]:
        out += ["", "Uncertified — measured before the canary existed, so their host "
                    "speed is unknown rather than confirmed equal:", ""]
        out += [f"- `{label(t)}`" for t in payload["uncertified"]]
    drifted = [r for r in payload["profiles"] if r.get("drifted_during_run")]
    if drifted:
        out += ["", "Host speed changed between the start and end of the run, so this "
                    "tree's own cases are not mutually comparable:", ""]
        out += [f"- `{label(r['tree'])}`: " +
                ", ".join(f"{k} {v:.2f}×" for k, v in sorted(r["drifted_during_run"].items()))
                for r in drifted]
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trees", nargs="+", type=Path,
                        help="job trees, or their results/ directories")
    parser.add_argument("--output", type=Path, help="write JSON here")
    parser.add_argument("--require-match", action="store_true",
                        help="exit nonzero unless every tree is certified and agrees")
    args = parser.parse_args()
    payload = compare(args.trees)
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(markdown(payload))
    if args.require_match and payload["verdict"] != "matched":
        raise SystemExit(f"host speed verdict: {payload['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
