"""Gather per-rank records from a GTFock scaling sweep into one table.

:source:`tests/pytests/gtfock_hpc_benchmark.py` writes one JSON file per rank per
point. This reduces them to one row per point and writes a CSV plus a JSON, so
the numbers that reach the documentation are the ones the machine produced::

    python gtfock_hpc_collect.py results/peptide_12395891 results/nanotube_12395892 \\
        --csv gtfock_hpc_results.csv --json gtfock_hpc_results.json

Reducing across ranks is not averaging. A parallel run takes as long as its
slowest rank, so the wall clocks are maxima; peak memory is reported both as the
worst rank (what a single process needed) and as the sum over ranks (what the
node needed), because those two answer different questions and only the second
one shows replicated storage getting more expensive with rank count.
"""

import argparse
import csv
import glob
import json
import os
import sys

# One row per point. Written in this order so the CSV reads left to right as
# "what was run", "what it cost", "was it the same answer".
_COLUMNS = [
    "system", "basis", "method", "nbf", "nshell", "puream",
    "arm", "jk_name", "ranks", "threads_per_rank", "total_cores",
    "iterations", "jk_calls",
    "scf_wall_s", "jk_wall_s",
    "speedup_vs_gtfock_n1", "jk_speedup_vs_gtfock_n1", "speedup_vs_direct",
    "peak_rss_max_mb", "peak_rss_sum_mb",
    "scf_energy", "dE_vs_direct_eh", "dE_across_ranks_eh",
    "process_grid", "local_task_shape",
    "host", "slurm_job_id", "slurm_nodelist",
]


def load_points(directories):
    """Collapse ``*.rank<N>.json`` files into one dict per (system, arm, ranks)."""
    by_key = {}
    for directory in directories:
        pattern = os.path.join(directory, "*.rank*.json")
        for path in sorted(glob.glob(pattern)):
            with open(path) as handle:
                record = json.load(handle)
            key = (record["system"], record["arm"], record["ranks"])
            by_key.setdefault(key, []).append(record)
    if not by_key:
        raise SystemExit(f"no *.rank*.json found under {', '.join(directories)}")

    points = []
    for (system, arm, ranks), records in sorted(by_key.items()):
        if len(records) != ranks:
            print(f"warning: {system}/{arm}/n{ranks} has {len(records)} rank "
                  f"records, expected {ranks}", file=sys.stderr)
        head = records[0]
        energies = [r["scf_energy"] for r in records]
        # An SCF that took a different number of iterations on different ranks
        # would mean the ranks disagreed about the density, so surface it rather
        # than reduce it away.
        iterations = sorted({r["iterations"] for r in records})
        point = {
            "system": system,
            "basis": head["basis"],
            "method": head["method"],
            "nbf": head["nbf"],
            "nshell": head["nshell"],
            "puream": head["puream"],
            "arm": arm,
            "jk_name": head["jk_name"],
            "ranks": ranks,
            "threads_per_rank": head["threads_per_rank"],
            "total_cores": head["total_cores"],
            "iterations": iterations[0] if len(iterations) == 1 else str(iterations),
            "jk_calls": head["jk_calls"],
            "scf_wall_s": max(r["scf_wall_seconds"] for r in records),
            "jk_wall_s": max(r["jk_wall_seconds"] for r in records),
            "peak_rss_max_mb": max(r["peak_rss_mb"] for r in records),
            "peak_rss_sum_mb": sum(r["peak_rss_mb"] for r in records),
            "scf_energy": head["scf_energy"],
            "dE_across_ranks_eh": max(energies) - min(energies),
            "process_grid": "x".join(str(n) for n in head["process_grid"]) if "process_grid" in head else "",
            "local_task_shape": "x".join(str(n) for n in head["local_task_shape"]) if "local_task_shape" in head else "",
            "host": head["host"],
            "slurm_job_id": head["slurm_job_id"],
            "slurm_nodelist": head["slurm_nodelist"],
        }
        points.append(point)
    return points


def add_derived(points):
    """Fill in the speedup and energy-difference columns, per system."""
    for system in {p["system"] for p in points}:
        rows = [p for p in points if p["system"] == system]
        base = next((p for p in rows if p["arm"] == "gtfock" and p["ranks"] == 1), None)
        direct = next((p for p in rows if p["arm"] == "direct"), None)
        for point in rows:
            # Speedup is only meaningful against the same algorithm at one rank;
            # for the reference arms it is left blank rather than invented.
            if base is not None and point["arm"] == "gtfock":
                point["speedup_vs_gtfock_n1"] = base["scf_wall_s"] / point["scf_wall_s"]
                point["jk_speedup_vs_gtfock_n1"] = base["jk_wall_s"] / point["jk_wall_s"]
            else:
                point["speedup_vs_gtfock_n1"] = ""
                point["jk_speedup_vs_gtfock_n1"] = ""
            if direct is not None and point is not direct:
                point["speedup_vs_direct"] = direct["scf_wall_s"] / point["scf_wall_s"]
                point["dE_vs_direct_eh"] = point["scf_energy"] - direct["scf_energy"]
            else:
                point["speedup_vs_direct"] = ""
                point["dE_vs_direct_eh"] = ""
    return points


def print_table(points):
    order = {"direct": 0, "df": 1, "pk": 2, "gtfock": 3}
    for system in sorted({p["system"] for p in points}):
        rows = sorted((p for p in points if p["system"] == system),
                      key=lambda p: (order.get(p["arm"], 9), p["ranks"]))
        head = rows[0]
        print(f"\n{system} / {head['basis']} / {head['method']}  "
              f"nbf={head['nbf']} nshell={head['nshell']} puream={head['puream']}  "
              f"on {head['host']}")
        print(f"{'arm':8s} {'ranks':>5s} {'thr':>4s} {'grid':>5s} {'iters':>5s} "
              f"{'SCF (s)':>9s} {'J/K (s)':>9s} {'spdup':>6s} {'RSS/rank':>9s} "
              f"{'RSS tot':>9s} {'energy (Eh)':>17s} {'dE vs direct':>13s}")
        print("-" * 118)
        for p in rows:
            speedup = f"{p['speedup_vs_gtfock_n1']:.2f}" if p["speedup_vs_gtfock_n1"] != "" else "--"
            de = f"{p['dE_vs_direct_eh']:.2e}" if p["dE_vs_direct_eh"] != "" else "--"
            print(f"{p['arm']:8s} {p['ranks']:5d} {p['threads_per_rank']:4d} "
                  f"{p['process_grid'] or '--':>5s} {str(p['iterations']):>5s} "
                  f"{p['scf_wall_s']:9.1f} {p['jk_wall_s']:9.1f} {speedup:>6s} "
                  f"{p['peak_rss_max_mb']:9.0f} {p['peak_rss_sum_mb']:9.0f} "
                  f"{p['scf_energy']:17.9f} {de:>13s}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directories", nargs="+",
                        help="result directories written by the SLURM script")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    points = add_derived(load_points(args.directories))
    print_table(points)

    if args.csv:
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=_COLUMNS)
            writer.writeheader()
            for point in points:
                writer.writerow({k: point.get(k, "") for k in _COLUMNS})
        print(f"\nwrote {args.csv}")
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(points, handle, indent=2, sort_keys=True)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
