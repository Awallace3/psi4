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
import collections
import csv
import glob
import json
import os
import sys

# GTFock's setup phases, in the order PDF_create runs them. The CSV needs a
# fixed column per phase, so this list is a copy of the engine's; a phase added
# there and not here still reaches the JSON, just not the table.
_DF_PHASES = ("metric", "factor", "int3c", "fit", "redist")


# One row per point. Written in this order so the CSV reads left to right as
# "what was run", "what it cost", "was it the same answer".
_COLUMNS = [
    "system", "basis", "method", "nbf", "nshell", "puream",
    "arm", "jk_name", "ranks", "threads_per_rank", "total_cores",
    "iterations", "jk_calls",
    "scf_wall_s", "jk_wall_s", "df_setup_s", "scf_remainder_s",
    "speedup_vs_gtfock_n1", "jk_speedup_vs_gtfock_n1", "speedup_vs_direct",
    "peak_rss_max_mb", "peak_rss_sum_mb",
    "scf_energy", "dE_vs_direct_eh", "dE_across_ranks_eh",
    "process_grid", "local_task_shape",
    # How the distributed density-fitting arm split the fitted tensor. Blank for
    # every other arm, since none of them has one.
    "naux", "nlocal_aux_min", "nlocal_aux_max", "nlocal_aux_sum",
    "nmetric_null", "local_tensor_mb_max", "local_tensor_mb_sum",
    # Where df_setup_s went. Only some of these phases divide over ranks, so
    # this is what says whether a setup that stopped improving stopped because
    # of the replicated metric factorization or for some other reason.
    *(f"df_phase_{name}_s" for name in _DF_PHASES),
    "host", "slurm_job_id", "slurm_nodelist", "df_setup_keys",
]


# Where a record was loaded from, stamped on it by load_points. Not a field the
# benchmark writes, and not a CSV column: it is only how a run off SLURM, where
# there is no job id, is told apart from another one.
_SOURCE_DIRECTORY = "_source_directory"


def _run_identity(record):
    """What distinguishes one launch from another, for a record of one rank.

    Under SLURM it is the job id together with the nodelist, not the per-rank
    `host`: the nodelist is the same string on every rank of a launch while
    `host` is not, so a single job spread over several nodes stays one point.

    Off SLURM there is no job id at all -- both fields come back None -- and
    every interactive run would otherwise share the one identity (None, None).
    The result directory identifies the run instead, since one launch writes its
    per-rank files under one directory even when its ranks span hosts.
    """
    job = record.get("slurm_job_id")
    if job:
        return ("job", str(job), str(record.get("slurm_nodelist") or "(no nodelist)"))
    return ("directory", record[_SOURCE_DIRECTORY], "")


def _describe_run(identity):
    kind, first, second = identity
    if kind == "job":
        return f"job {first} on {second}"
    return f"the run under {first} (no SLURM_JOB_ID)"


def _refuse_mixed_runs(system, arm, ranks, records):
    """Refuse to reduce a point whose records did not come from one single run.

    A point is one launch: `ranks` records, one per rank, from one job. If a
    sweep was repeated -- a job timed out, a node was drained -- and both result
    directories are handed over at once, the records land under the same
    (system, arm, ranks) key and a max over wall clocks or a sum over memory
    would silently span two runs on two pieces of hardware. That is exactly the
    drift the generated table exists to prevent, so it is fatal rather than a
    warning.
    """
    runs = {_run_identity(r) for r in records}
    if len(runs) > 1:
        listed = ", ".join(_describe_run(run) for run in sorted(runs))
        raise SystemExit(
            f"{system}/{arm}/n{ranks} collapses records from more than one run "
            f"({listed}); reducing them together would take a maximum over "
            "unrelated hardware. Pass one job's result directory at a time.")

    seen = collections.Counter(r["rank"] for r in records)
    duplicated = sorted(rank for rank, count in seen.items() if count > 1)
    if duplicated:
        raise SystemExit(
            f"{system}/{arm}/n{ranks} has more than one record for rank(s) "
            f"{duplicated}; the summed-memory and maximum-wall-clock columns "
            "would double-count. Pass each result directory once.")


def _refuse_incomplete_point(system, arm, ranks, records):
    """Refuse to reduce a point that is missing some of its ranks.

    A point declaring `ranks` ranks must carry one record from each of them. A
    job killed at its wall limit part way through a sweep leaves a truncated
    set: the two guards above still pass, because the survivors do come from one
    job and no rank repeats, yet `peak_rss_sum_mb` would then sum fewer ranks
    than the `ranks` column claims and the wall clocks would be a maximum over
    only the ranks that finished. The resulting row is indistinguishable from a
    complete point in the CSV the documentation tables are generated from, and a
    warning on stderr is invisible in a job log, so this is fatal.
    """
    found = sorted(r["rank"] for r in records)
    if found != list(range(ranks)):
        raise SystemExit(
            f"{system}/{arm}/n{ranks} is incomplete: expected {ranks} rank "
            f"record(s), ranks 0-{ranks - 1}, but found {len(found)}, "
            f"rank(s) {found}. Reducing it would report a node memory summed "
            "over fewer ranks than the row claims. Re-run that point.")


# A megabyte of fitted tensor, for the columns below: the engine reports doubles.
_MB_PER_DOUBLE = 8.0 / (1024.0 * 1024.0)


def _reduce_df_partition(records):
    """Reduce the distributed DF partition across the ranks of one point.

    The per-rank ``nlocal_aux`` is the whole claim this arm makes that a wall
    clock cannot check, so it is reduced three ways rather than one: the minimum
    and maximum say how evenly the auxiliary index divided, and the sum says
    whether it divided at all. A sum equal to ``naux`` with a maximum below it
    is a partition; a maximum equal to ``naux`` on every rank is a replicated
    tensor that would still give the right energy. Arms with no fitted tensor
    return empty strings, which the CSV writes as blanks rather than zeros.
    """
    parts = [r["df_partition"] for r in records if "df_partition" in r]
    if not parts:
        return {}
    local = [p["nlocal_aux"] for p in parts]
    doubles = [p["local_tensor_doubles"] for p in parts]
    return {
        "naux": parts[0]["naux"],
        "nlocal_aux_min": min(local),
        "nlocal_aux_max": max(local),
        "nlocal_aux_sum": sum(local),
        "nmetric_null": parts[0]["nmetric_null"],
        "local_tensor_mb_max": max(doubles) * _MB_PER_DOUBLE,
        "local_tensor_mb_sum": sum(doubles) * _MB_PER_DOUBLE,
    }


def _reduce_df_phases(records):
    """Reduce the DF setup phase clocks across the ranks of one point.

    Maxima, like every other wall clock here: the build is collective, so each
    phase costs the point what its slowest rank spent in it. That includes the
    time a rank spent waiting, which is the point -- a phase whose maximum far
    exceeds its minimum is not expensive, it is imbalanced.

    Records written before the benchmark measured this carry no phases and
    reduce to nothing, which the CSV writes as blanks rather than as zeros that
    would read like a phase that cost nothing.
    """
    per_rank = [r["df_setup_phases"] for r in records if r.get("df_setup_phases")]
    if not per_rank:
        return {}
    return {f"df_phase_{name}_s": max(p.get(name, 0.0) for p in per_rank)
            for name in _DF_PHASES}


def load_points(directories):
    """Collapse ``*.rank<N>.json`` files into one dict per (system, arm, ranks)."""
    by_key = {}
    for directory in directories:
        pattern = os.path.join(directory, "*.rank*.json")
        for path in sorted(glob.glob(pattern)):
            with open(path) as handle:
                record = json.load(handle)
            record[_SOURCE_DIRECTORY] = directory
            key = (record["system"], record["arm"], record["ranks"])
            by_key.setdefault(key, []).append(record)
    if not by_key:
        raise SystemExit(f"no *.rank*.json found under {', '.join(directories)}")

    points = []
    for (system, arm, ranks), records in sorted(by_key.items()):
        _refuse_mixed_runs(system, arm, ranks, records)
        _refuse_incomplete_point(system, arm, ranks, records)
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
            # Records written before the benchmark recorded which timer it read
            # carry no key; they still reduce, they just cannot say.
            "jk_timer_key": head.get("jk_timer_key", ""),
            "scf_wall_s": max(r["scf_wall_seconds"] for r in records),
            "jk_wall_s": max(r["jk_wall_seconds"] for r in records),
            # The DF three-index build, which JK::initialize() runs outside the
            # "JK: JK" timer, and the whole non-J/K remainder that contains it.
            # Both are blank on records written before the benchmark measured
            # them rather than back-filled with a zero that would read as a
            # measured absence.
            "df_setup_s": (max(r["df_setup_total_seconds"] for r in records)
                           if all("df_setup_total_seconds" in r for r in records) else ""),
            "scf_remainder_s": (max(r["scf_remainder_seconds"] for r in records)
                                if all("scf_remainder_seconds" in r for r in records) else ""),
            "df_setup_keys": "|".join(
                e["key"] for e in head.get("df_setup_records", [])),
            "peak_rss_max_mb": max(r["peak_rss_mb"] for r in records),
            "peak_rss_sum_mb": sum(r["peak_rss_mb"] for r in records),
            "scf_energy": head["scf_energy"],
            "dE_across_ranks_eh": max(energies) - min(energies),
            **_reduce_df_partition(records),
            **_reduce_df_phases(records),
            "process_grid": "x".join(str(n) for n in head["process_grid"]) if "process_grid" in head else "",
            "local_task_shape": "x".join(str(n) for n in head["local_task_shape"]) if "local_task_shape" in head else "",
            "host": head["host"],
            "slurm_job_id": head["slurm_job_id"],
            "slurm_nodelist": head["slurm_nodelist"],
        }
        points.append(point)
    return points


# The arms that scale over ranks, each measured against its own one-rank point.
# Comparing a fitted arm's four-rank wall clock to the exact arm's one-rank wall
# clock would put an approximation and a rank count into one number.
_SCALING_ARMS = ("gtfock", "gtfock_df")


def add_derived(points):
    """Fill in the speedup and energy-difference columns, per system."""
    for system in {p["system"] for p in points}:
        rows = [p for p in points if p["system"] == system]
        base_of = {arm: next((p for p in rows if p["arm"] == arm and p["ranks"] == 1), None)
                   for arm in _SCALING_ARMS}
        direct = next((p for p in rows if p["arm"] == "direct"), None)
        for point in rows:
            # Speedup is only meaningful against the same algorithm at one rank;
            # for the reference arms it is left blank rather than invented.
            base = base_of.get(point["arm"])
            if base is not None and point["arm"] in _SCALING_ARMS:
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
    order = {"direct": 0, "df": 1, "pk": 2, "gtfock": 3, "gtfock_df": 4}
    for system in sorted({p["system"] for p in points}):
        rows = sorted((p for p in points if p["system"] == system),
                      key=lambda p: (order.get(p["arm"], 9), p["ranks"]))
        head = rows[0]
        print(f"\n{system} / {head['basis']} / {head['method']}  "
              f"nbf={head['nbf']} nshell={head['nshell']} puream={head['puream']}  "
              f"on {head['host']}")
        # "setup" is the density-fitting cost that runs in preiterations(),
        # before JK::compute() opens its clock. Shown beside "J/K (s)" because a
        # fitted arm's J/K column is not comparable to an exact arm's without it.
        print(f"{'arm':9s} {'ranks':>5s} {'thr':>4s} {'grid':>5s} {'iters':>5s} "
              f"{'setup (s)':>9s} {'J/K (s)':>9s} {'SCF (s)':>9s} {'spdup':>6s} "
              f"{'RSS/rank':>9s} {'RSS tot':>9s} {'energy (Eh)':>17s} "
              f"{'dE vs direct':>13s}")
        print("-" * 129)
        for p in rows:
            speedup = f"{p['speedup_vs_gtfock_n1']:.2f}" if p["speedup_vs_gtfock_n1"] != "" else "--"
            de = f"{p['dE_vs_direct_eh']:.2e}" if p["dE_vs_direct_eh"] != "" else "--"
            setup = f"{p['df_setup_s']:9.1f}" if p["df_setup_s"] != "" else f"{'--':>9s}"
            print(f"{p['arm']:9s} {p['ranks']:5d} {p['threads_per_rank']:4d} "
                  f"{p['process_grid'] or '--':>5s} {str(p['iterations']):>5s} "
                  f"{setup} {p['jk_wall_s']:9.1f} {p['scf_wall_s']:9.1f} {speedup:>6s} "
                  f"{p['peak_rss_max_mb']:9.0f} {p['peak_rss_sum_mb']:9.0f} "
                  f"{p['scf_energy']:17.9f} {de:>13s}")


def rst_table(points):
    """Emit each system as a docutils simple table, ready to paste into the docs.

    The documentation table is generated rather than transcribed so that no
    number in :source:`doc/sphinxman/source/gtfock.rst` can drift away from the
    run that produced it.
    """
    order = {"direct": 0, "df": 1, "pk": 2, "gtfock": 3, "gtfock_df": 4}
    # The fitting setup and the non-J/K remainder are shown whenever any point
    # in the sweep measured them. A table that omits them puts a fitted arm's
    # partial "J/K (s)" beside an exact arm's complete one and invites the
    # quotient to be read as a ratio of Fock-build speed, which is the mistake
    # the "What the J/K timer does not cover" section of the documentation
    # exists to undo. Sweeps predating the instrumentation report neither, and
    # get the old eleven-column table rather than two columns of blanks.
    split = any(p["df_setup_s"] != "" for p in points)
    header = (["arm", "ranks", "thr", "grid", "iters"]
              + (["setup (s)"] if split else [])
              + ["J/K (s)"]
              + (["rest (s)"] if split else [])
              + ["SCF (s)", "speedup", "RSS/rank (MB)", "RSS node (MB)", "dE (Eh)"])
    for system in sorted({p["system"] for p in points}):
        rows = sorted((p for p in points if p["system"] == system),
                      key=lambda p: (order.get(p["arm"], 9), p["ranks"]))
        head = rows[0]
        body = []
        for p in rows:
            speedup = (f"{p['speedup_vs_gtfock_n1']:.2f}"
                       if p["speedup_vs_gtfock_n1"] != "" else "---")
            de = f"{p['dE_vs_direct_eh']:.1e}" if p["dE_vs_direct_eh"] != "" else "---"
            setup = p["df_setup_s"]
            # "rest" is the SCF with both the J/K timer and the fitting setup
            # removed, so setup + J/K + rest is the SCF column exactly. An arm
            # with no fitting setup has rest equal to the remainder already
            # reported, which is the same identity with a zero in it.
            rest = (p["scf_remainder_s"] - setup) if setup != "" and p["scf_remainder_s"] != "" else ""
            body.append(
                [p["arm"], str(p["ranks"]), str(p["threads_per_rank"]),
                 p["process_grid"] or "---", str(p["iterations"])]
                + ([f"{setup:.1f}"] if split else [])
                + [f"{p['jk_wall_s']:.1f}"]
                + ([f"{rest:.1f}" if rest != "" else "---"] if split else [])
                + [f"{p['scf_wall_s']:.1f}", speedup,
                   f"{p['peak_rss_max_mb']:.0f}", f"{p['peak_rss_sum_mb']:.0f}", de])
        widths = [max(len(row[i]) for row in [header] + body)
                  for i in range(len(header))]
        rule = "  ".join("=" * w for w in widths)
        def line(cells):
            return "  ".join(c.ljust(w) for c, w in zip(cells, widths)).rstrip()
        print(f"\n.. {system}: {head['basis']} nbf={head['nbf']} "
              f"nshell={head['nshell']} on {head['slurm_nodelist'] or head['host']}")
        print(rule)
        print(line(header))
        print(rule)
        for row in body:
            print(line(row))
        print(rule)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directories", nargs="+",
                        help="result directories written by the SLURM script")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--json", default=None)
    parser.add_argument("--rst", action="store_true",
                        help="also print the docs table, so it is generated "
                             "rather than transcribed by hand")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    points = add_derived(load_points(args.directories))
    print_table(points)
    if args.rst:
        rst_table(points)

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
