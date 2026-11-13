#!/usr/bin/env python3
"""Frozen paired review-fix timings. Fresh processes; never overwrite completed runs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import runpy
import statistics
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def inputs(args):
    systems = runpy.run_path(args.systems)["SYSTEMS"]
    base = dict(basis="def2-tzvp", scf_type="mem_df", reference="rhf",
                integral_package="libint2", orbital_optimizer_package="internal",
                e_convergence=1e-6, d_convergence=1e-6, maxiter=100,
                scf_df_k_layout="AUTO", scf_df_metric="cholesky",
                scf_k_fp32_convergence=1e-5, scf_e_conv_predicted=True)
    grid = dict(dft_basis_tolerance=1e-8, dft_radial_points_initial=50,
                dft_spherical_points_initial=194, dft_grid_switch_factor=100.0,
                dft_grid_switch_on="density", dft_grid_switch_diis="shift")
    cases = {}
    for name, method, memory in [
        ("nanotube_b3lyp_fast", "b3lyp", 64),
        ("nanotube_hf_fast", "hf", 64),
        ("nanotube_wb97mv_fast", "wb97m-v", 64),
        ("nanotube_wb97mv_tight", "wb97m-v", 46),
    ]:
        options = dict(base)
        if method != "hf":
            options.update(grid)
            options.update(dft_radial_points=99 if method == "wb97m-v" else 75,
                           dft_spherical_points=590 if method == "wb97m-v" else 302,
                           dft_vv10_postscf=False)
        cases[name] = dict(method=method, memory_gib=memory, options=options,
                           geometry=systems["nanotube"])
    assert not args.root.exists()
    save(args.root, dict(cases=cases, geometry_source_sha256=sha(args.systems),
                        note="Spot checks, not a recreation of every original report profile."))


def prepare(args):
    args.root.mkdir(parents=True, exist_ok=False)
    allowed = os.sched_getaffinity(0)
    topology = subprocess.check_output(["lscpu", "-p=CPU,CORE,SOCKET"], text=True)
    if args.cpus:
        cpus = [int(x) for x in args.cpus.split(",")]
    else:
        seen, cpus = set(), []
        for line in topology.splitlines():
            if line.startswith("#"):
                continue
            cpu, core, socket = map(int, line.split(","))
            if cpu in allowed and (core, socket) not in seen:
                seen.add((core, socket))
                cpus.append(cpu)
        cpus = cpus[:args.threads]
    assert len(cpus) == args.threads and set(cpus) <= allowed
    builds = {}
    for arm in ("before", "after"):
        lib = Path(getattr(args, arm)).resolve()
        core = next((lib / "psi4").glob("core*.so"))
        driver = lib / "psi4/driver/procrouting/scf_proc/scf_iterator.py"
        builds[arm] = dict(lib=str(lib), core=str(core), core_sha256=sha(core),
                           driver=str(driver), driver_sha256=sha(driver))
    manifest = dict(
        builds=builds, cases=json.loads(Path(args.inputs).read_text())["cases"],
        input_sha256=sha(args.inputs), script=str(Path(__file__).resolve()),
        script_sha256=sha(__file__), python=sys.executable, python_version=sys.version,
        host=os.uname().nodename, cpu_topology=topology,
        cpu_model=next(x.split(":", 1)[1].strip() for x in Path("/proc/cpuinfo").read_text().splitlines()
                       if x.startswith("model name")),
        threads=args.threads, cpus=cpus, interleave=args.interleave,
        repeats=3, pilot_excluded=True, source_commit=args.commit,
        baseline_commit="f21cb6880866c61fc80900af6a0a75642ada12c9",
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        created_unix=time.time(), note="Before vs corrected v3, not upstream.",
    )
    save(args.root / "manifest.json", manifest)
    print(args.root, flush=True)


def verify(m):
    assert sha(m["script"]) == m["script_sha256"]
    for b in m["builds"].values():
        assert sha(b["core"]) == b["core_sha256"]
        assert sha(b["driver"]) == b["driver_sha256"]


def worker(args, m):
    case = m["cases"][args.case]
    os.sched_setaffinity(0, m["cpus"])
    affinity_before_import = sorted(os.sched_getaffinity(0))
    import psi4
    assert Path(psi4.__file__).resolve().is_relative_to(Path(m["builds"][args.arm]["lib"]))
    psi4.set_num_threads(m["threads"])
    psi4.set_memory(int(case["memory_gib"] * 2**30))
    psi4.set_output_file(str(args.outdir / "output.dat"), False)
    psi4.set_options(case["options"])
    mol = psi4.geometry(case["geometry"])
    start, cpu = time.perf_counter(), time.process_time()
    energy, wfn = psi4.energy(case["method"], molecule=mol, return_wfn=True)
    wall = time.perf_counter() - start
    result = dict(
        case=args.case, arm=args.arm, energy=energy, wall_seconds=wall,
        process_cpu_seconds=time.process_time() - cpu,
        iterations=int(wfn.variable("SCF ITERATIONS")), nbf=wfn.basisset().nbf(),
        initial_grid_iterations=int(wfn.variable("SCF DFT INITIAL GRID ITERATIONS"))
        if wfn.has_variable("SCF DFT INITIAL GRID ITERATIONS") else 0,
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        affinity_before_import=affinity_before_import,
        final_jk_words=getattr(wfn, "memory_jk_", None),
        final_cache_words=getattr(wfn, "memory_collocation_", None),
        host=os.uname().nodename, load_after=os.getloadavg(),
    )
    save(args.outdir / "worker.json", result)
    psi4.core.close_outfile()


def timers(path):
    flat, blocks, counted, result = False, 0, False, {}
    for line in path.read_text().splitlines():
        text = line.strip()
        if re.match(r"^Module\s+User\s+System\s+Wall\s+Calls$", text):
            flat, counted = True, False
        elif text and set(text) in ({"-"}, {"*"}):
            flat = False
        elif flat:
            match = re.match(r"^(.*?)\s*:\s*([\d.]+)u\s+([\d.]+)s\s+([\d.]+)w\s+(\d+) calls", line)
            if match and not match[1].strip().startswith("|"):
                if not counted:
                    blocks += 1
                    counted = True
                result[match[1].strip()] = dict(wall_s=float(match[4]), calls=int(match[5]))
    assert blocks == 1 and result, f"Unexpected timer epochs: {blocks}"
    return result


def execute(args, m, case, arm, rep):
    verify(m)
    out = args.root / ("pilot" if rep == 0 else "measured") / case / f"rep{rep}_{arm}"
    if (out / "result.json").exists():
        print(f"Already complete: {out}", flush=True)
        return
    out.mkdir(parents=True, exist_ok=False)  # Incomplete attempts require a new run.
    scratch = out / "scratch"
    scratch.mkdir()
    env = os.environ | dict(
        PYTHONPATH=m["builds"][arm]["lib"], PYTHONNOUSERSITE="1",
        OMP_NUM_THREADS=str(m["threads"]), MKL_NUM_THREADS=str(m["threads"]),
        OPENBLAS_NUM_THREADS=str(m["threads"]), OMP_DYNAMIC="FALSE", MKL_DYNAMIC="FALSE",
        TMPDIR=str(scratch), SCRATCH=str(scratch), PSI_SCRATCH=str(scratch))
    for key in ("OMP_PROC_BIND", "OMP_PLACES"):
        env.pop(key, None)
    prefix = ["numactl", "--interleave=all"] if m["interleave"] else []
    command = prefix + [m["python"], m["script"], "worker", str(args.root),
                        "--case", case, "--arm", arm, "--outdir", str(out)]
    with (out / "console.log").open("w") as log:
        subprocess.run(command, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT,
                       timeout=3600, check=True)
    result = json.loads((out / "worker.json").read_text())
    result.update(repetition=rep, pilot=rep == 0, timers=timers(out / "timer.dat"))
    result["cache_lines"] = [x.strip() for x in (out / "output.dat").read_text().splitlines()
                             if "Cached " in x or "DFHelper Memory:" in x]
    verify(m)
    save(out / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("timers", "cache_lines")}), flush=True)


def summarize(args, m):
    rows = []
    for case in m["cases"]:
        records = {arm: [json.loads(p.read_text()) for p in sorted(
            (args.root / "measured" / case).glob(f"rep*_{arm}/result.json"))]
                   for arm in ("before", "after")}
        if any(len(r) != m["repeats"] for r in records.values()):
            continue
        b, a = records["before"], records["after"]
        bm, am = (statistics.median(r["wall_seconds"] for r in side) for side in (b, a))
        rows.append(dict(
            case=case, before_s=bm, after_s=am, change_percent=100 * (am / bm - 1),
            paired_ratios=[y["wall_seconds"] / x["wall_seconds"] for x, y in zip(b, a)],
            max_abs_delta_e=max(abs(y["energy"] - x["energy"]) for x, y in zip(b, a)),
            before_iterations=[x["iterations"] for x in b],
            after_iterations=[x["iterations"] for x in a],
            before_rss_gib=statistics.median(x["peak_rss_gib"] for x in b),
            after_rss_gib=statistics.median(x["peak_rss_gib"] for x in a),
        ))
    save(args.root / "summary.json", rows)
    print(json.dumps(rows, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("inputs", "prepare", "run", "worker", "summarize"))
    p.add_argument("root", type=lambda x: Path(x).resolve())
    for key in ("systems", "inputs", "before", "after", "cpus", "commit", "case", "arm"):
        p.add_argument("--" + key)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--interleave", action="store_true")
    p.add_argument("--outdir", type=Path)
    args = p.parse_args()
    if args.mode == "inputs":
        return inputs(args)
    if args.mode == "prepare":
        return prepare(args)
    m = json.loads((args.root / "manifest.json").read_text())
    if args.mode == "worker":
        return worker(args, m)
    if args.mode == "run":
        cases = [args.case] if args.case else list(m["cases"])
        for rep in range(m["repeats"] + 1):
            for i, case in enumerate(cases if rep % 2 else cases[::-1]):
                for arm in (("before", "after") if (rep + i) % 2 else ("after", "before")):
                    execute(args, m, case, arm, rep)
    summarize(args, m)


if __name__ == "__main__":
    main()
