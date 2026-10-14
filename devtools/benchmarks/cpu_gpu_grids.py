#!/usr/bin/env python3
"""Fresh true-CPU / GPU comparisons: same grids, allocation, threads and workload."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

import grid_evidence as grid

ARMS = ("fixed99", "aggressive99", "coarse75")
ROUTES = ("cpu", "host", "cuda")


def route_environment(science, route, environment, inherited):
    build = science / ("build_grid_cuda" if route == "cuda" else "build_grid_host")
    package = build / "stage/lib/psi4"
    libxc = environment.parents[2] / "software/libxc-7.1.2-cuda/lib64"
    libraries = [str(environment / "lib")]
    if route == "cuda":
        libraries.insert(0, str(libxc))
    if inherited.get("LD_LIBRARY_PATH"):
        libraries.append(inherited["LD_LIBRARY_PATH"])
    result = dict(inherited, PYTHONNOUSERSITE="1", PYTHONPATH=str(package.parent),
                  PATH=f"{environment}/bin:{build}/stage/bin:" + inherited.get("PATH", ""),
                  LD_LIBRARY_PATH=":".join(libraries))
    if route == "cpu":
        result["CUDA_VISIBLE_DEVICES"] = ""
    return result, package


def summarize(rows, repeats=3, tolerance=1e-5):
    groups = []
    for system in dict.fromkeys(r["system"] for r in rows):
        cells = {(a, b): [r for r in rows if r["system"] == system and r["arm"] == a
                         and r["backend"] == b] for a in ARMS for b in ROUTES}
        if not all(len(v) == repeats and all(r["ok"] for r in v) for v in cells.values()):
            continue
        runs = [r for v in cells.values() for r in v]
        for key in ("scientific_commit", "geometry_sha256", "threads", "memory"):
            assert len({r[key] for r in runs}) == 1, f"Unmatched {key}"
        for r in runs:
            gpu = r["backend"] != "cpu"
            assert all(r["options"][k] == gpu for k in ("USE_CUEST", "CUEST_XC", "CUEST_SAD"))
            for field in ("effective_route_options", "effective_scf_route_options"):
                assert all(r[field][k] == gpu for k in ("USE_CUEST", "CUEST_XC", "CUEST_SAD"))
            if not gpu:
                assert r["cuda_visible_devices"] == "", "CPU workers must hide GPUs"
            assert not r["options"]["CUEST_XC_TIMING"]
            final, initial = grid.ALL_ARMS[r["arm"]]
            assert (r["options"]["DFT_RADIAL_POINTS"], r["options"]["DFT_SPHERICAL_POINTS"]) == final
            assert (r["options"]["DFT_RADIAL_POINTS_INITIAL"],
                    r["options"]["DFT_SPHERICAL_POINTS_INITIAL"]) == (initial or (0, 0))

        def error(left, right):
            return max(abs(a["components"][k] - b["components"][k])
                       for a in left for b in right for k in grid.COMPONENTS)

        arms = []
        for arm in ARMS:
            routes = {}
            for backend in ROUTES:
                values = cells[arm, backend]
                times = [r["wall_s"] for r in values]
                median = statistics.median(times)
                representative = min(values, key=lambda r: abs(r["wall_s"] - median))
                grid_error = error(values, cells["fixed99", backend])
                cross_error = error(values, cells[arm, "cpu"])
                routes[backend] = dict(n=repeats, median_s=median, min_s=min(times), max_s=max(times),
                    representative=representative, grid_error_eh=grid_error,
                    cpu_gpu_error_eh=cross_error, grid_accuracy_pass=grid_error <= tolerance,
                    cpu_gpu_accuracy_pass=cross_error <= tolerance)
            for backend in ("host", "cuda"):
                routes[backend]["speedup"] = routes["cpu"]["median_s"] / routes[backend]["median_s"]
                routes[backend]["acceptable_speedup"] = (
                    routes["cpu"]["grid_accuracy_pass"] and routes[backend]["grid_accuracy_pass"]
                    and routes[backend]["cpu_gpu_accuracy_pass"])
            arms.append(dict(arm=arm, routes=routes))
        groups.append(dict(system=system, basis=grid.SYSTEMS[system], arms=arms))
    return dict(groups=groups, failures=[r for r in rows if not r["ok"]], tolerance_eh=tolerance,
                scope="Matched energy() wall; CPU has all cuEST flags off. No cross-job denominator.",
                same_final_switching_error_available_for_coarse75=False)


def campaign(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    receipt = json.loads(args.receipt.read_text())
    assert receipt["scientific_commit"] == args.scientific_commit and receipt["scoped_validation_passed"]
    assert not receipt["full_sapt_validation_passed"]
    envs, packages = {}, {}
    for route in ROUTES:
        envs[route], packages[route] = route_environment(args.science, route, args.environment, os.environ)
        expected = receipt["backends"]["cuda" if route == "cuda" else "host"]["core_sha256"]
        code = """import hashlib,json,sys,psi4
from pathlib import Path
assert Path(psi4.__file__).resolve().parent==Path(sys.argv[1]).resolve()
h=hashlib.sha256(Path(psi4.core.__file__).read_bytes()).hexdigest()
assert h==sys.argv[2]
print(json.dumps(dict(package=psi4.__file__,core_sha256=h)))
"""
        result = subprocess.check_output([sys.executable, "-c", code, str(packages[route]), expected],
                                         cwd=out, env=envs[route], text=True)
        grid.atomic(out / f"binary-{route}.json", json.loads(result))
    manifest = dict(job_id=os.getenv("SLURM_JOB_ID"), scientific_commit=args.scientific_commit,
        systems=args.systems, arms=ARMS, routes=ROUTES, repeats=args.repeats, threads=args.threads,
        microprofiling=False, memory_sampler=False, records=[],
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        grid_worker_sha256=hashlib.sha256(Path(grid.__file__).read_bytes()).hexdigest(),
        timing="Fresh serial processes; rotated/reversed grid+route order; energy() boundary only")
    manifest["scratch"] = "Persistent job-owned Lustre; same location policy for all routes"
    grid.atomic(out / "campaign.json", manifest)
    rows = []
    for system in args.systems:
        for repeat in range(args.repeats):
            order = [(a, r) for a in ARMS for r in ROUTES]
            if repeat % 2:
                order.reverse()
            shift = (repeat * 4) % len(order)
            order = order[shift:] + order[:shift]
            for arm, route in order:
                name = f"{system}-{arm}-{route}-{repeat+1}"
                directory = out / name
                directory.mkdir()
                with tempfile.TemporaryDirectory(prefix="cpu-gpu-grid-", dir=os.getenv("TMPDIR")) as scratch:
                    env = dict(envs[route], TMPDIR=scratch, SCRATCH=scratch, PSI_SCRATCH=scratch)
                    command = [sys.executable, str(Path(grid.__file__).resolve()), "--worker",
                        "--output", str(directory), "--system", system, "--arm", arm, "--variant", route,
                        "--repeat", str(repeat+1), "--threads", str(args.threads), "--memory", args.memory,
                        "--scientific-commit", args.scientific_commit,
                        "--expected-package", str(packages[route])]
                    print("START", name, flush=True)
                    try:
                        with (directory / "console.log").open("w") as log:
                            rc = subprocess.run(command, cwd=directory, env=env, stdout=log,
                                stderr=subprocess.STDOUT, timeout=args.case_timeout).returncode
                    except subprocess.TimeoutExpired:
                        rc = 124
                path = directory / "result.json"
                row = json.loads(path.read_text()) if path.exists() else dict(
                    system=system, arm=arm, backend=route, ok=False, error=f"No result; exit {rc}")
                row["result_path"] = str(path)
                if rc:
                    row["ok"] = False
                rows.append(row)
                manifest["records"].append(dict(name=name, returncode=rc, result=str(path)))
                grid.atomic(out / "campaign.json", manifest)
                grid.atomic(out / "evidence.json", summarize(rows, args.repeats))
                print("END", name, "rc", rc, "wall", row.get("wall_s"), flush=True)
                if rc or not row["ok"]:
                    return 1
    data = summarize(rows, args.repeats)
    assert len(data["groups"]) == len(args.systems)
    grid.atomic(out / "evidence.json", data)
    grid.atomic(out / "COMPLETE.json", dict(ok=True, count=len(rows),
        accuracy_pass=all(r["acceptable_speedup"] for g in data["groups"]
            for a in g["arms"] for k, r in a["routes"].items() if k != "cpu")))
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--science", type=Path, required=True)
    p.add_argument("--environment", type=Path, required=True)
    p.add_argument("--scientific-commit", required=True)
    p.add_argument("--systems", nargs="+", choices=grid.SYSTEMS, default=list(grid.SYSTEMS))
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--memory", default="112 GiB")
    p.add_argument("--case-timeout", type=float, default=3600)
    args = p.parse_args()
    if args.repeats < 1 or args.threads < 1:
        p.error("repeats and threads must be positive")
    raise SystemExit(campaign(args))
