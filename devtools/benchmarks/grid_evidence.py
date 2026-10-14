#!/usr/bin/env python3
"""Matched GPU grid experiment: serial rotated arms, fresh processes, no microprofiling.

The energy() boundary reproduces the October 4 D4(I)/iterative-GRAC workload.
It is not the CPKS/FDDS accuracy sweep. No historical timing is a denominator.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
import time

ARMS = {
    "fixed99": ((99, 590), None),
    "quarter99": ((99, 590), (50, 302)),
    "third99": ((99, 590), (65, 302)),
    "fixed75": ((75, 302), None),
    "coarse75": ((75, 302), (35, 194)),
}
EXTRA_ARMS = {"aggressive99": ((99, 590), (35, 194))}
ALL_ARMS = {**ARMS, **EXTRA_ARMS}
SYSTEMS = {"water": "cc-pvdz", "peptide": "6-31+g**",
           "nanotube": "6-31+g**", "protein83": "6-31+g**"}
COMPONENTS = [f"SAPT {key} ENERGY" for key in ("ELST", "EXCH", "IND", "DISP", "TOTAL")]
WATER = """0 1
O -0.702196054 -0.056060256 0.009942262
H -1.022193224 0.846775782 -0.011488714
H 0.257521062 0.042121496 0.005218999
--
0 1
O 2.268880784 0.026340101 0.000508029
H 2.645502399 -0.412039965 0.766632411
H 2.641145101 -0.449872874 -0.744894473
units angstrom
symmetry c1
no_reorient
no_com
"""


def atomic(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def timer_regions(timers):
    # Only serial parent boundaries, never parallel/thread-summed microtimers.
    def total(names):
        return sum(v["wall_time"] for k, v in timers.items()
                   if k.rsplit(";", 1)[-1].replace(" ", "") in names)
    return {
        "scf_s": total({"SCF:Driver"}),
        "xc_s": total({"RV:FormV", "UV:FormV"}),
        "functional_s": total({"cuESTXC:HostFunctional", "cuESTXC:CUDALibXCFunctional"}),
        "switch_s": total({"SCF:SwitchXCgrid"}),
    }


def summarize(rows, repeats=3, tolerance=1.e-5, arms=None):
    """Include only complete matched groups; retain all errors/failures separately."""
    configs = ARMS if arms is None else {arm: ALL_ARMS[arm] for arm in arms}
    assert "fixed99" in configs
    assert "fixed75" in configs or all(final != (75, 302) for final, _ in configs.values())
    groups = []
    for system in dict.fromkeys(row["system"] for row in rows):
        by_arm = {arm: [r for r in rows if r["system"] == system and r["arm"] == arm]
                  for arm in configs}
        if not all(len(v) == repeats and all(r.get("ok") for r in v) for v in by_arm.values()):
            continue
        assert all(r["options"]["CUEST_XC_TIMING"] is False for v in by_arm.values() for r in v)
        for key in ("geometry_sha256", "backend", "scientific_commit", "threads", "memory", "package"):
            assert len({r.get(key) for v in by_arm.values() for r in v}) == 1, f"Unmatched {key}"
        controls = {arm: by_arm[arm] for arm in ("fixed99", "fixed75") if arm in by_arm}
        base_median = statistics.median(r["wall_s"] for r in controls["fixed99"])
        arms = []
        for arm, runs in by_arm.items():
            same = "fixed75" if configs[arm][0] == (75, 302) else "fixed99"
            # Worst over all cross-repeat comparisons, not cherry-picked paired runs.
            def errors(ref):
                return {key: max(abs(a["components"][key] - b["components"][key])
                                 for a in runs for b in ref) for key in COMPONENTS}
            switching = errors(controls[same])
            quadrature = errors(controls["fixed99"])
            samples = [r["wall_s"] for r in runs]
            median = statistics.median(samples)
            representative = min(runs, key=lambda r: abs(r["wall_s"] - median))
            arms.append(dict(
                arm=arm, final=configs[arm][0], initial=configs[arm][1],
                n=len(runs), median_s=median, min_s=min(samples), max_s=max(samples),
                speedup=base_median / median, switching_error_eh=switching,
                final_grid_error_eh=quadrature, agreement_pass=max(quadrature.values()) <= tolerance,
                max_component_error_eh=max(quadrature.values()),
                max_grac_shift_error_eh=max(abs(a["grac_shifts"][key] - b["grac_shifts"][key])
                    for a in runs for b in controls["fixed99"] for key in ("A", "B")),
                # A complete median-nearest run supplies all detailed regions.
                representative=representative,
            ))
        groups.append(dict(system=system, basis=SYSTEMS[system], arms=arms))
    return dict(groups=groups, failures=[r for r in rows if not r.get("ok")],
                tolerance_eh=tolerance, expected_repeats=repeats,
                expected_systems=list(dict.fromkeys(row["system"] for row in rows)))


def worker(args):
    import psi4

    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(out)
    if args.expected_package:
        assert Path(psi4.__file__).resolve().parent == Path(args.expected_package).resolve()
    psi4.core.set_output_file(str(out / "psi4.out"), False)
    psi4.set_memory(args.memory)
    psi4.set_num_threads(args.threads)
    final, initial = ALL_ARMS[args.arm]
    opts = {
        "BASIS": SYSTEMS[args.system], "SCF_TYPE": "DF", "REFERENCE": "RHF",
        "SAPT_DFT_FUNCTIONAL": "PBE0", "SAPT_DFT_GRAC_COMPUTE": "ITERATIVE",
        "SAPT_DFT_GRAC_SEED_CATION": False, "SAPT_DFT_INDUCTION_TYPE": "NONE",
        "SAPT_DFT_DO_DHF": True, "SAPT_DFT_USE_EINSUMS": True,
        "ORBITAL_OPTIMIZER_PACKAGE": "INTERNAL",
        "USE_CUEST": True, "CUEST_XC": True, "CUEST_SAD": True,
        "CUEST_MIXED_PRECISION": False, "CUEST_XC_TIMING": False,
        "E_CONVERGENCE": 9, "D_CONVERGENCE": 8, "MAXITER": 150,
        "DFT_RADIAL_POINTS": final[0], "DFT_SPHERICAL_POINTS": final[1],
        "DFT_RADIAL_POINTS_INITIAL": initial[0] if initial else 0,
        "DFT_SPHERICAL_POINTS_INITIAL": initial[1] if initial else 0,
        "DFT_GRID_SWITCH_FACTOR": 10.0, "SAD_PRINT": 1, "BENCH": 0,
    }
    if args.system != "water":
        opts.update(PUREAM=True, DF_BASIS_SCF="def2-universal-jkfit", DF_BASIS_MP2="aug-cc-pvdz-ri")
    geo_file = Path(__file__).with_name("grid_geometries.json")
    geometry = WATER if args.system == "water" else json.loads(geo_file.read_text())[args.system]
    result = dict(system=args.system, arm=args.arm, repeat=args.repeat, options=opts,
                  memory=args.memory, threads=args.threads, geometry=geometry, ok=False,
                  geometry_sha256=hashlib.sha256(geometry.encode()).hexdigest(),
                  package=psi4.__file__, version=psi4.__version__, job_id=os.getenv("SLURM_JOB_ID"),
                  backend=args.variant, scientific_commit=args.scientific_commit)
    start = None
    try:
        psi4.set_options(opts)
        mol = psi4.geometry(geometry)
        result["nbf"] = psi4.core.BasisSet.build(mol, "BASIS", SYSTEMS[args.system]).nbf()
        assert result["nbf"] == dict(water=48, peptide=250, nanotube=548, protein83=922)[args.system]
        result["effective_grid_options"] = {k: psi4.core.get_option("SCF", k) for k in (
            "DFT_PRUNING_SCHEME", "DFT_GRID_NAME", "DFT_RADIAL_SCHEME",
            "DFT_RADIAL_POINTS", "DFT_SPHERICAL_POINTS", "DFT_GRID_SWITCH_FACTOR")}
        psi4.core.clean_timers()
        start = time.perf_counter()
        energy = psi4.energy("sapt(dft)-d4(i)", molecule=mol)
        result["wall_s"] = time.perf_counter() - start
        timers = psi4.core.get_timer_dict()
        assert not any("XC micro:" in k for k in timers), "Timing run was microprofiled"
        cuda = any("cuEST XC: CUDA LibXC Functional" in k for k in timers)
        assert cuda == (args.variant == "cuda"), "Unexpected functional backend"
        result.update(energy_eh=energy, components={k: float(psi4.variable(k)) for k in COMPONENTS},
                      grac_shifts={k: float(psi4.variable("SAPT DFT GRAC SHIFT " + k)) for k in ("A", "B")},
                      timers=timers, **timer_regions(timers))
        psi4.core.close_outfile()
        text = (out / "psi4.out").read_text()
        assert "cuESTJK: GPU-Accelerated" in text
        assert not any(s in text for s in ("MemDFJK: Density-Fitted", "DiskDFJK: Density-Fitted"))
        assert "SAD J/K backend: cuESTJK" in text
        result["initial_iterations"] = len(re.findall(r"@\S+ iter .*XC-INITIAL", text))
        result["final_iterations"] = len(re.findall(r"@\S+ iter .*XC-FINAL", text))
        result["all_iterations"] = len(re.findall(r"@\S+ iter ", text))
        if initial:
            assert result["initial_iterations"] > 0 and result["final_iterations"] > 0
        result["ok"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        if start is not None:
            result.setdefault("wall_s", time.perf_counter() - start)
    finally:
        psi4.core.close_outfile()
        atomic(out / "result.json", result)
    return 0 if result["ok"] else 1


def campaign(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(command=sys.argv, scientific_commit=args.scientific_commit,
                    harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    geometries_sha256=hashlib.sha256(Path(__file__).with_name("grid_geometries.json").read_bytes()).hexdigest(),
                    systems=args.systems, arms={arm: ALL_ARMS[arm] for arm in args.arms},
                    repeats=args.repeats, backend=args.variant,
                    job_id=os.getenv("SLURM_JOB_ID"),
                    timing="fresh-process energy() wall, imports/geometry/basis counts excluded; serial rotated arms",
                    microprofiling=False, memory_sampler=False, records=[])
    atomic(out / "campaign.json", manifest)
    rows = []
    for system in args.systems:
        for repeat in range(args.repeats):
            order = list(args.arms)
            if repeat % 2:
                order.reverse()
            shift = (repeat + 1 if repeat % 2 else repeat // 2) % len(order)
            order = order[shift:] + order[:shift]
            for arm in order:
                name = f"{system}-{arm}-{repeat + 1}"
                directory = out / name
                directory.mkdir()
                # Each process owns scratch and output. Fresh process startup is not the plotted boundary.
                with tempfile.TemporaryDirectory(prefix="grid-evidence-", dir=os.environ.get("TMPDIR")) as scratch:
                    env = dict(os.environ, TMPDIR=scratch, SCRATCH=scratch, PSI_SCRATCH=scratch)
                    command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                               "--output", str(directory), "--system", system, "--arm", arm,
                               "--repeat", str(repeat + 1), "--threads", str(args.threads),
                               "--memory", args.memory, "--variant", args.variant,
                               "--scientific-commit", args.scientific_commit,
                               "--expected-package", args.expected_package]
                    print("START", name, flush=True)
                    try:
                        with (directory / "console.log").open("w") as log:
                            rc = subprocess.run(command, env=env, cwd=directory, stdout=log,
                                                stderr=subprocess.STDOUT, timeout=args.case_timeout).returncode
                    except subprocess.TimeoutExpired:
                        rc = 124
                p = directory / "result.json"
                row = json.loads(p.read_text()) if p.exists() else dict(
                    system=system, arm=arm, repeat=repeat + 1, ok=False, error=f"No result; exit {rc}")
                row["result_path"] = str(p)
                if rc:
                    row["ok"] = False
                    row.setdefault("error", f"Worker exit {rc}")
                rows.append(row)
                manifest["records"].append(dict(name=name, returncode=rc, result=str(p)))
                atomic(out / "campaign.json", manifest)
                atomic(out / "evidence.json", summarize(rows, args.repeats, arms=args.arms))
                print("END", name, "rc", rc, "wall", row.get("wall_s"), flush=True)
                if rc or not row["ok"]:
                    return 1
    data = summarize(rows, args.repeats, arms=args.arms)
    data["campaign"] = manifest
    atomic(out / "evidence.json", data)
    assert len(data["groups"]) == len(args.systems)
    atomic(out / "COMPLETE.json", dict(ok=True, count=len(rows),
           accuracy_pass=all(a["agreement_pass"] for g in data["groups"] for a in g["arms"])))
    # A scientifically inaccurate reduced grid remains a completed experimental arm, not a missing case.
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--systems", nargs="+", choices=SYSTEMS, default=list(SYSTEMS))
    p.add_argument("--system", choices=SYSTEMS, default="water")
    p.add_argument("--arm", choices=ALL_ARMS, default="fixed99")
    p.add_argument("--arms", nargs="+", choices=ALL_ARMS, default=list(ARMS))
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--memory", default="112 GiB")
    p.add_argument("--case-timeout", type=float, default=900)
    p.add_argument("--variant", choices=["host", "cuda"], required=True)
    p.add_argument("--scientific-commit", required=True)
    p.add_argument("--expected-package", required=True)
    p.add_argument("--worker", action="store_true")
    args = p.parse_args()
    if args.repeats < 1 or args.threads < 1:
        p.error("repeats and threads must be positive")
    if len(set(args.arms)) != len(args.arms) or "fixed99" not in args.arms:
        p.error("arms must be unique and include fixed99")
    if any(ALL_ARMS[a][0] == (75, 302) for a in args.arms) and "fixed75" not in args.arms:
        p.error("75-final arms require the matched fixed75 control")
    return worker(args) if args.worker else campaign(args)


if __name__ == "__main__":
    sys.exit(main())
