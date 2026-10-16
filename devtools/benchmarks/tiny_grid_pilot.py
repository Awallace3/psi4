#!/usr/bin/env python3
"""Bounded, fresh-process initial-grid stability pilot; final grid stays 75x302.

One repeat is exploratory, not a robust speedup claim. Failed tiny grids remain
in the output. No direct-JK fallback, microprofiling, or GPU use in CPU mode.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import grid_evidence as grid

SCIENCE = "c28950041bbe8bb10617264a7b6ddc56a436cda1"
CORES = dict(cpu="5d3041f50b82a5fc5ea634ae6a0ee1cb4d2f420a8a53086a83cf1aef27ce1121",
             cuda="91d6018627e830cb7a088bf44b6e715c9a9694aaebb86907b78186f03735b8d7")
CONFIGS = {
    "fixed75": ((75, 302), None),
    "fixed99": ((99, 590), None),
    "tiny10": ((75, 302), (20, 110)),
    "tiny12": ((75, 302), (25, 110)),
    "tiny19": ((75, 302), (30, 146)),
    "tiny26": ((75, 302), (30, 194)),
    "coarse75": ((75, 302), (35, 194)),
    "tiny08": ((75, 302), (20, 86)),
}
SYSTEMS = ("water", "peptide", "protein83")


def install_configs():
    grid.ALL_ARMS.update(CONFIGS)
    for system in SYSTEMS:
        grid.SYSTEMS[system] = "aug-cc-pvdz"


def summarize(rows):
    """Separate switching error from final-grid error; never rank failed arms."""
    result = []
    for system in dict.fromkeys(r["system"] for r in rows):
        values = [r for r in rows if r["system"] == system]
        controls = {a: [r for r in values if r["arm"] == a and r.get("ok")]
                    for a in ("fixed75", "fixed99")}
        groups = []
        for row in values:
            item = {k: row[k] for k in (
                "system", "arm", "repeat", "ok", "result_path", "returncode") if k in row}
            item["initial_fraction"] = (
                CONFIGS[row["arm"]][1][0]*CONFIGS[row["arm"]][1][1]/(75*302)
                if CONFIGS[row["arm"]][1] else None)
            item["acceptable"] = False
            if row.get("ok") and all(controls.values()):
                for name, arm in (("switching", "fixed75"), ("reference", "fixed99")):
                    item[name + "_error_eh"] = max(
                        abs(row["components"][k]-ref["components"][k])
                        for ref in controls[arm] for k in grid.COMPONENTS)
                item["wall_s"] = row["wall_s"]
                item["speedup_vs_fixed75"] = min(r["wall_s"] for r in controls["fixed75"])/row["wall_s"]
                item["acceptable"] = max(item["switching_error_eh"], item["reference_error_eh"]) <= 1.e-5
                item.update({k: row[k] for k in (
                    "initial_iterations", "final_iterations", "all_iterations", "xc_s", "scf_s", "switch_s")})
            elif not row.get("ok"):
                item["error"] = row.get("error", "Worker failed")
            groups.append(item)
        stable = [r for r in groups if r["acceptable"] and r["initial_fraction"] is not None]
        result.append(dict(system=system, basis="aug-cc-pvdz", rows=groups,
            smallest_stable=min(stable, key=lambda r:r["initial_fraction"])["arm"] if stable else None,
            fastest_stable=min(stable, key=lambda r:r["wall_s"])["arm"] if stable else None,
            warning="Single-run rankings are exploratory; repeat controls and candidate before claiming a speedup."))
    return result


def worker(args):
    sys.argv.append("--einsums:no-attach-debugger")
    import psi4
    assert Path(psi4.__file__).resolve().parent == args.expected_package.resolve()
    core_sha = hashlib.sha256(Path(psi4.core.__file__).read_bytes()).hexdigest()
    assert core_sha == CORES[args.variant]
    geometry = grid.WATER if args.system == "water" else json.loads(
        Path(grid.__file__).with_name("grid_geometries.json").read_text())[args.system]
    psi4.set_num_threads(args.threads)
    # Count basis outside the timed boundary, including for new augmented inputs.
    mol = psi4.geometry(geometry)
    args.expected_nbf = psi4.core.BasisSet.build(
        mol, "BASIS", "aug-cc-pvdz", puream=True).nbf()
    args.ints_tolerance = 1.e-12
    args.scientific_commit = SCIENCE
    rc = grid.worker(args)
    path = args.output / "result.json"
    row = json.loads(path.read_text())
    row.update(core_sha256=core_sha, exploratory=True,
               harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               worker_sha256=hashlib.sha256(Path(grid.__file__).read_bytes()).hexdigest())
    grid.atomic(path, row)
    return rc


def campaign(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = dict(job_id=os.getenv("SLURM_JOB_ID"), scientific_commit=SCIENCE,
                    harness_commit=subprocess.check_output(
                        ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
                        text=True).strip(),
                    systems=args.systems, basis="aug-cc-pvdz", final_grid=[75, 302],
                    arms=args.arms, threads=args.threads, memory=args.memory,
                    variant=args.variant, repeats=1, sampling=False, microprofiling=False,
                    budget_s=args.budget, records=[],
                    scope="SAPT(DFT)-D4(I), PBE0, iterative unseeded GRAC, delta-HF, induction NONE; initial-grid stability only.")
    rows, exhausted = [], False
    for system in args.systems:
        for arm in args.arms:
            remaining = args.budget-(time.monotonic()-started)
            if remaining < 60:
                exhausted = True
                break
            directory = out / f"{system}-{arm}"
            directory.mkdir()
            with tempfile.TemporaryDirectory(prefix="tiny-grid-", dir=os.environ["TMPDIR"]) as scratch:
                env = dict(os.environ, TMPDIR=scratch, SCRATCH=scratch, PSI_SCRATCH=scratch)
                if args.variant == "cpu":
                    env["CUDA_VISIBLE_DEVICES"] = ""
                command = [sys.executable, str(Path(__file__).resolve()), "worker",
                    "--system", system, "--arm", arm, "--output", str(directory),
                    "--expected-package", str(args.expected_package), "--variant", args.variant,
                    "--threads", str(args.threads), "--memory", args.memory]
                print("START", system, arm, "remaining_s", round(remaining), flush=True)
                with (directory / "console.log").open("w") as log:
                    process = subprocess.Popen(command, cwd=scratch, env=env, stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        rc = process.wait(timeout=min(args.case_timeout, remaining-20))
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                        rc = 124
            path = directory / "result.json"
            row = json.loads(path.read_text()) if path.exists() else dict(
                system=system, arm=arm, ok=False, error="No complete worker result")
            row.update(result_path=str(path), returncode=rc)
            if rc:
                row.update(ok=False, error=row.get("error", f"Worker exit {rc}"))
            rows.append(row)
            manifest["records"].append(dict(system=system, arm=arm, returncode=rc, path=str(path)))
            grid.atomic(out / "campaign.json", manifest)
            grid.atomic(out / "summary.json", dict(groups=summarize(rows), complete=False))
            print("END", system, arm, "rc", rc, "wall_s", row.get("wall_s"), flush=True)
            # Do not spend the rest of an allocation on an unworkable large-system
            # control. Failed tiny arms, however, must not suppress larger tests.
            if not row["ok"] and arm in ("fixed75", "fixed99"):
                break
        if exhausted:
            break
    expected = len(args.systems)*len(args.arms)
    grid.atomic(out / "summary.json", dict(groups=summarize(rows),
        complete=len(rows)==expected, attempted=len(rows), expected=expected,
        failures=sum(not r.get("ok") for r in rows), budget_exhausted=exhausted))
    return 0 if len(rows)==expected else 75


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("worker", "campaign"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-package", type=Path, required=True)
    p.add_argument("--variant", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--memory", default="80 GiB")
    p.add_argument("--systems", nargs="+", choices=SYSTEMS, default=list(SYSTEMS))
    p.add_argument("--arms", nargs="+", choices=CONFIGS,
                   default=list(CONFIGS)[:-1])
    p.add_argument("--system", choices=SYSTEMS, default="water")
    p.add_argument("--arm", choices=CONFIGS, default="tiny10")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--case-timeout", type=float, default=1500.)
    p.add_argument("--budget", type=float, default=6600.)
    args = p.parse_args()
    if args.mode == "campaign" and not {"fixed75", "fixed99"} <= set(args.arms):
        p.error("Both fixed controls are required")
    if args.threads < 1 or args.case_timeout <= 0 or args.budget <= 0:
        p.error("Threads and time budgets must be positive")
    args.output = args.output.resolve()
    install_configs()
    return worker(args) if args.mode == "worker" else campaign(args)


if __name__ == "__main__":
    raise SystemExit(main())
