#!/usr/bin/env python3
"""Small, fresh-process SCF counterpoise IE parity matrix; no SAPT or dispersion."""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from saptdft_cuest_grac import geometry, atomic_json

CASES = [("water", "aug-cc-pvdz"), ("benzene", "aug-cc-pvdz"), ("peptide", "6-31+g**")]
FUNCTIONALS = ["m06-l", "m06", "pw6b95", "r2scan"]
ROUTES = ["cpu", "gpu-jk", "gpu-xc"]
FRAGMENTS = ["AB", "A", "B"]
EH_TO_KCAL = 627.5094740631


def options(basis, route):
    return dict(basis=basis, reference="rhf", scf_type="df", puream=True,
                df_basis_scf="def2-universal-jkfit", use_cuest=route != "cpu",
                cuest_xc=route == "gpu-xc", cuest_mixed_precision=False,
                cuest_sad=False, guess="sad", e_convergence=11, d_convergence=10,
                maxiter=200, fail_on_maxiter=True, dft_radial_points=99,
                dft_spherical_points=590, dft_pruning_scheme="robust")


def delta(energies):
    return energies["AB"] - energies["A"] - energies["B"]


def worker(a):
    import psi4
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(out)
    record = dict(ok=False, system=a.system, basis=a.basis, functional=a.functional,
                  route=a.route, fragment=a.fragment, job=os.getenv("SLURM_JOB_ID"),
                  psi4_module=psi4.__file__, version=psi4.__version__,
                  options=options(a.basis, a.route), geometry=geometry(a.system))
    try:
        assert Path(psi4.__file__).resolve().parent == a.package.resolve()
        psi4.core.set_output_file(str(out / "psi4.out"), False)
        psi4.set_memory("32 GiB")
        psi4.set_num_threads(8)
        try:
            psi4.core.get_global_option("CUEST_SAD")
        except Exception:
            # Older CUDA build predates this switch; its atomic SAD is CPU-only.
            record["options"].pop("cuest_sad")
        psi4.set_options(record["options"])
        mol = psi4.geometry(record["geometry"])
        assert mol.nfragments() == 2
        if a.fragment == "A":
            mol = mol.extract_subsets(1, 2)
        elif a.fragment == "B":
            mol = mol.extract_subsets(2, 1)
        mol.update_geometry()
        record["molecule"] = mol.to_string(dtype="psi4")
        psi4.core.clean_timers()
        start = time.perf_counter()
        energy, wfn = psi4.energy(a.functional, molecule=mol, return_wfn=True)
        record.update(energy_hartree=float(energy), wall_s=time.perf_counter()-start,
                      nbf=wfn.basisset().nbf(), meta=wfn.functional().is_meta(),
                      timer_records=psi4.core.get_timer_records())
        assert record["meta"], "Not a meta-GGA"
        assert math.isfinite(energy)
        psi4.core.close_outfile()
        text = (out / "psi4.out").read_text()
        record["gpu_jk_seen"] = "cuESTJK: GPU-Accelerated" in text
        assert record["gpu_jk_seen"] == (a.route != "cpu")
        names = ["/".join(t["timer_path"]) for t in record["timer_records"].values()]
        device = any("CUDA LibXC Functional" in n for n in names)
        host = any("Host Functional" in n and "GRAC" not in n for n in names)
        record["xc_route"] = "cuda-libxc" if device else "cuest-host-libxc" if host else "native-host"
        expected = "native-host" if a.route != "gpu-xc" else a.expected_xc
        assert record["xc_route"] == expected, (record["xc_route"], expected)
        record["ok"] = True
    except Exception as exc:
        record.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    atomic_json(out / "result.json", record)
    return 0 if record["ok"] else 1


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(results):
    rows = []
    for build, system, functional in itertools.product(["host", "cuda"], [c[0] for c in CASES], FUNCTIONALS):
        routes = {}
        for route in ROUTES:
            rs = {f: results.get((build, system, functional, route, f)) for f in FRAGMENTS}
            if all(r and r["ok"] for r in rs.values()):
                assert len({r["nbf"] for r in rs.values()}) == 1, "CP basis mismatch"
                es = {f: r["energy_hartree"] for f, r in rs.items()}
                routes[route] = dict(energies=es, ie_hartree=delta(es))
        for route in ROUTES[1:]:
            row = dict(build=build, system=system, functional=functional, route=route,
                       complete="cpu" in routes and route in routes)
            if row["complete"]:
                cpu, gpu = routes["cpu"], routes[route]
                diff = gpu["ie_hartree"] - cpu["ie_hartree"]
                errors = {f: gpu["energies"][f]-cpu["energies"][f] for f in FRAGMENTS}
                row.update(cpu_ie_hartree=cpu["ie_hartree"], gpu_ie_hartree=gpu["ie_hartree"],
                           delta_ie_hartree=diff, delta_ie_kcal_mol=diff*EH_TO_KCAL,
                           signed_total_errors_hartree=errors,
                           max_total_error_hartree=max(map(abs, errors.values())),
                           within_ie_1e_6_Eh=abs(diff) <= 1e-6)
            rows.append(row)
    return rows


def campaign(a):
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).resolve()
    builds = {"host": a.host_package.resolve(), "cuda": a.cuda_package.resolve()}
    binaries = {b: next(p.glob("core*.so")) for b, p in builds.items()}
    hashes = {b: digest(p) for b, p in binaries.items()}
    cases = CASES[:1] if a.preflight else CASES
    functionals = ["m06"] if a.preflight else FUNCTIONALS
    expected_count = len(cases)*len(functionals)*len(builds)*len(ROUTES)*len(FRAGMENTS)
    manifest = dict(job=os.getenv("SLURM_JOB_ID"), cases=cases, functionals=functionals,
                    routes=ROUTES, fragments=FRAGMENTS, packages={b:str(p) for b,p in builds.items()},
                    binary_sha256=hashes, harness_commit=subprocess.check_output(
                        ["git", "-C", str(script.parent), "rev-parse", "HEAD"], text=True).strip(),
                    script_sha256=digest(script), geometry_sha256={s:hashlib.sha256(geometry(s).encode()).hexdigest() for s,_ in CASES},
                    protocol="E_AB - E_A(ghost B) - E_B(ghost A), fixed geometry; no D4, no SAPT, no GRAC. "
                    "Each build compared to its own native CPU reference; historical CUDA build is not a source-matched ablation.",
                    tolerance="Report all errors; flag |IE GPU-CPU| > 1e-6 Eh, not a universal support certification.",
                    gpu=subprocess.check_output(["nvidia-smi","--query-gpu=name,uuid,driver_version","--format=csv,noheader"],text=True),
                    records=[])
    assert "H200" in manifest["gpu"]
    atomic_json(out / "manifest.json", manifest)
    results = {}
    # First case doubles as preflight; all errors are archived, never discarded.
    def comparisons_for_selection():
        return [r for r in summarize(results)
                if r["system"] in {s for s, _ in cases} and r["functional"] in functionals]

    for system, basis in cases:
        for functional, build, route, fragment in itertools.product(functionals, builds, ROUTES, FRAGMENTS):
            name = f"{system}-{functional}-{build}-{route}-{fragment}"
            work = out / name
            env = dict(os.environ, PYTHONPATH=str(builds[build].parent), PYTHONNOUSERSITE="1")
            if build == "cuda":
                env["LD_LIBRARY_PATH"] = str(a.cuda_lib_dir) + ":" + env.get("LD_LIBRARY_PATH", "")
            cmd = [sys.executable, str(script), "--worker", "--output", str(work),
                   "--package", str(builds[build]), "--system", system, "--basis", basis,
                   "--functional", functional, "--route", route, "--fragment", fragment,
                   "--expected-xc", "cuda-libxc" if build == "cuda" else "cuest-host-libxc"]
            with (out / f"{name}.log").open("w") as log:
                try:
                    p = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=240)
                    code = p.returncode
                except subprocess.TimeoutExpired:
                    code = 124
            record = dict(name=name, returncode=code)
            manifest["records"].append(record)
            file = work / "result.json"
            if file.exists():
                results[(build,system,functional,route,fragment)] = json.loads(file.read_text())
            print(name, "exit", code, flush=True)
            atomic_json(out / "manifest.json", manifest)
            atomic_json(out / "comparisons.json", comparisons_for_selection())
    assert hashes == {b:digest(p) for b,p in binaries.items()}, "Binary changed during campaign"
    comparisons = comparisons_for_selection()
    completed = (len(results) == expected_count and all(r["ok"] for r in results.values())
                 and all(r["returncode"] == 0 for r in manifest["records"]))
    atomic_json(out / "COMPLETE.json", dict(calculations_complete=completed, count=len(results),
        expected=expected_count, all_ie_within_tolerance=all(r.get("within_ie_1e_6_Eh",False) for r in comparisons)))
    return 0 if completed else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--preflight", action="store_true")
    for key in ("package", "host-package", "cuda-package", "cuda-lib-dir"):
        p.add_argument("--"+key, type=Path)
    for key in ("system", "basis", "functional", "route", "fragment", "expected-xc"):
        p.add_argument("--"+key)
    args = p.parse_args()
    sys.exit(worker(args) if args.worker else campaign(args))
