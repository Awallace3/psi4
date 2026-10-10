#!/usr/bin/env python3
"""Fresh-process meta-GGA SCF CP matrix; no SAPT/GRAC/D4, retain intrinsic VV10."""
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
FUNCTIONALS = ["m06-l", "m06", "pw6b95", "r2scan", "wb97m-v"]
ROUTES = ["cpu", "gpu-jk", "gpu-xc"]
FRAGMENTS = ["AB", "A", "B"]
EH_TO_KCAL = 627.5094740631
IE_TOLERANCE = 1e-5


def options(basis, route):
    return dict(basis=basis, reference="rhf", scf_type="df", puream=True,
                df_basis_scf="def2-universal-jkfit", use_cuest=route != "cpu",
                cuest_xc=route == "gpu-xc", cuest_mixed_precision=False,
                cuest_sad=False, guess="sad", e_convergence=11, d_convergence=10,
                maxiter=200, fail_on_maxiter=True, dft_radial_points=99,
                dft_spherical_points=590, dft_pruning_scheme="robust",
                dft_vv10_postscf=False, dft_vv10_radial_points=50,
                dft_vv10_spherical_points=146, dft_vv10_rho_cutoff=1e-8)


def delta(energies):
    return energies["AB"] - energies["A"] - energies["B"]


def validate_functional(wfn, name):
    """Prove wB97M-V was not silently replaced by a semilocal-only functional."""
    functional = wfn.functional()
    metadata = dict(name=functional.name(), meta=functional.is_meta(),
                    needs_vv10=functional.needs_vv10(), x_lrc=functional.is_x_lrc(),
                    x_omega=functional.x_omega(), x_alpha=functional.x_alpha(),
                    x_beta=functional.x_beta(), needs_grac=functional.needs_grac())
    assert metadata["meta"], "Not a meta-GGA"
    if metadata["needs_vv10"]:
        metadata["vv10_energy_hartree"] = float(wfn.variable("DFT VV10 ENERGY"))
        metadata.update(vv10_b=functional.vv10_b(), vv10_c=functional.vv10_c())
        assert math.isfinite(metadata["vv10_energy_hartree"])
    if name.lower() == "wb97m-v":
        assert metadata["needs_vv10"] and metadata["x_lrc"], "wB97M-V must retain VV10 and range separation"
        assert metadata["x_omega"] > 0
        assert abs(metadata["vv10_energy_hartree"]) > 1e-14, "VV10 energy was not included"
        assert not metadata["needs_grac"]
        for key, expected in dict(x_omega=.3, x_alpha=.15, x_beta=.85,
                                  vv10_b=6., vv10_c=.01).items():
            assert math.isclose(metadata[key], expected, rel_tol=0, abs_tol=1e-10), (key, metadata)
    return metadata


def validate_energy_components(wfn, energy):
    components = {key: float(wfn.variable(key)) for key in (
        "NUCLEAR REPULSION ENERGY", "ONE-ELECTRON ENERGY", "TWO-ELECTRON ENERGY",
        "DFT XC ENERGY", "DFT VV10 ENERGY")}
    assert all(math.isfinite(v) for v in components.values())
    assert math.isclose(energy, math.fsum(components.values()), rel_tol=0, abs_tol=1e-9), \
        "Returned energy does not include all SCF components, including VV10"
    assert math.isclose(energy, float(wfn.variable("DFT FUNCTIONAL TOTAL ENERGY")),
                        rel_tol=0, abs_tol=1e-9)
    return components


def validate_vv10_runtime_options(runtime_options):
    postscf = runtime_options["DFT_VV10_POSTSCF"]
    # Psi4's option binding may expose booleans as integer 0/1.
    assert type(postscf) in (bool, int) and postscf == 0, \
        "Self-consistent VV10 required (POSTSCF must be disabled)"


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
        record["vv10_runtime_options"] = {
            key: psi4.core.get_option("SCF", key) for key in (
                "DFT_VV10_POSTSCF", "DFT_VV10_RADIAL_POINTS",
                "DFT_VV10_SPHERICAL_POINTS", "DFT_VV10_RHO_CUTOFF")}
        validate_vv10_runtime_options(record["vv10_runtime_options"])
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
        record["functional_metadata"] = validate_functional(wfn, a.functional)
        record["energy_components_hartree"] = validate_energy_components(wfn, float(energy))
        if wfn.has_variable("SCF ITERATIONS"):
            record["scf_iterations"] = int(wfn.variable("SCF ITERATIONS"))
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


def build_environment(build, package, cuda_lib_dir):
    env = dict(os.environ, PYTHONPATH=str(package.parent), PYTHONNOUSERSITE="1")
    if build == "cuda":
        env["LD_LIBRARY_PATH"] = str(cuda_lib_dir) + ":" + env.get("LD_LIBRARY_PATH", "")
    return env


def provenance(package, env):
    """Pin staged drivers, basis data and resolved XC/cuEST libraries, not just core."""
    core = next(package.glob("core*.so"))
    ldd = subprocess.check_output(["ldd", str(core)], env=env, text=True)
    assert "not found" not in ldd, ldd
    files = {core, package / "metadata.py"}
    files.update(package.glob("driver/**/*.py"))
    basis = package.parents[1] / "share/psi4/basis"
    assert basis.is_dir(), basis
    files.update(p for p in basis.rglob("*") if p.is_file())
    cache = package.parents[2] / "CMakeCache.txt"
    if cache.exists():
        files.add(cache)
    libraries = {}
    for line in ldd.splitlines():
        words = line.split()
        if "=>" in words and any(k in words[0].lower() for k in ("libxc", "cuest")):
            path = Path(words[words.index("=>") + 1]).resolve()
            assert path.is_file(), path
            libraries[words[0]] = str(path)
            files.add(path)
    assert any("libxc" in name.lower() for name in libraries), libraries
    source = package.parents[3]
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    return dict(source=str(source), source_checkout_commit=commit, linked_libraries=libraries,
                note="Checkout commit is not by itself the compiled source identity; staged file hashes identify this tested build.",
                sha256={str(p): digest(p) for p in sorted(files)})


def process_result(record, returncode):
    record = dict(record, process_returncode=returncode)
    if returncode:
        record["calculation_reported_ok"] = record.get("ok", False)
        record["ok"] = False
        record["process_error"] = f"Process exited with code {returncode}"
    return record


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
                walls = {f: r.get("wall_s") for f, r in rs.items()}
                if all(isinstance(w, (int, float)) and math.isfinite(w) and w > 0
                       for w in walls.values()):
                    routes[route].update(wall_s=walls, cp_wall_s=math.fsum(walls.values()))
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
                           within_ie_1e_6_Eh=abs(diff) <= 1e-6,
                           within_ie_1e_5_Eh=abs(diff) <= IE_TOLERANCE)
                if "wall_s" in cpu and "wall_s" in gpu:
                    row.update(cpu_fragment_wall_s=cpu["wall_s"],
                               gpu_fragment_wall_s=gpu["wall_s"],
                               cpu_cp_wall_s=cpu["cp_wall_s"], gpu_cp_wall_s=gpu["cp_wall_s"],
                               cp_speedup=cpu["cp_wall_s"]/gpu["cp_wall_s"],
                               dimer_speedup=cpu["wall_s"]["AB"]/gpu["wall_s"]["AB"],
                               timing_scope="Wall around psi4.energy; CP=sum(AB,A,B); excludes Python startup/import.")
            rows.append(row)
    return rows


def campaign(a):
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).resolve()
    builds = {"host": a.host_package.resolve(), "cuda": a.cuda_package.resolve()}
    binaries = {b: next(p.glob("core*.so")) for b, p in builds.items()}
    hashes = {b: digest(p) for b, p in binaries.items()}
    build_provenance = {b: provenance(p, build_environment(b, p, a.cuda_lib_dir))
                        for b, p in builds.items()}
    cases = CASES[:1] if a.preflight else CASES
    functionals = ["m06"] if a.preflight else FUNCTIONALS
    if a.systems:
        cases = [case for case in cases if case[0] in a.systems]
    if a.functionals:
        functionals = [f for f in functionals if f in a.functionals]
    assert cases and functionals
    expected_count = len(cases)*len(functionals)*len(builds)*len(ROUTES)*len(FRAGMENTS)
    manifest = dict(job=os.getenv("SLURM_JOB_ID"), cases=cases, functionals=functionals,
                    routes=ROUTES, fragments=FRAGMENTS, packages={b:str(p) for b,p in builds.items()},
                    binary_sha256=hashes, build_provenance=build_provenance, harness_commit=subprocess.check_output(
                        ["git", "-C", str(script.parent), "rev-parse", "HEAD"], text=True).strip(),
                    script_sha256=digest(script), geometry_sha256={s:hashlib.sha256(geometry(s).encode()).hexdigest() for s,_ in CASES},
                    protocol="E_AB - E_A(ghost B) - E_B(ghost A), fixed geometry; no D4, no SAPT, no GRAC; "
                    "retain functional-intrinsic VV10, including for wB97M-V. "
                    "Each build compared to its own native CPU reference; historical CUDA build is not a source-matched ablation.",
                    tolerance="User acceptance |IE GPU-CPU| <= 1e-5 Eh; retain 1e-6 diagnostic flag.",
                    ie_tolerance_hartree=IE_TOLERANCE, worker_timeout_s=a.worker_timeout,
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
            env = build_environment(build, builds[build], a.cuda_lib_dir)
            cmd = [sys.executable, str(script), "--worker", "--output", str(work),
                   "--package", str(builds[build]), "--system", system, "--basis", basis,
                   "--functional", functional, "--route", route, "--fragment", fragment,
                   "--expected-xc", "cuda-libxc" if build == "cuda" else "cuest-host-libxc"]
            with (out / f"{name}.log").open("w") as log:
                try:
                    p = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=a.worker_timeout)
                    code = p.returncode
                except subprocess.TimeoutExpired:
                    code = 124
            record = dict(name=name, returncode=code)
            manifest["records"].append(record)
            file = work / "result.json"
            if file.exists():
                results[(build,system,functional,route,fragment)] = process_result(json.loads(file.read_text()), code)
            print(name, "exit", code, flush=True)
            atomic_json(out / "manifest.json", manifest)
            atomic_json(out / "comparisons.json", comparisons_for_selection())
    assert hashes == {b:digest(p) for b,p in binaries.items()}, "Binary changed during campaign"
    for pin in build_provenance.values():
        assert all(digest(Path(p)) == expected for p, expected in pin["sha256"].items()), "Build inputs changed during campaign"
    comparisons = comparisons_for_selection()
    completed = (len(results) == expected_count and all(r["ok"] for r in results.values())
                 and all(r["returncode"] == 0 for r in manifest["records"]))
    atomic_json(out / "COMPLETE.json", dict(calculations_complete=completed, count=len(results),
        expected=expected_count, final_provenance_verified=True, ie_tolerance_hartree=IE_TOLERANCE,
        all_ie_within_tolerance=all(r.get("within_ie_1e_5_Eh",False) for r in comparisons)))
    return 0 if completed else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--worker-timeout", type=float, default=240)
    p.add_argument("--systems", nargs="+", choices=[s for s, _ in CASES])
    p.add_argument("--functionals", nargs="+", choices=FUNCTIONALS)
    for key in ("package", "host-package", "cuda-package", "cuda-lib-dir"):
        p.add_argument("--"+key, type=Path)
    for key in ("system", "basis", "functional", "route", "fragment", "expected-xc"):
        p.add_argument("--"+key)
    args = p.parse_args()
    if not math.isfinite(args.worker_timeout) or args.worker_timeout <= 0:
        p.error("--worker-timeout must be finite and positive")
    sys.exit(worker(args) if args.worker else campaign(args))
