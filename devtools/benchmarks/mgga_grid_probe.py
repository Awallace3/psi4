#!/usr/bin/env python3
"""Decompose XC error using actual cuEST points, weights and density ingredients."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np

from mgga_ie import (atomic_json, build_environment, digest, geometry, options,
                     process_result, provenance)
from mgga_probe import validate_integrated_density, validate_mo_metadata


def preload_paths(prefix, shim):
    # Site GCC may embed DT_RPATH in a preload object, overriding LD_LIBRARY_PATH
    # before Python imports Psi4. Load the tested environment's runtimes first.
    return [Path(prefix)/"lib/libgcc_s.so.1", Path(prefix)/"lib/libstdc++.so.6",
            Path(shim)]


def runtime_maps(prefix):
    expected = {name: (Path(prefix)/"lib"/name).resolve()
                for name in ("libgcc_s.so.1", "libstdc++.so.6")}
    found = {name: set() for name in expected}
    for line in Path("/proc/self/maps").read_text().splitlines():
        path = line.split()[-1]
        if path.startswith("/"):
            for name in expected:
                if Path(path).name.startswith(name):
                    found[name].add(Path(path).resolve())
    assert all(found[name] == {path} for name, path in expected.items()), (
        "Unexpected C++ runtime selection", expected, found)
    return {name: str(path) for name, path in expected.items()}


def restricted_ingredients(raw):
    """cuEST one-spin rho/grad/tau -> native restricted functional inputs."""
    return dict(RHO_A=2*raw[:, 0],
                GAMMA_AA=4*np.sum(raw[:, 1:4]**2, axis=1),
                TAU_A=2*raw[:, 4])


def native_evaluation(psi4, seed, functional, coordinates, weights, raw=None):
    """Use native AO collocation/LibXC at supplied points, with tight screening."""
    chunk = 256
    basis = seed.basisset()
    extents = psi4.core.BasisExtents(basis, 1e-20)
    computer = psi4.core.RKSFunctions(basis, chunk, basis.nbf())
    computer.set_ansatz(2)
    computer.set_pointers(seed.Da())
    function = psi4.driver.dft.build_superfunctional(functional, True)[0]
    assert function.max_points() >= chunk
    energies, density_energies = [], []
    native = np.empty((len(weights), 5))
    for start in range(0, len(weights), chunk):
        stop = min(start+chunk, len(weights))
        vectors = [psi4.core.Vector.from_array(np.ascontiguousarray(v))
                   for v in (*coordinates[start:stop].T, weights[start:stop])]
        block = psi4.core.BlockOPoints(*vectors, extents)
        computer.compute_points(block)
        values = computer.point_values()
        n = stop-start
        native[start:stop, 0] = values["RHO_A"].np[:n]/2
        for k, name in enumerate(("RHO_AX", "RHO_AY", "RHO_AZ"), 1):
            native[start:stop, k] = values[name].np[:n]/2
        native[start:stop, 4] = values["TAU_A"].np[:n]/2
        output = function.compute_functional(values, n)
        energy_density = output["V"].np[:n].copy()
        assert np.isfinite(energy_density).all()
        energies.append(float(np.dot(weights[start:stop], energy_density)))
        if raw is not None:
            supplied = dict(values)
            supplied.update({k: psi4.core.Vector.from_array(np.ascontiguousarray(v))
                             for k, v in restricted_ingredients(raw[start:stop]).items()})
            output = function.compute_functional(supplied, n)
            energy_density = output["V"].np[:n].copy()
            assert np.isfinite(energy_density).all()
            density_energies.append(float(np.dot(weights[start:stop], energy_density)))
    # Accurate deterministic chunk summation without changing the quadrature.
    import math
    return (math.fsum(energies),
            math.fsum(density_energies) if raw is not None else None, native)


def worker(a):
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(out)
    result = dict(ok=False, functional=a.functional, fragment=a.fragment, package=str(a.package),
                  protocol="Same seed orbitals; actual C-API cuEST grid capture. "
                           "CPU collocation on each grid at basis tolerance 1e-20; "
                           "native LibXC on both native and captured density ingredients.")
    try:
        import psi4
        result["runtime_maps"] = runtime_maps(os.environ["CONDA_PREFIX"])
        assert Path(psi4.__file__).resolve().parent == a.package.resolve()
        assert not os.getenv("MGGA_CAPTURE_DIR")
        psi4.core.set_output_file(str(out/"psi4.out"), False)
        psi4.set_memory("32 GiB")
        psi4.set_num_threads(8)
        baseline = options("aug-cc-pvdz", "cpu")
        result["options"] = baseline
        result["geometry"] = geometry("benzene")
        psi4.set_options(baseline)
        mol = psi4.geometry(result["geometry"])
        if a.fragment == "A":
            mol = mol.extract_subsets(1, 2)
        elif a.fragment == "B":
            mol = mol.extract_subsets(2, 1)
        mol.update_geometry()
        result["molecule"] = mol.to_string(dtype="psi4")
        energy, seed = psi4.energy(a.functional, molecule=mol, return_wfn=True)
        ca, da = seed.Ca().np.copy(), seed.Da().np.copy()
        cocc = seed.Ca_subset("AO", "OCC").np.copy()
        overlap = psi4.core.MintsHelper(seed.basisset()).ao_overlap().np.copy()
        nocc = seed.nalpha()
        result.update(nbf=seed.basisset().nbf(), nelectron=2*nocc)
        np.testing.assert_allclose(cocc.T @ overlap @ cocc, np.eye(nocc), atol=1e-10, rtol=0)
        result["seed_energy_hartree"] = energy
        np.savez(out/"seed.npz", ca=ca, da=da, cocc_ao=cocc, overlap_ao=overlap)
        measurements = {}
        for route in ("cpu", "gpu-control", "gpu-capture"):
            psi4.set_options(dict(baseline, use_cuest=route != "cpu", cuest_xc=route != "cpu"))
            if route == "gpu-capture":
                capture = out/"capture"
                capture.mkdir()
                os.environ["MGGA_CAPTURE_DIR"] = str(capture)
            function = psi4.driver.dft.build_superfunctional(a.functional, True)[0]
            target = psi4.core.RHF(seed, function)
            validate_mo_metadata(target, ca.shape)
            target.force_occpi(psi4.core.Dimension([nocc]), psi4.core.Dimension([0]))
            target.Ca().np[:] = ca
            target.Da().np[:] = da
            target.epsilon_a().np[:] = seed.epsilon_a().np
            np.testing.assert_allclose(target.Ca_subset("AO", "OCC").np, cocc, atol=1e-12, rtol=0)
            psi4.core.clean_timers()
            target.form_V()
            quad = dict(target.V_potential().quadrature_values())
            validate_integrated_density(quad, 2*nocc)
            names = ["/".join(t["timer_path"]) for t in psi4.core.get_timer_records().values()]
            assert any("Host Functional" in n for n in names) == (route != "cpu")
            potential = target.Va().np.copy()
            assert np.isfinite(potential).all()
            measurements[route] = dict(quadrature=quad)
            np.save(out/f"{route}-v.npy", potential)
            if route == "cpu":
                blocks = target.V_potential().grid().blocks()
                cpu_points = np.concatenate([np.column_stack(
                    [b.x().np, b.y().np, b.z().np]) for b in blocks])
                cpu_weights = np.concatenate([b.w().np for b in blocks])
                result["cpu_grid_orientation"] = target.V_potential().grid().orientation().np.tolist()
                np.savez(out/"cpu-grid.npz", coordinates=cpu_points, weights=cpu_weights)
            target.V_potential().finalize()
            del target, function
            os.environ.pop("MGGA_CAPTURE_DIR", None)
        result["measurements"] = measurements
        control = measurements["gpu-control"]["quadrature"]["FUNCTIONAL"]
        captured = measurements["gpu-capture"]["quadrature"]["FUNCTIONAL"]
        assert abs(control-captured) < 1e-10, "Capture changed the XC energy"
        control_v = np.load(out/"gpu-control-v.npy")
        captured_v = np.load(out/"gpu-capture-v.npy")
        np.testing.assert_allclose(control_v, captured_v, atol=1e-10, rtol=0)
        info = json.loads((capture/"grid.json").read_text())
        count = info["npoints"]
        coordinates = np.fromfile(capture/"coordinates.f64", dtype=np.float64).reshape(count, 3)
        weights = np.fromfile(capture/"weights.f64", dtype=np.float64)
        raw = np.fromfile(capture/"density.f64", dtype=np.float64).reshape(count, 5)
        assert len(weights) == count
        assert np.isfinite(coordinates).all() and np.isfinite(weights).all() and np.isfinite(raw).all()
        integrated = float(2*np.dot(weights, raw[:, 0]))
        assert abs(integrated - measurements["gpu-capture"]["quadrature"]["RHO_A"]) < 1e-8
        result["capture_validated"] = True
        result["captured_integrated_density"] = integrated
        # Return to native settings before invoking native PointFunctions/LibXC.
        psi4.set_options(baseline)
        cpu_tight, _, _ = native_evaluation(psi4, seed, a.functional, cpu_points, cpu_weights)
        same_grid, same_density, native = native_evaluation(
            psi4, seed, a.functional, coordinates, weights, raw)
        native_integrated = float(2*np.dot(weights, native[:, 0]))
        validate_integrated_density({"RHO_A": native_integrated}, 2*nocc)
        result["native_integrated_density_on_cuest_grid"] = native_integrated
        np.save(capture/"native-ingredients.npy", native)
        cpu_energy = measurements["cpu"]["quadrature"]["FUNCTIONAL"]
        result["energy_decomposition_hartree"] = dict(
            cpu_original=cpu_energy, cpu_tight=cpu_tight,
            native_on_cuest_grid=same_grid, native_on_cuest_density=same_density, cuest=captured,
            cpu_screening=cpu_tight-cpu_energy,
            grid_effect=same_grid-cpu_tight,
            collocation_effect=same_density-same_grid,
            functional_adapter_effect=captured-same_density,
            total=captured-cpu_energy)
        difference = raw-native
        result["ingredient_errors"] = {
            name: dict(max_abs=float(np.max(np.abs(difference[:, k]))),
                       weighted_l1=float(np.dot(np.abs(weights), np.abs(difference[:, k]))),
                       relative_l1=float(np.dot(np.abs(weights), np.abs(difference[:, k])) /
                                         max(np.dot(np.abs(weights), np.abs(native[:, k])), 1e-300)))
            for k, name in enumerate(("rho_one_spin", "gradient_x", "gradient_y", "gradient_z", "tau_one_spin"))}
        from scipy.spatial import cKDTree
        distance, index = cKDTree(cpu_points).query(coordinates, workers=8)
        matched = distance < 1e-9
        result["coordinate_comparison"] = dict(
            cpu_count=len(cpu_weights), cuest_count=count, matched_count=int(matched.sum()),
            match_fraction=float(matched.mean()),
            nearest_distance_quantiles=np.quantile(distance, [0, .5, .95, 1]).tolist(),
            matched_weight_max_abs=float(np.max(np.abs(weights[matched]-cpu_weights[index[matched]])))
                if matched.any() else None)
        result["artifact_sha256"] = {str(p.relative_to(out)): digest(p)
                                     for p in capture.iterdir() if p.is_file()}
        result["ok"] = True
    except Exception as exc:
        result.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    atomic_json(out/"result.json", result)
    return 0 if result["ok"] else 1


def campaign(a):
    assert len(set(a.fragments)) == len(a.fragments), "Duplicate fragment selection"
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    package = a.host_package.resolve()
    env = build_environment("host", package, a.cuda_lib_dir)
    pin = provenance(package, env)
    script = Path(__file__).resolve()
    preloads = preload_paths(os.environ["CONDA_PREFIX"], a.shim.resolve())
    assert all(p.is_file() for p in preloads), preloads
    preload_hashes = {str(p.resolve()): digest(p) for p in preloads}
    manifest = dict(job=os.getenv("SLURM_JOB_ID"), package=str(package), provenance=pin,
                    shim=str(a.shim.resolve()), shim_sha256=digest(a.shim),
                    source_sha256=digest(script.with_name("mgga_capture.cc")),
                    script_sha256=digest(script), records=[],
                    preload_sha256=preload_hashes, fragments=a.fragments,
                    commit=subprocess.check_output(
                        ["git", "-C", str(script.parent), "rev-parse", "HEAD"], text=True).strip(),
                    gpu=subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
                        text=True))
    assert "H200" in manifest["gpu"]
    atomic_json(out/"manifest.json", manifest)
    import itertools
    for functional, fragment in itertools.product(("m06", "m06-l"), a.fragments):
        name = functional if fragment == "AB" else f"{functional}-{fragment}"
        child_env = dict(env, LD_PRELOAD=":".join(map(str, preloads)))
        child_env.pop("MGGA_CAPTURE_DIR", None)
        cmd = [sys.executable, str(script), "--worker", "--output", str(out/name),
               "--package", str(package), "--functional", functional, "--fragment", fragment]
        with (out/f"{name}.log").open("w") as log:
            try:
                code = subprocess.run(cmd, env=child_env, stdout=log,
                                      stderr=subprocess.STDOUT, timeout=1200).returncode
            except subprocess.TimeoutExpired:
                code = 124
        file = out/name/"result.json"
        record = process_result(json.loads(file.read_text()) if file.exists()
                                else dict(ok=False, error="Missing result.json"), code)
        manifest["records"].append(dict(functional=functional, fragment=fragment,
                                        returncode=code, result=record))
        atomic_json(out/"manifest.json", manifest)
        print(name, "exit", code, "error", record.get("error"), flush=True)
    assert digest(a.shim) == manifest["shim_sha256"]
    assert all(digest(Path(p)) == h for p, h in preload_hashes.items()), "Runtime changed"
    assert all(digest(Path(p)) == h for p, h in pin["sha256"].items()), "Build changed"
    complete = all(r["result"]["ok"] for r in manifest["records"])
    atomic_json(out/"COMPLETE.json", dict(calculations_complete=complete,
                                          final_provenance_verified=True,
                                          note="XC decomposition, not new interaction energies."))
    return 0 if complete else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--worker", action="store_true")
    for key in ("package", "host-package", "cuda-package", "cuda-lib-dir", "shim"):
        p.add_argument("--"+key, type=Path)
    p.add_argument("--functional")
    p.add_argument("--fragment", choices=("AB", "A", "B"), default="AB")
    p.add_argument("--fragments", choices=("AB", "A", "B"), nargs="+", default=["AB"])
    a = p.parse_args()
    raise SystemExit(worker(a) if a.worker else campaign(a))
