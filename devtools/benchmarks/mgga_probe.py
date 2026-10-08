#!/usr/bin/env python3
"""Fixed-orbital XC probes on existing binaries, without modifying scientific code."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from mgga_ie import (atomic_json, build_environment, digest, geometry, options,
                     process_result, provenance)

# Each profile differs from baseline by one option. Nominal settings alone do
# not establish identical CPU/cuEST point coordinates, orientations or weights.
PROFILES = {
    "baseline": {},
    "density-1e-18": {"dft_density_tolerance": 1e-18},
    "basis-1e-16": {"dft_basis_tolerance": 1e-16},
    "all-weights": {"dft_weights_tolerance": -1.0},
    "no-pruning": {"dft_pruning_scheme": "none"},
    "becke": {"dft_nuclear_scheme": "becke"},
    "stratmann": {"dft_nuclear_scheme": "stratmann"},
}


def validate_mo_metadata(target, shape):
    """Reject bare, uninitialized HF objects before occupied AO extraction."""
    assert target.nirrep() == 1, "Replay requires C1"
    assert target.nmo() == shape[1] and target.nmopi()[0] == shape[1], (
        "Uninitialized/inconsistent MO metadata: C_subset_helper source stride "
        "must equal the Ca column dimension", target.nmo(), target.nmopi()[0], shape)


def validate_integrated_density(quad, nelectron):
    # Both restricted routes store the total density in RHO_A (and RHO_B).
    # This generous quadrature tolerance is an invariant check, not IE accuracy.
    assert abs(quad["RHO_A"] - nelectron) < 1e-3, (
        "Invalid electron normalization; do not interpret XC differences",
        quad["RHO_A"], nelectron)


def worker(a):
    import numpy as np
    import psi4

    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(out)
    record = dict(ok=False, functional=a.functional, system="benzene", fragment="AB",
                  psi4_module=psi4.__file__, version=psi4.__version__,
                  protocol="One native CPU baseline SCF; identical C and D in every XC probe. "
                           "Profile changes affect quadrature only, not the seed orbitals.",
                  profiles=PROFILES, probes=[])
    try:
        assert Path(psi4.__file__).resolve().parent == a.package.resolve()
        psi4.core.set_output_file(str(out / "psi4.out"), False)
        psi4.set_memory("32 GiB")
        psi4.set_num_threads(8)
        baseline = options("aug-cc-pvdz", "cpu")
        try:
            psi4.core.get_global_option("CUEST_SAD")
        except Exception:
            baseline.pop("cuest_sad")
        record["baseline_options"] = baseline
        record["geometry"] = geometry("benzene")
        psi4.set_options(baseline)
        mol = psi4.geometry(record["geometry"])
        energy, seed = psi4.energy(a.functional, molecule=mol, return_wfn=True)
        ca = seed.Ca().np.copy()
        da = seed.Da().np.copy()
        nocc = seed.nalpha()
        assert seed.nirrep() == 1 and nocc == seed.nbeta()
        np.testing.assert_allclose(da, ca[:, :nocc] @ ca[:, :nocc].T, atol=1e-12, rtol=0)
        cocc_ao = seed.Ca_subset("AO", "OCC").np.copy()
        overlap_ao = psi4.core.MintsHelper(seed.basisset()).ao_overlap().np.copy()
        np.testing.assert_allclose(cocc_ao.T @ overlap_ao @ cocc_ao,
                                   np.eye(nocc), atol=1e-10, rtol=0)
        record.update(seed_energy_hartree=float(energy), nbf=seed.nso(), nocc=nocc,
                      seed_xc_energy_hartree=seed.variable("DFT XC ENERGY"),
                      density_sha256=hashlib.sha256(da.tobytes()).hexdigest(),
                      orbitals_sha256=hashlib.sha256(ca.tobytes()).hexdigest())
        np.savez(out / "seed.npz", ca=ca, da=da, cocc_ao=cocc_ao, overlap_ao=overlap_ao)
        baseline_cpu = None
        for profile, changes in PROFILES.items():
            pair = {}
            for route in ("cpu", "gpu-xc"):
                config = dict(baseline, **changes, use_cuest=route != "cpu",
                              cuest_xc=route == "gpu-xc")
                psi4.set_options(config)
                psi4.core.clean_timers()
                # RHF::form_V sets both D and occupied C. VBase.set_D alone
                # cannot drive cuEST, and set_Cocc is not exposed to Python.
                func = psi4.driver.dft.build_superfunctional(a.functional, True)[0]
                # A bare Wavefunction.build has nmopi=0 until form_Shalf.
                # Ca_subset uses that metadata as a source stride, so copying
                # the Ca array alone can silently upload the wrong AO orbitals.
                # Inherit the fully initialized seed's metadata and transform.
                target = psi4.core.RHF(seed, func)
                validate_mo_metadata(target, ca.shape)
                target.force_occpi(psi4.core.Dimension([nocc]), psi4.core.Dimension([0]))
                target.Ca().np[:] = ca
                target.Da().np[:] = da
                target.epsilon_a().np[:] = seed.epsilon_a().np
                uploaded = target.Ca_subset("AO", "OCC").np.copy()
                np.testing.assert_allclose(uploaded, cocc_ao, atol=1e-12, rtol=0)
                np.testing.assert_allclose(uploaded.T @ overlap_ao @ uploaded,
                                           np.eye(nocc), atol=1e-10, rtol=0)
                np.testing.assert_allclose(uploaded @ uploaded.T,
                                           seed.Da_subset("AO").np, atol=1e-12, rtol=0)
                target.form_V()
                quad = dict(target.V_potential().quadrature_values())
                potential = target.Va().np.copy()
                assert np.isfinite(potential).all()
                assert np.isfinite(quad["FUNCTIONAL"])
                validate_integrated_density(quad, 2*nocc)
                np.testing.assert_array_equal(target.Ca().np, ca)
                np.testing.assert_array_equal(target.Da().np, da)
                timers = psi4.core.get_timer_records()
                names = ["/".join(t["timer_path"]) for t in timers.values()]
                device = any("CUDA LibXC Functional" in n for n in names)
                host = any("Host Functional" in n for n in names)
                observed = "cuda-libxc" if device else "cuest-host-libxc" if host else "native-host"
                expected = "native-host" if route == "cpu" else a.expected_xc
                assert observed == expected, (observed, expected)
                probe = dict(profile=profile, route=route, options=config,
                             quadrature=quad, xc_route=observed, timers=timers,
                             ao_orbital_replay_validated=True, nmo=target.nmo(),
                             integrated_density_error=quad["RHO_A"]-2*nocc)
                pair[route] = (quad, potential)
                np.save(out / f"{profile}-{route}-v.npy", potential)
                record["probes"].append(probe)
                target.V_potential().finalize()
                del target, func
                atomic_json(out / "result.json", record)
            cpu_q, cpu_v = pair["cpu"]
            gpu_q, gpu_v = pair["gpu-xc"]
            if baseline_cpu is None:
                baseline_cpu = cpu_q["FUNCTIONAL"]
                assert abs(baseline_cpu - record["seed_xc_energy_hartree"]) < 1e-8, (
                    "Fixed-density CPU probe does not reproduce seed XC energy", baseline_cpu,
                    record["seed_xc_energy_hartree"])
            dv = gpu_v - cpu_v
            record.setdefault("comparisons", []).append(dict(
                profile=profile, delta_xc_hartree=gpu_q["FUNCTIONAL"]-cpu_q["FUNCTIONAL"],
                cpu_xc_shift_hartree=cpu_q["FUNCTIONAL"]-baseline_cpu,
                max_abs_potential_hartree=float(np.max(np.abs(dv))),
                frobenius_potential_hartree=float(np.linalg.norm(dv)),
                density_contraction_hartree=float(2*np.sum(da*dv))))
            atomic_json(out / "result.json", record)
        record["ok"] = True
    except Exception as exc:
        record.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    atomic_json(out / "result.json", record)
    return 0 if record["ok"] else 1


def campaign(a):
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).resolve()
    # Start with the validated host-LibXC build: CUDA placement is not needed
    # to reproduce this symptom, and the two binaries are not source matched.
    package = a.host_package.resolve()
    env = build_environment("host", package, a.cuda_lib_dir)
    pin = provenance(package, env)
    manifest = dict(job=os.getenv("SLURM_JOB_ID"), package=str(package), provenance=pin,
                    harness_commit=subprocess.check_output(
                        ["git", "-C", str(script.parent), "rev-parse", "HEAD"], text=True).strip(),
                    script_sha256=digest(script), profiles=PROFILES,
                    gpu=subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=name,uuid,driver_version",
                         "--format=csv,noheader"], text=True),
                    records=[], note="Fixed-density diagnostic; not an IE campaign or source-matched ablation.")
    assert "H200" in manifest["gpu"]
    atomic_json(out / "manifest.json", manifest)
    for functional in ("m06", "m06-l"):
        directory = out / functional
        cmd = [sys.executable, str(script), "--worker", "--output", str(directory),
               "--functional", functional, "--package", str(package),
               "--expected-xc", "cuest-host-libxc"]
        with (out / f"{functional}.log").open("w") as log:
            try:
                code = subprocess.run(cmd, env=env, stdout=log,
                                      stderr=subprocess.STDOUT, timeout=600).returncode
            except subprocess.TimeoutExpired:
                code = 124
        file = directory / "result.json"
        result = process_result(json.loads(file.read_text()) if file.exists()
                                else dict(ok=False, error="Missing result.json"), code)
        manifest["records"].append(dict(functional=functional, returncode=code,
                                        result=result))
        atomic_json(out / "manifest.json", manifest)
        print(functional, "exit", code, "error", result.get("error"), flush=True)
    assert all(digest(Path(p)) == h for p, h in pin["sha256"].items()), "Build inputs changed"
    complete = all(r["result"]["ok"] for r in manifest["records"])
    atomic_json(out / "COMPLETE.json", dict(calculations_complete=complete,
                                           note="Probe execution success is not parity certification."))
    return 0 if complete else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--worker", action="store_true")
    for name in ("host-package", "cuda-package", "cuda-lib-dir", "package"):
        p.add_argument("--"+name, type=Path)
    p.add_argument("--functional")
    p.add_argument("--expected-xc")
    a = p.parse_args()
    raise SystemExit(worker(a) if a.worker else campaign(a))
