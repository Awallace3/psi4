"""Standalone protein83 RHF energy diagnostic: no SAPT or monomer calculations.

Run once per fresh process, with the already-selected Psi4 environment.
This is a numerical reproducer, not an accepted timing/qualification gate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from peptide_scf import INPUT_SHA


def scf_options(backend):
    if backend not in ("cpu", "gpu"):
        raise ValueError("Explicit cpu/gpu backend required")
    return {
        "basis": "jun-cc-pvdz", "df_basis_scf": "aug-cc-pvdz-jkfit",
        "reference": "rhf", "scf_type": "df", "puream": True,
        "use_cuest": backend == "gpu", "cuest_mixed_precision": False,
        "cuest_gemm_min_dim": 0, "ints_tolerance": 1e-16,
        "e_convergence": 10, "d_convergence": 12, "maxiter": 100,
        "guess": "sad", "save_jk": True, "df_ints_io": "save",
    }


def energy_only(psi4, geometry, backend):
    """Exactly one public energy call on the full dimer."""
    psi4.set_options(scf_options(backend))
    molecule = psi4.geometry(geometry)
    return psi4.energy("hf", molecule=molecule, return_wfn=True)


def run(psi4, inputs_path, output, backend, threads):
    import numpy as np

    raw = inputs_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != INPUT_SHA:
        raise RuntimeError("Frozen jun-cc-pVDZ input manifest changed")
    inputs = json.loads(raw)
    point = inputs["systems"]["protein83"]
    geometry = point["geometry"]
    if hashlib.sha256(geometry.encode()).hexdigest() != point["geometry_sha256"]:
        raise RuntimeError("Frozen protein83 geometry changed")
    for item in inputs["basis_files"].values():
        basis = Path(psi4.core.get_datadir()) / "basis" / item["filename"]
        if hashlib.sha256(basis.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError(f"Installed basis differs: {basis.name}")
    if backend == "cpu" and (
        os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("SLURM_JOB_GPUS")
    ):
        raise RuntimeError("CPU control requires hidden GPUs and no GPU allocation")
    # A new directory prevents accidentally mixing fresh-process repeats.
    output.mkdir(parents=True, exist_ok=False)
    output = output.resolve()
    psi4.core.set_output_file(str(output / "psi4.out"), False)
    psi4.set_memory("96 GiB")
    psi4.set_num_threads(threads)
    report = {
        "status": "RUNNING", "scope": __doc__, "backend": backend,
        "options": scf_options(backend), "geometry": geometry,
        "input_sha256": INPUT_SHA, "geometry_sha256": point["geometry_sha256"],
        "threads": threads, "host_memory": "96 GiB",
        "psi4_file": psi4.__file__, "core_file": psi4.core.__file__,
        "psi4_version": psi4.__version__, "node": os.uname().nodename,
        "job_id": os.environ.get("SLURM_JOB_ID"), "accepted_speedup_count": 0,
        "entrypoint": 'psi4.energy("hf", molecule=dimer, return_wfn=True)',
    }
    started = time.perf_counter()
    try:
        energy, wfn = energy_only(psi4, geometry, backend)
        if wfn.basisset().nbf() != point["spherical_basis_counts"]["primary"]:
            raise RuntimeError("Unexpected AO dimension")
        # Freeze the converged full-dimer inputs for subsequent fixed-density
        # JK controls. Do not run any additional integral calculations here.
        arrays = {
            "Cocc": np.array(wfn.Ca_subset("AO", "OCC").np, copy=True),
            "D": np.array(wfn.Da().np, copy=True),
            "S": np.array(wfn.S().np, copy=True),
            "H": np.array(wfn.H().np, copy=True),
        }
        if not np.isfinite(energy) or not all(np.isfinite(a).all() for a in arrays.values()):
            raise RuntimeError("Nonfinite converged output")
        np.testing.assert_allclose(arrays["D"], arrays["Cocc"] @ arrays["Cocc"].T,
                                   atol=1e-12, rtol=1e-12)
        snapshot = output / "dimer-wavefunction.npz"
        np.savez(snapshot, **arrays)
        report.update(
            status="CONVERGED_DIAGNOSTIC_ONLY", energy_Eh=float(energy),
            nuclear_repulsion_Eh=float(wfn.molecule().nuclear_repulsion_energy()),
            nbf=wfn.basisset().nbf(),
            snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        )
    except BaseException as error:
        report.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - started
        report["timers"] = psi4.core.get_timer_dict()
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        psi4.core.flush_outfile()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--threads", type=int, required=True)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("Positive thread count required")
    import psi4
    prefix = args.prefix.resolve(strict=True)
    for module in (psi4, psi4.core):
        if not Path(module.__file__).resolve().is_relative_to(prefix):
            raise RuntimeError("Wrong Psi4 import: " + module.__file__)
    run(psi4, args.inputs, args.output, args.backend, args.threads)


if __name__ == "__main__":
    main()
