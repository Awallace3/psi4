#!/usr/bin/env python3
"""Real FDDS-only benchmark: prepare physical SCF caches once, replay identically.

SCF preparation is reported separately and never counted as dispersion speedup.
No random tensors, reduced quadrature, changed kernels, or D4 substitution.
CPU/GPU comparisons must use the same cache SHA256 and scientific options.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

SCIENCE = {
    "basis": "6-31+g**", "df_basis_scf": "def2-universal-jkfit",
    "df_basis_mp2": "aug-cc-pvdz-ri", "puream": True,
    "scf_type": "df", "reference": "rhf", "e_convergence": 9,
    "d_convergence": 8, "maxiter": 150,
    "dft_radial_points": 75, "dft_spherical_points": 302,
    "dft_grac_shift": .136, "cuest_mixed_precision": False,
}
FUNCTIONAL = "pbe0"
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


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def libraries():
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        path = line.split()[-1]
        if Path(path).name.startswith(("core.", "libcuest", "libcublas", "libcusolver",
                                      "libcudart", "libxc", "libmkl", "libint2", "libeinsums",
                                      "libopenblas", "libscipy_openblas")) and Path(path).is_file():
            paths.add(path)
    return {p: digest(p) for p in sorted(paths)}


def prepare(psi4, args):
    if args.system == "water":
        geometry = WATER
    else:
        spec = importlib.util.spec_from_file_location("timing_geometries", args.suite / "systems.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        geometry = module.SYSTEMS[args.system]
    molecule = psi4.geometry(geometry)
    molecule.reset_point_group("c1")
    molecule.fix_orientation(True)
    molecule.fix_com(True)
    molecule.update_geometry()
    if molecule.natom() > 89:
        raise ValueError("Campaign is restricted to protein89 and smaller (at most89 atoms)")
    if molecule.nfragments() != 2:
        raise ValueError("FDDS benchmark requires exactly two monomers")
    psi4.set_options({**SCIENCE, "use_cuest": False, "cuest_xc": False, "cuest_sad": False})
    arrays, scf = {}, []
    for label, real, ghost in (("A", 1, 2), ("B", 2, 1)):
        monomer = molecule.extract_subsets(real, ghost)
        start = time.perf_counter()
        energy, wfn = psi4.energy(FUNCTIONAL, molecule=monomer, return_wfn=True)
        scf.append({"monomer": label, "energy_Eh": energy, "seconds": time.perf_counter() - start})
        for name, value in (
            ("Cocc", wfn.Ca_subset("AO", "OCC")),
            ("Cvir", wfn.Ca_subset("AO", "VIR")),
            ("eps_occ", wfn.epsilon_a_subset("AO", "OCC")),
            ("eps_vir", wfn.epsilon_a_subset("AO", "VIR")),
        ):
            arrays[f"{name}_{label}"] = np.array(value, copy=True)
        arrays[f"D_{label}"] = arrays[f"Cocc_{label}"] @ arrays[f"Cocc_{label}"].T
        del wfn
        psi4.core.clean()
    # Exclusive directory creation prevents accidental replacement of an input cache.
    args.cache.mkdir(parents=True, exist_ok=False)
    np.savez(args.cache / "orbitals.npz", **arrays)
    metadata = {
        "system": args.system, "geometry": molecule.to_string(dtype="psi4"),
        "natom": molecule.natom(), "options": SCIENCE, "functional": FUNCTIONAL,
        "is_hybrid": True, "x_alpha": .25, "leg_points": 10, "leg_lambda": .3,
        "scf_preparation": scf, "orbitals_sha256": digest(args.cache / "orbitals.npz"),
        "suite_sha256": digest(args.suite / "systems.py") if args.system != "water" else None,
        "runner_sha256": digest(__file__), "core_sha256": digest(psi4.core.__file__),
        "source_pin": args.binary_pin, "threads": args.threads,
    }
    atomic_json(args.cache / "metadata.json", metadata)
    return metadata


def replay(psi4, args, metadata):
    from psi4.driver.procrouting.sapt import sapt_mp2_terms
    assert metadata["options"] == SCIENCE and metadata["functional"] == FUNCTIONAL
    assert digest(args.cache / "orbitals.npz") == metadata["orbitals_sha256"]
    assert metadata["natom"] <= 89
    molecule = psi4.geometry(metadata["geometry"])
    molecule.reset_point_group("c1")
    molecule.update_geometry()
    psi4.set_options({**SCIENCE, "use_cuest": args.arm == "gpu",
                      "cuest_xc": False, "cuest_sad": False, "cuest_gemm_min_dim": 0})
    primary = psi4.core.BasisSet.build(molecule, "BASIS", SCIENCE["basis"])
    auxiliary = psi4.core.BasisSet.build(molecule, "DF_BASIS_MP2", SCIENCE["df_basis_mp2"],
                                        "RIFIT", SCIENCE["basis"])
    with np.load(args.cache / "orbitals.npz", allow_pickle=False) as arrays:
        cache = {key: (psi4.core.Vector.from_array(arrays[key]) if key.startswith("eps_")
                       else psi4.core.Matrix.from_array(arrays[key])) for key in arrays}
    result = {
        "arm": args.arm, "system": metadata["system"], "ok": False,
        "natom": metadata["natom"], "nbf": primary.nbf(), "naux": auxiliary.nbf(),
        "occupied": [cache[f"Cocc_{s}"].np.shape[1] for s in ("A", "B")],
        "virtual": [cache[f"Cvir_{s}"].np.shape[1] for s in ("A", "B")],
        "threads": args.threads, "memory": args.memory, "options": SCIENCE,
        "binary_pin": args.binary_pin, "runner_sha256": digest(__file__),
        "cache_sha256": metadata["orbitals_sha256"], "metadata_sha256": digest(args.cache / "metadata.json"),
        "psi4_package": psi4.__file__, "core_sha256": digest(psi4.core.__file__),
        "job_id": os.environ.get("SLURM_JOB_ID"), "hostname": os.uname().nodename,
    }
    config = io.StringIO()
    with contextlib.redirect_stdout(config):
        np.show_config()
    result["numpy_version"] = np.__version__
    result["numpy_configuration"] = config.getvalue()
    pinv_calls = []
    original = np.linalg.pinv

    def timed_pinv(a, *positional, **kwargs):
        start = time.perf_counter()
        try:
            return original(a, *positional, **kwargs)
        finally:
            pinv_calls.append({"shape": list(a.shape), "seconds": time.perf_counter() - start})

    np.linalg.pinv = timed_pinv
    psi4.core.clean_timers()
    start = time.perf_counter()
    try:
        energies = sapt_mp2_terms.df_fdds_dispersion(
            primary, auxiliary, cache, metadata["is_hybrid"], metadata["x_alpha"],
            leg_points=metadata["leg_points"], leg_lambda=metadata["leg_lambda"])
        result["fdds_seconds"] = time.perf_counter() - start
        result["energies_Eh"] = {k: float(v) for k, v in energies.items()}
        result["timers"] = psi4.core.get_timer_records()
        calls = sum(int(value["n_calls"]) for key, value in psi4.core.get_timer_dict().items()
                    if "cuEST FDDS: contractions" in key)
        # Timer format is validated explicitly below; a missing GPU route is an error.
        result["gpu_timer_calls"] = calls
        exchange_calls = sum(int(value["n_calls"]) for key, value in psi4.core.get_timer_dict().items()
                             if "cuEST FDDS: hybrid exchange" in key)
        result["gpu_exchange_timer_calls"] = exchange_calls
        if args.arm == "gpu" and calls == 0:
            raise RuntimeError("Forced GPU benchmark did not execute FDDS GPU contractions")
        if args.arm == "cpu" and calls:
            raise RuntimeError("CPU baseline unexpectedly executed FDDS GPU contractions")
        if args.arm == "cpu" and exchange_calls:
            raise RuntimeError("CPU baseline unexpectedly executed GPU exchange preprocessing")
        if args.arm == "gpu" and args.require_gpu_exchange and exchange_calls != 4:
            raise RuntimeError("Optimized benchmark requires actual GPU X/Y work on both monomers")
        assert all(np.isfinite(v) for v in energies.values())
        result["ok"] = True
    finally:
        np.linalg.pinv = original
        result.setdefault("fdds_seconds", time.perf_counter() - start)
        result["pinv_calls"] = pinv_calls
        result["pinv_seconds"] = sum(v["seconds"] for v in pinv_calls)
        result["maxrss_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["libraries_sha256"] = libraries()
        atomic_json(args.output / "result.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", type=Path)
    p.add_argument("--system", choices=("water", "peptide", "nanotube", "protein83"), required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--arm", choices=("cpu", "gpu"), default="cpu")
    p.add_argument("--require-gpu-exchange", action="store_true")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--memory", default="96 GiB")
    p.add_argument("--binary-pin", required=True)
    p.add_argument("--expected-core-sha256", required=True)
    p.add_argument("--expected-package", type=Path, required=True)
    args = p.parse_args()
    args.cache, args.output = args.cache.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    os.chdir(args.output)
    sys.argv.append("--einsums:no-attach-debugger")
    import psi4
    sys.argv.pop()
    assert Path(psi4.__file__).resolve().parent == args.expected_package.resolve()
    assert digest(psi4.core.__file__) == args.expected_core_sha256
    psi4.core.set_output_file(str(args.output / "psi4.out"), False)
    psi4.set_num_threads(args.threads)
    psi4.set_memory(args.memory)
    try:
        metadata = prepare(psi4, args) if args.prepare else json.loads((args.cache / "metadata.json").read_text())
        print(json.dumps(replay(psi4, args, metadata), indent=2))
    finally:
        psi4.core.close_outfile()


if __name__ == "__main__":
    main()
