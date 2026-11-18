"""Measure a fresh SAPT-D4(I) energy call and retain signed fragment sums and timer paths."""
import argparse
import ctypes
import hashlib
import json
import socket
from pathlib import Path
import time

import numpy as np
import psi4


WATER = """0 1
O -0.702196054 -0.056060256 0.009942262
H -1.022193224 0.846775782 -0.011488714
H 0.257521062 0.042121496 -0.005218999
--
0 1
O 2.268880784 0.026340101 -0.000508029
H 2.645502399 -0.412039965 0.766632411
H 2.641145101 -0.449872874 -0.744894473
units angstrom
symmetry c1
no_reorient
no_com
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, help="Psi4 two-fragment geometry (default: water dimer)")
    parser.add_argument("--field", type=Path, help="JSON external_potentials in Psi4 format")
    parser.add_argument("--functional", required=True, choices=["HF", "PBE0"])
    parser.add_argument("--arm", required=True, choices=["cpu", "gpu"])
    parser.add_argument("--fsapt", action="store_true")
    parser.add_argument("--fragment-integrals", choices=["AUTO", "DF", "CUEST_JK", "CUEST_DFMO"],
                        help="Override the native F-SAPT fragment integral algorithm")
    parser.add_argument("--integral-direct", action="store_true", help="Use the cuEST integral-direct DF plan for SCF and SAPT")
    parser.add_argument("--coupled", action="store_true", help="Request per-source fragment response")
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--libxc", type=Path, help="Required loaded LibXC shared object")
    parser.add_argument("--einsums", action="store_true", help="Require the Einsums scalar backend")
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True, help="Pinned cuEST and CUDA 13.5 prefix")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.coupled and not args.fsapt:
        parser.error("--coupled needs --fsapt")
    if not Path(psi4.core.__file__).resolve().is_relative_to(args.stage.resolve()):
        raise RuntimeError(f"Unexpected Psi4 stage {psi4.core.__file__}")
    args.output.mkdir(parents=True, exist_ok=False)
    psi4.core.set_output_file(str(args.output / "psi4.out"), False)
    psi4.set_memory(args.memory)
    psi4.set_num_threads(args.threads)
    use_gpu = args.arm == "gpu"
    options = dict(basis="cc-pvdz", df_basis_scf="cc-pvdz-jkfit", df_basis_mp2="cc-pvdz-ri",
                   scf_type="df", freeze_core=True, guess="sad", maxiter=500,
                   sapt_dft_functional=args.functional, sapt_dft_do_dhf=True,
                   sapt_dft_grac_compute="ITERATIVE" if args.functional == "PBE0" else "NONE",
                   sapt_dft_induction_type="CPKS", sapt_dft_do_fsapt="FISAPT" if args.fsapt else "NONE",
                   fisapt_fsapt_filepath="none", fisapt_fsapt_ind_response=args.coupled,
                   sapt_dft_use_einsums=False, orbital_optimizer_package="internal",
                   use_cuest=use_gpu, cuest_mixed_precision=True)
    if args.integral_direct:
        if not use_gpu:
            parser.error("--integral-direct needs --arm gpu")
        options["cuest_df_integral_direct"] = True
    if args.fragment_integrals:
        options["fisapt_fsapt_integral_algorithm"] = args.fragment_integrals
    if args.coupled and args.functional == "PBE0":
        options["fisapt_fsapt_ind_xc_response"] = True
    if args.einsums:
        from psi4.driver.procrouting.sapt import sapt_proc
        if not sapt_proc.einsums_available:
            raise RuntimeError("Einsums scalar backend unavailable")
        options["sapt_dft_use_einsums"] = True
    psi4.set_options(options)
    geometry = args.geometry.read_text() if args.geometry else WATER
    kwargs = {"external_potentials": json.loads(args.field.read_text())} if args.field else {}
    mol = psi4.geometry(geometry)
    native = args.runtime.resolve() / "lib/libcuest.so.0.3.0"
    loaded = {Path(line.split()[-1]).resolve() for line in Path("/proc/self/maps").read_text().splitlines()
              if "/libcuest.so" in line or "/libxc.so" in line}
    mapped = {p for p in loaded if p.name.startswith("libcuest.so")}
    libxc = sorted(str(p) for p in loaded if p.name.startswith("libxc.so"))
    if args.libxc and libxc != [str(args.libxc.resolve())]:
        raise RuntimeError(f"Unexpected LibXC runtime {libxc}")
    with native.open("rb") as handle:
        native_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    if mapped != {native} or native_sha256 != (
            "cb810a15dff2d0c9c0c83cb56336bf80d77a74ace53f6e4d30921726679fc221"):
        raise RuntimeError(f"Unexpected cuEST runtime {mapped}")
    cudart = ctypes.CDLL("libcudart.so.13")
    version = ctypes.c_int()
    if cudart.cudaRuntimeGetVersion(ctypes.byref(version)) or version.value != 13050:
        raise RuntimeError(f"Unexpected CUDA runtime {version.value}")
    psi4.core.clean_timers()
    start = time.perf_counter()
    energy, wfn = psi4.energy("sapt(dft)-d4(i)", molecule=mol, return_wfn=True, **kwargs)
    elapsed = time.perf_counter() - start
    psi4.core.flush_outfile()
    with (args.output / "psi4.out").open() as handle:
        device_jk = any("cuESTJK: GPU-Accelerated" in line for line in handle)
    if device_jk != use_gpu:
        raise RuntimeError(f"Requested {args.arm} arm but cuESTJK dispatch was {device_jk}")
    timers = psi4.core.get_timer_records(False)
    terms = {key: float(wfn.variable(f"SAPT {key} ENERGY")) for key in ("ELST", "EXCH", "IND", "DISP")}
    sums = {}
    if args.fsapt:
        matrices = {key: np.asarray(wfn.variable("FSAPT_" + key))
                    for key in ("ELST_AB", "EXCH_AB", "INDAB_AB", "INDBA_AB", "EMPIRICAL_DISP")}
        np.savez(args.output / "fragments.npz", **matrices)
        sums = {key: float(matrix.sum()) for key, matrix in matrices.items()}
    result = dict(energy_hartree=energy, scalar_hartree=terms, fragment_sums_hartree=sums,
                  wall_seconds=elapsed, timers={key: row for key, row in timers.items()
                                               if row["timer_name"].startswith(("SAPT(", "F-SAPT"))},
                  options=options, host=socket.gethostname(), threads=args.threads, memory=args.memory,
                  geometry_sha256=hashlib.sha256(geometry.encode()).hexdigest(),
                  external_potentials=kwargs.get("external_potentials"), nbf=wfn.basisset().nbf(),
                  psi4_core=psi4.core.__file__, libxc=libxc, cuest=str(native), cuest_sha256=native_sha256, device_jk=device_jk, cuda_runtime=version.value)
    (args.output / "result.json").write_text(json.dumps(result, allow_nan=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
