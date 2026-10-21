#!/usr/bin/env python3
"""Fresh-process CPU/GPU SAPT(DFT) grid accuracy sweep (not a speedup assertion).

Run with the newly built Psi4 on PYTHONPATH:
    python devtools/saptdft_grid_sweep.py --output /tmp/grid-sweep --threads 2

Each reduced final grid has its own fixed-grid control, separating switching
error from quadrature error. Add --iterative-grac to test neutral/cation shifts.
--use-cuest is reserved for an appropriately built GPU allocation.
The default screens electrostatics/exchange with induction/dispersion disabled.
Use --full-sapt for CPKS induction and dispersion before drawing conclusions
about a complete interaction energy. --compact tests four representative grids.
"""

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time

SYSTEMS = {
    "water": """
    0 1
    O 0 0 0
    H 0 0 0.96
    H 0.93 0 -0.24
    --
    0 1
    O 0 0 3.0
    H 0 0 3.96
    H 0.93 0 2.76
    symmetry c1
    """,
    "hf": """
    0 1
    F 0 0 0
    H 0 0 0.92
    --
    0 1
    F 0 0 2.8
    H 0 0 3.72
    symmetry c1
    """,
}
COMPONENTS = ["SAPT ELST ENERGY", "SAPT EXCH ENERGY", "SAPT IND ENERGY", "SAPT TOTAL ENERGY"]
CONFIGS = [
    ((99, 590), None), ((99, 590), (65, 302)), ((99, 590), (50, 302)),
    ((75, 302), None), ((75, 302), (35, 194)),
    ((50, 194), None), ((50, 194), (25, 110)),
    ((25, 110), None), ((25, 110), (15, 50)),
]


def worker(args):
    import psi4

    config = json.loads(args.worker)
    final, initial = config["final"], config["initial"]
    output = pathlib.Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    psi4.core.set_output_file(str(output / "psi4.out"), False)
    psi4.set_num_threads(args.threads)
    psi4.set_memory("1 GiB")
    options = {
        "basis": args.basis, "scf_type": "df", "guess": "sad",
        "use_cuest": args.use_cuest, "cuest_mixed_precision": False,
        "orbital_optimizer_package": "internal",
        "dft_radial_points": final[0], "dft_spherical_points": final[1],
        "dft_pruning_scheme": "none",
        "e_convergence": 10, "d_convergence": 8, "maxiter": 150,
        "sapt_dft_functional": "pbe0",
        "sapt_dft_induction_type": "cpks" if args.full_sapt else "none",
        "sapt_dft_do_disp": args.full_sapt, "sapt_dft_do_dhf": True,
        "sapt_dft_grac_compute": "iterative" if args.iterative_grac else "none",
    }
    if not args.iterative_grac:
        options.update(sapt_dft_grac_shift_a=0.1, sapt_dft_grac_shift_b=0.1)
    if initial:
        options.update(dft_radial_points_initial=initial[0], dft_spherical_points_initial=initial[1])
    psi4.set_options(options)
    mol = psi4.geometry(SYSTEMS[args.system])
    started = time.perf_counter()
    result = {
        **config, "system": args.system, "basis": args.basis,
        "options": options, "psi4_version": psi4.__version__,
        "threads": args.threads, "backend": "cuest" if args.use_cuest else "cpu",
        "full_sapt": args.full_sapt,
    }
    try:
        energy = psi4.energy("sapt(dft)", molecule=mol)
        result.update(status="ok", energy=energy,
                      components={key: psi4.variable(key) for key in
                                  COMPONENTS + (["SAPT DISP ENERGY"] if args.full_sapt else [])},
                      grac_variables={key: value for key, value in psi4.core.variables().items()
                                      if "GRAC" in key})
    except Exception as exc:
        result.update(status="failed", error=str(exc))
    result["wall_s"] = time.perf_counter() - started
    psi4.core.close_outfile()
    text = (output / "psi4.out").read_text()
    result["initial_grid_iterations"] = len(re.findall(r"@\S+ iter .*XC-INITIAL", text))
    result["final_grid_iterations"] = len(re.findall(r"@\S+ iter .*XC-FINAL", text))
    result["nominal_initial_fraction"] = (
        initial[0] * initial[1] / (final[0] * final[1]) if initial else None)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def worker_result(completed, directory, final, initial, use_cuest):
    """A scientific result cannot override a failed runtime-admission worker."""
    def failed(message):
        return {"status": "failed", "final": final, "initial": initial,
                "error": message, "returncode": completed.returncode}
    if completed.returncode != 0:
        return failed(f"Worker exited {completed.returncode}: {completed.stderr[-4000:]}")
    if not (directory / "result.json").exists():
        return failed("Worker returned no result.json")
    if use_cuest:
        try:
            receipt = json.loads((directory / "runtime-admission.json").read_text())
            expected = json.loads(pathlib.Path(os.environ["CUEST_CORE_IDENTITY"]).read_text())
            version = receipt.get("native_version")
            if (receipt.get("status") != "passed" or type(receipt.get("payload_exit")) is not int
                    or receipt.get("payload_exit") != 0
                    or type(version) is not list or version != [0, 2, 50]
                    or any(type(part) is not int for part in version)
                    or not expected or receipt.get("before") != expected or receipt.get("after") != expected):
                return failed("cuEST runtime-admission receipt did not pass")
        except (OSError, ValueError, TypeError, KeyError):
            return failed("cuEST runtime-admission receipt is absent or invalid")
    return json.loads((directory / "result.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/tmp/saptdft-grid-sweep")
    parser.add_argument("--system", choices=SYSTEMS, default="water")
    parser.add_argument("--basis", default="cc-pvdz")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--case-timeout", type=float, default=300.0,
                        help="Maximum seconds per fresh-process calculation.")
    parser.add_argument("--iterative-grac", action="store_true")
    parser.add_argument("--full-sapt", action="store_true")
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--use-cuest", action="store_true")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    root = pathlib.Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    controls = {}
    configs = [CONFIGS[i] for i in (0, 2, 3, 4)] if args.compact else CONFIGS
    for final, initial in configs:
        label = f"{final[0]}x{final[1]}-" + (f"{initial[0]}x{initial[1]}" if initial else "fixed")
        directory = root / label
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "result.json").unlink(missing_ok=True)  # do not reuse stale results after a failed rerun
        command = [sys.executable, str(pathlib.Path(__file__).resolve()),
                   "--output", str(directory), "--system", args.system,
                   "--basis", args.basis, "--threads", str(args.threads),
                   "--worker", json.dumps({"final": final, "initial": initial})]
        if args.iterative_grac:
            command.append("--iterative-grac")
        if args.use_cuest:
            command.append("--use-cuest")
        if args.full_sapt:
            command.append("--full-sapt")
        if args.use_cuest:
            # The outer sweep does not create a handle. Admission/version proof
            # belongs in each actual scientific worker, not just its controller.
            if not os.environ.get("CUEST_CORE_IDENTITY") or not os.environ.get("CUEST_CONTRACT"):
                raise RuntimeError("GPU sweep requires the canonical cuEST runtime identity")
            command = [sys.executable, str(pathlib.Path(__file__).with_name("cuest_run_admitted.py")),
                       "--receipt", str(directory / "runtime-admission.json"),
                       "--script", str(pathlib.Path(__file__).resolve()), "--", *command[2:]]
        try:
            completed = subprocess.run(command, cwd=directory, capture_output=True, text=True,
                                       timeout=args.case_timeout)
        except subprocess.TimeoutExpired:
            row = {"status": "failed", "final": final, "initial": initial,
                   "error": f"Case exceeded {args.case_timeout} seconds"}
            completed = None
        if completed is None:
            pass
        else:
            row = worker_result(completed, directory, final, initial, args.use_cuest)
        if row["status"] == "ok":
            if initial is None:
                controls[tuple(final)] = row
            for name, ref in [("switch_error", controls.get(tuple(final))),
                              ("quadrature_error", controls.get((99, 590)))]:
                if ref:
                    row[name + "_eh"] = {key: abs(value - ref["components"][key])
                                        for key, value in row["components"].items()}
                    row[name + "_max_eh"] = max(row[name + "_eh"].values())
                    row[name + "_within_1e6_eh"] = row[name + "_max_eh"] <= 1.e-6
                    row[name + "_grac_error_eh"] = {
                        key: abs(value - ref["grac_variables"][key])
                        for key, value in row["grac_variables"].items()
                        if key in ref["grac_variables"]}
        rows.append(row)
        (root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(label, row["status"], f'{row.get("wall_s", 0):.3f}s',
              f'switch={row.get("switch_error_max_eh", float("nan")):.3e} Eh',
              f'grid={row.get("quadrature_error_max_eh", float("nan")):.3e} Eh', flush=True)
    if any(row["status"] != "ok" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
