#!/usr/bin/env python3
"""Small CPU/cuEST DF-JK numerical smoke test; NOT SAPT0/dispersion admission."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

NATIVE_SHA = "cb810a15dff2d0c9c0c83cb56336bf80d77a74ace53f6e4d30921726679fc221"
WATER = """0 1
O 0.000000 0.000000 0.000000
H 0.000000 0.000000 0.957200
H 0.926627 0.000000 -0.239987
--
0 1
O 0.000000 0.000000 2.900000
H 0.000000 0.957200 2.900000
H 0.926627 -0.239987 2.900000
units angstrom
symmetry c1
no_reorient
no_com
"""


def worker(args):
    import psi4
    stage = args.stage.resolve(strict=True)
    if not Path(psi4.__file__).resolve().is_relative_to(stage):
        raise RuntimeError(f"Wrong Psi4 Python installation: {psi4.__file__}")
    if not Path(psi4.core.__file__).resolve().is_relative_to(stage):
        raise RuntimeError("Wrong native Psi4 core")
    if (stage / "BUILT_FROM").read_text().strip() != args.pin:
        raise RuntimeError("Wrong native build pin")
    native = args.native.resolve(strict=True)
    if hashlib.sha256(native.read_bytes()).hexdigest() != NATIVE_SHA:
        raise RuntimeError("Wrong cuEST 0.3.0.1 library")
    psi4.set_num_threads(2)
    psi4.set_memory("2 GiB")
    psi4.core.set_output_file(str(args.output / f"{args.arm}.out"), False)
    psi4.set_options({
        "basis": "jun-cc-pvdz", "puream": True,
        "df_basis_scf": "aug-cc-pvdz-jkfit",
        "reference": "rhf", "scf_type": "df", "save_jk": True,
        "e_convergence": 1e-10, "d_convergence": 1e-10,
        "maxiter": 100, "use_cuest": args.arm == "gpu",
        "cuest_mixed_precision": False, "cuest_gridk": False,
        "cuest_directj": False,
    })
    energy, wfn = psi4.energy("hf", molecule=psi4.geometry(WATER), return_wfn=True)
    jk_name = wfn.jk().name()
    if (jk_name == "cuESTJK") != (args.arm == "gpu"):
        raise RuntimeError(f"Unexpected JK backend: {jk_name}")
    mapped = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and "/libcuest.so" in fields[5]:
            mapped.add(str(Path(fields[5]).resolve(strict=True)))
    if mapped != {str(native)}:
        raise RuntimeError(f"Wrong/multiple mapped cuEST libraries: {mapped}")
    result = {
        "status": "PASS_SCF_WORKER_ONLY", "arm": args.arm,
        "energy_hartree": float(energy), "jk": jk_name, "nbf": wfn.basisset().nbf(),
        "pin": args.pin, "core": psi4.core.__file__,
        "native": str(native), "native_sha256": NATIVE_SHA,
    }
    (args.output / f"{args.arm}.json").write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--pin", required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("cpu", "gpu"))
    args = parser.parse_args()
    if args.arm:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    for arm in ("cpu", "gpu"):
        scratch = args.output / f"scratch-{arm}"
        scratch.mkdir()
        env = dict(os.environ, TMPDIR=str(scratch), PSI_SCRATCH=str(scratch),
                   SCRATCH=str(scratch), CUDA_VISIBLE_DEVICES="" if arm == "cpu" else "0")
        subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
                        "--arm", arm], env=env, check=True, timeout=240)
    rows = {arm: json.loads((args.output / f"{arm}.json").read_text()) for arm in ("cpu", "gpu")}
    error = rows["gpu"]["energy_hartree"] - rows["cpu"]["energy_hartree"]
    passed = math.isfinite(error) and abs(error) <= 1e-7
    result = {"status": "PASS_SCF_ONLY" if passed else "FAIL_SCF_PARITY",
              "delta_hartree": error, "absolute_tolerance_hartree": 1e-7,
              "qualified_for_sapt0": False, "workers": rows}
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    if not passed:
        raise RuntimeError("CPU/cuEST SCF energies differ beyond the accepted tolerance")


if __name__ == "__main__":
    main()
