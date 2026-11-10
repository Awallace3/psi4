#!/usr/bin/env python3
"""JK-only same-geometry memory probes; synthetic coefficients, no XC or energies."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import time

CONFIGS = {
    "control922": ("6-31+g**", "def2-universal-jkfit", 922, 3735),
    "primary634": ("6-31g*", "def2-universal-jkfit", 634, 3735),
    "primary766": ("cc-pvdz", "def2-universal-jkfit", 766, 3735),
    "primary1293": ("aug-cc-pvdz", "def2-universal-jkfit", 1293, 3735),
    "primary1786": ("cc-pvtz", "def2-universal-jkfit", 1786, 3735),
    "aux4762": ("6-31+g**", "aug-cc-pvdz-jkfit", 922, 4762),
}


def structural_plan_gib(nbf, naux):
    """Unscreened structural estimate, explicitly NOT an allocator peak."""
    return 8 * (naux * nbf * (nbf + 1) / 2 + naux**2) / 2**30


def queried_workspaces(text):
    return [dict(allocation=m[1], queried_gib_rounded=float(m[2]))
            for m in re.finditer(
                r"Device memory, (AO pair list|DF int plan|J/K scratch):\s+([0-9.]+) \[GiB\]", text)]


def state(name, metadata, hold):
    directory = os.getenv("GRID_MEMORY_OUTPUT")
    if directory:
        from grid_memory import atomic
        data = dict(phase=f"basis_scan:{name}", monotonic_s=time.monotonic(), **metadata)
        atomic(Path(directory) / "phase.json", data)
        with (Path(directory) / "boundaries.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(event="state", **data)) + "\n")
    # Deliberate diagnostic plateau to observe persistent residency, not task polling.
    time.sleep(hold)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case", choices=CONFIGS, default="control922")
    parser.add_argument("--expected-package", required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--scientific-commit", required=True)
    parser.add_argument("--cpu-smoke", action="store_true", help="Water only; not GPU evidence")
    args = parser.parse_args()
    import numpy as np
    import psi4
    assert Path(psi4.__file__).resolve().parent == Path(args.expected_package).resolve()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    psi4.core.set_output_file(str(out / "psi4.out"), False)
    psi4.set_num_threads(args.threads)
    psi4.set_memory("12 GiB")
    psi4.set_options(dict(scf_type="df", use_cuest=not args.cpu_smoke, cuest_xc=False,
                          cuest_mixed_precision=False, cuest_xc_timing=False, puream=True))
    primary, auxiliary, expected_n, expected_q = CONFIGS[args.case]
    if args.cpu_smoke:
        geometry = "O 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.24\nsymmetry c1"
        primary, auxiliary = "cc-pvdz", "cc-pvdz-jkfit"
    else:
        geometry = json.loads(Path(__file__).with_name("grid_geometries.json").read_text())["protein83"]
    mol = psi4.geometry(geometry)
    basis = psi4.core.BasisSet.build(mol, "ORBITAL", primary, puream=True)
    aux = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", auxiliary, "JKFIT", primary, puream=True)
    n, q = basis.nbf(), aux.nbf()
    if not args.cpu_smoke:
        assert (n, q) == (expected_n, expected_q)
        assert structural_plan_gib(n, q) < 45, "Refusing a larger, unsized case"
    electrons = int(round(sum(mol.Z(i) for i in range(mol.natom())) - mol.molecular_charge()))
    nocc = (electrons + mol.multiplicity() - 1) // 2
    assert 0 < nocc <= n
    meta = dict(nbf=n, naux=q, nocc=nocc, xc_grid_constructed=False)
    result = dict(ok=False, case=args.case, primary=primary, auxiliary=auxiliary, **meta,
        diagnostic_only=True, eligible_for_speedup_bars=False, synthetic_coefficients=True,
        cpu_smoke=args.cpu_smoke, scientific_commit=args.scientific_commit,
        job_id=os.getenv("SLURM_JOB_ID"), package=psi4.__file__,
        structural_plan_gib=structural_plan_gib(n, q), not_an_allocator_peak=True)
    hold = 0.01 if args.cpu_smoke else 1.2
    jk = None
    try:
        state("before_JK", meta, hold)
        jk = psi4.core.JK.build_JK(basis, aux)
        jk.set_do_J(True)
        jk.set_do_K(True)
        jk.set_print(1)
        jk.set_memory(12 * 2**30 // 8)  # Psi4 words, NOT a negotiated device budget.
        jk.set_omega_alpha(1.0)
        jk.set_omega_beta(0.0)
        jk.initialize()
        if not args.cpu_smoke:
            assert jk.name() == "cuESTJK", "CPU fallback is not GPU evidence"
        state("plan_ready", meta, hold)
        coeff = np.random.default_rng(1701).standard_normal((n, nocc)) / math.sqrt(n)
        jk.C_left_add(psi4.core.Matrix.from_array(coeff))
        state("before_compute", meta, hold)
        for _ in range(3):
            state("JK_compute", meta, 0)
            jk.compute()
            assert np.isfinite(np.asarray(jk.J()[0])).all()
            assert np.isfinite(np.asarray(jk.K()[0])).all()
            state("after_compute", meta, hold)
        jk.C_clear()
        jk.finalize()
        state("after_finalize", meta, hold)
        result["ok"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        psi4.core.close_outfile()
        result["queried_workspaces"] = queried_workspaces((out / "psi4.out").read_text())
        if not args.cpu_smoke and result["ok"]:
            if {r["allocation"] for r in result["queried_workspaces"]} != {
                    "AO pair list", "DF int plan", "J/K scratch"}:
                result.update(ok=False, error="Missing library-queried allocation records")
        (out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
