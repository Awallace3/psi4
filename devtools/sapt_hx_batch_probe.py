#!/usr/bin/env python
"""Does one J/K call for both SAPT monomers' CPKS vectors cost less than two?

    python sapt_hx_batch_probe.py --suite GEOMETRIES_DIR --system protein83 --output OUT.json \
        [--basis aug-cc-pvdz] [--repeat 3] [--cuest]

SAPT(DFT) CPKS calls monomer A's and monomer B's Hessian-vector products separately, and
both run J/K on the same dimer-basis JK object. This times that J/K with the induction
shapes: C_left = Cocc (nbf x nocc_X), C_right = Cvir x^T (nbf x nocc_X), for X = A, B,
once per monomer and once batched. The orbitals come from one dimer SCF, split by the
monomers' occupied counts; J/K cost depends on the factor shapes, not on which orbitals.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--system", required=True)
    p.add_argument("--basis", default="aug-cc-pvdz")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--memory", default="100 GiB")
    p.add_argument("--cuest", action="store_true")
    args = p.parse_args()

    sys.argv.append("--einsums:no-attach-debugger")
    import psi4
    sys.argv.pop()
    sys.path.insert(0, str(args.suite))
    from systems import SYSTEMS
    assert args.system in SYSTEMS and args.system != "pla15_3pe2"

    psi4.set_memory(args.memory)
    psi4.set_num_threads(args.threads)
    psi4.core.set_output_file(str(args.output.with_suffix(".out")), False)
    molecule = psi4.geometry(SYSTEMS[args.system])
    psi4.set_options({"basis": args.basis, "scf_type": "df", "save_jk": True,
                      "use_cuest": args.cuest, "cuest_xc": args.cuest, "cuest_sad": args.cuest,
                      "cuest_mixed_precision": False})
    nocc = []
    for frag in (1, 2):
        sub = molecule.extract_subsets(frag)
        sub.update_geometry()
        nocc.append(int(round(sum(sub.Z(i) for i in range(sub.natom())) - sub.molecular_charge())) // 2)

    t0 = time.perf_counter()
    _, wfn = psi4.energy("hf", molecule=molecule, return_wfn=True)
    scf_s = time.perf_counter() - t0
    jk = wfn.jk()
    Cocc = wfn.Ca_subset("AO", "OCC").np
    Cvir = wfn.Ca_subset("AO", "VIR").np
    rng = np.random.default_rng(3)
    factors = []
    for n in nocc:
        x = rng.normal(size=(n, Cvir.shape[1])) * 1e-2
        factors.append((psi4.core.Matrix.from_array(np.ascontiguousarray(Cocc[:, :n])),
                        psi4.core.Matrix.from_array(np.ascontiguousarray(-Cvir @ x.T))))

    def timed(pairs):
        jk.C_clear()
        for left, right in pairs:
            jk.C_left_add(left)
            jk.C_right_add(right)
        t = time.perf_counter()
        jk.compute()
        return time.perf_counter() - t, [m.np.copy() for m in jk.J()], [m.np.copy() for m in jk.K()]

    timed(factors)  # warm-up
    rows = {"A": [], "B": [], "A+B separate": [], "A+B batched": []}
    max_diff = 0.0
    for _ in range(args.repeat):
        tA, JA, KA = timed(factors[:1])
        tB, JB, KB = timed(factors[1:])
        tAB, JAB, KAB = timed(factors)
        rows["A"].append(tA)
        rows["B"].append(tB)
        rows["A+B separate"].append(tA + tB)
        rows["A+B batched"].append(tAB)
        for a, b in zip(JA + JB + KA + KB, [JAB[0], JAB[1], KAB[0], KAB[1]]):
            max_diff = max(max_diff, float(np.abs(a - b).max()))
    summary = {k: {"min_s": min(v), "all_s": v} for k, v in rows.items()}
    record = {"system": args.system, "basis": args.basis, "nbf": wfn.basisset().nbf(),
              "nocc_monomers": nocc, "cuest": args.cuest, "threads": args.threads,
              "jk_name": jk.name(), "dimer_scf_s": scf_s, "timings": summary,
              "batched_vs_separate_max_abs_diff": max_diff,
              "saving_per_iteration_s": summary["A+B separate"]["min_s"] - summary["A+B batched"]["min_s"]}
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("system", "nbf", "nocc_monomers", "jk_name",
                                              "batched_vs_separate_max_abs_diff", "saving_per_iteration_s")}))
    for k, v in summary.items():
        print(f"  {k:14s} {v['min_s']:8.3f} s")


if __name__ == "__main__":
    main()
