#!/usr/bin/env python3
"""Compare coarse-convergence/final-refinement policies on the same core.

The explicitly hashed Python iterator overlay is the scientific change;
the existing C++ core is reused. Every process runs one policy independently.
"""
import functools
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import tiny_grid_pilot as pilot

POLICIES = dict(early1=(10., 1), full2=(1., 2), full1=(1., 1), early2=(10., 2))
BASES = dict(tiny10=((75, 302), (20, 110)), tiny12=((75, 302), (25, 110)),
             tiny19=((75, 302), (30, 146)), coarse75=((75, 302), (35, 194)),
             tiny08=((75, 302), (20, 86)), tiny05=((75, 302), (15, 74)))
for base, config in BASES.items():
    for policy in POLICIES:
        pilot.CONFIGS[base+"_"+policy] = config


def install_iterator(minimum, proofs):
    import psi4
    source = Path(__file__).resolve().parents[2] / "psi4/driver/procrouting/scf_proc/scf_iterator.py"
    name = "psi4.driver.procrouting.scf_proc._grid_policy_iterator"
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    @functools.wraps(module.scf_iterate)
    def iterate(wfn, *args, **kwargs):
        potential = wfn.V_potential()
        switched = bool(potential and potential.initial_grid())
        success = False
        try:
            result = module.scf_iterate(wfn, *args, xc_final_min_iterations=minimum, **kwargs)
            success = True
            if switched:
                assert not potential.initial_grid()
                assert wfn.variable("SCF DFT FINAL GRID ITERATIONS") >= minimum
                assert wfn._xc_grid_ready_for_finalize
            return result
        finally:
            if switched:
                proof = dict(converged=success, minimum_final_iterations=minimum)
                for field, variable in (
                    ("initial", "SCF DFT INITIAL GRID ITERATIONS"),
                    ("final", "SCF DFT FINAL GRID ITERATIONS")):
                    proof[field] = int(wfn.variable(variable)) if wfn.has_variable(variable) else 0
                proofs.append(proof)

    psi4.core.HF.iterations = iterate
    return hashlib.sha256(source.read_bytes()).hexdigest()


def worker(args):
    policy = args.arm.rsplit("_", 1)[1] if "_" in args.arm else "early1"
    factor, minimum = POLICIES[policy]
    args.switch_factor = factor
    proofs = []
    overlay = install_iterator(minimum, proofs)
    rc = BASE_WORKER(args)
    path = args.output / "result.json"
    row = json.loads(path.read_text())
    row.update(grid_policy=dict(name=policy, switch_factor=factor,
                                minimum_final_iterations=minimum),
               iterator_overlay_sha256=overlay,
               iterator_overlay_commit="recorded in campaign harness_commit",
               per_scf_grid_iteration_proofs=proofs)
    if row.get("ok") and any(proof["converged"] and proof["final"] < minimum for proof in proofs):
        row.update(ok=False, error="A converged SCF violated the requested final-grid minimum")
        rc = 1
    pilot.grid.atomic(path, row)
    return rc


BASE_WORKER = pilot.worker
if __name__ == "__main__":
    pilot.worker = worker
    raise SystemExit(pilot.main(worker_script=__file__))
