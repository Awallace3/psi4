#!/usr/bin/env python3
"""CPU execution gate for Python-only grid-policy changes; no performance claim."""
import argparse
import json
from pathlib import Path

import psi4
from grid_policy_pilot import install_iterator

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
psi4.core.set_output_file(str(args.output / "psi4.out"), False)
psi4.set_num_threads(2)
psi4.set_memory("1 GiB")


def options(**extra):
    psi4.core.clean_options()
    settings = dict(basis="sto-3g", scf_type="pk", guess="core",
        e_convergence=10, d_convergence=8, maxiter=100,
        dft_radial_points=75, dft_spherical_points=302,
        use_cuest=False, orbital_optimizer_package="internal")
    settings.update(extra)
    psi4.set_options(settings)


rows = []
for name, geometry, reference in (
    ("water", "O 0 0 0\nH 0 0 .96\nH .93 0 -.24\nsymmetry c1", "rks"),
    ("hydrogen", "0 2\nH 0 0 0\nsymmetry c1", "uks"),
):
    mol = psi4.geometry(geometry)
    options(reference=reference)
    expected = psi4.energy("pbe", molecule=mol)
    for factor, minimum in ((10., 1), (1., 2)):
        proofs = []
        overlay = install_iterator(minimum, proofs)
        options(reference=reference, dft_radial_points_initial=35,
                dft_spherical_points_initial=110, dft_grid_switch_factor=factor)
        actual = psi4.energy("pbe", molecule=mol)
        assert abs(actual-expected) < 2.e-9
        assert proofs and all(q["final"] >= minimum for q in proofs if q["converged"])
        rows.append(dict(system=name, factor=factor, minimum=minimum,
                         energy_error_eh=abs(actual-expected), proofs=proofs, overlay_sha256=overlay))

# A one-orbital atom converges immediately on each grid: forcing two final
# evaluations must still work when one final evaluation already meets tolerance.
mol = psi4.geometry("He 0 0 0\nsymmetry c1")
for minimum in (1, 2):
    proofs = []
    install_iterator(minimum, proofs)
    options(reference="rks", e_convergence=3, d_convergence=3,
            dft_radial_points_initial=35, dft_spherical_points_initial=110,
            dft_grid_switch_factor=1.)
    psi4.energy("pbe", molecule=mol)
    assert proofs[-1]["final"] == minimum
    rows.append(dict(system="already-converged-helium", minimum=minimum, proofs=proofs))

install_iterator(100, [])
options(reference="rks", maxiter=8, fail_on_maxiter=False, e_convergence=3, d_convergence=3,
        dft_radial_points_initial=35, dft_spherical_points_initial=110,
        dft_grid_switch_factor=1.)
try:
    psi4.energy("pbe", molecule=mol)
except psi4.driver.p4util.exceptions.SCFConvergenceError:
    rows.append(dict(test="maxiter_cannot_bypass_final_minimum", passed=True))
else:
    raise AssertionError("MAXITER silently bypassed the final-grid minimum")

for minimum in (0, -1, 1.5, True):
    install_iterator(minimum, [])
    options(reference="rks", dft_radial_points_initial=35, dft_spherical_points_initial=110)
    try:
        psi4.energy("pbe", molecule=mol)
    except psi4.driver.p4util.exceptions.ValidationError as error:
        assert "positive integer" in str(error)
    else:
        raise AssertionError(f"Invalid minimum accepted: {minimum}")
rows.append(dict(test="invalid_minima_rejected", passed=True))
psi4.core.close_outfile()
(args.output / "result.json").write_text(json.dumps(dict(ok=True, cases=rows), indent=2)+"\n")
print("PASS: RKS/UKS energy parity; exact 1 vs 2 final iterations; MAXITER guard; invalid minima.")
