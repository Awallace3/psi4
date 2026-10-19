#!/usr/bin/env python3
"""Export diagnostic XC microtimers for a small GRAC/PBE0 SCF.

GPU synchronization and CPU timer locks intentionally perturb execution.
Timer paths are inclusive; parallel entries report summed thread time.
Do not add these entries together or use this run for speedup claims.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--use-cuest", action="store_true")
    parser.add_argument("--fixed", action="store_true")
    args = parser.parse_args()

    import psi4

    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    psi4.core.set_output_file(str(out / "psi4.out"), False)
    psi4.set_num_threads(args.threads)
    psi4.set_memory("1 GiB")
    options = dict(
        basis="cc-pvdz", scf_type="df", reference="rks",
        use_cuest=args.use_cuest, cuest_xc=True, cuest_mixed_precision=False,
        cuest_xc_timing=True, dft_grac_shift=0.1,
        dft_radial_points=99, dft_spherical_points=590,
        dft_pruning_scheme="none", e_convergence=10, d_convergence=8,
        orbital_optimizer_package="internal",
    )
    if not args.fixed:
        options.update(dft_radial_points_initial=50, dft_spherical_points_initial=302)
    psi4.set_options(options)
    mol = psi4.geometry("O 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.24\nsymmetry c1")
    psi4.core.clean_timers()
    energy, wfn = psi4.energy("pbe0", molecule=mol, return_wfn=True)
    assert not wfn.V_potential().initial_grid()
    counters = {key: value for key, value in wfn.variables().items()
                if key.startswith("SCF DFT")}
    wfn.V_potential().finalize()  # include plan/workspace destruction
    timers = psi4.core.get_timer_dict()
    micro = {key: value for key, value in timers.items() if "XC micro:" in key}
    assert micro, "Profiling produced no XC microtimers"
    if args.use_cuest:
        assert any("cuestXCPotentialCompute" in key for key in micro)
    psi4.core.close_outfile()
    (out / "timers.json").write_text(json.dumps(dict(
        package=psi4.__file__, version=psi4.__version__, options=options,
        energy_eh=energy, counters=counters, timers=timers,
        microtimers=micro,
        caveat="Inclusive paths; thread sums and synchronized diagnostic timings are not additive wall time.",
    ), indent=2) + "\n")


if __name__ == "__main__":
    main()
