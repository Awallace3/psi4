#!/usr/bin/env python3
"""Focused VV10 correctness probe using an existing, unmodified GPU binary."""
import argparse
import json
from pathlib import Path

import psi4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-package", required=True)
    parser.add_argument("--cpu", action="store_true", help="Validate the probe on a local CPU build")
    args = parser.parse_args()
    assert Path(psi4.__file__).resolve().parent == Path(args.expected_package).resolve()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    psi4.set_num_threads(2)
    psi4.set_memory("2 GiB")
    original_g = psi4.core.RHF.form_G
    original_switch = psi4.core.VBase.set_grid
    events = []

    def snapshot(potential):
        functional = potential.functional()
        return dict(needs_vv10=functional.needs_vv10(), b=functional.vv10_b(),
                    c=functional.vv10_c(), initial=potential.initial_grid(),
                    points=potential.grid().npoints(),
                    quad=dict(potential.quadrature_values()))

    def form_g(wfn):
        original_g(wfn)
        if wfn.V_potential() is not None:
            events.append(dict(event="form_G", **snapshot(wfn.V_potential())))

    def set_grid(potential, name):
        before = snapshot(potential) if potential.grid() is not None else {}
        result = original_switch(potential, name)
        events.append(dict(event="set_grid", name=name, before=before, **snapshot(potential)))
        return result

    def immediate_final(wfn, *a, **kw):
        if wfn.V_potential().initial_grid():
            wfn.V_potential().set_grid("Final")
        return original_iterations(wfn, *a, **kw)

    psi4.core.RHF.form_G = form_g
    psi4.core.VBase.set_grid = set_grid
    original_iterations = psi4.core.HF.iterations
    records = []
    try:
        for postscf in (False, True):
            for name, switching, timing, early in (
                    ("fixed", False, False, False),
                    ("fixed_profiled", False, True, False),
                    ("switched", True, False, False),
                    ("switched_profiled", True, True, False),
                    ("immediate_final", True, False, True)):
                psi4.core.clean()
                psi4.core.clean_options()
                psi4.core.clean_variables()
                events.clear()
                options = dict(
                    basis="sto-3g", scf_type="df", guess="core", reference="rks",
                    use_cuest=not args.cpu, cuest_xc=True, cuest_mixed_precision=False,
                    e_convergence=12, d_convergence=10,
                    dft_radial_points=75, dft_spherical_points=302,
                    dft_vv10_postscf=postscf, cuest_xc_timing=timing,
                    dft_radial_points_initial=35 if switching else 0,
                    dft_spherical_points_initial=110 if switching else 0)
                psi4.set_options(options)
                case = out/f"{name}-post{int(postscf)}"
                case.mkdir()
                psi4.core.set_output_file(str(case/"psi4.out"), False)
                psi4.core.HF.iterations = immediate_final if early else original_iterations
                mol = psi4.geometry("O 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.24\nsymmetry c1")
                record = dict(mode=name, postscf=postscf, options=options, ok=False)
                try:
                    energy, wfn = psi4.energy("b97m-v", molecule=mol, return_wfn=True)
                    record.update(ok=True, energy=energy, final=snapshot(wfn.V_potential()),
                                  density=wfn.Da().to_array().tolist(),
                                  variables={k: float(v) for k, v in wfn.scalar_variables().items()},
                                  effective_options={k: psi4.core.get_option("SCF", k) for k in
                                      ("DFT_GRID_NAME", "DFT_PRUNING_SCHEME", "DFT_RADIAL_POINTS",
                                       "DFT_SPHERICAL_POINTS", "DFT_VV10_RADIAL_POINTS",
                                       "DFT_VV10_SPHERICAL_POINTS")})
                    del wfn
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                record["events"] = list(events)
                (case/"result.json").write_text(json.dumps(record, indent=2) + "\n")
                records.append(record)
                (out/"results.json").write_text(json.dumps(records, indent=2) + "\n")
                print(name, "postscf", postscf, "energy", record.get("energy"),
                      "quad", record.get("final", {}).get("quad"), flush=True)
                psi4.core.close_outfile()
    finally:
        psi4.core.RHF.form_G = original_g
        psi4.core.VBase.set_grid = original_switch
        psi4.core.HF.iterations = original_iterations
        psi4.core.close_outfile()
    return 0 if all(r["ok"] for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
