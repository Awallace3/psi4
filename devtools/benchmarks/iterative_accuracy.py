#!/usr/bin/env python3
"""Explain, rather than widen, the CPU/GPU accuracy gate under ITERATIVE GRAC.

With a fixed GRAC shift the two arms agree to well under 1e-6 Eh. Turning on
ITERATIVE GRAC puts two extra SCF solutions per monomer -- a neutral and a
doublet cation -- inside the timed region, and the shift derived from them
feeds the asymptotic correction of every later SAPT term. A backend difference
in either SCF is therefore amplified into the components, and the campaign's
1e-6 Eh tolerance starts reporting failures.

Raising the tolerance would hide the one thing worth knowing: whether the two
arms disagree because the GPU's arithmetic is less accurate, or because the two
arms converged to genuinely different SCF solutions. Those need opposite
responses. This separates them by checking the neutral and cation energies
independently: arithmetic noise moves both by a comparable small amount, while
a solution-selection difference leaves one converged pair agreeing to near
machine precision and the other offset by orders of magnitude more.

When the arms did pick different solutions, the variationally lower one is the
better answer, and which arm found it is reported rather than assumed.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import statistics

NAME = re.compile(r"^(?P<stem>.+)-(?P<mode>cpu|gpu)-(?P<repeat>\d+)$")
SHIFT = re.compile(r"^\s*GRAC shift Monomer (?P<label>\w+):\s*(?P<shift>-?[\d.]+)")
ENERGIES = re.compile(r"^\s*E_given\s*=\s*(?P<neutral>-?[\d.]+),\s*"
                      r"E_cation\s*=\s*(?P<cation>-?[\d.]+),\s*HOMO\s*=\s*(?P<homo>-?[\d.]+)")
COMPONENTS = [f"SAPT {term} ENERGY" for term in ("ELST", "EXCH", "IND", "DISP", "TOTAL")]
# A converged SCF pair reproduces across backends to roughly this much; anything
# larger is a different solution, not a different rounding of the same one.
SCF_NOISE_EH = 1e-5
# A cross-arm difference is only interpretable this far above the within-arm
# scatter; below it, the two arms are not distinguishable by these data.
SCATTER_MARGIN = 10.0


def grac_scf(psi4_out):
    """The neutral, cation, and HOMO energies behind each monomer's GRAC shift."""
    monomers, pending = {}, None
    for line in Path(psi4_out).read_text().splitlines():
        shift = SHIFT.match(line)
        if shift:
            pending = shift.group("label")
            monomers[pending] = {"shift": float(shift.group("shift"))}
            continue
        found = ENERGIES.match(line)
        if found and pending:
            monomers[pending].update({key: float(found.group(key))
                                      for key in ("neutral", "cation", "homo")})
            pending = None
    return monomers


def arm(directory):
    record = json.loads((directory / "result.json").read_text())
    if not record.get("ok"):
        return None
    return {"components": record["components_hartree"],
            "shifts": record["grac_shifts_hartree"],
            "grac_compute": record.get("grac_compute"),
            "monomers": grac_scf(directory / "psi4.out")}


def collapse(repeats):
    """One representative arm, plus the run-to-run scatter behind it.

    Neither backend is bitwise reproducible: threaded reductions and GPU work
    decomposition both reorder floating-point sums between runs. The scatter is
    tiny, but it sets the floor below which a cross-arm difference means
    nothing, so it is measured rather than assumed away. Medians resist a
    single outlying repeat.
    """
    median = lambda values: statistics.median(values)
    spread = lambda values: max(values) - min(values)
    components = {key: [r["components"][key] for r in repeats] for key in COMPONENTS}
    shifts = {label: [r["shifts"][label] for r in repeats] for label in repeats[0]["shifts"]}
    return {"components": {key: median(values) for key, values in components.items()},
            "shifts": {label: median(values) for label, values in shifts.items()},
            "component_scatter": max((spread(values) for values in components.values()), default=0.0),
            "repeats": len(repeats),
            "grac_compute": repeats[0]["grac_compute"],
            "monomers": repeats[0]["monomers"]}


def compare(cpu, gpu, tolerance):
    """Component deltas plus the SCF evidence for what produced them."""
    scatter = max(cpu["component_scatter"], gpu["component_scatter"])
    deltas = {key: abs(cpu["components"][key] - gpu["components"][key]) for key in COMPONENTS}
    monomers = {}
    for label in sorted(set(cpu["monomers"]) & set(gpu["monomers"])):
        c, g = cpu["monomers"][label], gpu["monomers"][label]
        entry = {key: abs(c[key] - g[key]) for key in ("shift", "neutral", "cation", "homo")}
        # Same solution: every SCF quantity agrees to near machine precision.
        # Different solution: the neutral stays tight while the cation jumps.
        entry["different_cation_solution"] = (entry["cation"] > SCF_NOISE_EH
                                              and entry["neutral"] <= SCF_NOISE_EH)
        entry["lower_cation_arm"] = "cpu" if c["cation"] < g["cation"] else "gpu"
        entry["cpu_cation"], entry["gpu_cation"] = c["cation"], g["cation"]
        monomers[label] = entry
    split = [entry for entry in monomers.values() if entry["different_cation_solution"]]
    worst = max(deltas.values())
    return {"max_component_delta_eh": worst,
            "within_tolerance": worst <= tolerance,
            "component_deltas_eh": deltas,
            "run_to_run_scatter_eh": scatter,
            "repeats": {"cpu": cpu["repeats"], "gpu": gpu["repeats"]},
            "delta_exceeds_scatter": worst > SCATTER_MARGIN * scatter,
            "max_shift_delta_eh": max((e["shift"] for e in monomers.values()), default=0.0),
            "monomers": monomers,
            "solution_selection_differs": bool(split),
            "better_arm": (sorted({e["lower_cation_arm"] for e in split})
                           if split else None),
            "verdict": verdict(worst, tolerance, split, scatter)}


def verdict(worst, tolerance, split, scatter):
    if worst <= tolerance:
        return "agrees within tolerance"
    if worst <= SCATTER_MARGIN * scatter:
        return ("exceeds tolerance but not the run-to-run scatter of either arm; "
                "the arms are not distinguishable by these data")
    if split:
        arms = sorted({e["lower_cation_arm"] for e in split})
        which = arms[0] if len(arms) == 1 else "/".join(arms)
        return (f"exceeds tolerance because the arms converged to different cation SCF "
                f"solutions; {which} found the lower one")
    return "exceeds tolerance with both arms on the same SCF solution: investigate arithmetic"


def analyze(results, tolerance=1e-6):
    arms = defaultdict(lambda: defaultdict(list))
    for path in sorted(Path(results).glob("*/result.json")):
        match = NAME.match(path.parent.name)
        if not match:
            continue
        loaded = arm(path.parent)
        if loaded:
            arms[match.group("stem")][match.group("mode")].append(loaded)
    rows = {}
    for stem, modes in sorted(arms.items()):
        if not (modes.get("cpu") and modes.get("gpu")):
            continue
        cpu, gpu = collapse(modes["cpu"]), collapse(modes["gpu"])
        row = compare(cpu, gpu, tolerance)
        row["grac_compute"] = cpu["grac_compute"]
        rows[stem] = row
    return rows


def markdown(rows, tolerance):
    out = [f"| Case | Max component Δ, Eh | Run-to-run scatter, Eh | Max GRAC shift Δ, Eh | "
           f"Max neutral Δ, Eh | Max cation Δ, Eh | Within {tolerance:g} Eh | Interpretation |",
           "|---|---:|---:|---:|---:|---:|:--:|---|"]
    for stem, row in rows.items():
        neutral = max((e["neutral"] for e in row["monomers"].values()), default=0.0)
        cation = max((e["cation"] for e in row["monomers"].values()), default=0.0)
        out.append(f"| {stem} | {row['max_component_delta_eh']:.2e} | "
                   f"{row['run_to_run_scatter_eh']:.1e} | {row['max_shift_delta_eh']:.2e} | {neutral:.2e} | {cation:.2e} | "
                   f"{'yes' if row['within_tolerance'] else 'no'} | {row['verdict']} |")
    out += ["", "The neutral and cation columns are the monomer SCF energies the GRAC shift is "
                "derived from. Where both agree to near machine precision, the arms solved the "
                "same problem the same way. Where the neutral agrees but the cation does not, "
                "the arms converged to different solutions of a near-degenerate open-shell SCF, "
                "and the component difference that follows is not a measure of GPU arithmetic "
                "error."]
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-6,
                        help="campaign accuracy gate in hartree (default 1e-6)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = analyze(args.results, args.tolerance)
    if not rows:
        raise SystemExit(f"no paired cpu/gpu case under {args.results}")
    print(markdown(rows, args.tolerance))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    unexplained = [stem for stem, row in rows.items()
                   if not row["within_tolerance"] and not row["solution_selection_differs"]
                   and row["delta_exceeds_scatter"]]
    if unexplained:
        print(f"\nUnexplained accuracy failures: {', '.join(unexplained)}")
    return 1 if unexplained else 0


if __name__ == "__main__":
    raise SystemExit(main())
