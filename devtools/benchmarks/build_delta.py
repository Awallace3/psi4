#!/usr/bin/env python3
"""Did this build change anything, and is the change bigger than the noise?

`summarize_saptdft_cuest.py` reduces one job tree to per-case medians. Two such
summaries answer "what did the build do" only if the two jobs differ by the
build and nothing else, so this tool is deliberately narrow: it takes exactly
two summaries, pairs them case by case, and reports the change in wall time,
speedup, and memory.

The reason it exists is that the obvious comparison is usually the wrong one.
The 2026-09-19 memory campaign was first read against the 2026-09-10 campaign,
which is the same protocol on the same partition -- but 55 commits separate the
two builds, including work on the GPU XC path, so the GPU arm getting faster
says nothing about the memory merge. The comparison that means something is
against the merge's own first parent, built from the same worktree with the same
harness. This tool does not know which of those it has been handed, so it prints
its two labels prominently and leaves the attribution claim to the caller.

Every change is judged against the run-to-run scatter in the two summaries it
came from, taken as the half-range of each arm's repeats. A difference inside
that band is reported as `~` rather than as a number with a sign, because three
repeats on a shared cluster do not resolve a two-percent effect and a report
that says they do is worse than one that says nothing.
"""
import argparse
from pathlib import Path
import json

# Medians move a little between allocations even when nothing changed. A change
# is called only when it clears the scatter of both summaries combined, so the
# band widens automatically for the noisy short cases rather than by a constant
# anyone has to tune.
def band(stats):
    """Half-range of one metric's repeats, or None when it is not resolvable."""
    if not isinstance(stats, dict):
        return None
    lo, hi = stats.get("min"), stats.get("max")
    if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
        return None
    return abs(hi - lo) / 2.0


def change(control, treatment):
    """Control median, treatment median, ratio, and whether it cleared the noise."""
    c_med = (control or {}).get("median")
    t_med = (treatment or {}).get("median")
    if not isinstance(c_med, (int, float)) or not isinstance(t_med, (int, float)):
        return None
    c_band, t_band = band(control), band(treatment)
    noise = (c_band or 0.0) + (t_band or 0.0)
    return {"control": c_med, "treatment": t_med,
            "delta": t_med - c_med,
            "ratio": (t_med / c_med) if c_med else None,
            "noise_band": noise,
            "resolved": abs(t_med - c_med) > noise,
            "scatter_measured": c_band is not None and t_band is not None}


def host_peak(row, arm):
    return (((row.get("memory") or {}).get("host_peak_rss_mib") or {})
            .get(arm) or {}).get("peak_rss_mib")


def device_peak(row):
    return (row.get("memory") or {}).get("device_peak_mib")


def key(row):
    return (row.get("system"), row.get("basis"))


def compare(control, treatment):
    """Pair two summaries case by case. Cases present in only one are named."""
    c_rows = {key(r): r for r in control.get("rows", [])}
    t_rows = {key(r): r for r in treatment.get("rows", [])}
    shared = [k for k in c_rows if k in t_rows]
    shared.sort(key=lambda k: (c_rows[k].get("nbf") or 0, k))
    rows, numeric_drift = [], []
    for k in shared:
        c, t = c_rows[k], t_rows[k]
        entry = {"system": k[0], "basis": k[1], "nbf": t.get("nbf"),
                 "nbf_agrees": c.get("nbf") == t.get("nbf"),
                 "cpu_wall_s": change((c.get("wall_s") or {}).get("cpu"),
                                      (t.get("wall_s") or {}).get("cpu")),
                 "gpu_wall_s": change((c.get("wall_s") or {}).get("gpu"),
                                      (t.get("wall_s") or {}).get("gpu")),
                 "cpu_host_peak_mib": change(host_peak(c, "cpu"), host_peak(t, "cpu")),
                 "gpu_host_peak_mib": change(host_peak(c, "gpu"), host_peak(t, "gpu")),
                 "gpu_device_peak_mib": change(device_peak(c), device_peak(t)),
                 "speedup": {"control": c.get("speedup"), "treatment": t.get("speedup")},
                 "max_abs_delta_hartree": {"control": c.get("max_abs_delta_hartree"),
                                           "treatment": t.get("max_abs_delta_hartree")}}
        # A build that was supposed to touch only memory must reproduce the
        # CPU-vs-GPU disagreement exactly. A change here is the headline, not a
        # footnote, so it is surfaced separately from the timing table.
        cd, td = (entry["max_abs_delta_hartree"]["control"],
                  entry["max_abs_delta_hartree"]["treatment"])
        if isinstance(cd, (int, float)) and isinstance(td, (int, float)) and cd != td:
            numeric_drift.append({"system": k[0], "basis": k[1],
                                  "control": cd, "treatment": td})
        rows.append(entry)
    return {"rows": rows,
            "control_only": sorted(f"{s}:{b}" for s, b in set(c_rows) - set(t_rows)),
            "treatment_only": sorted(f"{s}:{b}" for s, b in set(t_rows) - set(c_rows)),
            "numeric_drift": numeric_drift,
            "identical_numerics": not numeric_drift}


def cell(field, unit=""):
    """A change as `control -> treatment (ratio)`, or `~` inside the noise."""
    if not field:
        return "—"
    base = f"{field['control']:.2f}{unit} → {field['treatment']:.2f}{unit}"
    if not field["resolved"]:
        return f"{base} (~)"
    return f"{base} ({field['ratio']:.2f}×)"


def markdown(payload, control_label, treatment_label):
    out = [f"Control: `{control_label}` → Treatment: `{treatment_label}`. "
           "A `~` marks a difference inside the combined run-to-run scatter of "
           "the two jobs, which is not a measured change.", "",
           "| System | Basis | nbf | CPU wall s | GPU wall s | Speedup | "
           "CPU host peak MiB | GPU host peak MiB | GPU device peak MiB |",
           "|---|---|---:|---|---|---|---|---|---|"]
    for row in payload["rows"]:
        sp = row["speedup"]
        speed = "—"
        if isinstance(sp["control"], (int, float)) and isinstance(sp["treatment"], (int, float)):
            speed = f"{sp['control']:.2f}× → {sp['treatment']:.2f}×"
        out.append(
            f"| {row['system']} | {row['basis']} | {row['nbf']} | "
            f"{cell(row['cpu_wall_s'])} | {cell(row['gpu_wall_s'])} | {speed} | "
            f"{cell(row['cpu_host_peak_mib'])} | {cell(row['gpu_host_peak_mib'])} | "
            f"{cell(row['gpu_device_peak_mib'])} |")
    if payload["identical_numerics"]:
        out += ["", "Every case reproduced its CPU-vs-GPU energy difference exactly, "
                    "so the two builds are numerically identical on this suite."]
    else:
        out += ["", "**The two builds do not agree numerically.** A build intended to "
                    "change only memory must reproduce these exactly:", ""]
        out += [f"- {d['system']}/{d['basis']}: {d['control']:.6e} → {d['treatment']:.6e} Eh"
                for d in payload["numeric_drift"]]
    for name, missing in (("control", payload["treatment_only"]),
                          ("treatment", payload["control_only"])):
        if missing:
            out += ["", f"Absent from the {name}, so not compared: " +
                    ", ".join(f"`{m}`" for m in missing)]
    mismatched = [r for r in payload["rows"] if not r["nbf_agrees"]]
    if mismatched:
        out += ["", "**Basis dimension differs between the two jobs** — these rows are "
                    "not the same calculation:", ""]
        out += [f"- {r['system']}/{r['basis']}" for r in mismatched]
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("control", type=Path, help="summary.json of the baseline build")
    parser.add_argument("treatment", type=Path, help="summary.json of the build under test")
    parser.add_argument("--control-label", default=None)
    parser.add_argument("--treatment-label", default=None)
    parser.add_argument("--output", type=Path, help="write JSON here")
    parser.add_argument("--require-identical-numerics", action="store_true",
                        help="exit nonzero if any case's CPU-vs-GPU delta moved")
    args = parser.parse_args()
    control = json.loads(args.control.read_text())
    treatment = json.loads(args.treatment.read_text())
    payload = compare(control, treatment)
    c_label = args.control_label or str(args.control)
    t_label = args.treatment_label or str(args.treatment)
    payload["control_label"], payload["treatment_label"] = c_label, t_label
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(markdown(payload, c_label, t_label))
    if args.require_identical_numerics and not payload["identical_numerics"]:
        raise SystemExit("the two builds disagree numerically")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
