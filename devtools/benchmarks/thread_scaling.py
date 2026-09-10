#!/usr/bin/env python3
"""Measure CPU thread scaling and bound the extrapolation to unavailable widths.

Vendor GPU speedups are quoted against a wide CPU socket, so comparing them to
a narrow baseline overstates the GPU. Correcting for that needs a scaling
factor, and assuming a linear one is exactly the error being corrected. This
reads paired same-node runs at two thread counts and reports the measured
factor, then fits Amdahl's law to state what a wider baseline would plausibly
give and what the serial fraction caps it at.

A two-point Amdahl fit has no residual and cannot be validated by its own
inputs, so every extrapolated value is reported alongside the asymptote. Treat
them as an upper bound on the correction, not a measurement: they assume the
serial fraction is constant with width, which memory bandwidth contention makes
optimistic.

By default this scales total `energy()` wall time. `--timer` instead scales one
Psi4 flat timer, which is what a vendor DF-K claim actually needs: a kernel
speedup quoted against an N-core CPU run of the same kernel must be normalized by
how that kernel scales, not by how the enclosing method scales. The two differ
substantially here, so the choice is explicit rather than defaulted.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import statistics

NAME = re.compile(r"^(?P<system>.+?)-(?P<basis>.+)-cpu(?P<threads>\d+)-(?P<repeat>\d+)$")
TIMER_LINE = re.compile(r"^(.*?)\s*:\s*([\d.]+)u\s+([\d.]+)s\s+([\d.]+)w\s+(\d+) calls")


def timer_wall(path, name):
    """Wall seconds for one flat Psi4 timer, or None if it is absent.

    Only the flat section counts. The call-tree section below it repeats the same
    timer names indented under `|`, and summing those would double-count.
    """
    for line in Path(path).read_text().splitlines():
        if "|" in line:
            continue
        match = TIMER_LINE.match(line)
        if match and match.group(1).strip() == name:
            return float(match.group(4))
    return None


def load(results, timer=None):
    """Group `<system>-<basis>-cpuN-<repeat>` case directories by case and width.

    With `timer`, the measured quantity is that flat timer's wall time rather than
    the whole `energy()` call. A run whose timer.dat lacks the timer is dropped
    with a name, never silently treated as zero.
    """
    cases = defaultdict(lambda: defaultdict(list))
    for path in sorted(Path(results).glob("*/result.json")):
        match = NAME.match(path.parent.name)
        if not match:
            continue
        record = json.loads(path.read_text())
        if not record.get("ok"):
            continue
        threads = int(match.group("threads"))
        if record.get("threads") != threads:
            raise ValueError(f"{path.parent.name}: directory says {threads} threads, "
                             f"result.json says {record.get('threads')}")
        if record.get("mode") != "cpu":
            raise ValueError(f"{path.parent.name}: not a CPU run")
        if timer is None:
            measured = record["wall_s"]
        else:
            measured = timer_wall(path.parent / "timer.dat", timer)
            if measured is None:
                raise ValueError(f"{path.parent.name}: timer {timer!r} not in timer.dat")
        cases[(record["system"], record["basis"])][threads].append(measured)
    return cases


def amdahl(narrow_threads, narrow_s, wide_threads, wide_s):
    """Solve T(n) = serial + parallel/n through two measured points.

    Returns None when the two points imply no parallel speedup at all, since a
    nonpositive parallel part makes every extrapolation meaningless.
    """
    if wide_threads == narrow_threads:
        return None
    inverse = 1.0 / narrow_threads - 1.0 / wide_threads
    parallel = (narrow_s - wide_s) / inverse
    serial = narrow_s - parallel / narrow_threads
    if parallel <= 0 or serial < 0:
        return None
    return serial, parallel


def analyze(results, target_threads=56, timer=None):
    cases = load(results, timer)
    rows = []
    for (system, basis), widths in sorted(cases.items()):
        if len(widths) < 2:
            continue
        narrow, wide = min(widths), max(widths)
        narrow_s = statistics.median(widths[narrow])
        wide_s = statistics.median(widths[wide])
        row = {"system": system, "basis": basis,
               "narrow_threads": narrow, "wide_threads": wide,
               "narrow_median_s": narrow_s, "wide_median_s": wide_s,
               "repeats": {str(k): len(v) for k, v in sorted(widths.items())},
               "measured_speedup": narrow_s / wide_s,
               "parallel_efficiency": (narrow_s / wide_s) / (wide / narrow)}
        fit = amdahl(narrow, narrow_s, wide, wide_s)
        if fit:
            serial, parallel = fit
            projected = serial + parallel / target_threads
            row.update({"amdahl_serial_s": serial, "amdahl_parallel_s": parallel,
                        "serial_fraction_at_narrow": serial / narrow_s,
                        "target_threads": target_threads,
                        "projected_target_s": projected,
                        "projected_speedup_vs_narrow": narrow_s / projected,
                        "asymptotic_speedup_vs_narrow": narrow_s / serial})
        rows.append(row)
    return {"target_threads": target_threads,
            "measured_quantity": timer or "energy() wall time",
            "extrapolation_caveat": "Two-point Amdahl fit: no residual, assumes a "
                                    "width-independent serial fraction. Upper bound, not a "
                                    "measurement.",
            "rows": rows}


def markdown(summary):
    target = summary["target_threads"]
    text = [f"Scaling of **{summary['measured_quantity']}**.", "",
            "| System | Basis | Narrow | Wide | Narrow median, s | Wide median, s | Measured speedup | Parallel eff. | "
            f"Projected {target}T, s | Projected speedup | Asymptote |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary["rows"]:
        projected = (f"{row['projected_target_s']:.1f} | {row['projected_speedup_vs_narrow']:.2f}× | "
                     f"{row['asymptotic_speedup_vs_narrow']:.2f}×"
                     if "projected_target_s" in row else "— | — | —")
        text.append(f"| {row['system']} | {row['basis']} | {row['narrow_threads']}T | {row['wide_threads']}T | "
                    f"{row['narrow_median_s']:.2f} | {row['wide_median_s']:.2f} | "
                    f"{row['measured_speedup']:.2f}× | {row['parallel_efficiency'] * 100:.0f}% | {projected} |")
    text += ["", f"Projected columns are a two-point Amdahl fit evaluated at {target} threads. "
                 "The fit passes exactly through both measurements, so it has no residual and its "
                 "accuracy cannot be judged from these data. It assumes the serial fraction does not "
                 "grow with width, which memory bandwidth contention makes optimistic, so read the "
                 "projection as an upper bound on the baseline correction and the asymptote as the "
                 "ceiling no thread count can beat."]
    return "\n".join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--target-threads", type=int, default=56,
                        help="width to extrapolate to (default 56, NVIDIA's quoted CPU baseline)")
    parser.add_argument("--timer", metavar="NAME",
                        help="scale this Psi4 flat timer (e.g. 'JK: JK') instead of total "
                             "energy() wall time")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = analyze(args.results, args.target_threads, args.timer)
    if not summary["rows"]:
        raise SystemExit(f"no case had two thread widths under {args.results}")
    report = markdown(summary)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
