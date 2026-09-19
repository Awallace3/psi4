#!/usr/bin/env python3
"""Summarize sampled GPU/CPU activity and non-double-counted Psi4 flat timers."""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics

TIMER_LINE = re.compile(r"^(.*?)\s*:\s*([\d.]+)u\s+([\d.]+)s\s+([\d.]+)w\s+(\d+) calls")


def flat_timers(path):
    timers, blocks, flat, counted = {}, 0, False, False
    for line in Path(path).read_text().splitlines():
        stripped = line.strip()
        if stripped and set(stripped) == {"*"}:
            flat, counted = False, False
        elif re.match(r"^Module\s+User\s+System\s+Wall\s+Calls\s*$", stripped):
            flat, counted = True, False
        elif flat and len(stripped) > 20 and set(stripped) == {"-"}:
            flat = False
        elif flat:
            match = TIMER_LINE.match(line)
            if match and not match[1].strip().startswith("|"):
                if not counted:
                    blocks, counted = blocks + 1, True
                name = match[1].strip()
                entry = timers.setdefault(name, {"wall_s": 0.0, "calls": 0})
                entry["wall_s"] += float(match[4])
                entry["calls"] += int(match[5])
    if blocks != 1:
        raise ValueError(f"Expected one nonempty timer block, found {blocks}: {path}")
    return timers


def numeric(value):
    try:
        return float(value.strip().split()[0])
    except (ValueError, IndexError):
        return None


def activity(profile):
    rows = []
    for case in json.loads((profile / "profile-manifest.json").read_text()):
        if case["returncode"]:
            raise ValueError(f"Failed profile: {case['case']}")
        name = case["case"]
        result = json.loads((profile / name / "result.json").read_text())
        with (profile / f"{name}-gpu.csv").open() as stream:
            samples = [{key.strip().split(" [")[0]: value.strip() for key, value in row.items()}
                       for row in csv.DictReader(stream)]
        utilization = [numeric(row["utilization.gpu"]) for row in samples]
        utilization = [value for value in utilization if value is not None]
        if not utilization:
            raise ValueError(f"No valid GPU samples: {name}")
        with (profile / f"{name}-cpu.csv").open() as stream:
            cpu_samples = list(csv.DictReader(stream))
        first, last = cpu_samples[0], cpu_samples[-1]
        elapsed = float(last["unix_time"]) - float(first["unix_time"])
        cpu_seconds = (int(last["process_cpu_ticks"]) - int(first["process_cpu_ticks"])) / case["clock_ticks_per_second"]
        timers = flat_timers(profile / name / "timer.dat")
        rows.append({"case": name, "energy_wall_s": result["wall_s"],
                     "process_wall_s": case["end_unix"] - case["start_unix"],
                     "gpu_names": sorted({row["name"] for row in samples}),
                     "gpu_uuids": sorted({row["uuid"] for row in samples}),
                     "samples": len(utilization), "mean_gpu_util_percent": statistics.mean(utilization),
                     "zero_gpu_util_sample_fraction": sum(v == 0 for v in utilization) / len(utilization),
                     "at_least_90_gpu_util_sample_fraction": sum(v >= 90 for v in utilization) / len(utilization),
                     "max_gpu_util_percent": max(utilization),
                     "process_cpu_seconds": cpu_seconds, "mean_process_cpu_cores": cpu_seconds / elapsed,
                     "timers": timers, "components_hartree": result["components_hartree"],
                     "options": result["options"]})
    return {"sampling": "nvidia-smi/NVML 200 ms requests; driver utilization windows may overlap",
            "scope": "Whole fresh child process, including imports/setup; GPU activity is not SM occupancy or FLOP efficiency",
            "cases": rows}


def report(profile, result):
    evidence = profile.parent
    paired = json.loads((evidence / "paired-summary.json").read_text())["rows"]
    text = ["# Why the A100 speedup is below 10×", "",
            "## Hardware and utilization: measured, not assumed", "",
            "The CUDA-visible device was resolved via its PCI bus ID and NVIDIA UUID to **NVIDIA A100-SXM4-80GB**, "
            "UUID `GPU-7027904a-5a89-6206-de14-6fceedb9846c`—the same device recorded for the original benchmarks. "
            "**It was not an L40S.** Driver 595.71.05; the environment uses CUDA 13.3 packages / libcuEST 0.2.1.2.", "",
            "Allocation **13020320**, step **12**, completed `0:0` in 2m40s. These are new, single-run diagnostics, "
            "not replacements for the published three-run timing medians. GPU and process CPU activity were sampled "
            "at 200 ms intervals for each whole fresh child process, including imports/setup.", "",
            "| Profile | energy() wall, s | Mean GPU activity | Samples at 0% | Samples ≥90% | Mean CPU cores used |",
            "|---|---:|---:|---:|---:|---:|"]
    for row in result["cases"]:
        text.append(f"| {row['case']} | {row['energy_wall_s']:.2f} | {row['mean_gpu_util_percent']:.1f}% | "
                    f"{100*row['zero_gpu_util_sample_fraction']:.1f}% | "
                    f"{100*row['at_least_90_gpu_util_sample_fraction']:.1f}% | {row['mean_process_cpu_cores']:.2f} / 8 |")
    text += ["", "**The GPU was not continuously busy.** NVML GPU-utilization windows may overlap; "
             "these percentages measure sampled device activity, **not SM occupancy or fraction of peak FP64 FLOP/s**. "
             "They are not retrospective utilization measurements of the earlier runs. A Nsight executable is "
             "available, but this collection did not capture a kernel/CPU-stack trace.", "",
             "## The measured phase bottleneck", "",
             "Flat `timer.dat` sections were parsed once per process; their duplicated call-tree sections were excluded. "
             "Timers are inclusive/nested and must not all be added together. Values below are medians of the "
             "three original matched runs, not telemetry-run timings.", "",
             "| System / basis | CPU J/K, s | GPU J/K, s | J/K speedup | CPU XC, s | GPU XC, s | GPU total, s | XC / GPU total |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for system, basis in (("benzene", "aug-cc-pvdz"), ("nanotube", "6-31+g**")):
        times = {}
        for mode in ("cpu", "gpu"):
            timers = [flat_timers(profile / "reference-timers" / f"{system}-{basis}-{mode}-{repeat}.dat")
                      for repeat in (1, 2, 3)]
            times[mode] = {name: statistics.median(t[name]["wall_s"] for t in timers)
                           for name in ("JK: JK", "RV: Form V")}
        row = next(r for r in paired if r["system"] == system and r["basis"] == basis)
        total = row["wall_s"]["gpu"]["median"]
        cpu, gpu = times["cpu"], times["gpu"]
        text.append(f"| {system} / {basis} | {cpu['JK: JK']:.2f} | {gpu['JK: JK']:.2f} | "
                    f"{cpu['JK: JK']/gpu['JK: JK']:.1f}× | {cpu['RV: Form V']:.2f} | {gpu['RV: Form V']:.2f} | "
                    f"{total:.2f} | {100*gpu['RV: Form V']/total:.1f}% |")
    protein = flat_timers(profile / "reference-timers/protein157-gpu-1.dat")
    text += ["", "`JK: JK` excludes DF setup: the quoted 18–23× factors are **J/K-call speedups**, "
             "not full density-fitting or total-method speedups. They show that the J/K acceleration itself is substantial.", "",
             f"Protein157's GPU run spent **{protein['RV: Form V']['wall_s']:.2f} s** in XC potential formation "
             f"and **{protein['JK: JK']['wall_s']:.2f} s** in J/K, out of 486.21 s total. Its CPU comparison is separate.", "",
             "For nanotube, making its remaining ~3.01 s of GPU J/K calls free would improve total speedup "
             "only from 4.41× to about **4.65×** (holding everything else fixed). To reach 10× with the other "
             "phases unchanged, the ~41.38 s XC phase would have to fall below roughly **9.31 s**. "
             "A universal 10× expectation is particularly inappropriate for small systems with initialization "
             "and ~2 s of D4-call overhead.", "",
             "## Why XC/GRAC still stalls on the host", "",
             "The source confirms that `CUEST_XC=True` does **not** mean all functional work is GPU-native:", "",
             "1. `RV::compute_V` obtains grid densities on the GPU, then performs blocking device-to-host "
             "copies of weights and densities, plus a host transpose.",
             "2. [`evaluate_cuest_grac`](../../../../psi4/src/psi4/libfock/v.cc#L203-L251) builds one Psi4 "
             "functional worker and evaluates **256-point blocks serially on the CPU**. Its point loop includes "
             "map lookups, density/gradient conversion, and LibXC/GRAC functional evaluation.",
             "3. The non-GRAC LDA/GGA/meta-GGA branches have OpenMP regions, but the GRAC branch calls this "
             "serial helper instead. Thus eight requested threads do not parallelize this helper.",
             "4. The resulting potential is copied back to the GPU for AO integration, followed by another "
             "device-to-host AO-matrix copy. Buffers/weights/workspaces are repeatedly allocated or rebuilt.", "",
             "`RV: Form V` contains **all** these host and GPU phases, so its entire time cannot be assigned to "
             "the serial helper alone. The source plus device-idle samples identify a strong bottleneck candidate; "
             "exact helper, copy, allocation, and kernel costs still require finer timers or Nsight tracing.", "",
             "## Native FP64, mixed precision, and CUDA 13", ""]
    native = next(r for r in result["cases"] if r["case"] == "nanotube-native")
    mixed = next(r for r in result["cases"] if r["case"] == "nanotube-mixed")
    delta = max(abs(mixed["components_hartree"][k]-value) for k, value in native["components_hartree"].items())
    text += [f"Native FP64: **{native['energy_wall_s']:.2f} s** total, **{native['timers']['JK: JK']['wall_s']:.3f} s** J/K. "
             f"Mixed precision allowed: **{mixed['energy_wall_s']:.2f} s** total, "
             f"**{mixed['timers']['JK: JK']['wall_s']:.3f} s** J/K. Maximum component difference: **{delta:.3e} Eh**.", "",
             "There is no material end-to-end difference in this **single pair**. The baseline explicitly requests "
             "`CUEST_MIXED_PRECISION=False`; `cuESTJK::preiterations` calls `cuestSetMathMode(..., "
             "CUEST_NATIVE_FP64_MATH_MODE)` before iterative J/K work. The enabled flag permits mixed precision; "
             "it does **not prove** which internal cuEST kernel/precision was selected on the A100. A kernel trace "
             "would be needed for that claim. No L40S measurements or CUDA-12 control were performed, so this is "
             "**not evidence for or against a general CUDA-13 regression**.", "",
             "## Optimization priorities", "",
             "1. Add subphase timers/NVTX around GRAC host evaluation, density/potential copies, GPU density "
             "and AO integration, and allocation/setup; collect a Nsight Systems trace.",
             "2. Parallelize GRAC blocks with one independent functional worker/input set per thread; cache "
             "workers and hoist repeated map lookups. Preserve the tested spin factors and nonzero-shift controls.",
             "3. Reuse fixed-grid weights and work buffers; avoid repeated blocking transfers where possible.",
             "4. Longer term, evaluate GRAC on device to eliminate the host round trip. Benchmark and retest "
             "energies, densities, and orbital shifts before claiming a 10× total speedup.", "",
             "No performance-sensitive Psi4 implementation code was changed for this investigation.", "",
             "## Reproduce", "", "```bash",
             "python devtools/benchmarks/analyze_gpu_profile.py \\",
             "  devtools/benchmarks/results/phoenix-a100-20260909/profile \\",
             "  --output devtools/benchmarks/results/phoenix-a100-20260909/profile/summary.json \\",
             "  --report devtools/benchmarks/results/phoenix-a100-20260909/GPU_PROFILE.md",
             "```", "",
             "Raw sampled CSVs, per-case JSON and timers, original comparison timers, launcher, runner snapshots, "
             "device mapping, and file hashes are preserved in `profile/`. The first telemetry attempt failed "
             "before sampling because the launcher used an unavailable runtime API symbol; basis counts from "
             "its completed count stage were retained, and step 12 used PCI-bus mapping successfully.", ""]
    return "\n".join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = activity(args.profile)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.report:
        args.report.write_text(report(args.profile, result))
    print(f"Summarized {len(result['cases'])} profiled cases")


if __name__ == "__main__":
    main()
