#!/usr/bin/env python3
"""Render the committed Phoenix evidence without inventing missing measurements."""
import argparse
import json
from pathlib import Path

from summarize_saptdft_cuest import markdown, summarize


def load(path):
    return json.loads(path.read_text())


def deltas(reference, result):
    for key in ("geometry", "basis", "nbf", "threads", "memory", "psi4_version"):
        if reference.get(key) != result.get(key):
            raise ValueError(f"Mismatched {key}")
    return {key: result["components_hartree"][key] - value
            for key, value in reference["components_hartree"].items()}


def build_paired_summary(evidence):
    """Rebuild the six complete paired groups, retaining original accuracy misses."""
    initial = summarize(evidence / "raw/retry1/results")
    suite = summarize(evidence / "raw/suite2/results")
    rows = initial["rows"] + suite["rows"]
    expected = {(system, basis) for system in ("water", "benzene")
                for basis in ("cc-pvdz", "aug-cc-pvdz")}
    expected.update({("peptide", "6-31+g**"), ("nanotube", "6-31+g**")})
    if {(row["system"], row["basis"]) for row in rows} != expected or len(rows) != 6:
        raise ValueError("Expected exactly six complete paired system/basis groups")
    if not suite["complete"] or suite["failures"]:
        raise ValueError("Corrected suite campaign must be complete without execution failures")
    if any(row["paired_repeats"] != 3 or row["repeats"] != {"cpu": 3, "gpu": 3}
           for row in rows):
        raise ValueError("Every published paired row must contain three repeats per backend")
    count_path = evidence / "basis-counts.json"
    if count_path.exists():
        counts = load(count_path)["counts"]
        for row in rows:
            count = counts[row["system"] + "/" + row["basis"]]
            if count["dimer"] != row["nbf"]:
                raise ValueError("Recomputed basis count differs from measured calculation")
            row["nbf_monomer_a"] = count["monomer_a"]
            row["nbf_monomer_b"] = count["monomer_b"]
    return {"repeats_per_backend": 3, "accuracy_tolerance_hartree": 1e-6, "rows": rows,
            "note": "Original-grid benzene threshold misses retained; independent grid controls reported separately.",
            "source_campaigns": {"retry1": {"complete": initial["complete"], "failures": initial["failures"]},
                                 "suite2": {"complete": suite["complete"], "failures": suite["failures"]}}}


def render(evidence, paired=None):
    paired = paired if paired is not None else build_paired_summary(evidence)
    table = markdown({"complete": True, "rows": paired["rows"], "failures": [],
                      "accuracy_tolerance_hartree": paired["accuracy_tolerance_hartree"]})
    table = table.split("## Failed or incomplete measurements")[0]
    text = [table, "See [GPU utilization and bottleneck analysis](GPU_PROFILE.md) for sampled A100 activity, "
            "native/mixed-precision controls, and the XC/GRAC host bottleneck.", "", "## Protocol and hardware", "",
            "Measured on Phoenix, 2026-09-09, allocation **13020320**: one NVIDIA A100-SXM4-80GB "
            "and eight AMD EPYC 7543 cores (affinity 8–15). The node was shared, not exclusively reserved.", "",
            "Three fresh-process measurements per backend and system/basis; identical eight-thread settings "
            "for OpenMP/MKL/OpenBLAS. No discarded warm-up. Python imports and the preliminary nbf check "
            "are outside the timer; `energy()` includes its internal setup and GPU initialization. "
            "Do not use the historical timing suite's RHF/24-thread numbers as the CPU baseline.", "",
            "SAPT(DFT)-D4(I), PBE0, induction from delta-HF (`INDUCTION_TYPE=NONE`, `DO_DHF=True`), "
            "fixed GRAC shifts 0.136 Eh on both monomers, no automatic IP/GRAC calculation; 99×590 grid, "
            "SCF convergence 1e-9/1e-8, native double precision. The fixed shifts exercise GRAC but are "
            "not claimed to be physically determined for these systems. Response-based induction, "
            "GRAC gradients, and cuEST XC response are outside this evidence.", "",
            "Water/benzene use their default fitting bases. Peptide and nanotube use **spherical** "
            "6-31+G** (`PUREAM=True`), def2-universal-jkfit and aug-cc-pVDZ-RI fitting bases in **both** arms. "
            "Their nbf values (250/548) intentionally differ from the suite's Cartesian 260/574. "
            "Benzene is idealized, not the published S22 geometry. Full geometries/options are in each raw JSON.", "",
            "Build: source-equivalent commit `c7927d3ec9f992ef3906f4227404c99fe9413ae2`, Psi4 "
            "`1.12a1.dev631`, Release `-O3 -DNDEBUG`; libcuEST 0.2.1.2, Einsums 1.1.2, DFT-D4 3.7.0, "
            "LibXC 7.0.0, MKL 2025.3.0, CUDA packages 13.3, driver 595.71.05, Python 3.13.11. "
            "The recorded snapshot base plus patch matches all eight files changed through the source commit. "
            "See `initial-build-provenance.json`, `hardware.json`, and protocol snapshots.", "",
            "## GRAC accuracy controls", "",
            "**Do not interpret all original-grid rows as passing 1e-6 Eh.** Both benzene rows miss that "
            "predeclared component threshold; the largest difference is about 0.0011 kcal/mol. "
            "The following single-run aug-cc-pVDZ controls isolate quadrature sensitivity without changing "
            "the original timing table or loosening its threshold.", "",
            "| Control | Max component difference, Eh | Interpretation |",
            "|---|---:|---|"]
    raw = evidence / "raw"
    cpu = load(raw / "retry1/results/benzene-aug-cc-pvdz-cpu-1/result.json")
    gpu = load(raw / "retry1/results/benzene-aug-cc-pvdz-gpu-1/result.json")
    cpuxc = load(raw / "controls/gpu-cpuxc/result.json")
    fine_cpu = load(raw / "controls/cpu-finegrid/result.json")
    fine_gpu = load(raw / "controls/gpu-finegrid/result.json")
    zero = load(raw / "controls/gpu-unshifted/result.json")
    maxdiff = lambda a, b: max(map(abs, deltas(a, b).values()))
    text += [f"| GPU J/K + CPU XC versus CPU | {maxdiff(cpu, cpuxc):.3e} | Passes; isolates XC/grid effects |",
             f"| GPU XC versus CPU, 250×974 grid | {maxdiff(fine_cpu, fine_gpu):.3e} | Passes; denser-grid agreement |",
             f"| GPU nonzero GRAC versus zero shift | {maxdiff(zero, gpu):.3e} | Correction demonstrably changes the result |", "",
             f"The denser-grid single CPU/GPU timings were {fine_cpu['wall_s']:.2f}/{fine_gpu['wall_s']:.2f} s "
             f"({fine_cpu['wall_s'] / fine_gpu['wall_s']:.2f}×). This is a diagnostic **single pair**, not a three-run median.", "",
             "Step **13020320.9** completed `0:0`. Separately, **51 focused regression tests** passed "
             "(zero failures/errors/skips) on the same source-equivalent build in step **13020320.5**, "
             "including density/orbital GRAC checks, CPU GRAC, SAPT-D4(I), J/K, and GEMM tests. "
             "See `pytest-fixed-4.xml` and `validation-final.json`; this is not the full Psi4 suite.", "",
             "## Protein157: separate batch measurements", ""]
    p_gpu = load(raw / "protein157/results/protein157-gpu-1/result.json")
    text += [f"GPU job **13021499** completed `0:0` (batch elapsed 8m27s). Its single `energy()` "
             f"measurement was **{p_gpu['wall_s']:.2f} s ({p_gpu['wall_s']/60:.2f} min)**, "
             f"with **{p_gpu['nbf']} basis functions**. Same spherical 6-31+G** / fitting bases and fixed "
             "GRAC protocol; eight threads, 112 GiB Psi4 memory, 128 GiB allocated host memory, "
             "one A100 80 GB. Four cuEST builders were observed with no CPU DF J/K fallback.", "",
             "Largest reported allocations: DF integral plan **43.339 GiB**, J/K scratch **10.528 GiB**, "
             "AO pair list **0.007 GiB**. These are cuEST workspace reports, not an instrumented GPU peak. "
             "SLURM sampled step MaxRSS was **5,570,940 KiB** (~5.31 GiB); requested host memory is not usage.", "",
             "| Component | GPU energy, Eh |", "|---|---:|"]
    text += [f"| {key} | {value:.12f} |" for key, value in p_gpu["components_hartree"].items()]
    count_path = evidence / "basis-counts.json"
    if count_path.exists():
        count = load(count_path)["counts"]["protein157/6-31+g**"]
        text += ["", "| System | MonA own nbf | MonB own nbf | Dimer / ghosted-monomer SCF nbf |",
                 "|---|---:|---:|---:|",
                 f"| Protein157 | {count['monomer_a']} | {count['monomer_b']} | {count['dimer']} |", ""]
    p_cpu_path = raw / "protein157-cpu/results/protein157-cpu-1/result.json"
    if p_cpu_path.exists():
        p_cpu = load(p_cpu_path)
        if not p_cpu["ok"]:
            text += ["", "CPU job 13021636 did not produce a successful result; no timing ratio is reported."]
        else:
            diff = deltas(p_cpu, p_gpu)
            text += ["", f"CPU job **13021636**: {p_cpu['wall_s']:.2f} s; "
                     f"single-pair CPU/GPU ratio **{p_cpu['wall_s']/p_gpu['wall_s']:.2f}×**. "
                     "This is a **cross-node/cross-hardware comparison**, not an isolated same-host speedup.",
                     f"Maximum component difference: **{max(map(abs,diff.values())):.3e} Eh**.", ""]
    else:
        text += ["", "A matched CPU baseline was submitted separately as **13021636** "
                 "(eight CPUs, 128 GiB, 8-hour `embers` cap). The saved script requests `cpu-small`; "
                 "Phoenix's submission policy automatically remapped it to **`cpu-medium`**, with no "
                 "command-line partition override. **Its result is not yet "
                 "included here, so no protein157 speedup or CPU-agreement claim is made.** "
                 "The CPU partition may use different host hardware; any eventual timing ratio must disclose that.", ""]
    text += ["## Failures and completion accounting", "",
             "- Initial launcher step 13020320.6 failed before calculations because conda overwrote `BUILD`; "
             "the launcher now uses `PSI4_BUILD`.",
             "- Step 13020320.7 contains successful water/benzene child processes, each with zero return "
             "code and finalized JSON, but the overall step failed on peptide's generated Cartesian auxiliary basis. "
             "The 25th successful calculation was an **unpaired Cartesian peptide CPU attempt**, excluded from timing comparisons.",
             "- Step 13020320.8 again failed on peptide: naming a spherical fitting basis alone still inherited "
             "the primary Cartesian setting. This unpaired CPU attempt is also excluded. Both errors remain in `raw/`.",
             "- Step **13020320.10** completed `0:0`, including all 12 corrected spherical peptide/nanotube calculations. "
             "The benchmark does not quietly change only the GPU basis to obtain agreement.",
             "- Original-grid benzene accuracy misses remain visible above; they are not discarded runs.", "",
             "## Reproduce the summaries", "", "```bash",
             "python devtools/benchmarks/summarize_saptdft_cuest.py \\",
             "  devtools/benchmarks/results/phoenix-a100-20260909/raw/retry1/results --output /tmp/initial-summary",
             "# Expected nonzero exit: initial campaign failed later, and benzene misses the original-grid threshold.",
             "python devtools/benchmarks/summarize_saptdft_cuest.py \\",
             "  devtools/benchmarks/results/phoenix-a100-20260909/raw/suite2/results --output /tmp/suite-summary",
             "# Rebuild paired-summary.json directly from both raw campaigns, validate all six groups,",
             "# and regenerate this report (retaining original accuracy failures):",
             "python devtools/benchmarks/render_phoenix_report.py",
             "```",  "",
             "Exact run-time runner snapshots are in `protocols/*.py.txt`; the maintained runner is two "
             "directories above this evidence directory. Raw Psi4 outputs are retained under "
             "`/storage/project/r-cs207-0/awallace43/runs/psi4-cuest-timing/` on Phoenix and locally in "
             "`tmp/benchmark-results/`; `output-sha256.json` records their hashes. All five component energies, "
             "geometries, settings, timings, and run/step IDs are committed as raw JSON.", ""]
    return "\n".join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path,
                        default=Path(__file__).parent / "results/phoenix-a100-20260909")
    args = parser.parse_args()
    paired = build_paired_summary(args.evidence)
    (args.evidence / "paired-summary.json").write_text(json.dumps(paired, indent=2) + "\n")
    (args.evidence / "README.md").write_text(render(args.evidence, paired))


if __name__ == "__main__":
    main()
