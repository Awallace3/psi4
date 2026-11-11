# Canonical SAPT0 Blackwell pilot

User requested this pilot on 2026-11-11 and explicitly approved one two-hour
`gts-cs207 / gpu-rtxpro-blackwell / inferno` allocation (estimated maximum
$0.92 at submission planning). Leave H200 job 15447098 unchanged.

This helper-only branch starts at 9c663b0d52. Reuse the coherent native build
8bbbf411cb from job 15446271, with no installed Python overlays or binary changes.
Submission rejects any difference under `psi4/` between helper and native pins.
Do not merge newer unrelated, concurrently developed scientific changes here.

The hardware contract is RTX PRO 6000 Blackwell (`compute_120`), with an 80-GiB
explicit CUDA budget including a 12-GiB native reservation. This is a cap, not a
measured peak. The runtime verifies the actual current-PID GPU UUID, name, CUDA
architecture, physical capacity and absence of foreign GPU compute contexts.
Unknown GPUs and wrong architectures fail closed. CUDA 13.3 compiles for the
actual device at runtime; compatibility with cuEST 0.2.50 remains to be tested
inside the allocation. This document makes no GPU compatibility claim.

Run water CPU/GPU once first, then peptide with three fresh processes per arm
in alternating order. Eight scientific CPU threads and 96-GiB host memory in
both arms; one exclusive node requested with 160-GiB host memory and one GPU.
Spherical jun-cc-pVDZ, explicit matching auxiliary bases, convergence settings,
signed-component tolerances and all five term timing definitions are unchanged
from the H200 campaign. CPU/GPU comparisons use the same Blackwell host; do not
combine its CPU baseline with H200 measurements.

The existing public integer-keyword normalization issue concerns values zero
and one; this pilot uses block size 32 and cache size 4 GiB, so it does not need
the later scientific-source patch. The worker activates `--psiapi` before
enabling shell nounset, while preserving errors from the launcher and eval.

Only water and peptide are admitted for this initial Blackwell qualification.
No larger dimer or scientific-policy relaxation is automatic. Any compatibility,
SCF convergence, repeatability or signed-parity failure stops the pilot, preserves
the evidence, and produces no accepted speedup. In particular, a passing host
test or successful submission is not a passing GPU result.

Artifacts: per-job immutable manifest, test-only and submission receipts,
effective Slurm route, runtime config and hardware receipt, raw worker outputs,
term energies/timers and `water/comparison.json`, then `peptide/comparison.json`.
