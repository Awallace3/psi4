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

## Submission receipt

- Job **15497866**, helper pin `4ea6483cd1637f44b4fdc2595040c9c6a3c44610`.
- Native build remains `8bbbf411cb`, job 15446271.
- Remote source: `/storage/project/r-cs207-0/awallace43/gits/psi4.sapt0-blackwell-4ea6483cd1`.
- Output root:
  `/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/sapt0-blackwell-peptide-4ea6483cd1-01`.
- Actual readback: `Account=gts-cs207 QOS=inferno TimeLimit=02:00:00`,
  `Partition=gpu-rtxpro-blackwell,gpu-v100`,
  `TresPerNode=gres/gpu:rtx_pro_6000_blackwell:1`.
- The submission helper exited nonzero **after acceptance** because its initial
  partition comparison rejected Phoenix's appended default `gpu-v100`. The
  typed Blackwell GRES prevents V100 allocation. This was manually verified;
  no duplicate job was submitted and no job settings were changed.
- Test-only predicted Nov 11 at 19:43 Eastern; actual pending job initially had
  `StartTime=Unknown`. Do not present the test-only prediction as a reservation.
- Host suite: 178 tests passed before the two additional identity/shell tests;
  all 14 focused SAPT0 tests then passed. No GPU compatibility or speedup claim.

The route-check fix and this receipt are later local helper history. Do not
advance the detached remote source used by the queued job.
