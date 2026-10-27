# Current-stage split F-SAPT benchmark

User authorization: one GPU job and multiple CPU inferno jobs. No CPU
reference or same-gauge CPU oracle runs in the GPU allocation. No automatic
retry, scheduler modification, guard removal, or tolerance relaxation.

## Science and timing

- Fresh coherent whole stage at `d2a98abe3529027256efa678be2af6ab89c8a415`;
  response-workspace reuse and opt-in packed-S dispersion reuse are included.
- Exact per-source **HF CPHF**, spherical aug-cc-pVDZ, augmented JKFIT/RI,
  frozen core, matched `INTS_TOLERANCE=1e-16`, exchange/delta-HF scaling.
  This is not the cheaper production-scaled uncoupled induction partition.
  The previous production CPU timings are not used as its denominator.
- Three fresh molecular energy calls per system/arm: peptide, nanotube,
  protein83 (the previously confirmed suite protein entry).
- GPU cache ceiling: additional 64 MiB per call, within unchanged total live
  device guards. Actual cache occupancy/hits remain in execution metadata.
  This budget may not fit a large slab; do not assume a cache hit or gain.
- Instrumented whole-energy timing includes normal F-SAPT output, scientific
  setup/SCF/canonical work, required native audits, and full continuation.
  Runtime admission/primitive controls precede it. Diagnostic HF/fragment
  copies, NPZ writes and post-call numerical validation follow it.
  Scientific runtime/native synchronization remains enabled.
- `--benchmark` retains the existing HF/output Matrix/Vector owners until
  post-call snapshots, rather than inserting large diagnostic copies into the
  timer. It does not supply saved orbitals/localizations to production calls.
- Same-arm repeat/gauge tests, signed separated components, canonical
  energy and independent native CPU fixed-GPU-gauge oracles remain required.
  A speedup ratio is recorded only after three qualified samples in each
  arm and all three independent oracles. No completed worker alone is enough.
- Report the resource policy explicitly: 24-core CPU node versus one H200
  with 8 host threads; not a matched-host-thread or GPU-kernel-only ratio.
  This measures current CPU/GPU relative latency, not a pre/post optimization
  comparison of two whole GPU implementations.

## Allocation plan

| Allocation | Resource | Cap |
|---|---|---|
| Fresh whole build/CPU preparation | 8 CPUs, 48 GiB, no GPU | 3 h |
| GPU producer, all three systems | 1 H200, 8 CPUs, 160 GiB | 16 h |
| Peptide CPU reference + oracles | 24 CPUs, 160 GiB, exclusive Gold6226 | 2 h |
| Nanotube CPU reference + oracles | same CPU shape | 24 h |
| Protein83 CPU reference + oracles | same CPU shape | 24 h |

CPU jobs use node-local writable scheduler scratch, and depend `afterany` on
the GPU producer. References execute even if the corresponding GPU case was
unqualified, so valid CPU baseline work is not thrown away. Oracles and speedup
qualification require a complete matching GPU case; no GPU resubmission is
triggered by missing or failed data.

The GPU allocation starts with a fresh scaled water GPU call compared with
the **current build's** completed native CPU water control. This is a new
experimental batch with CPU build evidence plus existing component-level
Cachy evidence, not a claim that old H200 release `14377643` qualifies the
new code. The complete new GPU/CPU case qualification occurs in the separate
CPU jobs. No fake `PASS_SMALL_H200_ONLY` release is generated.

## Known failures remain visible

Nanotube previously failed orbital-resolved localization-repeat equivalence;
protein83 previously failed native RIFIT creation-workspace admission.
Workspace/cache changes do not purport to fix either. Each remains an
experimental point: a failed repeat stops that point without retry, and a
safe completed child allows the other distinct approved systems to run.
CPU references and original failed artifacts are preserved. No failed point
may be promoted into an accepted performance median by selecting samples.

## Entry points

Use `benchmark_suite.py` to prepare fresh per-role manifests and submit one
explicit role at a time after fresh account selection. `--test-only` runs
before every actual scheduler call, and durable attempt/response/job-ID files
prevent ambiguous automatic resubmissions. Source/helper paths and installed
science pins are separate and recorded; no scientific core/Python overlays.

Fresh build submission: **14465563**, `gts-cs207-chemx/cpu-small/inferno`,
source worktree
`/storage/project/r-cs207-0/awallace43/gits/psi4.disp0-benchmark-d2a98abe35`.
Build root:
`/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/phoenix-benchmark-build-d2a98abe35`.
The account selector reported monthly `gts-cs207` $0.89 below its $1 floor;
the full build cap is about $0.20. Build submission is not build completion,
GPU submission, or benchmark success.
