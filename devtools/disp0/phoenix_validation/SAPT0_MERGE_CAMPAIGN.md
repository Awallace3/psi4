# Canonical GPU SAPT0 integration and matched term timings

User requested the current Disp0 GPU implementation on `saptdft_cuest`, including
the target's existing uncommitted development, and selected **spherical
jun-cc-pVDZ**. This is canonical SAPT0(+delta-HF) via `energy("sapt(dft)")`,
`SAPT_DFT_FUNCTIONAL=HF`, **not** localized F-SAPT or the CPU CPKS split.

## Integration

- Target checkpoint: `e8ba9e52ac` (prior committed target `5f154adeb7`).
- Disp0 input: `4f0b9ee8e1`; includes the pending spherical qualification fixes.
- Merge: `8bbbf411cb`. Preserves native JK plan reuse/device response/DIIS;
  retains relative MATRIXPOWER fitting and FP64 before native plan construction;
  resolves the Einsums/cuBLAS bridge with `out="tensor"` and no redundant
  production induction reference evaluation.
- 391 host dispersion tests pass. These are not GPU numerical qualification.
- A fresh coherent native build is required; previous Disp0 receipts are not
  relabeled as tests of the merge.

## Matched experiment

Four standard `~/data/timing_test_suite/geometries/systems.py` dimers:
peptide, nanotube, protein83, protein157. PLA15 is excluded. Freeze the exact
geometry strings and source hash with `sapt0_timing.py prepare`. Do not change
suite geometries or use its old RHF/Cartesian timings as SAPT0 baselines.

Both arms: jun-cc-pVDZ, spherical aug-cc-pVDZ-JKFIT and aug-cc-pVDZ-RI,
frozen core, coupled HF induction, delta-HF enabled, no F-SAPT, no empirical
dispersion. CPU uses the existing native SAPT dispersion backend; GPU uses
the public automatic cuEST dispersion continuation and audited provider,
block edge 32, explicit 4 GiB packed-factor cache. No default cache change.
The first comparison explicitly retains default-off device CG and device DIIS;
both arms use the existing response iteration/SCF DIIS policy, with native
cuEST J/K on the GPU arm. Direct-J/Grid-K and response profiling are off.
This tests the GPU-enabled driver, not a claim that every operation is
device-resident. Device CG/DIIS optimizations require a separately labeled
qualification and comparison.

Run a small water CPU/GPU gate first. Then three **fresh-process** measurements
per arm/case, alternating CPU/GPU order, on the same allocated H200 host with
eight CPU threads in both arms. Report hardware and this CPU thread count;
this is not a comparison against a 24-core CPU node. Each energy-call timer
includes native provider construction, JIT, audits and cleanup reached inside
that call. Runtime import/admission outside the call is separately excluded
for both arms. No fitted speedups or overhead subtraction.

### Disjoint timing boundaries

- `elst`: `SAPT(DFT):elst`.
- `exch`: `SAPT(DFT):exch`.
- `ind`: `SAPT(DFT):ind` **minus** its nested delta-HF assembly.
- `disp`: `SAPT(DFT):disp`, including both signed dispersion components.
- `delta_hf`: dimer HF SCF + delta-HF assembly. Shared monomer HF references
  are **not** charged a second time. This is an attribution convention, not
  a counterfactual on/off estimate: the dimer SCF also initializes the shared JK.
- `shared_setup_other`: measured energy-call wall minus the five intervals.
- `total`: the entire actual energy call.

Report medians and all three samples; speedup = CPU median / GPU median.
Zero-resolution measurements have a null ratio, never infinity.

### Qualification

Require the same build, input/options, host, memory and thread counts; compare
all seven signed raw components plus total, not only summed energies.
CPU/GPU tolerance: atol 2e-8 Eh / rtol 1e-8 (the existing whole-energy policy).
Within-arm fresh-process repeats:
atol=rtol=1e-9. Check disjoint term-energy reconstruction to 1e-10 Eh.
No CPU dispersion fallback in GPU runs; require actual native GPU resources
and the audited provider's complete cleanup receipt. The CPU arm must leave
native GPU resources uninitialized.

Any failed numerical repeat, memory admission or job remains a failed/missing
case without an accepted speedup. Larger dimers require resource preflight;
no reduced basis or weakened tolerance may silently replace them.

## State

Merge committed and pushed. Native build **15446271** submitted using existing
read-only, pinned cuEST 0.2.50 / CUDA 13.3 / Einsums064 dependencies.
Route `gts-cs207-chemx / cpu-small / embers`, 8 CPUs / 48 GiB / 3 hours.
Root:
`/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/sapt0-merged-build-8bbbf411cb-01`.
Eight benchmark accounting/acceptance host contracts pass. No new timing
result yet.

Offline spherical shell counts from the exact checked-in basis files:

| Case | Primary | JKFIT | RIFIT | Single unscreened triangular JK tensor / GiB |
|---|---:|---:|---:|---:|
| peptide | 250 | 1308 | 1042 | 0.31 |
| nanotube | 548 | 2748 | 2240 | 3.08 |
| protein83 | 922 | 4762 | 3820 | 15.10 |
| protein157 | 1786 | 9182 | 7384 | 109.17 |

These are sizing estimates, not runtime basis queries or total peak memory.
The protein157 native plan is near H200 capacity before other allocations;
first qualify the smaller cases and measure actual memory. Its CPU DF setup
can require disk spill under the matched 96 GiB Psi4 budget. Preserve an
explicit memory failure rather than changing its basis.
