# Qualified small H200 correctness gate

## Final result

**14377643: PASS_SMALL_H200_ONLY**, completed2026-10-24 in29m42s.
Job, batch and step accounting are COMPLETED0:0; saved job status is exit0,
TERM0. Actual prolog confirms gts-cs207-chemx/inferno; the allocation was
1H200,4CPUs32GiB,1h cap. This follows the complete Cachy gate and successful
whole Phoenix build/CPU preparation14377089. No paid retry was submitted.

All10 fresh processes completed: CPU/GPU/independent same-gauge CPU oracle for
unscaled, scaled and displaced water, then a fresh scaled GPU repeat.
All23 required fields, signed components, output-file roundtrips, nonunit
scaling, changed energy and native/device cleanup passed. Final artifacts,
matrix/raw-input/output-file hashes and actual hardware were independently
inspected by task `b16f563b8`, not inferred from scheduler completion.

**Accepted timing samples:0. Large systems admitted:false. No GPU speedup.**

## Matched numerical case

HF SAPT(DFT)/spherical aug-cc-pVDZ, aug-cc-pVDZ-JKFIT/aug-cc-pVDZ-RI and
CC-PVTZ-MINAO; RHF/DF, frozen core, C assignment, exact per-source coupled HF
fragment response through the public CPKS option. Six atoms,82AOs, five
occupied/one frozen/four active orbitals per monomer.

The diagnostic uses `INTS_TOLERANCE=1e-16` in BOTH CPU/GPU arms after the Cachy
screening probe reproduced the default-cutoff nuclear-potential error.
Independent one-electron comparison remains1e-10; fragment matrices retain
atol2e-7/rtol2e-5; repeat fields retain1e-9. Repeat orbital labels/signs must
be proven by occupied overlaps, reconstruction and frozen-space checks.
Energy entries are only permuted, never densely rotated or sign scaled.

Scaled cases enable canonical exchange normalization and delta-HF induction
scaling. “Changed” translates all B atoms by+0.08Å. Virtual block size remains5.
The original production CPU comparisons use the cheaper scalar-rescaled
uncoupled partition and their native screening default, not this exact mode.

| GPU case | Diagnostic energy-call seconds | Interaction energy (Eh) | Maximum independent energy-matrix error (Eh) |
|---|---:|---:|---:|
| Unscaled | 428.154767 | -0.007157905735663632 | 1.08091314e-12 |
| Scaled | 428.883247 | -0.008584114667335198 | 1.37964293e-11 |
| Shifted B, scaled | 427.428554 | -0.008414604518010717 | 6.47697174e-13 |
| Fresh scaled repeat | 428.370688 | -0.008584114667449069 | Repeat energy-matrix difference6.07479345e-13 |

These are four different diagnostic/control calls, **not three performance
repeats of every system**. The timer encloses the energy call and observing
capture work; runtime admission and the physical primitive probe precede it.
It is not full-process/first-use benchmark time. CPU control calls took
2.786–3.301s. Thus the small diagnostic demonstrates correctness, not
acceleration; no speedup ratio is accepted.

## Actual device and cleanup evidence

- NVIDIA H200, UUID `GPU-75553820-c98c-532e-b792-5c9e20048a1d`;
  NVML143771MiB, driver615.71.09; actual NVRTC architecturecompute_90.
- CUDA13.3; cuEST0.2.50; cuda-bindings13.3.0/cuda-pathfinder1.8.4;
  Python/native Einsums world06404fa69ce9346443d7d4f19a250f6db9f0f62c.
- Every GPU case executed the physical batched-BLAS/high-index-IBO controls,
  native device-input DFMO and the full continuation. Each fragment plan
  records1040 strided-batched FP64 GEMMs,260 `response_finish` launches and
  66 `ibo_pair` launches, among other actual device operations.
- Every owner/session cleanup is complete, zero live explicit bytes/native
  owners; provider coefficient H2D bytes are0. This does not imply zero upstream
  HF-array transfers.
- Maximum tracked explicit device allocation is88794624bytes (84.681MiB).
  **Not total/peak VRAM:** native persistent/workspace and driver allocations
  are separate. Per-GPU-process peak RSS ranges997.4–1002.0MiB; this is not the
  allocation's total cgroup memory peak.

## Immutable provenance and limits

Source/helper `d471cd93fa1c7d81f58209771c33280c2fe2e9c0`;
whole build14377089, Psi4 1.12a1.dev867,271 CPU regressions/61 helper tests,
native CPU RHF and both exact public CPU preflights passed.

Root:
`/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/phoenix-h200-guard-fixed-d471cd93fa/job-14377643`

Gate-result SHA256:
`094b91f8347818fce849b24591c82368b07243fec1e98554693cdfd15a3f4d4e`

Runtime-config SHA256:
`ddf6c333a6f9c3a5ed638379a615cf96129142125b6b78da6a62a4d74cb8971a`

Original failed H20014322613 and Cachy attempts01–03 remain unchanged.
The old H200 timeout is not relabeled as a segfault or a success.
Protein/large-space resource and numerical gates, a matching production GPU
continuation or exact CPU performance baseline, and the declared three-repeat
CPU/GPU measurements remain outstanding. No large campaign or further paid
allocation is authorized by this one-gate result.
