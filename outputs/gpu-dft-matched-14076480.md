# Matched-grid DFT recovery — H200 job 14076480

## Status

- Peptide preflight: **8/8** functional/basis/DF-strategy comparisons complete.
- Protein83: **16/20** comparisons complete, with three timed repeats per engine.
- Protein83/WB97X/aug-cc-pVTZ fails in GPU4PySCF 1.8.1's range-separated
  three-center-integral batching:
  `df/int3c2e_bdiv.py:199`,
  `assert max(ao_pair_offsets[1:] - ao_pair_offsets[:-1]) < ao_pair_batch_size`.
  Both native-grid and shared-grid attempts fail in both DF strategies.
- The declared four-arm gate also withholds WB97X/cc-pVTZ timing repeats after
  the augmented arm fails. Failed/missing cases are **not** zero-time results.
- SLURM status: FAILED 1:0 after 1h45m, despite the usable partial panel.
  No paid retry has been submitted.

## What changed the memory conclusion

For Protein83/PBE/aug-cc-pVTZ, GPU4PySCF's sampled whole-device peak changes
from **4.368 GiB** under its native integral-direct DF-J strategy to
**101.700 GiB** when forced to consume a resident FP64 stored DF tensor.
Psi4/cuEST is **75.173 GiB** in either cohort.

Thus the earlier large GPU-memory advantage does **not** survive this stored-DF
control. This does not establish an intrinsic efficiency ranking: GPU samples
are lower bounds and include allocator reservations/workspaces. Route evidence
for the PySCF stored PBE arm records zero integral-direct J calls, 31 stored
DF contractions and an FP64 tensor of shape `(3735, 2411583)`,
72,058,100,040 bytes (67.109 GiB).

## Stored DF → stored DF: Protein83

Times are fresh-process launch-to-reap medians, **including imports**, setup,
SCF, validation, result writing and teardown. RSS is the maximum measured
worker high-water mark across the three repeats. GPU values are maximum
observed whole-device usage across those repeats, not exact allocator peaks.

| Method/basis | Psi4 s | PySCF s | Psi4 RSS GiB | PySCF RSS GiB | Psi4 GPU GiB | PySCF GPU GiB |
|---|---:|---:|---:|---:|---:|---:|
| SVWN/cc-pVTZ | 16.575 | 13.972 | 2.565 | 1.749 | 21.370 | 62.815 |
| SVWN/aug-cc-pVTZ | 39.249 | 25.613 | 4.921 | 1.751 | 75.173 | 101.700 |
| PBE/cc-pVTZ | 17.675 | 15.973 | 2.564 | 1.750 | 21.370 | 62.815 |
| PBE/aug-cc-pVTZ | 40.218 | 30.527 | 4.454 | 1.751 | 75.173 | 101.700 |
| M06-L/cc-pVTZ | 21.125 | 23.775 | 2.566 | 1.759 | 21.370 | 62.815 |
| M06-L/aug-cc-pVTZ | 53.823 | 51.784 | 4.454 | 1.778 | 75.173 | 101.700 |
| B3LYP/cc-pVTZ | 18.125 | 18.374 | 2.565 | 1.751 | 24.935 | 62.815 |
| B3LYP/aug-cc-pVTZ | 40.429 | 37.329 | 4.469 | 1.753 | 75.067 | 101.700 |

## Basis, grid and numerical checks

Protein83's live orbital dimensions match across engines:

| Basis | NBF, both engines | Retained orbitals, both engines |
|---|---:|---:|
| cc-pVTZ | 1786 | 1786 |
| aug-cc-pVTZ | 2806 | 2805 |

Each eligible Protein83 case uses the same actual nonzero quadrature:
**1,624,693 coordinate/total-weight pairs**. Psi4 stores 1,879,950 points;
PySCF stores 1,880,064, including additional zero-weight padding.
The actual coordinate/weight-multiset and overlap-spectrum/rank proofs passed;
equal point counts alone are not the acceptance criterion.

This is the exported **cuEST-source mesh**, **not the smaller native default**.
The request to use the smaller PySCF native grid inside cuEST remains open.

Across the 16 completed Protein83 comparisons, absolute median-energy
differences range from about 1.1e-9 to 2.47e-8 Eh. Peptide PBE/cc-pVTZ and
PBE/aug-cc-pVTZ agree within 6.8e-10 Eh. Peptide WB97X/cc-pVTZ and
WB97X/aug-cc-pVTZ retain approximately 8.98e-6 and 6.23e-6 Eh differences,
respectively, in both DF strategies. The latter still merit numerical review;
same route and same grid do not imply algebraically identical DF metrics.

Device and process NVML queries are separate and may miss different short-lived
transients; some short peptide process/device observations differ accordingly.
All charts label GPU measurements as sampled, retain RSS separately and avoid
claiming exact GPU high-water marks.

## Provenance and pending direct route

- Job: 14076480, `gts-cs207/gpu-h200/inferno`, NVIDIA H200.
- Harness: `267e5b7c96d024276fdb46e14575d087d13f7ee9`.
- Compiled core: `3ecc656deda2332abfb12d88542716f3c5fd69b6`,
  SHA256 `13b208c8fb8fec1e3aba075705af7d9ef8d6709d2b5661951774422b4c1de18c`.
- Both phases report unchanged runtime pins.
- Remote artifacts:
  `/storage/project/r-cs207-0/awallace43/runs/gpu-dft-compare/20261015-bohr-recovery-267e5b7-r1/control`.
- Local raw JSON evidence:
  `.pi/gpu_compare/bohr-14076480-evidence` (224 records).
- Page generator: `devtools/benchmarks/gpu_compare/control_report.py`.

The newly implemented cuEST 0.2.50 **four-center direct** adapter is separate:
build 14080574 passed; free H100 correctness job 14081007 is queued.
Neither that new core nor H100 timings are included in the H200 DF results.
