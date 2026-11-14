# Canonical GPU SAPT0 repeated-proof regression

## Measured baseline, not a kernel hypothesis

Blackwell job 15600635 passed water and three CPU/GPU peptide pairs at the
user-approved 1e-7 Eh absolute CPU/GPU criterion. Median dispersion was
3.990180716 s CPU and 233.838630306 s GPU. The immutable records remain at:

`/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/sapt0-blackwell-peptide-7d42fc2b36-01/job-15600635/`

Local retrieved JSON evidence is under
`/home/awallace43/data/sapt0-blackwell-15600635/`.

The three actual GPU phase records contain:

| GPU repeat | Dispersion (s) | Plan build (s) | Shell admission (s) | DFMO identity (s) |
|---|---:|---:|---:|---:|
| 0 | 233.8386 | 218.8103 | 211.3873 | 6.9001 |
| 1 | 235.0810 | 220.0574 | 212.5948 | 6.8749 |
| 2 | 233.5232 | 218.6376 | 211.2458 | 6.8207 |

These inclusive windows overlap: shell admission is inside plan build.
All runs have 468 shell admissions (122 primary, 346 auxiliary), 16 DFMO
factor calls and 64 dispersion blocks. Actual shell-create native calls
total only 0.0127–0.0130 s. Blocked execution is 7.36–7.44 s, itself including
the 6.82–6.90 s DFMO identity work. These are host wall windows, not CUDA events.

Cause: the canonical launcher forced `fresh-bytes-v1`, although the merged
science already supports immutable, process-owned sealed runtime admission.
The fresh path rehashes mapped cuEST/CUDA DSOs during every shell/DFMO
admission. A 4-GiB factor cache cannot cache these identity proofs.

## Red/green seam and fix

Command:

```sh
python3 -B -m unittest discover -s devtools/disp0/phoenix_validation -p test_sapt0_sealed.py
```

Before the launcher fix, this failed in the real canonical controller with:
`ValueError: This first campaign requires the existing fresh-byte runtime`.
Afterward the controller selects the early sealed bootstrap for both CPU and
GPU arms, with private aliases and the out-of-band retirement channel. A second
test verifies the actual energy-call helper invokes the whole GPU calculation
through the owner's guard and never falls back on admission failure.

This is a **launcher/runtime-policy fix**, not modified arithmetic. `psi4/`
and `psi4_runtime_support/` remain identical to merged native science
8bbbf411cb. Do not cache hashes by filename/mtime, delete audits, lower precision,
alter block/cache sizes, or reuse an old build proof to authenticate new code.
Create a new coherent whole stage/proof including the new launch helpers.
The build's existing CPU sealed-admission check must pass before GPU submission.

## Verification protocol

- Frozen source, coherent whole build and exact CPU sealed-admission receipt.
- Same Blackwell node/GPU, scientific settings, 8 CPU threads, memory and
  1e-7 Eh CPU/GPU tolerance; repeatability remains 1e-9.
- Sealed water gate, three fresh CPU/GPU peptide pairs.
- Separate cold cProfile worker and second-energy-in-one-process warm worker,
  excluded from medians. Each diagnostic must retain numerical agreement.
- `--runtime-ab`: additionally run the unchanged fresh-byte policy with three
  matched CPU/GPU pairs and identical diagnostics on this allocation.
- Both contracts explicitly configure the same CPU-only Einsums rc policy
  before scientific imports, use private HOME directories, disable persistent
  CUDA disk caches and enable the same packed-GEMM setting. These controls
  remove startup-policy confounders. The earlier historical job did not
  explicitly control all of them, so use the new same-allocation A/B to
  attribute the improvement, not just the old 234-second denominator.
- Whole subprocess time includes early sealing, admission, imports,
  calculation, serialization and exit. Energy-call time includes provider/JIT,
  audits and cleanup, but excludes early bootstrap in both measured contracts.
  Never claim moving a cost outside the energy timer removed it.
- Record unchanged owner `full_hash_calls` and `full_hash_bytes` across each
  sealed energy call; preserve live mapping/symbol/seal checks.
- `runtime-ab.json` only appears on matching same-node/GPU/science inputs,
  signed energies, and qualified samples. It includes per-phase records and
  GPU old/new ratios. CPU/GPU speedups remain in each `comparison.json`.
- No kernel speedup is inferred from cProfile or host-inclusive timers.

The frozen H200 job 15447098 is not modified by this work. It still selects the
legacy fresh-byte benchmark policy, so it is not evidence of the fixed path.
No new GPU success or speedup is claimed until the allocated A/B completes.
