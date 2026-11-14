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

## Verified first fix: Blackwell job 15611462

Build **15603125** completed in 11m09s, with 391 core tests and the CPU
sealed-admission proof passing. Exact build/helper pin:
`324a332974574c216a096428c9460f3d257f5e88`.
The matching GPU allocation **15611462** completed in 30m15s on embers with
exit 0 and no TERM. Water, both sets of three peptide pairs, profiles, warm
diagnostics, and `runtime-ab.json` all passed. This is a same-node/GPU A/B,
not a comparison assembled from unlike hardware.

Remote output:
`/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/sapt0-perf-blackwell-324a332974-01/job-15611462/`

Retrieved evidence:
`/home/awallace43/data/sapt0-blackwell-15611462/`.

| Metric | Fresh-byte control | Sealed candidate | Old/new |
|---|---:|---:|---:|
| Median GPU dispersion | 232.6136 s | 6.7227 s | 34.601x |
| Median GPU energy | 247.1586 s | 21.3225 s | 11.591x |
| Median whole GPU subprocess | 250.9617 s | 28.0973 s | 8.932x |

The improvement is not merely a shifted timing boundary. The candidate's
whole subprocess includes early sealed preparation, which took 2.247 s in
GPU repeat 0. Its owner full-hash count/bytes stayed **20 / 2706990506**
before and after energy; live checks grew from 83 to 21895 and mapped checks
from 20 to 4601. Seals/mapping checks were retained, not disabled.

| Candidate term (three-repeat median) | CPU (s) | GPU (s) | CPU/GPU |
|---|---:|---:|---:|
| elst | 0.0007999 | 0.0013233 | 0.604x |
| exch | 0.130693 | 0.053359 | 2.449x |
| ind | 2.118232 | 0.780531 | 2.714x |
| disp | 3.785626 | 6.722744 | 0.563x |
| delta_hf | 2.615267 | 10.729010 | 0.244x |
| total | 10.887706 | 21.322510 | 0.511x |

Do not compare this cold delta-HF window directly with the earlier 2.54 s
Blackwell pilot: the matched A/B explicitly disables persistent CUDA disk
caching for both contracts. The warm diagnostic's delta-HF drops to 1.360 s,
but warm dispersion remains 6.620 s. Warm total is 10.888 s; it is a single
diagnostic, not a three-repeat speedup.

Candidate largest raw CPU/GPU peptide discrepancy is 2.719e-9 Eh; total
error is 1.148e-10 Eh. The stricter old/new GPU agreement also passed.
Work remains 468 shells, 16 DFMO batches and 64 dispersion blocks.

### Remaining bottlenecks, not yet fixed

GPU repeat 0: plan build 1.920 s (shell admission 1.687 s), blocked dispersion
0.620 s (DFMO identity 0.052 s). The separate profiled sample reports two SDK
`verify_files` calls totaling 2.803 s, including 2.718 s in `canonical_headers`,
and one NVRTC `_compile` at 1.382 s. Profiled inclusive windows overlap and
include instrumentation overhead; do not sum them as disjoint device stages.

The catastrophic regression is fixed, but cold peptide GPU dispersion is
still 1.78x slower than CPU. Next work targets repeated SDK archive parsing,
per-session compilation and mapping-parser overhead while keeping the same
proof and numerical contracts. No further performance claim is made here.

## Follow-up candidates under host validation

1. `canonical_headers` now reuses only immutable, content-derived header
   metadata. Every call still hashes the complete archive first and validates
   its expected SHA/size; `verify_files` still rechecks every installed header,
   inventory, native identity and binding source. No pathname/mtime admission
   is cached. This removes duplicate decompression, not the first parse or
   full-byte archive checks. The regression was red with two decompressions
   for two identical admissions and checks caller-mutation isolation plus
   same-size changed archive/header rejection.
2. Sealed mapping parsing memoizes repeated lexical inode/device/name fields
   only inside one newly read snapshot. It returns every segment, including
   unrelated files and non-executable segments. No map snapshot or successful
   identity verdict is reused across checks. Its regression was red with five
   conversions of one inode per snapshot and requires fresh parsing next time.
3. Submission now rejects a sealed helper checkout not authenticated by its
   exact build proof and checks runtime-support source compatibility as well.

These candidates change identity/runtime-support Python, so they require a
new coherent build and numerical/performance run. The verified 324a332974
build and 15611462 evidence remain unchanged. NVRTC compilation has not yet
been cached or removed; its cold cost remains explicitly measured.

Host validation passed **393 core/identity tests, 194 Phoenix helper tests,
and 46 Cachy helper tests**. The mapping parser produced identical full-row
output in a synthetic 530-row host fixture: median time for 200 snapshots
fell from 0.3045 s to 0.1878 s (1.62x). This is a parser microbenchmark,
not GPU speedup evidence. New whole-build and GPU qualification are next.

The user selected **wait for free embers slots**, not paid validation.
At that decision all 50 slots were occupied; the monthly account was already
over-reserved. No inferno build or GPU follow-up is authorized. Recheck capacity
when a currently running embers job ends; do not cancel or alter other jobs.
