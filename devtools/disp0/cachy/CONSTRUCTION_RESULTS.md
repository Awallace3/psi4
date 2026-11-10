# Water construction result: admission dominates, not JIT threads

**Status:** `PASS_PAIRED_CONSTRUCTION_ONLY`. This is an RTX 3090 debug
diagnostic, not a numerical gate or an H200 speedup measurement.

Follow-up: [owned-runtime matched construction](OWNED_RUNTIME_RESULTS.md)
retains this baseline and compares explicit fresh-byte/owned admission on one
new coherent install. Its numerical qualification is tracked separately.

## Matched experiment

Six-atom water dimer; spherical aug-cc-pVDZ (82 AOs),
aug-cc-pVDZ-JKFIT (300 functions), aug-cc-pVDZ-RI (236 functions).
Two serialized fresh processes, ordered JIT threads **1 then 4**, one sample
each. Native parameter readback confirms the requested counts. Both use the
same basis-file hashes, geometry, fingerprints, FP64 mode and DF-plan policy.

Each process constructs independent JKFIT/RIFIT native contexts on one CUDA
session, observes their coexistence, then unwinds RIFIT, JKFIT and session.
There is no upstream SCF context, DFMO compute, or F-SAPT energy call.
Fresh per-arm CUDA cache paths do not prove a universally cold vendor cache.

## Measured host wall time

Seconds, rounded:

| Interval | JIT 1 | JIT 4 |
|---|---:|---:|
| Complete child, including bootstrap/admission | 155.560 | 155.073 |
| Session/construction/admission/cleanup interval | 149.840 | 149.337 |
| JKFIT owner build | 77.321 | 77.402 |
| RIFIT owner build | 66.339 | 65.815 |
| Sum of the two disjoint owner builds | 143.660 | 143.218 |
| Per-object admission blocks within those builds | 141.461 | 141.021 |
| All native creation calls within those builds | 0.167 | 0.175 |
| Separately timed creation synchronization | 0.036 | 0.035 |
| Native context creation, both contexts | 0.0166 | 0.0148 |
| Native DF-plan creation, both plans | 0.0765 | 0.0717 |

Rows overlap; do not add them. Native creation-call timing excludes the
separately recorded synchronization and is **not CUDA-event kernel time**.

The disjoint per-object admission blocks account for **over 98.46%** of
owner-build wall time in both arms. There are **156 JKFIT + 132 RIFIT = 288**
such blocks per arm, including the extra workspace-query admission checks.
This lower bound excludes combined context/FP64 and final-sync/identity
intervals, which also contain audits. Admission includes live checks,
mapped-byte hashing, ABI/symbol checks and their bookkeeping: this experiment
does not separately assign every admission second to SHA256 alone.

The matched source's per-object path calls fresh whole-DSO byte auditing for
each shell/parameter/basis/pair/plan object. JKFIT's 104 auxiliary shells alone
spend **50.717 s** in admission but **0.0041 s** in native shell creation in
the JIT-1 arm. This identifies the measured work to address, rather than
attributing the construction window to opaque vendor kernels.

The approximately 0.3% difference in total setup is **not an established
improvement** with one sample per setting. JIT threads do not explain the
dominant cost on this control. Keep the compatibility default at **1**.

## Cleanup and memory

Both processes report complete native/session cleanup, zero live tracked
bytes, zero native owners and no ambiguous native outcomes. Peak explicit
device allocations are **33,757,184 bytes (32.19 MiB)** in both arms.

These explicit allocations exclude opaque native/driver memory. CUDA free
memory observations are samples, not peaks; desktop co-tenancy can change
the baseline. The declared opaque reserve was unchanged at 512 MiB. This
experiment does not establish whole-workflow peak-memory admission or justify
changing the reserve. No GPU OOM occurred.

## Consequence for the performance work

Do not increase the default JIT count, share contexts, enlarge caches or remove
identity checks on the strength of this experiment.

The next target is the **identity-admission lifetime/interface**. Any reduction
in redundant per-object whole-library audits needs an explicit owned-loader /
immutability / invalidation contract, tamper/failure regressions, and unchanged
numerical and cleanup gates. Merely memoizing the present check or deleting it
would not satisfy the current contract.

Producer subphase timers are implemented but were not exercised by this
construction-only experiment. The measured H200 whole-call regression remains
unresolved; no Phoenix job was launched.

## Provenance

- Scientific install/source:
  `a13849f8b066752e8f1df999321047136ae9f8da`.
- Qualified launcher helper:
  `b71d3c3e3bfcac2828092a4ea37f82de7423d4c7`.
- Runtime config (`private-cpu-qual-04`) SHA256:
  `a4d3f107101949473985277a0a50100e2c832668dc629cc1b5a25b418b1f110c`.
- Water geometry SHA256:
  `67ca57effd611a5ca3d3199ae8da475445aa0d3a086f49fe5354f47677cfa212`.
- Raw comparison JSON SHA256:
  `f7bc5f7778f0f37fc53ef190d47b6e80e378ddc77698a806ac18f1ae0bb1c00c`.
- Raw results:
  `~/gits/psi4-disp0-perf-cachy/runs/construction-water-02/`
  (comparison JSON, per-arm launch/result JSON and stdout/stderr).
- Hardware: RTX 3090, compute_86, driver 615.78.08,
  UUID `GPU-d36fd194-bdd6-3d53-96c9-6a67e3225eac`.
- Resource bound: 4 CPU quota, 12 GiB RAM, no swap, 128 tasks,
  common nonblocking GPU lease/HOLD, 600 seconds per construction child.

The earlier compilation OOM and launcher/qualification failures remain
separately recorded in [CONSTRUCTION_PERFORMANCE.md](CONSTRUCTION_PERFORMANCE.md);
none contributes a timing sample.

After both processes completed and cleanup was verified, the common GPU lease
was acquired nonblockingly and both experiment-owned executable-copy trees
(`private-cpu-qual-03` and `-04`) were removed after exact inventory/byte checks.
The cleanup receipt is
`~/gits/psi4-disp0-perf-cachy/runs/executable-copy-cleanup.json`.
Durable source/build/install/dependency inputs and all run logs/JSON remain.
The two executable runtime configurations are now **expired** and must be
freshly qualified before reuse; historical pass artifacts were not rewritten.
No test GPU job remains running.
