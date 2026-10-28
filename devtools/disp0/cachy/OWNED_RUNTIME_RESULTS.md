# Owned runtime: matched water construction diagnostic

**RTX 3090, debug only. Not an H200 speedup or numerical qualification.**
Construction passed in both modes. The small restricted-water numerical gate
also passed; this is not a broad backend or Phoenix release.

## Matched experiment

Clean science/helper pin: `e5a16ad22c1dd5a514a898fc2ec80b0f6d493fb1`.
Whole install: `~/gits/psi4-disp0-perf-cachy/install-owned-e5a16ad22c`.
Both fresh CPU qualifications passed RHF/STO-3G at
`-1.1167593073781577 Eh`. The measured `install/` baseline remains untouched.

The six-atom water dimer has 82 spherical aug-cc-pVDZ AO functions,
300 JKFIT functions and 236 RIFIT functions. Geometry, installed basis bytes,
basis fingerprints, plan parameters, coefficient policy, native version/FP64,
host/device budgets and native ownership schedule match across modes.
The mode is the intended difference. Each mode ran fresh processes in JIT
order `[1,4]`, serialized under the common GPU flock/HOLD and
4-CPU/12-GiB/no-swap/128-task limits.

No SCF, DFMO production or energy calculation occurs in this construction test.
Each arm has its own CUDA cache directory; vendor-global cache state is unknown.
There is one sample per setting, and fresh-byte mode ran before owned mode.

## Host wall time

| Inclusive scope (seconds) | Fresh, JIT1 | Owned, JIT1 | Fresh, JIT4 | Owned, JIT4 |
|---|---:|---:|---:|---:|
| Entire child | 152.722 | 40.232 | 153.425 | 41.179 |
| Runtime admission | 1.881 | 3.069 | 1.866 | 3.090 |
| Construction + cleanup | 147.089 | 33.393 | 147.777 | 34.316 |
| Combined owner build | 141.083 | 28.719 | 141.732 | 29.566 |
| Disjoint per-object admission blocks | 138.960 | 27.935 | 139.523 | 28.758 |
| All native-create calls | 0.180 | 0.098 | 0.189 | 0.101 |

Inclusive rows overlap; do not sum them. The owned adapter trades a one-time
leased byte admission for repeated live-identity checks. It reports **17 full
hashes / 1,756,549,808 bytes**, 3,104 live checks and 2,677 mapped-byte receipts
per arm, with no invalidation. Bootstrap/whole-build qualification reads are
outside these adapter counters.

The observed reduction is host auditing, not faster FP64 kernels. JIT4 still
does not establish an improvement. Admission remains about 97% of owned
owner-build time: profile mapping traversal and live checks next, rather than
guessing that native computation is now the bottleneck.

## Memory and lifetime

All four arms completed native/session cleanup: zero native owners, zero
tracked live bytes, and **33,757,184 bytes (32.19 MiB)** peak tracked device
allocations. The 512 MiB opaque reserve did not change.

At JIT1, the entire fresh/owned device-free-memory sample sequences matched
exactly, including 23,532,142,592 free bytes while both native contexts coexist.
These are checkpoint observations, not an opaque-native peak or whole-workflow
memory bound. Desktop co-tenancy remains allowed. No GPU OOM occurred.

## Provenance

Under `~/gits/psi4-disp0-perf-cachy/runs/`:

| Artifact | SHA256 |
|---|---|
| `owned-fresh-cpu-02/runtime-config.json` | `6d6c3c7cf0a242514576e23fc38e870216a13e537a7f2625335907491a8db2d2` |
| `owned-owned-cpu-02/runtime-config.json` | `911baf5dfeff59e69ef82256ed90cff5afc4255fac539c264e6040e5dc37f75e` |
| `construction-fresh-water-03/comparison.json` | `e93746915dea4a5a570ea04a8a191f88c46a366d79571e9af71809362b49a2c3` |
| `construction-owned-water-03/comparison.json` | `e56cf2c25ed3ef3cae0a35d2990b5a15479c890cbc15d2e69bf7024a5ae65d17` |

Geometry SHA256:
`67ca57effd611a5ca3d3199ae8da475445aa0d3a086f49fe5354f47677cfa212`.
Device UUID: `GPU-d36fd194-bdd6-3d53-96c9-6a67e3225eac`;
driver 615.78.08; `compute_86`; canonical cuEST0.2.50.
Both modes used identical generated PTX:
`364a5b357adc29d9aba93e2b5ea5a0720e7486d59d1d13962deea58a2a61c6bc`.

## Independent numerical gate

`runs/owned-full-water-gate-01/gate-result.json` passed
`PASS_SMALL_CACHY_ONLY`; SHA256:
`eb2858382faca06c724eb0050e8147e673319d5aedf0d2280027c2587d7bc2f4`.
All ten fresh processes returned0: matched CPU/GPU and independent CPU replay
for unscaled, scaled and displaced geometry, followed by a fresh scaled GPU
repeat. All four GPU cases executed native cuEST/cuBLAS/custom-kernel work;
this CLI gate has no GPU-skip success path.

All **23 fields**, signed terms, nonunit scaling, changed energy and standard
output files passed. Maximum independent same-gauge energy-matrix/pair errors:

| Input | Maximum absolute error (Eh) |
|---|---:|
| Unscaled | 6.033e-13 |
| Scaled | 2.687e-12 |
| Displaced | 3.902e-12 |

The fresh repeat retained the original 1e-9 comparison thresholds and orbital
sign/permutation-only policy; its largest energy-matrix error was 3.476e-12 Eh.
Every GPU case retained the same 17 admitted hashes without invalidation and
completed native/session cleanup. The unscaled case performed 1,040 strided
batched GEMMs and 3,619 ordinary GEMMs, plus actual custom-kernel launches;
tracked peak device allocations were 88,794,624 bytes.

**Whole-call performance is not solved.** Diagnostic whole-energy calls still
took 155.5–158.6 s. The runtime reported about 232,000 live checks per GPU case.
These are correctness-run observations, not matched whole-workflow speedups;
accepted timing samples remain0. Preserve this residual overhead when choosing
the next profile-driven optimization.

## Remaining host overhead: CPU-only profile

`runs/owned-cpu-audit-profile-01` used the same qualified install and owned
library set, with `CUDA_VISIBLE_DEVICES=""`, no CUDA session/plan and no energy
calculation. This is a separate mechanism diagnostic, not a GPU timing sample.
The adapter retained exactly17 hashes throughout.

Unprofiled warm samples:

| Operation | Calls | Total wall time | Wall per call |
|---|---:|---:|---:|
| Full `check_live()` | 1,000 | 0.3611 s | 361.1 us |
| Full factory mapping audit | 20 | 1.9242 s | 96.21 ms |

Separate cProfile samples identify concrete follow-ups:

- 1,000 live checks invoke 17,000 file checks and **36,000 caller checks**.
  Caller validation repeats at each file boundary; constructing `Path` objects
  and repeating signal/PID/thread queries are material costs.
- 20 factory audits call `mapped_rows`40 times, resolving **46,040 pathnames**
  and invoking `lstat`415,400 times. Mapping traversal accounts for about92%
  of that profiled interval; it is not native computation.

Do not add these microdiagnostic times to the inclusive workflow timers or
extrapolate an H200 gain. The next bounded change should remove redundant work
inside a single live validation/mapping snapshot while preserving its entry,
exit, inode/device and revocation obligations—not introduce a cross-admission
mtime/mapping cache or drop identity checks.

The optimization remains explicit/opt-in. No cache-budget, native-context
sharing, arithmetic or reserve change is included, and no Phoenix job was run.

## Retired executable copies

After all workers exited, the common GPU lease was acquired and all four
experiment copy trees were verified byte-for-byte (77 files each) before
removal. Only those private `/tmp/psi4-disp0-perf-owned-*-cpu-0[12]` trees
were removed: 7,028,959,948 copied bytes in total. The exact whitelist and
verification hashes are recorded in
`runs/owned-executable-copy-retirement-01.json`.

The four runtime configurations are now **expired historical evidence**, not
reusable launch inputs. Requalify fresh executable copies before another run.
All durable sources, whole installs, build caches, runtimes, archives, inputs,
failed attempts, numerical results and CPU profile artifacts are retained.
No GPU job remains running.
