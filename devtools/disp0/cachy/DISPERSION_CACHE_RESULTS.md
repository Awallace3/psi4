# Native Cachy dispersion cache: parity, work reduction and paired timings

**PASS_NATIVE_DFMO_CACHE_ONLY**, 2026-10-26.
This is actual cuEST/CUDA execution, not a CPU shim or an uploaded-factor
producer. It is not whole Psi4/F-SAPT, physical basis/SCF, protein-admission or
H200 qualification.

## Provenance

- Source/helper/optimization:
  `643c2879a3b02ef39e7ed1bbd9713d2691c27857`.
- Exact old backend:
  `c54d98de7f02ce13c29b15e56bd8a68b408867f4`.
- Both arms use the same actual native plan, device, runtime, raw input spaces,
  correction arrays, energies and CUDA kernels. Only the backend orchestration
  and its explicit packed cache differ.
- GPU: RTX3090/compute86,
  `GPU-d36fd194-bdd6-3d53-96c9-6a67e3225eac`.
- Native cuEST **0.2.50, FP64 mode1**, MATRIXPOWER1, relative conditioning1,
  cutoff1e−12, Coulomb fraction1, long-range fraction/omega0, pair threshold0.
  Actual parameter/math-mode/version readbacks and exact loaded-file
  inode/device/SHA256 admission passed; no identity check was disabled.
- Raw run root:
  `/home/amwalla3/gits/psi4-disp0-cachy/runs/dispersion-cache-643c2879a3-01`.
- `result.json` SHA256:
  `c59c23985174515a9e4c256fad17c647c2b47058b2e53a020e9d7054aa282b13`.
- Baseline backend SHA256:
  `47d1dd080f3a9731137a19f8deb03ed8aac0548434557fc3a6fb05942c7bb6ab`.
- Candidate backend SHA256:
  `fbda897fc99fbe81e101fde78c6aad49fac228c294c41cc268578b8980234924`.
- Compiled PTX SHA256:
  `364a5b357adc29d9aba93e2b5ea5a0720e7486d59d1d13962deea58a2a61c6bc`.

## What was measured

The fixture has18 spherical primary functions,32 spherical auxiliary functions,
two explicit Gaussian centers, occupied dimensions2/3 and virtual dimensions9/7.
The virtual block size is3: three R and three S tiles.

The primitive-Gaussian metadata is explicit and structurally verified, **not**
a claim of admitted Psi4 contraction normalization or an SCF wavefunction.
Native DFMO produces all factors used by either production arm. Separately
downloaded native factors feed independent CPU oracles only; they are never
uploaded as production factors.

Timing starts before the entire blocked backend call and ends after completion
synchronization. It includes native factor production, mandatory fresh
identity/audit work, packing, contractions, reductions, cleanup and final
reporting. Initial session/plan construction and input staging are outside both
arms. This is **not vendor DFMO kernel-only timing**.

One warmup per arm precedes three paired canonical calls, alternating arm order.
Medians use every timed sample.

| Arm | Native factor batches | Contraction blocks | Median wall |
|---|---:|---:|---:|
| Exact old backend | 12 | 9 | 5.462297 s |
| Packed-S cache | 6 | 9 | 2.944363 s |

**Old/cache latency ratio:1.85517 (1.86×).**

Raw timed samples in repeat order:

- Old: `[5.517995768, 5.462297324, 5.461103964]` seconds.
- Cache: `[2.903402511, 2.944362562, 2.966060184]` seconds.

All three S slabs fit. Although the explicit ceiling was64MiB, the cache
allocated **only18,480 bytes**, with six reuse hits. Each native batch carries
the original five coefficient pairs. The improvement removes redundant S
production; it does not reduce the three GEMMs per contraction tile.

## Additional correctness/resource cases

- New backend with cap0:12 batches,9 blocks, no allocated cache.
- Tail-only cap2,640 bytes: one cached edge slab, two hits,10 batches,9 blocks.
- Canonical and nontrivial occupied-localized pair matrices/global signed
  components agree with independent full-tensor equations.
- A second changed-input seed passes both representations with a fresh
  per-call cache. There is no cross-call cache state.
- Worst pair error against the in-process independent oracle:
  **2.03288e−20** in the fixture's numerical component values, under unchanged
  `rtol=1e−11`, `atol=1e−12`.
- Saved old/cache Disp20 pair matrices for the first seed are bitwise equal.
  Exchange-dispersion matrices differ by at most2.64698e−23 (canonical) and
  1.05879e−22 (localized); do not call every output bitwise identical.
- Separate CPU replay verified the saved NPZ hashes and used explicit
  `(r,s,a,b)` loops, auxiliary-vector dot products and `U_A.T @ M @ U_B`
  rotations instead of the probe's `einsum` equations. It checked12 saved
  matrices for seed521 and4 for seed522; maximum replay discrepancy
  **1.35525e−20**.

The actual provider recorded128 native DFMO calls across warmups, timings,
validation cases and independent-oracle factor generation. Its coefficient
host-to-device count remained0; a poisoned host-upload method also guarded
every blocked production call. All production input spaces were staged before
those calls. Python-owned CUDA contractions recorded600 FP64 GEMMs plus
packing, amplitude, occupied-permutation and reduction kernels.

## Lifetime, machine and source isolation

- Provider and native plan cleanup both completed.
- Session cleanup completed with0 tracked live bytes and0 native owners.
  Native creation/destruction ambiguity lists and retained resources were empty.
- A post-process nonblocking lock check confirmed release of the common lease.
- Actual scope readback:4CPU,12GiB RAM, swap0, tasks128. Common lock and monitored
  HOLD, a600s scientific deadline, owned process-group supervision and cleanup
  remained active. No desktop/game/FDDS process was inspected or signalled.
- Peak explicitly tracked CUDA allocations:1,041,024 bytes, **not total/peak
  VRAM**. Boundary NVML observations included desktop/runtime state
  (1371/1389MiB); these are not peak measurements.
- A fresh private executable tmpfs root contains1,757,239,987 copied bytes of
  the exact CUDA closure plus canonical SDK headers/native library. Original
  installed environments/stages/source checkouts are unchanged.
- New CUDA manifest file SHA256:
  `93cb67657d78d96c9c812cb0b2200119f898f653b59e4724fe5201040237d027`.
- No Phoenix allocation, queued configuration or installed stage was changed.

## Meaning and limits

This validates bounded packed-S reuse and measures a1.86× improvement in this
small **native blocked-dispersion path**, including its required audits. It
does not establish a whole F-SAPT acceleration, realistic large-system
throughput, or a native-kernel-only speedup. Only three timed samples per arm
were collected; no confidence interval is inferred.

The feature remains explicitly opt-in through `factor_cache_bytes`, default0.
Larger cases need a separately admitted cache/live/native workspace budget and
coherent full-stage validation. Protein83's native-plan peak and nanotube's
localization-repeat issue are not repaired by this cache. Do not remove their
guards or reuse these fixtures as production speedup denominators.
