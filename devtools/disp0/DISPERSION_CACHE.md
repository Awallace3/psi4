# Bounded packed-S reuse for GPU dispersion

## Scope

This changes `blocked_device_dispersion`, not the older host-input
`blocked_dispersion`. The experimental `dispersion0_cuda`, `fisapt_cuda` and
`fragment_terms` entry points accept the explicit additional
`factor_cache_bytes` keyword. **Default0 preserves the previous residency
policy**; this does not introduce an automatic SAPT option or enable caching
in queued/installed runs.

An example opt-in is `factor_cache_bytes=64*1024**2`. That is an **additional
packed-cache cap**, not a promise that all native plans, factors, temporary
workspaces or the entire calculation fit in64MiB.

## What is reused

Every R tile previously recomputed the same five S-side native DFMO factors,
then packed them into `fa` and `bc`. The new per-call cache retains only the
packed FP64 results, including their correction rows:

```text
fa: [2Q+2, S*B]
bc: [2Q+2, S*A]
bytes per S slab = 8 * (2Q+2) * S * (A+B)
```

One contiguous allocation is created in the outer call scope, **before** any
nested R/S scopes. A deterministic greedy budget chooses slabs; a short tail
may fit when a full slab does not. Caching is skipped for one R tile.

The first R pass fills selected slots directly through validated `_pack(out=)`
views, in the original native-call order. Later R passes reuse those slots.
Unselected slabs follow the original native production path. Raw DFMO outputs,
coefficient packing and correction windows still expire in their original
S scopes. No full-size device copy, host factor cache, second localized gauge,
cross-call memoization, allocation retry or CPU fallback is introduced.

If `nC` of `nS` slabs fit and there are `nR` R tiles, native factor batches are:

```text
before: nR + nR*nS
after:  nR + nR*nS - (nR-1)*nC
all S cached: nR+nS
```

Contraction count remains `nR*nS`. R-major/S-minor accumulation order,
canonical denominators before occupied rotation, crossed-index orientation,
signed4/−2 prefactors, FP64 arithmetic and existing native ownership/audits/
synchronization are unchanged. Cache memory is released before return, and an
allocation/native failure remains fail-closed rather than triggering a retry.

The execution report now contains a `factor_cache` record with requested cap,
actual packed bytes, selected slab count, cache hits and separate R/S native
batch counts. This is resource/work evidence, not a speedup measurement.

## CPU verification

The new call-count regression initially failed because the old backend lacked
the cache interface. The implementation then passed:

- **235 CUDA CPU-side tests**, including real orchestration against the
  fake native ABI, CPU serial execution of packing/contraction kernels, and
  unchanged native-failure ownership checks.
- **5 Cachy fixture/closure tests**, including native-experiment Gaussian
  metadata and independent direct/localized tensor oracles.

Focused cache tests cover zero/tiny/partial/full budgets, a cached short tail,
unequal spaces, canonical and localized outputs, repeated changed inputs,
single-R bypass, nonnegative integer admission, pre-native memory rejection,
output alias/type/session guards, a native failure after one cache fill,
no recomputation retry, and complete cache cleanup.

```bash
PYTHONPATH=devtools/disp0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 -m unittest discover -s devtools/disp0 -p 'test_cuda*.py' -q

PYTHONPATH=devtools/disp0/cachy:devtools/disp0 OPENBLAS_NUM_THREADS=1 \
  python3 -m unittest test_dispersion_cache_probe test_response_reuse_probe -v
```

These are not actual CUDA/cuEST execution proof.

## Cachy experiment

`cachy/dispersion_cache_probe.py` compares the exact old Git backend at
`c54d98de7f02ce13c29b15e56bd8a68b408867f4` with the cache implementation on
actual cuEST0.2.50/CUDA. It uses:

- One explicitly owned native plan and exact FP64/MATRIXPOWER/relative1/
  cutoff1e−12 policy, actual native DFMO for **both** production arms, and
  device-input coefficients with a poisoned host-upload method.
- An explicit spherical primitive-Gaussian fixture. This exercises native
  integration/transport and dispersion-cache math, **not** a Psi4 basis-
  normalization, SCF, physical F-SAPT or whole-system admission claim.
- Independent CPU full-tensor oracles built from separately downloaded native
  factors. Those downloads are test data only and never production input.
- Canonical/localized, changed-input, no-cache and short-tail cache checks,
  plus three paired whole-blocked-call timings in alternating order.
- Fresh source/run/native-byte roots; the common nonblocking GPU lock, absent/
  monitored HOLD, 4CPU/12GiB/no-swap/128-task scope, a600s child deadline and
  whole-owned-process cleanup. No installed environment/stage or Phoenix
  allocation is changed.

The small9×7 virtual/block3 fixture should retain9 contraction blocks while
reducing native factor batches from12 to6 when all S slabs fit. Any observed
latency ratio applies only to this native blocked path and includes its
mandatory identity/audit overhead, **not vendor DFMO kernel-only timing**.
Whole F-SAPT and large-system speedups require separate qualified measurements.

## Actual native result

The Cachy experiment at source `643c2879a3` **passed**. Native factor batches
fell12→6 while contraction blocks stayed9. Three-pair median blocked wall time
fell5.462297→2.944363s (**1.86×**) with canonical/localized, changed-input,
cap0 and tail-only-cache checks passing. Complete native/session cleanup and
common-lease release were verified.

See [native Cachy results](cachy/DISPERSION_CACHE_RESULTS.md) for raw samples,
pair-matrix errors, independent artifact replay, pins and scope limits. This
is not whole F-SAPT or H200 performance evidence.
