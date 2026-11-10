# Grid versus basis GPU-memory diagnostics

These are **separate diagnostic runs**, never speedup measurements. The speed
campaign remains pinned to its original, sampler-free harness and scientific
binary. Do not change a source/build used by an active validation job.

## Questions and controls

1. Hold geometry, AO basis, auxiliary basis, backend and GPU constant; compare
   `fixed99`, `quarter99`, `third99`, `fixed75`, `coarse75`.
2. Compare initial- versus final-grid `form_G` phases and the grid transition.
   An unchanged final grid still has to fit; fewer early points need not lower
   whole-job peak residency.
3. Hold geometry, auxiliary basis and grid fixed; vary the primary basis.
   Changing molecule and basis together does not isolate NBF scaling.
4. Distinguish DF plan construction, steady JK, XC and simultaneous live
   wavefunctions. Record **NBF and Naux**, not NBF alone.
5. Keep host RSS, per-process GPU residency and device-wide free memory distinct.
   A host-memory budget is not a device-memory cap.

Nominal unpruned point ratios (not retained grid counts or GPU-memory ratios):

| Grid | Reference | Points/reference |
|---|---|---:|
| 50×302 initial | 99×590 final | 0.258517 |
| 65×302 initial | 99×590 final | 0.336073 |
| 75×302 final | 99×590 final | 0.387776 |
| 35×194 initial | 75×302 final | 0.299779 |

## Current evidence versus hypotheses

The prior report `~/docs/saptdft-memory-scaling.html` documents protein157
job 13055047 failing in dimer DF setup: approximately 101.339 GiB already
committed and a further 62.062 GiB requested. It attributes that request to
the **plan-construction temporary**, not XC. The report's fitted temporary
factor (~0.63 times persistent plan) is historical evidence, not an exact
coefficient promised by the current library.

A useful structural model is:

```
DF plan ~ 8 [Naux * screened_AO_pairs + Naux**2] bytes
GPU peak = max over time(context + all live plans + current workspaces
                         + XC/grid buffers + other library allocations)
```

Do not add maxima from different phases. Do not infer that smaller grids fix
a DF-setup OOM. Conversely, do not claim grid memory is negligible until its
residency is measured on the current build. The JK and XC source audits
below are complete; new GPU measurements are pending.

### Current JK source audit

At scientific checkpoint `10979f447`, the wrapper distinguishes:

- Cached primary/auxiliary GPU basis objects:
  `psi4/src/psi4/libmints/basisset.cc:1312–1382`.
- Persistent AO-pair and DF plans, and their separate creation temporaries:
  `psi4/src/psi4/libfock/cuESTJK.cc:120–187`. Neither query takes an XC grid.
- Per-call density/J/K matrices: `8*NBF**2*(2*do_J + do_K)` bytes.
- Per-call occupied coefficients: `8*NBF*nocc_max` bytes for symmetric
  builds, twice that for nonsymmetric builds; buffers are reused across
  densities: `cuESTJK.cc:339–379`.
- J/K scratch: the maximum returned by the J and individual K workspace
  queries, not their sum: `cuESTJK.cc:391–449`.
- No negotiated device budget in `cuESTCommon.h:74–135`, and
  `cuESTJK::memory_estimate()` returns zero (`cuESTJK.cc:303–305`).
- An **8,000,000,000-byte input** to the exchange workspace query/compute
  path (`cuESTJK.cc:386–389`), not a cap on the entire calculation.

The historical `0.63*plan` factor and slice-based working-memory model must
therefore remain empirical historical models. `CUEST_DFK_SLICES/MODULI`
are precision/emulation controls, not demonstrated total GPU-memory caps.
In this campaign mixed precision is explicitly **off**.

SAPT shares JK objects between monomer wavefunctions
(`sapt_proc.py:752–818`), so two wavefunctions do not imply two DF plans.
However, initialization of a supplied JK still reconstructs its plan, and
changing hybrid weights for SAPT bare exchange can reconstruct it again.
BasisSet-owned device caches can outlive JK finalization. These are distinct
lifetimes that the phase trace needs to distinguish.

Opaque cuEST queried workspaces are authoritative for their byte sizes;
the simple structural `Naux*AO_pairs` model is not an exact allocator formula.
The existing JK printout omits construction temporaries and explicit
matrices/coefficients, so it is not an adequate whole-job memory ledger.

### Current XC source audit

Here `N` is the **actual** cuEST point count, `B` the dense AO dimension,
and every array element is an 8-byte double. Fresh CUDA-LibXC explicit
grid-array totals (excluding opaque grid/plan/workspace storage) are:

| Functional path | LDA bytes | GGA bytes | meta-GGA bytes |
|---|---:|---:|---:|
| Restricted | `64*N` | `136*N` | `192*N` |
| Unrestricted | `104*N` | `272*N` | `352*N` |

Restricted GGA with GRAC is `160*N` bytes. For one million points this is
about 130 MiB RKS/GGA, 153 MiB RKS/GGA+GRAC, or 259 MiB UKS/GGA. These are
explicit-array **contributions**, not a model for total device residency.
Evidence in `psi4/src/psi4/libfock/v.cc`:

- Occupied coefficients `8*B*nocc` (sum of spin counts for UKS): 649–677.
- AO potential outputs `8*B**2` RKS / `16*B**2` UKS: 2104–2108, 4480–4488.
- Weights, densities and potentials: 2147–2204, 2241, 2471,
  4527–4589, 4659–4660, 4951–4964.
- CUDA LibXC simultaneous input, component and accumulated scratch:
  287–299, 326–348, 375–390.

The CUDA-LibXC workspace is **static process-wide, grow-only**, including
optional arrays that are not cleared when a later path no longer needs
them (`v.cc:201–250`). Finalizing a wavefunction does not release it.
Prior UKS, meta-GGA or GRAC evaluations can therefore leave more resident
memory than a fresh RKS run predicts. Fresh-process arms are essential.

GPU grid and XC plan construction each also query persistent **and temporary**
workspaces (`cubature.cc:3865–3897`, `v.cc:799–844`). Their costs cannot be
deduced from the explicit-array table. Density/potential integration uses
a **2,000,000,000-byte variable-buffer input**, not a total GPU-memory cap
or an unconditional 2 GB allocation (`v.cc:2111–2115`, 4491–4495).
No full `N*B` GPU collocation array is explicitly exposed by this wrapper;
cuEST's queries determine actual integration workspace sizes.

Normal grid switching destroys old VBase plans and grid ownership before
constructing replacements (`v.cc:738–745`, 1447–1495). It does not
intentionally retain two complete GPU grids. However, old process-wide
LibXC scratch remains live during replacement construction, and external
shared ownership of a grid could extend its lifetime. The diagnostic hooks
retain only scalars, not grids or wavefunctions.

VV10 has its own grid settings, grid and plan (`v.cc:769–780, 824–844`):
shrinking the main XC grid does not shrink those. VV10 is not part of the
PBE0-D4(I) speed workload, but matters for the general SCF helper.

Host-LibXC routes additionally copy full-grid weights/densities to the host,
transpose density data, and upload full-grid potentials. Smaller grids can
reduce both transfer and host-memory costs there. CUDA-LibXC largely removes
those transfers, but still has the process-wide device buffers above.

## Instrumentation

`grid_memory.py run` starts an external sampler and runs a Python script under
optional Python-boundary hooks:

```
python devtools/benchmarks/grid_memory.py run \
  --output /absolute/new/diagnostic-directory \
  --interval 0.2 --timeout 900 \
  --script /absolute/frozen-harness/devtools/benchmarks/grid_evidence.py -- \
  --worker --output /absolute/new/case-output \
  --system protein83 --arm coarse75 --repeat 1 --threads 8 \
  --memory '112 GiB' --variant host \
  --scientific-commit SCIENTIFIC_COMMIT \
  --expected-package /absolute/frozen-science/build_grid_host/stage/lib/psi4
```

Run this **inside a suitable GPU allocation** with the same environment,
library search path and staged Psi4 package as validated for that backend.
The diagnostic tool itself must also come from a committed, pinned harness
staged through Git. The example does not submit a job.

Artifacts:

- `manifest.json`: command, PID, job ID, requested sample interval, and
  `eligible_for_speedup_bars=false`.
- `samples.jsonl`: monotonic timestamps, actual query durations, per-GPU
  process-tree residency in MiB, host RSS/HWM snapshots and current phase.
- `boundaries.jsonl`: JK initialization, SCF initialization/finalization,
  RHF/UHF `form_G`, and Python-requested XC grid changes. Wavefunction
  metadata includes NBF, Naux, grid state, and host grid points/block extents
  where available. GPU grids bypass host block construction: their host
  counters are placeholders, not GPU point counts. These traces record
  `grid_points=null`, `grid_points_source="gpu-count-not-exposed"` and null
  host block/collocation sizes instead of treating zero as an actual count.
  No wavefunction references are retained.
- `summary.json`: sampled overall and per-phase peaks; unavailable accounting
  and failed samples are explicitly reported, never substituted with zero
  or device-wide totals.
- `console.log`: child stdout/stderr, including allocation-failure messages.

Limitations:

- These are **sampled lower bounds**, not exact allocation high-water marks.
  Short plan-construction temporaries can be missed. The largest observed
  sample interval is recorded, not just the requested interval.
- Hooks intercept Python-visible calls only. `form_G` includes C++ internal
  JK and XC work; it does not separately identify those internal peaks.
- `nvidia-smi` and event logging perturb the run. Do not merge diagnostic
  `result.json` files into speed evidence, even when microprofiling is off.
- `/proc` VmHWM covers process lifetime; a sampled per-phase HWM is not the
  phase's independent high-water mark.
- No CUDA synchronization, allocation interposition, memory-pool trimming or
  modified scientific binary is introduced by these hooks.
- Class-method hooks need a real Psi4 smoke test in addition to unit tests;
  GPU accounting needs a compute-node check, not login-node discovery.
- The current Python API does not expose cuEST's actual GPU point count.
  Do not infer retained points from host counters or nominal radial/angular
  products. Exact GPU counts require a separate source-level query.

If sampled traces leave the binding peak unresolved, the next targeted probe
should log exact workspace descriptor bytes and allocation/free lifetimes at
the source site identified by the audit, on a **new scientific checkpoint**.
Never silently present sampled residency as an exact allocator ledger.

## Validation and staged campaign

Local validation: **16 unit/supervisor tests passed**, including unavailable
accounting, nonzero child exits, and GPU host-placeholder metadata. The earlier
15-test version also passed on Phoenix. Real CPU Psi4 switching passed with 49
recorded boundaries, including JK initialization and both initial/final
`form_G` states. This validates the hooks, not GPU accounting.

`grid_memory_run.sh` requires the separate induction-NONE scoped gate and
matching core hashes. The original validation14044088 remains **FAILED**:
both 36-test GPU suites and the ordinary/iterative-GRAC water sweeps passed,
but all full-SAPT controls reached the explicit unsupported cuEST XC
`compute_Vx`/CPKS response guard. No full-SAPT validation is claimed.
`grid_workload_gate.py` recomputes switching and 75×302 accuracy from raw
components, rejects missing/skipped tests or any unrelated full-SAPT failure,
and records input hashes and this exclusion in a new receipt. It never
changes the original job's artifacts or writes a false COMPLETED marker.

The campaign runs one diagnostic per grid arm for water, nanotube and
protein83, separately for host/CUDA LibXC (30 fresh processes). Water gates
the larger cases, and missing grid/boundary metadata is an error. Explicitly
unavailable GPU point counts are accepted; host point counts must be positive.
The aggregate is `memory-evidence.json`; diagnostic child results are
explicitly marked ineligible for speedup bars.

This first campaign holds basis and geometry fixed **within each system**.
Its cross-system NBF variation is not an independent primary-basis scan.
After locating the binding phase, a controlled same-geometry basis/auxiliary
scan can target it without blindly repeating known over-capacity systems.
