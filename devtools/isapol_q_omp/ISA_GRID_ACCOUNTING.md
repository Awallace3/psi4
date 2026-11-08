# ISA grid accounting v1 (unqualified)

This child of `2fe7a711dd915e102fe6976f44d22571064aa31e` changes **accounting
only**. It does not change native ISA/SCF/grid/Q arithmetic, iteration limits,
Drho32, thresholds, memory/I/O/work caps or thread policy. No production
profile/cohort/dispatch is activated. Historical policies/FILES/receipts stay
byte-identical and correctly refuse these new sources.

## Source proof and narrow fast path

* `libisapol/isa_grid.cc`: `n_shell = radial_points-1`; each atom appends every
  radial/angular combination. `apply_becke()` changes weights, not row count.
* `libfock/cubature.cc`: the selector returns the smallest available Lebedev
  cardinality >= the request, clamped by IsaGrid to5294. The adjacent entries
  are194 and230, so request200 gives230.
* `isapol_native_partition.py:native_partition` verifies site count/origins,
  rejects ghosts/dummies and copies all four numerical grid options into
  `core.IsaGrid`. Every ISA site uses the full molecular grid.
* The pure shared helper tightens ONLY radial100/spherical200/smoothing3/
  scaling1/native-tabulated-radii/all-sites-unscreened. Thus the closed recipe
  has `sites*99*230`, not the old envelope `sites*99*406`.
  Everything else retains the old `min(5294,2*request+6)` envelope.
  Response100/200 and response-AUX integration400/2030 remain distinct.

No runtime point count, geometry, density, denominator, shape or AUX is
recomputed by this helper. Resource planning is still before native execution.

## Complete accounting, not parallel speed credit

`isa_resource_plan` retains every original term, including full-grid cached
atomic collocation even if uncached, all120 iterations, Drho32, MAIN/Q/history,
256MiB scratch and bounded Q payload/control memory (<=827392 extra bytes).
Only ISA-grid-row terms shrink. No division by threads or empirical speedup.

The new shared helper additionally charges staged-Q preparation/control at
**all** thread counts, including1/2 where the original arithmetic path avoids
that payload. With `T=(rank+1)^2`, `V=min(8,ceil(naux/64))`, and native block
`B=min(4096,floor(1048576/naux))`, the added operation allowance is:

```
sites * [points*(64+32*T+32*T*V)
         +max(1,ceil(points/B))*(4096+256*T*V)]
```

The existing `4*points*sites*T*naux` multiply/add-column charge stays intact.
The new terms cover flag/prefactor initialization/stores, repeated tiled
point-loop checks/loads and bounded task/worker setup/scans. The docstring
itemizes the per-loop allowances; no screening or partial-batch refund.
This is conservative source-level arithmetic accounting, **not a hard bound
on allocator/OpenMP runtime/vendor operations, RSS or wall time**. Existing
ledger exclusions remain unchanged.

## New seam, old policy unchanged

`resource_plan_isa_grid_v1.py` loads the SHA-pinned historical ledger, adjusts
only its single ISA stage and recomputes total work, stage work, numeric peak,
admission, cumulative-work/numeric refusals. All independent PFIT, packed
replay, quadrature and I/O gates remain intact. It loads the shared pure
helper from pinned bytes and extracts ONLY the existing verified scalar Q
workspace function (AST), never importing Psi4/NumPy/distribution.

Its source binding includes the unchanged grid selector/producer/recipe and
the new helper/runtime plus tested Q kernel. Results have
`policy_scientific_commit=None` and explicitly unqualified prediction status.
This is not a receipt, descriptor admission, installed qualification or policy
allowlist update. A later reviewed closed producer must bind the final commit
and actual installed implementation. Old25 results keep original provenance;
none of these calculations relabel them.

## Tests and pending compute validation

Pure stdlib only:

```bash
python3 -I -B devtools/isapol_q_omp/test_isa_grid_accounting.py
python3 -I -B devtools/isapol_q_omp/test_workspace.py
git diff --check
```

The new tests compare the real scalar runtime function with the new ledger,
reproduce the authorized17-site descriptor dimensions/refusal, check exact/
one-short work/byte/I/O boundaries and unchanged128T cap, general-recipe
fallback, native-table envelope, source tampering and old-source rejection.
The original workspace guard now permits ONLY the separately tested
`isa_resource_plan` body to differ; every other distribution byte and the
native high-thread tail remain frozen.

**Parent compute only, not locally run:** add
`tests/pytests/test_isapol_isa_grid_accounting.py` to the existing84-case native
union: **93 expected =34Q+18performance+32partition-response+9grid-row cases**.
The9 new public-API cases check actual native counts, atom offsets and general
rounding/clamping, without SCF. Also run existing distribution admission/
exact-plan/one-short molecular tests in authorized compute, without altering
their science tolerances. Fresh installed qualification is still required.

A minimal next phase can reuse the existing genuine build/qualification
engine into a fresh `ccasp-native-s66-qomp-<SCI>-build001/env` namespace, fitting
the existing `native_psi4.sh` pathname contract. Use unchanged approved original
configure bytes4356/SHA896949860e3eb012e07b2ec45556ab4580d87e36de094ca5567d800149803380,
new source/identity manifests and strict full installed-source mapping. Never
overlay8558, copy qualification claims or rename READY_BUILD_NOT_QUALIFIED.
No build/job/production profile change is authorized by this patch.
