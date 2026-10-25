# Explicit angular-refined full benzene candidate

Local implementation and engine-free tests only. No scientific acceptance,
native execution, remote composition, preflight or submission is established
here. The parent owns independent review, publication, staging and activation.
This is a **new own end-to-end calculation**, not grid-only integration,
tensor replay, a recovery of historical bytes, or a watchdog experiment.

## Exact effective configuration

`angular_plan.json` and `angular_profile.configuration(runtime)` bind exactly:

| Field | Original `pilot.json` | Explicit effective value |
|---|---|---|
| Q integration radial/angular | 200/590 | **200/974** |
| numerical memory bytes | 25769803776 (24 GiB) | **26843545600 (25 GiB)** |

The original profile SHA remains
`19170c907137c7883dfbceaa4a23e4b919ec2d97dac38343dc4559b8bffb0f0b`.
The effective profile is a separate canonical JSON SHA-256 identity, not the
original profile with a misleading label. It records parent SHA, both overrides,
the complete effective configuration, plan SHA, and profile/driver/job/batch/
shared-driver/runtime-source file hashes. Source binding adds actual commit/tree
and complete tools/routing hashes; runtime-manifest and derivative-admission
hashes are generated later by metadata preparation. No fabricated remote hashes.

Only two backward-compatible shared seams were added:

* `integrated_runtime.effective_profile()` optionally adds an explicit profile
  identity to source/derivative bindings. Original launchers return `None` and
  retain original binding semantics.
* `integrated_driver.capture_science` optionally accepts an explicit configuration,
  profile identity and post-pipeline completeness validator. Existing callers
  use the unchanged original configuration and export behavior.

No global pilot config or source runtime is monkeypatched. `angular_job.py`
uses `integrated_runtime.for_plan` to obtain a separate module namespace.
`pilot.py`, `pilot.json`, basis helpers/contracts, all installed scientific
Python routing, native sources and basis files remain unchanged.

## The actual physical path

`angular_job.py run` initializes the same admitted interpreter and exact
Einsums 1.1.3 retained handle, GNU/MKL loader closure and immutable native core.
It calls `angular_driver.py`, which delegates to the same integrated driver:

1. Verify fixture attribution/hash, 400 GiB free scratch, numerical threads 1,
   and production LW tolerance 1e-6. Set **Psi4 memory to 25 GiB**.
2. Admit exact native spherical MAIN414/138 shells and Cartesian AUX1146,
   including full normalized-shell fingerprints, before SCF.
3. Own PBE0/noGRAC SCF in the unchanged absolute dimer frame, with original
   SCF basis, grid75/302, thresholds and topology. Check post-SCF MAIN identity.
4. Original ISA iteration recipe/grid100/200, convergence and Drho10;
   independently construct full Q grid200/974. No changed tails, normalization,
   charge penalty, eta, rank, response approximation or physical tolerance.
5. The same `psi4.oeprop` bounded reference-H2H1 response pipeline uses original
   response grid100/200, all eleven static/dynamic Casimir nodes, rank4
   nonlocal input and strict production LW rank2 localization. Its
   `BoundedResources.max_bytes` receives the **same 26843545600 bytes** as Psi4.
   Work6.4e13, checkpoint I/O64 GiB, capacity `large` stay unchanged.
6. Original complete-cloud PFIT/static-reference penalty selection, original
   isotropic C6/C8/rank-limited C10 and full-tensor recoupled anisotropic
   properties. Rank2 C10 remains partial, not unrestricted-complete C10.
7. Before export, an additional completeness check requires all eleven
   response/LW/PFIT nodes, common static selection hash, finite strict LW
   diagnostics <=1e-6, 144 ordered isotropic and recoupled pairs, and complete
   ledger totals within unchanged work/I/O and explicit 25-GiB memory ceilings.
   This check never substitutes for native convergence, H2H1 residual,
   production LW, PFIT solved-status, tensor/reciprocity or recoupling gates.

The unchanged `DiagnosticSink` observes the **real** live F/Q/B/X and captures
the full lossless pre-LW input and signed charge-axis diagnostics before the
one strict factory call at each reached node. Any refusal propagates; no retry,
projection, rescaling, renormalization or replacement tensor. Partial capture
is **not physical acceptance** and never allows properties export. See
`INTEGRATED_CAPTURE.md` for the capture contract and output layout.

Physical success requires the complete original pipeline, valid exported
properties, all additional completeness checks, final exact loader/runtime/
source closure, unsignalled zero child exit and complete payload/batch
receipts. A grid-only error reduction, one passing node, complete capture,
or SLURM `COMPLETED` alone is insufficient.

## Basis evidence is derivative, not rebound

Original private probe path/SHA, original driver commit/tree/files and original
profile identity are still checked in their entirety by `prior_probe()`.
`derived_document()` preserves that evidence under `original_probe`; it adds
the new effective identity, driver/source binding and composed runtime SHA
separately. It explicitly says this is **not a new measurement**. Actual basis
metadata is rebuilt/compared in the single future allocation before own SCF.
No private geometry is stored in Git.

## Engine-free resource regression

`test_angular_full.py` executes the unmodified `_width` and `isa_resource_plan`
function ASTs with length-only grid and shell-metadata stubs. No engine/grid/
matrix calculation. Metadata: ns12, rank4, Q rows300, MAIN414/138 shells,
response/density AUX1146, atomic widths6x109/6x49, shape120, it120, Drho10;
iteration bound12x99x406=482328, actual response rows273240,
native state plan106842112 bytes.

| Accounting | 200/590 | 200/974 |
|---|---:|---:|
| Q points | 1408920 | 2325912 |
| standalone ISA numeric bytes | 23398743536 | 26333117936 |
| shared holdings bytes | 208454016 | 237797760 |
| total numeric bytes | **23607197552** | **26570915696** |
| ISA work | 43607749049472 | 44868796447872 |

Shared holdings are exactly `32*h + 2*32*273240 + 32*414*414 +
106842112 + 32 MiB`: caller Q once, two response-grid copies, MAIN buffers,
native state plan, diagnostics. The real `_Ledger` and `BoundedResources`
definitions refuse the candidate at24 GiB and admit it at25 GiB.
Modeled headroom272629904 bytes is **not an RSS guarantee**.
Conditional source-review complete-chain bound62337903183272 <64000000000000;
checkpoint I/O including diagnostics6968466000 <68719476736. These
conditional source-review values are not measured run totals or physical passes.
Scheduler48G remains separate from both numerical-memory bindings.

## Parent-owned remote preparation and test-only instructions

Do not execute locally. No publication, SSH, package installation, native
calculation or scheduler action was performed by this implementation worker.
After independent review/publication, the parent must stage a clean immutable
checkout of the reviewed commit/tree at the **new** source path in the plan.
Do not change any accepted source/stage/run or old ref.

Metadata-only preparation on Phoenix, with `C` and `T` replaced by reviewed
actual commit/tree (not placeholders at execution):

```bash
/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt/bin/python3.13 \
  -I -S -B \
  /storage/project/r-cs207-0/awallace43/gits/psi4.ccasp-benzene-angular-full-20261023/devtools/ccasp_phoenix_benzene/angular_job.py \
  prepare --commit C --tree T
```

This exclusively creates new `runtime-001`, `full-001` and scratch paths.
It composes the configured accepted package root/launcher plus complete
committed driver Python inventory using unchanged32MiB/5000-file bounds,
and links the accepted native core/data. It verifies old probe provenance,
then writes `full-001/metadata/preparation.json` with concrete pinned
`test_only_argv` and `future_submit_argv_not_executed`. It never calls sbatch.
On any failure, retain partial paths; no overwrite, cleanup or auto retry.

The parent should inspect the bounded preparation receipt and execute **only
its exact `test_only_argv`** during this phase:

```text
/opt/slurm/current/bin/sbatch --test-only --export=NONE --chdir=NEW_FULL_RUN
  NEW_SOURCE/devtools/ccasp_phoenix_benzene/angular_full.sbatch
  C T GENERATED_RUNTIME_SHA GENERATED_DERIVATIVE_SHA
```

Check route/capacity/storage/identity independently. Script pins explicit
gts-cs207-chemx/cpu-small/**embers**, one node/task,8 CPUs,48G,6h,
atl1-1-02-007-23-1 (marchnative), numericalthreads1, p4cuest_sapt Python3.13.
Compute guard/closure reject an incompatible node/interpreter/dependency set.
TERM-safe background child/wait and honest receipts are inherited; explicit
`--no-requeue`, no chain, watchdog, automatic retry or submission API.

Only the parent may eventually authorize activation of the **one** approved
allocation after its independent review and fresh preflight. No extra probe or
preflight allocation is implied here. Native/runtime/probe materialization and
physical success remain unverified locally.
