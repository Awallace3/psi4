# Synthetic native complete72 PFIT smoke

This is a **new synthetic-only procedure**, not the full benzene driver.
One free parent-reviewed Phoenix allocation is approved: two CPUs, 8G,
30 minutes, embers, same accepted build node. This worker implements/tests
locally and exits. Publication, staging, test-only and submission are separate
parent actions. No retry, chain, build, install, native fallback or paid QoS.

## Explicit new identity, unchanged old files

Entry points: `pfit_smoke_job.py`, `pfit_smoke_driver.py`,
`pfit_smoke_profile.py`, `pfit_smoke_plan.json`, `pfit_smoke.sbatch`.
All are opt-in additions. No old plan, profile, global runtime namespace,
production source, charge gate, or strict LW policy changes.
`integrated_runtime.for_plan` supplies a separate module namespace; its exact
composition, independently committed complete Python inventory, accepted-core
link, dependency hash and retained-Einsums initialization/closure checks are
reused. No candidate full driver is called.

Native source15cc9a7a9a2bd60e43ebc2533ec99686bca58374,
tree8f0aa8fabd476aafc6bc49b824c60ac935a86753,
coreb283901ddac01e6bf7210a7461fb71cd5b2333cf9e42b99d60271959b40ac7eb.
The plan uses the exact accepted `p4cuest_sapt/bin/python3.13`, Einsums1.1.3
preload and GNU-MKL/OpenMP dependency closure. No package changes.
Node/interpreter/allocation/thread guards precede scientific imports/CDLL.
The same retained handle lives through final loader, runtime, source and
procedure closure and terminal receipt writes, in the same interpreter.

The procedure identity and original full declaration provenance are separate:

* Procedure: `synthetic-complete72-static-mask-default-pfit/1`.
* Original complete declaration: base
  `d1b23d019d59d30b6ca9cff9018177cac285d341`,
  `pdef_profile.declared_variables/1`; full list is compared with the committed
  `pdef_plan.json:overrides.refinement.declared_variables`.
* Declaration canonical SHA256:
  `f3ed7ee6ab75cdfc090c7f246c48bf4d4f00a22b3f1aa30c1dc5abeec5fcd28e`.
  Original profile and plan byte hashes are also checked.
* Source binding retains accepted native identity separately from the new
  committed Python/tools/runtime identity. Additional source checks require
  the entire `psi4/` tree to equal the base and every base-relative change
  to be an added tools-directory file. Existing runtime checks also compare
  native source/GBS trees with the accepted native commit.

There is no private geometry, original probe derivation, molecular basis
construction or full25GiB effective config in this procedure. A new
`procedure-admission.json` explicitly says basis admission is **UNTESTED**;
it is not a basis-derivative admission or a new molecular measurement.
Runtime metadata/composition do not certify those untested admissions.

## Exact numerical procedure

Twelve fictitious C/H sites (six COPY members/type), rank2, proper alternating
identity/180-degree Cartesian rotations, 108 channels; no native fields or
integrals are generated. Thirty-two synthetic point labels and deterministic
PCG64(seed20261025) finite supplied fields, divided by sqrt(6). These are
synthetic channels, not Coulomb fields of a molecule. Supplied fields already
carry their declared local-axis interpretation.

Eleven frequency anchors, omega=0,.03,...,.30. The actual
`isapol_bounded.refinement_models` / `_refinement_node` path receives the real
complete72 declaration, cutoff1e-4, weight3/.001. It is not replaced by a
hardcoded model or direct static-helper construction.

`H1_11s_21c_A` is 4e-5 at the static node and becomes2e-4 at node3.
It remains a fitted **FREE** variable (anchor/strength0), not fixed zero.
`C1_10_10_A` drops from.02 to3e-5 after the static node; its exact current
anchor and frequency-dependent weight remain penalized. Deliberate COPY
anchor discrepancies cannot select parameters or replace the reference.
Every node must retain the identical71-selected mask/static-reference hash,
all72labels, complete six-member entry tables and current-node completeness
guard. No charge variable is declared.

1. Independently form all528 lower-triangle rows from explicit COPY entries,
   including both tensor off-diagonal orderings, **no point-pair weight**.
2. Actual `PackedDesignRows` with37-row blocks, two passes, all point pairs.
   Compare every row with independent algebra and require bit-identical
   pass digests; a third pass must refuse.
3. Before any solve require rank72, cond2(A)<=100 and, separately at each
   node, cond2(A.T A+diag(w))<=10000. No automatic cloud/seed/threshold retry.
4. Design-generated known target parameters are current anchors plus
   `.04*(1+sin(k+.1*t))`; the FREE extra is `.125+.001*t`.
   Targets are `A @ truth`, not molecular responses. The penalty has a
   measurable effect (>1e-8 maximum parameter shift), and the independent
   penalized solution must recover truth within1e-4 maximum bias.
5. Actual `refine_streamed` with **options=None**, explicitly exercising its
   default `NormalEquationsDSYSV` path. The underlying
   `isa_pfit_solve_rows_multi` assembles all528 rows and replays all528 for
   residuals. Its native adapter independently digests consumed design and
   targets and refuses changed replay content. A separate standalone
   PackedDesignRows digest is recorded; it is not mislabelled as an exposed
   native digest. No QR fallback.
6. At each of11nodes, construct the actual `IsaPfitProblem` via
   `refinement_problem`, inspect its72 unfixed full channel tensor entries,
   and call actual native `isa_pfit_solve` with DSYSV. Only one such dense
   problem is live at a time. Both native routes must report `Solved`,
   exactly72 free indices/rank, correct labels/frequency/528data rows,
   condition diagnostics, LAPACK info0, objective availability and budget.
7. Compare BOTH native solutions with independent augmented dense
   `numpy.linalg.lstsq`, never with another native solver as the oracle.
   Require nonzero FREE recovery, COPY/mirror tensor placement, unchanged
   static selection, effective penalty matrix and exact current penalty RHS.

### Objective and fixed tolerances

From `pfit.cc`: data rows enter DSYRK/DGEMM without normalization or point
weights; diagonal penalty `P=diag(w)` is decomposed into square-root rows,
with targets `sqrt(w)*anchor`. Thus the independent oracle solves

```text
min_z ||A z - y||² + sum_k w_k (z_k - anchor_k)²
augmented A = [A; diag(sqrt(w))]
augmented y = [y; sqrt(w)*anchor]
H = A.T A + diag(w), rhs = A.T y + w*anchor
```

There is no factor of2, row-count normalization, energy1/2, extra COPY
multiplicity penalty, static replacement anchor, or dynamic selection.
The native default struct starts with QR, but the real Python refinement
default explicitly selects DSYSV; the dense call does so too. All other
rank/rcond tolerances and default256MiB kernel cap remain unchanged.

Fixed parameter comparisons: atol1e-10, rtol1e-9. Fixed objective comparison:
atol1e-12, rtol1e-8. Independent row algebra: atol/rtol1e-13; current penalty
matrix/RHS: atol1e-15, rtol1e-13. Native normal-H rcond must be>=1e-6.
At the admitted cond(A)<=100 / cond(H)<=10000, roundoff amplification for
the normal equations is O(10000*eps), about2.2e-12 before dimension constants;
the chosen tolerances allow different BLAS accumulation order without
accepting the >1e-8 penalty-effect signal. Actual engine-free cond(A) is about
4.02 (normal conditioning about16.14), far from the admission limits.
These thresholds are fixed in the closed profile, not computed from a failing
native outcome or loosened after failure. QR/lstsq is independent-oracle only.

### Negative cases

Duplicate, incomplete (upward extra omitted) and out-of-rank declarations
must refuse in the real bounded constructor before solver dispatch. Third
replay is refused. A separate actual bounded model with weight_type0 and
zero supplied fields has a singular unpenalized design. Actual dense native
status must be `RankDeficient`, its parameters getter must throw, and actual
`refine_streamed` must raise its “did not solve” ValueError. No unsuccessful
fit is reported as a successful node or made solvable by fixing variables.
This is an expected negative test, not a retry of the positive solve.

## Resource arithmetic / bounded evidence

Closed test ceilings: numerical4GiB4294967296 (also actual Psi4 memory),
maxparameters72, maxwork48,000,000,000, test I/O32MiB,
48 run-output files, 16MiB aggregate, each JSON/map snapshot<=256KiB,
stdout/stderr<=1MiB each. There is no response/ISA scratch or full-node
256MiB diagnostic bundle. The plan's snapshot capacity48*256KiB=12MiB
fits below16MiB, with two bounded logs still fitting. Metadata/runtime
composition is a separate inherited32MiB/5000-file admission, not numerical
checkpoint traffic. No raw Git event logs or unbounded stdout bundles.
The32MiB test-I/O ceiling is not a claim about total filesystem traffic:
read-only hashing of the pinned core/dependency/source files during inherited
runtime admission/closure is separate, unchanged validation traffic.

Actual unmodified `_pfit_plan(32,108,72,11)`:
50,584,576bytes /87,583,488work.
Actual PackedDesignRows:
2,581,568bytes /71,533,824two-pass work, chunk18.
528*72*8=304,128bytes for one dense oracle design;
528*11*8=46,464bytes for targets; fields32*108*8=27,648bytes.

Source-counted dense arithmetic ceiling:
`(2*11+1)*528*72*108*108*4 = 40,794,513,408`.
This includes assembly AND residual replay for eleven positive dense problems
plus a conservative full-size singular assembly traversal.
`data_rows` uses two multiplies and an add/term,
rounded up to four work units for zeroing/writes. It is not a CPU instruction
or walltime estimate. Add real planner work, the standalone two-pass plan,
and a conservative500M small SVD/oracle/validation allowance:
**41,453,630,720 <48,000,000,000**. Negative zero-design streamed work is
covered by that allowance; there is no adaptive repetition.

Numerical overlap bound:
50,584,576 +2,581,568 +2*256MiB(default native work caps) +128MiB
(Python full-tensor/list, reference and small matrix overlap allowance)
= **724,254,784 <4GiB**. Input full tensors72*108²*8=6,718,464bytes,
with one dense problem live at a time. Streamed Python buffers independently
have16MiB caps. FourGiB numerical plus interpreter/vendor-library overhead
must fit scheduler8G; this is not measured RSS or a30-minute guarantee.
The tiny cloud has much lower arithmetic than the full2000-point fit.

Snapshots: design;11models (actual labels/entries, static masks/hash, current
anchors/weights, raw crossing anchor, target truth/reference);11solutions
(status, parameters, dense/reference errors, objectives, condition, digests);
negative cases; final result. Result snapshots say checks passed **pending
final closure**, not global completion. Final `PASS`/payload receipt is written
only after independent final loader/runtime/source/procedure closures and
output bounds pass. A loader failure cannot skip the remaining closures.
Primary errors and separate closure errors are retained. Batch completeness
additionally requires unsignalled zero child status; TERM sets143, waits on
the child and cannot write complete=true. Missing/truncated payload on a hard
kill is failure, never acceptance. Partial destinations are retained.

## Local checks (engine-free only)

```bash
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
P=/home/awallace43/miniconda3/envs/p4_ci/bin/python
"$P" -B -m unittest discover -s devtools/ccasp_phoenix_benzene -p 'test_*.py' -v
PYTHONPATH=tests/pytests "$P" -B -m unittest \
  test_isapol_bounded_campaign_source test_isapol_selection_source \
  test_isapol_scheme_source test_isapol_property_water_source -v
bash -n devtools/ccasp_phoenix_benzene/pfit_smoke.sbatch
git diff --check
```

`test_pfit_smoke.py` source-loads actual production Python using only the
marked fake core. It exercises the same driver, both actual wrappers,
declarations, streamed two-pass callback contract, independent objective,
negative cases, resource identities, exclusive metadata preparation, no-native
before guard, and honest final receipts/failure closure. Test-only native
stand-ins use NumPy normal equations; they are NEVER shipped as a runtime
fallback and do not certify compiled/native DSYSV, ABI, dependency closure,
native replay, native status, physical LW or molecular parity.

## Future parent-only preparation / test-only (NOT executed here)

After independent review/publication/staging to the new clean source:

```text
/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt/bin/python3.13
  -I -S -B
  /storage/project/r-cs207-0/awallace43/gits/psi4.ccasp-pdef-native-smoke-20261025/devtools/ccasp_phoenix_benzene/pfit_smoke_job.py
  prepare --commit REVIEWED_COMMIT --tree REVIEWED_TREE
```

New root `/storage/project/r-cs207-0/awallace43/runs/ccasp-pdef-native-smoke-20261025`,
`runtime-001`, `smoke-001`;
scratch `/storage/scratch1/3/awallace43/ccasp-pdef-native-smoke-20261025-smoke001`.
No existing root/destination is accepted, even after partial preparation.
No overwrite, cleanup or automatic retry.

Inspect generated `smoke-001/metadata/preparation.json` and procedure/runtime
pins. The ONLY generated scheduler argv is concrete **test-only**:

```text
/opt/slurm/current/bin/sbatch --test-only --export=NONE
  --chdir=.../ccasp-pdef-native-smoke-20261025/smoke-001
  .../psi4.ccasp-pdef-native-smoke-20261025/devtools/ccasp_phoenix_benzene/pfit_smoke.sbatch
  REVIEWED_COMMIT REVIEWED_TREE GENERATED_RUNTIME_SHA GENERATED_PROCEDURE_SHA
```

Preparation itself never invokes it and never generates a submit command.
Parent independently performs fresh route/preflight review and, if admitted,
the one approved synthetic job; no full-driver activation. The batch resolves
`srun` and requires `/opt/slurm/current/bin/srun` before PATH sanitization.

Even an eventual native smoke PASS is **not** molecular parity, fullcloud2000,
all11physical-node response, native LW, own-SCF, basis/frame measurement,
SCF/ISA/H2H1 success or final anisotropic-property acceptance. No charge-gate
change and no such calculation is within this grant.
