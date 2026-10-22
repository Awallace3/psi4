# Explicit integrated LW diagnostic — local preparation

No native execution is established by this document or its synthetic tests.
The coordinator owns publication, remote preparation, test-only/capacity
checks and the one conditional capture allocation. Do not run the frozen
P003 launchers, mutate its stage, or activate this preparation locally.

## Interface and default behavior

`bounded_properties(..., diagnostic_sink=None)` and the BOUNDED_DF oeprop
keyword of the same name are explicit opt-ins, never ambient Psi4 options.
The default forwards no new backend keyword, makes no observation, performs
no diagnostic reductions/output and keeps scientific arithmetic, defaults,
native C++, basis data, model, grid, penalty, tolerance and publication gates.
The existing common response/ISA pipeline is not duplicated.

The observer contract is trusted, synchronous and read-only: callbacks must
not mutate or retain live F/Q/B/X or the pre-LW operands. The concrete
`DiagnosticSink` fulfills that contract, retains only small binary vectors,
and is exercised for input immutability in synthetic seam tests. This is not
a sandbox for malicious arbitrary Python callbacks. The installed package
never imports devtools; the driver explicitly supplies the observer.

Hooks are finite and fail closed:

1. Native `native_constrained_ov` observes the vector from its **existing**
   `core.IsaAuxCoulomb(auxiliary).charges()` call, **anchor fit only**.
   It is the integral of the effective declared AUX functions, including
   actual Cartesian coefficients; there is no guessed normalization,
   nominal-charge substitution, clipping or second evaluation.
2. Before deleting anchor `fit`/`moments`, consume actual F, Q, and the same
   B returned by the existing `moments.anchor_legs(F)` contraction. Save
   actual lambda/eta, exact row-streamed operand hashes and small reductions.
3. After the existing H2H1 residual gate, consume the existing
   `self.legs[:,p:]` and `solved[:,p:]`. These are anchor B/X, **not** the
   contracted p×p target response. No new solve or reconstructed tensor.
4. `_localize_node` completes the actual full25 lossless snapshot and signed
   comparisons **before** invoking the unchanged strict public LW factory
   once. It catches nothing; original refusal propagates. No PFIT/publication
   is reached after that refusal.

Diagnostic selection is the declared original grid: up to eleven nodes,
one snapshot per reached node. Under the observed static-node refusal only
node1 is expected to be reached. No retry or search for another failing
frequency is performed. If the strict static gate unexpectedly passes, the
same original pipeline may continue under the approved scheduler walltime;
only completion of its original gates can publish accepted properties.
Capture completion alone is always labeled **not scientific acceptance**.

## Signed attribution

Let p be actual response AUX width, n=nocc*nvir and m=25*nsite. Q is (m,p),
anchor F (n,p), B=F@Q.T and existing X (n,m). Selector c has ones at25*a.

Retain q, `Q.T@c`, `deltaq=Q.T@c-q`, `Fq`, `Fdeltaq`, actual `Bc` and
`Bc-Fq-Fdeltaq`. After the one existing solve, retain Xc and:

```text
sfit = -(Fq).T @ X
sQ   = -(Fdeltaq).T @ X
sB   = -(Bc).T @ X
sX   = -B.T @ (Xc)
```

Save complete signed vectors and `sfit+sQ-sB`. The actual captured raw
response separately supplies:

```text
column-charge axis = alpha.T @ c = sum_b T[b,a,0,t]  ~ sB
row-charge axis    = alpha @ c   = sum_b T[a,b,t,0]  ~ sX
```

Both actual axes, site/component argmax, and raw-axis-minus-direct-B signed
vectors are saved, not merely rounded maxima. Do not equate axes by assumed
reciprocity. These are grouping/roundoff diagnostics, not new equality gates.

Each reduction stores ordinary ascending-index addition and `math.fsum`
over the **same rounded products** separately. Compensated deltaq/Fq and
other compensated vectors are not silently substituted into later ordinary
products or science. Scalar diagnostics do not establish native BLAS
arithmetic equivalence or a molecular root cause.

Only q/deltaq and O(n+m+p) vectors survive hooks. Row-streamed SHA-256 never
copies all F/Q/B/X. Whole matrices are neither retained nor serialized.
The actual raw m×m response and its original pair-layout view are copied
losslessly by `lw_capture.py`, not recomputed. Its established full25 format,
strict declarations, finite/size/hash checks and immutable bytes are retained.

## Bounds and output receipts

Opt-in only: p<=2048, n<=16384, <=12rank4 sites (m<=300), <=11nodes.
The default None path has none of these additional diagnostic restrictions.

The shared ledger reserves **32MiB numeric overhead throughout**, including
both reference-route reservation resets and retained-frequency workspace.
It charges upfront **256MiB I/O** for a **64MiB aggregate diagnostic output**
ceiling and conservative work
`128*n*(p+nf*m)+128*nf*m*m+128*p*m`. The early complete-frequency work check
includes that charge. Existing scientific budgets are not increased.
These are explicit plans, not measured RSS, walltime or a hard scratch quota;
dirty cache remains within the distinct48G scheduler allocation.

Exclusive0700 directories/0600 files, bounded metadata and manifest-last
bundles preserve partial failures without overwrite or cleanup:

```text
results/diagnostic/context.json
results/diagnostic/plan.json
results/diagnostic/anchor/{*.f64,manifest.json}
results/diagnostic/node-01/{*.f64,manifest.json}
results/diagnostic/node-01/lw-input/{*.f64,manifest.json}
results/diagnostic/node-01/raw-axis-comparisons/{*.f64,manifest.json}
results/diagnostic/node-01/pre-lw-complete.json
```

Only the last marker links a complete capture, solve/anchor manifests and
signed comparisons. It does **not** say strict LW passed. Overall result,
payload and batch receipts preserve nonzero failure and original traceback.
No output is called a recovered P003 tensor: P003 bytes remain unavailable.

## Explicit composed runtime and derivative admission

`integrated_plan.json` fixes new exclusive source/runtime/run/scratch paths.
Historical plans/helpers remain unchanged. In the future reviewed Phoenix
preparation action, `integrated_job.py prepare --commit C --tree T`:

- proves new diagnostic Git identity/clean state, exact routing allowlist,
  unchanged native/C++ and GBS Git trees, and unchanged original four
  scientific driver/basis/config files;
- copies **configured package-root files** from accepted native stage,
  materializes the **complete driver Python tree** from new committed source,
  and uses one explicit core symlink to accepted15cc/b283. It does not use raw
  source `psi4/__init__.py`/`extras.py` CMake templates;
- copies the accepted configured relocatable bin/psi4 script, records its
  inherited provenance, and links accepted immutable data;
- verifies exact package file set, hashes/source roles, link roles, bytecode
  absence and small composition bounds, then writes runtime manifest last;
- verifies the original private probe's external SHA, schema/status,
  **entire old source/driver/file/config/fixture/native binding** and canonical
  normalized-shell fingerprint pins;
- writes a separate derivative admission binding original probe provenance
  to new driver/Python/file pins and runtime-manifest SHA. The original probe
  is never edited or falsely relabeled as a measurement of the new driver;
- creates new absolute writable log/metadata parents, hashes the committed
  batch script and emits exact test-only/future-submit argv using the newly
  generated manifest hashes. It never invokes sbatch.

Large core/dependency hashes and imports belong to the later compute
allocation, not metadata composition. The composition receipt explicitly
marks the core hash as **expected, not measured during composition**.
Remote installed file layout and private probe bytes have not been inspected
under this author task; preparation must fail closed if they disagree.
Partial new preparation paths are never overwritten/reused automatically.

Inside the one allocated process, `initialize` admits the exact isolated
p4cuest_sapt Python, explicit admitted Einsums1.1.3 CDLL/hash/retained handle,
new absolute staged `bin/psi4 --psiapi`, new package/core/data/file roles,
unchanged isolated QC dependencies, numericalthreads1, and the original strict
mapped-loader closure. It uses new-stage/lib, isolated DEPS/lib/DEPS/python
and selected environment site-packages, not another Psi4 prefix. No install,
ELF/RPATH patch, version default, MKL override or dependency fallback.

Actual basis metadata is rebuilt/admitted **before SCF in that same single
allocation**, through unchanged basis helpers/GBS/core. Exact full shell
snapshots must match the old measured MAIN/AUX fingerprints, and post-SCF
MAIN must match admission. Derivative source/runtime pins are separately
checked. No extra basis-only allocation is inherently necessary; mismatch
stops nonzero rather than scheduling one.

The exact CDLL handle remains retained through the final after-capture
closure attempt, even on initialization/scientific failure. A failed final
closure is recorded separately without erasing the original scientific
traceback. New package/source hashes are rechecked. All tests of these paths
are temporary engine-free mocks or source-level checks, not actual admission.

## Coordinator-owned future activation

Only after independent review/publication on the NEW branch and immutable
remote checkout should the coordinator run metadata preparation. The generated
`capture-001/metadata/preparation.json` contains the concrete pinned argv:

```text
/opt/slurm/current/bin/sbatch --test-only --export=NONE --chdir=NEW_RUN
  NEW_SOURCE/devtools/ccasp_phoenix_benzene/integrated_capture.sbatch
  C T GENERATED_RUNTIME_SHA GENERATED_DERIVATIVE_SHA
```

Names above describe generated inputs, **not fabricated ready hashes**.
Final actual hashes cannot exist locally before authorized remote
materialization; preparation emits them. The committed script fixes
gts-cs207-chemx/cpu-small/explicitembers,1node1task8CPU48G6h,
same buildnode atl1-1-02-007-23-1, numericalthreads1/Psi4memory24GiB.
Fresh test-only route/capacity/storage checks are still required.

Scheduler walltime and TERM-safe child/wait/receipts replace watchdogs.
No dependency chain, resubmit, replay allocation, inferno or job control.
The science profile, fixture, production1e-6 gate and400GiB free-space check
(200GiB declared scratch budget, not a hard quota) remain unchanged.
