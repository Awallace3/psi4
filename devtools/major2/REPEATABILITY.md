# Blinded Protein157 numerical repeatability diagnostic

Separate from the failed frozen24-record primary benchmark. This does not
resume, relax or accept that cohort. `science.compare`, `science.summary`,
all81 prior host checks and all existing helper files are byte-identical.
No ratios, ranking, median wall times or speedup claims are generated.

## Exact pair, no compilation

BASE c56708fc6f6e8fcf2801868f0d3cb49bdb1bb0a2 / job14165998 /
core67c1e9021cc2fcaaf3b4ad3f98f6750ee39049c1aafe8cdfbe75b46c8335f224.
CAND21a7b4d9c922f293d6220cafd8143d70f7d85e18 / job14170562 /
core839e7bc2e52fe9d435689225fa718946710bb72d8147357ae3a8578fb8bdbf58.
Only these original paths/pins/cores/origins are allowlisted. Stage snapshots
are reused on login, **fully rehashed on the allocated H200 host**.
Complete source/stage/metadata/config/compile-DB/ABI/SASS/runtime/143driver
verification is mandatory there. Existing122 positive GPU checks and specific
baseline-red receipts are revalidated; they are not misreported as fresh tests.
Both origins say `compiled_in_this_job=false` in the diagnostic receipt.

The original CUDA13 identity helper from14170562 is also immutable and reused,
never rebuilt. Missing/changed helper is a concrete setup failure, not a reason
to compile, change runtime or guess CUDA device indices.

## Science and order

Exactly6 chemistry processes, B1,C1,C2,B2,B3,C3; Protein157 only. Same original
geometry, runner and PBE0 SAPT(DFT)-D4(I),6-31+g**, fixedGRAC.136/.136,
dHF, fullCPHF/CPKS, FP64,8threads,100GiB,35x194->75x302/factor1/min2,
e9/d8. Each process has a1800s fail-closed timeout; no process retry or
extra warmup/smoke/CPU benchmark is added. Raw timers are original, not edited.
Residual<1e-8, printed terminal SCF e9/d8 and all1e-8Eh component comparisons
remain fatal. Iteration differences are retained as diagnostic findings.

## Observation review

The current runner selects INTERNAL. Its first `scf_iterate` loop increments
iteration-1 to0 then calls `self.form_G()`. The OOO branch's lines617–628
are **not** the active path. In both exact bindings, RHF inherits the virtual
`HF::form_G` descriptor. A temporary RHF shadow calls this descriptor once.
The actual staged iterator frame's `self` must be the observed RHF. The actual
staged `proc.scf_helper` frame's local `banner` (already popped from kwargs)
must identify the expected stage, in order:
HF dimer, HF monomerA, HF monomerB, DFT monomerA, DFT monomerB.

First call must be iteration0/SAD, C1/RHF,nbf1786,natom157, closed-shell;
functionalHF/HF/HF/PBE0/PBE0. Initial dimer additionally requires2 physical
fragments/no ghost atoms and no preceding finalize. Actual nalpha/nbeta, class,
PID, Python identity/callsite, provenance/core/source/manifest and wrapper/hook
hashes are recorded. Native form_G call/return counts must match all printed
SCF rows. An additional read-only finalize wrapper proves chronological
completion and also calls its original descriptor once.

Before the first form_G per stage, copy actual Da; verify Db is numerically
equal (RHF). After original returns, read `wfn.jk().J()/K()`, exactly one
matrix each. They are the actual cuEST outputs, including its existing hybrid
exchange convention—not recomputed or rescaled physical exchange. Readonly
C-contiguous FP64 copies are stored in compressed NPZ with array byte hashes,
shapes,dtypes, archive hashes and exact counts. No scientific state setters,
J/K recomputation, density seeds, orbital writes or option changes are used.
All class shadows and the unchanged runner's HF proof adapter are restored in
try/finally, including Python signal exceptions. Before/captured/partial and
COMPLETE receipts are exclusive atomic,fsynced. TERM forwarding retains all
available partials; SIGKILL cannot run finally and is not claimed otherwise.

Only15 matrices/process,90 total. Uncompressed data2.139GiB; **hard persistent
NPZ cap2.25GiB total**,384MiB/process. Db is not stored redundantly. No
all-iteration matrices. Scalar printed energy/density trajectories are retained.
Full matrix norms/equality analysis runs only on allocated GPU-host CPUs.
All15 within/across process comparisons report exact byte hashes, bitwise
equality, numeric equality (signed zeros distinguished),maxabs and relative
Frobenius norm with `max(normA,normB)` denominator. First divergent stage/matrix
and first printed trajectory difference are reported, never a timing summary.

## Safety and handoff

New4h free embers/H2001/8CPU/128G/no-requeue allocation. Preserved10 old failed
roots include8 individually verified production records from14170562.
Bounded preservation stats/file sets on login are not fresh binary hashes.
New run/root/scratch/source pins cannot alias an old namespace. A permanent
exclusive submission lock consumes the attempt; numeric ID is fsynced before
acceptance checks/controller queries. No automatic retry. Actual submission
is explicitly parent-only after review; launch the official watcher only after
the numeric parent ID exists. No worker has submitted anything.

Host tests and staged import/temporary-shadow checks do not establish GPU
hook execution, SAD repeatability or a cause of the original drift.
If an actual seam/stamp/count fails, STOP and preserve partials, not substitute
later matrices. Raw timing observations remain blinded pending separate review.
