# Native Q / full-water benchmark only

Separate ops child/worktree based on a1e5231f1f; only this directory changes.
Frozen SCI/native tests, deployed validation ops and profiler835 are unchanged.
No new build, campaign framework, qualification/adoption route or production pin
override. Parent submits/stages; no native/quantum/network/cluster action locally.

## Two different measurements

1. **Synthetic constructor proxy**, OLD/CAND x1/2/4/8, fixed8CPU allocation.
   N=65,543,3 rank4 sites, AUX246, Q75x246, seed4103 and site seeds41/97/173.
   Uses the frozen5ae test builders:246 all-s shells on1 AUX centre, spherical,
   not the real mixed-shell water AUX, molecular quadrature or ISA shape field.
   Explicit label: `SYNTHETIC_ALL_S_WIDTH_MATCHED_NOT_MOLECULAR_Q`.
   One untimed warm-up,5 timed native constructors per cell; operand/basis/
   list construction and exports outside timing. Constructor includes native
   collocation/harmonics/screens/accumulation, not accumulator-only instructions.
   No Python serial oracle timed; the previously passing32 independent native
   oracle cases remain correctness evidence.

   Input arrays and exact bit hashes, JSON basis recipe description/site seeds/
   origins/screens/cutoff and complete small Q outputs are retained. Every call
   must match its warm-up's values/diagnostics/offsets, and all8 cells must match
   OLD1 bitwise and have identical frozen input hashes. Arrays are small/bounded;
   no full Ngrid x AUX matrix or scientific cache. Native batches remain<=8MiB,
   existing candidate payload/control charge unchanged. Python and native copies
   count toward the fixed20GiB process science budget/32GiB allocation; RSS is
   measured, not inferred solely from array sizes.

2. **Full water**,3 fresh8-thread runs per arm, interleaved
   OLD,CAND,CAND,OLD,OLD,CAND, same allocation/affinity. No orbital/state reuse.
   PBE0/aug-cc-pVTZ, actual partition400/2030 (Q2,429,910 rows),
   response100/200 (68,310 rows), H2H1, Drho32, all resources/cutoffs/thresholds/
   coverage unchanged,20GiB science memory. Process/MKL8, OpenBLAS1.

   Reads immutable accepted water fixture through835's original inventory/
   scientific checks. OLD uses exact claimed input. CAND uses a NEW deep copy
   with **exactly two changed scalar leaves**:
   `keywords.function_kwargs.native_monomer.implementation_sha256.{core,isapol_python}`.
   Exact diff/before/after hashes, used input and normalized input are saved.
   Native admission still checks the actual installed identity normally.

   Approved candidate identity comes from tested validation002:
   core `676ea1e1b616a4c4df4f221688b31b8ae7a367539e9187676141b3109c85c013`,
   Python `2b7758f3eec48bb8cfc3565d50505ba1ce1fd911cbf1d762aab7773549d3713d`.
   The validation record's `identity.implementation` is a direct pair.
   No835 main/receipt/fixture pins are patched or loosened.

Both arms use this same child runner and835 telemetry/hooks read-only.
Warm import/admission/thread checks for both prefixes precede all timing.
Each cell/full run has a new interpreter, owned CWD/cache/scratch before any
Psi4 import, retained through core finalization. Identity, paths, runtime version,
actual affinity and mapped BLAS/OpenMP pool counts are recorded before/after.
No new OMP_PROC_BIND/OMP_PLACES/NUMA policy or backend scaling experiment.

## Integrity / retained evidence

Pins passing validation002 result SHA
`1582867ceb6f1ab0b8bd9f7d3ddeba443c35b745436713cf3e828e8abac0b3f8`,
its82-case JUnit and snapshot equality; candidate identity and compiled5f83
source/tree; original evidence SHA44582da0...; qualified OLD receipts/runtime.
Before/after, independently seals OLD, candidate dependencies/all installed
Python, and hashes the **entire existing compiled candidate tree** against the
passing validation snapshot. Source gates restrict this child to4 new files and
require unchanged profiler835. GIT_OPTIONAL_LOCKS=0 is inherited/exported.

Whole-candidate snapshots, per-child commands/stdout/stderr/process times,
events/stages, pool/CPU/affinity/RSS, complete QCSchema/envelope/tensors/Cn, proxy
operands/Q and sampled scratch maxima are retained under new owned outputs.
An independent controller-process sampler observes scratch while a native child
may hold the GIL. Its256GiB logical/allocated sampled byte ceiling stops a
detected excess; no arbitrary inode reservation. Samples can miss transients;
byte/namespace/link-target integrity is not a filesystem-write sandbox.

Historical build001 and diagnostic001 remain FAILED and untouched.
New status `BENCHMARK_COMPLETE_NOT_ACCEPTED`, acceptance/adoption false.
TERM-safe wrapper forwards through active/finalization children, no false0,
requeue/retry or dependency chain. No installed-prefix/receipt modifications.

## Parent launch

Stage this new source through Git; source commit/tree are reported separately.
ROOT/SCRATCH are **nonexistent**, absolute/disjoint dedicated roots.
LOG is a separate fresh existing directory; never precreate ROOT/SCRATCH.

```bash
/opt/slurm/current/bin/sbatch --test-only \
  --chdir="$LOG" --output="$LOG/bench-%j.out" --error="$LOG/bench-%j.err" \
  "$SOURCE/devtools/isapol_q_bench/benchmark.sbatch" \
  --source "$SOURCE" --commit "$COMMIT" --tree "$TREE" \
  --root "$ROOT" --scratch-root "$SCRATCH" \
  --profiler-source /storage/project/r-cs207-0/awallace43/gits/psi4.ccasp-native-profile001 \
  --fixtures /storage/project/r-cs207-0/awallace43/runs/ccasp-native-profile-fixtures001/monomer-0001 \
  --validation-result /storage/project/r-cs207-0/awallace43/runs/ccasp-native-q-validate002/result.json \
  --evidence /storage/project/r-cs207-0/awallace43/runs/ccasp-native-q-validate-input001/evidence.json
```

Parent alone replaces --test-only after review/preflight. Fixed1node/1task/8CPU/
32GiB/2h free embers, cpu-small/gts-cs207-chemx, no requeue.2h is a finite cap,
not a measured completion estimate; retain partial evidence on failure.

## Analysis contract

No speedup claim before actual measurements. Report5 proxy constructor times/
cell, median/range, T1/TN and matched-thread OLD/CAND ratios; keep the proxy
label prominent. Full water: all3 times/arm, median/range and exclusive stage
accounting (never sum overlapping spans). Schema vs process vs allocation time
remain separate; `_stockholder_q` includes Python/native adapter costs.

Parent independently compares every local/refined tensor entry and all ordered
C6/C8/partial-C10 values, frequencies/coverage/residuals/iteration counts/flags,
against accepted water and repeats. Full recomputation has measured preexisting
variability; **do not require or claim full-pipeline bitwise equality** or relax
scientific thresholds. Proxy bitwise equality and full-run variability are
separate findings. Native/transport passes are not solver/science certification.

Pure tests (stdlib/fake children only; no NumPy/Psi4 import):

```bash
python3 -I -B devtools/isapol_q_bench/test_bench.py
bash -n devtools/isapol_q_bench/benchmark.sbatch
```
