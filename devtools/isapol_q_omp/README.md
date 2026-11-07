# Ordered native stockholder-Q: bounded low-thread follow-up

## Why this patch

The compiled5ae candidate's controlled benchmark15282426 reduced water's
eight-thread median schema time56.285→51.607s (8.31% less), but its synthetic
all-s Q constructor-call proxy regressed24.89%/18.16% at1/2 threads.
Constructor-call timing includes pybind marshalling; these are not pure native
accumulator timings or general molecular scaling results.

This new branch starts **exactly at5ae597cd9305588d14681d0da435834ef761c264**.
At Process thread count<=2, it restores the original point/component/AUX loop
before any staged-prefactor/active-flag allocation. Basis evaluation retains
its original worker policy (including two-thread collocation). At>=3 threads,
the5ae accumulation tail is unchanged; the additional branch is once per batch,
not inside the high-thread point loop. Changing only parallel_work's threshold
would leave the measured low-thread staged/tiled overhead in place.

No compiled correctness or performance claim is made for this follow-up yet.

Full analysis outside source:
`/home/awallace43/.cache/ccasp-q-benchmark001-analysis.md`.
It compares every local/refined tensor entry and every C6/C8/partial-C10 value.
Full-chain local tensors varied by up to2.017e-7 within OLD repeats and
2.291e-7 across arms; they are not a bitwise kernel oracle. All existing native-verified
flags are false; numeric transport passes do not grant solver certification.

## Scope and invariants

This minimal SCI child of5ae contains no build/benchmark orchestration changes.
Frozen q-omp/a1e, benchmark5b, profiler835, deployed sources and old receipts
remain untouched. No global BLAS/OpenMP setting or public thread knob changes.

1. `psi4/src/psi4/libisapol/partitioned_response.cc`:
   - Original site order, <=4096-point/8MiB collocation batches and full screen/
     basis validation remain unchanged.
   - Serial ratio/cutoff/negative/excluded/harmonic validation retains original
     point order and messages, before the parallel accumulation.
   - At1/2 threads, the original loop immediately validates/updates each point.
     It is whitespace-normalized identical to scientific1aaa's point loop,
     including Matrix add/get expressions, signed zeros and diagnostic order.
     Original batch collocation still precedes every sample check; empty grids
     still validate their complete screens.
   - Each batch prepares only its bounded point/component prefactors and active
     flags. Exclusions skip the update, not a synthetic zero addition. Screened
     basis zeros are still multiplied/added.
   - Existing `parallel_work` owns disjoint component/column intervals, at most
     **25*8=200 tasks**, with static scheduling and no nested OpenMP. Each output
     sees **all valid points in original order across all original batches**.
     There are no reductions, point partial sums or BLAS calls.
   - Direct output pointers replace Matrix accessors in the inner loop. No
     worker mutates metadata, counters, operands or any other output interval.
2. `psi4/driver/procrouting/isapol_distribution.py` is **unchanged from5ae**:
   it still charges native-Q scratch in **both ISA and MBIS** callers. No
   scientific options, caps, work ceilings or thresholds change.
3. New frozen-operand API tests and pure workspace/task-ownership tests only.

The compiled target must retain `-ffp-contract=off` and ordinary IEEE
non-reassociation flags. Bitwise equality is **expected and required on fixed
operands**, not yet verified for this unbuilt candidate.

### Memory

Unchanged collocation: min(4096,8MiB/(8*naux)) points. Additional per-batch
payload is `count*(8*(rank+1)^2+1)` bytes, at most**823296** at rank4, plus
**4096 bytes** explicitly reserved for bounded task-control overhead:
**827392 bytes maximum**, independent of grid size and atom/thread counts.
High-thread workspace remains per batch, never a full-grid cache/public knob.
Low-thread accumulation allocates none of these prefactors/active flags, but
admission deliberately retains the same conservative reservation at all thread
counts. There is no thread-dependent refund or relaxed admission boundary.
Old Q output and coordinate-block storage are unchanged.

Existing numeric admission becomes stricter by this additional reservation;
no caller's max_bytes/work/I/O cap is increased. For measured water the old
3,071,825,120-byte numeric plan would increase by827,392 bytes. Resource-plan
receipts for the old implementation remain old evidence, not candidate admission.

## Tests

Local **pure stdlib only**, no Psi4 import or native computation:

```bash
python3 -I -B devtools/isapol_q_omp/test_workspace.py
```

It checks workspace bounds, original row-budget edge, explicit charges in both
callers, exact disjoint output coverage with <=200 tasks, original1aaa loop
identity, early dispatch before allocations and unchanged5ae high-thread tail/
Python planner. Source tests require Git objects1aaa and5ae; they are not a
compiler, bitwise native test or timing verification.

**Parent compute allocation only:** build a fresh private candidate with a
matching installed Python/core. Do not copy a new `.so` into the qualified
1aaa environment, edit old receipts, or point the frozen835 profiler at a new
implementation: its identity rejection is intentional.

Run from a **new owned writable CWD**, with new owned scratch, never source or
fixtures. Set PATH/LD_LIBRARY_PATH to the selected private installation and its
original ordinary dependencies, OMP/MKL=8, OpenBLAS=1 (unchanged backend policy).
No new OMP_PROC_BIND/OMP_PLACES. Example after parent creates fresh OUT/SCR:

```bash
# Parent executes in OUT/cwd; SCI_PYTHON is the selected private installed Python.
# SOURCE points to this candidate Git commit; all paths below are absolute.
env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 \
  PSI_SCRATCH="$SCR" SCRATCH="$SCR" TMPDIR="$SCR" \
  "$SCI_PYTHON" -I -B -X "pycache_prefix=$SCR/cache" -m pytest -q \
  --rootdir="$SOURCE" --basetemp="$SCR/pytest" -o "cache_dir=$OUT/pytest-cache" \
  --junitxml="$OUT/q-omp.xml" \
  "$SOURCE/tests/pytests/test_isapol_q_parallel.py" \
  "$SOURCE/tests/pytests/test_isapol_native_performance.py" \
  "$SOURCE/tests/pytests/test_isapol_partitioned_response.py"
```

The **34 API cases** use frozen explicit Gaussian operands, no SCF. They
compare the original serial p/component/AUX arithmetic (using native fixed
basis/harmonic samples, no point reduction) to native1/2/4/8 via uint64 views.
Cases include Cartesian/spherical AUX, ranks0–4, multiple/permuted sites,
screens/zero columns, signed-zero/underflow/subnormals, cancellation, cutoff
neighbors, repeat/race-sensitive output checks, tile/8MiB/4096 boundaries,
empty grids and exception precedence across old batch boundaries.
Two added cases alternate1/2→4→1/2→8→1/2 with three collocation batches,
testing current-call dispatch, exact original sums/counters and site offsets.
The independent reference collocation in each case is bounded below4.3MiB.

Run that new test file against **both** the old1aaa installed core and the new
candidate core in separate fresh processes/CWDs. This validates the serial
oracle against the original native implementation, not merely against the new
one-thread path. Record imported identities and test-source hash.

Then parent can run existing distribution admission tests, including exact-plan
and one-byte-short refusals, in its authorized scientific allocation:

```bash
"$SCI_PYTHON" -I -B -X "pycache_prefix=$SCR/cache" -m pytest -q \
  -o "cache_dir=$OUT/pytest-cache" --basetemp="$SCR/distribution-pytest" \
  "$SOURCE/tests/pytests/test_isapol_distribution.py"
```

Native/build tests have **not** been executed locally. The new core requires a
normal Release/OpenMP-enabled build/relink and install of the updated Python
file. No binding or CMake interface changes are needed. Preserve existing
compiler floating-point flags; use a new build/source identity and private
manifest. Parent owns all Phoenix/embers resource admission and execution.

## Performance acceptance after correctness

After fixed-operand bitwise/diagnostic tests pass, benchmark original versus
candidate Q on **identical frozen operands** at1/2/4/8, same CPU/affinity,
memory/backend policy, recording constructor-only time and process peakRSS.
Keep full-chain recomputation comparisons separately labeled; never relax
thresholds or use their preexisting variability as a bitwise kernel gate.
Repeat end-to-end water only via a fresh benchmark-only derived identity
request, without mutating accepted inputs or frozen835 profiler pins.

Reuse the corrected a1e ops/build/oracle machinery in a separately reviewed
ops child pinned to this new SCI commit/hashes. Rerun the lightweight
same-source original34-case oracle, then one fresh private build/install and
84 native gates (34Q+18 performance+32 partition response). Use the unchanged
approved4356-byte configure record SHA896949860e3eb012e07b2ec45556ab4580d87e36de094ca5567d800149803380.
Keep all three generated Python outputs and both installed-test source
namespaces strict; no historical validate.py recovery is required on success.
Only actual build results supply new identities/receipt pins for a minimal
benchmark rebind; never guess them or weaken frozen runners' old guards.

No merge, push, deployment or new numerical/speed/scientific acceptance is
implied by this candidate commit.
