# Ordered native stockholder-Q OpenMP candidate

## Why this patch

Water0001 benchmark15278929 measured `_stockholder_q` at64.913s/1thread and
24.474s/8threads, **45.1%** of54.247s schema8. PFIT15.382s is a separate
backend experiment; all11 repeated P constructions cost only1.819s.
The Q span includes Python copies as well as native integration, so it is not
a native-accumulator-only timer. No speedup for this candidate is claimed yet.

Full analysis outside source:
`/home/awallace43/.cache/ccasp-native-perf-water001-analysis.md`.
It compares every local/refined tensor entry and every C6/C8/partial-C10 value.
Fresh full-chain results already vary by up to3.861e-7 in local tensors versus
thread1; they are not a bitwise kernel oracle. All existing native-verified
flags are false; numeric transport passes do not grant solver certification.

## Scope and invariants

This separate worktree/branch starts at8359155a10; the original profiler
worktree and its `devtools/isapol_native_profile/` files remain untouched.

1. `psi4/src/psi4/libisapol/partitioned_response.cc`:
   - Original site order, <=4096-point/8MiB collocation batches and full screen/
     basis validation remain unchanged.
   - Serial ratio/cutoff/negative/excluded/harmonic validation retains original
     point order and messages, before the parallel accumulation.
   - Each batch prepares only its bounded point/component prefactors and active
     flags. Exclusions skip the update, not a synthetic zero addition. Screened
     basis zeros are still multiplied/added.
   - Existing `parallel_work` owns disjoint component/column intervals, at most
     **25*8=200 tasks**, with static scheduling and no nested OpenMP. Each output
     sees **all valid points in original order across all original batches**.
     There are no reductions, point partial sums or BLAS calls.
   - Direct output pointers replace Matrix accessors in the inner loop. No
     worker mutates metadata, counters, operands or any other output interval.
2. `psi4/driver/procrouting/isapol_distribution.py`: explicitly charge native-Q
   scratch in **both ISA and MBIS** callers of this shared constructor. No
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
Workspace is freed/reused per batch, never a full-grid cache or public knob.
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
callers, and exact disjoint output coverage with <=200 tasks.

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

The **32 new API cases** use frozen explicit Gaussian operands, no SCF. They
compare the original serial p/component/AUX arithmetic (using native fixed
basis/harmonic samples, no point reduction) to native1/2/4/8 via uint64 views.
Cases include Cartesian/spherical AUX, ranks0–4, multiple/permuted sites,
screens/zero columns, signed-zero/underflow/subnormals, cancellation, cutoff
neighbors, repeat/race-sensitive output checks, tile/8MiB/4096 boundaries,
empty grids and exception precedence across old batch boundaries.

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
Repeat end-to-end water only via a newly qualified candidate request, without
mutating accepted inputs or frozen835 profiler pins.

No merge, push, deployment or new numerical/speed/scientific acceptance is
implied by this candidate commit.
