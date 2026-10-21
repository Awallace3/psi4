# One private Phoenix build/smoke allocation — proposal, not submitted

Attempt002 preserves failed job14254085 and its source/run directories.
That job failed before CMake because command recording tried to JSON-serialize
`Path` arguments. `call` now normalizes argv once before recording and executing
it. Two stdlib-only regression cases cover identical recorded/executed argv and
preservation of a nonzero child exit code; they execute no external command.
Attempt002 uses a new pinned `build-smoke-v2` source, `build-smoke-002` run and
`-attempt002` scratch parent. Resources, dependencies and science are unchanged.

Attempt002 reached CMake and correctly refused the selected environment's
outdated QCElemental. Attempt003 preserves that failure and exposes ONLY the
already-installed QCElemental0.50.4, QCEngine0.50.0 and OptKing0.5.0 packages
and exact dist-info directories from the same old dependency stage through
`run/deps/python`. No old Psi4 package, installation, download, requirement
change or whole-prefix overlay is involved. An explicit compute-only import/
version/path check runs before CMake; the native smoke verifies the same
module identities. New source/run/scratch are `build-smoke-v3`,
`build-smoke-003`, and `-attempt003`. Resources and science remain unchanged.

The four original adapter files at `32229e6d10817cd8682e9fdfcabf994e1e588c7e`
are unchanged. Independent adapter review is statically READY, answer SHA256
`a912115497d21f208b97834ff4e3e57c8b852e930930f8bb5e06cadb3c475e59`.
No adapter or benzene calculation is part of this job.

## Dependency selection

Use existing **p4cuest_sapt**, not p4qca. This is a deliberate profile deviation:
the reused Einsums 1.1.3 config records p4cuest_sapt's GNU 14.3 compiler, and both
the library and Python extension's unchanged RPATH name its `lib`. Its installed
Python 3.13.11/Libint 2.13.1/Libxc 7.0.0/gau2grid 2.0.8/NumPy 2.4.2/adcc 0.16.1/
MKL 2025.3/HDF5 1.14.6/fmt 12/spdlog 1.17/cpptrace 1.0.4/cpp-argparse 3.2/
pybind11 2.13.6/pytest 9.0.2 metadata and narrow input paths were inspected.
No import or numerical/backend compatibility is inferred from that metadata.

`build_smoke.py prepare --commit COMMIT --tree TREE` is a stdlib-only,
data-staging operation, intended only after the final source is Git-published.
It creates the fixed exclusive run/log/scratch parents and `run/deps`:

- byte-identical root `.cmake` files from the exact old Einsums and TargetHDF5
  config directories, with source/hash records; **zero relocation edits**.
  Relative `_IMPORT_PREFIX` now resolves to the new dependency-only prefix.
  TargetHDF5's explicit transitive paths already name the selected environment.
- only named Einsums/EinsumsExperimental/EinsumsPy header subtrees;
  exact 1.1.3 library/SONAME names; three Python package/extension files.
- no old `include/psi4`, whole stage `lib`, whole Python prefix, tests or
  `__pycache__` exposure. No existing file is modified, no ELF/RPATH patch,
  no package install or dependency rebuild.

Numerical hashes happen on compute. The exact shared library must match the
existing Phoenix campaign pin
`cf1f411130bc902a3ca74280cf23b3e7f325843de8cf38ec4a95fcb6f917faf2`.
Before scientific imports, the smoke child checks that pin, rejects an early
Einsums mapping, and keeps an exact `ctypes.CDLL(NOW|GLOBAL)` handle alive.
The environment's default Einsums 1.1.2 is never an accepted role.
No MKL threading/interface override is selected.
The exact optional D/lib/libEinsumsExperimental.so.1.1.3 is also hashed/pinned
in the compute receipt; only that additional Einsums family file is allowed.

Explicit OpenMP is selected at build time as this environment's `libomp.so`.
Its `libiomp5.so` and `libgomp.so.1` aliases resolve to that same file, while
unversioned `libgomp.so` resolves to a distinct GCC runtime and is not selected.
Actual maps/hashes, including whichever MKL threading library naturally loads,
are recorded before/after tests. Multiple OpenMP or wrong-prefix key libraries
are refused; no backend equivalence to workstation evidence is claimed.

## Build operations in order, only after submission grant

1. Confirm allocation, pinned HEAD/tree, clean source and unchanged accepted
   adapter hashes. Bind exact environment Python/compiler paths, clear site
   compiler/include/library/Python overrides, verify cc1/cc1plus.
2. Outer configure: pinned local packages, all optional add-ons off except
   Einsums, required find/insist pins, disconnected fetch settings. Inspect
   generated dependency commands/config graph before any compilation.
3. Execute only generated lapack/hdf5 **config-data** install scripts into
   private stage; refuse scripts describing numerical/executable installation.
4. Explicit **direct inner configure**, using the generated initial cache but
   correcting flags explicitly: numeric shell ordering **2**, O3 Release,
   C++20, unity ON/batch8, march-native. Do not execute automatic EP core flags.
   Verify actual compile graph and per-isapol `-ffp-contract=off`; refuse
   old-source/p4qca/dependency-build/network paths.
5. Direct inner `--target core --parallel 1`, then `update_version --parallel 1`,
   then `cmake --install INNER --prefix PRIVATE_STAGE` (project only).
   Never outer/default/all/install builds or a shared numerical install.
6. Record private core SHA/ELF metadata. Obtain PsiAPI from **this** absolute
   `stage/bin/psi4 --psiapi`, with explicit selected Python. Start smoke using
   `-I -S -B`, then explicitly add only stage, isolated Einsums and selected
   environment package roots. Initialize empty PYTHONPATH before evaluating
   PsiAPI, preserve its command status, and bind PSIDATADIR to this stage's
   share/psi4 before scientific imports. Bind job-owned TMPDIR/SCRATCH/PSI_SCRATCH.

All commands retain full phase logs, exact argv and natural exit codes.
The shell waits for an owned `srun` child; a TERM/INT/HUP or nonzero exit cannot
write `complete:true`. No watchdog, lifecycle experiment, retry or chain.

## Five exact smoke selectors

Stored literally in `build_smoke_plan.json`, source-checkout pytest, importlib
mode, plugin autoload/cache disabled, fresh temp directory and JUnit:

1. `test_isapol_restricted_state.py::test_restricted_state_snapshot_matches_native_wavefunction`
2. `test_isapol_recoupled_dispersion.py::test_closed_dipole_and_sine_rank_one_symmetry`
3. `test_isapol_recoupled_dispersion.py::test_frames_metadata_no_rotation_no_distance_and_ordered_pairs`
4. `test_isapol_bounded_oeprop.py::test_refinement_parameter_limit_defaults_to_64_and_passes_through`
5. `test_isapol_bounded_oeprop.py::test_live_oeprop_publishes_latest_result_and_clears_it_on_refusal`

These cover a real HF/STO-3G restricted water state; analytic angular/native
recoupling; opt-in 72 pass-through at the property ABI; and a tiny PBE0/STO-3G
water/32-point/three-node property chain. The pass-through test queries the
word `benzene` as preset metadata only; it builds no benzene molecule.
The water chain is analytic DF-centre distribution, **not** molecular ISA
benzene or production scientific-pilot acceptance. The smoke uses one
numerical thread and 4 GiB Psi4 memory inside the 48G allocation.

Allocation: one node/task, 8 allocated CPUs, **one compiler process at a time**,
48G, 08:00:00, gts-cs207-chemx/cpu-small/explicit embers, no GPU.
Only `sbatch --test-only` is permitted during preparation. Exact final commit,
tree, script hashes, test-only result and final proposed submission argv are
published in the external build-plan handover, not guessed here.

Read-only static build-script review `dd9c6e8559c36f1bfd53faf6352be46f9`
identified PsiAPI's unset-PYTHONPATH failure and two targeted binding corrections;
all were corrected before publication. It is not a compute/runtime admission.
