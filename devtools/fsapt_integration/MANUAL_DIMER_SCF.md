# Manual cachy dimer SCF investigation

Worktree: `~/gits/psi4-fsapt-opt-cachy/source`.
Edit `build.sh` directly at the **repository root**; all generated artifacts
live under `./tmp/`. This workflow does not overwrite any sibling previously
qualified installation, but editing/advancing the source means it no longer
matches the old frozen qualification.

```bash
cd ~/gits/psi4-fsapt-opt-cachy/source
nvim build.sh                     # repo-root editing; C++ LSP uses the link below
./build.sh                         # build/install, then GPU cold + warm peptide SCFs
./build.sh build                   # build only, no GPU calculation
./build.sh run --repeats 1         # existing install, fresh-process cold SCF only
./build.sh compare --repeats 2     # separate CPU and GPU processes, each cold + warm
./build.sh paths                   # print resolved paths; no setup/build/GPU work
PSI4_CUEST_SCF_TIMERS=0 ./build.sh run  # instrumentation-off control
FSAPT_BUILD_JOBS=2 ./build.sh build    # inherited by nested Ninja, not only outer build
```

Default: 24-atom peptide **dimer-only HF/DF SCF**, jun-cc-pVDZ /
aug-cc-pVDZ-JKFIT, FP64, E10/D12, four host threads. No monomers, localization,
FSAPT, or independent reference calculations obscure the SCF startup.
Uses SAPT(DFT)'s `scf_helper("SCF")` call and `SAVE_JK`/`DF_INTS_IO=SAVE`
policy. The `SAPT(DFT):Dimer SCF` timer excludes explicit wavefunction teardown,
which has its own `Manual wavefunction cleanup` timer.
`--e-convergence`, `--d-convergence`, `--maxiter`, `--repeats` are editable;
the test itself is `devtools/fsapt_integration/peptide_scf.py`.
The frozen geometry manifest is `tmp/inputs.json`.

Layout:

```text
build.sh
compile_commands.json -> tmp/obj/psi4-core-prefix/src/psi4-core-build/compile_commands.json
tmp/
  obj/             # CMake/Ninja objects, native core in nested psi4-core build
  install/         # this manual workflow's installation
  runtime/         # private cuEST/CUDA runtime
  inputs.json
  runs/
```

The compile database becomes available after the first successful native build.
The script refuses to replace a different existing root compilation database.
`tmp/` and the database link are local, ignored build artifacts.

Each invocation creates `tmp/runs/scf-<timestamp>-<unique>/{cpu,gpu}/`:

- `cold.out`, `warm-1.out`: Psi4 output, **timer table appended at the bottom**;
- `cold-timers.json`, `warm-1-timers.json`: complete native timer paths/calls;
- `scf-result.json`: options, energies, full timings, import paths, run status;
- `console.log`: also ends with timing summaries;
- `../build.json`: last build HEAD/diff hash and whether it was dirty.

Cold means first SCF in a fresh Python process, **not** a cleared driver/JIT/disk
cache. Warm repeats share runtime handles but construct fresh SCF objects.
They are diagnostic contrasts, not independent timing replicates. No CUDA
warm-up or added synchronization occurs before the first SCF.

## New opt-in subtimers

Set `PSI4_CUEST_SCF_TIMERS=1` (default in build.sh) to record:

- CUDA device discovery, context/stream, cuBLAS, cuSOLVER, cuEST handle creation;
- AO shells/basis setup, workspace query/allocation/create/cleanup;
- Mints AO-pair and one-electron plans, including cached basis setup;
- overlap/kinetic/potential allocation, charge packing/H2D, workspace query,
  compute API, workspace freeing, D2H, and output freeing;
- JK plan validation, primary/auxiliary bases, AO pairs, DF workspace and plan;
- workspace host allocation, `cudaMalloc`, and `cudaFree`.

These are **inclusive host wall timers**, nested under the existing SCF stages;
do not add parents and children together. No new CUDA barriers are inserted:
an asynchronous compute call can enqueue quickly and have its wait charged to
workspace free or D2H. This locates the host-visible latency without pretending
to measure device kernel time or identify JIT internals that cuEST does not expose.
Use a separate Nsight/CUDA-event experiment if that remaining split matters.

The native changes are in `psi4/src/cuest_runtime.cc`,
`psi4/src/psi4/libfock/cuESTCommon.h`, `cuESTJK.cc`, and
`psi4/src/psi4/libmints/{basisset,mintshelper}.cc`.

## Safety and provenance

The runner acquires the shared `~/.cache/psi4-cachy/gpu0.lock` nonblockingly
and respects `gpu0.HOLD` before starting. Do not run another unmanaged GPU
calculation concurrently. The shared dependency environment is read-only;
cuEST/CUDA runtime files are private copies. No qualified prefix or queued
Phoenix checkout is modified. The older separate manual root is preserved,
not moved or deleted. Dirty manual edits are allowed and recorded;
`run` uses the **last installation**, so rebuild after changing C++ or Python.
RTX 3090 numbers are debugging evidence, not H200 speedup claims.
