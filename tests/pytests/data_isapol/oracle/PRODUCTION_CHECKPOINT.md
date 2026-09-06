# Production ISA-A checkpoint development workflow

These opt-in tools exercise actual CamCASP density/basis/grid construction and
export one frozen update for `psi4.core.isa_a_fit_step`. They do not implement
native Psi4 Drho-C fitting, the ISA controller, or end-to-end properties.

## Protocol distinction

The local `H2O-isagrid/H2O.cks` inspected for this increment requests **Doo-C and
BVLS**, not Drho-C and LU. Never compare its final constrained coefficients to
an unconstrained LU solution and call that a solver regression.

`prepare_isa_water_run.py` creates a distinctly named adapted protocol:

- archived orbitals/OBS, Cartesian aVTZ molecular AUX, spherical aVTZ AtomAux
  with ISA set2, radial 100/angular request 200;
- explicit NN density fits with lambda 1000 and 0, eta/gamma zero;
- **Drho-C, Algorithm A, LU**;
- archived convergence/activation/tail controls, no response or GDMA stages;
- source-default numerical integration controls: this source rejects the archive's
  `SET Num-Int-Pars` block, so the preparer explicitly omits and records it.
  Initial execution resolved angular request 200 to **230 actual** points;
- omit newer `W-TAILS / FIX = ON` syntax: the inspected source defaults tails ON
  but does not recognize that explicit keyword. This adaptation is also recorded.

It is **not the modern ISA-Pol preset**. Expanded input and hashes of included
basis/input/orbital files are written beside the run. Orbital provenance is
inherited; no new SCF is performed.

## Prepare isolated source and inputs

Run from the Psi4 worktree root, substituting local reference paths:

```bash
TOOLS=tests/pytests/data_isapol/oracle
CAMCASP_SOURCE=/path/to/CamCASP
ARCHIVE=/path/to/H2O-isagrid
python "$TOOLS/capture_isa_checkpoint.py" --camcasp "$CAMCASP_SOURCE" \
  --destination "$PWD/.pi/audit/production-camcasp"
python "$TOOLS/prepare_isa_water_run.py" --archive "$ARCHIVE" \
  --camcasp "$CAMCASP_SOURCE" --destination "$PWD/.pi/audit/production-water"
```

Destinations must not exist. Neither tool modifies the reference tree/archive.
All source anchors must match uniquely before instrumentation is written.
Do not redistribute the generated upstream source or executable.

## Build the reference

Use a **serial** make: this legacy Makefile can compile the same module twice
under `-j`, causing `.mod` races. The local development build command is:

```bash
cd .pi/audit/production-camcasp
GIT_CEILING_DIRECTORIES="$PWD" make -j1 NAME=camcasp LIBS='-llapack -lblas' LDFLAGS='' \
  FFLAGS='-O2 -DG77 -DCADPAC -DGAMESS -DSAPT2002 -DSIGNED_INTEGER -DERF -DF2003 -fno-backslash -fimplicit-none -fallow-argument-mismatch -ffp-contract=off' \
  FFLAGS2='-O2 -DG77 -DCADPAC -DGAMESS -DSAPT2002 -DSIGNED_INTEGER -DERF -DF2003 -fallow-argument-mismatch -ffp-contract=off'
```

This is source/compiler dependent, not a promise that every CamCASP revision
builds unchanged. Record actual command/compiler, linked libraries, executable
hash and logs. The Git ceiling prevents the scratch build from reporting the
parent Psi4 repository's revision as CamCASP provenance; use the captured source
hashes, not ambient VCS strings. `plan.md` records current evidence/blockers.

## Capture and replay

Run the instrumented executable in the new water directory, feeding
`H2O-expanded.cks` to stdin and retaining stdout/stderr. Set reference environment
paths as required by your installation. `ISAPOL_CHECKPOINT_CALL=N` selects the
1-based `make_Stilde_stockholder_A` call; unset means tracing is disabled. Use a
fresh run directory per capture: output `isapol-checkpoint.dat` is opened with
`status='new'` and is never overwritten. The latest instrumenter rejects anything
other than unperturbed density ISA, Drho-C, A, LU, and uncontracted atomic shells.

The current producer emits **v2**; the reader also accepts v1. A checkpoint is
complete only with the final `END` record after solved D and (in v2) raw shape
s-projection are available. See `BASIS_RECONSTRUCTION.md` for descriptor details. Fortran `STOP` may return a misleading shell status; always verify
completion and parse the stream. Sampling preserves actual neighbour/site/batch
order, signed samples, tail-processed shapes and shell component normalization.
The input overlap is captured **before** damping/ridge; RHS is captured **before**
the LU solve overwrites it. Raw distances are recomputed from point coordinates,
not read from CamCASP's already weighted/capped `rA2` workspace.

```bash
export PATH=/home/awallace43/miniconda3/envs/p4_ci/bin:$PATH
export PYTHONPATH="$PWD/build_camcasp_psi4_joint/stage/lib"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -P \
  tests/pytests/data_isapol/oracle/replay_isa_checkpoint.py \
  .pi/audit/production-water/isapol-checkpoint.dat \
  --report .pi/audit/production-water/replay.json
```

The report includes absolute and globally scaled errors for metric/RHS/D/population,
normalized solve residual, excluded points, extension path and stream hash. CLI
failure means tolerance was exceeded; it must not be suppressed. Its default
scaled tolerance is 1e-9, not an independently established production tolerance.
Keep `run-provenance.json` and `capture-provenance.json` with it. Synthetic parser
unit tests are **format tests only** and do not establish production parity.

## Measured execution

The local adapted reference compiled and converged in 53 iterations. First and
activated oxygen checkpoints (68,310 points / 109 functions) passed C++ replay.
Largest observed coefficient absolute error was 2.8911e-12; largest normalized
solve residual was 5.3526e-17. Seven final shape/tail files were byte-identical
across early-traced, activated-traced and untraced runs. See
`../camcasp_isa_production_evidence.json` and root `plan.md` for precise errors,
hashes, artifact paths and limitations. These are production sampled-kernel
checks, not native wavefunction-to-property parity.

The v2 follow-up exports full density/basis/shape descriptors, validates independent
reconstruction, adds activated H1 replay and provides three portable bounded basis
fixtures. Its maximum H1 replay residual is 7.9801e-17. See
`BASIS_RECONSTRUCTION.md` and `../camcasp_isa_basis_evidence.json`. Large full-fit
streams still remain local; bounded basis fixtures are not full replay fixtures.

## Full exported-input provider reconstruction

The optional `--reconstruct-providers` replay mode requires v2 descriptors. It uses
`IsaAFitProvider` to reconstruct **all** recorded atomic basis and molecular density
samples, primitive function metadata, raw squared distances and analytic weighted
overlap in C++, then solves the frozen fit and projects its raw shape coefficients.
The Python adapter only converts descriptor indices and storage; the independent
polynomial oracle is unchanged. Each compared input, metric/RHS/D/population, raw
shape and normalized residual is reported. The default replay path continues to
use captured numerical samples.

```bash
python -P tests/pytests/data_isapol/oracle/replay_isa_checkpoint.py \
  .pi/audit/basis-water-activated/isapol-checkpoint.dat \
  --reconstruct-providers \
  --report .pi/audit/basis-water-activated/provider-replay.json
```

Use the staged environment from `plan.md`. Choose a **new report path**, preserving
existing sample-only evidence. Full streams contain 68,310 points; the portable
97-point descriptor fixtures are not complete fitting inputs. This replay still
uses **captured quadrature, old coefficients, screened/tail-processed shape and
shape-sum samples and a supplied density neighbour list**. It does not reconstruct
active tails, generate Drho-C coefficients/basis recipes, execute a controller, or
establish end-to-end parity. The current captures belong to the adapted legacy
reference track, not a future CamCASP Libint2 branch. See `plan.md` for measured
results and any blockers; the existing scaled tolerance is not a scientific error
budget for unrelated protocols.

## Whole-sweep controller transition capture (in development)

`capture_isa_sweep.py` extends the diagnostic harness in a **new** source copy.
It preserves the v2 atom format and adds a separate `ISAPOL_SWEEP_STATE 1` sidecar.
Select an ordinary iteration using `ISAPOL_CHECKPOINT_SWEEP=N`; simultaneous
single-call/sweep selectors are rejected. Initial scope is three-site ordinary
A/W/Drho-C/LU, Func-1/Fit-3, s-block weighting, no DIIS/symmetry/decoupled or
self-consistent tail loops. Unsupported controls fail explicitly.

```bash
python -P tests/pytests/data_isapol/oracle/capture_isa_sweep.py \
  --camcasp /path/to/CamCASP --destination /new/scratch/source
# Build serially using the reference compiler/flags above; prepare a new run.
ISAPOL_CHECKPOINT_SWEEP=21 /new/scratch/source/bin/camcasp \
  < H2O-expanded.cks > reference.log 2>&1
python -P tests/pytests/data_isapol/oracle/replay_isa_sweep.py /new/run \
  --report /new/run/controller-replay.json
```

Output is exclusively created as `isapol-atom-{1,2,3}.dat` plus
`isapol-sweep-state.dat`. The sidecar records configured/active controls; PRE and
committed POST full D/D0/W/W0 vectors; saved charges; convergence flags/deltas;
shape neighbours; and Func-1 cutoff/A/b state. It associates exact atom call IDs
and writes a final marker only after all atom streams and the sweep commit.
No diagnostic evaluation or state-changing open is added. Meaningless undefined
reference IP and unused Func-1 parameter slots are deliberately not exported.
The existing archive/captures and single-call mode remain separate.

Initial reference `ISAcharge` can be NaN because legacy pre-fit rescaling divides
zero saved charges by their zero sum. That third charge field is retained as an
explicit diagnostic token/undefined numeric value, never fed to the controller.
The two saved shape charges and all coefficients remain strictly finite. Undefined
tail-function index -1 is accepted only for an undefined tail; a defined tail must
have the supported function and valid parameters.

The strict reader cross-checks each v2 atom stream against the sidecar before
reconstructing a C++ controller step. It compares full fitted/mixed states, saved
charges, deltas/flags, next controls and defined tail parameters. The source's
stale saved-A gate versus the deterministic C++ tail policy remains an explicit
limitation. This is **one exported-input controller transition**, not independent
native generation or end-to-end parity. See `plan.md` for actual build/run/test
status; capture scaffolding alone is not a passing production gate.

## Full C++ controller trajectory from exported initial inputs

After validating the initial whole-sweep capture, run:

```bash
python -P tests/pytests/data_isapol/oracle/run_isa_controller.py /initial/run \
  --report /new/path/cpp-controller-trajectory.json
```

This calls `IsaAController.run` from the captured first entry state; subsequent
shapes, tails and activation decisions are generated in C++. It retains history,
final coefficients/tails, timing and initial-stream hashes. It still uses exported
basis descriptors, fixed density coefficients and quadrature. The CLI fails on
nonconvergence, but convergence is **not** a final-reference or end-to-end parity
certificate. Compare the final state with a separately captured converged reference:

```bash
python -P tests/pytests/data_isapol/oracle/compare_isa_trajectory.py /trajectory.json \
  --initial /initial/run --reference /final/run --report /new/path/comparison.json
```

The strict comparator requires matching iteration count, convergence flags, deltas,
MaxDelta, coefficients, charges, controls and tail cutoffs/parameters. It checks
history consistency and fixed cutoff/configuration agreement at captured endpoints.
It retains the existing joint A/b scaled denominator and reports individual parameter
errors separately. Endpoint checks do not certify every intermediate reference state.
Distinguish numerical trajectory differences from represented-function/property
agreement; the latter must not silently replace a failed raw-parameter gate.

## Remaining acceptance work

- Typed C++ explicit-basis/fixed-density sampling, co-centred overlap, shape mapping
  and frozen-fit assembly are implemented; continue native recipe/DF generation
  separately, keeping the polynomial reconstruction as an independent oracle.
- Add explicit iteration/phase selectors and intermediate activation transitions;
  capture actual states, not merely requested settings.
- Implement and validate active tails and the synchronous ISA controller separately.
- Preserve trace-on/off checks when extending the diagnostic format.

The source inspection also found that `overlap_integrals.F90` does not apply
all-block W-Eps as mathematically intended when `s_block_only` is false. This
workflow retains the true source metric and uses s-block-only. It does not certify
the all-block upstream implementation.
