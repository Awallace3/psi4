# cuEST/cuBLAS CPHF/CPKS acceleration — handoff

## Current objective and scope

User requested implementing cuEST + cuBLAS acceleration for CPHF/CPKS.
After clarification, selected **GPU JK/GEMM first (Recommended)**:
- accelerate HF and existing supported LDA/GGA CPKS response builds;
- CPKS uses **CUEST_XC=false**, retaining its native CPU XC response kernel;
- leave GPU XC response and existing MGGA/VV10 response-kernel limitations
  explicit, rather than silently returning missing XC contributions.

Tasks: #21 inspect, #22 implement, #23 validate.
#21 and #22 are complete; #23 validation is in progress. Response edits and
regression tests exist, but neither GPU numerical validation nor speedup
evidence is complete. The checkpoints below are chronological; use the
latest checkpoint for current state.

## Concurrent benchmark: keep separate

Lower-rung water benchmark **13976174** is queued on free embers, watched
by **b04540077**. Harness pin **e44ae0eb8f1a4c076e5a1ad54114eeb57ea9d83c**
in `/home/awallace43/gits/psi4__worktrees/cuest-dft-rungs-20261012` and
remote `/storage/project/r-cs207-0/awallace43/gits/psi4.cuest-dft-rungs-20261012`.
Freeze that checkout and both historical staged packages while queued/running.
Do not rebuild/upgrade its environment or staged binaries for response work.
Task #18 is pending while its automatic watcher remains active; resume on
terminal notification. Detailed benchmark continuation is in `mgga-check.md`.

## Verified cuEST APIs

NVIDIA primary documentation:
- release notes: https://docs.nvidia.com/cuda/cuest/0.2.2/release-notes.html
- exchange: https://docs.nvidia.com/cuda/cuest/0.2.2/c-api/methods/exchange/index.html
- Coulomb: https://docs.nvidia.com/cuda/cuest/0.2.2/c-api/methods/coulomb/index.html
- XC primitives: https://docs.nvidia.com/cuda/cuest/0.2.2/c-api/methods/exchange-correlation/index.html

Latest documented version is **0.2.2**. Nonsymmetric DF exchange appeared in
**0.2.0**, and supports one left coefficient matrix with batched right matrices.
Both coefficient buffers are occupied-index-first, row-major;
output is batched AO K. Plan parameters encode α/r + β erf(ωr)/r.
General response D can be passed to `cuestDFCoulombCompute`.
Nonsymmetric grid-density primitives are also available, but out of the
initial native-XC response scope. cuEST does not supply a complete CP solver.

Phoenix environment `/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt`
has conda `libcuest`/`libcuest-dev` **0.2.2.2**; library symlinks resolve
to `libcuest.so.0.2.2`.
Verified header macros major=0/minor=2/patch=2 and actual declarations of
`cuestDFNonsymmetricExchangeCompute`, its workspace query, and
`cuestDFCoulombCompute` in included DF headers. The umbrella `cuest.h` only
includes those headers; searching that file alone misleadingly returned false.
No environment changes were made.

## Source map and preliminary findings

Development cwd HEAD **9317f406b2b66f33b65866cbc44a4189f913339e**, with
user-owned working changes. Preserve them.

- `psi4/src/psi4/libfock/cuESTJK.cc:330–545` already handles distinct
  left/right coefficients via nonsymmetric exchange, currently one RHS per
  call. Current dirty code handles varying/zero occupied columns and reuses
  per-build coefficient scratch.
- `cuESTJK.cc:140–166`: α/β/ω are encoded in the DF plan.
- `cuESTJK.h:109–112`: enabling long-range exchange also enables combined K.
- `psi4/src/psi4/libscf_solver/rhf.cc:397–637`:
  `onel_Hx`, `twoel_Hx_full`, transforms, combined/uncombined return paths.
- `uhf.cc:444–780`: corresponding spin-blocked paths.
- Response uses `R = −Cvir Xᵀ`, then `D = Cocc Rᵀ`.
  RHF singlet assembly is 4J − α(K+Kᵀ) − β(wK+wKᵀ) + 4Vx;
  UHF assembles spin Coulomb sums with factor 2.
- Current response assembly sets α/β=1 under USE_CUEST but still accesses
  separate wK, and only includes K when `is_x_hybrid`; inspect/fix the
  **pure long-range, α=0** case because cuEST embeds wK in K.
  Preserve combined/uncombined operator ordering and triplet behavior.
- `psi4/src/cuest_gemm.cc:188–329` already provides
  `psi::cuest_chain_gemm` using cuBLAS on the cuEST stream, retaining chain
  intermediates on device. It has no public C++ declaration/header yet.
  It supports one-irrep matrices only; symmetry-blocked and zero-size chains
  need deliberate CPU fallback. Use existing
  `CUEST_GEMM_MIN_DIM` transfer-work threshold (default 256; zero forces
  otherwise eligible chains), rather than inventing inconsistent heuristics.
- `v.cc:2912–2940` and `5641…`: GPU XC response guards are intentional.
  Native RKS response also rejects MGGA and VV10; preserve these limitations.
- `tests/pytests/test_cuest_jk.py` already covers asymmetric K, weighting,
  α reset, changing/zero occupied buffers and math-mode switching.
  Preserve these user-owned test changes.

Independent read-only delegate **dcac8a982dbfbdb5caaf137fb298b46c7**
(*Trace CPHF response JK*) completed; its verified result was retrieved.

## Implementation in progress (2026-10-12)

- `JK::returns_combined_exchange()` exposes the preweighted-K/no-wK contract;
  cuEST overrides it. RHF/UHF response now consumes K once for global/partial/
  pure long-range hybrids, independent of the global USE_CUEST flag.
- `JK::set_omega` is virtual; cuEST rebuilds a live plan when omega changes,
  matching its existing alpha/beta lifecycle handling.
- `psi4/src/cuest_gemm.h` publishes the existing cuBLAS chain internally.
- `libscf_solver/cpscf_gemm.h` supplies scoped doublet/triplet and four-factor
  one-electron products, FP64 chains with device intermediates, existing
  threshold semantics, and native fallbacks for empty/symmetry/small products.
- RHF/UHF response uses these transforms. Projection tracks each returned
  component's input, including separate J/K outputs; UHF uses flattened spin
  indices and can initialize both AO and SO coefficient sets in a mixed batch.
- `tests/pytests/test_cuest_cphf.py` covers native separate-operator projection,
  mixed-symmetry UHF batches, fixed-orbital RHF/UHF HF/SVWN/BLYP/PBE0/CAM-B3LYP/
  LRC-wPBE comparisons, forced/bypassed GPU transforms with timer assertions,
  solver residuals, and explicit unsupported-XC-kernel guards.
- Added omega-reset regression to existing `test_cuest_jk.py`.
- Added scope/transfer/threshold documentation to `sapt.rst`.

Background tasks (automatic notifications; do not poll merely to wait):
- **b7432d2af**: CPU response baseline, using the pre-edit staged binary;
  expect a failing separate UHF projection or mixed-symmetry regression.
- **bba53deac**: CPU build of changes (log `/tmp/cuest-cpscf-cpu-build.log`).
- **b6dc5abda**: read-only Phoenix dependency-cache inspection for a new,
  isolated GPU build. No remote response build or job has been submitted yet.

Local execution must start outside the source root, e.g. `/tmp`, to avoid
importing the source `psi4` package instead of the staged binary. Python is
`/home/awallace43/miniconda3/envs/p4_ci/bin/python` (3.13), PYTHONPATH is
`build_saptdft_cuest/stage/lib` (absolute). `git diff --check` and Python
syntax compilation pass. Do not mark #22/#23 complete based on those checks.

### Latest validation checkpoint

- CPU build **bba53deac** succeeded and staged the changed C++ core.
- First baseline test **b7432d2af** failed in the harness (Matrix Python
  dimensions are `coldim()`/`rowdim()`, not C++ `colspi()`/`rowspi()`).
  Corrected; no claim that this was a reproduced response bug.
- **b44fd6612** and **bc0ea0b8f** pass both native separate-operator projection
  cases, then timeout at mixed-symmetry UHF after 180/240 s. The SCF is already
  complete in under a second; timeout is post-SCF. Task #24 tracks this blocker.
- Stack-trace probe **bb0918fb1** runs only mixed-symmetry with Python
  faulthandler timeout 15 s, outer limit 50 s.
- Independent unaffected-suite check **b5b03a49e** excludes mixed-symmetry.
- Review delegate **db1d93c765768ec0c27d60a9812eff24f** completed; verified
  result retrieved. No static response algebra/indexing flaw found, but live
  omega plan rebuild could leave invalid objects on failure. Now destruction
  invalidates the plan first, compute rejects uninitialized plans, and
  preiterations cleans partial state before explicit reconstruction. Added
  finalize/reinitialize checks to omega regression. GPU execution still needed.
- Dependency inspection **b6dc5abda** completed. New validation payload:
  `devtools/benchmarks/cpscf_validate.sh`, isolated build/stage; SBATCH script:
  `/home/awallace43/docs/saptdft/cuest/cpscf/20261012-cpscf-validation-01/job.sbatch`.
  Uses H200/8CPU/64G/2h **embers**, read-only installed historical dependencies.
  **Not staged/submitted yet**; resolve CPU timeout before snapshot/push.
- The planned GPU suite expects 25 CPSCF and one JK test, no skips. Update
counts if adding cases. Freeze its new source pin once submitted.
- Lower-rung water archive independently audited (see mgga-check.md):
  90/90 SCFs, 20/20 comparisons pass. Larger-control submission **b831edefd**
  is running independently on same frozen DFT harness pin.

### Mixed-symmetry root cause and correction

Unaffected CPU check **b5b03a49e**: **6 passed, 18 GPU skips, one mixed test
deselected**, 16.17 s.
Faulthandler **bb0918fb1** shows a post-SCF library debugger wait, not slow SCF.
Native GDB probe **be710da59** catches **SIGSEGV in JK::compute_D()**.
The cache checked batch length and symmetry but not row/column dimensions,
so switching SCF SO coefficients to dense AO response coefficients reused
incompatible density buffers and indexed beyond the coefficient irrep count.

`jk.cc::compute_D` now checks row/column dimensions before cache reuse;
`allocate_JK` similarly checks dimensions before reusing the result buffers.
Existing mixed-symmetry regression is the end-to-end seam and now covers
both RHF/UHF, one- and two-electron actions, and separate J/K MO projection.
GPU suite expected count is now **26 CPSCF + 1 JK**, no skips.
Rebuild **b7febb186** is running (log `/tmp/cuest-cpscf-cache-build.log`);
rerun full CPU suite on its completion. Do not stage GPU snapshot until green.
No debugger/probe instrumentation was added to source.

Larger lower-rung jobs submitted: benzene **13979120**, peptide **13979121**;
watcher **be728fd68**, free embers, same frozen scientific harness/binaries.

### CPU regression green — ready for isolated GPU staging

Build **b7febb186** succeeded. Full CPU test **b335ec839**:
**8 passed, 18 GPU-only skips**, 15.37 s. Both RHF/UHF mixed-symmetry cases
now pass; the original cached-buffer crash no longer reproduces.
Task #24 blocker resolved; #22 implementation complete, #23 GPU validation
next. This is not GPU correctness/speedup evidence.

Validation-only source freezer:
`/home/awallace43/docs/saptdft/cuest/cpscf/20261012-cpscf-validation-01/freeze_source.py`.
Uses an alternate git index and commit-tree, includes existing dirty baseline
and required untracked timing/helper/test/payload files, excludes handoff
documents, then creates a separate validation branch/worktree. It asserts
development HEAD and index are unchanged. Do not push/submit until freezing
and shell/Python syntax checks succeed. GPU suite expects 26 CPSCF + 1 JK,
with zero skips.

### GPU candidate staging and shared-JK correction

First isolated snapshot **e3208fc03823d285cf23fbc5f46369bd6a4e0328** was
created/pushed on `cuest-cpscf-validation-20261012` using an alternate index.
Development HEAD/index were verified unchanged. First freeze attempt only
failed because devtools is ignored; explicit `git add -f` of the listed
payload fixed it. No user index mutation or branch advance occurred.
Remote stage/dry-run **b3204d72e** is running; **no GPU response job submitted**.
Do not submit this first candidate: it is superseded by the following finding.

SAPT sets shared cuEST exchange weights to (1,0) before assigning that builder
to monomer `cphf_Hx`. A hybrid's response cannot assume combined K is already
weighted for its functional. Added `cpscf_exchange.h`: derive a scalar only
when the plan's full/long-range mixture is compatible/proportional to the
functional's operator, include inactive omega=0 beta correctly, and reject
incompatible range mixtures explicitly. Bare-K PBE0 response scales by .25
without rebuilding/mutating the shared plan, preserving subsequent SAPT use.
RHF/UHF fixed-orbital GPU regressions now test the bare-K borrowed-plan case
and incompatible CAM omega. Test count remains **26 + 1**.

CPU rebuild **b1d5db373** runs (log `/tmp/cuest-cpscf-exchange-build.log`).
After green CPU regression, create/push a NEW `...-v2` validation snapshot,
with `cpscf_exchange.h` included, and update a NEW stage/batch artifact rather
than racing the first staging command's inputs. Old branch/pin can remain;
do not delete/repoint any worktree. Freeze source helper now defaults to v2.
Task #23 remains in progress; GPU numerical validation/speedup still pending.

### Latest checkpoint — mixed-symmetry solver correction

The first candidate's remote staging/dry-run completed, but it must not be
submitted. Dry-run pseudo-ID 13979687 is not an actual job. Candidate v2
includes the borrowed bare-exchange correction and additional symmetry tests.

New mixed-symmetry **solver** regression passes RHF but native GDB catches
UHF SIGSEGV in `Matrix::apply_denominator`. The solver stored AO/SO flags for
each flattened alpha/beta input, then indexed them by equation instead of
alpha index. All three preconditioner selections now use `c1_input_[2*i]`.
This is separate from the already-fixed JK density/result cache crash.

CPU rebuild **bd61c9c5b** is running. After success, rerun the full latest
suite: expected **10 CPU passes, 20 GPU-only skips**. The isolated GPU payload
now requires **30 CPSCF + 1 JK** executed tests with zero skips.

New artifacts:
`/home/awallace43/docs/saptdft/cuest/cpscf/20261012-cpscf-validation-v2-01/`
contains `stage_remote.sh` and `job.sbatch`. Freeze/push NEW branch
`cuest-cpscf-validation-20261012-v2` only after the CPU regression is green.
No response GPU job has been submitted. Keep old candidate/pin/worktrees.

Lower-rung water is complete and published in PR #12; larger controls
13979120/13979121 remain under automatic watcher **be728fd68**. See
`mgga-check.md`; do not mutate their source pin, staged packages, or environment.

### Corrected CPU suite green; v2 frozen

Build **bd61c9c5b** succeeded. Full regression **b40d3584e**:
**10 passed, 20 GPU-only skips**, 15.63 s. RHF and UHF mixed-symmetry
solvers converge and reproduce their RHS via Hessian action.

Frozen validation v2 pin:
**0294125a547e50cf9126c851c67498f028fe1e7d**, branch
`cuest-cpscf-validation-20261012-v2`. Alternate-index freezer verified the
development HEAD and index unchanged. Local validation worktree is
`/home/awallace43/gits/psi4__worktrees/cuest-cpscf-validation-20261012-v2`.
Staging/push/dry-run **b4ada1fb5** is running. No actual response GPU job
submitted yet. After staging succeeds: live embers headroom and duplicate
check, submit ONE H200/8CPU/64G/2h validation, capture ID/effective QoS,
start automatic watcher. Require 30 CPSCF + 1 JK tests with no skips.

### GPU validation submitted

Actual job **13980976**, effective route read back:
**gts-cs207-chemx / gpu-h200 / embers**, H200:1, 8 CPU, 64G, 2h.
Submission succeeded with 24/50 embers slots used beforehand. Initially
PENDING; automatic watcher **b9114a614** is active. Do not poll merely to
wait. Dry-run pseudo-ID 13980967 is not a submitted job.

Source pin remains **0294125a547e50cf9126c851c67498f028fe1e7d**.
Run root:
`/storage/project/r-cs207-0/awallace43/runs/psi4-cpscf/20261012-validation-v2-01`.
On terminal notification: inspect allocation/batch accounting and bounded
logs, retrieve result metadata/XML, require COMPLETED + VERIFIED.json,
31 actual test passes/zero skips and pinned fresh-package imports. Build
errors, skips, or failures keep #23 incomplete. Confirm actual H200 and
prolog QoS inside the allocation. This job is correctness validation, not
a response speed benchmark.

## Validation constraints

### Response commit and separate FDDS agent

User requested committing the current work then redirected FDDS implementation
to a separate workmux Pi thread. Scoped response commit **4f47db835a**
on `saptdft_cuest` includes 13 source/header/test/documentation files.
Separate SAD/lifecycle/basis-parser changes remain uncommitted and preserved.
Only the response subsection of sapt.rst was staged.

FDDS branch **saptdft_cuest_disp** starts from this commit; workmux handle
**saptdft-cuest-disp**, new sibling worktree. Launch/working confirmation
task **b6865d732** is running. Self-contained prompt:
`/home/awallace43/docs/saptdft/cuest/fdds-agent-prompt.md`.
The agent must implement/test/document optional GPU FDDS dispersion and
commit on its own branch, no automatic merge. Shared environment/historical
packages and response validation worktree are read-only for it.

Response job **13980976** and its pin **0294125a547e** remain unchanged;
the commit does not establish GPU validation or speedups. Task #23 pending
while coordinating FDDS launch; watcher **b9114a614** remains active.

First FDDS launch **b6865d732** failed because project default panes only
run the editor/environment shell, with no `pi` pane. Branch/worktree already
exist at correct commit. Retry **b70c900a2** uses existing worktree,
`--open-if-exists -l pi -H`, explicit parent session psi4, and the global
named Pi layout (no global config changes, no duplicate worktree).
Wait for its working-status confirmation before claiming the agent is running.

FDDS agent **confirmed working** by **b1f6204b2**. Layout/agent flags
conflict prevented b70c900a2; b8627c19f created the Pi session using
invocation-only `fdds-workmux.yaml` but prompt injection did not start work.
Explicit `workmux send ... -f fdds-agent-prompt.md` succeeded; capture
shows the received prompt and active todo tracing FDDS contractions.
No global config changes were made.

Handle **saptdft-cuest-disp**, branch **saptdft_cuest_disp**, worktree
`../saptdft-cuest-disp`, base **4f47db835a**. Parent tmux session **psi4**.
Automatic completion watcher **ba9506d5e** uses `workmux wait` (24h).
Task #26 launch complete; #27 implementation/validation in progress, owner
workmux:saptdft-cuest-disp. On notification capture/review its changes,
tests, commit and handoff; do not blindly merge. No auto-merge authorized.

### Response GPU allocation failed in preflight

Watcher **b9114a614** completed: job **13980976** FAILED 1:0, job/batch
elapsed 12 seconds, payload step elapsed five seconds. No GPU numerical
pass established. Bounded stdout/stderr/configure/build log inspection
**b6304f159** is running; diagnose before any new submission.
Task #23 remains in progress. FDDS agent continues independently on
`saptdft_cuest_disp`, watcher **ba9506d5e**; its implementation is not affected
or cancelled. Do not mutate the frozen failed response snapshot to hide
its provenance; use a new validation pin/run if code/payload needs correction.

Root cause verified by **bf690dd45**: the site GCC module sets `GCC_ROOT`
to its GCC12 installation; conda GCC14 then cannot find `cc1`. Unsetting
`GCC_ROOT` or setting it to the conda environment restores actual executable
frontend paths. Conda activation / LD_LIBRARY_PATH rebinding alone do not.
No package/environment changes were made. FDDS agent was informed.

New isolated validation branch/worktree **cuest-cpscf-validation-20261013-v3**,
pin **ae8cedca770e1bc1a7a6091d313cfc6dec8887d9**, is based on unchanged v2
scientific code and changes ONLY the payload: unset GCC_ROOT after module
loading and require executable C/C++ frontends, recording compiler.txt.
New run **20261013-validation-v3-01** and local artifacts
`/home/awallace43/docs/saptdft/cuest/cpscf/20261013-cpscf-validation-v3-01/`.
Stage/preflight/free-embers submission **b2ed5ce54** is running. On success
capture its actual job ID and start automatic watcher. v2/13980976 evidence
and frozen source remain intact. Suite still requires 30 CPSCF + 1 JK,
zero skips. No GPU code-correctness or response-speedup evidence yet.

Retry submission **b2ed5ce54 succeeded**: actual job **13991128**,
effective **gts-cs207-chemx / gpu-h200 / embers**, H200:1 / 8 CPU / 64G /
2h, initially PENDING. Preflight verified executable conda cc1/cc1plus.
24/50 slots used before submission. Dry-run pseudo-ID **13991127** is not
an actual job. Automatic watcher **b1a5e47d4** active; do not poll to wait.
Latest pin **ae8cedca770e**, run **20261013-validation-v3-01**.
On terminal notification audit actual compiler/build/import metadata,
H200/prolog QoS, 30 CPSCF + 1 JK test passes with no skips, COMPLETED and
VERIFIED.json. Preserve original 13980976 failure evidence.

### FDDS implementation handed back; review/GPU pending

FDDS workmux watcher **ba9506d5e** completed (agent idle/done, NOT GPU
validation completed). Branch **saptdft_cuest_disp** clean at
**6e2e7acce1**; implementation **9fd3a2aa620f**, final CPU-tested
scientific source **099706616a818041f56072c3b2471d6c04451cc8**.
New persistent `CuestFDDSContractions` accelerates uncoupled plus five
hybrid sums; native integral/grid/kernel/susceptibility/quadrature work
remains CPU. Exact handoff: `../saptdft-cuest-disp/fdds-check.md`.

Coordinator audited native CPU XML: **23 executed passes, zero
errors/failures/skips**, 17.362 s; actual staged core SHA256 matches
`21d4115d948e7e35ad727ffbac1dc7ad5c51b023e80d0e2e9d6c1ba9fbc1b038`.
Those are 13 FDDS/SAPT + 10 CPSCF native tests, not GPU evidence.

FDDS job **13991729** queued on free embers H200, frozen pin **099706616a**,
agent watcher **b8a0442d4**, 20 FDDS tests/zero skips required. Previous
agent job 13991223 cancelled per agent's handoff; original artifacts retained.
No speedup claim, no merge. Agent remains owner of its GPU monitoring.

Read-only reviews running; agent instructed to hold source/tests stable
(handoff updates allowed):
- Numerical integration: **d5e1deca3ef549e9f0121bf2f701a1cc7**
- CUDA safety: **db6495209f845237937ebdb05a606bc46**
Retrieve each verified result with bg_result on notification, then send
actionable issues to workmux:saptdft-cuest-disp. Task #27 remains in progress.
Response #23 waiting on its independent job **13991128**.

Numerical review **d5e1deca3ef549e9f0121bf2f701a1cc7** retrieved.
Coordinator confirmed two P2 edge cases: empty one/five output matrices
bypass the host budget, and exported tensor helpers dereference null dfh_
on supported all-empty objects. Device-capacity/runtime-error test coverage
is also incomplete (risk, not demonstrated bug). Findings saved/sent via
`/home/awallace43/docs/saptdft/cuest/fdds-review-numerics.md`.
Agent still instructed to hold source until outstanding CUDA safety review
**db6495209f845237937ebdb05a606bc46** is retrieved; then authorize fixes
on its branch and native regression rerun. No merge or GPU pass claim.

CUDA review **db6495209f845237937ebdb05a606bc46** retrieved: no proven
new GEMM mapping/formula/lifetime bug. Conditional shared-handle pointer-mode
ownership risk and untested capacity/fault/unequal-nonempty-dimension paths
remain. Both reviews complete; source freeze released.
Actionable fixes/tests sent to the FDDS agent in
`/home/awallace43/docs/saptdft/cuest/fdds-review-followup.md`; agent to fix
the two confirmed P2 issues, add unequal-nonempty coverage and rerun native
FDDS+CPSCF tests/commit on its branch. Review-fix watcher **bd14639ab**
active. Keep job13991729/pin099706616a immutable; its results validate
pre-review code, not future fixes. No job cancellation or auto-merge authorized.

Review-fix watcher **bd14639ab** ended on an agent yield, NOT finished work:
host-budget fix **c90fc65bcc** committed, agent reports 12 native
regression/recovery passes, but null-helper guard and other coverage are
still in progress. Agent launched an isolated helper-crash reproduction.
Do not mark #27 complete from workmux's done status alone.

Coordinator requested atomic `../saptdft-cuest-disp/fdds-review-fixes-complete.json`
only after fixes, expanded full native suite and final commits/handoff.
Marker records current head, actual pass/error/skip counts, relative JUnit
path and honest GPU validation status. Event-based (inotify, no busy polling)
watcher **bf4d481f6** waits for it using
`/home/awallace43/docs/saptdft/cuest/watch_fdds_review_marker.py`.
On notification independently verify marker/source/XML and source fixes.
The marker establishes CPU review-fix completion only, never GPU correctness.

Marker watcher **bf4d481f6** completed. Agent HEAD
**7d51234f290555851e43bcc0d9edb589ea541b1e** (clean); review-fixed scientific
source **8810da5743cc7350e7620ecb708cdd3e47baa739**. Coordinator independently
verified marker/current HEAD and XML: **51 executed CPU passes, zero
errors/failures/skips**, 18.484 s (41 FDDS/SAPT + 10 CPSCF).
Fixes: c90fc65bcc empty budgets; efde54e03b exported helper guards and
persisted DFHelper shape metadata; 8810da5743 unequal-nonempty coverage
and explicit shared cuBLAS HOST mode check.

Final focused read-only review **de8a82e1774677ed98dc51ceddf57be58** checks
DFHelper shape-query compatibility and fix correctness. Agent instructed to
hold source/tests and delay fresh GPU submission until result. Existing
job13991729 is pre-review pin099706616a; it cannot validate these fixes.
After clean review coordinate one fresh independent free-embers v3 pin/run,
52 FDDS tests required, no existing job cancellation/mutation or auto-merge.
GPU capacity/fault-injection cases remain untested and explicit.

Final review **de8a82e1774677ed98dc51ceddf57be58** retrieved: **no new
correctness defect**; confirms fixes, DFHelper effective shape compatibility
and meaningful regression coverage. Source freeze released.
Agent authorized ONE new isolated free-embers FDDS v3 validation, scientific
pin **8810da5743cc7350e7620ecb708cdd3e47baa739**, 52 executed tests/zero skips.
No cancellation/mutation of old13991729 or response13991128; no auto-merge.

Submission-event watcher **b7c08063f** waits for agent's atomic
`fdds-v3-submission.json` in its worktree, recording verified actual job ID,
pin, QoS, expected52 tests, run directory and watcher. Agent owns terminal
GPU monitoring/audit. On marker notification capture actual ID and ensure
the coordinator receives meaningful terminal evidence rather than an idle
workmux yield. Script: `.../docs/saptdft/cuest/watch_fdds_submission.py`.

FDDS v3 submission watcher **b7c08063f** completed: actual job **13992504**
on **gts-cs207-chemx / gpu-h200 / embers**, H200:1 / 8 CPU / 64G / 2h,
initially PENDING. Agent watcher **bba95ab46** monitors SLURM.
Scientific source/tests pinned **8810da5743cc7350e7620ecb708cdd3e47baa739**;
separate harness **e60b32507b3069192032705b1d4161dd38886511** only changes
validation scripts/handoff, independently verified via diff (no scientific
source/test differences). Harness relaxes nounset only around conda activation
and restores it; GCC_ROOT isolation retained. Live slots37/50 at submission.
Run:
`/storage/project/r-cs207-0/awallace43/runs/psi4-fdds/20261012-validation-03`.

Coordinator terminal-evidence watcher **b8e927285** uses event-based
`watch_fdds_submission.py --terminal` for the agent's atomic
`fdds-validation-13992504-terminal.json`. Agent instructed to report
true outcome/counts/pins/local evidence (tests0 for preflight failure);
pass requires52 actual tests/zero errors/failures/skips, correct staged
import/core/libraries, actual H200/GPUroute and VERIFIED+COMPLETED.
Both existing FDDS13991729 and response13991128 remain untouched.
No GPU numerical/speedup claim or merge until attributable evidence.

### FDDS v3 failed — focused diagnosis required

Coordinator evidence watcher **b8e927285** completed with FAILED marker:
job13992504 ran/built/imported correctly on H200; partial pytest progress
**50 passes, one failure, one unfinished final SAPT GPU case**.
No final JUnit, VERIFIED.json or COMPLETED. Final case entered native
debugger-wait fault handler; TERM ended allocation after1h57m57s, job/batch
FAILED143:0, payload CANCELLED0:15. Counts are progress tokens, NOT certified
JUnit; reported errors0 does not rule out the unreported native crash.

Agent handoff records explicit user QoS switch in its own thread:
v3 only changed from original embers to inferno at unchanged2h limit.
Actual prolog confirms inferno. This approval does not authorize new paid
diagnostics. Runtime resolved conda CUDA13.3.29/cuBLAS13.6.0.2, not site
nvcc12.6.1 runtime. Core hash:
`3fe5a446db657ecb82c6557973eab85f08c161729ab16d047df7c16fb9d61c73`.
Failure evidence: `../saptdft-cuest-disp/build_fdds_cpu/gpu-13992504/`;
agent handoff commit **e2032b65a5**.

Coordinator suspects a TEST fixture error for unequal nonhybrid: available
budget262 doubles (computed from B3occ/7vir), but A2occ/8vir requires276
doubles, so CPU reference may throw before GPU comparisons. Under a valid
budget, A has one block and B two, hence9 blocks/threefrequencies, not
hardcoded12. This is a hypothesis pending fail-fast traceback, NOT an
attributed GPU numerical bug.

Diagnosis/fix instructions sent:
`/home/awallace43/docs/saptdft/cuest/fdds-gpu-diagnosis-request.md`.
Agent to run one short independent free-embers diagnostic, read-only use of
existing compiled pin if safe, separate assertion and native-GDB SAPT
processes with strict timeouts/debugger-wait breakpoint, then fix/regress/
commit proven causes. No blind full-suite rerun, cancellation or auto-merge.
Task #28 tracks blocker; #27 stays incomplete.

Event watcher **b8ed003d1** waits for agent's atomic
`fdds-gpu-diagnosis-complete.json` (outcome fixed or blocked, current HEAD,
jobs, confirmed causes/tests/local evidence). Script:
`watch_fdds_review_marker.py --diagnosis`.
A diagnosis marker is not a full GPU validation pass.

Local `build_saptdft_cuest/CMakeCache.txt` is **ENABLE_cuEST=OFF**.
Local `nvidia-smi` fails: driver/library version mismatch (NVML 580.178).
CPU compilation/tests are possible; GPU correctness needs an isolated
Phoenix build/stage, never either running benchmark's binaries.
New cluster validation defaults to free embers. No inferno approval applies.
Do not commit unrelated user changes on the development branch merely to
stage a build; use an isolated, auditable validation branch if required.

Known Phoenix GPU-build dependencies from the lifecycle cache:
- compiler: `p4cuest_sapt/bin/x86_64-conda-linux-gnu-{cc,c++}`;
- BLAS/LAPACK: `p4cuest_sapt/lib/libmkl_rt.so`;
- Libxc CMake directory: `p4cuest_sapt/lib/cmake/Libxc`;
- cuEST headers/library: environment `include/`, `lib/libcuest.so`;
- Release, OpenMP ON, cuEST ON, dkh OFF.

Suggested regression surface:
1. Nonsymmetric/batched response J/K, zero/differing occupied spin counts.
2. HF, LDA/GGA, global hybrid, and pure/partial long-range hybrid Hessian
   actions: combined and separate operators, multiple RHS and transposes.
3. CPHF/CPKS solution residuals and polarizabilities against native CPU
   references and cuEST-JK-with-CPU-GEMM controls.
4. Forced GEMM threshold zero, high-threshold CPU fallback, symmetry behavior.
5. Clear CUEST_XC=true and MGGA/VV10 rejection for out-of-scope kernels.

No numerical speedup/support claim until the new response code is built and
its executed tests pass. Keep partial implementation and validation blockers
visible rather than marking #22/#23 complete.
