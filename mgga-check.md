# Meta-GGA CPU/cuEST interaction-energy checks — handoff

Status: 2026-10-08 continuation. Original matrix/recovery jobs are terminal;
fixed-density diagnostic job **13886607** completed but is **scientifically
invalid** (orbital replay metadata; see below), and strict
PW6B95 CPU-control job **13886663** completed with attributable convergence
failures and one converged monomer (not a complete IE).
Corrected fixed-density job **13886797** completed and passed replay invariants.
Actual-grid decomposition job **13888500** failed before SCF (runtime linkage).
Corrected decomposition job **13893248** completed with validated capture.
Ghost-monomer closure job **13895464** completed successfully on `embers`.
The original scientific jobs and **ωB97M-V water preflight 13904944 are terminal**.
The preflight used explicitly approved inferno and passed all 18 workers,
four comparisons and final provenance; full-XC IE errors are about 1.17e-7 Eh,
with 5.30× / 6.10× complete-IE speedups (host/CUDA LibXC).
First explicit-VV10-setting jobs **13911317 / 13911319 / 13911321**
(water / benzene / peptide) failed **before SCF** because my harness used
`is False` on Psi4's valid integer-zero Boolean option. Fixed and regression
tested (17 tests plus 54 actual option-record replays); no new scientific
results. User explicitly approved three corrected **inferno** retries:
**13940495 / 13940497 / 13940499** (water / benzene / peptide).
All corrected retries are **terminal**: water and peptide completed,
benzene failed on one 300-second host GPU-JK-only AB timeout.
**All six ωB97M-V full-GPU comparisons pass 1e-5 Eh**; 11/12 comparisons
are complete and pass, with that one diagnostic control explicitly incomplete.
Final speedups are water **5.47×/5.65×**, benzene **31.71×/41.50×**,
peptide **28.98×/40.24×** (host/historical CUDA LibXC).
The audited final speedup/support and dimer-energy tables are posted to
[cuEST PR #12](https://github.com/Awallace3/psi4/pull/12).
See **Final corrected ωB97M-V audit and PR publication** at the end for
source pins, archive paths, provenance and the optional recovery decision.
User selected **Keep documented gap (Recommended)** after publication:
the requested full-GPU benchmark scope is complete, and the extra GPU-JK
control remains explicitly incomplete. No recovery is planned or authorized.
The MGGA campaign is terminal; another inferno recovery needs fresh approval.

**New lower-rung request:** prepare SVWN, BLYP, B3LYP, PBE0 and ωB97X
controls on the same three systems/builds/routes, using free embers only.
Harness extended in separate worktree at **e44ae0eb8f**, with 20 passing
tests. Water preflight **13976174** is submitted on **embers**, initially
**PENDING / Priority**; benzene/peptide wait for its audit. Watcher **b04540077**.
See **Lower-rung numerical extension** at the end; no results claimed yet.
The host benzene AB/A/B fixed-density XC
decomposition demonstrates the combined backend-quadrature contribution to
the interaction-energy discrepancy; its difference from the archived optimized
SCF IE error is about **2e-8 Eh** for both M06 and M06-L.
The audited JSON/Markdown/HTML report is complete and render-checked; paths
and final closure are recorded at the end of this handoff.
PW6B95's original strict CPU protocol has a reproducible convergence limitation;
no complete IE exists. No scientific C++ fix or universal support claim was made.
User follow-up adopts **1e-5 Eh for IE agreement**: all **47 complete comparisons**
pass, including **24/24 full-GPU rows**. Of 60 planned comparisons, 13 remain
incomplete: 12 PW6B95 and one ωB97M-V GPU-JK control. The earlier archived report
retains its original 1e-6 Eh diagnostic classification; the new ωB97M-V
addition uses the adopted criterion and does not overwrite that report.

## Goal and immediate next steps

The user asked to quantify ordinary **SCF interaction-energy** agreement between
CPU and GPU on our smaller systems, after Lori reportedly said meta-GGAs are
unsupported in Psi4+cuEST. This is explicitly **not a SAPT(DFT) investigation**.

1. Reproduce the evidence summary from the archived `comparisons.json` files
   listed below. Keep unsuccessful and missing cases visible.
2. Diagnose benzene M06-L/M06 differences with the shared cuEST XC route.
   Start with matched grid/pruning and density-screening/threshold sensitivity,
   then inspect density/τ conventions, functional derivatives and AO assembly.
   These are hypotheses, not established causes. A fixed-density potential
   comparison would help separate XC implementation differences from SCF feedback.
   Completion criterion: identify and demonstrate the mechanism with a controlled
   CPU/GPU comparison, rather than merely finding a setting that lowers the error.
3. Diagnose PW6B95 ghost-monomer convergence on the CPU reference first.
   Inspect the SCF residual/energy histories. If changing damping, guess, optimizer,
   grid or convergence criteria, label a new protocol and apply it consistently
   across routes. Completion criterion: obtain converged matched references or
   document a reproducible convergence limitation; do not certify absent IEs.
4. Write a consolidated results summary, distinguishing IE cancellation, total
   energy errors, functional/build coverage, and unresolved cases.

Use free `embers` for small follow-ups. Later approvals covered the water
preflight and exactly three corrected ωB97M-V retries; they do not authorize
an additional paid recovery job.

## What is established

- M06-L, M06 and r²SCAN completed across water, benzene and peptide, both builds,
  and all three routes: **36 complete comparisons** to the corresponding CPU
  references (9 system/functional pairs × 2 builds × 2 GPU routes).
- GPU J/K-only agreement is near numerical noise:
  maximum absolute IE difference **7.3544e-10 Eh**;
  maximum underlying total-energy difference **1.1544e-9 Eh**.
- Full cuEST XC exceeds our strict **1e-6 Eh IE threshold** for benzene M06-L
  and M06 in both builds: four failed comparison rows, not four distinct systems.
- Maximum absolute IE differences for full cuEST XC:

  | Functional | Maximum IE error (kcal/mol) | Case |
  |---|---:|---|
  | M06-L | 0.00228655 | benzene |
  | M06 | 0.00380991 | benzene |
  | r²SCAN | 0.000523458 | benzene |

- Largest full-cuEST-XC IE error: **6.07148e-6 Eh**.
  Largest underlying total-energy error: **6.14938e-5 Eh**.
  Good IE agreement can therefore conceal cancellation of larger total errors.
- Host-LibXC and CUDA-LibXC versions of the cuEST-XC route have almost identical
  errors. This points toward the shared cuEST XC path rather than specifically
  CUDA functional evaluation, but the builds are not source/version matched:
  this is suggestive evidence, not a proven attribution.
- All tested r²SCAN IEs pass. Peptide/r²SCAN full-XC errors are
  −1.71690e-8 Eh (host LibXC) and −1.74081e-8 Eh (CUDA LibXC).
- PW6B95 has **no complete IE comparison**. CPU ghost-monomer calculations also
  fail convergence; some benzene workers hit the 240-second timeout. This does
  not establish a GPU-specific incompatibility.
- These results establish working restricted SCF cases, not universal support
  for all meta-GGAs, unrestricted references, gradients, Hessians, response, or
  Laplacian-dependent functionals.

## Protocol

Counterpoise-corrected interaction energy at fixed geometry:

```text
IE = E_AB − E_A(ghost B) − E_B(ghost A)
delta_IE = IE_GPU − IE_CPU
1 Eh = 627.5094740631 kcal/mol
```

Each dimer and ghost monomer is calculated in a fresh process. Ghost extraction
uses `extract_subsets(1, 2)` and `(2, 1)`. Dimer/monomer basis counts are checked
equal. No SAPT, GRAC, D4 or other empirical dispersion is included.

- Water and benzene: aug-cc-pVDZ.
- Peptide: spherical 6-31+G**.
- Auxiliary basis: def2-universal-jkfit, DF SCF, C1.
- Functionals: `m06-l`, `m06`, `pw6b95`, `r2scan`.
- Restricted closed-shell SCF; 8 CPU threads; Psi4 memory 32 GiB.
- FP64 (`CUEST_MIXED_PRECISION=False`), CPU SAD in every route.
- Grid: 99 radial × 590 spherical points; `DFT_PRUNING_SCHEME=ROBUST`.
- `E_CONVERGENCE=11`, `D_CONVERGENCE=10`, `MAXITER=200`,
  `FAIL_ON_MAXITER=True`. Worker timeout: 240 seconds.
- Exactly one calculation per configuration: these are numerical checks, not
  statistical timing measurements.

Three routes are compared **within each build**:

| Route | USE_CUEST | CUEST_XC | Meaning |
|---|---|---|---|
| `cpu` | false | false | Native CPU J/K and XC |
| `gpu-jk` | true | false | cuEST J/K, native CPU XC |
| `gpu-xc` | true | true | cuEST J/K and density/potential assembly; functional placement depends on build |

Routing is asserted using the cuESTJK output header and XC timer names, not
options alone. `CUDA LibXC Functional` must appear for CUDA full-XC, and
`Host Functional` for the host-LibXC cuEST-XC route.

## Evidence and job outcomes

Local archive root:
`/home/awallace43/docs/saptdft/cuest/mgga-ie/`

| Directory beneath root | Job | Outcome |
|---|---|---|
| `20261007-preflight/results-13884485/` | 13884485 | Water/M06; 18/18 SCFs, all IE comparisons pass |
| `20261008-matrix/results-13884503/` | 13884503 | Partial matrix; 184 workers recorded: 160 exit 0, 16 exit 1, 8 timeout 124 |
| `20261008-r2scan-recovery/results-13885267/` | 13885267 | Peptide/r²SCAN; 18/18 SCFs, all IE comparisons pass |

Each results directory has `manifest.json`, `comparisons.json`, per-worker
`result.json` and `psi4.out`, plus captured process logs. The complete jobs have
`COMPLETE.json`; the partial matrix does not. Sibling `metadata/` and `logs/`
directories preserve environment/linkage and scheduler output.

The matrix terminated after **1:57:49**, with batch exit 143 and a terminated
srun step. The script requests a TERM warning before its 2-hour wall limit and
responds by terminating the payload. Do not describe this as successful completion
or assume preemption without additional scheduler evidence.

The main matrix yields 32 complete comparison rows; recovery supplies the four
missing peptide/r²SCAN rows. Combine those complete rows, **not** the preliminary
water/M06 preflight repeats. Retain PW6B95 incomplete rows separately.

Remote archive root:
`/storage/project/r-cs207-0/awallace43/runs/psi4-mgga-ie/`
with the same three campaign directory names.

## Harness and reproducibility

This working tree is the user's dirty development tree. Preserve its changes.
Experiment code lives in a separate worktree:

```text
Local:  /home/awallace43/gits/psi4__worktrees/cuest-mgga-ie-20261007
Branch: cuest-mgga-ie-20261007
Remote: /storage/project/r-cs207-0/awallace43/gits/psi4.cuest-mgga-ie-20261007
HEAD:   b1183da010ec2186ea402285902ab455a6cacf87
```

Files under `devtools/benchmarks/`:

- `mgga_ie.py`: worker, route checks, campaign, comparison generation.
- `mgga_ie_job.sh`: TERM-aware SLURM wrapper.
- `test_mgga_ie.py`: four passing local unit tests; not a substitute for GPU tests.
- `saptdft_cuest_grac.py`: reused geometry generator and atomic JSON writer only.
  Benzene is the prior idealized dimer, **not published S22 geometry**.
- `saptdft_suite_geometries.json`: peptide coordinates.

Harness revisions:

- Preflight: `b63bdcf91a25ffc3f07228e4d3d6ff516a2ab9f3`.
- Main matrix: `81b8f0f9126d18f5307e06ce3685f207300068d1`.
- Recovery: `b1183da010ec2186ea402285902ab455a6cacf87`.

The main/recovery revisions additionally hash staged Python drivers, basis files,
metadata, CMake cache and resolved LibXC/cuEST libraries, verifying them unchanged
at normal campaign completion. The interrupted matrix did not reach that final
verification. Its initial hashes are preserved in the manifest.
The preflight has only core-library hashes and captured `ldd` paths.

Process failures invalidate comparison inputs even if a worker wrote `ok:true`
before crashing at teardown (fixed after preflight). Campaign exit success means
calculations completed; **check `all_ie_within_tolerance` separately**.

Existing build packages (both use `p4cuest_sapt` Python environment):

```text
Host LibXC:
/storage/project/r-cs207-0/awallace43/gits/psi4.saptdft-cuest-lifecycle-20261003/build_lifecycle/stage/lib/psi4

CUDA LibXC (older historical build):
/storage/project/r-cs207-0/awallace43/gits/psi4.saptdft_cuest/build_cuda_libxc/stage/lib/psi4

CUDA library directory prepended for CUDA-build processes:
/storage/project/r-cs207-0/awallace43/software/libxc-7.1.2-cuda/lib64
```

Use manifest hashes/version strings as build evidence. A source checkout's
current commit alone is not proof of the compiled binary's source revision.
The host build comes from the validated lifecycle snapshot; the CUDA build is
older. No scientific C++ fixes were made during this investigation.

The wrapper requests H200, 8 CPUs, 64G, 2 hours, `gts-cs207-chemx`, **embers**.
Preflight/recovery override walltime to 30 minutes. It reuses the pinned
Protein157 scratch helper from remote worktree
`psi4.saptdft-cuest-protein157-20261004` at
`7d897a7f203297754087ea77536619c378191c8b`.
It restores TMPDIR/SCRATCH/PSI_SCRATCH after `srun` resets the environment.

Run local checks from the experiment's `devtools/benchmarks/`:

```bash
python3 -m unittest test_mgga_ie
bash -n mgga_ie_job.sh
```

For future subsets, `mgga_ie.py` supports `--systems peptide` and
`--functionals r2scan` (or other listed choices); pass those after the wrapper's
three arguments: absolute harness path, pinned commit, unique run directory.
Make explicit new options/configurations for any diagnostic protocol changes.

## Lori branch versus our CUDA bridge

Fetched and inspected:

- `lori/xptl_v1110_wwK_rb1` at `39583456b27944493efc491a49d01ec4c403d1bc`.
- `upstream/cuest_xptl_v1110` at `02972872c0b0c7cb65f5ed716a7dd06e7b8ab78f`.

Both have meta-GGA cuEST density/potential paths using **host LibXC**.
Lori's branch includes restricted/unrestricted M06 and PW6B95 test definitions
in `tests/pytests/test_cuest.py`. Test presence alone is not a passing test result
or official support declaration. We have not obtained Lori's exact statement,
failing input, or intended build, and have not tested her branch directly.

Our branch adds device-functional evaluation absent from the inspected Lori tip:

- `52431fc2c0`: restricted CUDA-LibXC bridge.
- `60322c4275`: unrestricted CUDA-LibXC bridge.
- `4a6164ef71`: persistent workspace and device reductions.

Relevant code: `psi4/src/psi4/libfock/v.cc` and
`psi4/src/psi4/libfunctional/LibXCfunctional.cc`.
LibXC itself supplies CUDA meta-GGA kernels; the bridge calls
`xc_mgga_exc_vxc` with device functional handles/arrays.
Primary reference:
https://gitlab.com/libxc/libxc/-/raw/7.1.0/src/work_mgga_inc.c

## Remote operating rules

Read the Phoenix skill before new submissions:
`/home/awallace43/.pi/agent/skills/phoenix-slurm-experiments/SKILL.md`,
plus `references/phoenix.md` and `references/psi4.md`.

Use its local `scripts/phx` wrapper for remote commands and
`scripts/watch_jobs.sh` for background monitoring. Sync code through git, use new
run directories, and preserve existing builds/evidence. Keep heavy work on compute
nodes. Explicitly request `--qos=embers`; inferno requires new user approval.

The earlier SAPT/Protein157 report is separate:
`~/docs/saptdft/cuest/breakdown-2026-10-07.html`.
It has not been repurposed as a meta-GGA report.

## Continuation: archive replay and fixed-density probes

Archived comparisons were independently regenerated from `manifest.json` worker
exit codes and per-worker results using the new `mgga_archive.py`. The regenerated
rows exactly match the archived comparison files. Recovery supersedes four
incomplete peptide/r²SCAN rows, leaving **36 complete, four IE failures, and
12 incomplete PW6B95 comparisons**. Original process failures remain in the
derived report rather than being erased by recovery.

Derived evidence:
`/home/awallace43/docs/saptdft/cuest/mgga-ie/consolidated-replay-20261008.json`.
The `--assert-parity` replay exits 1 deliberately because failed/incomplete cases
remain. That is a scientific regression signal, not a harness unit-test failure.

New experiment commit:
`3629b8d0ce9c9cf313ff213ce4e0b0943433fac0`, pushed to the existing experiment
branch and fast-forwarded on Phoenix. Seven local unit tests pass; Python compile
and wrapper shell syntax checks pass. No scientific C++ changes were made.
Do not advance the remote harness checkout while this queued job verifies its
commit pin.

**Job 13886607**, `mgga-fixed-density`, requests H200, eight CPUs, 64G,
30 minutes, `gts-cs207-chemx`, explicit `embers`.
Initial effective scheduler state: PENDING / Priority; QOS=embers,
Partition=gpu-h200, StartTime=Unknown. Verify prolog QoS after it starts.

Run directory:
`/storage/project/r-cs207-0/awallace43/runs/psi4-mgga-ie/20261008-fixed-density-01/`.
Wrapper selects `MGGA_DRIVER=mgga_probe.py`. It runs only the validated
**host-LibXC** build, benzene **AB**, M06 and M06-L, in separate fresh processes.
Each worker obtains one native CPU baseline SCF, then evaluates both routes at
exactly the same density and occupied orbitals. `RHF.form_V()` supplies both
`set_D` and the non-Python-exposed `set_Cocc`; no SCF optimization occurs during
the potential comparisons.

Seven one-option profiles: baseline; density cutoff 1e-18; basis cutoff 1e-16;
disabled weight cutoff; no angular pruning; Becke nuclear weights; Stratmann
nuclear weights. All profiles reuse baseline seed orbitals. Nominal CPU/cuEST
settings do **not** prove identical grid coordinates, orientations, nuclear
weights, or screening. Nuclear-weight options may be ignored by the cuEST grid
path; such a result must not be called a matched-grid comparison.

Per-worker artifacts include seed C/D arrays and hashes, CPU/GPU potential
matrices, XC quadrature values, route timer assertions, XC-energy differences,
potential max/Frobenius norms and density contractions.
Baseline CPU fixed-density XC energy must reproduce the seed XC energy within
1e-8 Eh. This new API path has **not yet been validated on the compute node**;
failure is possible and must remain visible. Probe success is not IE certification.

Background monitor uses the skill's `watch_jobs.sh` at 180-second intervals.
On completion, inspect `.batch` accounting and process exit codes, download
the compact diagnostic outputs, verify `COMPLETE.json`, then interpret sensitivity.
Mechanism remains unresolved until controlled evidence supports it.

## Continuation: PW6B95 CPU-reference diagnosis

Read-only review of the main matrix establishes:

- Water CPU ghost A/B fail at 200 iterations in both builds, with endpoint
  RMS commutator residuals ranging from 3.73e-8 to 2.08e-6 against the requested
  1e-10. Host B passes the final energy-step criterion but not the residual.
- Peptide host CPU ghosts fail at 200 iterations: A residual 1.31e-6,
  B residual 7.78e-8. **Peptide PW6B95 CUDA-build workers were never reached**;
  their missing coverage is not a recorded SCF failure.
- Benzene CPU ghosts time out at 240 seconds, after 68–69 iterations, in both
  builds. Endpoint residuals are 4.38e-6–7.80e-6. Their 200-iteration outcome
  is unknown. The roughly 3.3 s/iteration runtime makes the old timeout
  inadequate to diagnose a full 200-iteration run.
- All five available CPU dimers converge (water/benzene in both builds,
  peptide host). Ghost-basis convergence, not universal PW6B95 SCF failure,
  prevents counterpoise IE comparisons.
- GPU-JK water ghosts also fail; GPU-JK benzene ghosts time out. Full GPU-XC
  benzene ghosts reach explicit 200-iteration failures; water A converges but
  B fails. Thus failure classes are shared but individual convergence outcomes
  are route-dependent.
- `D_CONVERGENCE` uses the reported RMS `|[F,P]|` residual here, not a literal
  successive-density difference. Endpoint energies from failed iterations
  must never be treated as converged reference energies.

Representative archive evidence:
`water-pw6b95-host-cpu-{A,B}/psi4.out:414–419`,
`benzene-pw6b95-host-cpu-{A,B}/psi4.out:290–301`,
`peptide-pw6b95-host-cpu-{A,B}/psi4.out:436–441`.
Worker return codes and missing directories are recorded in the main manifest.

New isolated worktree/branch, preserving the queued XC job's checkout pin:

```text
Branch: cuest-mgga-pw6b95-20261008
Local:  /home/awallace43/gits/psi4__worktrees/cuest-mgga-pw6b95-20261008
Remote: /storage/project/r-cs207-0/awallace43/gits/psi4.cuest-mgga-pw6b95-20261008
Commit: e2e030263c43bcab53f1b05673beb16475929dc8
```

New files: `mgga_pw6b95.py`, `mgga_pw6b95_job.sh`,
`test_mgga_pw6b95.py`. Ten local unit tests pass including inherited harness
checks; compile and Bash syntax checks pass.

**Job 13886663** requests eight CPUs, 64G, 90 minutes, `cpu-small`,
`gts-cs207-chemx`, explicit `embers`, no GPU. Initial effective state:
RUNNING, StartTime=2026-10-08T10:58:38 (cluster local time), QOS=embers.
Compute-node prolog confirms `QOS: embers`; captured environment confirms
`QOS=embers`.

Run:
`/storage/project/r-cs207-0/awallace43/runs/psi4-mgga-ie/20261008-pw6b95-cpu-01/`.

Six fresh-process host-build **CPU-only** ghost calculations:

1. Water A/B, original strict baseline, worker timeout 600 seconds.
2. Water A/B, labeled `damped-20` solver protocol:
   `DAMPING_PERCENTAGE=20`, `DAMPING_CONVERGENCE=1e-12`, timeout 600 seconds.
3. Benzene A/B, original SCF settings, only worker timeout extended to
   1200 seconds (`timeout-only`).

All retain original geometry/basis/grid, `MAXITER=200`, strict E/D thresholds,
and `FAIL_ON_MAXITER=True`. No IE is computed from these monomer-only controls.
Current source rejects MGGA rotated response builds
(`v.cc`, `RV::compute_Vx_full`), so a speculative SOSCF recovery was not submitted.
This source observation alone is not proof of the exact compiled binary's code.

The campaign writes `DIAGNOSIS.json`, not an IE completion marker.
An attributable SCF failure or timeout can mean the diagnostic finished,
but `all_converged` remains separately recorded. Setup errors and unexpected
process crashes are not accepted as scientific diagnostic endpoints.
Check process exit codes, endpoint histories and final provenance verification.
If a solver protocol succeeds, run matching AB/A/B across all routes before
making any new IE comparison.

## Continuation: shared-XC source/API inspection (not yet causal evidence)

Read-only inspection of the **current dirty source**, not a source-matched
reconstruction of the tested binaries, identifies incomplete correspondence of
nominal CPU/cuEST settings:

- Radial nodes/weights and angular counts per shell do reach cuEST
  (`libfock/cubature.cc:3817–3860`).
- `DFT_NUCLEAR_SCHEME` is parsed but not forwarded to the cuEST molecular-grid
  API (`cubature.cc:3863–3890,4435`). Native CPU uses it
  (`cubature.cc:3905,4006`).
- Native `OrientationMgr` and its oriented point placement are bypassed by
  the cuEST branch's early return (`cubature.cc:3895–3905,4005`).
- Native weight cutoff, basis-extents shell screening, distant-point removal
  and blocking do not reach cuEST grid/plan creation
  (`cubature.cc:4008,4291–4324,4439,4482–4499`).
- `DFT_DENSITY_TOLERANCE` does propagate to LibXC components
  (`driver/procrouting/dft/superfunctionals.py:79–82`,
  `libfunctional/superfunctional.cc:436–443`,
  `libfunctional/LibXCfunctional.cc:301–308`).
- cuEST host/CUDA component accumulators skip an entire point contribution if
  inspected component outputs are nonfinite (`v.cc:124–146`,
  `cuest_xc_gpu.cu:61–82`); native LibXC instead checks for NaNs and throws
  (`LibXCfunctional.cc:692–735`). Whether this guard fires in the discrepant
  cases is unknown.

The visible restricted conversion uses total `rho=2*rho0`, gradient invariant
`gamma=4*|grad rho0|^2`, and `tau=2*tau0`. Native CPU's tau/potential half factors
cannot be compared by eye without the external assembly contract. There is no
demonstrated factor-of-two bug.

Installed Phoenix cuEST headers provide additional, version-adjacent evidence:

```text
Prefix: /storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt/include/cuest/
types/cuest_parameter_types.h:698–710
  Atom/molecular-grid parameter attributes expose only NONE (reserved).
  SHA256 d9f34c35b4765fc841983498c8788ecb28a2f6937f0fbe6d07183d518ff69ae1
exchange_correlation/xc_integration_grid_compute_api.h:141–144
  TOTAL weights are documented as angular × radial × Becke.
  SHA256 fc40d6547ee97a7409ad7bc5e63ce75f2a54448134f5593c388228c6668a9789
exchange_correlation/xc_potential_compute_api.h:371–378
  Packed derivative channels and externally applied integration weights.
  SHA256 b24a97b2afaecb22d923eac2f29a34748b98c2214a58deb705f10ea84a4bb18b
```

Official API documentation independently describes the total weights as
angular × radial × Becke:
https://docs.nvidia.com/cuda/cuest/c-api/methods/exchange-correlation/index.html
(retrieved 2026-10-08). This documents the API, not exact correspondence to
Psi4's Becke implementation, angular frame, or tested binary behavior.

Interpret job 13886607 with backend-separated changes:

1. Nuclear scheme, basis cutoff and weight cutoff should move the CPU result
   but leave cuEST invariant if the inspected plumbing matches the binary.
   Compute **GPU shifts** from stored per-probe quadrature values as well as
   the already reported CPU shifts and inter-backend differences.
2. Density cutoff changes LibXC in both routes; it does not establish matched
   cuEST collocation screening.
3. No pruning changes shell counts in both routes, but not orientation or
   partition equivalence.
4. A closer Becke CPU match is suggestive only. A mechanism demonstration
   requires actual grid-coordinate/weight comparison or another controlled
   experiment isolating the differing operation.
5. Identical-orbital energy discrepancies isolate XC quadrature from SCF
   feedback, not grid construction from functional/collocation differences.
   If necessary, next controls are identical-point density/gradient/tau
   comparison, nonfinite-output counters, and orbital-rotation finite
   differences versus `2 Tr(Vxc δD)`.

### Fixed-density job 13886607: execution passed, scientific validation failed

Job and `.batch` completed exit 0 in 4:48; two workers exit 0;
`COMPLETE.json` reports successful probe execution; prolog confirms `embers`.
Downloaded archive:
`/home/awallace43/docs/saptdft/cuest/mgga-ie/20261008-fixed-density-01/`.
**Do not use its sensitivity or energy/potential differences for attribution.**
`INVALID.json` records the failure without modifying the original result files.

The CPU integrates approximately 84 electrons, but replayed cuEST density
integrals are 3322.315639 (M06) and 3292.695865 (M06-L).
This invalidates the premise that the two paths received identical densities,
despite equality of the copied `Ca`/`Da` arrays.

Source inspection identifies a harness initialization error:
`Wavefunction::common_init` leaves `nmopi=0`
(`libmints/wavefunction.cc:498–510`).
The fresh bare `Wavefunction.build` passed to `core.RHF` did not run
`HF.form_Shalf`, which normally initializes MO dimension metadata
(`libscf_solver/hf.cc:787`).
`C_subset_helper` uses `nmopi` as the source stride when extracting occupied
orbitals (`wavefunction.cc:881`). Thus `RHF.form_V` uploads incorrectly extracted
AO occupied coefficients even though native CPU integration uses the correct
copied density matrix. This is a **probe-harness error**, not demonstrated
scientific cuEST disagreement.

A corrected replay must initialize/inherit MO metadata, validate AO occupied
coefficients and their density/overlap normalization, and explicitly reject
an integrated electron count inconsistent with 84 before interpreting XC data.
The source mechanism still needs a controlled corrected compute-node replay.

### Corrected replay queued: job 13886797

Experiment branch `cuest-mgga-ie-20261007` advanced only after job 13886607
was terminal, to `5b36502690d0d125be7e7666c4565f380e269f14`.
PW6B95's separate checkout/commit remains untouched.

The corrected probe constructs `core.RHF(seed, functional)` from the fully
initialized CPU seed instead of a bare `Wavefunction.build`. It validates
`nmo/nmopi` against the copied matrix columns, copies orbital energies, compares
the actual `Ca_subset("AO", "OCC")` to the seed, checks overlap orthonormality,
checks AO occupied-density reconstruction, and rejects an integrated electron
count differing from 84 by more than 1e-3. This is a coarse physical invariant,
not a claim about 1e-6 Eh IE accuracy.

Ten local unit tests pass, including regression guards rejecting zero-MO
metadata and the observed bad electron count. Compile/shell checks pass.
The compute-node replay now passes these invariants (see next subsection).

Job **13886797**: H200, eight CPUs, 64G, 30 minutes,
`gts-cs207-chemx / gpu-h200 / embers`.
Initial scheduler state PENDING / Priority; StartTime=Unknown; effective
QOS=embers verified. Verify compute-node prolog after launch.
Run:
`/storage/project/r-cs207-0/awallace43/runs/psi4-mgga-ie/20261008-fixed-density-02/`.
Background watcher uses `watch_jobs.sh -i 180`.

### Corrected fixed-density job 13886797: valid replay, XC discrepancy remains

Job, `.batch`, and `srun` completed exit 0 in 4:37. Both functional workers
exit 0; `COMPLETE.json` confirms successful execution with final build hashes
verified by campaign code. Prolog confirms `embers`. Local archive:
`/home/awallace43/docs/saptdft/cuest/mgga-ie/20261008-fixed-density-02/results-13886797/`.

All 28 backend/profile evaluations pass AO occupied-orbital, overlap,
density reconstruction, finite-output and coarse electron-count checks.
Maximum electron integral deviation is 6.76e-5 from 84.

| Benzene AB, identical density | M06 | M06-L |
|---|---:|---:|
| Baseline XC energy GPU−CPU / Eh | +6.15802968e-5 | −1.42495228e-5 |
| Prior converged total energy GPU−CPU / Eh | +6.14938076e-5 | −1.42833705e-5 |
| Baseline potential max absolute difference / Eh | 9.2164e-5 | 5.2974e-5 |
| Baseline integrated electrons, CPU | 83.99999158 | 83.99999328 |
| Baseline integrated electrons, cuEST | 83.99993242 | 83.99993613 |

The fixed-density XC differences reproduce the prior AB total-energy
differences closely (within 0.24% relative). Thus SCF feedback is not required
to produce the discrepancy. This does **not** yet separate quadrature/grid
differences from collocation, ingredient conventions or functional adapters.
These AB-only XC probes produce no new interaction energies.

Sensitivity results, with backend changes kept separate:

- Density cutoff 1e-18 and basis cutoff 1e-16 change neither XC difference
  meaningfully (roundoff ~1e-14 Eh).
- Disabling native weight cutoff moves CPU by approximately −5.24e-11 Eh;
  cuEST stays invariant. Not an explanation of 1e-5–1e-4 Eh discrepancies.
- Removing pruning moves CPU M06 by −2.28e-6 and cuEST by +3.34e-7 Eh;
  for M06-L, CPU −1.16e-6 and cuEST +8.09e-8 Eh. Discrepancies remain.
- CPU Becke moves M06 by +3.06e-6 and M06-L by +5.89e-6 Eh; cuEST stays
  invariant. Remaining discrepancies: +5.85e-5 and −2.01e-5 Eh.
- CPU Stratmann moves M06 by −1.13e-6 and M06-L by −3.37e-6 Eh;
  cuEST stays invariant. Remaining discrepancies: +6.27e-5 and −1.09e-5 Eh.

The tested nuclear schemes alone do not explain the effect; a smaller
difference for one profile must not be called a fix. Actual grid/ingredient
comparison is the next isolating experiment.

### PW6B95 CPU controls 13886663: reproducible original-protocol limitation

Job, `.batch`, and `srun` completed exit 0 after 32:40.
`DIAGNOSIS.json` verifies all six endpoints attributable and final provenance
unchanged; `all_converged=false`. Scientific worker failures are deliberately
kept separate from successful completion of the diagnostic.
Prolog confirms `embers`. Local archive:
`/home/awallace43/docs/saptdft/cuest/mgga-ie/20261008-pw6b95-cpu-01/results-13886663/`.

| Host CPU ghost/control | Worker exit | Last iteration | RMS residual | Outcome |
|---|---:|---:|---:|---|
| Water A, baseline | 1 | 200 | 1.99793e-7 | SCFConvergenceError |
| Water B, baseline | 1 | 200 | 4.60515e-8 | SCFConvergenceError |
| Water A, damped-20 | 1 | 200 | 1.37536e-6 | SCFConvergenceError |
| Water B, damped-20 | 0 | 37 | 1.39538e-14 | Converged |
| Benzene A, timeout-only | 1 | 200 | 4.70774e-6 | SCFConvergenceError |
| Benzene B, timeout-only | 1 | 200 | 4.80933e-6 | SCFConvergenceError |

Only water B's damped run supplies a converged total energy:
−76.52311554673426 Eh. Damping switches off by iteration 35; the final two
iterations are ordinary DIIS. This is partial solver recovery, not a recovered
counterpoise IE. No other endpoint energy in this control is a reference.

The extended benzene worker limit removes the previous runtime ambiguity:
both host CPU ghosts reach the specified 200 iterations and fail rather than
timing out. Their minimum residuals in the last 20 iterations are 1.08e-6
and 2.07e-6. Water baseline failures also reproduce independently.

This establishes a **reproducible convergence limitation of the original
strict host-CPU protocol**. It does not prove PW6B95 can never converge under
another solver/grid protocol, nor identify the underlying residual-floor
mechanism. No matched AB/A/B recovery was obtained, so there remain zero valid
PW6B95 IE comparisons. The single successful monomer must not be combined
with unconverged or differently configured calculations.

### Actual-grid decomposition queued: job 13888500

Experiment branch advanced only after the previous jobs were terminal, to
`ae2131f6fed35876fc5e996804031b7690f0f616`.
New files: `mgga_capture.cc`, `mgga_grid_probe.py`,
`test_mgga_grid_probe.py`; wrapper supports `MGGA_DRIVER=mgga_grid_probe.py`.
Eleven local unit tests pass; Python compile and shell syntax checks pass.
The C-API capture shim is **not yet compiled or scientifically validated**.
No scientific Psi4/cuEST binary was modified.

Job **13888500**: H200, eight CPUs, 64G, one hour,
`gts-cs207-chemx / gpu-h200 / embers`.
Initial state PENDING / Priority, StartTime=Unknown; effective QOS=embers.
Run:
`/storage/project/r-cs207-0/awallace43/runs/psi4-mgga-ie/20261008-grid-decomposition-01/`.
Watcher uses `watch_jobs.sh -i 180`. Verify prolog QoS after launch.

The job builds a small external `LD_PRELOAD` shim against the installed cuEST
C API and CUDA runtime on the compute node. It forwards original calls, then
captures actual grid coordinates, total weights and raw one-spin rho/gradient/tau
arrays. It uses no Psi4 C++ private-layout/ABI access. Headers and compiled shim
are part of the diagnostic evidence. Compiler is GCC 12.3.0 with CUDA 12.6.1.

Each M06/M06-L worker uses one CPU seed and evaluates native CPU, cuEST with
capture inactive, then cuEST with capture active. Capture energy/potential must
match the inactive control within 1e-10; AO orbital, density normalization,
finite-array and captured-integral checks also gate interpretation.

Native `RKSFunctions` and native `SuperFunctional` then evaluate:

1. Native CPU points/weights with basis tolerance 1e-20, controlling the native
   screening difference from the original 1e-12 reference.
2. Actual cuEST coordinates/weights with **native AO density ingredients**.
3. The same cuEST weights with **captured cuEST density ingredients**, evaluated
   by native LibXC (`rho=2*rho0`, `gamma=4*|grad rho0|²`, `tau=2*tau0`).
4. Original cuEST-XC energy.

The telescoping differences separate native screening, grid, collocation/
ingredient and functional-adapter effects. Native density integrated on cuEST
points must also pass the 84-electron invariant. Coordinate counts, nearest-point
matches, matched-point weight differences and ingredient error norms are saved.
This is intended to **demonstrate a mechanism**, not find an arbitrary setting
that reduces error. It remains AB-only and supplies no new counterpoise IEs.
Compiler/setup failures, malformed captures, or failed invariants must stay visible.

### Grid-decomposition job 13888500: setup failure, no scientific results

Job and `.batch` failed exit 1 after 15 seconds; the srun step failed exit 1.
The C-API shim compiled successfully, but both workers failed to import Psi4:
site GCC-12 `libstdc++.so.6` lacks `GLIBCXX_3.4.31`, which the pinned Psi4 core
requires. No SCF or grid capture ran. `COMPLETE.json` records false; missing
worker result files and traceback logs remain in the archive.
Prolog confirms `embers`.
Local archive:
`/home/awallace43/docs/saptdft/cuest/mgga-ie/20261008-grid-decomposition-01/`.

The compiled shim's `readelf -d` reports an embedded **DT_RPATH** pointing at
the site GCC-12 lib64 directory. That preload object's runtime search overrides
the intended conda `LD_LIBRARY_PATH`; ordinary `ldd core.so` without the preload
still resolves conda and did not catch the problem.
This is an instrumentation environment error, not a cuEST scientific result.

### Linkage-corrected decomposition queued: job 13893248

Experiment commit `63c2f381d7d1b9d2c60fe4f5974127b2edb4260c`.
The shim requests an explicit conda RUNPATH. More importantly, workers preload
the existing, tested conda `libgcc_s.so.1` and `libstdc++.so.6` **before** the
capture shim. Their hashes are recorded and verified unchanged after the run.
Worker `/proc/self/maps` checks require those actual runtime objects, preventing
silent site-library substitution. Import failures now write an attributable
worker `result.json` when possible.

Twelve local tests pass, including preload-order regression; compile/shell syntax
checks pass. Compute-node import/capture validation remains pending.

Job **13893248**: H200, eight CPUs, 64G, one hour,
`gts-cs207-chemx / gpu-h200 / embers`.
Initial state PENDING / Priority, StartTime=Unknown; effective QOS=embers.
Run:
`/storage/project/r-cs207-0/awallace43/runs/psi4-mgga-ie/20261008-grid-decomposition-02/`.
Background monitor uses `watch_jobs.sh -i 180`. Verify prolog QoS and inspect
capture/control checks before interpreting the energy decomposition.

### Grid decomposition 13893248: validated AB mechanism

Job, `.batch`, srun and both workers completed exit 0 in 4:10.
`COMPLETE.json` confirms execution complete and final provenance verified.
Actual runtime maps select the tested conda libgcc and `libstdc++.so.6.0.34`.
Prolog confirms `embers`. Local archive:
`/home/awallace43/docs/saptdft/cuest/mgga-ie/20261008-grid-decomposition-02/results-13893248/`.

The passive capture agrees with its inactive control: XC energy difference
<1e-10 Eh by assertion; potential max change 6.66e-16 (M06),
4.44e-16 Eh (M06-L). Captured and native-collocated density integrals on cuEST
points agree at displayed precision (~83.99993 electrons). All replay,
finite-data, runtime-map and provenance checks pass.

Telescoping XC-energy differences at the **same density** (Eh):

| Contribution | M06 | M06-L |
|---|---:|---:|
| Native basis-screening control | +7.11e-15 | +2.84e-14 |
| **Change to actual cuEST coordinates/weights** | **+6.15802938e-5** | **−1.42495247e-5** |
| Native versus cuEST collocation/ingredients on cuEST grid | 0.0 | 0.0 |
| Functional-adapter remainder | +2.80e-12 | +1.55e-12 |
| Total cuEST−CPU XC difference | +6.15802966e-5 | −1.42495231e-5 |

Thus the AB **energy** discrepancy is reproduced with **native CPU AO
collocation and native LibXC**, simply by supplying cuEST's actual quadrature.
It is not necessary to invoke different density/τ conventions or CUDA functional
kernels to explain these cases. This is a demonstrated mechanism, not just
a grid setting that reduced the error.

Actual CPU grid: 890,358 retained points; cuEST: 983,376 points, including
188,758 points with exactly zero total weight. Only 4,116 cuEST points match
a retained CPU coordinate within 1e-9 bohr (0.4186%). Median nearest-coordinate
distance is 0.06976 bohr. The CPU orientation matrix is nonidentity; the shared
Psi4/cuEST path bypasses that native orientation and native weight partition/
retention controls.
The experiment isolates the **combined backend quadrature difference**;
orientation versus nuclear partition versus retention is not individually
decomposed. It is not a proof of every potential-matrix element's correctness.

Native/cuEST ingredient differences at nonzero-weight points are roundoff:
rho max ≤1.35e-13, gradient-component max ≤1.48e-12, one-spin tau max
≤3.41e-12; weighted relative L1 errors ≤6.89e-16.
The original per-worker `ingredient_errors.max_abs` includes zero-weight points,
where large differences occur. Those maxima must **not** be used as evidence
of a physical density/τ discrepancy; weighted norms and nonzero-weight maxima
are the meaningful equivalence checks. Source captures and original statistics
remain untouched.

This controls the host-LibXC build and benzene AB for M06/M06-L only. The similar
historical CUDA-build pattern remains suggestive, not a source-matched CUDA
ablation. Ghost-monomer captures are required before claiming direct, complete
counterpoise attribution rather than extrapolating the AB result.

### Counterpoise closure queued: job 13895464

Experiment commit `677a4f675cd3941d2859bb722ca644a4a494db09` adds
`--fragments A B` to the same validated capture/decomposition harness.
It extracts ghosts with `(1,2)` and `(2,1)`, exactly as the original IE harness.
Worker normalization uses each molecule's actual electron count (42 for these
monomers), and AO basis dimension is recorded. The C-API shim and physical
decomposition protocol are unchanged. New records explicitly separate
nonzero-weight ingredient maxima from all-point maxima.
Twelve local tests pass.

Job **13895464**: H200, eight CPUs, 64G, 30 minutes,
`gts-cs207-chemx / gpu-h200 / embers`.
Initial state PENDING / Priority, StartTime=Unknown; effective QOS=embers.
Run:
`/storage/project/r-cs207-0/awallace43/runs/psi4-mgga-ie/20261008-grid-ghosts-01/`.
Background watcher uses `watch_jobs.sh -i 180`. Verify prolog QoS on launch.

After completion, combine the two AB results from 13893248 with four A/B
results from 13895464 **only after checking identical staged-build provenance
and basis dimensions**. For each contribution compute `AB − A − B`.
Compare the resulting **fixed-density XC IE difference** with the prior
host-build converged total-energy IE difference; keep their distinction visible.
There are still no new self-consistent GPU route calculations or recovered
PW6B95 interaction energies in this closure experiment.

### Final-report generator prepared (pre-closure record)

Local-only analysis files, outside the pinned experiment checkout:

```text
/home/awallace43/docs/saptdft/cuest/mgga-ie/analysis/consolidate.py
/home/awallace43/docs/saptdft/cuest/mgga-ie/analysis/test_consolidate.py
```

Four report unit tests and Python compile checks pass. The generator refuses
partial/failed capture runs, requires identical staged-build and runtime hashes
across AB/A/B jobs, verifies captured artifact hashes and basis dimensions, and
checks native CPU IE drift against the archived reference. It retains all 48
comparison rows, invalid diagnostics, original process failures and scope limits.
It generates audited JSON, Markdown and standalone HTML only after all input
gates pass. No final report has been generated yet.

Once 13895464 is archived locally, run:

```bash
python3 /home/awallace43/docs/saptdft/cuest/mgga-ie/analysis/consolidate.py \
  --archive /home/awallace43/docs/saptdft/cuest/mgga-ie \
  --harness /home/awallace43/gits/psi4__worktrees/cuest-mgga-ie-20261007/devtools/benchmarks \
  --output /home/awallace43/docs/saptdft/cuest/mgga-ie/report-20261008
```

The output directory must not exist; choose a new output if revising the report.
Render-check HTML in light/dark/narrow views before presenting it. The visual
language may reuse the previous project's report palette, but meta-GGA content
and files are separate from the SAPT/Protein157 report.

### Final ghost closure and report — complete

Job **13895464** completed: job, batch and srun `COMPLETED / exit 0`,
elapsed **7:58**. Downloaded stdout verifies effective `embers` QoS.
All four scientific workers pass (`ok` and `capture_validated`); `COMPLETE.json`
records calculations complete and final provenance unchanged.

Archive:

```text
/home/awallace43/docs/saptdft/cuest/mgga-ie/20261008-grid-ghosts-01/results-13895464/
```

The consolidated audit verified identical staged-build provenance, capture
source hashes and preloaded runtime hashes across AB and ghosts, all captured
artifact hashes, and the common **384-function counterpoise basis**. The
initialized monomer replay checks use 42 electrons. The final native CPU IE
differs from the archived reference by −3.12639e-13 Eh for M06 and
+4.83169e-13 Eh for M06-L.

For each controlled contribution, compute **AB − A − B** at native CPU seed
densities:

| Contribution / Eh | M06 | M06-L |
|---|---:|---:|
| Native screening control | +1.06581410e-14 | −2.84217094e-14 |
| Actual-grid contribution | **+6.09192264e-6** | **+3.66340345e-6** |
| Collocation/ingredients | −3.55271368e-15 | 0 |
| Functional-adapter remainder | −2.64677169e-12 | −2.60058641e-12 |
| Fixed-density XC IE difference | **+6.09192000e-6** | **+3.66340082e-6** |
| Archived optimized SCF IE difference | +6.07148141e-6 | +3.64337711e-6 |
| SCF-reference residual | −2.04385913e-8 | −2.00237054e-8 |

Thus **the combined quadrature-grid effect quantitatively reproduces the
fixed-density XC IE discrepancy**, directly including both ghost monomers.
The residual against optimized SCF includes seed/SCF and J/K numerical effects;
it is not isolated density relaxation. This is a host-build M06/M06-L result,
not a source-matched historical CUDA ablation or a newly optimized GPU SCF IE.
Orientation, partition and point retention remain combined rather than separately
isolated. Potential-matrix/response equivalence is not universally certified.

Across all six captures, each cuEST grid has 188,758 exactly zero-weight points.
On nonzero-weight support, maximum absolute ingredient differences are
ρ ≤ 1.35004e-13, any gradient component ≤ 1.47793e-12, and one-spin τ ≤ 3.41061e-12.
Capture-active versus capture-inactive potential changes are ≤ 3.24741e-15 Eh.
Large raw maxima at zero-weight points remain preserved, not mistaken for
physical disagreement.

The report generator ran successfully with **all evidence gates intact**:

```text
/home/awallace43/docs/saptdft/cuest/mgga-ie/report-20261008/mgga-results.json
/home/awallace43/docs/saptdft/cuest/mgga-ie/report-20261008/mgga-results.md
/home/awallace43/docs/saptdft/cuest/mgga-ie/report-20261008/mgga-results.html
```

JSON retains all 48 comparison rows, including 32 strict passes, four strict
failures and 12 incomplete PW6B95 comparisons, plus provenance SHA256 values,
original process outcomes and invalid diagnostic records. The four IE failures
remain failures; the report identifies their mechanism rather than lowering
the threshold. PW6B95 still has no complete IE.

Verification:

- Report analysis: **4/4 unit tests pass**.
- Experiment harness: **12/12 unit tests pass**.
- Chromium: desktop 1280×900 light/dark and mobile 375×812 dark checked.
  No document horizontal overflow, no clipped SVG labels, no external scripts
  or stylesheet resources; theme toggle works.
- Expanding the all-comparisons table exposes exactly 48 rows:
  32 PASS, four FAIL, 12 INCOMPLETE.
- Screenshots `desktop-light.png`, `desktop-dark.png`, `mobile-dark.png` live
  beside the report. The design follows the existing cuEST report palette;
  this remains a separate ordinary-SCF artifact.

Optional loopback-only annotation session:
`http://127.0.0.1:4387/session/997e8d971abb83aa`.
Background task `bd3fb7074` waits for review feedback and wakes this agent on
delivery. It is a review listener, not a scientific job. A remote viewer must
forward remote port 4387 from their own device using their normal Pi-host SSH
destination; no public sharing was performed.

The four original investigation/reporting goals are complete. Future work, if
requested, should be a separately approved scope: a source-matched grid-parity
implementation and regression suite, a matched full AB/A/B PW6B95 recovery
protocol, or broader functional/reference/property coverage. Do not claim those
extensions have already been tested.

### User follow-up: IE acceptance threshold

Use **|ΔIE| ≤ 1e-5 Eh** (0.00627509 kcal/mol) as the user's acceptance criterion.
All **36/36 complete comparisons pass**; the maximum remains 6.07148141e-6 Eh.
The four original 1e-6 Eh failures are therefore passes at the revised tolerance.
Do not relabel the 12 incomplete PW6B95 comparisons as passes or apply the IE
tolerance to dimer total energies. No calculations or raw evidence changed.
The report generated above intentionally preserves its original stricter
diagnostic classification; a revised presentation must label both thresholds.

Converged AB total energies were read from successful archived `result.json`
workers, including the r²SCAN recovery. Host CPU/cuEST-XC values (Eh):

| System | Functional | CPU E_AB | cuEST-XC E_AB |
|---|---|---:|---:|
| water | M06-L | −152.86835098187 | −152.86835118957 |
| water | M06 | −152.82110696674 | −152.82110768492 |
| water | PW6B95 | −153.05347524974 | −153.05347514635 |
| water | r²SCAN | −152.81142041427 | −152.81142054696 |
| benzene | M06-L | −464.48051992162 | −464.48053420500 |
| benzene | M06 | −464.14779962107 | −464.14773812726 |
| benzene | PW6B95 | −465.16244627465 | −465.16249042883 |
| benzene | r²SCAN | −464.26585064617 | −464.26584124151 |
| peptide | M06-L | −497.01987988478 | −497.01987814474 |
| peptide | M06 | −496.75895551755 | −496.75895294223 |
| peptide | PW6B95 | −497.68077470649 | unavailable |
| peptide | r²SCAN | −496.79868741346 | −496.79868777021 |

GPU-JK-only dimer energies agree with the corresponding CPU values within
1.155e-9 Eh across available builds/cases. Successful PW6B95 dimers do not
recover its missing counterpoise IEs.

### User follow-up: archived GPU speedups

Read timings only from successful worker `result.json` records in the original
matrix and r²SCAN recovery. `wall_s` surrounds `psi4.energy` (including its SCF
setup/finalization), not Python startup/import, scheduler wait or process launch.
For a complete IE, sum the three independently measured AB/A/B wall times.
The harness sets eight Psi4 CPU threads; archived environment/prolog identifies
an NVIDIA H200. These are single archived runs, not repeated performance trials.
Ranges below are across M06-L, M06 and r²SCAN, **not repeat-run uncertainty**.

| System | Host-LibXC full-XC CP speedup | Historical CUDA-LibXC full-XC CP speedup |
|---|---:|---:|
| water | 1.13–1.19× | 1.21–1.27× |
| benzene | 11.40–11.69× | 15.53–15.70× |
| peptide | 4.49–4.58× | 6.51–6.82× |

Representative M06 complete-IE times (seconds, CPU versus full cuEST XC):

| System | Host CPU → GPU | Historical CUDA-build CPU → GPU |
|---|---:|---:|
| water | 5.163 → 4.458 | 5.311 → 4.401 |
| benzene | 177.210 → 15.503 | 176.409 → 11.359 |
| peptide | 59.522 → 13.004 | 60.726 → 9.079 |

Assessment: full cuEST XC gives useful wall-time acceleration on benzene and
peptide at the user's IE tolerance. Water is too small for a compelling benefit,
especially once process startup is included. GPU-JK-only is not a useful speedup
here: complete-IE median CPU/GPU-JK ratios are 0.62–0.72 in the host build and
0.74/1.04/1.00 for water/benzene/peptide in the historical CUDA build.

For host benzene/M06 AB, native CPU spends 50.900 of 61.881 seconds in
`RV: Form V`; GPU-JK-only reduces iterative JK from 2.237 to 0.0838 seconds but
native XC takes 93.195 of 96.645 seconds. Full cuEST XC takes 5.161 seconds
overall. This demonstrates the XC bottleneck, not the cause of the slower native
XC timing in the mixed route; that regression was not separately diagnosed.

Historical host/CUDA builds are not source-matched: do not attribute their
speedup difference solely to CUDA LibXC. Iteration counts can vary, so these are
time-to-convergence ratios, not normalized kernel-throughput benchmarks. Failed
PW6B95 CP references have no defensible complete-IE speedup.

The user accepts the numerical discrepancy; its demonstrated cause remains
quadrature grids. No density-fitting-error estimate or exact-integral comparison
was performed, so “within density-fitting differences” is a tolerance judgment,
not a proven attribution.

### PR update and ωB97M-V coverage — in progress

User requested the meta-GGA speedup table in the cuEST PR and ωB97M-V in the
tested list with speedups. Updated **Awallace3/psi4#12** (the user's
`saptdft_cuest` PR), not Lori's upstream integration PR #3479.
Inserted a marked `Ordinary SCF meta-GGA support comparisons` section immediately
before Testing and verified the existing body is otherwise byte-preserved.
It includes the measured table, 1e-5 Eh acceptance, all 36 complete passes,
the 12 unresolved PW6B95 rows, scientific scope/provenance and **pending** ωB97M-V.

Local exact inserted section:
`/home/awallace43/docs/saptdft/cuest/mgga-ie/analysis/pr-mgga-section.md`.
Body SHA256 before:
`793242a3d6b515622e610012a149581a3a914fcc5b5dd9a04a173c48287d3311`;
after:
`21a61e9a1715c5874d8c90e090ea300a5cfae68b7cc2638ebee243e851a0a328`.
The section markers allow replacing only this section on subsequent updates;
always fetch the live PR body and preserve unrelated edits.

New dedicated experiment worktree (original archives/harness remain pinned):

```text
local: /home/awallace43/gits/psi4__worktrees/cuest-mgga-wb97mv-20261008
remote: /storage/project/r-cs207-0/awallace43/gits/psi4.cuest-mgga-wb97mv-20261008
branch: cuest-mgga-wb97mv-20261008
commit: 0857cee287301b58b400161563c564458567c7a8
```

Committed and synced via git. `mgga_ie.py` now includes `wb97m-v` by default,
retains legacy 1e-6 diagnostic plus 1e-5 acceptance, and emits dimer and full-CP
speedups only for complete, successful matched records with valid timings.
Returned functional must report MGGA + LRC + VV10 and ω>0; finite nonzero
`DFT VV10 ENERGY` is required. Intrinsic VV10 is **not disabled** or replaced
by D3/D4. The ordinary SCF CPU/GPU routes and convergences are unchanged.
**15 unit tests pass**, Python compile and shell syntax pass.

Water preflight **13904944**, dry-run pseudo-ID 13904943 (not another submission):
H200, eight CPUs, 64G, two-hour request, 300-second per-worker timeout,
`gts-cs207-chemx / gpu-h200 / embers`. Initial effective queue:
`PENDING / Priority`, QOS=embers. Prolog confirmation awaits launch.
18 fresh-process workers: two historical builds × three routes × AB/A/B;
four expected CPU/GPU comparisons.

```text
remote: /storage/project/r-cs207-0/awallace43/runs/psi4-mgga-ie/20261008-wb97mv-water-01/
local submission: /home/awallace43/docs/saptdft/cuest/mgga-ie/20261008-wb97mv-water-01/submission.json
```

Background watcher **b407c1652** (`watch_jobs.sh -i 180 phoenix 13904944`)
wakes this agent on terminal state. Do not poll it. Source delegate
`d9c8da4197e4055ec63da2c1f9a9ade82` independently inspects functional/VV10
plumbing and will notify on completion; retrieve with `bg_result`.

**Next on water terminal notification:** inspect batch/worker exits and
scientific artifacts, download archive, check prolog embers, finite nonzero VV10,
range separation, route timers and final staged provenance. Then submit separate
benzene and peptide jobs at the same commit/protocol using explicit embers,
new run directories and completion-aware watchers. Do not advance either
experiment checkout while jobs still pin it.
When audited ωB97M-V results are available, replace pending coverage in the
marked PR section with its per-system/build IE errors and complete-IE speedups;
do not silently include it in the existing three-functional ranges.
Any errors/missing references must remain visible. No new scientific results
are claimed at submission time.

### Independent ωB97M-V source check retrieved

Delegate `d9c8da4197e4055ec63da2c1f9a9ade82` completed and its answer was
hash-verified. Canonical registration is `wB97M-V` / alias `WB97MV`,
LibXC `HYB_MGGA_XC_WB97M_V` (`libxc_functionals.py:145`).
Current source supports MGGA, long-range cuEST exchange and a separate cuEST
VV10 nonlocal route; host-LibXC means host functional evaluation, not native
CPU XC fallback. The source check is not proof of the historical binaries.

Standard expected parameters: ω=0.3, short-range HF α=0.15, additional long-range
β=0.85 (total LR HF=1), VV10 b=6, C=0.01. Parameter values come from linked
LibXC and should be recorded from the returned functional.
Current source's `DFT_VV10_POSTSCF` default is false
(`read_options.cc:2041`), so the queued harness does not deliberately select a
post-SCF-only variant. Its nonzero-VV10/LRC guards are already active.

Before the full follow-up, after the queued water pin is no longer active,
strengthen recording with explicit `DFT_VV10_POSTSCF=False`, matched explicit
VV10-grid settings, returned α/β/b/C and the five public SCF energy components.
Check total energy equals nuclear + one-electron + two-electron + semilocal XC
+ VV10 within 1e-9 Eh, and equals `DFT FUNCTIONAL TOTAL ENERGY`; record rather
than infer the actual historical-binary runtime options. If that changes the
harness commit, repeat water under the final pin so the published three-system
ωB97M-V matrix has identical declared protocol. Do not advance the checkout
while 13904944 still pins it.

Public energy variables and output labels are confirmed in
`scf_iterator.py:1130–1178`; `needs_vv10()` or mere variable presence alone is
not sufficient. The existing worker requires finite nonzero VV10. cuEST RKS
nonlocal energy/potential is added at `v.cc:2425–2499`; range-separated exchange
parameters are forwarded in `cuESTJK.cc:140–166` and the combined scaled K is
handled in `rhf.cc:239–255,340–372`.

### Water preflight moved to inferno with user approval

After an eight-hour embers wait, user explicitly requested **“submit to inferno”**.
Approval is scoped to water job **13904944**, not future benzene/peptide jobs.
Moved the existing queued job in place, with no duplicate submission:

```bash
scontrol update JobId=13904944 QOS=inferno TimeLimit=02:00:00
```

Readback: `13904944 PENDING inferno gpu-h200 (None)`. Same job ID, pinned
commit, run directory, H200/8CPU/64G and two-hour request. Existing background
watcher **b407c1652** still watches the correct job. Prolog QoS confirmation and
scientific validation await launch/completion.

The first QoS-only update was rejected with “Requested time limit is invalid
(missing or exceeds some limit)”; fresh `sbatch --test-only --qos=inferno`
with two hours was valid. Re-stating the **unchanged** TimeLimit in the in-place
update succeeded. Dry-run identifier **13911207** is not a submitted job.
Local `submission.json` retains original submitted QoS=embers plus the approved
effective QoS=inferno and command/readback evidence. Do not treat this as
permission to charge future jobs to inferno.

### Water preflight passed; final three-system campaign submitted

**13904944**: job, batch and srun `COMPLETED / 0:0`, elapsed **2:49**.
Downloaded archive verifies prolog effective **inferno** and successful
`COMPLETE.json`: 18/18 workers, four complete comparisons, all pass 1e-5,
final unchanged staged provenance verified. Independent `mgga_archive.replay`
exactly reproduces all four stored rows and finds no process failures.
All returned functionals are MGGA/LRC/VV10, ω=0.3, with nonzero VV10 in
every AB/A/B and route. Independently summing the five printed energy components
matches returned total energies within **2.84217e-14 Eh**, proving VV10 inclusion.

Archive and audit:

```text
/home/awallace43/docs/saptdft/cuest/mgga-ie/20261008-wb97mv-water-01/results-13904944/
/home/awallace43/docs/saptdft/cuest/mgga-ie/20261008-wb97mv-water-01/PREFLIGHT-AUDIT.json
```

| Build | Full-XC ΔIE / Eh | CPU CP / s | GPU CP / s | CP speedup |
|---|---:|---:|---:|---:|
| host | −1.17115491e-7 | 25.6448 | 4.8413 | 5.30× |
| historical CUDA | −1.17121814e-7 | 29.9292 | 4.9040 | 6.10× |

GPU-JK-only max |ΔIE| is **1.17986914e-7 Eh** for this preflight, not the
near-roundoff errors for the earlier three functionals. It is slightly slower
than CPU. Native host VV10 consumes 7.3777 of 9.269 AB seconds, explaining why
ωB97M-V benefits substantially on water unlike the VV10-free functionals.
No general mechanism attribution for this new GPU-JK error is claimed.

After all preflight workers were terminal, strengthened the harness:
explicit self-consistent VV10 (`POSTSCF=False`, 50×146, rho cutoff 1e-8);
returned α/β/ω/b/C checks (0.15/0.85/0.3/6/0.01); public five-component and
`DFT FUNCTIONAL TOTAL ENERGY` equality within 1e-9 Eh.
**16 unit tests**, Python compile, shell syntax and diff-whitespace checks pass.
Committed, pushed and remote fast-forward verified:

```text
ca1c35bed6e0e48407adb13c0535763267eed4b7
```

No scientific C++ or staged build changed. The previous water preflight remains
separate; water is repeated under the final explicit protocol.

Three independent final jobs, each H200/8CPU/64G, two hours, 300-second worker
timeout, 18 workers and four comparisons:

| System | Submitted job | Dry-run pseudo-ID (not submitted) | Initial effective state/QoS |
|---|---:|---:|---|
| water | **13911317** | 13911316 | PENDING / Priority / embers |
| benzene | **13911319** | 13911318 | PENDING / Priority / embers |
| peptide | **13911321** | 13911320 | PENDING / Priority / embers |

Account/route: `gts-cs207-chemx / gpu-h200 / embers`.
Headroom was 8/50 before submission. All three `sbatch --test-only` validations
passed; all actual submissions accepted. No inferno approval for these jobs.
Remote runs:
`/storage/project/r-cs207-0/awallace43/runs/psi4-mgga-ie/20261009-wb97mv-{water,benzene,peptide}-01/`.
Local submission manifest:
`/home/awallace43/docs/saptdft/cuest/mgga-ie/20261009-wb97mv-matrix-01/submission.json`.
Watcher **b9d6bbd55** monitors all three with `watch_jobs.sh -i 180`, and wakes
this agent once all are terminal. **Do not advance either pinned experiment
checkout before retrieval.** Background polling is unnecessary.

Updated the marked PR #12 section with measured water results/speedups and
clearly pending final jobs. Unrelated PR content was preserved and readback
verified. Latest PR body SHA256:
`5b22c693d425eb56f0d6173e8c4a7df55d352d1ec0fccd6e892848f8a7e309a7`.
The legacy speedup ranges remain explicitly scoped to M06-L/M06/r²SCAN;
ωB97M-V water is a separate measured table, not silently folded into them.

**Next:** on matrix terminal notification, inspect batch/worker outcomes and
all scientific gates, download each archive to the matching local name, verify
prolog embers and consistent staged hashes across final runs, independently
replay rows and inspect configuration/component/route/timing records.
Publish final per-system/build ωB97M-V IE errors and speedups in the marked
live PR section, preserving unrelated edits. Compare final water to preflight
but exclude the preflight repeat from coverage counts. Retain failed/incomplete
cases and keep task #9 in progress until its unresolved evidence is handled.

### Final attempts failed before SCF — Boolean binding guard fixed

Watcher **b9d6bbd55** delivered ALL-TERMINAL with job/batch/srun failure 1:0:

| System | Job | Elapsed |
|---|---:|---|
| water | 13911317 | 0:56 |
| benzene | 13911319 | 1:07 |
| peptide | 13911321 | 0:48 |

Downloaded all three archives. Each has 18 worker errors, **zero energy
results**, `calculations_complete=False` and final unchanged provenance verified.
All effective prologs are embers, and both build staged-hash maps exactly match
the successful preflight 13904944. This is not evidence of a new scientific
functional, convergence, or runtime-library failure.

The exact traceback points to:

```python
assert record["vv10_runtime_options"]["DFT_VV10_POSTSCF"] is False
```

Every captured value is integer **0**, meaning the intended disabled setting.
`0 is False` is false in Python. The assertion ran before molecule construction
and `psi4.energy`, so no SCF started. This observed value/trace identifies the
cause directly; speculative alternative hypotheses were unnecessary.

Preserved original artifacts and added a `SETUP-FAILURE.json` beside each
`20261009-wb97mv-{water,benzene,peptide}-01` archive. The first matrix remains
failed; never rewrite its worker or COMPLETE records to success.

Extracted the actual call-site gate into `validate_vv10_runtime_options`.
The captured-option regression went **red** against the old identity check
and **green** after accepting Boolean False or integer 0, while rejecting
enabled/unknown values. Replayed the gate on **all 54 real captured option
records** successfully. Seventeen unit tests, Python compile, shell syntax and
diff-whitespace checks pass. No physical SCF/VV10 setting or staged build changed.
This proves the erroneous setup gate is repaired, not that unrun final SCFs pass.

Fix committed, pushed and remote fast-forward/HEAD verified:

```text
45c7902013948221e18219a50f3a18dc934ba772
```

Both experiment checkouts can now be pinned to this fix for **new run directories**.
All previous pinned jobs are terminal. No retry has been submitted yet:
latest `job_status.sh phoenix` reports **embers submit slots used: 50/50**.
Previous inferno approval only covered preflight 13904944; do not silently reuse it
for three new charged jobs.

PR #12 marked section now records the setup failures, the fix/regression evidence
and ready-but-unmeasured corrected retries; successful preflight numbers remain.
Unrelated body preserved and live readback verified. Latest body SHA256:
`8d167a931bd1a7e33e3f4af58aa6763b79e26deb05b7756f4a9d799a8e5e67ee`.

Task #11 (diagnose/repair this setup gate) is complete. Task #9 remains pending,
blocked by #12 (submission capacity / retry QoS approval). Next decision:
explicit inferno approval for three H200/8CPU/64G jobs, two-hour limit each,
or retain prepared retries until at least three free embers slots open.
After actual retry acceptance, record job IDs/manifest, freeze both checkout pins,
start a notification-aware watcher, and follow the existing full scientific audit
before updating final support/speedup rows. Any subsequent failure is a new
diagnostic problem; do not label the full three-system campaign complete now.

### Corrected retries submitted with fresh inferno approval

Structured approval:
“Submit the three corrected benchmark retries with inferno?” →
**“Approve inferno”**. Scope is exactly three new H200/8CPU/64G retry jobs,
two-hour limit each; no blanket paid recovery/follow-up approval.

All three `sbatch --test-only --qos=inferno` validations and actual submissions
succeeded at fixed harness pin **45c7902013948221e18219a50f3a18dc934ba772**:

| System | Actual job | Dry-run pseudo-ID (not submitted) | Initial queue |
|---|---:|---:|---|
| water | **13940495** | 13940494 | PENDING / Priority / inferno |
| benzene | **13940497** | 13940496 | PENDING / Priority / inferno |
| peptide | **13940499** | 13940498 | PENDING / Priority / inferno |

Readback separately verifies each job has **2:00:00**, **8 CPUs**, **64G**,
and **gres/gpu:h200:1**. The explicit command-line `--qos=inferno` overrides
the batch script's default embers directive. Prolog verification awaits launch.
Same 300-second worker timeout, physical SCF/VV10 settings and staged builds.
All previous failed runs remain intact; no old job was cancelled or rewritten.

Remote new runs:
`/storage/project/r-cs207-0/awallace43/runs/psi4-mgga-ie/20261010-wb97mv-{water,benzene,peptide}-01/`.
Local submission/approval manifest:
`/home/awallace43/docs/saptdft/cuest/mgga-ie/20261010-wb97mv-matrix-01/submission.json`.

Watcher **bb728b28b**:
`watch_jobs.sh -i 180 phoenix 13940495 13940497 13940499`.
It wakes this agent after all three are terminal; do not poll merely to wait.
Freeze both experiment checkouts at 45c7902013 until retrieval. Do not commit
new reports into the pinned checkout while these jobs remain queued/running.

PR #12 marked section now includes the corrected retry IDs and explicit inferno
approval, while retaining failed-setup history and the separate measured water
preflight. Unrelated content preserved and roundtrip verified. Latest body hash:
`c700904f13edb0e5223382f76996979854209e8b73f19c125171d48a25f633f4`.
Task #12 (submission capacity/approval) is complete; #9 is **in progress** awaiting
the corrected scientific results. No new final-matrix pass or speedup claim.

**Next on notification:** retrieve watcher logs (do not reconfirm terminal via
bg_status), download three new archives, inspect each batch/worker outcome and
COMPLETE marker. Check effective inferno prologs, returned functional/configuration,
VV10/energy components, routes, matching staged hashes and final unchanged
provenance. Independently replay comparisons and calculate per-system/build CP
timings. Preserve any new errors/missing references; do not certify absent IEs.
Update only the marked live PR section with audited final coverage/speedups.
Exclude preflight repeats and all setup-failure runs from complete-case counts.

### Final corrected ωB97M-V audit and PR publication

Watcher **bb728b28b** completed; all archives retrieved and audited:

| System | Job | Terminal state | Successful workers | Elapsed |
|---|---:|---|---:|---|
| water | 13940495 | COMPLETED 0:0 | 18/18 | 2:58 |
| benzene | 13940497 | FAILED 1:0 | 17/18 | 41:05 |
| peptide | 13940499 | COMPLETED 0:0 | 18/18 | 34:17 |

All prologs verify effective inferno. All final staged-provenance checks pass;
staged input maps match across the three jobs. Each archive independently
replays its stored comparisons, including benzene's explicit incomplete row.
The 53 successful records retain MGGA/LRC/VV10 and nonzero VV10 energy.
Returned settings: ω=0.3, α=0.15, β=0.85; b=6, C=0.01; self-consistent
VV10 (`DFT_VV10_POSTSCF=0`), 50×146 nonlocal grid, density cutoff 1e-8.
Maximum energy-component sum drift: **8.526512829121202e-14 Eh**.

Complete-IE times sum AB + A(ghost B) + B(ghost A), around `psi4.energy`:

| System | Build | CPU / s | Full GPU / s | Speedup | GPU−CPU IE / Eh |
|---|---|---:|---:|---:|---:|
| water | host | 22.918 | 4.192 | 5.47× | −1.171123926724249e-7 |
| water | cuda | 23.634 | 4.182 | 5.65× | −1.1713289893577894e-7 |
| benzene | host | 515.743 | 16.265 | 31.71× | −6.685184473553818e-6 |
| benzene | cuda | 567.608 | 13.677 | 41.50× | −6.686622100460227e-6 |
| peptide | host | 439.199 | 15.154 | 28.98× | +6.078087153582601e-7 |
| peptide | cuda | 503.757 | 12.519 | 40.24× | +6.073031784126215e-7 |

**All six full-GPU rows pass 1e-5 Eh.** ωB97M-V has 11 complete/pass
comparisons and one incomplete GPU-JK-only control. Across the original
matrix and this addition: **47 complete/pass out of 60 planned**, with
12 PW6B95 and one ωB97M-V control incomplete. All **24 full-GPU comparisons**
for M06-L/M06/r²SCAN/ωB97M-V pass; six benzene rows exceed the tighter
1e-6 diagnostic threshold. Maximum |ΔIE| is **6.686622100460227e-6 Eh**
(0.004195918717518498 kcal/mol).
ωB97M-V's five complete GPU-JK-only controls reach |ΔIE|=3.5629361150313343e-7
and underlying |ΔE|=1.1237573147582225e-5 Eh, so do not extend the earlier
three-functional near-roundoff JK claim to this functional.
ωB97M-V full-GPU maximum underlying |ΔE|=1.2667098587826331e-5 Eh;
the combined maximum remains M06's 6.149380755e-5 Eh.
The M06 grid-attribution experiment was not repeated for ωB97M-V.

Missing worker: `benzene-wb97m-v-host-gpu-jk-AB`, timeout return **124**
at 300 s. Last printed iteration 20 has ΔE=5.47402e-11 and density
residual 1.05636e-12: energy criterion 1e-11 is not met.
There is no result JSON or certified final energy. Preserve the incomplete
control rather than treating the last printed energy as converged.
Both benzene full-GPU comparisons use complete CPU/full-GPU AB/A/B records
and remain valid despite the enclosing batch's failed status.

Local archive root:
`/home/awallace43/docs/saptdft/cuest/mgga-ie/`
under `20261010-wb97mv-{water,benzene,peptide}-01/results-{13940495,13940497,13940499}`
(pair each system with its matching job).
Remote root: `/storage/project/r-cs207-0/awallace43/runs/psi4-mgga-ie/`.

Manifest SHA256 by system:
- water: `e78d53aa40a226fb48cf37d944864566595c30fef0d5c1d48de2823a121c98dc`
- benzene: `633fd6c3241e2a59592e049f224a7784aa1048854e8ecdff6ced0e49579dff02`
- peptide: `2ac3d293c4483090a885446faf8c03237a26dc4db18fd669bc42c18be305aa63`

Scientific worker pin **45c7902013948221e18219a50f3a18dc934ba772**.
After all jobs were terminal, report-only commit
**beb0cacbb36283214a4f1da1e172bf2cf1690136** was pushed to
`cuest-mgga-wb97mv-20261008`. It updates measurement notes and adds compact
`devtools/benchmarks/results/phoenix-h200-20261008-mgga/wb97mv-results.json`.
All 17 harness tests pass; JSON timings/IEs match archived rows exactly,
dimer rounding agrees within 6e-13 Eh, source manifest digests match,
and `git diff --check` passes.
[Public measurement notes and rows](https://github.com/Awallace3/psi4/tree/beb0cacbb36283214a4f1da1e172bf2cf1690136/devtools/benchmarks/results/phoenix-h200-20261008-mgga).

PR #12 now contains the final functional coverage, six measured speedups
and six dimer total-energy pairs; outside-marker text was byte-preserved
and API readback verified.
Previous body SHA256: `c700904f13edb0e5223382f76996979854209e8b73f19c125171d48a25f633f4`.
New body SHA256: `ed4bb5f2a699aff728e35e9d9285d2b252b786fd77f8097cf29c0d21233f925e`.
Local publication evidence:
`/home/awallace43/docs/saptdft/cuest/mgga-ie/analysis/pr-mgga-final-update.json`.

**Scope accepted:** after publication, user selected
**Keep documented gap (Recommended)** rather than another control rerun.
The requested full-GPU results are complete and published; recovery is
out of scope, not silently treated as a passing control. Tasks #13 and #9
can close on this explicitly accepted scope. No follow-up is pending.
Another inferno job would require fresh, job-scoped approval; the existing
three-retry approval is exhausted. No scientific C++ changes or additional
cluster submissions were made to recover this control.

### LDA/hybrid versus meta-GGA implementation audit

User requested checking LDA/hybrid cuEST agreement with MGGA and clarified
**Implementation audit**, not another numerical-control campaign.
Reviewed current HEAD `9317f406b2b66f33b65866cbc44a4189f913339e`, including
user-owned dirty JK files. No code changes, tests/builds or cluster jobs.

Normal SCF dispatch/scaling is consistent: shared 1/4/5-channel
density/gradient/τ collocation, per-component LibXC rung selection and α
weighting, matching host/CUDA spin factors, common hybrid α/β/ω propagation
without double-counting exact exchange, and separate VV10 grid/energy.
No MGGA-only implementation discrepancy was established.

Common caveat: cuEST's molecular grid bypasses native orientation,
nuclear-partition/weight filtering and host basis-block post-processing.
Shared radial/angular/pruning settings are not proof of identical quadrature
for LDA/hybrids either. This agrees with the previous controlled M06-family
grid diagnosis, not a blanket density-fitting explanation.

Separate source-level API concern: `JK::set_omega` only changes the field;
cuEST's already-initialized integral plan has no matching ω invalidation,
unlike α/β setters. Ordinary SCF sets ω before initialization, so this does
not establish a defect in our benchmark results. A reused-JK ω-mutation
regression would be needed before any fix; none was run.

Tests already define SVWN/B3LYP/M06/B97M-V cases at 5e-6 Eh, GRAC
XC-isolated controls and broad conditional LRC/H₂ coverage at 5e-4 Eh.
These were **not rerun**, and they do not establish named LDA/hybrid
three-dimer IE agreement at 1e-5 Eh or host/CUDA backend parity.
Detailed file/line evidence, limitations and targeted follow-ups:
`/home/awallace43/docs/saptdft/cuest/mgga-ie/analysis/lda-hybrid-mgga-implementation-audit.md`.
No PR-body update was made for this source-only audit.

### Lower-rung numerical extension

User now explicitly asked to **run the lower rungs and extend the measured
CPU/GPU table**. Planned controls: SVWN (LDA), BLYP (GGA), B3LYP/PBE0
(ordinary GGA hybrids), ωB97X (range-separated GGA hybrid, no -D/-V).
Same water/benzene/peptide geometries/bases, both staged historical builds,
CPU/GPU-JK/full-GPU routes, all three CP fragments. Planned **270 workers,
60 comparisons, 30 full-GPU rows**. Acceptance stays **1e-5 Eh IE**.

New isolated worktree:
`/home/awallace43/gits/psi4__worktrees/cuest-dft-rungs-20261012`;
remote `/storage/project/r-cs207-0/awallace43/gits/psi4.cuest-dft-rungs-20261012`.
Commit **e44ae0eb8f1a4c076e5a1ad54114eeb57ea9d83c** pushed.
Original MGGA defaults are retained; lower controls require explicit selection.
Harness now checks returned rung/name, hybrid fractions/range separation and
absence of unintended VV10/GRAC for these controls, while preserving ωB97M-V
checks. Pure SVWN/BLYP may retain an inactive beta=1 at omega=0; that
bookkeeping default is recorded rather than falsely rejected as LRC.

Tests-first red import confirmed missing control support; implementation now
passes **20 tests**, batch shell syntax and `git diff --check`.
Scientific E/D thresholds and maxiter are unchanged; worker timeout increases
from 300 to **900 seconds** to avoid premature process truncation.
Staged executables and p4cuest_sapt environment exist; scratch-helper pin
**7d897a7f203297754087ea77536619c378191c8b** verified.

**Free embers only.** Earlier inferno approvals do not apply.
Live initial queue check showed **14/50 embers slots used**.
First submit water with all five controls: 90 workers, 20 comparisons,
one H200/8 CPU/64 GiB allocation, **2-hour** limit.
After successful water functional/route/component/provenance audit,
submit benzene and peptide with **4-hour** limits, same resources and pin.
No stranded dependencies or speculative paid recovery.

Preparation/submission evidence:
`/home/awallace43/docs/saptdft/cuest/mgga-ie/20261012-dft-rungs-matrix-01/submission.json`.
Water output:
`/storage/project/r-cs207-0/awallace43/runs/psi4-mgga-ie/20261012-dft-rungs-water-01`.
Job ID and actual readback are still pending at this preparation checkpoint.
Track #17 (preparation), #18 (submit/monitor), #19 (audit/extend table).
Publish measured lower-rung rows only after raw results/replay verification;
currently the LDA/GGA/ordinary-hybrid entries remain unmeasured.

**Staging recovery checkpoint:** first `phx` fetch/worktree call on `phoenix`
timed out at 110 s before submission. Alternate `phoenix2` read-only checks
confirmed fetched origin pin e44ae0eb8f and no matching queued job/run.
An initially visible incomplete checkout then disappeared through Git's own
abort cleanup; a second inspection found no directory and no registered
lower-rung worktree. Nothing was removed/repointed by this agent.
Background **b74e8dedc (Stage Phoenix DFT controls)** now uses phoenix2,
600-second SSH limit, reuses the already fetched pin, preserves any existing
branch only if it matches that pin, creates a new worktree/run directory and
runs **sbatch --test-only** (not actual submission).
Task #20 tracks this blocker; #17 is pending behind it. On notification:
read staging logs, verify dry-run/resources and remote pin, then close #20/#17
and start #18 before submitting water once. Recheck embers headroom, parse
the actual job ID, read effective resources/QoS, update the manifest and start
a notification-aware watcher. No scientific jobs submitted at this checkpoint.

Staging **b74e8dedc completed successfully**. Remote checkout exactly matches
e44ae0eb8f, tracked files clean; fresh logs/metadata directories created.
Water `sbatch --test-only --qos=embers` accepted the H200/8CPU/64G/2-hour
shape. **13976164 is a dry-run pseudo-ID, not an actual job.**
Tasks #20/#17 are complete; #18 is in progress confirming the actual single
water submission on phoenix2. Preserve/freeze the experiment checkout once
that job is accepted; benzene/peptide are still not submitted.

### Lower-rung water preflight submitted

Actual job **13976174** accepted once via `sbatch --parsable`; before
submission there were **15/50 embers slots** used and no matching queued
water job. Separate `scontrol` readback verifies:
**gts-cs207-chemx / gpu-h200 / embers**, **2:00:00**, **8 CPUs**, **64G**,
**gres/gpu:h200:1**; initial state **PENDING / Priority**.
Effective prolog QoS verification awaits launch. This is not dry-run 13976164.
The script explicitly exports `MGGA_DRIVER=mgga_ie.py`.

Frozen local and remote harness pin:
**e44ae0eb8f1a4c076e5a1ad54114eeb57ea9d83c**.
No commits or checkout advances while this job is queued/running.
Water exercises all five controls, both builds, three routes and CP fragments:
**90 workers, 20 comparisons, 10 full-GPU rows**.
Outputs: `20261012-dft-rungs-water-01/results-13976174/` under the existing
remote `/storage/project/r-cs207-0/awallace43/runs/psi4-mgga-ie/` root.

Watcher **b04540077 (Watch lower rung water)**:
`watch_jobs.sh -i 180 phoenix2 13976174`, notification-aware and ordinary
shell survive-reload opted in. Do not poll merely to wait. On terminal notice,
read bounded watcher logs and retrieve the archive. Verify H200, effective
embers prolog, all worker exits, actual functional name/rung/α/β/ω, zero
unintended VV10/GRAC, energy components, routes, matching bases, replay and
final staged-file hashes. Preserve failures/timeouts rather than certifying
printed energies. Analyze complete comparisons even if acceptance fails,
but distinguish support/runtime completeness from IE threshold agreement.

Only after successful runtime/provenance preflight, submit benzene and peptide
with same frozen pin and 4-hour **embers** limits; check current two-slot
headroom and both dry-runs first. No inferno approval, no dependencies and
no cancellation/mutation of unrelated jobs. Task #18 remains in progress,
#19 pending. No new lower-rung accuracy/speedup values are claimed yet.

### Lower-rung water completed and audited

Watcher b04540077 completed: job **13976174** and its .batch/.0 steps
COMPLETED 0:0, allocation elapsed **00:08:23**. Archive retrieval b45fbe5cf
succeeded; archive SHA256
`561545864d1329ad960c8ecddd871ae8d9ac9110efa06e56f542bf3169484ecd`.
Local results:
`/home/awallace43/docs/saptdft/cuest/mgga-ie/20261012-dft-rungs-water-01/results-13976174`.

Independent frozen-harness replay exactly reproduces comparisons:
**90/90 workers exited 0; 20/20 comparisons complete and pass 1e-5 Eh**.
Actual functional names/rungs/alpha/beta/omega, no unintended VV10/GRAC,
expected routes and physical options, H200, effective embers prolog and
final unchanged-staged-file provenance passed. Maximum energy-component
accounting residual **5.685e-14 Eh**.

Full-GPU max IE error **9.513e-10 Eh**; all-route max **1.19614e-7 Eh**
is GPU-JK-only. Full-GPU complete-IE speedups **0.86–1.04×**: tiny water
has no convincing GPU speed advantage. Preserve these measured slowdowns.
Audited ten-row table:
`/home/awallace43/docs/saptdft/cuest/mgga-ie/analysis/lower-rung-water-13976174.md`.

Benzene/peptide can now be submitted at the SAME frozen harness pin, each
4h/free embers, after live headroom and both dry-runs. Do not mutate either
historical staged package/environment. PR table not yet extended; retain
separation from the new CPSCF response build.

Follow-up submission **b831edefd succeeded**: **benzene 13979120** and
**peptide 13979121**, each **gts-cs207-chemx / gpu-h200 / embers**, 4h,
H200:1 / 8CPU / 64G, PENDING/Priority at readback. Both dry-runs passed
(pseudo IDs 13979117/13979119 are not submitted jobs); 19/50 embers slots
were used before submission. Frozen pin remains e44ae0eb8f, no dependencies.
Watcher **be728fd68** runs `watch_jobs.sh -i 180 phoenix2 13979120 13979121`
with automatic notification and ordinary-shell survive-reload enabled.
On terminal notice retrieve/archive/audit both result sets, then extend the
marked PR #12 comparison section, preserving failed/missing rows.

Audited water rows are now appended inside PR #12's marked comparison
section, explicitly retaining 0.86–1.04× speedups and unmeasured larger
cases. All text outside the markers was verified unchanged.
Published body SHA256:
`32c57d3dd60c43d7d91cdda694b40101a636ee4b8266599d21cddab8540dab3f`.
Local body:
`/home/awallace43/docs/saptdft/cuest/mgga-ie/analysis/pr-12-with-lower-rung-water.md`.
No response-validation claim was added to the SCF benchmark section.

### Larger lower-rung controls audited

Watcher **be728fd68** completed: benzene **13979120**, job/batch 0:0,
40m50s; peptide **13979121**, job/batch 0:0, 16m06s. Both archives retrieved
by **b89f2db23**; remote/local SHA256 match. Independent replay and full
attribution audit succeeded: **180/180 workers, 40/40 comparisons pass
1e-5 Eh**; campaign including water is **270/270 workers, 60/60 passes**.

Actual H200/effective embers, frozen harness e44ae0eb8f, actual functional
names/rungs/alpha-beta-omega, no VV10/GRAC, routing, options, CP bases,
energy-component sums, matching build hashes and final unchanged staged-file
provenance verified. Audit initially used literal one-space prolog matching;
fixed whitespace-aware matching against actual `QOS:       embers` (no
scientific data failure).

Full-GPU CP speedups: **benzene 8.07–9.30×**, **peptide 3.55–5.36×**.
Max |GPU−CPU IE| across both GPU routes:
benzene **1.083017252767604e-6 Eh**, peptide **2.537900911647739e-7 Eh**.
Single time-to-convergence measurements, own-build CPU anchors; not repeated
averages or a source-matched LibXC ablation. No density-fitting-only attribution.

Audit artifacts:
`/home/awallace43/docs/saptdft/cuest/mgga-ie/analysis/lower-rung-followups{.md,-audit.json}`.
Audit/replay script:
`.../20261012-dft-rungs-matrix-01/audit_followups.py`.
Archive SHA256:
- benzene `8933f659fd043f42a61c4b8bf39f868851e6e718fb83568cb7d58ca9874f9dac`
- peptide `b1280eaf258d448c0912a7566e1f69f23604bf51ddb30ecd8dc8216444b9919b`

PR update **bbb1743fc** is running: adds the 20 full-GPU larger-system
rows, replaces obsolete queued/unmeasured text only within owned markers,
preserves prior MGGA/ωB97M-V/water results and all outside-marker text.
Do not claim publication until its readback succeeds. Task #19 remains
in progress; #18 monitoring complete. Response GPU correctness is separate
under job **13980976**, watcher **b9114a614**; see `cphf-check.md`.

Publication **bbb1743fc succeeded** and readback matched exactly. PR #12
now contains the 20 larger-system full-GPU rows and completed campaign
summary. Existing MGGA/ωB97M-V/water tables and all outside-marker text
preserved. Published body SHA256:
`473fd3ce3287932c4bfef93635106ec956a193e5c7f3cbcdbbe1e085d4e71d3f`.
Body saved at `.../analysis/pr-12-with-lower-rung-followups.md`.
Task #19 complete; resume #23 on response-validation terminal notification.

Report-feedback listener **bd3fb7074** failed with interrupted Lavish server
poll (SERVER_ERROR), not a scientific calculation failure. Original artifact:
`/home/awallace43/docs/saptdft/cuest/mgga-ie/report-20261008/mgga-results.html`.
Resumed loopback-bound tracked poll **b83b0c322** with automatic notification;
no artifact edit or forced reopening. Feedback remains queued until delivery.
On notification read its complete feedback/end response; on another server
failure inspect the Lavish server log before retrying. FDDS diagnostics remain
independent under event watcher **b8ed003d1**; no FDDS GPU pass established.
