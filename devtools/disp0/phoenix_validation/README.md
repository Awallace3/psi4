# Fresh Phoenix native build and CPU validation

This is a new build of the current scientific source, not recovery or
relabeling of any previous failed Psi4 installation. It does not submit
protein-size F-SAPT or certify GPU correctness.

The dependency environment, complete prepared CUDA13 development SDK and
installed Einsums2/Waggle pair from allocation14254131 are read-only
build/runtime inputs. That allocation's original Psi4
installation and failed status remain unchanged. The validator requires the
special Einsums commit `06404fa69ce9346443d7d4f19a250f6db9f0f62c`, one native
world, and a newly built, versioned, cuEST-enabled Psi4 stage. It never imports
the old Psi4 core or overlays Python scientific files.

The CPU job:

1. Verifies clean exact-pin source with complete history and annotated v1.11.
2. Inventories CUDA13.3 and creates a fresh private developer-link locator.
3. Pins canonical cuEST native bytes and the complete 59-header aggregate.
4. Configures and builds a whole new Psi4 stage with native cuEST enabled,
   native Einsums disabled (Python E2 is independent), CPU LibXC, and portable
   x86-64-v3 CPU code. HF validation does not need GPU LibXC.
5. Runs the standalone CPU/mock/serial regressions, then imports the new stage
   before E2, checks identities/maps, and executes a native CPU RHF smoke test.

No environment/package installation, GPU allocation/context, old-root changes,
automatic retry, or GPU/scientific fallback is performed. CUDA_VISIBLE_DEVICES
is empty throughout. A successful result is **CPU_BUILD_VALIDATION_ONLY**.
Missing dependencies or incompatible ELF maps fail closed.

`submit.py` supplies every resource option explicitly, tests the route first,
records submission provenance, and permits only one submission per fresh run
root. The job uses 8 CPUs, 48 GiB, 3 hours, and defaults to free embers.
User-approved inferno is selected explicitly with `--qos inferno --account ...`
after running the Phoenix account selector; no post-start QoS move is needed.
Build parallelism
is six; the extra CPUs accommodate orchestration/linking. Watch the captured
job ID with the user-level Phoenix skill's `watch_jobs.sh`.

The two existing uncommitted fixture edits in the development checkout are
not part of this pinned build and are neither committed nor discarded.

## First attempt: preserved failure 14308890

Source `f20fb2b1d7` stopped before SDK preparation/configuration because this
launcher's header serialization sorted `Path` components instead of POSIX
filename strings. Read-only inspection confirmed all 59 real Phoenix headers
match BOTH canonical aggregate digests and all mandatory individual hashes.
This was a launcher defect, not evidence of a changed SDK.

The corrected serializer retains the same canonical expected hash and header
count. Its regression distinguishes root `cuest.h` from nested `cuest/...`
ordering and rejects a changed byte. No original artifact, dependency or guard
threshold is modified. A replacement submission requires a new pin/run root;
the submission helper never retries it automatically.

## Second attempt: preserved failure 14309318

Source `28036fc61f` passed the canonical header check, then stopped before
configuration because the shared toolkit locator lacks `cusparse.h`,
`nvJitLink.h` and their unversioned development links. The subsequent requested
inferno update was rejected because the job was no longer pending; no paid
replacement was silently submitted.

The already prepared read-only `opt064-...-14254131-0/cuda13` dependency has
all required headers, compiler support, static archives and development links.
New submissions explicitly inventory that complete SDK, not the incomplete
shared locator, and construct another fresh private locator in the new run.
The cheap submit-time filesystem check now rejects missing SDK inputs before
any allocation; full byte/ELF validation remains inside the CPU job.

## Verified CPU build: 14309728

Source `3343e673c2f52cb825fb714756289cff0cc59c38` completed on
`gts-cs207-chemx / cpu-small / inferno` in 9m34s. Job, batch and step accounting
all report COMPLETED0:0; runtime status records exit0 and no TERM.

Saved result is PASS / CPU_BUILD_VALIDATION_ONLY: Psi4 `1.12a1.dev843`,
244 CPU/mock/serial regressions, 8 launch guards, 5 independent E2 API checks,
one matching E2 `06404fa...` world, and native CPU RHF
`-1.1167143250625702 Eh`. No stderr was recorded.

Artifacts:
`/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/phoenix-current-3343e673c2/job-14309728/`

The new executable is `build/stage/bin/psi4` beneath that artifact root.
`result.json` retains the source pin, binary hashes, mapped paths and proof
hashes. GPU correctness is explicitly false and protein F-SAPT admission is
false. This successful CPU build is not an H200 gate or a timing measurement.
# Full-fragment qualification follow-up

## Production CPU v2 / C++ comparisons

The user-requested CPU batch uses default production scalar-rescaled fragment
induction (`IND_RESPONSE=false`, exchange/induction scaling true). It does not
invoke the new exact per-source solver. `submit_cpu.py` selects one of three
bounded paired CPU allocations; `compare_cpu.py` runs a small native water
preflight followed by three independent fresh-process calls per backend,
alternating their order on the same exclusive24-thread CPU node.
Only peptide can launch initially; the larger two submitters require the
successful exact-pinned peptide result and terminal0:0 job/batch/step accounting.
Actual scheduler allocations, CPU model and scratch filesystem/free space are
admitted inside the allocation before any scientific calls.

Both arms share HF/DF/CPKS canonical SAPT terms, frozen core, spherical
aug-cc-pVDZ, augmented JKFIT/RI, delta-HF, MINAO and convergence settings.
The native blocked dispersion default48 and optimized v2 default24 are
recorded, rather than forcing either backend onto the other's block size.
Whole energy time includes SCFs and all fragment work; exclusive phase timers
are nonoverlapping. Diagnostic arrays and checks are processed afterward,
without retaining a FISAPT object or DFHelper. F-SAPT disk output is disabled.

Successful results require correct dispatch, finite full matrices, canonical
signed scalars including both dispersion components, full fragment sums,
zero external/frozen slots, converged localizers and frozen/active-preserving
occupied-gauge permutations. Raw results and failures are preserved. Scratch
peaks are explicitly sampled logical sizes, not exact filesystem/cgroup peaks.
The native localizer's1000-sweep default and Python's50-sweep default are
recorded; a printed localization failure is fatal to qualification.

This CPU-v2/C++ ratio is **not** a GPU speedup. The current GPU correctness
gate uses exact per-source response, a different fragment approximation.

## Exact-response GPU diagnostic gate

`COMPLETION_PLAN.md` records the agreed full F-SAPT scope (peptide, nanotube,
protein83; spherical aug-cc-pVDZ), not a dispersion-only benchmark.

The optional `submit.py --prepare-gpu-runtime` step prepares frozen private
cuda-bindings wheels and a CUDA loader manifest **inside the CPU allocation**.
It then exercises the public exact-coupled CPU fragment path, unscaled and
delta-HF/exchange scaled. No CUDA context is created in this preparation job.
`result.json` remains CPU-build evidence only; also inspect
`gpu-runtime/preparation-result.json`, both `cpu-fragment-*/result.json` and
the enclosing job status.

`fragment_gate.py` is a diagnostic fresh-process CPU/GPU entrypoint. Its GPU
arm requires the exact prepared configuration SHA, native cuEST HF, an owned
audited provider, actual batched FP64 BLAS controls, high-index IBO reductions,
no CPU fragment integral/localizer fallback and final cleanup. It captures
raw HF and final matrices for independent comparison. Its output status
explicitly does **not** claim independent correctness or accepted timings.

The separate `submit_gate.py` route is one H200, 4 CPUs, 32 GiB, at most
one hour. It requires both completed native CPU preflights and a prepared
configuration SHA. QoS/account are explicit; inferno still requires fresh
scoped approval and live account selection.

`run_gate.py` invokes ten fresh sequential diagnostic processes: matched
CPU/GPU and independent same-GPU-gauge CPU replay for unscaled, scaled and
changed inputs, then a fresh GPU repeat. The oracle rebuilds native CPU AO
JK, verifies HF projectors and one-electron inputs, reconstructs IAO charges
and checks final-gauge stationarity independently, then computes the signed
fragment components using CPU DFHelper/native JK. Both signed dispersion
pair matrices are required separately, not merely their sum. Output-file
roundtrips and nonunit scaling are also checked. These are **small-case**
diagnostics, not protein admission or any of the accepted timing repeats.

Helpers may have a newer separately recorded Git pin than the scientific
stage, but they must use its SHA-verified launcher and untouched whole
scientific Python/core inventory. They never overlay driver code or modify
the source/build being used by job14321813.
