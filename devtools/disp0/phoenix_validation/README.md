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
