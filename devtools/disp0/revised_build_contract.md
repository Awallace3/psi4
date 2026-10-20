# Revised CPU build contract — executable proposal, not allocation approval

## Source and scope

E2 source stays exactly `d17bb76e4ed3a07fbbc470e6813a5c134dfe0e62`;
all Psi4 scientific files under `psi4/` must equal
`587e172a33afaadb00eb7f574a5089533f7fc755`.
Only tracked `devtools/` configuration, packaging, probes and tests change.
No C++/CUDA scientific algorithm, E2 source patch, binary patch or source
overlay. No coordinator source, SDK, environment, run, docs or data changes.

The original failed jobs14182584/14193865/14196330 and failed artifact-prefix
proof remain failed and preserved. The b3/8f/9e tuple is explicitly rejected
by the new admission. This contract does not repair or reinterpret it.

## Input identities before a future submission

Two **read-only pinned inputs** are mandatory:

1. A complete CUDA13 development-tree manifest produced by
   `runtime_contract.py inspect-sdk --source ABSOLUTE_SDK --out NEW_JSON`.
   This command is documented, **not executed in offline preparation**.
   Full inventory hashing may be substantial; do not move heavy inspection
   onto a login node to meet a deadline. Supply/review a suitable existing
   immutable manifest or separately clear its bounded preparation.
2. An exact conda explicit dependency lock and SHA256, preferably the
   unchanged `environment-explicit.txt` from14196330 (GNU14.3 build20,
   CMake4.3.2, matching Apiary/Clang resource setup). A new environment is
   installed from the lock; the original environment is never cloned/edited.

Required CUDA identities: nvcc13.3.73, CUDA/CUDART header13030, compiler
support/bin/NVVM/libdevice inputs, static runtime/development archives and
all eight required public component headers and critical transitive
`crt/host_config.h`, `driver_types.h`, `vector_types.h`, `builtin_types.h`,
and `nv/target`.
The complete recursive include/bin/NVVM input trees are fingerprinted,
including symlinked subdirectories; cycles and transitive changes fail.
Mandatory libraries:

| Component | Genuine SONAME |
|---|---|
| cudart | libcudart.so.13 |
| cublas / cublasLt | libcublas.so.13 / libcublasLt.so.13 |
| cusolver / cusparse | libcusolver.so.12 / libcusparse.so.12 |
| nvJitLink | libnvJitLink.so.13 |

Component SONAME12 for CUDA13 cuSOLVER/cuSPARSE is legitimate; **site CUDA12
selection or CUDA12 nvJitLink/CUDART/cuBLAS is not**. Missing/ambiguous
components, wrong headers, old-build/site12 paths and changed hashes fail.
No PATH search or fallback download supplies missing CUDA components.

`prepare_sdk` revalidates the manifest and creates a NEW mode0700 private
locator. Compiler/header/support trees remain read-only inputs. Its own
`lib64` supplies genuine unversioned developer aliases only to their
attested real SONAMEs. It never aliases a SONAME12 request to a SONAME13
binary or changes the shared CUDA13 source.

## Configuration, packaging and installed proof

`revised_einsums_hook.cmake` is injected with
`CMAKE_PROJECT_Einsums_INCLUDE`. It seeds private component locations but
does **not** precreate `CUDA::cudart`, which would skip E2's backend setup.
Deferred validation checks every created CUDA imported library's canonical
file and SHA, rehomes only already-attested locations to private locators,
checks header roots and missing transitive targets, and requires all six
mandatory bound targets.

Deferred install rules install the actual `_core` target and the complete
generated Python package (rc.py, helpers, subpackages and stubs). Existing
native E2 install rules remain; built shared dependencies are installed too.
Intended installed E2 paths are exactly:

* `_core`: `$ORIGIN/..`, private SDK `lib64`, new dependency environment `lib`.
* native/dependency libraries: `$ORIGIN`, the same SDK and environment.

New dtags are requested and **actual ELF RUNPATH is checked afterward**.
Old-build/site12/empty/absolute-NEEDED paths fail. Complete installed package
inventory and pure-Python bytes must match the actual build package, not a
hand-selected overlay. `--psiapi` must choose the whole new Psi4 stage.

`revised_runtime_probe.py` imports Psi4 core before Python E2, verifies one
actual installed E2d17/GNU14 world, all CUDA component maps/identities and
actual loader closure. The tiny one-thread layout/view/pool/replay probe
then runs. Small physical CPU tests must execute on this new stage:
26 cases, all physical cases executed, 64 retained signed Disp20/Exch-Disp20
comparisons at the predeclared1e-8 Eh/1e-8 relative physical tolerance.
Only the inapplicable mock-CUDA request test may be skipped on a real build.

An opt-in pytest plugin writes a **same-test-process** runtime/binary/XML
proof; a parent-created historical XML or diagnostic import is insufficient.
Both API and physical process proofs audit every relevant actual native
mapping, including Psi4/dependencies and lazy imports, not just six CUDA
names or a separate `ldd` process. System-library policy handles both Debian
and RHEL `lib64`; it never permits a scientific/site library merely because
its ABI filename looks plausible.

`build_audit.py` captures the real CMake compile databases, Ninja commands,
expanded response-file commands (`compdb -x`) and actual Ninja header
dependencies. Effective compiler/include/library/link paths and CUDA-header
identities are validated before acceptance. Verbose build logs are retained.
NVCC prepend/append flags, ambient CUDA/build paths and compiler launchers are
sanitized; unexpanded response files or missing actual include/link evidence
fail closed. These are metadata audits, not diagnostic compilations.
CUDA-specific transitive basenames and namespaces are identity-bound even
when a shadow header appears beneath an otherwise allowed environment/source
root. Linker scripts/version maps and packed/relative linker-file paths are
audited as inputs, not mistaken for harmless scalar flags.
`revised_admission.py collect` binds all proofs, source equality and SDK/
install target records but reports only `PENDING_JOB_EXIT`.
After the allocation ends, `admit` additionally requires unchanged evidence,
status exit0/no signal, actual job **and batch** COMPLETED0:0 and the approved
CPU route. This is `CPU_RUNTIME_PREPARATION_ONLY`, never GPU correctness.

## Exactly one proposed CPU allocation

* Jobs: **1**, no array/retry/requeue or benchmark campaign.
* Partition candidate: **cpu-small**, subject to fresh exact-route test-only.
* Account/QoS: **gts-cs207-chemx / embers** (free, no inferno approval inferred).
* Resources: **1 node, 1 task, 8 CPUs, 48G host, no GPU, 02:00:00 maximum**.
* Worst case:16 CPU-hours,96 GiB-hours host reservation, **$0 embers charge**.
* Fresh detached source, input receipts, env/cache/build/install/run roots;
  persistent job-owned scratch. Require conservative64GiB project/quota and
 64GiB scratch headroom plus adequate inode quota before submission.
* Explicit `--no-requeue`, TERM-safe status, stop on first failure.
* No actual submission until fresh explicit CPU approval, completed exact
  recipe push/source identity, complete SDK/dependency input identities,
  exact-route test-only, live headroom and no-duplicate check.

SBATCH proposal: `revised_cpu_build.sbatch`; payload:
`revised_cpu_payload.sh`. Both require an explicit approval-reference input;
that flag is not itself proof that approval exists. Required source/input
pins are rechecked inside the allocation. No SDK inspection, dependency
installation, E2/Psi4 build or diagnostic compilation ran in this offline step.

## Shared GPU coordination supersedes independent requests

Do **not** independently submit the historical GPU gate/timing SBATCH files.
Sole owner is `saptdft-cuest-disp` (FDDS), planning one H200/8CPU/128GiB/8h
interactive allocation under its own approved account/ceiling. No grant/
UUID/window is established by this recipe. Disp0 has no allocation and no
immediate READY need; never hold the owner's GPU idle for this CPU work.

The old `interactive-01` controller failed before allocation and is immutable/
superseded. Current owner-announced controller is
`P/gits/psi4.cuest-shared-interactive-20261020-v2` (short pin9a026ca149);
future manifest is
`P/runs/psi4-cuest-shared/20261020-interactive-02/allocation.json`,
lock `gpu-step.lock` beside it. Wait for the owner’s actual allocated manifest,
full job/device/source provenance and assigned window; use **only** its step
controller, never direct srun or lock bypass. The common lock spans the whole
exclusive step and cleanup; missing proof quarantines the allocation.

After actual runtime READY, request only a fitting roughly30min scalar gate
(minimum4CPU/32G). Final benchmarks/CPU controls must be exclusive, with own
isolated immutable stage/output, thread/science/hardware/co-tenancy provenance.
512GiB protein runs do **not** fit128GiB and must not launch. No separate
CPU-baseline campaign, GPU duplicate or account switch is inferred. Any work
that cannot fit requires separately explicit approval. No08:00 speedup
promise; Phase2 remains blocked.
