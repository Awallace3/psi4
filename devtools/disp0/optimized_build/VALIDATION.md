# Preparation evidence, 2026-10-21

These are local fixtures/metadata/configure-only tests, not a build or scientific
validation. No actual job ID is recorded here.

- Initial 13 mock/metadata guards: PASS, 1.203s.
- Additional real CMake **LANGUAGES NONE** configure/generate fixtures:
  7 PASS, 0.637s. Fake non-ELF library bytes; compile commands deliberately
  `cmake -E false`; no compiler detection or compilation.
- Initial combined input rerun failed **before collection** because system
  `/usr/bin/python3` lacks pytest (`b9495b323`). No package installed.
- Re-executed using existing pytest-capable
  `/home/awallace43/miniconda3/envs/p4_einv2/bin/python`:
  `-B -m pytest --noconftest -q tests/pytests/test_disp0_sdk_inputs.py`:
  **38 PASS, 0.27s**, task `b4b50307c`. No Psi4/Einsums native import.
- Final one-shot/environment guards plus configure fixtures:
  `python3 -B -m unittest discover -s devtools/disp0/optimized_build -p 'test_*.py' -v`:
  **21 PASS, 2.694s**.
- `bash -n` on both `build.sbatch` and `payload.sh`: PASS.
- `git diff --stat 587e172a33afaadb00eb7f574a5089533f7fc755 HEAD -- psi4/`:
  empty; tracked scientific source unchanged.
- `git diff --check`: PASS.

Read-only reviews:

- `d90202093f49ccd3d0ceb24a0d5cbf008`: exact064 native Python install replaces
  d17 packaging workaround; required Waggle companion identified; clean GitInfo
  and explicit installed Waggle import/maps checks added.
- `d3bf5e8c1a9afbedd1f3f3afcd7552950`: P1 inherited `SBATCH_*` override fixed
  by sanitized environment plus complete CLI resource shape on both scheduler
  calls, regression added. P2 automatic ccache selection fixed by
  `EINSUMS_WITH_CCACHE=OFF`, assertion added.
- `d8335ebad2a994f7f44cb28fbb2ad77c5`: follow-up confirms those resource,
  ccache, Waggle/RUNPATH and clean-GitInfo fixes; **no scoped blocker found**.
  Read-only source review, not an executed build/test.

Execution admission remains fresh-source/route/quota/headroom/duplicates;
full SDK/input identity checks occur in the new CPU job before configuration.
Old input attempt UNKNOWN/RESERVED and all failed roots remain untouched.
Actual optimized CPU build/API and signed physical/GPU checks remain unpassed.

## Corrected decoder/loader boundary and interactive protocol

Fresh user `yes` approves one corrected same-resource build; subsequent user
instruction selects one interactiveCPU allocation rather than batch retries.
Old14249189 and old UNKNOWN remain unchanged.

- Red regression:1536MiB virtual mapping under inherited1GiB AS fails
  `OSError: [Errno12] Cannot allocate memory`,0.231s. No physical pages touched.
- Before final IPC test addition:33 local guards PASS1.537s, including
  corrected mapping/separate budget and interactive acquisition mocks.
-39 archive/input guards PASS0.28s, including new negative asserting neither
  SDK manifest nor receipt is written when mandatory closure fails.
- Both payload and interactive-step Bash syntax checks PASS.
- Actual CUDA-positive loader trace, native build, APIs, physical/device tests
  remain UNRUN; local virtual-map tests do not stand in for any of them.
- Added actual local fork/IPC inheritance probe: decoder hardAS1GiB and
  separately started worker actual hardAS8GiB observed; rejects non-ldd
  request and captures raw response, no library loaded. Combined **34 PASS,
  4.070s**. This is the final pre-review local run, not Phoenix readiness.

SDK review `d01c2d55c` found one evidence blocker: timeout/oversize exception
handling lost partial loader output. Fixed with bounded captured stdout/stderr,
explicit timeout/truncation/byte counts, fail-closed status and worker-level
fork regressions that verify durable raw-file/ack equality. **36 guards PASS
1.669s**, **39 SDK guards PASS0.29s**. Narrow fix confirmation pending.

Independent follow-ups `d5c46a79b` and `dd7ba9065` exhausted artificial
turn limits and produced **no committed answer**; neither is counted as a pass.
Known SDK blocker is addressed and worker-level tests pass. Freezing current
code after direct inspection; advisory `df014bf6e` reviews the frozen SDK and
interactive-allocator code while independent Git staging/preflight proceeds.

Final advisory `df014bf6e`: no remaining SDK evidence/budget blocker; one
allocator blocker (missing requested TRES and GPU resources only in AllocTRES).
Fixed by mandatory parseable requested+allocated CPU/node/memory proof and GPU
rejection in both, with negative regressions and unit-normalized48GiB positive.
**38 combined guards PASS2.012s**. Narrow TRES-fix confirmation pending.
No resource shape/science change; new03 source rather than adopting staged02.

Live14250580 proved Phoenix's interactive default is Requeue1. The strict
guard correctly refused GO but the lease was unused before accounting
PREEMPTED0:0 at09:46:30Eastern. Do not infer build success from exit0.
The later old-ID flag correction made no update because its initial controller
query failed. Evidence and source03 remain immutable.

Prepared explicit opt-in correction for a separately approved NEW own ID,
with own UID/job-name checks, single durable update and unchanged-resource/
time verification. New negative/positive mocks cover missing approval,
already-zero/no update, exactly one authorized update and wrong owner/name.
**42 local guards PASS2.524s**. No replacement allocation requested by tests.

Reviewd132 caught missing start/end timestamps being treated as equalNone.
Fixed mandatory full ISO timestamps, valid calendar/positive interval, direct
post-update key comparison; missing/malformed/backwards and changed start/end
negative regressions. **45 guards PASS4.772s**. Scoped confirmation pending.
Parent explicitly confirmed fresh replacement authority already valid:
one8CPU/48GiB/2hfreeembers/chemx, bundled new-own-ID flag correction and
immediate build without another approval question.

## Live build step0: compilation/install completed, audit parser failed

14254131.0 exited1/nonsignalled after2596.56s. Both Einsums native/Python
and whole Psi4 stage reached installation. The subsequent metadata audit
rejected literal Ninja-escaped `\$ORIGIN/../`; no installed proof/API ran.
The failed step/result/partial audit remain unchanged.

Exact local regression reproduced that rejection before fix. Anchored literal
ORIGIN recognition now accepts only exact escaped/unescaped forms; unknown
variables/lookalikes and foreign CUDA paths still reject. An optional fresh
external audit directory prevents writes into the preserved build run.
**48 local guards PASS5.275s**, **96 legacy runtime guards PASS4.60s**,
validation continuation Bash syntax PASS. No binary/linker/scientific-source
change, configure/build/install or RPATH patch is involved.

`validate_install.sh` is a new same-allocation validation continuation. It
reads the completed immutable step0 install, writes ONLY new validation/scratch
outputs, audits effective metadata, runs installed-world/API checks and
requires the selected binary tuple unchanged before/after. Original step0
is not relabeled successful. GPU/physical correctness remains unpassed.

Validation step1 stopped after52.88s on an unexpanded `@` token. Actual
captured Ninja JSON identifies only Apiary's Python stub-generator fragment
list, not a compiler response. Exact local auxiliary-rule regression went
red, then passed after separating shell command segments/actual compiler
positions. Actual compiler/link responses, foreign compilers/includes/cwd,
cache wrappers and unidentified compiler wrappers still reject; `.cc` source
arguments are no longer mistaken for compiler executables. Auxiliary rules
remain captured raw, not falsely labeled semantically audited.
**52 guards PASS3.624s**, **96 legacy runtime guards PASS4.87s**.
