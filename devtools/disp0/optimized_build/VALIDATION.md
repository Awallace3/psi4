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
