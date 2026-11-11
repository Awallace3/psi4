# Native CPKS + existing GPU F-SAPT integration

## Source boundaries

- Target baseline: `bb00386a9ae68f211a1a73d21d29fc49f90cbfdd`
  on `saptdft_cuest`.
- CPU donor: `fsapt-cpks-native-cpu`, pinned at
  `57bff3baa8`. Port the **net eight-file change** from `d7b5a01d4c`;
  do not merge its unrelated upstream history or revive the removed Python
  response framework.
- GPU donor `saptdft_cuest_disp0` at `4f0b9ee8e1` is already an ancestor
  through merge `8bbbf411cb`. Its fragment electrostatics, exchange,
  induction/response, dispersion, IAO/IBO, scaling and workflow sources are
  present. No second GPU merge is needed.
- Never update the running/queued Phoenix benchmark's checkout, installation,
  manifests or run directories. New qualification gets a separate root.

The CPU port reuses native FISAPT PCG with monomer `cphf_Hx`, fixes the
`WAbs -> wA` source loading bug, clears unmatched source columns and handles
zero sources. This also changes legacy coupled HF source loading, so test
both XC-on and legacy routes.

Branch-specific integration guards reject `USE_CUEST` for the new CPU option
at both Python driver entrypoints; reject unsupported functionals/options
before SCF/cache work; accept the driver's normalized all-None external-field
mapping. Numerical CPU tests explicitly disable cuEST and Einsums.

**GPU fragment response remains HF-only.** Native CPU HF/SVWN/PBE CPKS
admission is not evidence for GPU KS fragment response. No default changes.

## Gates, in order

1. Host-only CUDA/driver contracts and source syntax checks. These are not
   GPU evidence.
2. Private coherent cachy sm_86 build and four native numerical cases:
   HF/SVWN/PBE directional fragment sums versus canonical CPKS, plus
   HF XC-on versus legacy cell-by-cell matrices. Run `cachy_build.sh ROOT PIN`.
3. Fresh real-GPU small HF fragment gate: signed individual terms, both
   directional induction matrices, full fragment matrices, sum closure,
   scaled/unscaled routes, repeated/changed inputs, frozen/unequal spaces,
   cleanup and no host-factor/solver fallback. Reuse the existing
   `devtools/disp0/phoenix_validation/fragment_gate.py` and same-gauge CPU
   oracle, with a **new audited runtime configuration** for this installation.
   Its existing 24/8 thread policy is a Phoenix policy, not a cachy default.
   Unsupported UKS/Cartesian/link/external cases remain rejection tests.
4. On tiny admitted systems run compute-sanitizer memcheck and initcheck
   (racecheck when shared-memory kernels change). Require zero GPU skips,
   exact source/import/library provenance, and recorded peak memory.
5. Practical peptide case from the user's `~/data/timing_test_suite/`.
   Use identical frozen inputs and same-wavefunction/fixed-localization-gauge
   CPU/GPU comparisons; independent SCF/IBO gauges are not cellwise oracles.
6. Only after cachy passes: a separate Phoenix sm_90 build, architecture
   parity gate, then matched timing/scaling on embers. Inferno needs approval.

## Practical inputs and improvement priorities

Use the literal geometries in
`~/data/timing_test_suite/geometries/systems.py`; record source and geometry
SHA256 values with each run. Do not edit that suite or use historical
RHF/Cartesian timings as F-SAPT baselines.

Continue the existing campaign's **spherical jun-cc-pVDZ** primary and explicit
aug-cc-pVDZ-JKFIT / aug-cc-pVDZ-RI fitting sets, with frozen core. JK-based
fragment terms must use JKFIT, not substitute RIFIT. This is a new fragment
campaign, not an extension of canonical SAPT0 timing receipts.

| Case | Where first | Purpose |
|---|---|---|
| Tiny unequal dimer / water | cachy | CPU CPKS and signed HF GPU correctness, sanitizers |
| peptide (24 atoms) | cachy | Practical bounded integration/memory check |
| nanotube (42 atoms) | Phoenix | Asymmetric spaces, localization repeatability |
| protein83 | Phoenix | Measured native-plan and response memory admission |
| protein157 | Phoenix H200, conditional | Scale only after full peak-memory admission |

PLA15/3PE2 is excluded; it requires a separate explicit request.
No promise that protein157 fits: the existing canonical campaign estimates
109 GiB for one unscreened triangular JK tensor before fragment response
factors/workspaces. Measure rather than reduce science silently.

Before optimizing, retain the current GPU result as the baseline and instrument
native factor construction, localization, each signed term, response setup /
Hx / solve, transfers, allocations and cleanup. Investigate existing nanotube
localized-gauge repeat failures and protein83 native-plan peaks before a
performance claim. Evaluate response workspace reuse and bounded packed-factor
cache within explicit memory limits; do not choose FP64 algorithms based on
RTX3090 speed. Report cachy timings as debug-only.

GPU KS response is a separate increment: connect the existing monomer XC
Hessian machinery to fragment source solves, preserving exact-exchange
fractions, signs, per-source true residuals and source-sum closure. Start
with CPU-qualified HF/SVWN/PBE; do not enable hybrids/GRAC by implication.

## Initial state

User authorized small cachy tests. Read-only preflight found RTX3090, 24 GiB,
about 1.3 GiB device memory in use, 23 GiB available host RAM and 36 GiB free
disk. Existing source/build/install sizes suggest a separate roughly 3 GiB
root is practical; never remove another agent's files to make space.
The persistent cuEST 0.2.50 runtime exists under
`~/gits/psi4-xdm-cuest-cachy/runtime/`. Shared environments and other roots
remain read-only. Build, numerical and GPU results are pending.

Local validation: `python3 -m unittest discover -s devtools/disp0
-p 'test_cuda*.py' -v` passed **254 tests**, including the new CPU/GPU
admission boundaries. Python compilation, `bash -n cachy_build.sh` and
`git diff --check` passed. These are host-only checks.

The additional independent CPU fragment-response contracts passed (3 tests,
257 host checks total). The isolated cachy build at `a4d0339029` compiled and
installed successfully (`BUILD_OK`), but its launcher stopped before Python
tests: `--psiapi` expanded unset `PYTHONPATH` under `set -u`. Define an empty
path before activation and use fresh attempt directories on retries.
No native numerical or GPU pass is claimed from that attempt.

Retry at **`924dd8da21c132a2a3efe57a0931fdbf68dbaf3d` passed**:
`BUILD_OK`, exact installed Python/core import assertions, and all four
native CPU CPKS numerical tests (19.62 s, zero skips).
Receipt: cachy
`~/gits/psi4-fsapt-cpks-integration-cachy/runs/cpu-cpks-924dd8da21c132a2a3efe57a0931fdbf68dbaf3d-RJJl0n/junit.xml`.
The installed science stays pinned there while a separate helper checkout
prepares sealed-runtime qualification. Reuse the existing
`qualify_construction_root.py` provenance checks, now explicitly admitting
this dedicated integration root; do not relax byte/import/runtime checks.

Sealed runtime qualification at the same science pin also passed:
`runs/sealed-cpu-01/whole-build-proof.json`, native CPU RHF smoke
`-1.1167593073781577 Eh`. Runtime config SHA256:
`59fa03f193b2d8b5b1637c98b3653cc61969fbc8d585caf21b3373f6e1b54ccc`.
The independent small HF GPU gate in `runs/hf-fragment-gate-01` **failed**
before the GPU energy calculation; its CPU arm completed. The public
`energy()` keyword normalizer converted an explicitly supplied integer
zero-byte cache budget to `False`, rejected by the strict CUDA admission.
A focused regression also reproduced the analogous block-size-one bug.
Preserve both integer controls in `kwargs_lower`; keep rejecting actual
booleans, strings and floats rather than relaxing CUDA admission.
This runtime receipt alone is not a GPU pass.

The existing fragment input freezer now includes explicit `jun-cc-pvdz`
for peptide/nanotube/protein83, preserving JKFIT/RIFIT separation and
checked spherical function counts. Protein157 remains a frozen geometry
only, not admitted by that three-case fragment harness.

For subsequent science changes, `cachy_build.sh ROOT PIN PREFIX` supports
a fresh `ROOT/install-owned-*` prefix so the passing CPU baseline install
and failed GPU artifacts are preserved. Rebuild/reinstall normally; no Python
overlay. Freeze a new runtime proof for the new exact science pin.

## Current qualified increment

**`b02569493979602e418c05911430eaf1f406e47a`** passed a fresh coherent build,
all four native CPU CPKS cases again (19.78 s, zero skips), and the independent
sealed-runtime CPU smoke. Installation:
`~/gits/psi4-fsapt-cpks-integration-cachy/install-owned-b025694939`.
The previous `install/` is preserved.

The real RTX3090 small HF fragment gate now **passes**:
`runs/hf-fragment-gate-02/gate-result.json`,
SHA256 `98273240702912c81ba75c22a61279f4972cceb86700d50ec348b9d944dcbc95`.
All ten fresh CPU/GPU/same-gauge-oracle/repeat cases completed, covering
unscaled, scaled and displaced water dimers. Maximum signed aggregate CPU/GPU
difference is `1.3376e-11 Eh`; largest scaled same-gauge induction-cell
difference is `1.2113e-11 Eh`. Strict repeat and ownership/cleanup gates pass.
This is **small HF correctness**, not GPU KS response, peptide admission,
sanitizer qualification, or a speedup.

Runtime config: `runs/sealed-cpu-02/runtime-config.json`,
SHA256 `00c471d835ab97bdc820268178abe584f8bef7fde0962116b660c9745eee790d`.
Frozen practical jun-cc-pVDZ manifest: `runs/fsapt-jun-inputs-01/inputs.json`,
SHA256 `988d3b5c6b80b611c6fecc5b6292d5fde51d8f1be89b0a1bb7a017561f4e337a`.

`cachy_followup.py` reuses this exact gated installation for (a) water
memcheck/initcheck with fresh-repeat parity or (b) peptide CPU/GPU/same-gauge
oracle/repeat parity. It requires the matching small release receipt, a
nonblocking common GPU lease and <=4 CPU / 12 GiB host / no-swap scope.
Workers are capped at 20 minutes, the whole follow-up at 50 minutes.
The peptide budget follows the existing 20 GiB device / 12 GiB declared-native
admission policy and checks free VRAM plus a desktop margin before launch;
those limits are not measurements of peak consumption. No nanotube/protein
run or H200 speedup is admitted by this launcher.
