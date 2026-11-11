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
