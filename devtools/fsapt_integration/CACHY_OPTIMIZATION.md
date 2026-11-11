# Localization / electrostatics candidate

This development candidate is **not GPU-qualified** and has no accepted speedup.
The protein83 Phoenix baseline remains frozen at
`b641cd4254751bfd42a461d0d0fe1c5647765b58`.
Use a separate `~/gits/psi4-fsapt-opt-cachy` root.

## Changes

- Ordered IBO Jacobi: one 128-thread block per sweep, instead of one per pair.
  Frozen pairs precede active pairs with the same ascending indices, reductions,
  angles and DROT signs. Shared storage is reused only after an unconditional
  end-of-pair barrier. `sweep_algorithm="pairwise"` retains the reference launch
  schedule. This removes launch overhead, **not** the serial pair dependency.
- Fragment electrostatics: skip exactly zero-charge point potentials. A GPU
  check authenticates the host scheduling indices against resident charges;
  negative and fractional charges remain active. `active_atoms=None` retains
  the all-atom reference. No potential batching or basis-validation caching is
  included yet.

## Required evidence

1. Host equation/contract tests, coherent sm_86 build, native CPU CPKS tests.
2. Exact raw one-sweep GPU bits (rows, rotation, gradient) between pairwise and
   fused launches; multi-sweep outputs/history at 1e-12. Include frozen/active
   blocks and atom indices above 128.
3. `run_gate.py --optimization-probe`: fresh-process small CPU/GPU/oracle
   controls, plus same-provider optimized/all-atom electrostatics comparison.
   Compare every fragment field, repeat stability, launch coverage and cleanup.
4. New shared-memory kernel: memcheck, initcheck, racecheck and synccheck.
   The prior full-runtime sanitizer attempt failed the startup loader guard
   before CUDA. That is **not** a clean sanitizer result. Do not bypass the
   authenticated loader or relax scientific-library provenance.
5. Practical peptide E10/D12 parity, independent CPU same-gauge oracle,
   before/after results, and device-memory accounting. RTX 3090 timings are
   debug-only.

No candidate Phoenix timing submission until the applicable gates pass.
Compare speedups only on Phoenix, with CPU references on CPU nodes.
