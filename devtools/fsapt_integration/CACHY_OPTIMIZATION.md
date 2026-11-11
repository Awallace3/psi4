# Localization / electrostatics candidate

This development candidate has **small-system GPU numerical qualification only**,
not sanitizer clearance or an accepted speedup.
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

## Build concurrency diagnosis

Initial private-root builds at outer `-j4`, then `-j2`, reached the enforced
12-GiB cgroup limit and were OOM-killed. The generated Ninja rule launches
`cmake --build .` inside `psi4-core` without forwarding the outer job limit.
`test_cachy_build.py` reproduces that missing inheritance with a real, tiny
Ninja ExternalProject. The build helper now exports
`CMAKE_BUILD_PARALLEL_LEVEL=$FSAPT_BUILD_JOBS` (default 4, allowed 1–4) in
addition to the outer flag. Retry with `FSAPT_BUILD_JOBS=2`; do not raise the
desktop memory limit as a workaround.

## Small-system numerical result

Science/helper pin: `ba9251b785043df16dfa60acf64a7ddcbfca0ea9`.
Private prefix: `~/gits/psi4-fsapt-opt-cachy/install-owned-ba9251b785`.
Run: `runs/hf-fragment-gate-03` in that root.

- Coherent build/import and four native CPU CPKS tests passed, zero skips.
- Ten fresh CPU/GPU/oracle/repeat cases passed, including changed geometry and
  scaled/unscaled terms; independent same-gauge comparisons cover 23 fields.
- Four physical localization fixtures, each checked over three raw sweeps:
  rows, rotations and gradient bits match the pairwise launch path exactly.
  Four-iteration final outputs/history also have zero difference. Fixtures
  include overlapping pairs at 257 atom/minimal indices for both powers.
- Nine charge-mask acceptance/rejection cases passed in each GPU worker.
  Expected rejections retire their own sessions; no invalidated session reuse.
- Water point-potential calls: **12 → 6**. Same-provider optimized/all-atom
  maximum matrix difference: **8.881784197001252e-16 Eh**; maximum total
  difference across the four workers: **1.7763568394002505e-15 Eh**.
- Repeat maximum field difference: **3.053487671045296e-11** (`IndAB_AB`),
  within the unchanged 1e-9 comparison policy.

Receipts:

| File | SHA256 |
|---|---|
| `gate-result.json` | `ca95afb3a2950842622677296bf3cc7ed9d728ef0d6c3257aa03ac2a0031df0f` |
| `optimization-launch-result.json` | `20559e826486083a5882dbc59802cbfdb833aa99ee601ae9bddc5cff0d2efd25` |

Practical peptide E10/D12 attempt `runs/peptide-e10-d12-01` **failed** its strict
fresh-process repeat check after all four workers completed. CPU/GPU energy,
signed aggregate and independent same-gauge oracle checks passed. One
`IndBA_AB` element (index `[3,33]`) differed by **1.25201928e-9 Eh**:
`0.022914703320116348` versus `0.022914702068097068`, exceeding the unchanged
`atol=rtol=1e-9` policy. Do not relabel this attempt as passed or drop that
element. Saved SCF/fragment diagnostics were inspected without another GPU run.

### Peptide failure attribution (saved artifacts, no rerun)

The scaled-induction difference is dominated by the upstream delta-HF scaling
factor, not a large difference in the unscaled fragment response:

| Quantity | First GPU process | Repeat |
|---|---:|---:|
| Dimer HF energy / Eh | -494.07407982275714 | -494.07407982246701 |
| Induction scale | 1.5540798133217653 | 1.5540797285499057 |
| Delta-HF scalar / Eh | -0.001663096740895785 | -0.0016630964864527854 |

After matching only orbital labels/signs with the existing policy, use the
identity `A-B = (fA-fB)*(B/fB) + fA*(A/fA-B/fB)` for diagnosis. At the failing
`IndBA_AB[3,33]`, the factor contribution is **1.2499499666904604e-9 Eh**,
leaving **2.0693132276769496e-12 Eh** residual. Maximum unscaled `IndBA_AB`
difference is **1.3315355762433256e-12 Eh**. This decomposition does **not**
change the production output, comparison threshold, or FAILED verdict.

Both peptide workers actually used 11 fused sweeps per monomer and 115
ordered pairs per sweep: **22 sweep launches instead of 2530 pair launches**.
Point-potential calls were **48 → 24**. Same-provider optimized/all-atom
electrostatics agreed within **1.7763568394002505e-15 Eh** per matrix element
in the first worker. These are work-count/numerical results, not speedups.

Full-runtime sanitizers remain blocked by the previously identified
injection/startup-loader mismatch. These results do not establish an H200
speedup or a full memory-peak comparison.
