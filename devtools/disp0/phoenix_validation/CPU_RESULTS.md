# Production CPU F-SAPT comparisons

## Scope

HF SAPT(DFT), full fragment F-SAPT, spherical aug-cc-pVDZ with augmented
JKFIT/RI, frozen core, delta-HF, exchange/induction scalar scaling enabled,
exact per-source fragment response disabled. Both arms share the canonical
Einsums-v2 driver; only fragment continuation dispatch differs.

Scientific stage: `1aed320cdb8b031c281b4642001a750aa51c549f`, whole native
build14321813. The Einsums-v2 speedup history through `1592851675` and the
optimized blocked C++ dispersion implementation are included.
Each system uses one exclusive 24-thread Gold6226 allocation, alternating
backend order, three fresh processes per arm. Water preflights are excluded.
Timings below are the **whole energy call**, not subprocess or kernel time.

## Qualified measurements

Seconds; median [minimum, maximum], n=3 per backend:

| System | Job | C++ | Einsums-v2 | v2/C++ time |
|---|---|---|---|---|
| peptide | 14324190 | 21.293 [21.135, 21.540] | 22.032 [22.009, 22.899] | 1.035 |
| nanotube | 14326813 | 148.126 [146.541, 148.748] | 156.061 [153.973, 156.988] | 1.054 |
| protein83 | 14324700 | Not yet collected | Not yet collected | — |

Thus v2 is about **3.5% slower on peptide and 5.4% slower on nanotube** in
these full-call measurements. This is not a historical pre/post-v2 benchmark:
both measured implementations already contain their respective optimizations.
No confidence intervals or protein extrapolations are inferred from three runs.

Peptide's six samples were also requalified offline under the new component
ledger. Timed-engine/resource/shared-runtime fingerprints match the original
helper; measured times and original artifacts were not altered.

Nanotube job14326813 completed0:0 (job/batch/step), elapsed16m27s,
no TERM. Its allocation admission, both water preflights, all six numerical
ledgers, three paired full-matrix comparisons and same-arm repeats passed.
Occupied mappings were identity; maximum paired matrix discrepancy:
**6.82989220734953e-11 Eh**. Retained MO rank is740 for742AOs, with virtual
counts732/662. Both implementations show the same fitted residual, approximately
0.6413–0.6425 microEh in the fragment sum versus the JK canonical total;
no matrix correction was applied. Canonical algebra residual is at most
1.57e-17Eh. Both signed canonical dispersion scalars are checked, but separate
signed native dispersion pair matrices are not exposed by this CPU harness.

## Nanotube exclusive phases

Median seconds, n=3 per arm; repeated SCF calls are summed within each sample
before taking its median. Fragment totals are also summed per sample first.
Medians of separate phases need not add to the whole-call median.

| Phase | C++ arm | Einsums-v2 arm |
|---|---:|---:|
| Common SCFs | 61.030 | 61.440 |
| Common JK cache | 2.712 | 2.702 |
| Common electrostatics | 0.012 | 0.012 |
| Common exchange | 2.529 | 2.528 |
| Common canonical induction | 35.153 | 35.377 |
| Fragment electrostatics | 12.603 | 14.342 |
| Fragment exchange | 3.958 | 4.083 |
| Fragment induction | 7.377 | 8.818 |
| Fragment dispersion | 21.445 | 25.457 |
| All instrumented fragment phases, including localization/setup | 46.065 | 53.598 |
| Whole-call time outside instrumented phases | 0.532 | 0.340 |

The measured extra time is primarily in the fragment continuation, particularly
dispersion; this is a phase observation, not a diagnosis of its implementation.

Maximum child RSS across repeats: C++26.045GiB, v2 25.993GiB. This is process
`ru_maxrss`, not a cgroup/page-cache peak. Scratch logical size sampled every
2seconds peaked at7.791GiB in each arm, not an exact disk high-water mark.
Actual node: `atl1-1-03-003-30-1.pace.gatech.edu`;24CPU affinity admitted;
local scratch XFS had1.451TB free before calls.

## Failures and missing results

- Original nanotube14324682 remains FAILED: first v2 calculation completed
  science but the cancellation-biased net-total validator rejected it.
  No full matrices were saved before that guard, so its155.904s observation
  is **not** an accepted timing sample. The corrected retry above used fresh
  processes and a fresh output root, not retrospective promotion.
- Protein83 job14324700 retains the original helper and existing watcher.
  Its results have not yet been collected here; no automatic retry authorized.
- H200 correctness gate14322613 failed on a600s GPU-child timeout.
  No accepted GPU timings or speedups exist. Its exact-response fragment mode
  differs from this production-scaled CPU mode: these CPU times must not be
  used as its speedup denominator without a qualified matching GPU method.

## Provenance and reproduction

Remote prefix: `/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/`

- Peptide: `production-cpu-005f3b4c71/peptide/job-14324190/`.
- Nanotube: `production-cpu-ledger-abd4c1bc46/nanotube/job-14326813/`.
- Protein83: `production-cpu-005f3b4c71/protein83/job-14324700/`.
- Input SHA256: `f8c96058e9387ab14179ce263a2d4896dbe9cc2e3bf1286654784825ff178e38`.
- Dependency-config SHA256: `4d255b795caae75a9df879b81b26fd84b10a79d0400f2831b5dbcdebad4afa6c`.
- Corrected helper: `abd4c1bc468ce7f759988e3af2f8c985ff6c48bf`.
- Nanotube raw collection: background task `b7b791acf`; offline peptide
  requalification: `b34e990d7`.

Each run retains `comparison-result.json`, `allocation.json`,
`job-status.env`, per-child `result.json`/`matrices.npz`, stdout/stderr and
paired-repeat proofs. Parent manifests bind helper/science/input/config and
resource identities. Recompute whole-call medians/ranges from the six accepted
child `whole_energy_seconds` records. Group each child's exclusive `phases`
by name and sum before taking medians; sum backend-prefixed phases for its
fragment total. Retain all failed points separately.
