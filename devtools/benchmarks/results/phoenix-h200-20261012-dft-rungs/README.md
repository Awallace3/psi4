# Lower-rung DFT CPU/cuEST controls

Status: prepared, not yet submitted or measured.

## Scope

Extend the existing water/benzene/peptide meta-GGA counterpoise comparison with:

| Rung | Functional | Expected semilocal family | Exact exchange |
|---|---|---|---|
| 1 | SVWN | LDA | none |
| 2 | BLYP | GGA | none |
| 4 | B3LYP | GGA | 20% |
| 4 | PBE0 | GGA | 25% |
| 4 | ωB97X | GGA | range separated; intrinsic parameters retained |

These are **not** ωB97X-D or ωB97X-V. No added D3/D4, GRAC or SAPT.
No nonlocal VV10 is expected for these five controls.

Both historical staged host-LibXC and CUDA-LibXC builds are retained, with
each compared against its own native CPU reference. Routes: CPU, cuEST J/K
with native XC, and full cuEST J/K + XC. Every AB/A(ghost B)/B(ghost A)
calculation uses a fresh process.

## Protocol and acceptance

Reuse the existing geometry/basis/DF/FP64 protocol:
- water/benzene: aug-cc-pVDZ; peptide: spherical 6-31+G**;
- def2-universal-jkfit auxiliary basis; CPU SAD; eight CPU threads;
- Psi4 memory 32 GiB; 99×590 ROBUST XC grid;
- E=1e-11, D=1e-10, maxiter=200, fail on nonconvergence;
- `IE = E_AB − E_A(ghost B) − E_B(ghost A)`;
- accept **|GPU−CPU IE| ≤ 1e-5 Eh**, retain the 1e-6 diagnostic flag;
- retain individual total-energy differences to expose cancellation;
- report complete-IE speedup from the sum of three `psi4.energy` wall times,
  excluding imports/startup/queue time.

The worker timeout is **900 s**, increased from the previous 300 s so
near-converged controls are not unnecessarily truncated. This changes the
process time budget, not scientific convergence thresholds.

## Submission plan

Use free **embers** only; no inferno approval applies to this campaign.
First run all five controls on water as a staged-build/functional preflight:
**90 workers, 20 CPU/GPU comparisons** across both builds.
Verify all definitions, component accounting, routes, provenance and complete
records before submitting benzene and peptide. Planned total:
**270 workers and 60 CPU/GPU comparisons**, of which 30 are full-GPU rows.

One H200, eight CPUs, 64 GiB allocation memory per job.
Water: two-hour allocation; benzene/peptide: four-hour allocations.
All fit embers' eight-hour limit. This does not guarantee successful SCF or
completion before preemption; partial/failing records remain explicit.
No dependencies that can become stranded are submitted.

The existing TERM-safe `mgga_ie_job.sh` and archive replay are reused.
Pin the experiment checkout and stage inputs; final unchanged-provenance
verification is required. Existing historical build-source mismatch and
CPU/cuEST quadrature differences remain limitations, not new functional
failures or a demonstrated density-fitting attribution.

## Completion

Replace pending coverage in the measured rung/system table only after an
independent worker replay, raw timing audit, functional-metadata and
energy-component checks, effective-QoS and final-provenance verification.
Keep unsuccessful cases visible. Extend the marked section of cuEST PR #12
without modifying unrelated text.
