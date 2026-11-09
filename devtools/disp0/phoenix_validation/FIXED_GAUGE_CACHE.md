# Protein83 fixed-gauge cache diagnostic

## Why this experiment

Native fitting-policy correction`e0da305c02` passed six physical JK tests
and the small H200 gate, but Protein83/SVP producer15337409 still failed
strict cross-process energy repetition. That failure remains visible;
neither its413s samples nor this experiment qualify a whole-system speedup.

The first saved result contains a separate measured work bottleneck:

- Device-input blocked dispersion:224.041s.
- Native producer cumulative wall:219.040s; its nested completion waits
  account for211.221s. These are inclusive host timers, **not kernel clocks**.
- R slabs29, S slabs30, contraction blocks870.
- With64MiB cap, only the short S tail fits:8,522,336bytes,1slab,28hits.
  Native batches:29R+842S=871.
- All packed S slabs require
  `8*(2*2488+2)*698*(58+49)=2,974,295,264bytes` (2.770GiB).
  A4GiB cap predicts29R+30S=59native batches, still870contractions.

This predicts eliminated work, **not runtime proportional to call count**.
No automatic cache-budget change or production default is introduced.

## Fixed numerical inputs

The probe consumes SHA-verified`raw-hf.npz` and`fragment-matrices.npz`
from15337409/r0, plus the frozen Protein83/def2-SVP input manifest. It
uses captured HF matrices/energies and the active blocks of the captured
IBO rotations. The rotations are **diagnostic input only**, not a new
production path or an independent localization oracle.

The original installed`prepare_hf` creates device spaces/corrections.
The original installed`blocked_device_dispersion` runs four times in one
owned RIFIT provider/session:64MiB,4GiB,4GiB,64MiB. No source overlay,
factor download/re-upload, SCF, response, full F-SAPT or CPU fallback runs.
Changing only cache residency preserves the R-major contraction order.

Every call must:

1. Match the exact predicted native/cache/contraction counts.
2. Return live device bytes to their pre-call value and respect the
   original120GiB device limit, including12GiB native reservation.
3. Avoid host numerical uploads inside the blocked path.
4. Match every signed scalar and pair matrix against the first call at
   rtol1e-12/atol1e-12. No SCF-repeat tolerance changes.

All four outputs are saved in`dispersion-pairs.npz`, sealed in the worker
receipt, and replayed by the controller. Normal provider/session cleanup
and owned-process reaping remain mandatory. Source/build admission and the
sealed runtime are unchanged; only the outer diagnostic launcher is newer.

## Limits and launch

This isolates the blocked call, including required native ownership checks,
packing, integration, contractions, reductions and synchronization. It is
**not** a vendor-kernel-only speedup, a full F-SAPT timing or a CPU/GPU
comparison. The standalone RIFIT context also omits full-workflow response
and JKFIT residency: its measured peak cannot establish their coexistence.
Any promising result needs a separate full-workflow memory/numerical gate.

Use the existing`scf_repeat_probe.py --mode cache-profile --kind gpu`
with the exact candidate config, original r0 result/hash and a fresh root.
The existing embers H200/8CPU/160GiB/30min scheduler path is reused.
Success is`PASS_FIXED_GAUGE_CACHE_DIAGNOSTIC`; accepted speedup count
remains0. No source/native rebuild is needed because scientific functions
and their existing explicit cache-cap interface are unchanged.

## Submitted H200 diagnostic15357943

All158 host helper tests passed; independent read-only review found no
concrete launch blockers. These are not physical cache execution evidence.

Frozen outer pin:`d87d61188ea3d5869f07d8b08108fd773e4480d1`,
`/storage/project/r-cs207-0/awallace43/gits/psi4.disp0-cache-probe-d87d61188e`.
Science/build remains`e0da305c02`, unchanged from the completed physical
six-test native-JK and small F-SAPT gates.

Submitted **15357943**; effective read-back route:
`gts-cs207-chemx / gpu-h200 / embers`,1H200,8CPUs,160GiB,30min,
initially PENDING for Priority. Test-only15357942 is not the allocation.
Root:
`/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/fixed-gauge-cache-d87d61188e-01`.
Manifest SHA256:
`00f96f57921d344a4d019dc8c374ed97a7ae218fb58eb159216708129ee90f17`.
Runtime-config SHA256:
`dcfada35840335785602df7a92e8b1f67761b5c3dae6dea5690a60559a850760`.
Official hourly watcher:`b217404f7`.

The separate candidate fixed-density reconstruction control15356916 is
still a different experiment. No full Protein83 CPU reference/oracle,
accepted ratio, native-repeat waiver or production cache-default change
is implied by this submission.

## Completed15357943: blocked dispersion11.861× faster

The H200 allocation completed in8min48s; job/batch/step exited0 without
TERM. The worker returned0, reported no retirement and was reaped; owned
scratch cleanup and provider/session cleanup completed. The controller
replayed all four sealed pair/scalar outputs. Status:
`PASS_FIXED_GAUGE_CACHE_DIAGNOSTIC`, accepted full-speedup count0.

Actual device: NVIDIA H200/sm90, driver615.71.09, cuEST0.2.50 with native
FP64 and the unchanged relative-MATRIXPOWER/cutoff1e-12 RIFIT policy.

| Cap | Call order | Blocked wall/s | Native producer wall/s | Native batches |
|---|---|---:|---:|---:|
| 64MiB | A1 | 224.11257319001015 | 218.63574419985525 | 871 |
| 4GiB | B1 | 18.994964473997243 | 14.785218328353949 | 59 |
| 4GiB | B2 | 18.984791721915826 | 14.783711718511768 | 59 |
| 64MiB | A2 | 226.3733985470608 | 220.24149632884655 | 871 |

Two-sample medians:225.24298586853547s versus18.989878097956534s,
**11.861212836999384× for the blocked call**. This includes native
production, packing, contractions, reductions and required audits, not
vendor-kernel-only timing. Two samples per cap are not a general scaling study.

All870contraction blocks remain. All30S slabs fit, versus one tail:
cache hits840versus28. Maximum absolute pair-matrix difference from A1
is8.131516293641283e-20Eh; maximum signed-scalar difference is
8.673617379884035e-19Eh, comfortably inside the unchanged1e-12 checks.
No CPU numerical upload occurred inside the blocked calls.

Actual packed-cache storage grew8,522,336→2,974,295,264bytes.
The first small-cache call peaked at12.1313GiB explicit device storage;
the **session-lifetime** peak after the large-cache calls was14.8934GiB,
plus the unchanged12GiB native reservation. Live bytes returned to the
same12,389,943,696-byte prepared-state baseline after every call.
Later A2's peak counter is cumulative, not a separate A2 footprint.

Seals beneath`fixed-gauge-cache-d87d61188e-01/job-15357943/`:

| Artifact | SHA256 |
|---|---|
| `diagnostic-result.json` | `e817677fefbfbb1964bef9fa10e0def14a34f3863a6c98ede024f0c27b35b749` |
| `r0/probe-result.json` | `c910f4aa99ba1bb9cf0ee86aef4444b55fe0b34c4a16d14fede2a71c44e5e7be` |
| `r0/dispersion-pairs.npz` | `2996c90ef34510b9a4bd3d608bf3fb14b94e2ad0ffc88de7802ba67a1830be63` |

**Not yet proved:** coexistence of the larger cache with the full
JKFIT/RIFIT/response workflow, end-to-end energy speedup, independent
protein CPU oracle or native SCF repeatability. The frozen full benchmark
helper also intentionally caps its cache argument at1GiB; this diagnostic
does not bypass that helper or silently change its production policy.
The next integration gate should execute the original full fragment
workflow on fixed saved HF inputs, prove actual memory coexistence and
compare every original output field before any new whole-energy claim.

## Next: full fixed-HF coexistence gate

The outer-only`cache-coexistence` mode calls the original installed
`fragment_terms` once with the4GiB cap, both owned auxiliary plans and the
complete exact-response workflow. It holds raw HF matrices/energies and
canonical scaling inputs fixed, but **recomputes IAO/IBO localization**;
saved localized orbitals are comparator output only, not production input.
No SCF or whole energy call runs.

It compares all23original fields with the captured result using the unchanged
strict`compare_arrays` criteria: label/sign permutation only, no energy
rotations or relaxed tolerance. Original dispersion scalar checks, actual
870contractions/59RIFIT calls/5JKFIT calls, provider/session cleanup and
the existing declared device budget remain mandatory. The parent replays
all field comparisons from sealed NPZ files.

The existing source-boundary observer measures localization, felst, fexch,
find and fdisp with synchronization at boundaries, retaining other-work
residuals. These are **intrusively instrumented fixed-HF fragment windows**,
not unprofiled energy baselines; no whole-workflow ratio is calculated by
subtracting overhead or splicing them into earlier runs. A successful physical
run establishes actual coexistence on the allocated H200 and reports explicit
tracked peak plus the unchanged opaque native reservation, not a measured
opaque-native peak or a general memory model.

This does not change the frozen whole-energy helper's1GiB argument cap,
qualify Protein83 SCF repeatability, change the production cache default,
or introduce any accepted CPU/GPU speedup.

### Full coexistence gate submitted15366266

All161 host helper tests and7 source-boundary observer tests passed.
Independent read-only launch review found no concrete API, ownership,
provenance, numerical or trace blockers. Physical integration remains pending.

Frozen outer pin:`d8ba38d32eca184648adc2bc507148bb2d453bbb`,
`/storage/project/r-cs207-0/awallace43/gits/psi4.disp0-cache-full-d8ba38d32e`.
Science/build remains`e0da305c02`. Submitted **15366266** on the read-back
route`gts-cs207-chemx / gpu-h200 / embers`,1H200,8CPUs,160GiB,30min,
initially PENDING for Priority. Test-only15366265 is not the allocation.
The first scheduler read-back timed out; the idempotent`phoenix2` fallback
confirmed the same job and route. No duplicate submission or job change.

Root:
`/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/fixed-hf-cache-full-d8ba38d32e-01`.
Manifest SHA256:
`507cc44c438ea23b56f0212520356badefdeba7220a0ba6ed83bc661fb7cc803`.
Runtime-config SHA256:
`d3288c346d8d8f2a75884ec2e0b27a398d5778ffa22cec260f0a98fd36b02635`.
Official hourly watcher:`ba7c081fb`.

The served`~/docs/saptdft/cuest/fsapt.html#protein-cache` now separates
the11.86× fixed-input dispersion result from qualified peptide CPU/H200
speedups and the still-failed whole-protein repeat checks. The new chart's
six hover marks, even-sample medians, basis/NBF labels and responsive
dark/light/mobile/ToC behavior passed browser checks without console errors.
The existing full report regression also passed, including original driver
selectors, exact16-stage schema, rainbows and preserved numerical ratios.
