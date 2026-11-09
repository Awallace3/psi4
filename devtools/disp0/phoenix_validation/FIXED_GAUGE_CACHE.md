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
