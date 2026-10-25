# Full-system H200 continuation

The user requested the full job after the successful small H200 gate14377643.
This is a new bounded submission, not an automatic extension/retry of that gate.

## Declared run

- One H200,8CPUs,160GiB SLURM memory,8h maximum, explicit inferno.
  Live account selection before submission; the verified current H200 rate
  gives a worst-case charge of$11.31. Eight hours is a cap, not a measured ETA.
- Sequential peptide, nanotube, protein83 from input manifest
  `f8c96058e9387ab14179ce263a2d4896dbe9cc2e3bf1286654784825ff178e38`.
  Spherical aug-cc-pVDZ and the frozen augmented JKFIT/RI basis files.
- Three fresh CPU/GPU pairs per system, and a fresh independently reconstructed
  native CPU same-gauge oracle for **every** GPU result:27 processes total.
  Each GPU process additionally runs the small physical BLAS/IBO primitives.
- Reuse the complete immutable scientific stage from build14377089,
  source pin
  `d471cd93fa1c7d81f58209771c33280c2fe2e9c0`).
  Helpers are committed/staged separately; no installed scientific overlay.
- New configuration changes only the declared resource policies:
  8threads,96GiB Psi4 host memory,120GiB session device limit,12GiB opaque-native
  reserve. These are limits/reserves, **not measured peaks or memory admission**.
  Existing exact simultaneous allocation/workspace/localization/response
  guards remain enabled. Actual device capacity is verified through NVML;
  it is not inferred from the requested SLURM host memory.
- Dispersion block24 (instead of water diagnostic block5); unchanged FP64,
  exact per-source coupled response, frozen core, exchange/delta-HF scaling
  and matched1e-16 integral screening. No substitution of G-shell auxiliaries.
- Native retained rank is checked against the independently computed CPU
  orbital space, including the full retained-subspace overlap/projector,
  not assumed equal to the AO dimension. The nanotube has
 742AOs but its earlier native calculation retained740MOs.

## Qualification, not a production speedup claim

`run_full.py` uses the existing full-fragment diagnostic continuation with
all23fields, signed occupied-pair reductions, standard disk outputs, native
CPU one-electron/DF-JK/IAO reconstruction, and no CPU fragment fallback in the
GPU arm. Same-arm repeat checks permit only validated occupied labels/signs,
not rotations of energy entries; expected dimensions come from frozen input
metadata rather than the result under test.

`full_observe.py` records synchronized CUDA free/total/tracked allocation
snapshots plus NVML memory and queried native owner workspace subtotals while
JKFIT/RIFIT plans are live. Upstream cuESTJK remains live at fragment entry;
that lifetime is explicitly recorded, not mistaken for memory covered by
the session reserve. Observations are stage samples, not peak-VRAM measurements.
No native plan creation order/lifetime or public scientific algorithm is changed.

Whole-energy diagnostic times and subprocess elapsed times are retained.
Raw-HF/matrix capture, probes and oracle work make this a **diagnostic**
batch, not accepted first-use production timing. `accepted_timing_count`
and `accepted_speedup_count` stay0. In particular, the earlier production
C++/v2 CPU times use an approximate scalar-rescaled fragment partition and
24threads; they are not denominators for this exact-response8-thread GPU run.

Only successful signed full matrices/independent oracle/repeats can admit
a large point. A failure or deadline stops the allocation; remaining cases
are explicitly incomplete. Partial checkpoints and all failure outputs are
retained. No automatic retry, silent CPU fallback, reduced basis, increased
tolerance or removed memory/work guard is permitted.

## Artifacts and launch

`submit_full.py` seals the original small config/release hashes, derived
configuration, frozen inputs, exact helper pin and resource command in a
fresh run root. It requires a clean helper checkout and runs exact-route
`sbatch --test-only` before one `sbatch --parsable` call. An ambiguous scheduler
response is never automatically retried.

`full_job.sbatch`/`full_payload.sh` retain TERM-safe job-status markers.
Allocation-local `scontrol` validates account, inferno, H200 partition/GRES,
8CPUs,160GiB and8h before science. The existing runtime then binds actual
H200UUID/compute90, rejects foreign compute contexts, freshly authenticates
the mapped library closure and verifies complete native/session cleanup.

`full-result.json` is atomically checkpointed before/after each fresh process
and completed comparison. Each `SYSTEM-rN-ARM` directory contains its own
`result.json`, hashed raw HF, complete fragment matrices and standard output
files. Job IDs, effective routing and final accounting must be recorded after
submission; this document alone is not proof that anything was queued.
