# Water and peptide driver-stage diagnostics on Cachy

User approved running smaller cases while Phoenix SSH is unavailable.
These are **RTX 3090 desktop diagnostics, not H200 speedup measurements**.
The hourly Phoenix-access monitor remains separate.

## Selected inputs and timing scope

| Case | Atoms | Spherical aug-cc-pVDZ NBF | Work |
|---|---:|---:|---|
| Water control | 6 | 82 | Fresh CPU, GPU and independent CPU oracle |
| Timing-suite peptide | 24 | 356 | Fresh CPU, GPU and independent CPU oracle; separate profiled GPU repeat |

Both use scaled HF-reference full F-SAPT/aug-cc-pVDZ and exact per-source
coupled response. The peptide uses virtual block size24 and the H200 case's
64MiB GPU factor-cache ceiling. CPU factor caching remains disabled.
No fA.dat/fB.dat post-processing assignments are required.

Nanotube and both protein systems are outside this bounded desktop diagnostic;
Protein157 remains explicitly excluded by user instruction. No larger case is
automatically attempted after the smaller runs finish.

Each baseline worker retains and hashes its actual post-exit `timer.dat`.
The parser reads only one nonempty flat timer epoch, uses wall rather than
user CPU columns, and excludes the duplicate nested tree. Missing timers remain
null; parent/child intervals are not added together.

The separate cProfile worker retains `profile.pstats` and inclusive SAPT
function windows. Those intervals overlap, include host/identity/native-call
costs, have profiling overhead, and do not establish individual GPU kernel times.
Oracle and profiled workers are not independent baseline timing samples.
One fresh pair is not a statistically qualified new speedup.

## Frozen installation and outer launcher

The existing private Cachy installation and authenticated launch helpers stay
unchanged at scientific pin:
`0433ec0452679cfc2ec2f7226ae941c0fc6719ef`.

- Root: `/home/amwalla3/gits/psi4-disp0-perf-cachy`
- Stage: `install-owned-sealed-0433ec0452`
- Reference config: `runs/sealed-owned-cpu-04/runtime-config.json`
- Reference config SHA256:
  `4034046cb74140a7ade016e8cd290d61fa316f405cccccee1c238bd8f8d6b187`
- Separate outer-launcher checkout: `profile-helper-b852eb9f1c`
- Outer-launcher pin: `b852eb9f1c461bfc77dae39738fab7792fff84d1`
- Suite module SHA256:
  `72668390d24f134316c8e779931a2f65efcd9289ecea8f1cf20a72733262e3bb`

The H200 result used scientific pin`927283f8ed`. The recorded tree difference
from this Cachy-gated prefix consists of cold locator binding and sealed
retirement diagnostics (`sapt_cuda_runtime.py`, `psi4_runtime_support/sealed.py`),
not the F-SAPT arithmetic. The experiment records both pins and the differing
paths; this is the same peptide/method/work partition, **not an identical
whole-build CPU/GPU comparison with the H200**.

No rebuild, shared-environment modification, scientific-module overlay, native
library substitution, weakened identity check or advanced frozen source is used.
The new outer controller invokes the original proof-authenticated bootstrap,
admission, environment and owned-child helpers from the unchanged source.

## Desktop resource policy

- One worker at a time, four host threads, 400% CPU quota.
- Cgroup host cap12GiB, zero swap and128 tasks.
- Per-worker deadline30min, controller deadline90min, service hard cap95min.
- Common nonblocking GPU lock inherited by owned children; HOLD is honored.
- Fixed RTX3090 UUID and fresh sealed aliases/cache HOME/scratch per worker.
- Water: Psi4 host workspace2GiB, device budget1GiB, native allowance512MiB.
- Peptide: Psi4 host workspace6GiB, device budget20GiB, native allowance12GiB.
- Before each GPU worker require the budget plus512MiB free VRAM and a
  utilization sample no greater than10%. All native workspace and actual
  free-memory allocation guards remain active; rejection is not retried.

At initial inspection the desktop reported24576MiB total,22517MiB free,
1661MiB used and0% GPU utilization. These are point samples, not exclusive
ownership or a future peak-memory guarantee. Other desktop processes and
their memory are never modified.

## Started 2026-11-03

Service: `psi4-cachy-stage-profile-20261103-01.service`.
Output root:
`/home/amwalla3/gits/psi4-disp0-perf-cachy/runs/stage-profile-20261103-01`.
Log: `/home/amwalla3/gits/psi4-disp0-perf-cachy/stage-profile-20261103-01.log`.
Completion monitor: `b7c49b167`.

The controller checkpoints `profile-result.json`, preserves per-worker logs,
raw matrices, output files and timers, and requires the existing signed
component/energy checks and independent CPU oracle. The separate profiled
peptide call must additionally pass the existing strict fresh-repeat check.
At submission only the service/resource readback was verified; no new timing
or correctness outcome is claimed until the actual workers complete.

Local stdlib-only parser/selection/desktop-admission tests: **6 passed**.

## First checkpoint: water completed, peptide CPU active

The fresh water CPU/GPU pair and independent fixed-GPU-gauge CPU oracle all
returned0 and passed the existing signed component/energy/matrix checks.
Both baseline workers retained complete single-epoch `timer.dat` files.
These are single **RTX3090 debug-only** observations, not accepted speedups.

| HF-reference F-SAPT/aug-cc-pVDZ window | CPU4 (s) | RTX3090 +4 host threads (s) |
|---|---:|---:|
| Whole energy call | 3.227362 | 26.886316 |
| Dimer SCF | 0.450 | 8.646 |
| Monomer A SCF | 0.335 | 0.411 |
| Monomer B SCF | 0.322 | 0.382 |
| SAPT JK/cache | 0.016 | 0.042 |
| Canonical exchange | 0.014 | 0.006 |
| Canonical induction/correction | 0.208 | 0.070 |
| Full fragment continuation | 1.809 | 17.261 |

The CPU fragment aggregate sums the seven disjoint CPU fragment timers; GPU
uses the single enclosing GPU F-SAPT timer. Timers are rounded to milliseconds,
whereas whole-call seconds come from the existing higher-resolution worker
clock. Canonical electrostatics rounds to0.000s in both workers, not missing
execution; no ratio is calculated from that rounded zero.

This tiny control shows substantial GPU dimer-SCF and fragment-continuation
overhead on the desktop. It does not rank H200 bottlenecks or predict the
peptide outcome. At this checkpoint the peptide CPU reference has started;
its GPU worker, oracle and separate profile are not yet complete.
