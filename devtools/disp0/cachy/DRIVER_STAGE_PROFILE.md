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

## Completed: water and peptide passed

The controller finished at **2026-11-03T11:57:40Z**, recording
`PASS_SMALL_RTX3090_DRIVER_DIAGNOSTICS_NOT_H200_SPEEDUP`.
All seven workers returned0 with owned child groups reaped. Both independent
CPU oracles checked all23 fields, and the separate profiled peptide GPU call
passed the existing strict fresh-repeat check.

The monitor reported exit3 because the completed transient systemd unit was
garbage-collected, not because a scientific worker failed. Completion was
subsequently established from the final controller checkpoint, worker status/
hash receipts and completed log. No retry was launched.
The service journal records56min30.331s elapsed and an8.5G host-cgroup
memory peak (human-rounded journal units), below the12GiB cap. The scientific
checkout remains clean at`0433ec0452`, and `BUILT_FROM` still identifies that
same exact commit.

### Whole-call observations

| HF-reference F-SAPT/aug-cc-pVDZ | CPU4 energy (s) | RTX3090 +4 host threads energy (s) | CPU4 inclusive (s) | GPU inclusive (s) |
|---|---:|---:|---:|---:|
| Water | 3.227362 | 26.886316 | 15.472824 | 40.046503 |
| Peptide | 1538.003742 | 114.928005 | 1546.180725 | 128.923484 |

These are one baseline process per arm, **not medians or a qualified new
speedup**. Cachy's CPU is an AMD Ryzen7 2700X Eight-Core Processor, using four
threads under the explicit four-CPU quota. Do not compare its denominator with
the24-core Gold6226 Phoenix reference, or predict H200 bottlenecks from the
3090's restricted FP64 throughput.

### Peptide driver-step wall times

| `sapt_proc.py` window | CPU4 (s) | RTX3090 +4 host threads (s) |
|---|---:|---:|
| Dimer SCF | 18.072 | 13.601 |
| Monomer A SCF | 9.101 | 2.936 |
| Monomer B SCF | 9.295 | 2.857 |
| SAPT JK/cache | 2.349 | 0.996 |
| Canonical electrostatics | 0.001 | 0.002 |
| Canonical exchange | 2.256 | 0.286 |
| Canonical induction/correction | 29.473 | 3.435 |
| Full fragment continuation | 1467.051 | 90.681 |
| Dispersion dispatch/metadata, not fragment dispersion | 0.071 | 0.067 |

The CPU fragment sum comprises localization0.219, partition0.001,
fragment localization0.085, electrostatics4.182, exchange2.039,
exact source-resolved induction1449.482 and dispersion11.043s.
The GPU fragment timer contains all those terms together and must not be
called a dispersion-only interval. Its nested blocked-dispersion diagnostic
is19.548715s; that is not the same boundary as the complete CPU `fdisp0` call,
so no matched dispersion-only ratio is claimed.

### Separate profiled peptide worker

The profiled energy call takes143.624762s versus114.928005s in the unprofiled
baseline; profiling materially changes elapsed time. Observed inclusive
windows include:

| Window | Calls | Profiled inclusive (s) |
|---|---:|---:|
| Source-resolved induction `solve_source` | 88 | 42.936578 |
| Response `apply` | 1361 | 41.891806 |
| Native identity `_audit_maps` | 1650 | 23.554415 |
| Sealed `mapped_rows` | 1767 | 16.775188 |
| Native plan `build` | 2 | 21.088295 |
| `blocked_device_dispersion` | 1 | 21.091242 |
| `localize_iao_device` | 2 | 0.206007 |

These windows are nested/overlapping, include native waiting and host audits,
and have incomplete profiler coverage of the enclosing workflow. They must
not be summed into a synthetic disjoint GPU-stage breakdown or relabeled as
kernel timings. They identify useful next inspection targets on Cachy only.

### Numerical, work and cleanup checks

- Water: largest same-gauge field-entry difference3.48067e-12; whole-energy
  GPU-minus-CPU difference-3.57351e-12Eh.
- Peptide: largest same-gauge field-entry difference1.25069e-11; whole-energy
  difference-1.27876e-9Eh between independent SCF calculations.
- Separate profiled peptide fresh-repeat largest field error8.77555e-11,
  below the unchanged1e-9 repeat tolerances.
- All seven result/matrix/standard-output/timer hash receipts were rechecked.
- Peptide tracked explicit device peak4,477,018,672 bytes,196 dispersion
  compute blocks,145 RIFIT calls and64MiB cache ceiling match the H200 work
  ledger. This tracked peak does not measure opaque native allocations.
- Native/session cleanup completed with zero live tracked bytes and native
  owners. After completion the desktop again reported1661MiB used,
  22517MiB free and0% GPU utilization; this is a point sample, not peak telemetry.

| Artifact under `runs/stage-profile-20261103-01/` | SHA256 |
|---|---|
| `profile-result.json` | `327b6c406177a09fc205820c043178198c0eb835a5e31af301b828a6a5e55f37` |
| `water-cpu/result.json` | `156e444796db94d89ee2252b8cb0911828f84a603480d5b84bb7f9c8c4cfe58f` |
| `water-gpu/result.json` | `695bdce9c298d92c877d95aac10080929c2a6705c3194cbbcbb2ee42b8d758f8` |
| `peptide-cpu/result.json` | `3ce77b462635b82bbff4cfcb75af992b42297ff51ddc7dc3286e36a44fcb1822` |
| `peptide-gpu/result.json` | `31c18bda45e52de9ec6b914c6698cea63f483a01a7738f560642bd4babbed566` |
| `peptide-cpu/timer.dat` | `549a4ab9a7dbe682c0aa2a46228aa993b511fd6d891b6499941b2407ac40df36` |
| `peptide-gpu/timer.dat` | `d891c4c5df54a94370a1fd96ff3113cdc8c7fe8d43c1488f992ff9c007c38b47` |
| `peptide-gpu-profiled/profile.pstats` | `80523d4eedd53dc71b543b3ec9b3e1af7a6f67ab0babef2640649bd64e2dffef` |

## Separate synchronized fragment-term follow-up

Two **GPU-only** follow-ups completed with
`PASS_RTX3090_FOUR_TERM_DIAGNOSTICS`. No CPU timing or oracle was rerun:
the existing strict repeat checker compared all23 fields with the independently
CPU-qualified GPU baselines above, at unchanged atol/rtol1e-9.

The observer traces source boundaries in the authenticated, unchanged installed
`fragment_terms` function and synchronizes its existing owned session before
closing each adjacent window. It does not replace scientific functions or edit
the installed stage. The four windows are disjoint; DISP includes rotation
preparation, `prepare_hf` and blocked dispersion. Preparation/localization,
plans, scaling, downloads and cleanup outside these boundaries remain overhead.

| HF-reference F-SAPT/aug-cc-pVDZ fragment window | Water RTX3090 (s) | Peptide RTX3090 (s) | Peptide fragment share |
|---|---:|---:|---:|
| Electrostatics | 0.706218 | 7.203907 | 6.409% |
| Exchange | 0.039286 | 0.388051 | 0.345% |
| Exact coupled induction | 1.827227 | 50.917317 | 45.300% |
| Dispersion, including preparation | 8.374780 | 20.741801 | 18.453% |
| Workflow overhead residual | 6.634464 | 17.336926 | 15.424% |
| Outer factory/cleanup residual | 6.817025 | 15.812998 | 14.068% |
| Full traced fragment | 24.399000 | 112.401000 | 100% |

All six parts close to the recorded full fragment in each run. These are
**instrumented wall times, not GPU kernel clocks or matched per-term speedups**.
The peptide fragment was112.401s with this observation, versus90.681s in the
earlier unprofiled process. Do not rescale the traced parts into that baseline,
splice them into the H200 result, or infer H200 bottleneck shares.
Water's overhead residuals account for55.1% of its traced fragment.

Both workers returned0 and reaped owned groups. Largest strict-repeat field
errors were1.0559192409331786e-12 for water and1.0249645576720923e-10 for peptide
(the latter is a localization field, not an energy-unit claim).
The frozen scientific checkout remains clean at0433ec0452. After completion
the GPU point sample was1667MiB used,22511MiB free,0% utilization.

- Run: `runs/fragment-term-profile-20261103-01/`
- Helper checkout: `profile-helper-51b5f0ad41`
- Helper pin: `51b5f0ad41b6d8e5facdbe44186051010a629c07`
- Original per-case config hashes unchanged.
- Four host threads;400% CPU quota;12GiB host cap;zero swap;128 tasks.
- Per-worker600s;controller1200s;service hard cap1300s.
- Workflow SHA256:
  `8d0c630ac343c5bc0e4d8aa0fe22e9294c9c66168bb4c17da33be17fb15e23c1`.
- Observed source markers:95elst,97exch,99ind,103disp,112after-terms.
- Local observer/parser/admission tests:12passed; independent inspect-only
  safety review found no concrete blocker before launch.

The remote final artifacts were hash-rechecked:

| Artifact under the follow-up run | SHA256 |
|---|---|
| `term-profile-result.json` | `8c48b17012ef2c02f47297980581f3044c25ef760332db8f39f20c15de4ca6bb` |
| `water-term-windows.json` | `a0c4a386672527e826e7c9c6257113d5f62c0999261350709d2f8bab6af0e2ef` |
| `peptide-term-windows.json` | `65c7e17d06e5ccb7983c55c039bde550e3a1593923bffbaac15c50e25d345a91` |
| `water-gpu/result.json` | `b65ce1792fc3b04e65a138b39eb8a3226ccd0fb10496584d3f294f6dd3031ced` |
| `peptide-gpu/result.json` | `c58e131cdadb3a2ce45bd77998c99b715bce445ea08a0182f2da9507cfd41c1d` |
| `water-gpu/timer.dat` | `3a02a9e1679f104d160ed911b6674eaec4cde46d17d4e3cdca6b161ea0e64e7c` |
| `peptide-gpu/timer.dat` | `9bcca689151e9fd249aaec1a0b9f8fdae7750093c6cb9c8f5ef616ca5917d1cc` |
