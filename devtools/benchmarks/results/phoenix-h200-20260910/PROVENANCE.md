# Provenance

Everything in this directory derives from the Phoenix jobs below, run on
2026-09-10 and 2026-09-11 against one build. The raw per-case trees are not
committed; `regenerate.sh` rebuilds every table here from them.

## Build

| | |
|---|---|
| Source commit | `e971d2957b` ("Include iterative GRAC shifts in benchmark timings"), branch `saptdft_cuest` |
| Staged extension | `build_saptdft_cuest/stage/lib/psi4/core.cpython-313-x86_64-linux-gnu.so` |
| sha256 | `bc7b9620cd41f69e0d90ac83939a7ce9570c291cebdc574da2e771bae1e4227c` |
| Psi4 | `1.12a1.dev636` |
| libcuEST | 0.2.1.2 (conda-forge) |
| Einsums / pyeinsums | 1.1.2 |
| MKL | 2025.3.0; numpy 2.4.2; libblas 3.11.0 |
| DFT-D4 | 3.7.0 |
| Python | 3.13.11 |
| Driver script | `saptdft_cuest_grac.py`, sha256 `1ae3257474559fe658f94b4e4a2c9d55fd281b554bfc19ec8b25cc4d09f3663d` |
| Geometries | `saptdft_suite_geometries.json`, sha256 `17640f5394055437a1286546d83224614acee6237deb7ed2329f0f14546e45cd` |

Each job verified the commit and the extension's sha256 before running anything,
and verified that the staged binary contains the GRAC host-functional symbol.
Every case asserts `grac_compute == "ITERATIVE"` in its own `result.json`.

## Jobs

| Job | SLURM ID | Partition / QoS | Hardware | Purpose |
|---|---|---|---|---|
| A | 13060539 | gpu-h200 / embers | 1x NVIDIA H200 143771 MiB, driver 595.71.05; 8 cores of Intel Xeon Platinum 8562Y+ | Paired CPU/GPU, 6 cases x 3 repeats, alternating order. **PREEMPTED at 02:16:17** with three nanotube cases outstanding, and later shown to have run on a threefold-degraded host. **RETIRED** — superseded by A3, kept only as evidence of the degradation |
| A2 | 13065746 | gpu-h200 / embers | same shape as A, **canary-verified healthy** | The three nanotube cases job A did not reach: CPU repeats 2 and 3, GPU repeat 3. The first healthy same-host measurement in this campaign, and now the independent check on A3. Not merged — see below |
| C | 13061073 | cpu-small / embers | 24 cores of Intel Xeon Gold 6226 | CPU-only 8 vs 24 thread scaling, same node |
| B | 13060540 | gpu-h200 / embers | 1x NVIDIA H200; 8 cores of Platinum 8562Y+, **canary-verified** | protein157: 3 GPU repeats then one 8-thread CPU repeat in the same allocation. GPU repeats landed (486.24, 486.57, 488.21 s). **PREEMPTED twice and now terminal** — requeued 2026-09-11 07:58:43 onto the same node and the same eight cores, preempted again at 09:59:08 after 2:00:25, 52 iterations into the GRAC monomer A cation SCF. No same-host CPU result |
| D2 | 13066284 | cpu-small / **inferno** | 24 cores of Intel Xeon Gold 6226 | protein157 CPU baseline. **COMPLETED 02:20:03**, 8365.25 s |
| A3 | 13080182 | gpu-h200 / embers | same shape as A, **canary-verified healthy** | Rerun of job A's full paired campaign, 6 cases x 3 repeats. **COMPLETED 00:39:28**, exit 0:0, 36/36 cases rc=0. This is the paired table |

Jobs A3 and C ran the same six systems with the same driver, settings, and
binary; they differ only in hardware and thread width. **They are different nodes
with different CPUs**, which matters for how their numbers may be combined — see
`CPU_BASELINE.md`.

### Why job A3 exists, and what it found

Job A's speedups were same-host ratios and were individually valid for the host
it got, but that host ran its CPU work about three times slower than another
gpu-h200 allocation of the same CPU model on the same binary. Both arms were
degraded and not by the same factor, so no arithmetic recovered the healthy-host
number from job A's tree: its figures were an upper bound, and the cross-node
comparison against job C was an estimate whose sign was not determined. Job A3
re-measured the whole paired campaign on an allocation that records its own
throughput, before and after, via `common.inc`'s `host_canary`.

The canary reads healthy at both ends — 83.3 then 84.2 GF/s per core, scalar
47.6 Miter/s both ways, 2800 MHz — against 84.15 GF/s from the solo gpu-h200
probe, so the host is within 1% of a node doing nothing else. All 36 cases
returned rc=0 with `grac_compute == "ITERATIVE"` asserted in their own
`result.json`, 3 repeats everywhere, repeat spread 0.3-4%.

So the planned replacement happened: **A3's speedups replace job A's
throughout**, and every range in the report collapses to a measurement. Job A's
numbers are retired, not corrected, and must not be quoted; the direct A/A3
comparison puts its CPU-side deficit at 2.89-3.19x, inside the 2.97-3.47x that
had been inferred from a completely different job pair. A3 also confirms A2
independently: 7.56x against 7.58x on nanotube, from separate jobs on separate
days with separate canaries.

### Why neither job A nor job A2 is merged into the paired table

Job A's preemption left `nanotube-6-31+g**-cpu-2` as a stub directory holding a
`psi4.out` and no `result.json`, and `-cpu-3`/`-gpu-3` unstarted. Job A2 reran
exactly those three with the same binary, settings, and node shape, writing its
own run directory. The plan was to present the two trees as one campaign through
`merge_case_trees.py`.

That plan did not survive A2's canary, which reads **healthy**. A2's three cases
are repeats 2 and 3 of a case job A measured at repeat 1 on a host later shown
to be degraded threefold, so merging would place both hosts inside one row and
take a median across them:

| `nanotube-6-31+g**` | job A (degraded) | job A2 (healthy) |
|---|---|---|
| cpu | 1078.82 s | 349.31 s, 356.15 s |
| gpu | 113.19 s, 113.45 s | 46.51 s |

A median of those is a number neither machine produced, and `merge_case_trees.py`
would have refused it — that guard is exactly for this. A2's repeat indices are 2
and 3 besides, which the summarizer cannot place on their own. So `regenerate.sh`
builds the paired table from **job A3 alone**: one host, certified healthy before
and after, all six cases at all three repeats. A and A2 stay in RAW and in
`host-speed.md`, because the comparison between them is what established the
deficit in the first place, and A2 is reported in `CPU_BASELINE.md` as the
independent check on A3's headline number.

A2's CPU repeat reproduces job A's returned energy to 1e-15 Eh and its GPU
repeat to 1e-12 Eh, with identical GRAC shifts, so the two trees differ in host
speed and in nothing else.

### Why D2 is inferno and the rest is not

The protein157 CPU baseline was attempted twice on embers and preempted both
times: 13061074 at 01:06:40, and 13063342 at 01:15:19. The second had converged
both GRAC shifts (monomer A 0.04531118, monomer B 0.05039562) in about 62 minutes
and entered the 1786-function dimer SCF before it was killed. `psi4.energy()` has
no interior checkpoint, so the run cannot be chunked into 8-hour pieces and a
preemption at hour four costs all four hours. Inferno was requested and
explicitly approved for this one job. Everything else here is embers.

## Settings, identical in both arms

`SAPT_DFT_GRAC_COMPUTE=ITERATIVE` for every case, with no exceptions: both GRAC
shifts are determined by neutral and doublet-cation SCFs inside the timed
`energy()` call. PBE0, DF SCF, 99/590 grid, SCF convergence 1e-9/1e-8, 112 GiB
Psi4 memory, 8 threads (jobs A/A2/A3/B) or 8 and 24 (job C),
`SAPT_DFT_INDUCTION_TYPE=NONE` with `SAPT_DFT_DO_DHF=True`,
`CUEST_MIXED_PRECISION=False`. The arms differ only in `USE_CUEST`.

## Observed failure mode: a hung case is not a slow case

`nanotube-6-31+g**-cpu-2` in job A stopped producing output after 40 s and stayed
alive — it did not recur in A3, whose three nanotube CPU repeats all finished
near 349 s, so this is intermittent rather than case-specific. Its log ends with

```
PID: 2116552 on atl1-1-02-012-23-0 ready for attaching debugger. Once attached set i = 1 and continue
```

That string is in `libEinsums.so.1.1.2`, whose `--einsums:no-attach-debugger`
option is documented as "do not provide mechanism to attach debugger on detected
errors". Einsums installs signal handlers by default and parks the process on a
detected error instead of aborting, so an intermittent fault presents as an
indefinite hang with a zero exit path, no traceback, and no partial result. Only
the per-case `timeout` recovered it, at the cost of the full 7200 s budget. The
same case succeeded on repeat 1 and repeat 3 with the same binary and input.

Anyone extending this campaign should either pass `--einsums:no-attach-debugger`
or have the launcher kill a case as soon as that marker appears in its log. Do
not raise the per-case timeout to accommodate it: that converts a fast failure
into a slow one.

## Observed failure mode: protein157 in aug-cc-pVDZ does not fit an A100

Jobs 12904893 and 12904894 (gpu-a100, an earlier bench tree at
`bench/saptdft_cuest`, not this campaign) both died within 25 s:

```
Failed to allocate device buffer: out of memory
  requested 127.215 GiB; device has 78.631 GiB free of 79.251 GiB total
```

This is deterministic, not a transient or a contention artifact — the request
exceeds the card. protein157 in aug-cc-pVDZ is nbf 2491 / naux 9182
(`results/probe/probe.protein157.aug-cc-pvdz.json`), and 80 GB is simply the
wrong card for it.

It does not threaten job B (13060540), which runs protein157 in **6-31+G\*\***
(nbf 1786) on an **H200** (143771 MiB). The aux basis is `def2-universal-jkfit`
and depends on the atoms, not the orbital basis, so naux is unchanged between
the two and the buffer ratio is driven by nbf alone: a `naux·nbf²` allocation
scales to ~65 GiB and a `naux·nbf·nocc` one to ~91 GiB. Both fit, the second
without much room. The exact allocation model was not derived — neither candidate
reproduces 127.215 GiB exactly, so these are bounds on the scaling rather than a
prediction — but both point the same way and the margin is real.

The practical rule: protein157 needs an H200, and the basis is what decides
whether it fits at all.
