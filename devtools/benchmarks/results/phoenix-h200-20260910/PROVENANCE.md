# Provenance

Everything in this directory derives from the Phoenix jobs below, all run on
2026-09-10 against one build. The raw per-case trees are not committed;
`regenerate.sh` rebuilds every table here from them.

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
| A | 13060539 | gpu-h200 / embers | 1x NVIDIA H200 143771 MiB, driver 595.71.05; 8 cores of Intel Xeon Platinum 8562Y+ | Paired CPU/GPU, 6 cases x 3 repeats, alternating order. **PREEMPTED at 02:16:17** with three nanotube cases outstanding |
| A2 | 13065746 | gpu-h200 / embers | same shape as A | The three nanotube cases job A did not reach: CPU repeats 2 and 3, GPU repeat 3 |
| C | 13061073 | cpu-small / embers | 24 cores of Intel Xeon Gold 6226 | CPU-only 8 vs 24 thread scaling, same node |
| B | 13060540 | gpu-h200 / embers | 1x NVIDIA H200 | protein157 GPU |
| D2 | 13066284 | cpu-small / **inferno** | 24 cores of Intel Xeon Gold 6226 | protein157 CPU baseline |
| A3 | 13080182 | gpu-h200 / embers | same shape as A | **Canary-verified rerun of job A's full paired campaign.** Job A's protocol verbatim; the only changes are the host canary and a 3600s per-case timeout. Submitted because job A's host was degraded threefold, which bounds its speedups to a range rather than pinning them |

Jobs A and C ran the same six systems with the same driver, settings, and binary;
they differ only in hardware and thread width. **They are different nodes with
different CPUs**, which matters for how their numbers may be combined — see
`CPU_BASELINE.md`.

### Why job A3 exists

Job A's speedups are same-host ratios and are individually valid for the host it
got, but that host ran its CPU work about three times slower than another
gpu-h200 allocation of the same CPU model on the same binary. Both arms were
degraded and not by the same factor, so no arithmetic recovers the healthy-host
number from job A's tree. Job A's own figure (7.03x for benzene aug-cc-pVDZ) is
an upper bound; the cross-node comparison against job C is an estimate whose
sign is not determined, because a healthy 8562Y+ core beats job C's Gold 6226
by anywhere from 1.13x to 1.95x depending on what the phase is bound by. See
`CPU_BASELINE.md`. Job A3 re-measures the whole paired
campaign on an allocation that records its own throughput, before and after, via
`common.inc`'s `host_canary`. Run `host_speed.py` on the A3 tree first: if its
canary shows a healthy host, A3's speedups replace job A's throughout and the
range collapses to a number; if it shows another degraded host, that is itself
the finding, and A3 is resubmitted rather than averaged in.

### Jobs A and A2 are one campaign

Job A's preemption left `nanotube-6-31+g**-cpu-2` as a stub directory holding a
`psi4.out` and no `result.json`, and `-cpu-3`/`-gpu-3` unstarted. Job A2 reran
exactly those three with the same binary, settings, and node shape, writing its
own run directory so job A's provenance metadata stays untouched. `regenerate.sh`
presents the two trees as one through `merge_case_trees.py`, which symlinks
rather than copies — so every case still points at the job that produced it — and
which refuses if both trees claim a completed copy of the same case. The stub
loses to A2's completed rerun; that is the only collision it resolves silently.

Reading the nanotube row therefore means reading across two allocations. They are
the same hardware shape but not the same physical node.

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
Psi4 memory, 8 threads (job A) or 8 and 24 (job C),
`SAPT_DFT_INDUCTION_TYPE=NONE` with `SAPT_DFT_DO_DHF=True`,
`CUEST_MIXED_PRECISION=False`. The arms differ only in `USE_CUEST`.

## Observed failure mode: a hung case is not a slow case

`nanotube-6-31+g**-cpu-2` in job A stopped producing output after 40 s and stayed
alive. Its log ends with

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
