# Protein157 memory-floor bracketing

## Current round: explicitly approved inferno pair

The three first-round embers jobs were preempted, not OOM-killed:
full400GB sampled383.56GiB RSS; replay200GB228.37GiB; replay100GB133.81GiB.
None completed. User explicitly approved **two full runs at100GB and85GB**,
gts-cs207/inferno,192threads,1400GiB Slurm/2600GiB scratch,16h limits.
Account quote$128.84available and$23.13/job worst-case; recheck before sbatch.
Use a **new output campaign**, `--approved-inferno`, and verify effective QoS.
Command-line sbatch account/QoS/time override the script's safe embers defaults.
This does not authorize further paid retries or the later cgroup-limit tests.

85GB is just above the source projection admission gate83.3963GB at192threads
and auxiliary maxshell7. This is a declaration threshold, not an RSS bound.
Full checkpoints/results are required; strict numerical parity stays separate.

## Historical first round (harness b7e391df51)

User asks how low peak RSS can go before OOM. The previous800GB full calculation
completed at685.72GiB RSS but failed strict1e-8Eh parity. Neither that setting
nor Slurm1400GiB was a demonstrated minimum.

This first bounded round:
- Full SAPT(PBE0), Psi4 **400GB decimal**, otherwise original conditions.
- FDDS-only replays at **200GB** and **100GB**, using the exact matrices,
  densities and orbital energies checkpointed in completed job15547644.
- Same validated native diagnostic build70e01e2baf/hash83a4dc1b..., no rebuild
  or production changes. New harness commit recorded independently.
-192threads, exclusivecpu-gnr,1400GiB Slurm,2600GiB scratch,embers8h.
  Hold the Slurm cap fixed initially to measure algorithmic RSS floors rather
  than combine blocking changes with file-cache pressure/OOM.
- FDDS-only results **exclude** SCF, MP2/exchange-dispersion and inherited
  allocator/cache state. They are not whole-job memory minima. Replays compare
  FDDS energies to the same-input800GB checkpoint; full runs compare all five
  components to original1200GB head. Do not relax1e-8Eh.
- Native probes and2s samples record in/outcore selection, buffers, RSS,
  page-cache-inclusive cgroup memory, scratch and OOM events.

After this round, bisect surviving/failing budgets with a small number of
jobs, confirm the candidate in a full run, then test a lower **Slurm limit**
with explicit page-cache margin. Classify OOM, admission rejection, timeout,
preemption, scratch exhaustion, and parity failure separately. Do not call
a timeout/preemption the memory floor; do not call a finite calculation that
fails the energy tolerance a fully validated working point.

Below roughly300GB the FDDS AO store may switch from RAM to disk, and SCF
can change storage earlier. This may increase wall time/scratch; keep no
automatic retries or inferno changes. Seek fresh approval if8h no longer fits.
The source contains per-thread memory admission gates, so the floor is for
**192threads and these options**, not an absolute method-wide minimum.

Historical source of inputs:
`/storage/scratch1/3/awallace43/psi4-pr44-protein157-memory-v1/results/800/15547644`.
Immutable config/manifest and submission IDs are saved in each new campaign.
