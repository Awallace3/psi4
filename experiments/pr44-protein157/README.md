# PR44 protein157 before/after

Two **explicitly approved** gts-cs207/inferno jobs, one per exact PR44 base/head,
reusing the validated native builds and private pydantic2 runtime from the
smaller embers campaign. This harness is not part of the production PR.

- Atom/coordinate tokens copied unchanged from `~/data/timing_test_suite/geometries/protein157.xyz`
  (whitespace/comment normalized).
  User clarified protein157 after initially naming protein89. Atoms 1–118 are
  peptide A, 119–157 ligand B; both neutral singlets. No recentering/rotation.
- SAPT(PBE0)/aug-cc-pVDZ, delta-HF, hybrid FDDS, SINGLE GRAC. 2491 primary,
  9182 JK-fit, **7384 FDDS auxiliary** functions (`DF_BASIS_MP2`, not
  `DF_BASIS_SAPT`). Historical unpruned 75×302 grid made explicit.
- Same 1200 GB **decimal** Psi4 declaration (1117.587 GiB), 192 threads,
  exclusive cpu-gnr, 1400 GiB SLURM memory, 2600 GiB node-local scratch filter,
  12-hour limit. Require at least 2000 GiB actually free inside the job.
- Base uses its legacy implementation; head uses AUTO algorithm/storage,
  split monomers. Production memory safety factor remains 0.8.
- Sample process RSS, cgroup memory including page cache, and logical/allocated
  scratch every 2 s in a separate spawned process (native FDDS retains the GIL).
  Also record process high-water RSS and cgroup peak/events. Scratch peaks are
  sampled lower bounds, not file-write totals. Never label reduced disk use as
  reduced process RSS. Same resources, but independent nodes: not a speed test.
- Require unsignalled exit zero, complete finite energies, valid sampler,
  hashes/basis dimensions, no OOM, and compare components at the predeclared
  1e-8 Eh tolerance. The nanotube coupled-FDDS parity issue remains open.
- No resumable FDDS checkpoint: retain partial logs/samples on failure, do not
  automatically launch costly retries. No cleanup of shared remote data.

Before submission: commit/push/fetch via GitHub; verify clean checkout, quota,
both `sbatch --test-only` plans, and run the skill's `inferno_account.sh` for
cpu-gnr / 192 CPUs / 12 hours / count 2 immediately before actual submission.
Require gts-cs207 to cover both; do not silently switch accounts.

```sh
python experiments/pr44-protein157/submit.py \
  /storage/scratch1/3/awallace43/psi4-pr44-protein157-283ae89a32-inferno-v1 --test-only
# After cost check:
python experiments/pr44-protein157/submit.py \
  /storage/scratch1/3/awallace43/psi4-pr44-protein157-283ae89a32-inferno-v1
```

Historical (different revisions; not current PR results): protein157 scratch
1032.8 → 643.0 GiB, RSS 798.8 → 980.4 GiB. The new measurements must be
reported independently, including any RSS increase or energy mismatch.
