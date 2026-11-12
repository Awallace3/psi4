# PR44: Phoenix dimer validation

This experiment-only branch does not change either production revision.
It compares PR43's published head (`8f46e95cba`) with PR44 (`283ae89a32`).
The later local-only PR43 lifecycle follow-up is not silently mixed in.

Native dependencies and compilers come from `p4qca`. Python runs in a private
`--system-site-packages` venv under the campaign, with QCElemental 0.50.4,
QCEngine 0.50.0 and Pydantic 2.13.3 pinned to match the successful local build.
The shared environment remains unchanged. Initial build job 15496716 compiled
the base but failed before calculations: staged QCElemental 0.50.4 tried to
import Pydantic-2 APIs from p4qca's Pydantic 1.10.21. Preserve that failure;
it is not a numerical failure. Each harness revision gets its own immutable
manifest while existing job outputs remain under their original job IDs.

Pilot 15509332 stopped before calculation at the scratch equality assertion.
The payload now selects and binds all three scratch variables inside the srun
step, before importing Psi4, and records incoming paths for diagnosis.

## Scope

The peptide (24 atoms, 260 AOs) and ethene–nanotube (42 atoms, 574 AOs)
are PR43's timing-suite geometries, now evaluated as full SAPT(DFT) dimers,
including delta-HF, automatic single-pass GRAC, and hybrid FDDS dispersion.
Geometry numeric tokens are copied from the original timing suite.
Basis, integration grid, threads and convergence match PR43's workload class.
Auxiliary bases and GRAC settings are explicitly pinned in `config.json`.

Eight independent CPU embers calculations:

* Base, head AUTO, and head forced FDDS AO disk at 4 GiB (peptide) /
  16 GiB (nanotube).
* Head AUTO at 2 / 8 GiB to exercise tighter admission.

SLURM allocations are twice the Psi4 declaration, allowing scratch page-cache
headroom. This is **not** PR43's 1.25x cgroup experiment, a universal memory
bound, or a statistical timing benchmark. One run per point, potentially on
different CPUs. No 80% policy change is made. No protein-scale claim is made.

## Evidence and acceptance

Each case records input/config/harness/native hashes, exact source SHA, loaded
core path, environment packages, actual host/SLURM allocation, wall time,
process peak RSS, sampled cgroup memory (including page cache), sampled scratch
high-water size, full Psi4 output, component energies, and exceptions.
Sampling uses a separate process so native calls holding the GIL cannot pause
the monitor; its small memory overhead is included in cgroup readings.
Sampling is not an exact high-water bound. Scratch is local to each allocation.

Compare completed DISP/TOTAL energies with the base at absolute tolerance
1e-8 Eh. Report missing, preempted, OOM and gracefully rejected cases explicitly;
never count partial runtimes as speedups. Inspect printed storage choices to
verify the forced disk route. SLURM .batch/step status and result JSON must agree.

`job.sbatch build` builds both frozen sources in p4qca and runs the small SAPT tests.
Only submit science after its manifest and successful completion are verified;
there are no fragile afterok chains. `job.sbatch science` takes an absolute harness
directory, campaign root and case index. Scripts use background children and
TERM-safe completion markers. Resubmission creates a new job-ID output directory.

Revisit the 80% factor only after inspecting phase peaks and auditing FDDS
allocations not covered by MemoryClaim (including dense matrices and BLAS
workspace). Passing these examples alone does not justify raising it.
