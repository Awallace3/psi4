# Protein157 RSS diagnosis (experiment-only native probes)

Parent `60d7d7bc71`: frozen PR44 production head283ae89a32 plus the existing
experiment harness. Native changes only emit `PR44_MEM` phase/buffer plans,
RSS and RSS high-water marks when PR44_MEMORY_TRACE is set. No allocations,
blocking policies or scientific expressions are changed.

Hypotheses: larger budget-dependent temporary blocks; overlap with retained
AO/tensor buffers; allocator retention; or an unavoidable larger working set.
Do not infer causality from whole-run peaks alone. Match time-stamped native
phase and allocation plans to the independently spawned 2s resource sampler.

First build/regress on embers. Planned controlled comparison: same instrumented
head, same protein157 options/geometry and 192threads; Psi4 1200GB versus800GB
(decimal), keeping SLURM memory1400GiB fixed to separate algorithmic budget
effects from cgroup page-cache pressure. Explicit embers8h and no requeue.
Historical full runs took6h45–6h50. Timeout/preemption is not a scientific
failure or permission for an automatic inferno retry.

Record checkpoints of the small SCF matrices/densities at FDDS entry to make
subsequent FDDS-only experiments possible without repeating SCF. The checkpoint
does not by itself reproduce the full process allocator/cache history.
Require finite component energies within1e-8Eh against the original high-budget
head. Changes to that tolerance are not permitted to hide discrepancies.

This is not production timer CSV logic and must never be added to PR44.
