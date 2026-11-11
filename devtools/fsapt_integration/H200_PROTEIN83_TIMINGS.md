# Protein83 split CPU/H200 campaign: incomplete

**No accepted GPU timing or speedup.** The first large-system GPU worker failed
in dimer SCF before F-SAPT; the comparison job correctly rejected that producer.
Do not divide the CPU sample times by the failed GPU allocation's elapsed time.

Frozen science/helper: `b641cd4254751bfd42a461d0d0fe1c5647765b58`.
Input SHA256: `988d3b5c6b80b611c6fecc5b6292d5fde51d8f1be89b0a1bb7a017561f4e337a`.
Protein83, jun-cc-pVDZ, E10/D12, two requested samples per arm, no Python tracing.
All allocations used Phoenix **embers**, account `gts-cs207-chemx`.
CPU references used CPU-only `cpu-gnr` nodes with 48 threads; H200 host budget
was eight threads. Different CPU resource classes are intentional, not an
equal-host-resource comparison.

| Role | Job | Result |
|---|---|---|
| Build/CPU qualification | 15474904 | Passed |
| CPU sample 0 | 15474909 | Completed; **11983.806214201031 s** |
| CPU sample 1 | 15474910 | Completed; **10928.757684147917 s** |
| H200 samples | 15474911 | Water controls passed; first protein83 dimer SCF did not converge in 100 iterations; **zero accepted samples** |
| CPU comparison/oracle | 15474912 | Rejected incomplete GPU producer; no oracle or accepted ratio |

Both CPU producers report `PASS_SAMPLES_AWAITING_CROSS_ARM`, exit 0, no TERM,
and completed owned-scratch cleanup. Their samples are preserved. The final
CPU-repeat, cross-arm and same-gauge oracle checks were **not** reached.
The failed GPU job also completed owned-scratch cleanup; this was not a recorded
OOM, time limit, or preemption.

Remote campaign:

```text
/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-fsapt/protein83-two-run-b641cd4254-01/
  cpu0/job-15474909/sample-0/
  cpu1/job-15474910/sample-0/
  gpu/job-15474911/sample-0/
  compare/job-15474912/
```

Each role has `split-result.json`, `job-status.env` and bounded worker logs.

## SCF failure: energy criterion, not density criterion

At GPU iteration 100, the printed residual was **5.71616e-13**, below D12,
but the energy change was **1.57090e-8 Eh**, above E10. Iterations 79–100
already had residuals between about 5.7e-13 and 8.6e-13 while energy changes
continued to fluctuate at roughly 1e-9–1e-8 Eh (minimum in this window
4.28827e-10 Eh). This looks like an energy-evaluation noise floor, not simply
an SCF density that needs more iterations.

The trace alone does not identify the underlying cuEST integral/reduction
operation. A fixed-density repeated native J/K and energy-contraction
diagnostic is the next discriminating test; do not blindly increase MAXITER
or relax E10 to turn this failed run into an accepted benchmark.

No automatic retry, tolerance change or inferno submission has been made.
