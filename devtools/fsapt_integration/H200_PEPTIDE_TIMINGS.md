# H200 peptide F-SAPT timings — job 15472290

## Provenance and status

- Phoenix `gts-cs207-chemx / gpu-h200 / embers`, 2026-11-11.
- Allocation completed in 37:01; batch and step exit codes zero.
- Result: `PASS_NUMERICAL_TIMING_SANITIZER_OUTSTANDING`.
- Science `08984a56e6dee1e052f06187b97bbdced4d2ae12`, native build
  job `15471744`; production sources identical to `b025694939`.
- Timing helper `ade1014462f907a4ea9451ceaefe6af08476a307`.
- Peptide from `~/data/timing_test_suite`, spherical jun-cc-pVDZ, 250 AOs,
  HF F-SAPT including delta-HF, E10/D12, block 24, factor cache zero.
- Eight CPU threads in both arms on the same H200 host. Same Lustre scratch;
  these are practical workflow wall times, not storage-independent compute.
- Frozen input SHA256:
  `988d3b5c6b80b611c6fecc5b6292d5fde51d8f1be89b0a1bb7a017561f4e337a`.
- Result SHA256:
  `da6f8250cd5bcba163b84223e41c6860b205a99bd8baeab1fdf245f080d50836`.

Remote artifact directory:

```
/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-fsapt/peptide-steps-ade1014462-01/job-15472290
```

Local retrieved result: `.pi/fsapt-h200-15472290/timing-result.json`.

## Whole-workflow baseline

Three fresh processes per arm, no source tracer. Normal qualification hooks,
native audits, JIT and provider cleanup inside the energy-call timer remain
charged; runtime admission and post-call diagnostic serialization are excluded.
Speedup is median CPU time divided by median GPU time.

| Arm | Samples (s) | Median (s) |
|---|---|---:|
| CPU | 239.3123, 240.6095, 240.9939 | 240.6095 |
| H200 | 32.9505, 45.0599, 32.7909 | 32.9505 |

**Baseline speedup: 7.30×.** The slower GPU repetition is retained, not discarded.

## Separately instrumented step/term comparisons

Three additional fresh processes per arm. Full-precision Psi4 timers plus
synchronized GPU source-block wall windows; not kernel timings. These samples
must not be mixed with the baseline or used to reconstruct its breakdown.

| Step / term | CPU median (s) | H200 median (s) | CPU/H200 |
|---|---:|---:|---:|
| Dimer SCF | 4.8273 | 5.2131 | 0.93× |
| Monomer A SCF | 2.1440 | 0.9973 | 2.15× |
| Monomer B SCF | 2.1559 | 0.9828 | 2.19× |
| Canonical JK preparation | 0.2382 | 0.0773 | 3.08× |
| Canonical electrostatics | 0.001028 | 0.001236 | 0.83× |
| Canonical exchange | 0.2167 | 0.0194 | 11.19× |
| Canonical induction | 3.5962 | 0.2791 | 12.89× |
| Delta-HF assembly only | 0.00001572 | 0.00001313 | negligible |
| Fragment localization | 0.1857 | 0.8427 | 0.22× |
| Fragment electrostatics | 1.5859 | 6.0613 | 0.26× |
| Fragment exchange | 0.5324 | 0.0273 | 19.47× |
| Fragment induction | 221.0165 | 9.8195 | 22.51× |
| Fragment dispersion | 2.9583 | 2.5215 | 1.17× |
| Instrumented total | 239.7632 | 52.1050 | 4.60× |

GPU fragment overhead outside the five source-block windows: **25.0530 s**
median. Remaining residual: CPU **0.2695 s**, GPU **0.2064 s**. No ratio is
assigned to these nonmatching residual scopes. Each individual sample's
disjoint ledger closes; medians of rows need not sum to the median total.

GPU term windows exclude subsequent scaling/downloads, retained in fragment
overhead. Localization measures corresponding stages, not identical algorithms.
Canonical induction excludes nested delta-HF assembly; dimer SCF is charged once.
Fragment dispersion supplies canonical dispersion, with no duplicate row.

Instrumentation changes GPU total materially (52.11 s versus 32.95 s baseline).
Thus fragment electrostatics/localization and overhead are **diagnostic leads**,
not established uninstrumented kernel regressions. Fragment induction is the
largest CPU cost and has a 22.51× instrumented reduction; baseline improvement
must still be verified after any optimization.

## Numerical gates

- Small ten-process H200 physical gate passed.
- Six CPU/GPU pairs: maximum energy difference **2.8165e-9 Eh**;
  maximum signed aggregate difference **2.9816e-9 Eh**.
- Ten strict within-arm repeat comparisons passed, including traced versus
  untraced workers. Maximum field difference **9.6142e-10**.
- Independent same-gauge oracle: 23 fields, maximum absolute difference
  **1.7609e-9 Eh**.
- All 14 outer child processes succeeded; owned scratch cleanup completed.
- Sanitizer qualification remains **outstanding** because the loader rejected
  compute-sanitizer before CUDA execution. This is not memory certification.
- HF fragment-response timing is not a GPU KS/XC CPKS qualification.

Ratios were independently recomputed from all three raw samples; the raw
`instrumented_raw_samples` ledgers were checked for additive closure to `1e-8 s`.

Reproduce those arithmetic checks from the retrieved artifact:

```bash
python3 - <<'PY'
import json, statistics
from pathlib import Path
d = json.loads(Path(".pi/fsapt-h200-15472290/timing-result.json").read_text())
assert d["status"] == "PASS_NUMERICAL_TIMING_SANITIZER_OUTSTANDING"
assert d["scratch_cleanup"]["complete"]
for mode in ("baseline", "instrumented_steps"):
    for name, row in d[mode].items():
        for arm in ("cpu", "gpu"):
            values = row[arm]["samples_seconds"]
            assert len(values) == 3
            assert statistics.median(values) == row[arm]["median_seconds"]
        ratio = row["cpu"]["median_seconds"] / row["gpu"]["median_seconds"]
        assert abs(ratio - row["speedup"]) < 1e-12
        print(mode, name, ratio)
for rows in d["instrumented_raw_samples"].values():
    for row in rows:
        assert abs(sum(v for k, v in row.items() if k != "total") - row["total"]) < 1e-8
PY
```
