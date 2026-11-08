# Protein83/def2-SVP: upstream HF repeatability investigation

## Reproduced symptom

H200 job15318800 completed two full Protein83/SVP calculations without OOM.
The unchanged strict repeat guard rejected their interaction-energy
difference4.018948517731102e-9Eh. Source1cad631980, outer387b36b92b;
all source/config/input identities and original artifacts remain frozen.
See `DEF2_STAGE_CAMPAIGN.md` for job roots, timings and hashes.

Retrieved immutable result, raw-HF, fragment and output files:
`/tmp/fsapt-protein83-def2-15318800/r0/` and`r1/`.
Both NPZ files per process were matched to the original result's SHA256 seals.

This fast local replay uses the actual unchanged scalar guard and exits1:

```bash
PYTHONPATH=devtools/disp0/phoenix_validation python3 - <<'PY'
import json
from pathlib import Path
import numpy as np
from repeat_gauge import close
r = Path("/tmp/fsapt-protein83-def2-15318800")
a, b = [json.loads((r / f"r{i}/result.json").read_text()) for i in (0, 1)]
close(np.asarray(a["energy_Eh"]), np.asarray(b["energy_Eh"]),
      "Signed repeat scalar energy_Eh")
PY
```

Actual replay:0.3s, AssertionError at the original1e-9 absolute/relative
tolerance. This reproduces the symptom; it does not make a new timing sample.

## Localization is not the only failing check

For diagnosis only, separately calling the unmodified `compare_arrays`
on the sealed snapshots fails its no-continuous-IBO-rotation check:
26/6084 monomer-A overlap entries, maximum1.19783048e-8 against1e-9.
This is not an accepted check of all23fields; later checks still stop at
the first failure. No energy matrix was rotated and no tolerance changed.

## Upstream evidence

Total interaction-energy delta tracks the dimer-minus-monomer HF energy
delta to2.5e-12Eh, the dispersion change. Native final SCF energies:

| Process | Dimer (Eh) | A (Eh) | B (Eh) |
|---|---:|---:|---:|
| r0 | -1752.51662916069608 | -984.96510089460605 | -767.55240649620782 |
| r1 | -1752.51662916650912 | -984.96510089731237 | -767.55240649529310 |

The raw AO overlap is identical; nuclear-attraction matrices differ by at
most2.842170943040401e-14. In contrast:

| Quantity | Maximum absolute r0/r1 difference |
|---|---:|
| A density matrix | 1.5456329549579095e-9 |
| A Coulomb matrix | 1.9041657139950985e-9 |
| A exchange matrix | 2.0030201941523273e-9 |
| A occupied orbital energies | 5.760862009651646e-9 |
| B density matrix | 6.105501867970098e-11 |

A density's S-weighted Frobenius difference is1.5658025446141463e-8;
B's is2.1233075031261974e-10. AO metric eigenvalues range from
0.0002229914929360157 to6.617150593293588, condition29674.453075177567.
Canonical occupied overlap orthogonality residuals remain about3e-14;
this does not prove identical occupied projectors to the strict repeat
threshold.

Both runs requested E/D convergence10 and INTS_TOLERANCE1e-16, with
mixed precision disabled. Last dimer iterations:

- r0 iter20: energy change-7.95808e-12, density residual2.02180e-11.
- r1 iter22: energy change-5.72982e-11, density residual4.96634e-12.

Monomer A stopped at18/19iterations with residuals3.84258e-11 /
1.36895e-11. Monomer B stopped at18 in both, residuals5.40858e-11 /
5.40898e-11. Merely observing that these satisfy SCF convergence does not
establish a process-independent native integral/Fock operator. Tightening
SCF is a hypothesis to test, not a demonstrated fix.

## Source clues, not yet established causes

In `psi4/src/psi4/libfock/cuESTJK.cc`, `preiterations()` sets
`CUEST_NATIVE_FP64_MATH_MODE` only **after** native AO-pair and DF-plan
creation. `psi4/src/cuest_runtime.cc` initializes the shared cuEST handle
without explicitly setting that math mode. Thus plan construction precedes
the requested FP64 selection for a newly initialized handle. Whether cuEST
uses the handle's math mode during plan creation needs a controlled probe;
the ordering alone does not prove this caused the observed drift.

The JK wrapper also reads and prints `DF_FITTING_CONDITION` through
`condition_`, but its current plan-parameter setup does not pass that value
to cuEST. The Python fragment plan separately configures its fitting cutoff.
This is an option-handling discrepancy to investigate, not permission to
change cutoffs or assume the two paths already match.

Next discriminating tests should separate fixed-density JK/native-plan
repeatability from SCF stopping sensitivity, using one frozen wavefunction
and unchanged numerical guards. No unchanged full benchmark retry or
independent CPU qualification was submitted from this failed producer.

## Fixed-density control

`scf_repeat_probe.py` prepares a bounded native-JK control, not another full
F-SAPT retry. It uses the already qualified1cad631980 whole installation,
sealed loader/CPU admission, the exact completed r0 result SHA, its original
runtime closure and frozen SVP basis/geometry files. Raw-HF payload hashes are
checked inside each worker before loading the monomer-A coefficients.
No source overlay, new installed native code, SCF, localization or F-SAPT
calculation is performed. Accepted speedup count is always0.

One30min embers/H200 allocation runs three fresh sealed processes. Each
process builds three sequential native JK plans: one cold shared-handle
construction and two warm rebuilds. Each plan evaluates J/K four times
from identical coefficient bytes and is finalized before the next plan.
Actual constructed density hashes/residuals are retained too. This separates:

1. Same-plan/kernel accumulation drift.
2. Warm-plan reconstruction drift at fixed global state.
3. Cold-versus-warm effects.
4. Fresh-process variation with one frozen wavefunction.

The probe retains J/K matrices, signed fixed-density closed-shell
`2 Tr(D deltaJ) - Tr(D deltaK)` changes, real native CUDA-context UUID/hardware,
compute-call counts and cleanup receipts. Cold means first C++ cuEST JK plan
in that process, not an assertion that all GPU/library caches were empty.
Cold/warm differences alone cannot distinguish math-mode inheritance from
other initialization/cache effects; a subsequent one-variable code experiment
would still be required.

The outer batch wrapper permits only the two explicit driver names,
`stage_scaling.py` and`scf_repeat_probe.py`; each submitter explicitly chooses
its driver. Source/build proof pins, thread policies, resource route,
retirement channels, group reaping and scratch ownership remain required.
No numerical guard, reserve or workspace budget is relaxed.

### Control submitted

Frozen outer control pin:
`5900959ae09e6d3b477253871ffa994a42ac4af8`, checkout
`/storage/project/r-cs207-0/awallace43/gits/psi4.disp0-fixed-jk-5900959ae0`.
All148host helper tests, batch shell syntax and whitespace checks passed.
They cover coefficient reuse, output ownership, cleanup on failure,
rejection of CPU fallback, signed trace calculations and native-context
identity checks; they are not GPU correctness evidence.

Submitted **15319813** successfully, after finalized CPU sealed admission
and frozen reference/config/input checks. Effective route:
`gts-cs207-chemx / gpu-h200 / embers`, oneH200,8hostCPUs,160GiB,
30min cap, initially PENDING for Priority. Test-only15319812 is not the
submitted job. Root:
`/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/fixed-density-jk-5900959ae0-01`.

- Manifest SHA256:
  `3d88759f6ffd13fbcea12ff442742177abfc4259355c80349ee40566b09d82c3`.
- Diagnostic runtime config SHA256:
  `a1082c53dd0c67d17b52b42134ca0c056678e9439c3e64aee533f3d2fc78988c`.
- Reference result SHA256:
  `59838c6873d4f386aeccb4bdad7f699bf4111d354c1bc7113dfdbe04b218cbf5`.
- Hourly official watcher:`b4a66d90a`.

The installed native/scientific source remains1cad631980, not the outer
helper pin. No native precision-ordering change is present in this control.
The pinned SDK documents default math mode as potentially mixed precision
and native-FP64 mode as FP64 throughout; the header does not establish how
plan creation depends on the current mode. No causal conclusion is claimed
before the control, and no SAPT timing or speedup will be accepted from it.
