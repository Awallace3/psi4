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
