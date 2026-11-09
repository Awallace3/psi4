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

### Fixed-density GPU result: reconstruction, not compute-call noise

Job15319813 completed0:0 in54s without TERM. All three fresh workers
returned0 without retirement, native plan finalization returned, groups
were reaped and owned scratch cleanup completed. All workers used the same
actual H200/sm90, driver615.71.09, with no foreign compute context.

All36native compute calls used bitwise-identical coefficients **and**
constructed densities (density SHA256
`e0b92751636455378be72a31405f4516be75216274efa0acbe93d2bb493fde5c`).
Results:

| Comparison | Observed result |
|---|---|
| Repeated computes on one plan | J bitwise identical; K differences at most4.44e-16 |
| Same-plan fixed-D JK energy | Differences at most4.10e-16Eh |
| Cold to warm plan | Fixed-D JK energy differences up to1.92e-9Eh |
| Warm to warm plan | Fixed-D JK energy differences up to1.48e-9Eh |
| Fresh process, same plan position | Fixed-D JK energy differences up to1.78e-9Eh |

Thus different SCF stopping points are **not necessary** to produce
nanohartree-scale differences: rebuilding the native DF operator suffices.
Warm plans also differ after the wrapper has selected FP64, so the late
math-mode assignment is not established as the sole cause. This localizes
the issue to reconstruction/native-plan state rather than repeated
J/K compute calls at fixed state; it does not identify the particular
internal factorization/reduction or prove the absolute accuracy.

Final diagnostic SHA256:
`73aaf895ff4fc488b4771a2d5a57ec011d0c0d52f04e4a7249917e00a4c95669`.
Worker report SHA256 values:
`d8a9b5047a42140a016b4d366dbcbd6c5ec27eaefc381a6ea20c47c47aadab46`,
`6847b7d4575eabb9d239b8c1918f157219aeeccdad6ebee05b4c91c9cc1820b2`,
`e923dfae90b9cbdf541be1362b074e7a7f7192dc7b11f46a637680876ccaf90f`.
Accepted speedup count remains0.

### Independent CPU control

The probe now has a separate CPU-only role: one24-core Gold6226,
160GiB,30min embers allocation with actual local scratch, no GPU GRES,
and CUDA devices hidden. It evaluates the same saved coefficients with
native MemDFJK/DiskDFJK, explicitly grants DF memory, retains the actual
backend/density and compares its J/K matrices against all nine sealed GPU
plan results. The original independent AO-JK oracle tolerances,
rtol2e-7 /atol2e-8, are unchanged. Signed fixed-D energy differences are
reported, not silently turned into a new scalar acceptance threshold.

This is a fixed-wavefunction accuracy diagnostic, **not** a CPU SCF,
full fragment oracle, whole-energy timing or accepted speedup. Its purpose
is to distinguish an absolute operator-accuracy problem from a
cross-construction repeatability floor before proposing a correction or
any change to the experimental acceptance contract. No repeat or
independent-oracle tolerance has been changed.

The public Psi4 `DF_FITTING_CONDITION` option defaults to1e-10
(`psi4/src/read_options.cc`), distinct from the C++ constructor's initial
member value1e-12 before options are read. The wrapper prints/retains the
option but does not configure it into the cuEST DF plan. Native default
conditioning and the relative/absolute cutoff semantics must be checked
before attributing the reconstruction floor solely to unavoidable rounding.
The completed controls above used the unchanged1cad631980 implementation.

## Candidate native policy correction (built; not yet GPU-validated)

The pinned0.2.50 SDK confirms its defaults: relative cutoff1e-12 and QR
metric fitting. Psi4's CPU metric power uses a relative eigenvalue cutoff;
`MemDFJK` passes `DF_FITTING_CONDITION` into DFHelper, whose usual default
is1e-10. The Python cuEST fragment factory already explicitly requests
MATRIXPOWER instead of the default QR. The native SCF wrapper did neither.

Candidate changes in`psi4/src/psi4/libfock/cuESTJK.cc`:

- Pass the actual`condition_` option into the native fitting cutoff.
- Explicitly select relative conditioning and MATRIXPOWER, matching the
  CPU metric-inverse policy rather than relying on the vendor QR default.
- Select requested FP64 math before AO-pair/DF-plan creation, not afterward.
- Report the actual selected metric policy in the native JK header.

This corrects ignored configuration and policy ordering; it is not a change
to repeat/oracle tolerances. The three offline source-contract checks went
red against the old wiring and green after the correction. They do not
constitute native compilation or numerical evidence.

The new GPU regression`test_cuest_jk_honors_fitting_condition_like_cpu`
compares tight1e-10 and deliberately loose0.02cutoffs on the small water
dimer, verifies the CPU control actually changes, requires the GPU operator
to change too, and checks CPU/GPU J/K parity at each cutoff. This numerical
test has been syntax-checked but **not yet executed** on a candidate build.
A new coherent native build and physical GPU gates were required; no old
proof, frozen source or stage is to be advanced or relabeled.

## Independent CPU control15321631 completed

Frozen CPU outer helper:`af107b744fe2374d0fc274c100737aac504fdd71`,
`/storage/project/r-cs207-0/awallace43/gits/psi4.disp0-fixed-jk-cpu-af107b744f`.
Submitted15321631 on`gts-cs207-chemx / cpu-small / embers`,
24Gold6226cores,160GiB,30min, exclusive local-NVMe constraint and **no GPU**.
Test-only15321630 is not the job. Root:
`/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/fixed-density-jk-cpu-af107b744f-01`.
Submission/config SHA256:
`f35f6a1aae59f14072af6d269767c03867704d7965f431e9667ed19e29dc62e9` /
`c0192c0abc091dba6308c117b7aef0ebbc96a0c33d446e8f3cdff56cede5ef49`.
Hourly watcher:`b13e7d1ad`.

The job completed with exit0/no TERM; its worker returned0 with no retirement
and was reaped. Scratch was actual XFS local storage and cleanup completed.
It used **DiskDFJK**, not cuEST, with requested fitting condition1e-10,
24threads and GPUs hidden. No SCF/F-SAPT or GPU work was performed.
All four CPU J/K computations and all constructed densities were bitwise
identical. Input coefficient/density hashes match the GPU controls exactly.

All nine GPU plan results passed the **unchanged** independent AO-JK policy,
rtol2e-7 /atol2e-8. Maximum absolute GPU-minus-CPU discrepancies:

- J:2.014850508658128e-10.
- K:4.519051799434237e-12.
- Signed fixed-D JK energy changes range from-8.847443340941831e-10 to
  -2.806572773887816e-9Eh.

Thus the observed reconstruction differences are below the existing
fixed-gauge AO-JK accuracy criterion while exceeding the separate1e-9
cross-process energy repeat threshold. This is **not** a full fragment/energy
oracle and does not validate a whole-system speedup. The native-policy
correction still requires its own new build and physical regressions; this
CPU control used the original1cad631980 whole stage.

Final CPU diagnostic SHA256:
`f2a3db20e20130b8b0b6665862083a4e57817c261f121a70ff314715a647c842`.
Worker report SHA256:
`62c5eafd84f7a3555193ecaf40a22343a38c9fdca428121be149006f898da49c`.
Candidate native-fix source pin:
`e0da305c02b55a33c6cb6c250072d552c0606555`; source sync is separate from
native build/qualification. No repeat or oracle tolerance was changed.

## Candidate native build submitted

Frozen source:
`/storage/project/r-cs207-0/awallace43/gits/psi4.disp0-scf-fit-e0da305c02`,
pin`e0da305c02b55a33c6cb6c250072d552c0606555`.
Submitted **15322281** for a new whole sm90 build, CPU guards, small native
CPU controls and sealed-runtime admission. Effective route:
`gts-cs207-chemx / cpu-small / embers`,8CPUs,48GiB, three-hour cap,
no GPU; initially PENDING for Priority. Test-only15322280 is not the job.
Root:
`/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/phoenix-scf-fit-build-e0da305c02`.
Manifest SHA256:
`15ca7b4ca03189dfd1b90bf36e688a7907518ee11fdb64395afd04da475a618e`.
Hourly official watcher:`bc0d4d880`.

The old1cad631980 installation, controls and data remain immutable.
Submission is not compilation success or native/GPU validation. A completed
exact-pin proof is required before candidate GPU tests; the fitting-cutoff
regression must execute with zero GPU skips before any candidate speedup claim.

### Build15322281 completed

The fresh sm90 candidate build completed in10min04s, job/batch/step exit0,
no TERM. All386 core CPU guards and150 helper tests passed (one optional
literal-input-suite skip). CPU sealed-runtime admission passed and reaped
its owned groups; it created no CUDA session. This is compilation/CPU
qualification, **not GPU correctness or a new speedup**.

Artifacts beneath
`phoenix-scf-fit-build-e0da305c02/job-15322281/`:

| Artifact | SHA256 |
|---|---|
| `result.json` | `0096ded04f300bff1ca6f352c3e06c0f20b856a736cb84f0fb7aeb9548e56c3b` |
| `cpu-sealed-admission/preflight-result.json` | `ec85124145351a0686f54bbfdca3ef5424db08dcaa153c6e8109cbd267b1033c` |
| `cpu-sealed-admission/runtime-admission.json` | `7233fd168728e20d996d8d6cc78fc392be7fce462f10fa3d8a61a69a9b3ee2ce` |
| `cpu-sealed-admission/runtime-config.json` | `4da9bef9884e48e83564a8b3d9f1a05f56699d4a738bdbd17b7f4e093b7db5d1` |

Next physical gate: the six frozen-source`test_cuest_jk.py` tests, requiring
each named case once and zero skips/failures/errors in JUnit, then the existing
small H200 F-SAPT gate (signed controls, independent same-gauge CPU oracle,
repeat and changed geometry). The new outer helper runs against the completed
build above; it neither advances the source checkout nor overlays installed
code. Both results remain non-benchmark gates with accepted speedup count0.

### Physical native/small GPU gate submitted15323651

Outer launcher pin:`a3434ede733a4ee7f97d4133dd14e3784086f2c1`,
`/storage/project/r-cs207-0/awallace43/gits/psi4.disp0-native-tests-a3434ede73`.
All154 host launcher tests and the three native fitting source guards passed;
the independent launch review found no concrete blockers. Neither mocks nor
source guards are numerical GPU evidence.

Submitted **15323651**, effective route read back as
`gts-cs207-chemx / gpu-h200 / embers`,1H200,8CPUs,160GiB,30min.
Initially PENDING for Priority. Test-only15323649 is not the allocation.
Root:
`/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/native-jk-regression-a3434ede73-01`.
Manifest SHA256:
`a34f68b9de17359807ea851fc7873e1d1e40918ac00329e346db70313d7766e1`.
Runtime-config SHA256:
`08082fff7d5eb31f1e8463a5f2d26d51097c80c589c5e5f3747dfafd9f7a6f1e`.
Official hourly watcher:`be362e84f`.

The immutable native source remains`e0da305c02`; no Protein83 benchmark
retry or new accepted speedup is implied by this correctness submission.

### Physical gate15323651 passed

Completed in3min32s; job, batch and step exited0 with no TERM. The real
device was NVIDIA H200/sm90, driver615.71.09. All **six native JK tests
passed with zero skips, failures or errors**, including the requested-cutoff
CPU/GPU comparison. The worker returned0 with no retirement; its owned
process group was reaped.

The frozen small gate also passed: unscaled, scaled and changed-geometry
CPU/GPU calculations, all three independent same-gauge CPU oracles, and the
fresh GPU repeat returned0. Its status is`PASS_SMALL_H200_ONLY` with
`gpu_small_case_correctness_verified=true`. Scratch cleanup completed.
The combined status is`PASS_NATIVE_JK_AND_SMALL_GPU_GATES`.

| Artifact beneath`native-jk-regression-a3434ede73-01/job-15323651/` | SHA256 |
|---|---|
| `diagnostic-result.json` | `1108303b704f7cda227324ef705bc1157b82fb4077029c370def47d2727790ca` |
| `r0/probe-result.json` | `b62beea247956f4b7713bedeb3094408d4bf2616d436b1b0fd526b419e540f8b` |
| `r0/native-jk.xml` | `8e1dd1d9b4e33cae6df968fd636bb94ee7072e4ddfd897240660b805255b8e8b` |
| `small-gate/gate-result.json` | `e56dd6ae7125f3b73648dfc0dd52fbcaa3ce2fa771364e0d0b1234eeca0b4709` |

These validate the native configuration correction at small size, not
Protein83 repeatability or a whole-system speedup. The next controlled
experiment uses the unchanged frozen Protein83/def2-SVP geometry and
three fresh unprofiled GPU processes followed by a separate stage trace.
All original repeat and independent-oracle tolerances remain unchanged;
CPU reference jobs wait for a complete successful GPU producer.

### Corrected Protein83/SVP producer submitted15337409

Submitted **15337409** using the same frozen outer`a3434ede73` and native
source/build`e0da305c02`. Read-back route:
`gts-cs207-chemx / gpu-h200 / embers`,1H200,8CPUs,160GiB,8h cap,
initially PENDING for Priority. Test-only15337408 is not the allocation.
Root:
`/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/native-fit-stage-e0da305c02-01-protein83-def2-svp-gpu`.
Manifest SHA256:
`9dfc4639e01d6ed7a92b4aa19980b3025accac42091af2f75dd2d163454039ba`.
Runtime-config SHA256:
`8c2214fa5c18991d62530776c930d156c285c94b86c9f08ec31e44874ea1d102`.
Official hourly watcher:`badef46ed`.

The83-atom/766-AO case uses the original frozen SVP input manifest
`e93a40f2ca961461cb98ba423b8eff5f199d7b201a0925185c80fff610dfe327`,
96GiB Psi4 host budget,120GiB device limit and12GiB native reservation.
Only native fitting policy differs from the prior candidate. It must pass
the original three-process strict repeat checks before the separate trace
and independent CPU reference/oracle qualification. No automatic retry,
relaxed tolerance, Protein157 calculation or accepted speedup was added.
