# Restricted meta-GGA SCF CPU/cuEST comparisons

This is ordinary counterpoise SCF, **not SAPT(DFT)**. IE is
`E_AB - E_A(ghost B) - E_B(ghost A)` at a fixed geometry.
The acceptance threshold is **|CPU/GPU ΔIE| ≤ 1e-5 Eh**; the original
1e-6 Eh flag is retained as a tighter diagnostic, not the acceptance gate.

## Existing accuracy evidence

M06-L, M06 and r²SCAN each have 12 complete comparisons: three systems,
two historical builds and two GPU routes against each build's CPU reference.
**36/36 pass at 1e-5 Eh.** At 1e-6 Eh, four benzene M06-family rows fail.
All r²SCAN IEs pass either threshold.

| Functional | Largest full-XC absolute IE difference / Eh | kcal/mol | Status |
|---|---:|---:|---|
| M06-L | 3.64385326e-6 | 0.00228655244 | Pass at 1e-5 |
| M06 | 6.07148141e-6 | 0.00380991211 | Pass at 1e-5 |
| r²SCAN | 8.34184e-7 (rounded) | 0.000523458145 | Pass at 1e-5 |
| PW6B95 | unavailable | unavailable | 12 incomplete comparisons; CPU ghost failures |
| ωB97M-V | 6.68662210e-6 | 0.00419591872 | 11 complete comparisons pass; six full-XC rows complete; one GPU-JK control timeout |

Together, **47/47 complete comparisons pass at 1e-5 Eh**, out of 60 possible.
Thirteen remain incomplete: twelve PW6B95 and one ωB97M-V GPU-JK control.
All **24/24 full-XC comparisons for M06-L/M06/r²SCAN/ωB97M-V** pass.
This is not a claim that incomplete comparisons pass.

For M06-L/M06/r²SCAN, GPU-JK-only maximum |ΔIE| is 7.35440153e-10 Eh and maximum underlying
|ΔE| is 1.15437615e-9 Eh. Full cuEST XC maximum underlying |ΔE| is
6.14938076e-5 Eh: IE agreement includes cancellation of larger total errors.

Host benzene AB/A/B fixed-density controls attribute M06/M06-L differences
to the combined actual quadrature-grid effect, not an unsupported-functional
failure or demonstrated density-fitting error. Grid-only CP contributions are
6.09192264e-6 / 3.66340345e-6 Eh; archived optimized SCF IE discrepancies are
6.07148141e-6 / 3.64337711e-6 Eh. Their residuals are approximately 2e-8 Eh.
Orientation, nuclear partition and retention were not separately isolated.

## Complete-IE wall-time speedups

**NVIDIA H200 versus eight Psi4 CPU threads; full GPU J/K + XC route.**
Each IE cost sums successful AB/A/B `wall_s` around `psi4.energy`.
Ranges span M06-L, M06 and r²SCAN, not repeated-run uncertainty.

| System | Host-LibXC full-XC speedup | Historical CUDA-LibXC full-XC speedup |
|---|---:|---:|
| water | 1.13–1.19× | 1.21–1.27× |
| benzene | **11.40–11.69×** | **15.53–15.70×** |
| peptide | **4.49–4.58×** | **6.51–6.82×** |

Representative M06 complete-IE times:

| System | Host CPU / GPU, seconds | Historical CUDA-build CPU / GPU, seconds |
|---|---:|---:|
| water | 5.163 / 4.458 | 5.311 / 4.401 |
| benzene | 177.210 / 15.503 | 176.409 / 11.359 |
| peptide | 59.522 / 13.004 | 60.726 / 9.079 |

GPU-JK-only is generally slower or approximately tied because native XC
dominates these cases. Host benzene/M06 AB spends 50.900 of 61.881 CPU seconds
in native XC; full cuEST XC takes 5.161 seconds overall.

## Protocol and scope

Restricted DF, FP64, CPU SAD; spherical basis; def2-universal-jkfit;
99×590 ROBUST XC grid; E=1e-11, D=1e-10, maxiter=200, fail_on_maxiter=True.
Water/benzene aug-cc-pVDZ; peptide 6-31+G**. Benzene is idealized, not S22.
No GRAC, SAPT or added D3/D4. **ωB97M-V retains its intrinsic VV10 and
range-separated exchange**, both checked from the returned functional,
including a finite nonzero `DFT VV10 ENERGY`.

No unrestricted, derivative, response or universal meta-GGA support claim.
The historical builds are not source-matched: their timing difference is not
a controlled CUDA-LibXC ablation. Timings are single time-to-convergence runs,
include SCF setup/finalization, exclude Python startup/import and scheduler
wait, and can include different iteration counts.
PW6B95 has no defensible complete-IE speedup without matched converged references.

## ωB97M-V water preflight

Job **13904944** completed (job/batch/srun exit 0; elapsed 2:49) after
explicit user-approved in-place move from embers to inferno.
All 18 workers passed and final staged provenance was verified.
These measurements are distinct from the M06-family/r²SCAN ranges above.

| Build | Full-XC ΔIE / Eh | CPU CP / s | GPU CP / s | Full-XC CP speedup |
|---|---:|---:|---:|---:|
| host LibXC | −1.17115491e-7 | 25.6448 | 4.8413 | **5.30×** |
| historical CUDA LibXC | −1.17121814e-7 | 29.9292 | 4.9040 | **6.10×** |

All four water comparisons (including GPU-JK-only) pass at 1e-5 Eh;
GPU-JK-only max |ΔIE| is 1.17986914e-7 Eh, not the near-roundoff values
seen for the other three functionals. The returned functional is MGGA/LRC/VV10,
ω=0.3, with finite nonzero VV10 in every fragment/route.

The final follow-up made self-consistent VV10 explicit (`DFT_VV10_POSTSCF=False`,
50×146 VV10 grid, rho cutoff 1e-8), records α/β/b/C, and checks the returned
SCF energy includes the five components including VV10. Water was repeated
alongside benzene and peptide under that final harness pin; the preflight is
not silently spliced into a differently declared final protocol.
The first explicit-option final jobs 13911317/13911319/13911321 failed before
SCF because the harness compared Psi4's integer-zero boolean option with
`is False`. All 54 runtime records report the intended disabled value 0;
no energies were computed. They are preserved as setup failures, not
unsupported-functional or convergence failures.

A captured-option regression reproduces the error and passes after accepting
both boolean False and integer 0 while rejecting enabled/unknown values.
The corrected gate accepts all 54 captured records.
Seventeen local unit tests, Python compile and shell syntax checks pass.
The corrected retry retains the same physical VV10/SCF protocol.

## Final ωB97M-V full-XC results

Corrected jobs **13940495 (water), 13940497 (benzene), 13940499 (peptide)**,
explicitly approved inferno, harness **45c7902013948221e18219a50f3a18dc934ba772**.
All six full-XC comparisons have successful CPU/GPU AB/A/B endpoints and
pass the **1e-5 Eh** IE criterion. Values below use final data, not preflight
repeats. Host and historical CUDA builds are compared to their own CPU reference
within each job's allocation.

| System | Build | CPU CP / s | Full-GPU CP / s | CP speedup | GPU−CPU IE / Eh |
|---|---|---:|---:|---:|---:|
| water | host LibXC | 22.918 | 4.192 | **5.47×** | −1.17112e-7 |
| water | historical CUDA LibXC | 23.634 | 4.182 | **5.65×** | −1.17133e-7 |
| benzene | host LibXC | 515.743 | 16.265 | **31.71×** | −6.68518e-6 |
| benzene | historical CUDA LibXC | 567.608 | 13.677 | **41.50×** | −6.68662e-6 |
| peptide | host LibXC | 439.199 | 15.154 | **28.98×** | +6.07809e-7 |
| peptide | historical CUDA LibXC | 503.757 | 12.519 | **40.24×** | +6.07303e-7 |

The actual shared runtime configuration is MGGA/LRC/VV10, ω=0.3,
α=0.15, β=0.85 (total LR HF=1), b=6, C=0.01, self-consistent VV10,
50×146 NL grid and rho cutoff 1e-8. The 53 successful workers have finite
nonzero VV10 and checked total-energy components; maximum component-sum
drift is 8.52651e-14 Eh. Staged hash maps match across all three jobs, all
final provenance checks pass, and independent worker replay exactly reproduces
the stored complete and incomplete comparison rows.

**The all-route campaign is partial:** water/peptide completed all 18 workers;
benzene completed 17/18. Its host **GPU-JK-only AB control** hit the
300-second worker timeout, return code 124, and has no final result JSON.
Last recorded iteration 20 has ΔE=+5.47402e-11 Eh and density residual
1.05636e-12; that is not a certified converged final energy. Its IE/speedup
remain absent. No full-XC endpoint was lost, so the benzene full-XC rows above
are valid. Do not equate an incomplete extra control with full-XC failure.

All 11 complete ωB97M-V rows pass. Known GPU-JK-only max |ΔIE| is
3.56294e-7 Eh and max underlying |ΔE| is 1.12376e-5 Eh; the earlier
three-functional near-roundoff statement does not apply to range-separated
ωB97M-V. Its differences were not mechanism-decomposed or attributed to a
specific density-fitting error.

Rounded dimer total energies / Eh (complete full-XC endpoints):

| System | Build | CPU E_AB | Full-GPU E_AB |
|---|---|---:|---:|
| water | host | −152.833773631394 | −152.833784887710 |
| water | historical CUDA | −152.833773630238 | −152.833784887724 |
| benzene | host | −464.336419367889 | −464.336431787057 |
| benzene | historical CUDA | −464.336419119947 | −464.336431787046 |
| peptide | host | −496.875039056492 | −496.875042247467 |
| peptide | historical CUDA | −496.875039090925 | −496.875042248328 |

Machine-readable final values, manifest/staged-map digests and limitations:
[`wb97mv-results.json`](wb97mv-results.json).

## Provenance

Original matrix: job 13884503, interrupted after 184 workers (160 success,
16 error, eight timeout); its final unchanged-build verification was not reached.
Recovery 13885267 fills peptide r²SCAN. Invalid replay/capture attempts
13886607 and 13888500 are excluded from scientific conclusions.
Corrected fixed-density 13886797, validated actual-grid AB 13893248 and
ghost closure 13895464 establish the mechanism. CPU PW6B95 controls 13886663
document reproducible original-protocol failures, not an impossibility theorem.

Archived raw records, independent replay and JSON/Markdown/HTML reports:
`~/docs/saptdft/cuest/mgga-ie/` on the experiment workstation.
The ωB97M-V follow-up records staged driver/core/library/basis hashes,
harness commit, geometry hashes, GPU identity, all worker exit codes, both IE
thresholds, component energies, and dimer/complete-IE speedups.
