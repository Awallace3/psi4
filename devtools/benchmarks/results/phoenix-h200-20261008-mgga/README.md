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
| ωB97M-V | 1.17121814e-7 (water preflight only) | 0.00007349 (rounded) | Water 4/4 complete pass; final three-system campaign pending |

GPU-JK-only maximum |ΔIE| is 7.35440153e-10 Eh and maximum underlying
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

The final follow-up makes self-consistent VV10 explicit (`DFT_VV10_POSTSCF=False`,
50×146 VV10 grid, rho cutoff 1e-8), records α/β/b/C, and checks the returned
SCF energy includes the five components including VV10. Water will be repeated
alongside benzene and peptide under that final harness pin; the preflight is
not silently spliced into a differently declared final protocol.
Sixteen local unit tests, Python compile and shell syntax checks pass.

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
