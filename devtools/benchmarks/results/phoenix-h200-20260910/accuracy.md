| Case | Max component Δ, Eh | Run-to-run scatter, Eh | Max GRAC shift Δ, Eh | Max neutral Δ, Eh | Max cation Δ, Eh | Within 1e-06 Eh | Interpretation |
|---|---:|---:|---:|---:|---:|:--:|---|
| benzene-aug-cc-pvdz | 2.08e-06 | 1.2e-10 | 1.17e-04 | 3.20e-07 | 1.17e-04 | no | exceeds tolerance because the arms converged to different cation SCF solutions; gpu found the lower one |
| benzene-cc-pvdz | 2.34e-06 | 8.9e-12 | 1.19e-04 | 2.50e-07 | 1.20e-04 | no | exceeds tolerance because the arms converged to different cation SCF solutions; gpu found the lower one |
| nanotube-6-31+g** | 9.28e-08 | 6.3e-10 | 2.00e-08 | 1.49e-06 | 1.45e-06 | yes | agrees within tolerance |
| peptide-6-31+g** | 4.72e-08 | 3.7e-10 | 1.90e-07 | 4.20e-07 | 4.30e-07 | yes | agrees within tolerance |
| water-aug-cc-pvdz | 5.05e-08 | 1.9e-12 | 2.00e-08 | 4.00e-08 | 5.00e-08 | yes | agrees within tolerance |
| water-cc-pvdz | 5.84e-08 | 3.9e-13 | 2.00e-08 | 4.00e-08 | 5.00e-08 | yes | agrees within tolerance |

The neutral and cation columns are the monomer SCF energies the GRAC shift is derived from. Where both agree to near machine precision, the arms solved the same problem the same way. Where the neutral agrees but the cation does not, the arms converged to different solutions of a near-degenerate open-shell SCF, and the component difference that follows is not a measure of GPU arithmetic error.
