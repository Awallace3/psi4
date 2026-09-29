| Case | Max component Δ, Eh | Run-to-run scatter, Eh | Max GRAC shift Δ, Eh | Max neutral Δ, Eh | Max cation Δ, Eh | Within 1e-05 Eh | Interpretation |
|---|---:|---:|---:|---:|---:|:--:|---|
| protein83-6-31+g** | 1.58e-07 | 4.5e-09 | 1.70e-07 | 1.45e-06 | 1.48e-06 | yes | agrees within tolerance |
| protein83-aug-cc-pvdz | 2.42e-07 | 4.1e-10 | 1.30e-07 | 1.14e-06 | 1.21e-06 | yes | agrees within tolerance |

The neutral and cation columns are the monomer SCF energies the GRAC shift is derived from. Where both agree to near machine precision, the arms solved the same problem the same way. Where the neutral agrees but the cation does not, the arms converged to different solutions of a near-degenerate open-shell SCF, and the component difference that follows is not a measure of GPU arithmetic error.
