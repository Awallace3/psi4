| Case | Max component Δ, Eh | Run-to-run scatter, Eh | Max GRAC shift Δ, Eh | Max neutral Δ, Eh | Max cation Δ, Eh | Within 1e-05 Eh | Interpretation |
|---|---:|---:|---:|---:|---:|:--:|---|
| protein157-6-31+g** | 8.93e-07 | 0.0e+00 | 1.00e-08 | 8.24e-06 | 9.01e-06 | yes | agrees within tolerance |

The neutral and cation columns are the monomer SCF energies the GRAC shift is derived from. Where both agree to near machine precision, the arms solved the same problem the same way. Where the neutral agrees but the cation does not, the arms converged to different solutions of a near-degenerate open-shell SCF, and the component difference that follows is not a measure of GPU arithmetic error.
