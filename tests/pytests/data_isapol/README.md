# ISA-Pol reference data

Reference values used by `../test_isapol.py`. They exercise CamCASP 6.0.051
compatibility infrastructure.

| file | contents |
| --- | --- |
| `camcasp_atomprop.dat` | `AtomProp(Z)` for `Z = 0 … 83` — Slater radius, Bondi vdW radius, Grimme vdW radius, Grimme C6, covalent radius. All in atomic units, 17 significant digits. |
| `camcasp_grid_h2o.npz` | The ISA integration grid for water at `n_r = 8`, `n_a = 110`, `k_mu = 3`, `rscale = 1.0` — 3 × 7 × 110 = 2310 points — plus the per-atom offsets and the geometry it was built from. |

`oracle/griddump` emits both, linking directly against CamCASP's
`src/atoms.f90` and `src/gdma/atom_grids.F90`. Regenerating them needs a
CamCASP checkout, which is why the results are committed rather than computed
at test time. The regeneration tooling is not tracked and is not needed by the
retained tests.

The grid is deliberately small. The full production grid (`n_r = 80`, `n_a = 590`,
139 830 points) also matches bit for bit in every coordinate, but 4.5 MB of doubles
does not belong in a test fixture. One 1-ulp difference remains on that grid: the
association of the `4π` factor in the quadrature weights.
