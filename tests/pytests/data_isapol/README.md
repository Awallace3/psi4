# ISA-Pol reference data

Reference values used by `../test_isapol.py` and `../test_isapol_fit.py`.
The original fixtures exercise CamCASP 6.0.051 compatibility infrastructure;
the water fitting fixture below is explicitly a matched-sample kernel oracle,
not a converged ISA-A property reference.

| file | contents |
| --- | --- |
| `camcasp_atomprop.dat` | `AtomProp(Z)` for `Z = 0 … 83` — Slater radius, Bondi vdW radius, Grimme vdW radius, Grimme C6, covalent radius. All in atomic units, 17 significant digits. |
| `camcasp_grid_h2o.npz` | The ISA integration grid for water at `n_r = 8`, `n_a = 110`, `k_mu = 3`, `rscale = 1.0` — 3 × 7 × 110 = 2310 points — plus the per-atom offsets and the geometry it was built from. |
| `camcasp_casimir_freq.dat` | The imaginary-frequency Gauss–Legendre quadrature (`omega`, `tm1sq`, `weight`) for every even `n_freq` from 2 to 10, at three values of `omega0`. |
| `camcasp_prand.dat` | The `dprand()` stream: the first 256 deviates for each of four seeds, plus checkpoints at draws 1000, 10 000 and 100 000. |
| `camcasp_fit_points.npz` | The 2000-point fit-point cloud for water and for HCl at the protocol's defaults, plus the cube centre, its half-width and the geometry each was built from. |
| `camcasp_recoupling.dat` | The anisotropic dispersion recoupling coefficients — 393 `(n, L1, L2, J)` blocks, 4673 terms, as exact root-rational fractions. This checks coefficient data, not an anisotropic property engine. |
| `camcasp_isa_fit_water.npz` | Frozen ISA-A fits for O/H1/H2 at three active option settings, on identical water grid/density/shape/basis samples. Includes modified metrics, RHS, solved coefficients and integrated populations from source-extracted Fortran arithmetic. |
| `camcasp_isa_fit_water.json` | Water/edge fixture provenance, source/generator/data hashes, actual density/basis definitions and limitations. |
| `camcasp_isa_fit_edges.npz` | Source-extracted synthetic fitting checks: signed samples, exact denominator cutoff with nonzero damping, automatic ridge eligibility and nonzero exponent-cap contributions. Not a physical density/basis. |

`oracle/griddump` emits the first two, linking directly against CamCASP's
`src/atoms.f90` and `src/gdma/atom_grids.F90`; `oracle/freqdump` the third,
generated from `src/casimir/casimir.f90`; `oracle/pranddump` and
`oracle/latticedump` the next two, generated from `src/random.f90` and
`src/lattice.F90`; `oracle/parse_cncode.py` the last, parsed from
`src/casimir/c6code.f90 … c12code.f90`. See `oracle/README.md` for how to rebuild
them and regenerate these files; you need a CamCASP checkout, which is why the
results are committed rather than computed at test time.

The grid is deliberately small. The full production grid (`n_r = 80`, `n_a = 590`,
139 830 points) also matches bit for bit in every coordinate, but 4.5 MB of doubles
does not belong in a test fixture. See `psi4/src/psi4/libisapol/SPEC.md` §3.5.5 for
what was measured on it and for the one remaining 1-ulp difference (the association
of the `4π` factor in the quadrature weights).
