# ISA-Pol reference data

Reference values used by `../test_isapol.py` and the `../test_isapol_lw_*.py`
tests. They exercise CamCASP 6.0.051 compatibility infrastructure.

| file | contents |
| --- | --- |
| `camcasp_atomprop.dat` | `AtomProp(Z)` for `Z = 0 … 83` — Slater radius, Bondi vdW radius, Grimme vdW radius, Grimme C6, covalent radius. All in atomic units, 17 significant digits. |
| `camcasp_grid_h2o.npz` | The ISA integration grid for water at `n_r = 8`, `n_a = 110`, `k_mu = 3`, `rscale = 1.0` — 3 × 7 × 110 = 2310 points — plus the per-atom offsets and the geometry it was built from. |
| `camcasp_casimir_freq.dat` | The imaginary-frequency Gauss–Legendre quadrature (`omega`, `tm1sq`, `weight`) for every even `n_freq` from 2 to 10, at three values of `omega0`. |
| `camcasp_fit_points.npz` | The PFIT fit-point clouds (CamCASP `RANDOM` lattice, seed 1, `lolim = 2`, `hilim = 4` van der Waals radii): 2000 points each for water and for HCl, and the 500-point water cloud of the `H2O_props` reference case. Each comes with its cube centre, cube half-width and the geometry (`Z`, bohr) it was built from. |
| `orient_local/` | Retained ORIENT/CamCASP water declarations (sites, frames, frequency headers, manifest) for the LW tests and the `oracle/extract_lw_*.py` parsers; see its README for attribution. |

`oracle/griddump` emits the first two, linking directly against CamCASP's
`src/atoms.f90` and `src/gdma/atom_grids.F90`; `oracle/freqdump` the third,
generated from `src/casimir/casimir.f90`; `oracle/latticedump` the fit
points, from `src/random.f90` and `src/lattice.F90`. Regenerating them needs a
CamCASP checkout, which is why the results are committed rather than computed
at test time. The regeneration tooling is not tracked and is not needed by the
retained tests. `camcasp_fit_points.npz` has sha256 `aa450134…86870f1`. As
provenance breadcrumbs only: it is byte-identical to the copy in the
unpublished integration commit `4189ded9cc8f319c8bc21bbe78960f0fab7e2eff`,
which took it from an unpublished local checkout at `202c42217138`. Neither
commit is on a public remote and `latticedump` is untracked, so regenerating
the clouds from this repository alone has not been demonstrated; the tests
check against this committed copy and the Maclaren stream values in
`../test_isapol.py`.

The grid is deliberately small. On the local non-FMA build, the full production grid
(`n_r = 80`, `n_a = 590`, 139 830 points) also matches bit for bit in every coordinate,
but 4.5 MB of doubles does not belong in a test fixture. One 1-ulp difference remains
on that grid: the association of the `4π` factor in the quadrature weights. No
cross-platform bitwise identity is claimed.
