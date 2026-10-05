# ISA-Pol reference data

Reference values used by `../test_isapol.py`, the `../test_isapol_lw_*.py`
tests and the Casimir–Polder dispersion tests. They exercise CamCASP 6.0.051 compatibility infrastructure.

| file | contents |
| --- | --- |
| `camcasp_atomprop.dat` | `AtomProp(Z)` for `Z = 0 … 83` — Slater radius, Bondi vdW radius, Grimme vdW radius, Grimme C6, covalent radius. All in atomic units, 17 significant digits. |
| `camcasp_grid_h2o.npz` | The ISA integration grid for water at `n_r = 8`, `n_a = 110`, `k_mu = 3`, `rscale = 1.0` — 3 × 7 × 110 = 2310 points — plus the per-atom offsets and the geometry it was built from. |
| `camcasp_casimir_freq.dat` | The imaginary-frequency Gauss–Legendre quadrature (`omega`, `tm1sq`, `weight`) for every even `n_freq` from 2 to 10, at three values of `omega0`. |
| `camcasp_fit_points.npz` | The PFIT fit-point clouds (CamCASP `RANDOM` lattice, seed 1, `lolim = 2`, `hilim = 4` van der Waals radii): 2000 points each for water and for HCl, and the 500-point water cloud of the `H2O_props` reference case. Each comes with its cube centre, cube half-width and the geometry (`Z`, bohr) it was built from. |
| `camcasp_casimir_h2o_vdz_l3.json` | The decoded CASIMIR stage of the `tests/H2O_props/psi4` declaration (`SCFcode psi4`, cc-pVDZ, PBE0, `AC NONE`, ALDA+CHF, constrained-NN DF, 100×400 grid; prefix `water_L3`, localization rank limit 3): the printed isotropic `00 00 0` row of each type-pair block of `water_L3_casimir.out`, a census of its recoupled rows, the declared stage settings, and the rank isotropics `tr(α_ll)/(2l+1)` of `water_L3_0f10.pol`, the localized file that stage read. Read by `../test_isapol_primary_casimir_reference.py` and `../test_isapol_casimir_rank4_truncation.py`. sha256 `3cfc05970c8e7182b16d1227a3e6c5ca0309d20425338276be59ad7aa801b1c2`. |
| `camcasp_casimir_h2o_vdz_l4.json` | The same reference response localized at rank limit 4 (`water_L4`), a different declared model; same layout. Read by `../test_isapol_casimir_rank4_truncation.py`. sha256 `23abdd9183fc59ba54541807e262ce837fffcc11f9a72df1b4f540f14a1ba8a6`. |
| `h2o_props_psi4_basis/NOTICE` | The CamCASP MIT notice (and the MolSSI-BSE notice) that the two Casimir fixtures' `notice` fields point to. Only the notice is shipped; no basis records are. |
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

## Casimir–Polder fixtures and historical evidence

The two `camcasp_casimir_h2o_vdz_*.json` files decode printed CamCASP output
text only; no CamCASP source is read or run. They are byte-identical to the
copies in the unpublished candidate `4189ded9cc8f319c8bc21bbe78960f0fab7e2eff`
(provenance breadcrumb only, like the other identifiers here). Their decoder,
`oracle/read_casimir_out.py`, was tracked in that unpublished history (e.g.
`d680666432`) and later untracked; it is not shipped, and regeneration from
this repository alone has not been demonstrated.
`../test_isapol_camcasp_local_pol.py` carries its 33 scalars and six printed
coefficients as literals transcribed from CamCASP `examples/properties/H2O`
(`H2O_aTZ`, weight type 4); that capture is not shipped either.

The following are recorded measurements, not regression data; no test
recomputes them:

- A native declared-model chain (`REFERENCE_AUX_ATOMAUX_SLATER_TAIL_REFSCF_TAILACT1e-6_REFIT_DFPROP_ISA_A`,
  rank-3 LW, 100×400 response grid) was once compared with the printed L3
  row. Worst relative residual over `n = 6…12`: O–O `3.06e-5`, H–O `9.56e-5`,
  H–H `2.02e-4`, the last of the same order as the reference's own H2/H3
  asymmetry (`> 1e-4`). It is a different model from the reference and these
  numbers are not a stage-level calculation or an accuracy claim.
- The constants 8.400508 / 81.842910 / 700.674100, once quoted as the printed
  O–O C6/C8/C10 (unpublished commit `eb49a236ea`), were a transcription error
  and appear in no CamCASP file. The printed row is 8.367451 / 81.67585 /
  701.1518 / 4950.050, and it is the Casimir integral of `water_L3_0f10.pol`
  to `<= 3.2e-7` relative (asserted by the primary reference test).
- For the `H2O_aTZ` example: its two routes to the static molecular isotropic
  polarizability (translated rank-4 distributed, 9.247357; summed refined
  local, 9.271584) differ by 0.26%; its printed `00(ll)` recoupled rows match
  `(-1)^l sqrt(2l + 1) tr(α_ll)/(2l+1)` to `< 1e-5` relative; its rank-4
  distributed sum rules close to 7.7e-7; and its `COPY H1` declaration holds in
  local, not global, axes.
