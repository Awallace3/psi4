# End-to-End Wiring Specification (Task 7)

Specification for chaining Tasks 2–6 and publishing the seven public variables. Records
the architectural decisions taken on 2026-08-17 so the implementation does not have to
re-litigate them.

## Decision 1: the Python driver orchestrates the SCF triple

`FrozenResponseContext::create` requires three wavefunctions — the GRAC-corrected
reference, the neutral precursor, and the cation. `AtomicPolarizabilityCalculator`
currently holds a single `wfn_`.

**Decided:** the driver runs the three SCFs; the calculator receives them.

- Add a driver-level routine (under `psi4/driver/procrouting/`) that runs the three SCFs
  explicitly and hands the results to the calculator.
- Change `AtomicPolarizabilityCalculator` to take the three wavefunctions and **fail closed
  with a named exception** if any is missing or inconsistent (mismatched molecule, basis,
  or geometry).
- `OEProp` remains the publication entry point and keeps its existing dispatch. A bare
  `OEProp` call on a single wavefunction must continue to fail closed with the existing
  clear message rather than silently producing partial output.

Rationale: multi-SCF workflows belong in the driver by Psi4 convention, and a property
class should not drive SCF. This does mean the pipeline is not reachable from a bare
`OEProp` call alone; that is accepted.

Note this narrows the plan's Task 7 interface line ("one OEProp call publishes ...").
Publication still happens in one call, but the caller must supply the SCF triple.

## Decision 2: bond-graph derivation stays fail-closed on disconnected graphs

Non-covalent complexes (e.g. a water dimer) are rejected. The reviewed model is a monomer,
LW localization over a disconnected graph is not well defined, and atomic polarizabilities
and dispersion coefficients are computed per monomer and only then combined. Revisit only
if cluster support becomes a goal.

## Stage chain

All stages exist and are individually tested. The wiring must connect them in this order,
validating between stages rather than at the end only:

```text
(wfn_grac, wfn_neutral, wfn_cation)
  -> FrozenResponseContext::create
  -> compute_isa_weights
  -> ISAPolResponseProvider::compute_isapol_response(grid)   [SitePairResponse per frequency]
  -> derive_bond_graph(molecule)                             [fail closed if disconnected]
  -> localize_lw(response, graph, residual_tolerance)        [per frequency]
  -> generate fit points + evaluate_point_response           [PointResponseData]
  -> derive_pdef_constraints(molecule)                       [active-variable mask]
  -> refine_wsm(localized, point_response, constraints, options)  [RefinedL3Model per frequency]
  -> compute_dispersion(models, grid)                        [C6, C8, C10, C12]
  -> pack + rotate to global Cartesian                       [(3,6) static, (33,6) dynamic]
  -> publish seven array variables
```

`make_casimir_grid(10, 0.5)` supplies the grid and must be the same object handed to
`compute_dispersion`, which validates the protocol grid.

## Frame hazard (must not be got wrong)

`derive_pdef_constraints` returns a mask expressed in the **molecular frame** when
`site_axes` is empty, which is what `refine_wsm` needs because its harmonics are global. If
a caller supplies non-identity local axes, the mask indexes variables in *those* frames and
must not be handed to `refine_wsm` unchanged. Assert the frame convention at the call site;
a silent mismatch produces plausible-looking wrong anisotropy.

Separately, the reviewed protocol runs `symmetry c1`, `no_com`, `no_reorient` with the
molecule in the `xz` plane, i.e. **not** Psi4's canonical frame. Verified on 2026-08-17
that `derive_pdef_constraints` still detects `C2v(Z)` geometrically under those flags and
returns 170 active / 104 independent variables, so declared `c1` does not defeat the mask.
Any refactor must preserve that: keying the mask off the *declared* point group instead of
the geometry would silently disable all constraints.

## Publication contract

Publish only after every validation gate passes; never publish partial arrays.

| Variable | Shape | Content |
| -------- | ----- | ------- |
| `ATOMIC POLARIZABILITIES` | `(3, 6)` | static global Cartesian, packed `xx, xy, xz, yy, yz, zz` |
| `ATOMIC DYNAMIC POLARIZABILITIES` | `(33, 6)` | frequency-major, 11 blocks x 3 atoms |
| `ATOMIC POLARIZABILITY FREQUENCIES` | `(11, 1)` | static point plus ten mapped nodes |
| `ATOMIC C6` / `C8` / `C10` / `C12` | `(3, 3)` | isotropic `00 00 0`, `hartree * bohr^n` |

Atom order is `O, H1, H2`. The static tensor must equal the zero-frequency block of the
dynamic output exactly.

## Grid quality

Do not use the SCF test fixtures for parity. Measured behaviour:

| DFT grid | ISA grid | electron-count error | LW charge sum | tightest LW tolerance |
| -------- | -------- | -------------------- | ------------- | --------------------- |
| `50 / 12` | `30/10/12` | `2.7e-02` | `2.4e-03` | `1e-02` |
| `302 / 50` | `60/18/24` | `1.8e-07` | `1.5e-08` | `1e-06` |
| `590 / 99` | `100/24/32` | `6.9e-09` | `8.7e-10` | `1e-08` |

The `1e-4` parity gate requires at least the middle row. Pin the parity grid explicitly in
options and record it, rather than inheriting a default.

## Tests to add for this task

Extend `tests/pytests/test_atomic_polarizabilities.py`, which already holds the reviewed
literals and passes 11 tests. Required additions:

- published shapes `(3,6)`, `(33,6)`, `(11,1)`, four `(3,3)`
- all published values finite
- static block equals the zero-frequency dynamic block
- per-atom global tensors exactly symmetric at every frequency
- H2O C2 relation `alpha_H2 = S_x alpha_H1 S_x` on published output
- dispersion matrices symmetric, and H1/H2 rows equal
- fail-closed behaviour: missing any of the three wavefunctions raises a named
  `PsiException` and publishes nothing (assert none of the seven variables exist afterwards)
- the six property comparisons against the reviewed literals at `rtol=1e-4, atol=1e-5`

Do not loosen tolerances, replace literals, or read reference data from pytest.
