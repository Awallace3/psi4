# CamCASP Atomic Polarizability Parity Design

## Purpose

Replace the current unverified Psi4 atomic-polarizability post-SCF behavior with a test-driven implementation whose results agree directly with the documented CamCASP properties workflow.

The scientific reference follows Figure 1 of the CamCASP user guide:

```text
Psi4 orbitals
    -> CamCASP distributed polarizabilities and point-to-point responses
    -> ORIENT localization
    -> PFIT WSM refinement
    -> localized frequency-dependent atomic polarizabilities
    -> CASIMIR distributed dispersion coefficients
```

Production pytest must not invoke CamCASP, ORIENT, PFIT, or CASIMIR. It compares Psi4 results against hard-coded values produced by a tracked local regeneration script.

## Current-State Problems

The current branch contains an incomplete implementation in `psi4/src/psi4/libmints/atomic_polarizability.{cc,h}` and broad-range tests in the misspelled file `tests/pytests/test_atomic_polarizabilites.py`.

The existing implementation is not a faithful CamCASP reimplementation. In particular, it uses heuristic free-atom or volume partitioning, rescales uncoupled frequency dependence from a static response, and does not reproduce the ORIENT/PFIT localization and refinement stages. The existing tests accept broad numerical ranges and print CamCASP values without asserting component-wise parity.

These approximations and broad-range assertions are not acceptable under this specification.

## Goals

1. Reproduce CamCASP's localized, WSM-refined atomic polarizabilities in Psi4.
2. Compare full static atomic dipole-dipole tensors in global Cartesian coordinates.
3. Compare the same dipole-dipole tensors at the static point and ten imaginary frequencies.
4. Reproduce CASIMIR's isotropic atom-pair C6, C8, C10, and C12 coefficients.
5. Develop each property with an ordinary failing pytest followed by the implementation needed to make it pass.
6. Preserve a reproducible local provenance workflow without committing generated reference JSON or external packages.

## Non-Goals

- Production Psi4 must not call external CamCASP, ORIENT, PFIT, or CASIMIR executables.
- Production pytest must not clone software, access the network, or read locally generated JSON.
- Broad plausibility ranges are not substitutes for reference comparisons.
- The historical L2 values near 6.51 a.u. for O and 1.38 a.u. for H are not normative after adopting the complete L3 model.
- This work does not add anisotropic spherical Cn components to the public Psi4 API. The initial Cn API contains the isotropic `00 00 0` atom-pair coefficients.

## Canonical Scientific Protocol

### Molecule and electronic structure

Use the Psi4-backed CamCASP H2O properties example as the canonical system.

- Geometry, in Bohr:
  - O: `(0.0000000000, 0.0000000000, 0.0000000000)`
  - H1: `(-1.4536519600, 0.0000000000, -1.1216873200)`
  - H2: `( 1.4536519600, 0.0000000000, -1.1216873200)`
- Charge: 0
- Multiplicity: 1
- Psi4 orientation: `symmetry c1`, `no_com`, and `no_reorient`
- Orbital method: PBE0
- Orbital basis: aug-cc-pVTZ (`aVTZ` in the CamCASP input)
- Asymptotic correction: Psi4 GRAC protocol used by CamCASP
- Experimental ionization potential: `12.62063 eV`
- CamCASP H2O input HOMO value: `-0.3989 hartree`
- CamCASP response kernel: ALDA+CHF
- Point grid: CamCASP `Options Tests` grid for a deterministic, tractable regression calculation

The regeneration script must make these choices explicit rather than relying silently on CamCASP defaults.

### Frequency grid

Use the CamCASP/CASIMIR default quadrature:

- one static point at zero frequency
- ten nonzero Gauss-Legendre imaginary-frequency points
- base frequency parameter `0.5 a.u.`

The exact generated frequencies are stored in local JSON and copied as hard-coded pytest literals.

### Localization and refinement

Use one complete model for all accepted properties:

- nonlocal CamCASP polarizability rank: L4 (`NL4` output), matching the standard properties workflow
- ORIENT localization method: Lillestolen-Wheatley (`LW`)
- ORIENT localization limit: L3
- PFIT WSM limit: L3
- PFIT hydrogen limit: L3
- PFIT penalty weight: 4
- PFIT cutoff: `0.0001`
- local axes: the symmetry-related definitions from `camcasp-bin/tests/H2O_props/psi4/H2O.axes`:

```text
Axes
  H1  z global Z x from H2 to H1
  H2  z global Z x from H1 to H2
End
```

L3 is required on every atom because CamCASP maps L1 to coefficients through C6, L2 through C10, and L3 through C12. Lowering only the hydrogen limit would deliberately truncate higher-order coefficients involving H.

Changing localization rank, WSM rank, hydrogen rank, weighting, cutoff, axes, grid, basis, functional, kernel, or asymptotic correction defines a different reference model and requires intentional regeneration and review of every hard-coded value.

### Coordinate convention

CamCASP and PFIT may report spherical tensors in atom-local frames. Psi4's public dipole-dipole output is global Cartesian.

For atom A, convert the spherical dipole-dipole block to a local Cartesian 3x3 matrix using CamCASP's documented real-spherical ordering, then rotate it using the orthonormal local-to-global matrix `R_A`:

```text
alpha_A_global = R_A @ alpha_A_local @ R_A.T
```

The regeneration script must validate `R_A @ R_A.T = I`, preserve right-handed frames, and store the spherical block, local Cartesian block, rotation matrix, and global Cartesian block in the local JSON.

## Orient Installation

`devtools/regenerate-camcasp.sh` installs or verifies Orient as an ignored external dependency.

Default behavior:

1. Clone `https://gitlab.com/anthonyjs/orient.git` into `<repo>/orient` using the public branch/ref.
2. Pin and record the selected Orient commit; the initial known public reference is `d8d861098c8f548e2cf230c387c8431d9418650a`.
3. Prefer the supplied non-graphical Linux binary `x86-64/gfortran/exe/orient-5.0.11-ng`.
4. Link it through `orient/bin/orient` and prepend the required directory to `PATH`.
5. Run a non-graphical smoke test before starting reference generation.
6. Accept `ORIENT_EXE=/absolute/path/to/orient` as an override.

The Orient checkout, source, and binaries remain untracked. This avoids adding GPL material to Psi4's LGPL production history. If the supplied binary is incompatible, the script must fail with build guidance rather than silently skipping ORIENT. A source build uses `make OPENGL=no` and requires the dependencies documented by Orient.

## CamCASP Installation

The regeneration script uses `CAMCASP`, defaulting to `<repo>/camcasp-bin`.

It must:

1. Verify or clone the CamCASP binary distribution.
2. Record the CamCASP version and Git commit; the initial checkout is version 7.2.2 patch 003 at commit `b474442`.
3. Idempotently unpack the supplied gfortran binaries for `camcasp`, `cluster`, `process`, `pfit`, and `casimir`.
4. Create or update links in `camcasp-bin/bin`.
5. Create a local `psi4.sh` wrapper for the selected staged Psi4 executable.
6. Set `CAMCASP`, `ARCH=x86-64`, `PATH`, `PSIPATH`, and isolated scratch paths required by CamCASP.
7. Verify all five CamCASP executables and the Psi4 wrapper before running the calculation.

`PSI4_EXE` defaults to `<repo>/build_camcasp/stage/bin/psi4` and is overridable. The script records the Psi4 version, Git revision, dirty status, and executable path. The Psi4 build used for reference generation supplies orbitals to CamCASP; the new Psi4 post-SCF implementation is not used to create its own reference values.

## Reference-Generation Script

Create `devtools/regenerate-camcasp.sh` as the auditable source of the hard-coded test values.

### Documented controls

The beginning of the script contains a prominently commented scientific configuration block equivalent to:

```bash
LOCALIZATION_LIMIT=3
WSM_LIMIT=3
HYDROGEN_LIMIT=3
PFIT_WEIGHT=4
PFIT_CUTOFF=0.0001
N_FREQUENCIES=10
FREQUENCY_SCALE=0.5
```

Comments must state:

- L1 supports dispersion through C6, L2 through C10, and L3 through C12.
- `WSM_LIMIT` must not exceed `LOCALIZATION_LIMIT`.
- Reducing `HYDROGEN_LIMIT` truncates higher-order H-containing coefficients.
- Rank or penalty changes can alter even the fitted dipole-dipole subset.
- Values from a changed protocol must not replace pytest literals without reviewing the complete protocol change.

### Pipeline stages

The script performs these stages in order:

1. Validate prerequisites and create isolated working and scratch directories under `.camcasp-reference/`.
2. Install or verify Orient.
3. Install or verify CamCASP.
4. Materialize an explicit H2O `.clt` input containing the canonical protocol.
5. Run the Psi4-backed CamCASP properties calculation to produce distributed static and dynamic polarizabilities and point-to-point responses.
6. Run ORIENT LW localization at all eleven frequencies.
7. Run PFIT WSM refinement at all eleven frequencies with L3/L3/L3, weight 4, and cutoff `1e-4`.
8. Run CASIMIR through C12.
9. Parse and validate the complete outputs.
10. Write `.camcasp-reference/atomic-polarizabilities.json`.
11. Print copy-ready Python literals for the production pytest module.

The script uses the documented CamCASP `localize.py` workflow rather than bypassing ORIENT or PFIT.

### Local JSON schema

The generated, untracked JSON contains:

- schema version and generation timestamp
- CamCASP, Orient, and Psi4 versions, commits, and executable paths
- checksums of scientific inputs
- geometry, atom labels, atom order, units, functional, basis, IP, kernel, and grid settings
- localization, WSM, hydrogen, weight, and cutoff settings
- exact static and imaginary frequencies
- complete L3 localized and WSM-refined spherical polarizability model at every frequency
- local Cartesian dipole-dipole tensors
- local-to-global rotation matrices
- global Cartesian dipole-dipole tensors
- C6, C8, C10, and C12 isotropic atom-pair matrices
- source output paths and relevant checksums

The JSON exists only for local provenance and diagnostics. Add `.camcasp-reference/` and `orient/` to `.gitignore`. Do not commit the JSON.

### Failure behavior

The script exits nonzero at the first failed stage and retains logs. Error messages identify the failed stage and relevant output path.

It rejects:

- missing or non-executable Psi4
- failed Orient checkout, binary validation, or smoke test
- missing or failed CamCASP executables
- nonzero CamCASP, ORIENT, PFIT, or CASIMIR exits
- absent static or dynamic frequency blocks
- a frequency count other than eleven under the accepted configuration
- unexpected atom labels or order
- incomplete L3 polarizability output
- incomplete C12 output
- non-finite values
- non-orthonormal or left-handed local frames
- nonsymmetric Cartesian dipole-dipole tensors beyond parsing tolerance
- nonsymmetric isotropic atom-pair Cn matrices beyond parsing tolerance

## Psi4 Public Output Contract

The in-process Psi4 implementation exposes these wavefunction array variables.

### Static dipole-dipole polarizabilities

`ATOMIC POLARIZABILITIES`

- shape: `(natom, 6)`
- units: atomic units of dipole polarizability
- frame: global Cartesian
- packed columns: `xx, xy, xz, yy, yz, zz`

### Frequency-dependent dipole-dipole polarizabilities

`ATOMIC DYNAMIC POLARIZABILITIES`

- shape: `(11 * natom, 6)`
- ordering: frequency-major atom blocks
- frequency index 0: static
- frequency indices 1 through 10: increasing CamCASP quadrature points
- packed columns: `xx, xy, xz, yy, yz, zz`

`ATOMIC POLARIZABILITY FREQUENCIES`

- shape: `(11, 1)`
- units: atomic units
- first value: exactly zero
- remaining values: the ten CamCASP/CASIMIR quadrature points

### Dispersion coefficients

Expose four symmetric atom-pair matrices:

- `ATOMIC C6`
- `ATOMIC C8`
- `ATOMIC C10`
- `ATOMIC C12`

Each has shape `(natom, natom)` and contains CASIMIR's isotropic `00 00 0` coefficient in the corresponding atomic units (`hartree * bohr^n`).

The implementation must retain the complete L3 frequency-dependent local model internally because C8, C10, and C12 cannot be reconstructed correctly from only the public dipole-dipole subset.

## Production Pytest Design

Replace the misspelled broad-range test with `tests/pytests/test_atomic_polarizabilities.py`.

A module-scoped fixture performs the fixed PBE0/aug-cc-pVTZ calculation and invokes the atomic-polarizability post-SCF pipeline once. The fixture returns the wavefunction and does not load local JSON.

The test module hard-codes named reference arrays copied from a successful regeneration run. Adjacent comments record:

- CamCASP and Orient versions/commits
- geometry, method, basis, IP, kernel, and frequency protocol
- L3/L3/L3 limits
- LW localization
- PFIT weight and cutoff
- the regeneration command

Implement independent tests:

1. `test_atomic_dipole_dipole_matches_camcasp`
2. `test_dynamic_atomic_dipole_dipole_matches_camcasp`
3. `test_atomic_c6_matches_camcasp`
4. `test_atomic_c8_matches_camcasp`
5. `test_atomic_c10_matches_camcasp`
6. `test_atomic_c12_matches_camcasp`

Each test first checks variable existence, shape, and finiteness, then performs its numerical comparison. Failures identify the property and, where applicable, frequency, atom or atom pair, and Cartesian component.

### Numerical tolerances

- Static and dynamic Cartesian tensors: `rtol=1e-4`, `atol=1e-5`
- Frequency grid: `rtol=1e-10`, `atol=1e-12`
- C6, C8, C10, and C12 matrices: `rtol=1e-4`, `atol=1e-5`

Also test:

- symmetry of each reconstructed Cartesian 3x3 tensor
- symmetry of each Cn atom-pair matrix
- the expected H2O C2 relation between H1 and H2 global tensors
- isotropic traces as diagnostic assertions, never as substitutes for component-wise comparison

No test may use only positivity checks, broad ranges, printed values, or comparisons between mismatched basis sets or localization models.

## TDD Sequence

Tests are ordinary red tests during development, not permanent `xfail` tests.

1. Write the static tensor comparison and confirm that it fails against the incomplete implementation.
2. Implement the minimal faithful response, localization, and WSM behavior needed to pass the static L3 dipole-dipole subset.
3. Add the dynamic tensor and frequency-grid comparison; confirm red, then make green.
4. Add C6 comparison; confirm red, then implement and make green.
5. Add C8 comparison; confirm red, then implement and make green.
6. Add C10 comparison; confirm red, then implement and make green.
7. Add C12 comparison; confirm red, then implement and make green.
8. Run the focused module and relevant Psi4 regression suite. The completed branch requires all six property tests to pass.

A failure in a later property must not weaken or remove an earlier passing assertion.

## Acceptance Criteria

The work is accepted when all of the following are true:

1. `devtools/regenerate-camcasp.sh` can provision or verify the ignored CamCASP and Orient installations and complete the Figure 1 workflow.
2. The script produces validated local JSON containing the complete L3 model and C6 through C12 provenance.
3. Production pytest contains hard-coded references and has no runtime dependency on that JSON or external programs.
4. Psi4's static global Cartesian atomic dipole-dipole tensors match CamCASP ORIENT/PFIT L3 references within the specified tolerance.
5. Psi4's static-plus-ten-frequency dipole-dipole tensors and frequency grid match the references.
6. Psi4's isotropic atom-pair C6, C8, C10, and C12 matrices match CASIMIR.
7. All shapes, coordinate conventions, orderings, units, and symmetry invariants satisfy this document.
8. All focused tests pass without `xfail`, broad numerical ranges, or skipped scientific stages.
