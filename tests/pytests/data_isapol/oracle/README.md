# The CamCASP oracles

A handful of small Fortran programs, each built against your own CamCASP checkout,
produce every reference number in `../`.

## `griddump` — element table and integration grid

`griddump` is a ~50-line Fortran program that links directly against CamCASP's own
`src/atoms.f90` and `src/gdma/atom_grids.F90` and dumps the ISA integration grid and
the element table.  It is what produced `../camcasp_atomprop.dat` and
`../camcasp_grid_h2o.npz`, and it is the reference `libisapol` is held to.

```
CAMCASP=~/gits/CamCASP ./make_oracle.sh          # builds ./oracle-build/griddump
cd oracle-build
./griddump <<'IN'
3
8 110 3 1.0
8  0.0  0.0 -0.12425761
1  0.0 -1.43191407 0.98633103
1  0.0  1.43191407 0.98633103
IN
```

which writes `radii.dat` and `grid.dat`.  The second input line is
`n_r  n_a  k_mu  rscale`; atom lines are `Z x y z` in bohr.

## `freqdump` — imaginary-frequency quadrature

`freqdump` is generated rather than written: `make_freq_oracle.sh` slices the
`rlow`/`wlow` Gauss-Legendre tables and the whole of `SUBROUTINE frequencies` out of
`src/casimir/casimir.f90` and wraps them in a driver, so there is no hand
transliteration between CamCASP and the reference.  It produced
`../camcasp_casimir_freq.dat`.

```
CAMCASP=~/gits/CamCASP ./make_freq_oracle.sh     # builds ./oracle-build/freqdump
echo "0.3 10" | ./oracle-build/freqdump
```

Input is `omega0 n_freq` on stdin; output goes to stdout, one line per frequency,
`k  omega(k)  tm1sq(k)  weight(k)`, with `k = 0` the static point.  `n_freq` must be
even and at most 10 (CamCASP's `MAXF`); odd counts leave a frequency uninitialised
in CamCASP itself, which is why `CasimirGrid` rejects them.

## `latticedump` and `pranddump` — random deviates and the fit-point cloud

Also generated rather than written.  `make_lattice_oracle.sh` lifts the whole of
`src/random.f90`'s `MODULE random`, `MODULE radii`'s double-precision van der Waals
table, and the three numerically relevant fragments of `src/lattice.F90` — the
radii/centre/`dmax` setup, the `RANDOM` draw loop and `subroutine add` — and wraps
them in a driver with stubs for the surrounding CamCASP type system.  Between them
they produced `../camcasp_prand.dat` and `../camcasp_fit_points.npz`; the
reformatting is done by `make_fixtures.py`.

```
CAMCASP=~/gits/CamCASP ./make_lattice_oracle.sh   # builds both into ./oracle-build
echo "1 40" | ./oracle-build/pranddump            # seed, count
./oracle-build/latticedump <<'IN'
1 2000 2.0 4.0
3
8.0  0.0  0.0 -0.12425761
1.0  0.0 -1.43191407 0.98633103
1.0  0.0  1.43191407 0.98633103
IN
./make_fixtures.py                                 # rewrites both fixtures in ../
```

`latticedump`'s first input line is `seed  npoints  lolim  hilim`, the second the
atom count, then one `Z x y z` line per atom in bohr.  It prints the cube centre,
its half-width and every accepted point.

CamCASP's `SET Lattice` block reaches this code with `seed = 1`: `RANDOM nlat` sets
`seed = 0`, and the later `SEED 1` calls `sdprnd(1)` at parse time.  Without a
`SEED` directive the generator self-seeds with 0 on its first `dprand()`.

## `parse_cncode.py` — anisotropic dispersion recoupling coefficients

Not a Fortran program: `src/casimir/c6code.f90 … c12code.f90` are themselves generated
Fortran, so the oracle is a parser rather than a build.  `parse_cncode.py` reduces the
393 `(L1, L2, J)` blocks to their exact root-rational-fraction coefficients and writes
both `../camcasp_recoupling.dat` and the table `libisapol` compiles in.  It asserts as
it goes — order conservation, the triangle rule, that every loop bound is exactly a
rank's component range, and that it consumed every character of every right-hand side
— so a file it cannot account for is an error, not a silent partial parse.

```
CAMCASP=~/gits/CamCASP ./parse_cncode.py
```

It rewrites two files in place:

* `../camcasp_recoupling.dat`, the committed fixture;
* `psi4/src/psi4/libisapol/recoupling_data.inc`, the generated C++ table.

Both are committed, so this only needs re-running if CamCASP's tables change.  See
SPEC.md §9.3 for why the coefficients are transcribed rather than re-derived, and for
the invariants the tests check them against.

## `make_isa_fit_fixture.py` — frozen water ISA-A fitting update

With `p4_ci` active and the staged Psi4 selected:

```bash
eval "$(build_camcasp_psi4_joint/stage/bin/psi4 --psiapi)"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python tests/pytests/data_isapol/oracle/make_isa_fit_fixture.py --camcasp /path/to/CamCASP
```

The script extracts the executable RHS loop from `Chi_rho_w_overlap` and the
metric-modification loop from `make_Stilde_stockholder_A` into a temporary Fortran
adapter, then solves with LAPACK DGESV. The adapter **injects** basis samples and
an analytic weighted overlap; it does not invoke CamCASP basis construction or
molecular density fitting. The input is water RHF/STO-3G AO density, a small
16/110 ISA grid, and a diagnostic unnormalized s/p/s atomic basis. All sampled
inputs are identical between C++ and Fortran. Three cases exercise zero and
activated W-Eps, s-only/all-function weighting, damping and Positive-W.

This establishes the frozen update arithmetic boundary, **not** Drho-C parity,
production AtomAux equivalence, tail/controller correctness, or converged water
properties. No SCF is run by pytest. `camcasp_isa_fit_water.json` records the exact
provenance and SHA256 hashes. Extracted Fortran source and the executable live only
in a temporary directory. The sampling and metric adapters are ours; source loops
are not rewritten in Python. Independent NumPy tests provide a separate algebraic
check and cover cutoff, signed samples, malformed inputs and singular solves.
The same executable-source adapter also generates `camcasp_isa_fit_edges.npz`,
which independently exercises signed sums, nonzero damping at/below the cutoff,
positive old coefficients under auto/non-auto ridge selection, and nonzero
exponent-capped contributions. These synthetic samples are not a physical density
or basis; their purpose is to reach branches the water fixture cannot.

## Why the results are committed and the tool is not run at test time

CamCASP is not redistributable and `atom_grids.F90` is GDMA's, under GPL-2-or-later.
No CamCASP *source* is checked in here: `stubs.f90` and `driver.f90` are ours, and the
`make_*.sh` scripts extract what they need from your own checkout into a scratch build
directory.  The binaries they produce are local development tools and are never linked
into, or shipped with, Psi4.  `parse_cncode.py` is the one exception to "results only":
its output is numerical constants, which ship, with permission, in `libisapol`.

## Build flags matter

Do not add `-ffast-math`, `-march=native`, or `-fdefault-real-8`.  Each one silently
changes the answer, and two of them cost a day each to find:

* `-fdefault-real-8` would promote the element table's literals to double precision.
  CamCASP does not use it, so `R_Slater(O)` is really `0.60000002384185791` Å.
  See SPEC.md §3.5.3.
* `-march=native` lets gfortran contract `r*x + c` into an FMA, which moves grid
  points by 1 ulp relative to a stock CamCASP build.  See SPEC.md §3.5.5.

`freqdump`, `pranddump` and `latticedump` are pure double precision and do no
contractible arithmetic, so they are insensitive to all of the above; the flag
discipline still applies for consistency.

To regenerate the committed fixtures after changing any of this, re-run the commands
above and rebuild the `.npz` — see `../README.md` for its contents.
