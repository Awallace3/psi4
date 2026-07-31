# Task 4 Slice B report: native real-space ISA weights

## Implementation

Implemented the production `compute_isa_weights(shared_ptr<const FrozenResponseContext>, ISAOptions)` factory and kept the existing uniform-weight path explicitly test-only as `ISAWeights::create_test_only`.

The factory:

- evaluates the frozen `Da + Db` AO density in deterministic point/local-AO order on the exact retained response grid;
- constructs a deterministic auxiliary atom-centred shell grid using mapped Gauss-Legendre radial nodes (plus the origin) and a named Gauss-Legendre-polar/uniform-azimuth exact product rule;
- starts every pro-atom from the normalized one-GTO `alpha=1` profile;
- performs simultaneous real-space stockholder updates with pointwise log-sum-exp;
- stores positive log profiles and uses shape-preserving PCHIP interpolation;
- activates continuous, charge-conserving exponential tails and solves their exponents by bracketed bisection;
- uses normalized radial overlap convergence and fails closed at the iteration limit;
- finalizes probabilities on the exact sealed response-grid order, closes each row to unity, checks pointwise density conservation and integrated population conservation, and binds the result to the exact input context pointer;
- validates context/grid/site/basis/density/integration/options/tail/convergence/finiteness conditions;
- records electron counts, iteration/convergence data, populations, residuals, radial profiles/tails, the named grid profile/radius table, and a deterministic context/options digest.

No MBIS, uniform production weights, nearest-atom assignment, or caller-array production fallback was introduced.

## Tests added

`tests/pytests/test_native_isa_weights.py` adds clean synthetic fixtures for:

- exact one-centre unity and population conservation;
- identical two-centre inversion symmetry;
- site-order column equivariance;
- density-scaling invariance and population scaling;
- pointwise finite/nonnegative/unity invariants;
- log-PCHIP shape preservation and exponential-tail continuity;
- fail-closed nonconvergence and nonfinite density parameters.

It also adds a small STO-3G H2O SCF test built from an actual sealed GRAC context. The test checks exact-grid pointwise unity, discrete electron-population conservation, and a three-level auxiliary-grid refinement trend without CamCASP numerical literals.

## TDD and validation

RED was run against the main staged core before implementation:

```text
PYTHONPATH=/home/awallace43/gits/camcasp_psi4/build_camcasp/stage/lib \
  python -m pytest -q .../tests/pytests/test_native_isa_weights.py -x

FAILED test_one_atom_is_exact_unity_and_conserves_population
AttributeError: psi4.core has no attribute _atomic_polarizability_test_isa
```

No build was performed in the isolated worktree, as required. Local non-build checks completed:

- `python -m py_compile tests/pytests/test_native_isa_weights.py`
- `git diff --check`

The parent must build and run the new synthetic and SCF tests. Numerical tolerances/refinement monotonicity remain subject to that first compiled run.
