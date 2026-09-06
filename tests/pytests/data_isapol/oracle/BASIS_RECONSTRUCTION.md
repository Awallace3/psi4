# Descriptor reconstruction audit

This is opt-in development validation of **exported** CamCASP representations,
not a native Psi4 basis/DF provider. Read `PRODUCTION_CHECKPOINT.md` for producer
setup and `plan.md` for current measured evidence.

## Checkpoint v2

The reader accepts v1 and v2. The current producer emits v2, extending v1 with:

- complete runtime atomic, molecular-density and selected-atom shape bases:
  representation, site labels/charges/bohr coordinates, primitive exponents,
  effective contraction coefficients and 1-based shell site/l/first/last indices;
- the actual `Rho%D` vector and `Rho%Basis`, not an assumed equivalent AUX;
- the actual molecular density neighbour array, including its trailing zero
  padding (source uses `ANY(site == list)`; zeros match no valid site);
- old shape coefficients and explicit shape-to-AtomAux s-shell map; the raw map
  allocation is retained, while its active prefix is determined by shape basis
  size and unused entries must be zero;
- raw new shape coefficients immediately after s-projection, **before DIIS or
  mixing**. The complete `END` marker now follows this projection.

The atomic descriptor must agree with the original frozen-fit metadata. The map
must be one-to-one and target s shells. Invalid indices, counts, representations,
nonfinite data, incomplete streams and malformed neighbour padding fail explicitly.

Capture occurs immediately after the reference evaluates the actual density.
In this source `evaluate_FuncExpansion` opens `Rho%D` and does not release it;
no extra density calculation or state-changing load is introduced by capture.
Both the old unmodified-source build and earlier v1 artifacts are preserved.

## Independent arithmetic

`reconstruct_isa_basis.py` implements:

- Cartesian GAMINT ordering and angular multiplier
  `sqrt((2*l-1)!! / ((2*i-1)!!*(2*j-1)!!*(2*k-1)!!))` for `x^i*y^j*z^k`;
- real regular solid harmonics derived by differentiating the Legendre polynomial
  and expanding `(x+i*y)^m`, with rational polynomial coefficients accumulated
  before conversion to floating point. No component table is copied from the
  reference. Convention: no Condon–Shortley phase, DALTON component order,
  anomalous spherical p order **x,y,z**;
- contracted shell radial factors `sum_p C_p exp(-alpha_p*r^2)` using exported
  **effective** coefficients, without renormalizing a second time;
- linear molecular density contraction with actual neighbour screening;
- co-centred analytic Gaussian overlap from polynomial products and Gaussian
  moments, with explicit W-Eps integrability and s-block-only treatment;
- an independent per-shell radial normalization identity and explicit shape-map
  consistency / raw coefficient projection checks.

For normalized effective coefficients, each shell's radial normalization is

```
K = (2*l-1)!! * pi^(3/2) / 2^l
    * sum_pq C_p*C_q / (alpha_p+alpha_q)^(l+3/2) = 1
```

Supported angular momenta are **S through G**. Multi-centre analytic overlap is
not implemented by this audit (sampled molecular density is multi-centre).
Cartesian and spherical basis evaluation support contracted shells; the frozen
fit producer still requires primitive atomic bases. Contraction behavior is
covered algebraically, not claimed to be production-contracted-basis parity.

Bare no-tail shape samples can be reconstructed and clipped according to that
branch. **Active exponential tails are not reconstructed from bare Gaussian
coefficients**; the report explicitly excludes that check. Shape projection and
shape-basis mapping remain testable in the active-tail state.

## Run and retain bounded fixtures

With the staged environment selected as in `plan.md`:

```bash
python -P tests/pytests/data_isapol/oracle/reconstruct_isa_basis.py \
  .pi/audit/basis-water-first/isapol-checkpoint.dat \
  --report .pi/audit/basis-water-first/basis-audit.json \
  --sample-count 97 \
  --fixture-output tests/pytests/data_isapol/camcasp_isa_basis_first.json
```

Repeat for activated oxygen and hydrogen. The audit measures 97 deterministic
points spread across the actual neighbour-grid sequence plus the complete atomic
metric; it does **not** claim all 68,310 density/basis samples were reconstructed.
Reports retain selected source indices, absolute and globally scaled errors,
source stream/tool hashes, tolerance and pass/fail. The default scaled threshold
is 1e-10; measured errors must still be inspected independently.

Portable fixtures retain full basis/density/shape descriptors, complete overlap
and coefficient vectors, and only selected basis/density/shape samples. Export is
limited to 513 selected points and happens only after successful audit. These are
**basis-descriptor audit fixtures**, not complete frozen-fit replay fixtures.
Companion `camcasp_isa_basis_evidence.json` supplies producer/protocol provenance.
Unit tests use `load_fixture()` and `audit()` without CamCASP or SciPy.

Source cross-references (local CamCASP, not redistributed):
`types_primary.F90::basis_set`, `basis_operations.F90::norm_contraction_coeffs`,
`basis_functions.F90::evaluate_gtos_{cart,spher}`, and
`function_expansion_operations.F90::evaluate_FuncExpansion_neighbours_1`.
