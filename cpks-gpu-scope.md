# Proposed GPU LDA/GGA `compute_Vx`

**Status: design / illustrative diff, not an implemented or compiled port.**
This scope does not change the scientific binary, submit a job, or change the
Protein157 grid benchmark. mGGA, VV10, direct J/K, analytic nuclear Hessians and
the separate SAPT dispersion/FDDS work are excluded.

## 1. cuEST provides the necessary density primitive

Verified against the installed cuEST 0.2.2 header on Phoenix:

`$CONDA_PREFIX/include/cuest/exchange_correlation/xc_potential_compute_api.h`

- L500: `cuestXCNonsymmetricDensityCompute`.
- L546: its workspace query.
- L406: `cuestXCPotentialCompute`.

[NVIDIA's versioned API documentation](https://docs.nvidia.com/cuda/cuest/0.2.2/c-api/methods/exchange-correlation/index.html)
describes separate left/right coefficient arrays, each `numOccupied × nao`,
on the GPU. The GGA density output is point-major
`[rho, grad_x, grad_y, grad_z]`. Rank must be positive.

In Psi4's factor convention, write `D = L R^T`, with `L,R` shaped
`nao × rank`, and transpose each for the cuEST call:

```cpp
// Real SDK entry point. Parameter/workspace RAII omitted here.
CHECK_CUEST(cuestXCNonsymmetricDensityComputeWorkspaceQuery(
    handle, plan, approximation, parameters, &variable_budget,
    &temporary_descriptor, rank, d_LT, d_RT, d_delta_density));
CHECK_CUEST(cuestXCNonsymmetricDensityCompute(
    handle, plan, approximation, parameters, &variable_budget,
    temporary_workspace, rank, d_LT, d_RT, d_delta_density));
```

The intended bilinear contraction is

```text
delta_rho(r)   = sum_i L_i(r) R_i(r)
grad_delta_rho = sum_i [grad L_i(r) R_i(r) + L_i(r) grad R_i(r)]
```

There is no positive-definite factorization in this construction: changing
`R -> -R` changes the sign, and `L != R` represents a nonsymmetric AO density.
The documentation explicitly supports nonsymmetric factors; signed behavior
and exact normalization still need a small runtime SDK test. Do not present
header inspection as a completed numerical validation.

**Never evaluate LibXC on `delta_rho` as though it were a physical density.**
Evaluate its first/second derivatives at the physical SCF reference, and
contract those derivatives with the signed response fields. Do not clip
negative response values or replace them by their absolute value.

For real scalar LDA/GGA, only the symmetric part of `D` affects XC:
`rho[D] == rho[D^T]`; a purely antisymmetric perturbation gives zero XC
response. This does not justify symmetrizing or otherwise changing the
nonsymmetric density passed to the separate J/K response.

`cuestXCPotentialCompute` integrates user-supplied grid fields back to the AO
basis; the caller applies the total integration weights **exactly once**.
Its GGA documentation has a factor-four convention. Establish the exact
adapter with a direct AO quadrature probe, rather than transplanting a
spin-summed formula blindly.

`cuestXCNonsymmetricDerivativeCompute` is a basis/grid (nuclear derivative)
routine, **not** the electronic response kernel needed here.

## 2. Native behavior that must remain unchanged

Source locations refer to the current, dirty worktree, not to a claim that
these lines are identical to the PR's committed HEAD:

- `psi4/src/psi4/libfock/v.cc:3023`: `RV::compute_Vx_full`.
- `psi4/src/psi4/libfock/v.cc:5743`: `UV::compute_Vx`.
- `psi4/src/psi4/libfock/v.h`: public response normalization and spin ordering.
- `psi4/src/psi4/libfunctional/superfunctional.cc:660`: spin-adapted triplet
  derivative combinations; `:735`: GRAC is applied only for `deriv == 1`.
- `psi4/src/psi4/libfunctional/LibXCfunctional.cc:734` onward: native
  unpolarized second derivatives; `:899` onward: polarized layouts.
- `psi4/src/psi4/libscf_solver/rhf.cc:488–524`: existing response factors.
- `psi4/src/psi4/libscf_solver/uhf.cc:575–650`: same, including spin reordering.

Requirements:

1. RKS singlet **and triplet**, UKS alpha/beta cross-coupling, LDA and GGA.
2. Arbitrary real dense response matrices, signed/nonsymmetric inputs, batches,
   zero RHSs, AO and symmetry-blocked representations.
3. Preserve RKS's half-alpha-output convention, not the ordinary SCF potential
   normalization. Preserve UKS interleaving `[a0,b0,a1,b1,...]`.
4. Screen using the **reference** density and `v2_rho_cutoff_`, not the signed
   response. RKS uses total reference rho; UKS uses `rho_a + rho_b`.
   Match the native strict/non-strict comparisons at the cutoff, including
   UKS's slightly different LDA/GGA equality behavior.
5. Native GRAC response uses the underlying semilocal derivative-2 kernel,
   **not** a newly differentiated GRAC interpolation. Keep native exclusions,
   including **UKS GRAC and RKS-triplet GRAC**. The native triplet path clones
   a polarized superfunctional, retains GRAC, and throws during allocation
   (`superfunctional.cc:115–155,635–645,660–666`). RKS-singlet GRAC uses the
   uncorrected semilocal kernel.
6. Preserve component weights, range parameters, density thresholds and user
   tweaks in host/device and polarized clones. Hybrid exact exchange remains
   in the existing J/K response; do not count it again in this kernel.
7. Keep explicit unsupported errors for mGGA, VV10 and missing required
   functional derivatives. Do not fall into the empty CPU-grid block loop.

### RKS packing, including the non-obvious factors

Let `n,g` be the total reference density and gradient (twice the alpha
reference quantities for closed-shell orbitals). Let `u,h` be the density
and gradient from the nonsymmetric response `Dx`, **without doubling it**.
For the singlet, evaluate unpolarized LibXC at `(n, gamma = g.g)` and retain

```text
A = f_rho,rho     B = f_rho,gamma
C = f_gamma,gamma
S = f_gamma
t = 2 g.h
```

The native AO result is represented by the following weighted coefficients:

```text
p0 = w [A u + B t]
p  = 2 w [S h + (B u + C t) g]

Vx_mu,nu = sum_points {
    p0 phi_mu phi_nu
  + p . [grad(phi_mu) phi_nu + phi_mu grad(phi_nu)]
}
```

This follows from the CPU intermediate `T = 0.5*p0*phi + p.grad(phi)`,
followed by `phi^T T + T^T phi`. It is not an invitation to add another
factor of two after the cuEST AO integration.

For a triplet, evaluate a polarized functional at the closed-shell
reference, then reproduce `SuperFunctional::compute_functional`:

```text
A = (f_rhoa,rhoa - f_rhoa,rhob) / 2
B = (f_rhoa,sigmaaa - f_rhoa,sigmabb) / 4
C = (f_sigmaaa,sigmaaa - f_sigmaaa,sigmabb) / 8
S = f_sigmaaa / 2 - f_sigmaab / 4
```

Use the same `p0,p` equations. An independent oracle is **half** the UKS
alpha response to `(delta_a,delta_b)=(Dx,+Dx)` for the singlet, or
`(Dx,-Dx)` for the triplet, at a closed-shell reference. A sign change of the
singlet kernel is not a triplet implementation.

### UKS packing

At each grid point define

```text
x = [rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb]
z = [u_a, u_b,
     2 g_a.h_a,
     h_a.g_b + g_a.h_b,
     2 g_b.h_b]

delta_v = H_f(x) z

p0_a = w delta_v_rhoa
p_a  = w [2 v_sigmaaa h_a + v_sigmaab h_b
          + 2 delta_v_sigmaaa g_a + delta_v_sigmaab g_b]

p0_b = w delta_v_rhob
p_b  = w [2 v_sigmabb h_b + v_sigmaab h_a
          + 2 delta_v_sigmabb g_b + delta_v_sigmaab g_a]
```

`H_f` is a symmetric `5 × 5` Hessian. LibXC stores its blocks as:

```text
v2rho2:     [a_a, a_b, b_b]                         (3)
v2rhosigma: [a_aa, a_ab, a_bb, b_aa, b_ab, b_bb]     (6)
v2sigma2:   [aa_aa, aa_ab, aa_bb, ab_ab, ab_bb, bb_bb] (6)
```

The first derivatives `v_sigma` are still necessary: `fxc` alone misses
the terms proportional to `h_a,h_b`. A zero alpha perturbation can have a
nonzero alpha output through beta coupling, and conversely.

At exact UKS cutoff equality, split the scalar action into two pieces:
mask `H_rho,rho * delta_rho` with `rho_a + rho_b > cutoff`, and mask the
GGA scalar additions and gradient action with `rho_a + rho_b >= cutoff`.
A single mask on the complete `H_f * z` cannot reproduce native equality
behavior. The compact equations above describe the unmasked action.

## 3. Proposed diff: one response implementation behind VBase

**The following hunks are interface/algorithm sketches, not an apply-ready
patch.** New helper names do not imply an existing cuEST API.

```diff
--- a/psi4/src/psi4/libfock/v.h
+++ b/psi4/src/psi4/libfock/v.h
@@ VBase protected cuEST state
+    // Own buffers/parameters with the V object, not process-global mutable state.
+    std::unique_ptr<CuestXCResponse> cuest_response_;
+
+    // Common LDA/GGA validation, reference setup, AO/SO conversion and dispatch.
+    void compute_Vx_cuest(const ResponseBatch& input,
+                          std::vector<SharedMatrix> ret,
+                          ResponseSpin channel);

@@ VBase public response interface
+    // An additional factored entry point. Existing dense entry points stay.
+    // Each pair means Dx = left * right^T, in the declared representation.
+    // CPU default materializes Dx and dispatches the existing native method.
+    virtual void compute_Vx_factored(const ResponseFactors& input,
+                                    std::vector<SharedMatrix> ret);

@@ RV public response interface
+    void compute_Vx_factored(const ResponseFactors& input,
+                            std::vector<SharedMatrix> ret) override;
+    void compute_Vx_full_factored(const ResponseFactors& input,
+                                 std::vector<SharedMatrix> ret, bool singlet);

@@ UV public response interface
+    void compute_Vx_factored(const ResponseFactors& input,
+                            std::vector<SharedMatrix> ret) override;

--- a/psi4/src/psi4/libfock/v.cc
+++ b/psi4/src/psi4/libfock/v.cc
@@ RV::compute_Vx_full
+    // Validate size, representation, reference and LDA/GGA support first.
+#ifdef USING_cuEST
+    if (use_cuest_xc()) {
+        return compute_Vx_cuest(ResponseBatch::dense(Dx), ret,
+            singlet ? ResponseSpin::RKS_Singlet : ResponseSpin::RKS_Triplet);
+    }
+#endif
     throw_if_cuest_unsupported(/* existing diagnostic */);
     // Existing CPU implementation remains intact.

@@ UV::compute_Vx
+#ifdef USING_cuEST
+    if (use_cuest_xc()) {
+        return compute_Vx_cuest(ResponseBatch::dense(Dx), ret, ResponseSpin::UKS);
+    }
+#endif
     throw_if_cuest_unsupported(/* existing diagnostic */);
     // Existing CPU implementation remains intact.

@@ VBase::finalize_impl
+    // Release response buffers/parameters before destroying their plan/grid.
+    cuest_response_.reset();
```

The factored dispatch must actually reach the new GPU path:

```cpp
// Proposed definitions; materialize() means the exact existing matrix product.
void RV::compute_Vx_factored(const ResponseFactors& in, Outputs ret) {
    compute_Vx_full_factored(in, ret, true);
}
void RV::compute_Vx_full_factored(const ResponseFactors& in,
                                 Outputs ret, bool singlet) {
    if (use_cuest_xc()) {
        compute_Vx_cuest(ResponseBatch::factored(in), ret,
            singlet ? ResponseSpin::RKS_Singlet : ResponseSpin::RKS_Triplet);
    } else {
        compute_Vx_full(materialize(in), ret, singlet);
    }
}
void UV::compute_Vx_factored(const ResponseFactors& in, Outputs ret) {
    if (use_cuest_xc()) {
        compute_Vx_cuest(ResponseBatch::factored(in), ret, ResponseSpin::UKS);
    } else {
        compute_Vx(materialize(in), ret);
    }
}
```

These definitions also need the usual cuEST compile guards; the snippets
omit those for readability. The base default calls dense `compute_Vx`, not
itself, while RV's full-factored CPU branch preserves the triplet argument.

The private implementation belongs in new
`psi4/src/psi4/libfock/cuest_xc_response.{h,cc}`. Keep device algebra in the
existing `cuest_xc_gpu.{h,cu}`; keep native CPU response unchanged.

```cpp
// Proposed implementation outline in cuest_xc_response.cc.
// Each compute_* wrapper owns/checks the corresponding SDK parameters,
// WorkspaceQuery, bounded variable budget, temporary workspace and status.
void CuestXCResponse::apply(const ResponseBatch& batch, Outputs& outputs) {
    validate_lda_gga_and_derivative_capabilities();
    validate_reference_and_final_grid();

    // Values shared across this RHS batch, rebuilt on every public invocation
    // initially. This deliberately avoids a stale cross-call fxc cache.
    compute_total_integration_weights();
    compute_reference_density_from_D_or_matching_occupied_factors();
    evaluate_reference_vsigma_and_fxc();  // host LibXC OR CUDA LibXC
    project_rks_spin_channel_if_needed();

    for (const auto& rhs : batch.in_public_spin_order()) {
        compute_signed_transition_density(rhs);  // nonsymmetric SDK API
        contract_and_pack_response(rhs);         // equations above; weights once
        integrate_response_potential(rhs);       // cuestXCPotentialCompute
        copy_result_and_restore_requested_symmetry(rhs, outputs);
    }
}
```

Reference preparation must work when a caller has supplied only `set_D`,
not `set_Cocc`. Native response accepts that. Do not require occupied
coefficients to exist, or use stale ones from a previous SCF. Dense reference
density contraction is a correct fallback; use occupied factors only when
their consistency with the current reference is established.

### Reuse the factors that the solver already has

```diff
--- a/psi4/src/psi4/libscf_solver/rhf.cc
+++ b/psi4/src/psi4/libscf_solver/rhf.cc
@@ RHF::twoel_Hx_full: XC contribution
-    // Materialize Dx[i] = Cl[i] * Cr[i]^T, then compute_Vx_full.
+    // Reuse these factors; Cr already contains the response minus sign.
+    // Allocate outputs from factor/basis dimensions, not by forming dense Dx.
+    potential_->compute_Vx_full_factored(
+        ResponseFactors::paired(Cl, Cr), Vx, singlet);

--- a/psi4/src/psi4/libscf_solver/uhf.cc
+++ b/psi4/src/psi4/libscf_solver/uhf.cc
@@ UHF::twoel_Hx: XC contribution
-    // Materialize Dx_a, Dx_b and interleave the dense matrices.
+    // JK stores all alpha factors, then all beta factors.
+    // The XC adapter must explicitly reorder them to a0,b0,a1,b1,...
+    potential_->compute_Vx_factored(
+        ResponseFactors::interleave_spins(Cl, Cr, nvecs), Vx);
```

No solver-vector layout, hybrid J/K scaling, direct J/K selection, or
mathematical definition of the Hessian action changes.
Factor views are borrowed only for the duration of the call: the solver clears
its J/K factor vectors afterward. Complete or explicitly synchronize dependent
device reads before releasing/reusing those buffers; never retain borrowed
host pointers in the response workspace.

For symmetry-blocked factors, reuse the correct irrep pairing/permutation
from the J/K AO transformation. Applying a square-density symmetry-removal
routine independently to rectangular factors is not sufficient.
An initial optimized factor path can be C1-only, provided non-C1 inputs
still reach the **GPU-grid dense adapter**, not an empty CPU-grid fallback.
Outputs must have the dimensions and perturbation irrep of the existing
`Matrix::doublet(left,right,false,true)` result, derived without materializing
it. Validate compatible alpha/beta representations and perturbation symmetry.
The AO result is real symmetric; its SO representation may carry a nonzero
perturbation irrep and must not be forced into the totally symmetric irrep.

There is an existing native UKS batch-output indexing issue at
`v.cc:6240–6247`: it tests `Dx[i]->nirrep()` while writing outputs
`ret[2*i],ret[2*i+1]`. Use the actual pair metadata in the GPU adapter.
Mixed-representation native batches need a targeted native fix or a
per-pair oracle before they can serve as a parity baseline; do not reproduce
this bug in the new implementation.

### Preserve arbitrary dense `compute_Vx` callers

The SDK API takes factors, not an arbitrary dense response matrix. Avoid
Cholesky, squaring coefficients, or an SVD in every iterative matvec.
An exact bounded-memory fallback is a column-identity decomposition:

```text
D = sum_tiles L_tile R_tile^T
L_tile = columns of the AO identity selected by this tile
R_tile[nu,i] = D[mu_i,nu]
```

Compute each tile's signed grid density and sum it before applying the
kernel. This preserves arbitrary dense inputs without truncation, but costs
more contractions than the occupied-rank solver path. Treat it as a
compatibility path, not the performance route. Skip zero-rank factors rather
than passing rank zero to the SDK.

## 4. Functional adapters and memory

CUDA LibXC uses `xc_functional_device()` with the existing explicit
`XC_FLAGS_ON_DEVICE` initialization. Host LibXC uses host arrays on the
**same cuEST grid**, not a second independently pruned CPU grid.

- LDA: obtain `xc_lda_fxc`.
- GGA: obtain `v_sigma` using the first-derivative API, and the three Hessian
  blocks using `xc_gga_fxc`. A combined first/second-derivative API can replace
  the two calls after capability checks.
- Polarized derivative layouts and RKS triplet clones must match native
  `LibXCFunctional::build_polarized`, including tweaks/omega/cutoffs.
- Check derivative capabilities and finite values. Do not copy SCF's
  component-skipping behavior and silently turn an invalid response into zero.
- A CUDA-enabled library's symbols/headers are not proof that each requested
  functional/device derivative actually works: require device smoke tests.

The installed CUDA LibXC 7.1.2 header declares, and `libxc.so.15` exports,
`xc_lda_fxc`, `xc_gga_fxc`, `xc_lda_vxc_fxc` and `xc_gga_vxc_fxc`.
The GGA combined call can supply `v_sigma` and all required Hessian blocks
without requesting an XC energy. This is symbol/header verification, not
an executed CUDA derivative test.

The host-LibXC build currently does not compile `cuest_xc_gpu.cu`:
`psi4/src/psi4/libfock/CMakeLists.txt:30`. Support host functional evaluation
and packing in C++, with the necessary transfers, without introducing an
unconditional nvcc requirement. The CUDA-LibXC build uses CUDA packing kernels.
Compile-check for the nonsymmetric XC API (or require an appropriate SDK
version for this feature); the current `find_package(cuEST 0.2 ...)` alone
does not establish that every accepted installation exports these symbols.
Older SDKs should retain the explicit unsupported diagnostic, not a link error
or a silently different CPU-grid result.

**Start with one RHS (one alpha/beta pair for UKS) on the device at a time.**
Reuse allocation capacity, but initially cache numerical reference/derivative
values only within a single public invocation. A later cross-matvec cache
needs a real immutable response epoch or complete configuration/reference
revision tracking; pointer identity or unchanged grid dimensions is not enough.

Illustrative minimum GGA storage, in doubles per grid point:

| Route | Reference, weights, accumulated derivatives | One RHS's density and packed potential |
|---|---:|---:|
| RKS singlet | 9 | 8 |
| UKS | 27 | 16 |

These exclude per-component LibXC scratch, SDK workspace, AO outputs,
coefficient buffers, transient triplet polarized derivatives and allocation
overhead. Also budget host mirrors, retained buffer capacities, caller-owned
outputs for the entire batch and simultaneously live SCF/response plans:
one-RHS device processing does not bound total host/batch memory.
Stream components and immediately project triplet derivatives.
Query and report SDK workspace; do not blindly reserve a fresh 2 GB inside
every RHS. cuEST documents additional allocations outside its variable budget.

The proposed implementation is `O(Ngrid)` in grid fields, not
`O(Ngrid * NBF * Nrhs)`. Dense output is still `O(NBF^2)`, and the large DF
J/K plans remain. Smaller grids reduce XC fields and work, not the dominant
basis-dependent DF storage. Whole-grid SDK outputs remain whole-grid outputs:
do not promise arbitrary point streaming through an API with no point-offset
argument.

Finalization releases the response workspace before its grid/plan. Grid
switches, changed reference/spin/functional settings and failed setup cannot
leave valid-looking cached values behind. Response is prepared on the selected
**final grid**, not on the SCF initial grid.

## 5. Implementation sequence and acceptance gates

1. **SDK contract probe:** ordinary versus nonsymmetric `L=R`, `R=-L`,
   `L != R`, transpose invariance, zero/empty handling, direct AO reference
   values/gradients, scalar and GGA potential integration with exactly-once
   weights. Also probe each required CUDA LibXC derivative entry point.
2. **Dense public response:** RKS LDA/GGA singlet, then triplet, then UKS.
   Independent host/CUDA functional adapters share the same packing equations.
   Keep mGGA/VV10 explicit errors.
3. **Factored solver path:** reuse existing J/K factors, maintain UKS ordering
   and non-C1 fallback, retain dense public calls for other response users.
4. **Numerics before speed:** match native kernels and full iterative solver
   residuals; then benchmark setup, repeated actions and complete solves.

Required tests extend `tests/pytests/test_cuest_cphf.py`:

- Build an actual GPU-grid V object with `CUEST_XC=true` from initialization.
  Existing GPU J/K response comparisons retain CPU XC, and the current
  GPU-XC response test checks the unsupported guard. Toggling an option after
  building a CPU grid is not a GPU response test.
- Fixed-orbital native-versus-GPU actions: LDA/GGA, RKS singlet/triplet and UKS,
  pure, global-hybrid and range-separated cases.
- Dense-versus-factored actions; signed, nonsymmetric, antisymmetric, zero and
  mixed-RHS batches; batch permutations and shrinking active batches.
- UKS cross-spin-only RHSs and an empty occupied spin. A zero input for one
  spin must not erase a nonzero cross-spin output.
- AO/SO representations, nontrivial symmetry, **nonzero perturbation irreps**
  and compatible output symmetry; explicit RKS-triplet/UKS GRAC rejection.
- Central differences of native uncorrected semilocal potentials with the
  documented RKS normalization; signed linearity and adjoint consistency.
  Do **not** use the derivative of a GRAC-corrected potential as the oracle
  for the native semilocal GRAC response convention.
- Exact cutoff boundary behavior, unsupported derivative flags, nonfinite
  derivative errors, workspace failures and exception-safe finalization.
- Repeated calls after reference changes, functional parameter changes,
  final-grid changes and independent V objects; no stale reference fields.
- Completed CPKS solver residuals/polarizabilities and a small SAPT induction
  calculation on CPU/GPU, not an `induction=NONE` timing proxy.
- CPU-only builds/runs remain usable without a device; host and CUDA LibXC
  configurations tested separately.

Suggested kernel-action goal is approximately `1e-9` absolute / `1e-8`
relative in native FP64 on identical grids, calibrated against the actual
CPU/quadrature oracle rather than treated as a universal tolerance.
The earlier `1e-5 Eh` sparse-grid **energy** criterion is not a response-kernel
or solver-residual acceptance threshold.

Instrument disjoint response stages: reference density, first derivatives,
`fxc`, signed response density, Hessian contraction/packing, AO integration,
transfers and symmetry transforms. Report cold setup and repeated actions
separately, plus total solve and iteration counts; CUDA completion must be
included when timing asynchronous calls. Add measured response bars to the
report only after correctness gates pass.

### Checks performed for this scope

- Installed cuEST header inspected; versioned NVIDIA contracts checked.
- Installed CUDA LibXC derivative declarations and shared-library exports
  inspected; device execution not performed.
- Native spin adaptation, density contractions and first/second derivative
  layouts inspected.
- Independent NumPy algebra checks (toy Hessian, **not cuEST/LibXC tests**):
  factor versus dense rho/gradient errors below `2.2e-14`;
  UKS packed-field centered-difference error `6.7e-10`;
  native symmetrized-T versus packed AO error below `1.5e-14`;
  RKS triplet projection versus half UKS-alpha error below `4.5e-16`.
- No compiled GPU response implementation or response speedup is claimed.
- Two read-only source reviews agreed on the RKS/UKS algebra and identified
  the native GRAC-triplet exclusion, cutoff-equality masks and UKS output
  indexing issue addressed above.
