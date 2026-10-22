# FDDS integral-production replacement

## Separate frozen run and development

Job14302772 tests science3a7820d7a4, not this refactor. Its original source,
controller, build and outputs must remain unchanged. Supplier preparation
source62f79dbea3 is a separate local candidate. Its fresh CPU build passed;
the broad native gate is **not green** (results below). No new GPU integral
producer, residency or speedup is established.

## Implemented seam

`libsapt_solver/fdds_tensor_provider.{h,cc}` hides integral preparation and
storage behind logical shape, first-axis read, derived create and write.
The native adapter owns the existing DFHelper. FDDS contraction, QR and
exchange consumers no longer register spaces or manage two integral rounds.

* Tensor storage is FP64, contiguous, last index fastest; slices are half-open.
* Registered integral kinds distinguish raw Coulomb and metric(-1/2) factors.
  Derived QR/X/Y storage cannot overwrite integral names.
* Name, slice, capacity and byte-overflow checks precede backend transfers.
  Zero-length transfers perform no backend I/O.
* Matrix reads retain the existing three-dimensional NumPy view metadata and
  leave padded backing-buffer tails alone.
* No full tensor is copied into a second registry/cache. Only shape/kind/name
  metadata is duplicated; the native disk/core policy is unchanged.
* Source ownership and scratch cleanup follow the provider's lifetime.
  No backend exception is converted into a retry.

The native adapter keeps raw `DIRECT`/`DIRECT_iaQ`, metric power0, the hybrid
`DIRECT_iaQ` metric power-0.5 round, existing memory in doubles, thread count,
metric-fold routing and pre-metric/core-release flags. Both rounds' outputs
survive clearing the transformation registrations. Native production remains
the default in CPU and cuEST builds.

**This is a host-buffer interface**, not a device-pointer lease. A later
device adapter must not be described as resident merely because its integral
kernel executes on a GPU.

## Why fitted DFMO cannot be relabeled raw

The canonical0.2.50
`density_fitted_mo_integrals_compute_api.h` explicitly states that its output
is transformed by the fitting metric. Its plan supports QR or matrix-power
fitting, relative/absolute conditioning and a fitting cutoff; no raw
metric-power0 output control was found in those public headers.

FDDS's raw tensors are needed alongside the original auxiliary Coulomb
metric, overlap, QR factors and contact projection. If a fitted factor is
`F = B J^(-1/2)` with a thresholded spectral inverse, then
`F J^(1/2) = B P_retained`, not generally `B`. Reconstructing raw tensors this
way can discard components and change the auxiliary convention. Gram parity
alone does not prove FDDS tensor parity.

Consequently the next cuEST adapter must first prove the exact native
metric(-1/2) factor/gauge/rank contract. A fitted-only producer paired with a
native raw producer is a **mixed producer**, not DFHelper-free FDDS. Full
replacement needs a verified raw Coulomb endpoint or an independently
validated raw integral kernel. Neither a different fitting algorithm nor a
lower cutoff is an implicit workaround.

## Contact projection is a different operator

`project_densities()` first contracts Coulomb three-center AO integrals with
the density, applies the existing Coulomb metric inverse, and then contracts
the three-center **contact overlap**
`T[P,Q,R] = integral chi_P(r) chi_Q(r) chi_R(r) dr`.
The last stage uses Libint's delta operator and primitive precision0, not
Coulomb DFMO and not two-center overlap. No matching contact endpoint was
identified in the canonical public header inventory.

A custom Gaussian contact kernel could fuse the R contraction directly into
the P,Q output, avoiding a materialized auxiliary cube. Before integration it
needs primitive/contraction normalization, Cartesian and pure-shell ordering,
shell permutations, general angular momenta, symmetric accumulation, the
existing factor2 and the no-primitive-screening contract verified against
Libint. Such a kernel is not part of the initial native supplier refactor.

## Gates

`devtools/test_fdds_tensor_provider.py` compiles the actual interface against
an explicit in-memory host adapter: layouts/tails, names, capacity, overflow,
empty transfers, immutable source inputs and repeated changing writes.
This model does not execute the native storage implementation or CUDA.

`tests/pytests/test_fdds_tensor_provider.py` checks actual small physical
raw/fitted tensors against independent Mints AO integrals and NumPy metric
powers, including unequal monomers, pqQ/Qpq permutations, prefixes and raw
survival across the fitting round. Existing blocked exchange, response,
empty/memory, projection and native-XC SAPT tests remain required.

The new native adapter must compile and pass those gates before extension.
Future cuEST production additionally needs exact SDK/runtime admission,
actual DFMO calls and mapped output proof, rank/cutoff and basis convention
tests, bounded workspace/stream completion/fault cleanup, then unchanged
integrated energy gates. The response pinv cutoff stays1e-13; DF fitting
condition stays1e-12. These are different contracts.

## Local evidence and unresolved gate

Fresh CPU-only core SHA256:
`82d7aef688643fe3961b16659588c2cc34ae8b5804d434b0a0265ec2dc28cf77`.
Import and linked-library provenance, XML and full logs are retained under
`build_fdds_provider_native_20261022/native-gate-v1/`.

The broad gate executed184 tests:168 passed,16 failed, zero errors/skips.
Its passing subsets were64 FDDS,4 independent supplier tensor oracles,
8 projection,51 CPHF,6 lifecycle and35 SCF options. Each of the four tensor
oracles reported maximum absolute error8.881784197001252e-15 a.u.

All16 failures are the existing switched-SCF CPHF assertion comparing final
grid point counts (66186 versus66187, `test_cuest_cphf.py:1073`).
This is unresolved, not waived or established as pre-existing; the full gate
must not be reported as a pass. Historical and isolated controls are being
used to distinguish baseline/runtime/test-order behavior from a regression.

Two historical-oracle setup attempts are also retained, not numerical
failures: the first encountered unset `PYTHONPATH` under Bash nounset; the
second used the current fixture with a historical core that has no
`CUEST_FDDS_PINV` option (four setup errors, zero comparisons). The explicit
`fdds_historical_supplier_oracle.py` compatibility driver removes only that
disabled, unused option for historical comparison; it is forbidden on current
cores and is not a candidate admission gate.

That explicit historical compatibility run passed all four tensor oracles
(zero errors/failures/skips,0.581s), maximum error9.152401059253634e-15 a.u.
Its retained receipt records eight removals (setup/teardown for four cases)
and historical core SHA256
`c08b6341ce7b29099225fc83fde6dec336c6bc3e38caba0ca5840b7c6722d23f`.
Evidence: `build_fdds_cpu/provider-oracle-20261022-v3/`. This establishes that
the same independent tensor oracle accepts the old supplier and the candidate;
it does not resolve the separate CPHF gate or establish a GPU producer.
