# Mandatory directive: no DFHelper in production dispersion

The user explicitly confirmed that this work must stop using DFHelper for the
production SAPT0 dispersion implementation. This directive supplements
`AGENT_HANDOFF.md`; it does not authorize changing another agent's sources,
environments, jobs or allocations.

## Required production route

**cuEST 0.2.50 DFMO → device-resident metric-transformed MO factors →
cuBLAS contractions and custom CUDA packing/denominator/signed-reduction
kernels, orchestrated from Python.**

* Do not call `psi4.core.DFHelper` for production dispersion integral generation,
  transformations, factor packing or intermediate storage.
* Do not obtain factors indirectly from a DFHelper-produced cache, disk tensor
  or host-side transform. A GPU contraction after a host DFHelper transform
  does not satisfy this directive.
* Do not silently fall back to DFHelper, NumPy factor generation or another CPU
  integral/transform implementation when the cuEST route is unavailable.
* Audit the actual dispersion call chain and factor provenance, not only the
  name of its Python entrypoint or wrapper. Keep device ownership, layout,
  stream ordering and lifetimes explicit.
* The live cuEST handle/basis/plan factory remains unfinished at handoff. If
  that blocks execution, implement and audit it or report BLOCKED; do not hide
  the gap with a host factor provider.

## Correctness/reference exception

An independent existing CPU/native FISAPT reference may use DFHelper solely as
an explicitly labeled, separate correctness oracle. Its factors and caches must
not feed the production GPU calculation. Reference and production measurements
must remain distinguishable.

The existing default CPU/non-HF implementations must not be removed or broken
just to make a source-wide search contain no DFHelper references. The prohibition
applies to the new production dispersion route, including its transitive factor
providers. Historical tests and CPU references are not GPU coverage.

## Unchanged scientific and reporting gates

* Native FP64; both signed Disp20 and Exch-Disp20, occupied-pair/global sum rules,
  changed inputs and multiple block sizes/geometries/bases.
* Exact cuEST0.2.50 identity/loaded CUDA closure; explicit full-Coulomb,
  MATRIXPOWER/relative-cutoff policies and native metric/rank validation.
  Do not double-whiten already metric-transformed DFMO output.
* Demonstrate real DFMO/CUDA/cuBLAS execution. CPU qualifier-shim tests are not
  device or physical correctness evidence.
* Count remaining CPU work, transfers, preparation and cleanup in end-to-end
  timing. Report kernel-only timings separately.
* No greater-than10x speedup claim from tiny cases, theoretical work estimates
  or unmatched CPU/GPU calculations.
* No broader F-SAPT expansion before the signed GPU dispersion gate passes.
