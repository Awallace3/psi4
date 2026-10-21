# Opt-in general FDDS GPU pseudoinverse candidate

Candidate only until native/GPU/physical gates pass. `CUEST_FDDS_PINV` defaults
false and has no effect unless `USE_CUEST` is true. All four FDDS sites retain
their original `rcond=1e-13`, giving 22 calls for the hybrid ten-point workflow.
No Hermitian assumption, ordinary inverse/solve, rank approximation, changed
quadrature or modified fitting/ALDA policy.

Private `_cuest_fdds_pinv` accepts borrowed finite square C-contiguous FP64
arrays. Empty/rectangular/strided/other dtype/threshold/capacity rejection
returns None before enqueue. Numerical, CUDA and cuSOLVER failures throw,
without a CPU retry. A private owned cuSOLVER handle uses cuEST's stream.
cuBLAS HOST pointer mode is checked, not mutated.

Three O(n²) buffers hold the SVD input/scaled VT, U, and VT/output. Workspace
and arithmetic are checked before allocation against 80% of free device
memory. O(n) singular values go to the host for the exact strict
`s > rcond * s_max` mask; factors remain resident for FP64 reconstruction.
SVD status is checked before reconstruction. Stream drain precedes all
borrowed-owner, host-vector and device-buffer destruction on error.

Input row-major A is column-major Aᵀ. General SVD of Aᵀ, left scaling of VT,
and `scaled_VT.T @ U.T` produce `(Aᵀ)+` column-major, therefore A+ row-major
when downloaded. Input arrays are never modified or packed.

Tests cover nonsymmetric/repeated inputs, rank deficiency, zeros, singular
values below/equal/above cutoff, native/opt-out/threshold/layout fallbacks,
read-only inputs, input freshness/ownership, preflight errors, Python
error-propagation without CPU retry, and hybrid/nonhybrid integrated FDDS.
Actual GPU post-enqueue fault injection and forced low-device-capacity paths
remain explicit runtime-coverage gaps.

The 14250361 timing plan is protein83 only, five retained controls:
CPU0/baseGPU0/candidateGPU0/candidateGPU1/baseGPU1, fresh processes, one H200,
eight threads/96 GiB, same physical cache and verified local XFS. External
runner bytes are frozen at d599720a06 for all arms. No cross-allocation timing
ratio or extrapolated peptide/nanotube result is permitted. A fresh budget
must cover remaining build, gates, all declared controls, cleanup and grace.
