# Approved supplied-local anisotropic contraction boundary

Status: user selected **Direct contraction** after reviewing the distinction from
CamCASP recoupled coefficients. Implementation, canonical build/staging and binary
identity checks passed; 120 focused tests and 1002 combined ISA/FDDS tests passed.
Independent source review found no must-fix defect. This validates the bounded
supplied-local contraction, not native/localization or recoupled-coefficient parity.

This is an undamped, nonretarded contraction of explicitly supplied reciprocal real
local response tensors. It does not implement localization, native responses,
PFIT-to-site inference, or CamCASP C_n(t,u,J) output. Never extract diagonal blocks
of a distributed response implicitly and call them localized.

## Conventions

Use existing regular real Racah H_lm(x)=r^l C_lm(x/r), no Condon–Shortley phase,
component order 10,11c,11s,... (dipoles z,x,y). Rank lists are explicit subsets
of1–4. All blocks between the declared ranks are supplied; numerical zero is an
explicit model zero, not an absent block. Partial block inputs are unsupported.
Coordinates are bohr; F is a proper local-to-global Cartesian frame. Transform
alpha_global=D(F) alpha_local D(F)^T with the existing harmonic convention.
Require exact symmetry of finite supplied local tensors; do not symmetrize or
require positive definiteness. This strict policy avoids silently changing inputs.

For R=origin_B-origin_A, d_l=(2l-1)!!, and V=Q_A^T T Q_B:

```
T_lm,kn(R) = (-1)^l/(d_l*d_k) H_lm(grad_R) H_kn(grad_R) (1/|R|)
```

Anchors: Cartesian dipole T=(I-3*u*u^T)/R^3, permuted to z,x,y;
on positive z, T_l0,k0=(-1)^k binomial(l+k,l)/R^(l+k+1);
T_AB(R)=T_BA(-R)^T. Coulomb derivatives through degree8 must not be obtained by
passing unsupported rank8 to the rank4 multipole-transform API.

For tau_lm,kn(u)=R^(l+k+1)T_lm,kn(R):

```
C_n = sum_f w_f sum_(a,a',b,b': ranks sum+2=n)
      alpha_A[a,a'] alpha_B[b,b'] tau[a,b] tau[a',b']
E_n = -C_n / R^n
```

Each ordered component quadruple occurs exactly once. Weights already include
1/(2*pi); no extra half, CP factor or off-diagonal multiplicity. Include odd
orders7,9,11 from mixed-rank response blocks. max_order ranges6–12, default12.
Energy is explicitly truncated at max_order; do not claim PSD guarantees for a
truncated expansion. Frequency arrays match exactly, are strictly increasing and
nonnegative; static frequency has zero weight, and some weight must be positive.

## Results and validation requirements

Return orientation-resolved scalar coefficients, not recoupled angular tensors.
All A/B site pairs use actual geometry; reject coincident sites. Preserve model
provenance, labels, ranks/axes, frames, grid/weights, distance and per-order energies.
Report included/missing **ordered** rank quadruples per order and distinguish
completeness within the supplied model from unrestricted ranks>=1 completeness.
Unrestricted C_n can need rank n-5: full rank4 covers general C6–C9, not general
C10–C12. This differs from complete rank4 isotropic C12. Tensor-model completeness
never certifies physical adequacy. Reject malformed dimensions/frames, nonfinite
inputs/intermediates, and overflow. Own input snapshots and return copies.

Independent tests must include dipole tensor closed forms, high-rank axial and
low-rank off-axis interaction signs, a hand-derived mixed-rank nonzero C7,
noncommuting/global rotations and A/B exchange, Lorentz-frequency normalization,
rank-pair Frobenius factors binomial(2l+2k,2l), scalar-block isotropic C6–C12,
independent A/B orientation averages of anisotropic inputs, rank coverage/zeros,
all pairs, static exclusion, ownership and explicit input/overflow rejection.
Existing upstream high-rank tables are not a qualified sole oracle.

Validated coverage limitations: the independent A/B orientation-average test uses
rank1/rank2 A and a diagonal dipole B, whose three-permutation mean is state-specific,
not general SO(3) cubature. High-rank off-axis element validation is indirect
(axial signs, Frobenius identities and covariance); no independent full degree8
reference tensor is claimed. Resource ceilings bound storage, not execution time.
Arithmetic conservatively rejects intermediate values outside double range even
when later cancellation might rescue a representable result; tiny final underflow
is allowed. See `anisotropic_dispersion.h` for the concrete storage ceilings.

Mathematical derivation/scope audit: delegate d8368ffebbf06f8fe90549fcb0c04a2df,
verified answer SHA25669188036c59c6965104354d34e03fca09a5e8adcd314872c2201db73603a1bb9.
