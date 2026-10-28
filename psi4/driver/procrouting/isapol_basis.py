# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Explicit basis recipes and verified MAIN adaptation, independent of ISA iteration.

Centres are bohr; shell coefficients include normalization. MAIN conversion
preserves the occupied orbitals without fitting density or repairing normalization.
"""
from dataclasses import dataclass
import functools
import math
import numpy as np
from psi4 import core


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must explicitly declare a nonempty name/origin')



def _finite(value, name, minimum=None):
    if not np.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f'Invalid {name}')



def _owned(value):
    a = np.array(value, dtype=float, copy=True)
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite numerical snapshot')
    # Immutable backing buffer, not merely a reversible writeable flag.
    return np.frombuffer(a.tobytes(), dtype=float).reshape(a.shape)



@dataclass(frozen=True)
class ShellRecipe:
    centre: int
    l: int
    exponents: tuple
    coefficients: tuple

    def __post_init__(self):
        object.__setattr__(self, 'exponents', tuple(self.exponents))
        object.__setattr__(self, 'coefficients', tuple(self.coefficients))
        if type(self.centre) is not int or self.centre < 0 or type(self.l) is not int or self.l not in range(5):
            raise ValueError('Shell centre/rank invalid (S-G only)')
        if not self.exponents or len(self.exponents) != len(self.coefficients):
            raise ValueError('Shell primitive arrays must match')
        if not np.isfinite(self.exponents).all() or min(self.exponents) <= 0:
            raise ValueError('Shell exponents must be positive finite')
        if not np.isfinite(self.coefficients).all() or not any(self.coefficients):
            raise ValueError('Shell effective coefficients must be finite and nonzero')



@dataclass(frozen=True)
class BasisRecipe:
    name: str
    origin: str
    representation: str
    centres: tuple
    shells: tuple

    def __post_init__(self):
        _text(self.name, 'basis name')
        _text(self.origin, 'basis origin')
        object.__setattr__(self, 'centres', tuple(tuple(float(x) for x in c) for c in self.centres))
        object.__setattr__(self, 'shells', tuple(self.shells))
        if self.representation not in ('Cartesian', 'Spherical'):
            raise ValueError('Explicit Cartesian GAMINT or spherical DALTON required')
        c = np.asarray(self.centres)
        if c.ndim != 2 or c.shape[1] != 3 or not len(c) or not np.isfinite(c).all():
            raise ValueError('Centres must be finite bohr triples')
        if not self.shells or any(not isinstance(s, ShellRecipe) or s.centre >= len(c) for s in self.shells):
            raise ValueError('Invalid basis shells')

    def build(self, role):
        shells = []
        for r in self.shells:
            s = core.IsaGaussianShell()
            s.centre, s.l = r.centre, r.l
            s.exponents, s.coefficients = r.exponents, r.coefficients
            shells.append(s)
        return core.IsaExplicitBasis(getattr(core.IsaBasisRole, role),
                                    getattr(core.IsaBasisRepresentation, self.representation),
                                    self.centres, shells)



@dataclass(frozen=True)
class ShellTransformDiagnostic:
    shell: int
    rank: int
    condition: float
    training_residual: float
    heldout_residual: float
    overlap_residual: float



@dataclass(frozen=True)
class MainAdaptation:
    recipe: BasisRecipe
    basis: object
    transform: np.ndarray
    occupied: np.ndarray
    diagnostics: tuple
    orthonormality_residual: float
    global_overlap_residual: float
    transformed_orthonormality_residual: float
    method: str = 'deterministic_shell_collocation_svd_no_truncation'



def _psi_samples(basis, points):
    """Unscreened bounded BasisFunctions sampling; never a density provider."""
    points = np.asarray(points, dtype=float)
    vectors = [core.Vector.from_array(np.ascontiguousarray(points[:, j])) for j in range(3)]
    extents = core.BasisExtents(basis, 0.0)
    block = core.BlockOPoints(*vectors, core.Vector.from_array(np.ones(len(points))), extents)
    evaluator = core.BasisFunctions(basis, len(points), basis.nbf())
    evaluator.set_deriv(0)
    evaluator.compute_functions(block)
    local = list(block.functions_local_to_global())
    result = np.zeros((len(points), basis.nbf()))
    result[:, local] = np.asarray(evaluator.basis_values()['PHI'])[:len(points), :len(local)]
    if len(local) != basis.nbf() or not np.isfinite(result).all():
        raise ValueError('Unscreened MAIN sampler did not return every finite AO')
    return result



def _directions(n, phase):
    i = np.arange(n) + .5
    z = 1 - 2 * i/n
    angle = i * (math.pi * (3-math.sqrt(5))) + phase
    return np.column_stack((np.sqrt(1-z*z)*np.cos(angle), np.sqrt(1-z*z)*np.sin(angle), z))



@functools.lru_cache(maxsize=None)
def _dalton_terms(l):
    """Per-m (norm, ((c, k, r2 power), ...)) of the Legendre-derivative harmonics."""
    from numpy.polynomial import Legendre, Polynomial
    p = Legendre.basis(l).convert(kind=Polynomial)
    return tuple((math.sqrt(math.factorial(l-m)/math.factorial(l+m)*(2 if m else 1)),
                  tuple((c, k, (l-m-k)//2) for k, c in enumerate(p.deriv(m).coef) if c != 0))
                 for m in range(l+1))



def _dalton_polynomials(l, xyz):
    """Independent Legendre-derivative solid harmonics, no reference data."""
    r2 = np.sum(xyz*xyz, axis=1)
    z = xyz[:, 2]
    xy = xyz[:, 0] + 1j*xyz[:, 1]
    components = []
    for m, (norm, terms) in enumerate(_dalton_terms(l)):
        radial = np.zeros(len(xyz))
        for c, k, power in terms:
            radial += c * z**k * r2**power
        components.append(norm * radial * xy**m)
    if l == 1:
        return np.column_stack((components[1].real, components[1].imag, components[0].real))
    return np.column_stack([components[m].imag for m in range(l, 0, -1)] +
                           [components[0].real] + [components[m].real for m in range(1, l+1)])



def explicit_main_overlap(recipe):
    """Independent Gaussian-product/Gauss-Hermite exact polynomial overlap.

    Each primitive pair uses order floor((la+lb)/2)+1 in each Cartesian
    coordinate, exact for degree la+lb. This is not molecular ISA quadrature,
    not inferred by applying the fitted transformation to the Psi4 metric.
    """
    from itertools import product
    offsets = np.cumsum([0] + [2*s.l+1 for s in recipe.shells])
    S = np.zeros((offsets[-1], offsets[-1]))
    quadrature = {}
    for i, a in enumerate(recipe.shells):
        A = np.array(recipe.centres[a.centre])
        for j, b in enumerate(recipe.shells[:i+1]):
            B = np.array(recipe.centres[b.centre])
            n = (a.l+b.l)//2+1
            if n not in quadrature:
                x, w = np.polynomial.hermite.hermgauss(n)
                ix = np.array(list(product(range(n), repeat=3)))
                quadrature[n] = (x[ix], np.prod(w[ix], axis=1))
            nodes, weights = quadrature[n]
            # All primitive pairs of the shell pair share one evaluation.
            points, scaled = [], []
            for alpha, ca in zip(a.exponents, a.coefficients):
                for beta, cb in zip(b.exponents, b.coefficients):
                    p = alpha+beta
                    points.append((alpha*A+beta*B)/p + nodes/math.sqrt(p))
                    scaled.append(ca*cb*math.exp(-alpha*beta/p*np.dot(A-B, A-B))/p**1.5 * weights)
            points, scaled = np.concatenate(points), np.concatenate(scaled)
            va, vb = _dalton_polynomials(a.l, points-A), _dalton_polynomials(b.l, points-B)
            block = va.T @ (scaled[:, None]*vb)
            S[offsets[i]:offsets[i+1], offsets[j]:offsets[j+1]] = block
            S[offsets[j]:offsets[j+1], offsets[i]:offsets[i+1]] = block.T
    return _owned(S)



def adapt_main(wfn, *, caller_converged):
    """Verified MAIN conversion; thread-local serial MKL for reproducible small SVDs.

    No process-wide BLAS/OpenMP setting is changed. Non-MKL builds retain their
    backend behavior. Heavy ISA and response work remains independently parallel.
    """
    return core._isa_serial_blas_call(lambda: _adapt_main(wfn, caller_converged=caller_converged))



def _adapt_main(wfn, *, caller_converged):
    """Return exact effective MAIN descriptors and verified C_DALTON = T C_Psi4.

    For each shell solve E T = P by full-rank SVD (64 angular samples at each
    of three exponent-scaled radii). Reject condition >1e8 or scaled errors
    >2e-11. Validate on disjoint 71-point directions, phase/radii. Independently
    check T^T S_exp T against the Mints shell self-overlap (analytic native
    co-centred metric). Independently construct the entire explicit MAIN
    overlap using Gaussian-product/finite Gauss-Hermite polynomial integration;
    compare T^T S_exp T to Mints (2e-11) and both original/transformed occupied
    orthonormalities to I (2e-9).
    No singular-value truncation, normalization, orbital fitting or repair.
    S-G pure MAIN only; Cartesian S/P accepted as identically sized angular
    spaces, Cartesian D and above explicitly rejected.
    """
    if caller_converged is not True:
        raise ValueError('Actual wavefunction requires caller_converged=True declaration')
    if not isinstance(wfn, core.Wavefunction) or wfn.nirrep() != 1:
        raise ValueError('Actual C1 Wavefunction required')
    if (wfn.nalpha() != wfn.nbeta() or not wfn.same_a_b_orbs() or wfn.nalpha() < 1
            or wfn.doccpi()[0] != wfn.nalpha() or wfn.soccpi()[0] != 0):
        raise ValueError('Restricted closed-shell occupied spatial orbitals required')
    basis = wfn.basisset()
    mol = wfn.molecule()
    centres = tuple((mol.x(i), mol.y(i), mol.z(i)) for i in range(mol.natom()))
    S = np.asarray(core.MintsHelper(basis).ao_overlap())
    C = np.array(wfn.Ca_subset('AO', 'OCC'), copy=True)
    Cb = np.asarray(wfn.Cb_subset('AO', 'OCC'))
    if C.shape != (basis.nbf(), wfn.nalpha()) or not np.isfinite(C).all() or not np.array_equal(C, Cb):
        raise ValueError('Restricted occupied coefficients inconsistent')
    ortho = float(np.max(np.abs(C.T @ S @ C - np.eye(C.shape[1]))))
    if not np.isfinite(ortho) or ortho > 2e-9:
        raise ValueError(f'MAIN occupied orthonormality failure: {ortho}')
    T = np.zeros_like(S)
    shells, diagnostics = [], []
    for j in range(basis.nshell()):
        sh = basis.shell(j)
        l = sh.am
        if l > 4 or (l >= 2 and not sh.is_pure()):
            raise ValueError('MAIN supports spherical S-G, not Cartesian D or higher')
        r = ShellRecipe(int(basis.shell_to_center(j)), int(l),
                        tuple(sh.exp(k) for k in range(sh.nprimitive)),
                        tuple(sh.coef(k) for k in range(sh.nprimitive)))
        shells.append(r)
        single = BasisRecipe('MAIN shell', 'actual GaussianShell.coef effective values',
                             'Spherical', centres, (r,))
        E = single.build('AtomAux')
        start = basis.shell_to_basis_function(j)
        end = start + 2*l + 1
        origin = np.array(centres[r.centre])
        scale = 1/math.sqrt(max(r.exponents))
        def points(n, phase, radii):
            return np.concatenate([origin + scale*a*_directions(n, phase) for a in radii])
        train = points(64, .0, (.4, 1.1, 2.3))
        held = points(71, .417, (.67, 1.47, 2.89))
        e = np.asarray(E.evaluate(train.tolist()))
        p = _psi_samples(basis, train)[:, start:end]
        u, sigma, vt = np.linalg.svd(e, full_matrices=False)
        condition = float(sigma[0]/sigma[-1]) if sigma[-1] > 0 else float('inf')
        if not np.isfinite(condition) or condition > 1e8:
            raise ValueError(f'MAIN shell {j} collocation rank/condition failure: {condition}')
        t = (vt.T / sigma) @ (u.T @ p)
        def error(a, b):
            return float(np.max(np.abs(a-b))/max(np.max(np.abs(b)), np.finfo(float).tiny))
        tr = error(e @ t, p)
        hr = error(np.asarray(E.evaluate(held.tolist())) @ t, _psi_samples(basis, held)[:, start:end])
        ov = error(t.T @ np.asarray(E.overlap()) @ t, S[start:end, start:end])
        if not np.isfinite([tr, hr, ov]).all() or max(tr, hr, ov) > 2e-11:
            raise ValueError(f'MAIN shell {j} validation failure: train={tr}, held={hr}, overlap={ov}')
        T[start:end, start:end] = t
        diagnostics.append(ShellTransformDiagnostic(j, l, condition, tr, hr, ov))
    recipe = BasisRecipe(basis.name(), 'actual wavefunction MAIN GaussianShell.coef (not original/ERD)',
                         'Spherical', centres, tuple(shells))
    main = recipe.build('Orbital')
    # Independent global held-out orbital sample test, after all shell tests.
    pts = np.concatenate([np.array(c) + _directions(83, .913)*1.73 for c in centres])
    cp = T @ C
    a, b = np.asarray(main.evaluate(pts.tolist())) @ cp, _psi_samples(basis, pts) @ C
    if np.max(np.abs(a-b)) > 2e-11 * max(1., np.max(np.abs(b))):
        raise ValueError('Transformed occupied orbitals failed independent global sampling')
    independent_overlap = explicit_main_overlap(recipe)
    ov_global = float(np.max(np.abs(T.T @ independent_overlap @ T - S)))
    transformed_ortho = float(np.max(np.abs(cp.T @ independent_overlap @ cp - np.eye(cp.shape[1]))))
    if not np.isfinite([ov_global, transformed_ortho]).all() or ov_global > 2e-11 or transformed_ortho > 2e-9:
        raise ValueError(f'Independent global MAIN overlap/orthonormality failure: {ov_global}, {transformed_ortho}')
    return MainAdaptation(recipe, main, _owned(T), _owned(cp), tuple(diagnostics), ortho,
                          ov_global, transformed_ortho)
