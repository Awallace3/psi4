"""Explicit adapted Cartesian Drho-C / ordinary ISA-A partition through Q.

No default basis recipe, hidden SCF, reference-file I/O, AO-density substitute,
charge rescaling, or reference-parity assertion. Effective shell coefficients
include normalization. MAIN is derived from the actual restricted C1 wavefunction.
Numerical shell collocation is a *verified representation change*, not a fit of
orbitals/density or a repair of their normalization. See ``adapt_main``.
"""
from dataclasses import dataclass
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
class SiteRecipe:
    label: str
    origin: tuple
    atomic: BasisRecipe
    shape: BasisRecipe
    shell_map: tuple
    rank: int
    tail_cutoff: float
    tail_allowed: bool

    def __post_init__(self):
        _text(self.label, 'site label')
        object.__setattr__(self, 'origin', tuple(float(x) for x in self.origin))
        object.__setattr__(self, 'shell_map', tuple(self.shell_map))
        if len(self.origin) != 3 or not np.isfinite(self.origin).all():
            raise ValueError('Invalid site origin')
        if type(self.rank) is not int or self.rank not in range(5):
            raise ValueError('Q rank must be 0-4')
        _finite(self.tail_cutoff, 'explicit tail cutoff in bohr', 1e-8)
        if type(self.tail_allowed) is not bool:
            raise ValueError('tail_allowed must be bool')
        for b in (self.atomic, self.shape):
            if any(len(s.exponents) != 1 or b.centres[s.centre] != self.origin for s in b.shells):
                raise ValueError('AtomAux/Shape must be primitive and co-centred at its site')
        if any(s.l != 0 for s in self.shape.shells):
            raise ValueError('Shape must be s-only')
        core.IsaShapeMap(self.atomic.build('AtomAux'), self.shape.build('Shape'), self.shell_map)


@dataclass(frozen=True)
class GridRecipe:
    radial_points: int
    spherical_points: int
    becke_smoothing: int
    radius_scaling: float
    radius_policy: str
    neighbour_policy: str

    def __post_init__(self):
        if self.radius_policy != 'native_tabulated_bragg_slater':
            raise ValueError('Only explicit native_tabulated_bragg_slater radii supported')
        if self.neighbour_policy != 'all_sites_unscreened_full_molecular_grid':
            raise ValueError('Only all-sites unscreened full molecular quadrature supported')
        for name, low in (('radial_points', 2), ('spherical_points', 1), ('becke_smoothing', 0)):
            v = getattr(self, name)
            if type(v) is not int or v < low:
                raise ValueError(f'Invalid {name}')
        _finite(self.radius_scaling, 'radius_scaling', np.finfo(float).tiny)


@dataclass(frozen=True)
class ControllerRecipe:
    # All controls required: no silent reference/default recipe inference.
    convergence: float
    max_iterations: int
    w_eps: float
    positive_lambda: float
    positive_max_alpha: float
    positive_auto: bool
    damping: float
    s_block_only: bool
    density_cutoff: float
    w_eps_activation: float
    positive_activation: float
    tail_activation: float
    mixing: float
    mixing_skip: int
    tail_iteration_limit: int
    fix_tails: bool

    def __post_init__(self):
        for name in ('positive_auto', 's_block_only', 'fix_tails'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be bool')
        for name, low in (('max_iterations', 1), ('mixing_skip', 0), ('tail_iteration_limit', 0)):
            if type(getattr(self, name)) is not int or getattr(self, name) < low:
                raise ValueError(f'Invalid {name}')
        for name in ('convergence', 'w_eps', 'positive_lambda', 'positive_max_alpha',
                     'damping', 'density_cutoff', 'w_eps_activation', 'positive_activation',
                     'tail_activation', 'mixing'):
            _finite(getattr(self, name), name, 0.)
        if self.convergence <= 0 or not 0 <= self.mixing <= 1:
            raise ValueError('Invalid convergence/mixing')

    def build(self, sites):
        o = core.IsaAControllerOptions()
        f = core.IsaAFitOptions()
        fit_fields = ('w_eps', 'positive_lambda', 'positive_max_alpha', 'positive_auto',
                      'damping', 's_block_only', 'density_cutoff')
        for name in self.__dataclass_fields__:
            setattr(f if name in fit_fields else o, name, getattr(self, name))
        o.fit = f
        o.tail_cutoffs = [s.tail_cutoff for s in sites]
        o.tail_allowed = [s.tail_allowed for s in sites]
        o.convergence_included = [True] * len(sites)
        return o


@dataclass(frozen=True)
class PartitionRecipe:
    name: str
    origin: str
    track: str
    auxiliary: BasisRecipe
    sites: tuple
    grid: GridRecipe
    controller: ControllerRecipe
    drho_profile: str

    def __post_init__(self):
        _text(self.name, 'recipe name')
        _text(self.origin, 'recipe origin')
        object.__setattr__(self, 'sites', tuple(self.sites))
        if self.track != 'explicit_cartesian_drho_c_isa_a':
            raise ValueError('Only explicitly adapted Cartesian Drho-C/ISA-A supported; no modern preset')
        if self.auxiliary.representation != 'Cartesian':
            raise ValueError('Molecular AUX must explicitly be Cartesian')
        if not self.sites or len({s.label for s in self.sites}) != len(self.sites):
            raise ValueError('Unique nonempty site labels required')
        def signatures(b):
            return [(b.centres[s.centre], s.l, s.exponents, s.coefficients) for s in b.shells]
        if signatures(self.auxiliary) == [v for site in self.sites for v in signatures(site.atomic)]:
            raise ValueError('Distinct AtomAux required: identical AUX/AtomAux initialization needs separate unconstrained Drho bookkeeping')
        if self.drho_profile not in ('strict1e-9', 'Drho1e-2'):
            raise ValueError('Declare strict1e-9 or user-authorized Drho1e-2 comparison profile')


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


def _dalton_polynomials(l, xyz):
    """Independent Legendre-derivative solid harmonics, no reference data."""
    from numpy.polynomial import Legendre, Polynomial
    r2 = np.sum(xyz*xyz, axis=1)
    z = xyz[:, 2]
    xy = xyz[:, 0] + 1j*xyz[:, 1]
    components = []
    p = Legendre.basis(l).convert(kind=Polynomial)
    for m in range(l+1):
        radial = np.zeros(len(xyz))
        for k, c in enumerate(p.deriv(m).coef):
            if c != 0:
                radial += c * z**k * r2**((l-m-k)//2)
        norm = math.sqrt(math.factorial(l-m)/math.factorial(l+m)*(2 if m else 1))
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
            block = np.zeros((2*a.l+1, 2*b.l+1))
            for alpha, ca in zip(a.exponents, a.coefficients):
                for beta, cb in zip(b.exponents, b.coefficients):
                    p = alpha+beta
                    centre = (alpha*A+beta*B)/p
                    points = centre + nodes/math.sqrt(p)
                    va, vb = _dalton_polynomials(a.l, points-A), _dalton_polynomials(b.l, points-B)
                    factor = ca*cb*math.exp(-alpha*beta/p*np.dot(A-B, A-B))/p**1.5
                    block += factor * (va.T @ (weights[:, None]*vb))
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


def one_gto_initialization(sites):
    """ONE-GTO alpha0=1: first minimum ABSOLUTE exponent difference, coefficient 1.

    Shape shells are primitive. Atomic initial D is separately zero: ordinary A
    does not infer initial atomic density from w0; no previous negative ridge
    coefficients exist. For the supported distinct-basis branch, source
    initialize_D0 zeros D/D0; initialize_w0 separately chooses the unit GTO.
    Identical molecular/atomic AUX initialization is not supported here.
    """
    initial = core.IsaSweepState()
    initial.atomic_coefficients = [[0.] * s.atomic.build('AtomAux').nfunction for s in sites]
    shapes = []
    for s in sites:
        k = min(range(len(s.shape.shells)), key=lambda i: abs(s.shape.shells[i].exponents[0]-1.))
        w = [0.] * len(s.shape.shells)
        w[k] = 1.
        shapes.append(w)
    initial.shape_coefficients = shapes
    return initial


def final_shape_samples(shapes, state, sites, points):
    """Stored final Func-1/Fit-3 tails only; never fit tails at this boundary."""
    n = len(sites)
    if not n or any(len(v) != n for v in (shapes, state.coefficients.shape_coefficients, state.tails)):
        raise ValueError('Final shape/site/state dimensions must match exactly')
    return tuple(_owned(core.IsaGaussianShape(b, c).sample(
        points, tail, bool(state.apply_tails and site.tail_allowed)))
        for b, c, tail, site in zip(shapes, state.coefficients.shape_coefficients, state.tails, sites))


@dataclass(frozen=True)
class NativePartitionResult:
    recipe: PartitionRecipe
    main: MainAdaptation
    auxiliary: object
    coulomb: object
    density: object
    drho: object
    controller: object
    initial: object
    trajectory: object
    grid: object
    grid_points: np.ndarray
    grid_weights: np.ndarray
    density_samples: np.ndarray
    shape_samples: tuple
    q: object
    drho_metric_condition: float
    provenance: str
    comparison_status: str = 'not_evaluated_no_reference'

    @property
    def converged(self):
        return self.trajectory.state.converged

    def require_q(self):
        if self.q is None:
            raise RuntimeError(f'ISA-A did not converge: {self.trajectory.termination}; Q unavailable')
        return self.q


def native_partition(wfn, recipe, *, caller_converged):
    """Actual restricted SCF -> verified MAIN -> native Drho-C -> ordinary A -> Q.

    Returns inspectable nonconvergence with q=None. Native objects own copies of
    input arrays; result snapshots are independent of the wavefunction/recipe.
    Native result records (drho/history) remain caller-mutable, not live caches.
    Q rows/site/component convention and cutoff diagnostics are labeled by the
    current native Q API. No response/alpha/Cn computed by this factory.
    """
    if not isinstance(recipe, PartitionRecipe):
        raise TypeError('Explicit immutable PartitionRecipe required')
    main = adapt_main(wfn, caller_converged=caller_converged)
    mol = wfn.molecule()
    geometry = main.recipe.centres
    if len(recipe.sites) != mol.natom() or tuple(s.origin for s in recipe.sites) != geometry:
        raise ValueError('Site order/origins must exactly match actual wavefunction nuclei')
    if recipe.auxiliary.centres != geometry:
        raise ValueError('Molecular AUX centres must exactly match actual MAIN geometry/order')
    if any(mol.Z(i) <= 0 for i in range(mol.natom())):
        raise ValueError('Ghost/dummy partition sites unsupported')
    auxiliary = recipe.auxiliary.build('MolecularAux')
    atomic = [s.atomic.build('AtomAux') for s in recipe.sites]
    shapes = [s.shape.build('Shape') for s in recipe.sites]
    options = recipe.controller.build(recipe.sites)
    go = core.IsaGridOptions()
    for name in ('radial_points', 'spherical_points', 'becke_smoothing', 'radius_scaling'):
        setattr(go, name, getattr(recipe.grid, name))
    grid = core.IsaGrid(mol.clone(), go)
    if any(not np.isfinite(grid.alpha(i)) or grid.alpha(i) <= 0 for i in range(mol.natom())):
        raise ValueError('Native tabulated Slater radius unavailable')
    pts = np.column_stack((grid.x(), grid.y(), grid.z()))
    weights = np.asarray(grid.w())
    grids = []
    # Each site integrates over the entire Becke-weighted molecular grid,
    # NOT merely its generating atom's radial/angular subgrid.
    for site in recipe.sites:
        g = core.IsaNoTailGrid()
        g.points, g.weights = pts.tolist(), weights.tolist()
        g.density_sites = list(range(len(recipe.auxiliary.centres)))
        g.shape_sites = list(range(len(recipe.sites)))
        grids.append(g)
    coulomb = core.IsaAuxCoulomb(auxiliary)
    drho = coulomb.fit_drho_c(main.basis, core.Matrix.from_array(main.occupied), 1000.)
    density = core.IsaFixedDensity(auxiliary, drho.coefficients)
    samples = density.evaluate(pts.tolist(), list(range(len(geometry))))
    controller = core.IsaAController(atomic, shapes, [s.shell_map for s in recipe.sites], density, grids, options)
    initial = controller.initialize(one_gto_initialization(recipe.sites))
    trajectory = controller.run(initial)
    # Iteration-only caches need not overlap the much larger final-Q and response
    # workspaces. Preserve a fully owned, rerunnable controller, not its cache.
    controller = controller.without_prepared_cache()
    provenance = (f'native restricted C1 wfn; caller declares SCF converged; {recipe.name}; '
                  f'recipe origin: {recipe.origin}; actual MAIN effective coef + validated shell collocation; '
                  f'Drho-C lambda1000 unrescaled; {recipe.drho_profile} comparisons not evaluated; '
                  'ordinary ISA-A; full molecular IsaGrid/all sites unscreened (no screening parity); '
                  'stored final tails; global Cartesian axes/bohr/atomic units; no reference orbitals or density')
    sampled_shapes, q = (), None
    if trajectory.state.converged:
        sampled_shapes = final_shape_samples(shapes, trajectory.state, recipe.sites, pts.tolist())
        total = np.sum(sampled_shapes, axis=0)
        qsites = []
        for i, site in enumerate(recipe.sites):
            s = core.IsaMultipoleSamples()
            s.points, s.weights = pts.tolist(), weights.tolist()
            s.shape, s.shape_sum = sampled_shapes[i].tolist(), total.tolist()
            s.auxiliary_sites = list(range(len(geometry)))
            qs = core.IsaMultipoleSite()
            qs.label, qs.origin, qs.rank, qs.samples = site.label, site.origin, site.rank, s
            qsites.append(qs)
        q = core.IsaPartitionedMultipoles(auxiliary, qsites, provenance, recipe.controller.density_cutoff)
    return NativePartitionResult(recipe, main, auxiliary, coulomb, density, drho, controller,
                                 initial, trajectory, grid, _owned(pts), _owned(weights), _owned(samples),
                                 sampled_shapes, q, float(np.linalg.cond(np.asarray(drho.metric))), provenance)
