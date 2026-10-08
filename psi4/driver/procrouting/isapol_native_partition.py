# Copyright (c) 2007-2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""Explicit Drho-C / ordinary ISA-A partition through distributed multipoles.

Recipes declare the basis, grid, initialization and tail model. No hidden SCF,
reference-file I/O, charge rescaling or default recipe.
"""
from dataclasses import dataclass
import numpy as np
from psi4 import core
from .isapol_basis import BasisRecipe, MainAdaptation, adapt_main, _text, _finite, _owned

# The Drho-C LU solve is refined with an exact residual on the same factors (the
# native fit_drho_c default stays 0, plain LU). Nonconvergence refuses; there is
# no fallback to the plain LU coefficients.
DRHO_REFINEMENT_ITERATIONS = 10


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
    atomic_initialization: str = 'zero_atomic_D0'

    def __post_init__(self):
        _text(self.name, 'recipe name')
        _text(self.origin, 'recipe origin')
        object.__setattr__(self, 'sites', tuple(self.sites))
        # Spherical (2l+1) and Cartesian ((l+1)(l+2)/2) shells span different
        # AUX spaces and therefore define different fits/partitions.
        tracks = {'explicit_cartesian_drho_c_isa_a': 'Cartesian',
                  'explicit_spherical_drho_c_isa_a': 'Spherical'}
        if self.track not in tracks:
            raise ValueError('Only explicitly adapted Cartesian/spherical Drho-C/ISA-A supported; no modern preset')
        if self.auxiliary.representation != tracks[self.track]:
            raise ValueError(f'Track {self.track} requires a {tracks[self.track]} molecular AUX, '
                             f'not {self.auxiliary.representation}')
        if not self.sites or len({s.label for s in self.sites}) != len(self.sites):
            raise ValueError('Unique nonempty site labels required')
        def signatures(b):
            return [(b.centres[s.centre], s.l, s.exponents, s.coefficients) for s in b.shells]
        # Exactly concatenating AtomAux starts D0/D from site Drho blocks;
        # distinct bases start at zero. The declared branch must match.
        identical = signatures(self.auxiliary) == [v for site in self.sites for v in signatures(site.atomic)]
        if self.atomic_initialization not in ('zero_atomic_D0', 'drho_partitioned_atomic_D0'):
            raise ValueError('Declare zero_atomic_D0 or drho_partitioned_atomic_D0 atomic initialization')
        if identical != (self.atomic_initialization == 'drho_partitioned_atomic_D0'):
            raise ValueError('drho_partitioned_atomic_D0 requires, and identical AUX/AtomAux requires, '
                             'an AtomAux concatenating exactly into the molecular AUX')
        if self.drho_profile not in ('strict1e-9', 'Drho1e-2'):
            raise ValueError('Declare strict1e-9 or user-authorized Drho1e-2 comparison profile')


def one_gto_initialization(sites):
    """ONE-GTO alpha0=1: first minimum ABSOLUTE exponent difference, coefficient 1.

    Shape shells are primitive. Distinct AtomAux starts atomic D/D0 at zero,
    independently of w0. Identical AUX uses ``drho_partitioned_initialization``.
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


def drho_partitioned_initialization(sites, auxiliary, coefficients):
    """Atomic D0/D from the molecular Drho block on each site; w0 as ONE-GTO.

    Transcribed with attribution from CamCASP (MIT) ``src/stockholder.F90``
    ``initialize_D0``: when the atomic AUX sets are the AUX1 subsets of, or are
    identical to, the molecular AUX, ``D0(1:naux) = D(1:naux) = DFrho(first:last)``
    over that site's function range; otherwise both stay zero. Only the exact
    identical-basis case is supported here, so ``first:last`` is unambiguous.
    ``DFrho`` is the density expansion actually being partitioned -- the same
    coefficient vector handed to ``IsaFixedDensity`` -- not a second fit.
    This is a declared initialization branch, not a convergence aid.
    """
    if len(coefficients) != sum(2*s.l+1 if auxiliary.representation == 'Spherical'
                                else (s.l+1)*(s.l+2)//2 for s in auxiliary.shells):
        raise ValueError('Drho coefficient count does not match the molecular AUX')
    initial = core.IsaSweepState()
    atomic, offset, cursor = [], 0, 0
    for site in sites:
        basis = site.atomic.build('AtomAux')
        width = 0
        for shell in site.atomic.shells:
            if (auxiliary.centres[auxiliary.shells[cursor].centre], auxiliary.shells[cursor].l,
                auxiliary.shells[cursor].exponents, auxiliary.shells[cursor].coefficients) != \
               (site.atomic.centres[shell.centre], shell.l, shell.exponents, shell.coefficients):
                raise ValueError('AtomAux must concatenate exactly into the molecular AUX in order')
            width += 2*shell.l+1 if auxiliary.representation == 'Spherical' else (shell.l+1)*(shell.l+2)//2
            cursor += 1
        if width != basis.nfunction:
            raise ValueError('AtomAux block width does not match its native function count')
        atomic.append([float(c) for c in coefficients[offset:offset+width]])
        offset += width
    if cursor != len(auxiliary.shells) or offset != len(coefficients):
        raise ValueError('Site AtomAux blocks do not tile the molecular AUX')
    initial.atomic_coefficients = atomic
    initial.shape_coefficients = one_gto_initialization(sites).shape_coefficients
    return initial


def final_shape_samples(shapes, state, sites, points, *, tails=None):
    """Stored final Func-1/Fit-3 tails only; never fit tails at this boundary.

    ``state.tails`` lags one shape update. ``native_partition`` explicitly uses
    ``trajectory.final_tails``, the postconvergence refit; surrogate states
    default to their own stored tails.
    """
    n = len(sites)
    tails = state.tails if tails is None else tails
    if not n or any(len(v) != n for v in (shapes, state.coefficients.shape_coefficients, tails)):
        raise ValueError('Final shape/site/state dimensions must match exactly')
    return tuple(_owned(core.IsaGaussianShape(b, c).sample(
        points, tail, bool(state.apply_tails and site.tail_allowed)))
        for b, c, tail, site in zip(shapes, state.coefficients.shape_coefficients, tails, sites))


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
            reason = 'multipoles not requested' if self.converged else 'ISA-A did not converge'
            raise RuntimeError(f'{reason}: {self.trajectory.termination}; Q unavailable')
        return self.q


def native_partition(wfn, recipe, *, caller_converged, build_multipoles=True):
    """Actual restricted SCF -> verified MAIN -> native Drho-C -> ordinary A -> Q.

    Returns inspectable nonconvergence with q=None. Native objects own copies of
    input arrays; result snapshots are independent of the wavefunction/recipe.
    Native result records (drho/history) remain caller-mutable, not live caches.
    Q rows/site/component convention and cutoff diagnostics are labeled by the
    current native Q API. No response/alpha/Cn computed by this factory.
    ``build_multipoles=False`` runs the same iteration without sampling final
    shapes or constructing Q for the density-fit AUX. Response adapters can then
    sample final shapes on their explicit integration grid and response AUX.
    """
    if not isinstance(recipe, PartitionRecipe):
        raise TypeError('Explicit immutable PartitionRecipe required')
    if type(build_multipoles) is not bool:
        raise TypeError('build_multipoles must be bool')
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
    drho = coulomb.fit_drho_c(main.basis, core.Matrix.from_array(main.occupied), 1000.,
                              max_refinement_iterations=DRHO_REFINEMENT_ITERATIONS)
    density = core.IsaFixedDensity(auxiliary, drho.coefficients)
    samples = density.evaluate(pts.tolist(), list(range(len(geometry))))
    controller = core.IsaAController(atomic, shapes, [s.shell_map for s in recipe.sites], density, grids, options)
    declared = (drho_partitioned_initialization(recipe.sites, recipe.auxiliary, drho.coefficients)
               if recipe.atomic_initialization == 'drho_partitioned_atomic_D0'
               else one_gto_initialization(recipe.sites))
    initial = controller.initialize(declared)
    trajectory = controller.run(initial)
    # Iteration-only caches need not overlap the much larger final-Q and response
    # workspaces. Preserve a fully owned, rerunnable controller, not its cache.
    controller = controller.without_prepared_cache()
    provenance = (f'native restricted C1 wfn; caller declares SCF converged; {recipe.name}; '
                  f'recipe origin: {recipe.origin}; actual MAIN effective coef + validated shell collocation; '
                  f'Drho-C lambda1000 unrescaled; {recipe.drho_profile} comparisons not evaluated; '
                  f'ordinary ISA-A; atomic initialization {recipe.atomic_initialization}; '
                  'full molecular IsaGrid/all sites unscreened (no screening parity); '
                  'postconvergence-refit final tails; global Cartesian axes/bohr/atomic units; '
                  'no reference orbitals or density')
    sampled_shapes, q = (), None
    if build_multipoles and trajectory.state.converged:
        sampled_shapes = final_shape_samples(shapes, trajectory.state, recipe.sites, pts.tolist(),
                                             tails=trajectory.final_tails)
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
