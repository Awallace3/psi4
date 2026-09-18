"""DF-centre distributed multipoles: CamCASP's ``DistPolAlgorithm = 'DF'`` rule.

CamCASP's distributed-polarizability default is ``'DF'``
(``polarizability.F90:219``), whose ``dist_polarizabilities_DF`` (line 1593)
assigns each auxiliary function **wholly** to the centre that function sits on
(``centre = 0``, ``isitdist = .true.``) and then forms
``alpha^(a,b)_{t,u}(w) = - Qa(k,t) C_{k,l}(w) Qb(l,u)``.  It never forms a
stockholder weight, so no ISA-A fixed point is a prerequisite for it.

This is a DIFFERENT DECLARED MODEL from the ISA-A shape partition that
``isapol_native_partition`` produces, not a better-converged version of it.  The
two assign the same molecular density to sites by different rules and their site
multipoles, site polarizabilities and dispersion coefficients may never be quoted
as agreeing.  Choosing between them is a model declaration.

Two forms of the same rule are provided:

``grid``
    ``Q(a,t,k) = sum_p w_p R_t(r_p - R_a) chi_k(r_p)`` for ``k`` on ``a`` and
    zero otherwise, evaluated on the caller's molecular quadrature.  It needs no
    new arithmetic at all: ``auxiliary_sites=[a]`` already zeroes every function
    not centred on ``a`` (``explicit_basis.cc``) and ``shape == shape_sum`` makes
    the stockholder ratio identically one, so the shipped sampling constructor
    computes exactly this rule.  It inherits the molecular grid's quadrature
    error, worst in relative terms on the charge row because ``int(chi_k)`` of a
    diffuse fitting function is the hardest moment for a molecular grid to
    integrate.

``analytic``
    the same rule in closed form.  Because every factor is co-centred the
    integral has an elementary value: the Racah regular solid harmonic ``R_lm``
    is a homogeneous polynomial of degree ``l``, the GAMINT Cartesian angular
    factor is a monomial, and the radial part is a contraction of s-type
    Gaussians, so

        Q(a,(l,m),k) = N_k sum_i c_i sum_q C_q^{lm}
                       prod_d Gamma((n_d+1)/2) / zeta_i**((n_d+1)/2)

    with ``n = q + p(k)`` and odd powers vanishing.  Nothing is sampled, so this
    form carries no quadrature defect whatever.  On PBE0/cc-pVDZ water with the
    cc-pVDZ-JKFIT AUX the two forms differ by 2.78e-07 absolute / 6.65e-09
    relative.  The difference is spread over every rank -- 2.7e-08 relative on the
    charge row, 4.0e-09 at rank 1, 8.2e-10 at rank 2, 9.3e-09 at rank 3 -- so it
    is grid error, not one bad moment.  That the charge row is the grid's worst
    is confirmed independently: the closed form reproduces the directly derived
    ``int(chi_k)`` to 7.1e-15 where the grid form is 2.78e-07 off it.

The harmonic monomial coefficients ``C^{lm}`` are recovered from the SHIPPED
``core.isa_regular_multipoles``, never reimplemented, so this module cannot
disagree with the rest of the library about what a Racah component is.  Both
recoveries are *gated*, not merely reported: an expansion that does not
reproduce the shipped evaluator, or a Cartesian convention that does not
reproduce ``IsaExplicitBasis.evaluate``, is refused rather than used.

WHERE THIS RULE IS SOUND.  The DF rule is basis-sensitive in a way the
stockholder rule is not, because a function's whole moment is charged to its own
centre with no partition to damp it.  On the reference's own declared
aug-cc-pVTZ-RI auxiliary set it is well behaved and reaches the reference's site
ratio (``SPEC.md`` section 7).  On a MAIN-matched JKFIT auxiliary set it is
numerically hopeless: static distributed blocks reach 1.3e+03 while summing to
about 9.8, and the closed form yields a NEGATIVE hydrogen rank-3 scalar.  A
negative site response is a broken model, so ``site_isotropic_gate`` refuses it
here rather than letting it propagate into a polarizability or a C6.  That
refusal is a property of the declared (rule, auxiliary basis) pair; it is never
worked around by loosening the gate.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from psi4 import core

#: GAMINT Cartesian power order per shell angular momentum, matching
#: ``IsaExplicitBasis``'s stored function order.  Not a sortable convention: the
#: order is part of the column identity of every AUX matrix in this library.
CARTESIAN_POWERS = {
    0: ((0, 0, 0),),
    1: ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    2: ((2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)),
    3: ((3, 0, 0), (0, 3, 0), (0, 0, 3), (2, 1, 0), (2, 0, 1), (1, 2, 0),
        (0, 2, 1), (1, 0, 2), (0, 1, 2), (1, 1, 1)),
    4: ((4, 0, 0), (0, 4, 0), (0, 0, 4), (3, 1, 0), (3, 0, 1), (1, 3, 0),
        (0, 3, 1), (1, 0, 3), (0, 1, 3), (2, 2, 0), (2, 0, 2), (0, 2, 2),
        (2, 1, 1), (1, 2, 1), (1, 1, 2)),
}

#: Highest rank the shipped Racah evaluator and the GAMINT power table cover.
MAX_RANK = 4

#: Declared point sets for the two recoveries below.  They are fixed model
#: constants, not sample sizes to be converged: the recovered numbers are the
#: exact monomial coefficients of a polynomial, so a different draw gives the
#: same answer to the gated tolerance.  They are pinned so that a given build
#: reproduces a given Q bitwise, and they are the same two constants the
#: archived DF-centre audit ran at.
HARMONIC_SEED = 20260910
CONVENTION_SEED = 20260911
#: Worst admissible reconstruction error.  Measured through rank 3 the harmonic
#: recovery lands at 2.31e-14 and the convention check at 4.44e-16 against a
#: basis magnitude of 2.88, so these gates are more than three orders of
#: magnitude above the observed error and still tight enough to catch a real
#: ordering, sign or normalization change.
HARMONIC_TOLERANCE = 1.0e-10
CONVENTION_TOLERANCE = 1.0e-10

#: Both DF-centre forms produce AUX columns, so the only representation they can
#: declare is the fitted one.  A direct-OV Q has occupied-virtual product
#: columns and no auxiliary centre for a function to sit on, so the DF rule is
#: not even definable there.
REPRESENTATION = 'fitted_density_coefficients'


#: The DF-centre forms this module produces, as the public path names them.
DF_CENTRE_DISTRIBUTIONS = ('df_centre_analytic', 'df_centre_grid')
#: Every distributed-multipole rule the pipeline can declare.  ``isa_a`` is the
#: shipped stockholder shape partition and lives in ``isapol_native_partition``;
#: it is listed here only so one tuple names the whole choice.
DISTRIBUTIONS = ('isa_a',) + DF_CENTRE_DISTRIBUTIONS


def df_centre_multipoles(distribution, auxiliary, sites, rank, *, points=None, weights=None):
    """Produce the named DF-centre Q.

    The two forms are the same declared rule, so they take the same arguments
    apart from the grid the sampled one integrates on.  Handing a grid to the
    closed form, or omitting it from the sampled one, is refused rather than
    ignored: which form ran is part of what the result means.
    """
    if distribution == 'df_centre_analytic':
        if points is not None or weights is not None:
            raise ValueError('The closed-form DF-centre rule samples nothing; do not hand it a grid')
        return analytic_df_centre_multipoles(auxiliary, sites, rank)
    if distribution == 'df_centre_grid':
        if points is None or weights is None:
            raise ValueError('The grid DF-centre rule needs the molecular quadrature it integrates on')
        return grid_df_centre_multipoles(auxiliary, sites, rank, points, weights)
    raise ValueError(f'Unknown DF-centre distribution {distribution!r}; '
                     f'expected one of {DF_CENTRE_DISTRIBUTIONS}')


def double_factorial(n):
    """``n!!`` for odd ``n``, with the empty product for ``n <= 0``."""
    value = 1.0
    while n > 0:
        value *= n
        n -= 2
    return value


def cartesian_normalization(l, powers):
    """GAMINT angular normalization of one Cartesian function of an l-shell.

    The shell's effective coefficients already carry the radial normalization,
    so this is only the factor that relates a Cartesian monomial to the shell's
    own ``(l,0,0)`` reference.
    """
    return math.sqrt(double_factorial(2*l-1)
                     / (double_factorial(2*powers[0]-1)
                        * double_factorial(2*powers[1]-1)
                        * double_factorial(2*powers[2]-1)))


def monomials(l):
    """Degree-``l`` Cartesian monomial exponents, in the recovery's own order."""
    return tuple((a, b, l-a-b) for a in range(l, -1, -1) for b in range(l-a, -1, -1))


@dataclass(frozen=True)
class HarmonicExpansion:
    """Monomial expansion of every Racah ``R_lm`` through ``rank``, read off the
    shipped evaluator and gated against it."""
    rank: int
    seed: int
    tolerance: float
    coefficients: tuple
    residuals: tuple

    @property
    def residual_max(self):
        return max(self.residuals)

    def block(self, l):
        """``(nmonomial, 2l+1)`` coefficients of the ``l`` harmonics."""
        return self.coefficients[l]


def harmonic_expansion(rank, *, seed=HARMONIC_SEED, tolerance=HARMONIC_TOLERANCE):
    """Recover ``C^{lm}_q`` from ``core.isa_regular_multipoles`` and gate it.

    ``R_lm`` is a homogeneous polynomial of degree ``l``, so its monomial
    coefficients are determined by its values; they are solved for here rather
    than transcribed, which is what keeps this module in exact agreement with
    the shipped evaluator's sign, ordering and normalization conventions.
    """
    if not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank <= MAX_RANK:
        raise ValueError(f'DF-centre multipole rank must be an integer 0 to {MAX_RANK}, not {rank!r}')
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 1:
        raise ValueError(f'Harmonic recovery seed must be a positive integer, not {seed!r}')
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError(f'Harmonic recovery tolerance must be finite and positive, not {tolerance!r}')
    rng = np.random.default_rng(seed)
    coefficients, residuals = [], []
    for l in range(rank+1):
        mons = monomials(l)
        points = rng.normal(size=(8*len(mons)+16, 3))
        design = np.array([[x**m[0] * y**m[1] * z**m[2] for m in mons] for x, y, z in points])
        target = np.array([core.isa_regular_multipoles(l, list(p))[l*l:] for p in points])
        block, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        residual = float(np.max(np.abs(design @ block - target)))
        if not np.isfinite(residual) or residual > tolerance:
            raise RuntimeError(f'Racah rank-{l} monomial recovery failed: residual {residual!r} '
                               f'exceeds {tolerance!r}; the shipped evaluator and this '
                               'module disagree about what a Racah component is')
        coefficients.append(block)
        residuals.append(residual)
    return HarmonicExpansion(rank, seed, float(tolerance),
                             tuple(coefficients), tuple(residuals))


def verify_cartesian_convention(basis, built, *, seed=CONVENTION_SEED,
                                tolerance=CONVENTION_TOLERANCE):
    """Gate the GAMINT ordering/normalization against the shipped collocation.

    The closed form below walks ``basis.shells`` and ``CARTESIAN_POWERS`` itself,
    so it must first prove that this walk lands on exactly the columns
    ``IsaExplicitBasis.evaluate`` produces, in exactly that order, with exactly
    that normalization.  Returns ``(error, magnitude)``; raises if the error
    exceeds ``tolerance`` relative to the magnitude.
    """
    if basis.representation != 'Cartesian':
        raise ValueError('The DF-centre rule needs a Cartesian GAMINT molecular AUX; '
                         f'this recipe declares {basis.representation}')
    rng = np.random.default_rng(seed)
    points = rng.normal(scale=1.5, size=(64, 3))
    reference = np.asarray(built.evaluate(points.tolist()))
    mine = np.zeros_like(reference)
    column = 0
    for shell in basis.shells:
        if shell.l not in CARTESIAN_POWERS:
            raise ValueError(f'Cartesian AUX shell l={shell.l} exceeds the tabulated '
                             f'GAMINT power order (l <= {MAX_RANK})')
        delta = points - np.asarray(basis.centres[shell.centre])
        r2 = np.einsum('pd,pd->p', delta, delta)
        radial = sum(c*np.exp(-z*r2) for z, c in zip(shell.exponents, shell.coefficients))
        for powers in CARTESIAN_POWERS[shell.l]:
            mine[:, column] = cartesian_normalization(shell.l, powers)*radial*(
                delta[:, 0]**powers[0]*delta[:, 1]**powers[1]*delta[:, 2]**powers[2])
            column += 1
    if column != reference.shape[1]:
        raise ValueError(f'Cartesian AUX function count mismatch: this module walks {column} '
                         f'columns, the shipped basis has {reference.shape[1]}')
    error = float(np.max(np.abs(mine-reference)))
    magnitude = float(np.max(np.abs(reference)))
    if not np.isfinite(error) or error > tolerance*max(magnitude, 1.0):
        raise RuntimeError(f'GAMINT Cartesian convention check failed: {error!r} against '
                           f'magnitude {magnitude!r}; the closed-form DF-centre walk does '
                           'not reproduce the shipped collocation')
    return error, magnitude


def closed_form_charge_rows(basis, nsite):
    """``int chi_k`` per auxiliary function, charged wholly to its own centre.

    An independent derivation of the DF rule's charge row: ``R_00 = 1``, so the
    ``(a, 00)`` row is just the integral of each function on ``a``.  It is
    derived here from the Gaussian moment directly, with no harmonic expansion
    entering, which is what makes it a real cross-check on the rank-0 block of
    both forms rather than a restatement of one of them.  It checks only rank 0;
    the higher ranks are cross-checked by running both forms against each other.
    """
    rows = np.zeros((nsite, sum(len(CARTESIAN_POWERS[s.l]) for s in basis.shells)))
    column = 0
    for shell in basis.shells:
        for powers in CARTESIAN_POWERS[shell.l]:
            if not any(p % 2 for p in powers):
                rows[shell.centre, column] = cartesian_normalization(shell.l, powers)*sum(
                    c*math.prod(math.gamma(.5*(p+1))/z**(.5*(p+1)) for p in powers)
                    for z, c in zip(shell.exponents, shell.coefficients))
            column += 1
    return rows


@dataclass(frozen=True)
class DFCentreMultipoles:
    """A DF-centre Q and everything that had to hold for it to be produced."""
    form: str
    rank: int
    values: np.ndarray
    labels: tuple
    origins: tuple
    representation: str
    provenance: str
    diagnostics: dict = field(default_factory=dict)

    def sites(self):
        """The declared ``IsaMultipoleSite`` axes, carrying no samples."""
        out = []
        for label, origin in zip(self.labels, self.origins):
            site = core.IsaMultipoleSite()
            site.label, site.origin, site.rank = label, list(origin), self.rank
            out.append(site)
        return out

    def partition(self):
        """The Q as a ``core.IsaPartitionedMultipoles`` supplied-values record.

        Going through the shipped class rather than keeping the array loose means
        the DF-centre rule reaches ``IsaDistributedResponse`` -- and therefore the
        same ``-Q C Q^T`` arithmetic, the same finiteness checks and the same
        reciprocity diagnostic -- as the stockholder rule, with no second
        implementation of the contraction anywhere.
        """
        return core.IsaPartitionedMultipoles(core.Matrix.from_array(self.values),
                                             self.sites(), self.representation,
                                             self.provenance)


def _declared_axes(auxiliary, sites, rank):
    """Refuse anything but an exact site-to-auxiliary-centre correspondence."""
    if not isinstance(rank, int) or isinstance(rank, bool) or not 1 <= rank <= MAX_RANK:
        raise ValueError(f'DF-centre multipole rank must be an integer 1 to {MAX_RANK}, not {rank!r}')
    centres = np.asarray(auxiliary.centres)
    if len(sites) != len(centres):
        raise ValueError(f'The DF-centre rule charges every auxiliary function to its own '
                         f'centre, so it needs one site per auxiliary centre: {len(sites)} '
                         f'sites against {len(centres)} centres')
    for i, site in enumerate(sites):
        # Exact, not approximate: a site displaced from the centre it collects
        # would silently make this a different, undeclared distribution rule.
        if not np.array_equal(centres[i], np.asarray(site.origin, dtype=float)):
            raise ValueError(f'Auxiliary centre order must match site order exactly; site '
                             f'{site.label} at {tuple(float(x) for x in site.origin)} '
                             f'against centre {tuple(float(x) for x in centres[i])}')
    return tuple(s.label for s in sites), tuple(tuple(float(x) for x in s.origin) for s in sites)


def analytic_df_centre_multipoles(auxiliary, sites, rank, *, expansion=None,
                                  charge_tolerance=1.0e-10):
    """The DF-centre rule in closed form, with every precondition gated.

    ``auxiliary`` is the recipe's ``BasisRecipe`` (Cartesian GAMINT); ``sites``
    are the recipe's site declarations, one per auxiliary centre and in the same
    order.  No grid, no partition and no ISA-A fixed point enters.
    """
    labels, origins = _declared_axes(auxiliary, sites, rank)
    built = auxiliary.build('MolecularAux')
    convention_error, convention_magnitude = verify_cartesian_convention(auxiliary, built)
    if expansion is None:
        expansion = harmonic_expansion(rank)
    elif expansion.rank != rank:
        raise ValueError(f'Supplied harmonic expansion is rank {expansion.rank}, not {rank}')
    m = (rank+1)**2
    nfunction = sum(len(CARTESIAN_POWERS[s.l]) for s in auxiliary.shells)
    q = np.zeros((len(sites)*m, nfunction))
    column = 0
    for shell in auxiliary.shells:
        base = shell.centre*m
        for powers in CARTESIAN_POWERS[shell.l]:
            norm = cartesian_normalization(shell.l, powers)
            offset = 0
            for l in range(rank+1):
                mons = monomials(l)
                block = expansion.block(l)
                for t in range(2*l+1):
                    value = 0.
                    for qi, mon in enumerate(mons):
                        c = block[qi, t]
                        if c == 0.:
                            continue
                        n = (mon[0]+powers[0], mon[1]+powers[1], mon[2]+powers[2])
                        # Every odd Cartesian power integrates to zero exactly;
                        # skipping them is the identity, not a screening cutoff.
                        if n[0] % 2 or n[1] % 2 or n[2] % 2:
                            continue
                        radial = 0.
                        for z, ci in zip(shell.exponents, shell.coefficients):
                            radial += ci*math.prod(
                                math.gamma(.5*(d+1))/z**(.5*(d+1)) for d in n)
                        value += c*radial
                    q[base+offset+t, column] = norm*value
                offset += 2*l+1
            column += 1
    if not np.isfinite(q).all():
        raise RuntimeError('Closed-form DF-centre Q is not finite')
    charge = closed_form_charge_rows(auxiliary, len(sites))
    charge_error = float(np.max(np.abs(q[[i*m for i in range(len(sites))]]-charge)))
    charge_magnitude = float(np.max(np.abs(charge)))
    if not np.isfinite(charge_error) or charge_error > charge_tolerance*max(charge_magnitude, 1.0):
        raise RuntimeError(f'Closed-form DF-centre charge row disagrees with the direct '
                           f'Gaussian moment by {charge_error!r}')
    return DFCentreMultipoles('analytic', rank, q, labels, origins, REPRESENTATION,
        f'DF-centre rule in closed form; CamCASP dist_polarizabilities_DF '
        f'(DistPolAlgorithm=DF); every auxiliary function wholly on its own centre; '
        f'rank {rank}; AUX {auxiliary.name}; no grid, no stockholder weight, no ISA-A '
        f'fixed point; harmonic coefficients recovered from the shipped Racah '
        f'evaluator at seed {expansion.seed}',
        dict(harmonic_residuals=expansion.residuals,
             harmonic_residual_max=expansion.residual_max,
             harmonic_seed=expansion.seed,
             convention_error=convention_error,
             convention_magnitude=convention_magnitude,
             charge_row_error=charge_error,
             charge_row_magnitude=charge_magnitude,
             quadrature_defect='none; nothing is sampled'))


def grid_df_centre_multipoles(auxiliary, sites, rank, points, weights, *,
                              charge_tolerance=None):
    """The DF-centre rule on the caller's molecular quadrature.

    This is the shipped sampling constructor with the two inputs the DF rule
    implies: ``auxiliary_sites=[a]`` keeps only the functions centred on ``a``,
    and ``shape == shape_sum`` makes the stockholder ratio identically one.  No
    new arithmetic and no new gate; the ``denominator_cutoff`` is zero because
    the ratio is exactly one everywhere and nothing may be excluded.
    """
    labels, origins = _declared_axes(auxiliary, sites, rank)
    built = auxiliary.build('MolecularAux')
    points = np.ascontiguousarray(points, dtype=float)
    weights = np.ascontiguousarray(weights, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] != weights.shape[0]:
        raise ValueError('DF-centre grid needs matching (npoint,3) points and (npoint,) weights')
    unit = np.ones(points.shape[0])
    declared = []
    for i, site in enumerate(sites):
        samples = core.IsaMultipoleSamples()
        samples.points, samples.weights = points.tolist(), weights.tolist()
        samples.shape, samples.shape_sum = unit.tolist(), unit.tolist()
        samples.auxiliary_sites = [i]
        item = core.IsaMultipoleSite()
        item.label, item.origin, item.rank = site.label, list(site.origin), rank
        item.samples = samples
        declared.append(item)
    partition = core.IsaPartitionedMultipoles(built, declared,
        f'DF-centre rule on the supplied molecular quadrature; CamCASP '
        f'dist_polarizabilities_DF (DistPolAlgorithm=DF); every auxiliary function '
        f'wholly on its own centre; rank {rank}; AUX {auxiliary.name}; '
        f'{points.shape[0]} points; stockholder ratio identically one', 0.)
    values = np.asarray(partition.values)
    m = (rank+1)**2
    charge = closed_form_charge_rows(auxiliary, len(sites))
    charge_error = float(np.max(np.abs(values[[i*m for i in range(len(sites))]]-charge)))
    if charge_tolerance is not None:
        magnitude = float(np.max(np.abs(charge)))
        if not np.isfinite(charge_error) or charge_error > charge_tolerance*max(magnitude, 1.0):
            raise RuntimeError(f'Grid DF-centre charge row is {charge_error!r} from the '
                               f'closed-form Gaussian moment, beyond the declared '
                               f'{charge_tolerance!r}')
    return DFCentreMultipoles('grid', rank, values, labels, origins, REPRESENTATION,
        partition.provenance,
        dict(grid_points=int(points.shape[0]),
             charge_row_error=charge_error,
             charge_row_magnitude=float(np.max(np.abs(charge))),
             excluded_denominators=tuple(partition.excluded_denominators),
             negative_ratios=tuple(partition.negative_ratios),
             quadrature_defect='the molecular grid error, worst in relative terms '
                               'on the charge row'))


def site_isotropic_gate(raw, rank, *, labels):
    """Refuse a DF-centre static response with a negative site isotropic scalar.

    On a MAIN-matched auxiliary set the DF rule can charge so much of a diffuse
    function's moment to a light centre that the site's own static rank-1
    isotropic response comes out negative -- measured as -5.856 on hydrogen at
    rank 3.  That is not a badly converged number to be tightened, it is a
    broken model, and it must not reach a polarizability or a C6.  ``raw`` is the
    static distributed tensor with axes ``(site, site, component, component)``.
    """
    raw = np.asarray(raw)
    m = (rank+1)**2
    if raw.ndim != 4 or raw.shape[:2] != (len(labels),)*2 or raw.shape[2:] != (m, m):
        raise ValueError('Site isotropic gate needs a (site,site,component,component) '
                         'static distributed tensor matching the declared rank')
    scalars = {}
    for i, label in enumerate(labels):
        for l in range(1, rank+1):
            lo = l*l
            trace = float(np.trace(raw[i, i, lo:lo+2*l+1, lo:lo+2*l+1]))/(2*l+1)
            scalars[(label, l)] = trace
    negative = {k: v for k, v in scalars.items() if v < 0}
    if negative:
        worst = min(negative, key=negative.get)
        raise RuntimeError(
            f'The DF-centre rule on this auxiliary basis gives a negative static site '
            f'isotropic response: site {worst[0]} rank {worst[1]} is {negative[worst]!r}. '
            f'A negative site response is a broken model, not an unconverged one; declare '
            f'a proper RI auxiliary set (the reference case uses aug-cc-pVTZ-RI) or the '
            f'stockholder ISA-A rule instead. {len(negative)} of {len(scalars)} site/rank '
            f'scalars are negative.')
    return scalars
