"""DF-centre distributed multipoles: CamCASP's ``DistPolAlgorithm = 'DF'`` rule.

CamCASP's distributed-polarizability default is ``'DF'``
(``polarizability.F90:219``), whose ``dist_polarizabilities_DF`` (line 1593)
assigns each auxiliary function **wholly** to the centre that function sits on
(``centre = 0``, ``isitdist = .true.``) and then forms
``alpha^(a,b)_{t,u}(w) = - Qa(k,t) C_{k,l}(w) Qb(l,u)``.  No stockholder
weight is formed.  This is a different declared model from a stockholder
(ISA/MBIS) partition; their site properties must not be quoted as agreeing.

Every factor is co-centred, so the rule is produced in closed form:

    Q(a,(l,m),k) = N_k sum_i c_i sum_q C_q^{lm}
                   prod_d Gamma((n_d+1)/2) / zeta_i**((n_d+1)/2)

with ``n = q + p(k)`` and odd powers vanishing.  No quadrature error.

``C^{lm}`` is recovered from ``core.isa_regular_multipoles`` and the Cartesian
convention from ``IsaExplicitBasis.evaluate``; both recoveries are gated and
refused on mismatch.

The rule is basis-sensitive: each function's whole moment goes to its own
centre.  It is well behaved on the reference's aug-cc-pVTZ-RI auxiliary set; on
a MAIN-matched JKFIT set static blocks reach 1.3e+03 while summing to about 9.8,
and the H rank-3 scalar is negative.  No tolerance repairs that.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from psi4 import core

#: GAMINT Cartesian power order, matching ``IsaExplicitBasis``.  Part of the
#: column identity of every AUX matrix; do not sort.
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

#: Fixed seeds for the two recoveries.  Any draw gives the same exact polynomial
#: coefficients to the gate; pinning makes Q bitwise reproducible.
HARMONIC_SEED = 20260910
CONVENTION_SEED = 20260911
#: Worst admissible reconstruction error; observed ~1e-14, so a real ordering,
#: sign or normalization change still fails.
HARMONIC_TOLERANCE = 1.0e-10
CONVENTION_TOLERANCE = 1.0e-10

#: DF-centre Q has fitted-density AUX columns.
REPRESENTATION = 'fitted_density_coefficients'


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

    ``R_lm`` is homogeneous of degree ``l``, so its coefficients are solved from
    its values, inheriting the evaluator's sign, order and normalization.
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

    The closed form walks ``basis.shells`` with ``CARTESIAN_POWERS``; this checks
    that walk reproduces ``IsaExplicitBasis.evaluate`` column for column.
    Returns ``(error, magnitude)``; raises if error exceeds ``tolerance`` relative
    to the magnitude.
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

    Independent rank-0 cross-check of the closed form (``R_00 = 1``), from
    Gaussian moments without the harmonic expansion.
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
    are the recipe's site declarations, one per auxiliary centre, in order.
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
        f'rank {rank}; AUX {auxiliary.name}; no grid, no stockholder weight, no '
        f'stockholder fixed point; harmonic coefficients recovered from the shipped Racah '
        f'evaluator at seed {expansion.seed}',
        dict(harmonic_residuals=expansion.residuals,
             harmonic_residual_max=expansion.residual_max,
             harmonic_seed=expansion.seed,
             convention_error=convention_error,
             convention_magnitude=convention_magnitude,
             charge_row_error=charge_error,
             charge_row_magnitude=charge_magnitude,
             quadrature_defect='none; nothing is sampled'))
