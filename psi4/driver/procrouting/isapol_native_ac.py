# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Declared multipole/Tozer-Handy asymptotic correction, as its own gated policy.

This module exists because a reference protocol we compare against declares its
SCF asymptotic correction as

    .DFTAC / MULTPOLE / TANH / "0.46380 0.46380 3.0 4.0"

which is a Tozer-Handy correction whose asymptotic branch is a Fermi-Amaldi
multipole potential, joined to the density-functional potential by a tanh
interpolation between 3 and 4 Bragg-Slater radii, with the monomer ionization
potential supplied as input.  That is NOT the gradient-regulated form the
``FIXED_GRAC`` policy admits, and a different asymptotic-correction form is a
different model.  It is therefore added here as a separate policy with its own
declaration, its own producer, its own convergence record and its own
validator; ``FIXED_GRAC`` and ``validate_correction`` were not widened to cover
it, and no GRAC parameter was refitted to imitate it.

The model
---------
    v_xc^AC(r) = (1 - f(x)) v_xc^DFA(r) + f(x) [ c_FA v^FA(r) + Delta ]
    v^FA(r)    = -V_H^mult(r) / N                       (Fermi-Amaldi)
    Delta      = I + eps_HOMO                           (Tozer-Handy constant)
    x(r)       = min_A |r - R_A| / R_A^BS

``Delta`` is a prediction, not a fitted parameter: adding a constant c to v_xc
shifts every eigenvalue by c, so a Kohn-Sham potential that tends to -1/r can
only be consistent with a HOMO at eps_HOMO if it tends to -1/r + (I+eps_HOMO).
With ``shift_mode='variational'`` it is recomputed from the current HOMO each
iteration, which is what makes the declaration IP-driven rather than
shift-driven.  ``c_FA = 1 - a_x`` for a global hybrid, because the a_x fraction
of exchange is already the asymptotically correct nonlocal operator; that too
is a derivation, but it is carried as a declared field so that a bracket can
report other values instead of hiding one.

Implementation facts that are part of the model's identity
---------------------------------------------------------
* The splice is applied to the POINTWISE potential, never to the GGA integrand.
  Weighting the integrand by (1-f) instead leaves a grad(f) surface term
  supported in the splice shell, and that term is not small here: it is worth
  +0.22 Eh in the LUMO expectation value for water/aug-cc-pVTZ.  Concretely
  Psi4's own XC matrix is used unchanged and the local correction
  f*[c_FA v^FA + Delta - v_xc^{DFA,pw}] is added to it, so the inner region is
  bit-identical to Psi4's PBE0 and the pointwise DFA potential is only ever
  evaluated where f > 0.
* v_xc^{DFA,pw} = v_rho - 2[grad(v_gamma).grad(rho) + v_gamma lap(rho)], with
  grad(v_gamma) = v_{rho,gamma} grad(rho) + v_{gamma,gamma} grad(gamma) and
  grad(gamma) = 2 (grad grad rho).grad(rho).  Integrating this pointwise
  potential does not reproduce Psi4's integrand-form XC matrix near a nucleus,
  which is exactly why f must vanish identically inside b1 rather than merely
  be small there.
* Every join form here returns f identically zero inside b1.  The asymptotic
  branch is a multipole expansion with a pole at its origin, so a raw tanh that
  is only ~1e-6 at a nucleus multiplies a pole and is not negligible.  How the
  join is made to vanish is an implementation choice, and therefore a declared
  join variant, not a repair.
* The correction has no energy functional.  ``energy`` is the plain hybrid
  functional evaluated at the corrected density, so it must lie ABOVE the plain
  SCF minimum; a lower value would mean the correction had been mislabelled.
* The multipole moments are analytic AO multipole integrals of the total
  density, not grid quadrature.

What this module deliberately does not do
-----------------------------------------
The iteration here is not Psi4's SCF and produces no SCF seal.  It verifies the
seal of the uncorrected wavefunction it starts from, records that it did, and
then invalidates that seal by replacing the orbitals; ``require_scf_seal`` will
correctly refuse the result afterwards.  ``validate_declared_ac`` checks the
module's own record instead, and no seal is ever manufactured.  Orbitals
produced here may only enter the response chain through this policy, so that
their provenance is never reported as "no SCF asymptotic correction".
"""
import dataclasses
import math
from numbers import Real
import numpy as np
from psi4 import core
from .isapol_native_correction import require_scf_seal

AC_POLICY = 'DECLARED_MULTPOLE_AC'

#: Bragg-Slater radii in bohr.  ``psi4`` is the table in ``cubature.cc``;
#: ``camcasp`` is ``libisapol/tables.cc`` ``rslater_ang`` converted with that
#: file's own bohr.  They differ for hydrogen, so they are different models.
CAMCASP_BOHR = 0.529177249
BRAGG_SLATER_TABLES = {
    'psi4': {1: 0.661, 2: 0.661, 3: 2.740, 4: 1.984, 5: 1.606,
             6: 1.323, 7: 1.228, 8: 1.134, 9: 0.945, 10: 0.900},
    'camcasp': {1: 0.50/CAMCASP_BOHR, 2: 0.50/CAMCASP_BOHR, 6: 0.70/CAMCASP_BOHR,
                7: 0.65/CAMCASP_BOHR, 8: 0.60/CAMCASP_BOHR, 9: 0.50/CAMCASP_BOHR,
                10: 0.50/CAMCASP_BOHR},
}
JOIN_FORMS = ('none', 'linear', 'tanh', 'tanh_raw')
ORIGINS = ('nuccharge', 'com', 'atom0')
SHIFT_MODES = ('none', 'variational', 'fixed')

#: Strictest thresholds a caller may declare for the AC iteration.  These are
#: admission limits, not defaults: a looser declaration is refused rather than
#: accepted with a warning.
MAX_ENERGY_THRESHOLD = 1.e-8
MAX_GRADIENT_THRESHOLD = 1.e-6

_KEYS = ('PHI', 'PHI_X', 'PHI_Y', 'PHI_Z',
         'PHI_XX', 'PHI_YY', 'PHI_ZZ', 'PHI_XY', 'PHI_XZ', 'PHI_YZ')
_D1 = ('PHI_X', 'PHI_Y', 'PHI_Z')
_HESS = {(0, 0): 'PHI_XX', (1, 1): 'PHI_YY', (2, 2): 'PHI_ZZ',
         (0, 1): 'PHI_XY', (0, 2): 'PHI_XZ', (1, 2): 'PHI_YZ'}


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


@dataclasses.dataclass(frozen=True)
class AcDeclaration:
    """One asymptotic-correction model.  Every field is part of its identity.

    Nothing here is inferred from an option name, a method name or a
    wavefunction.  ``ionization_potential`` in particular is a declared input in
    Hartree, never computed from a HOMO or a Delta-SCF.
    """
    ionization_potential: float
    join: str = 'tanh'
    b1: float = 3.0
    b2: float = 4.0
    tanh_k: float = 1.0
    bragg_table: str = 'psi4'
    fa_scale: float = 0.75
    multipole_order: int = 2
    origin: str = 'nuccharge'
    shift_mode: str = 'variational'
    shift_value: float = 0.0

    def __post_init__(self):
        if self.join not in JOIN_FORMS:
            raise ValueError(f'unknown asymptotic-correction join form; expected one of {JOIN_FORMS}')
        if self.bragg_table not in BRAGG_SLATER_TABLES:
            raise ValueError('unknown Bragg-Slater radius table declaration')
        if self.origin not in ORIGINS:
            raise ValueError('unknown multipole origin declaration')
        if self.shift_mode not in SHIFT_MODES:
            raise ValueError('unknown Tozer-Handy shift mode declaration')
        if type(self.multipole_order) is not int or not 0 <= self.multipole_order <= 3:
            raise ValueError('multipole order must be an explicit integer 0..3')
        if not all(_finite(v) for v in (self.ionization_potential, self.b1, self.b2,
                                        self.tanh_k, self.fa_scale, self.shift_value)):
            raise ValueError('asymptotic-correction declaration requires finite real scalars')
        if self.ionization_potential <= 0:
            raise ValueError('declared ionization potential must be positive (Hartree)')
        if not 0. < self.b1 < self.b2:
            raise ValueError('splice radii must satisfy 0 < b1 < b2 (Bragg-Slater radii)')
        if self.tanh_k <= 0:
            raise ValueError('tanh sharpness must be positive')
        if not 0. <= self.fa_scale <= 1.:
            raise ValueError('Fermi-Amaldi scale must lie in [0, 1]')
        for name in ('ionization_potential', 'b1', 'b2', 'tanh_k', 'fa_scale', 'shift_value'):
            object.__setattr__(self, name, float(getattr(self, name)))

    def label(self):
        shift = {'none': 'noshift', 'variational': 'var',
                 'fixed': 'fix%+.6f' % self.shift_value}[self.shift_mode]
        return ('%s_b%g-%g_k%g_%s_fa%.2f_L%d_%s_%s_ip%.5f'
                % (self.join, self.b1, self.b2, self.tanh_k, self.bragg_table,
                   self.fa_scale, self.multipole_order, self.origin, shift,
                   self.ionization_potential))


#: The reference protocol's own declaration for the water monomer, transcribed
#: from its input file: MULTPOLE asymptotic branch, TANH join over 3->4 Bragg
#: radii, IP 0.46380 Eh.  ``fa_scale`` and ``shift_mode`` are the derivations
#: named in the module docstring, not part of the transcription.
REFERENCE_WATER_DECLARATION = AcDeclaration(ionization_potential=0.46380)


@dataclasses.dataclass(frozen=True)
class AcConvergence:
    """Achieved AC iteration diagnostics.  Not an SCF seal and not a claim of one."""
    iterations: int
    delta_energy: float
    orbital_gradient: float
    energy_threshold: float
    gradient_threshold: float
    shift: float
    shift_clamped: int
    homo: float
    lumo: float
    energy: float
    reference_energy: float
    grid_points: int


@dataclasses.dataclass(frozen=True)
class DeclaredAcProvenance:
    """Owned snapshot of the declared correction actually in the orbitals.

    ``shift`` is the converged Tozer-Handy constant, which is part of the
    potential's identity and so part of the policy fingerprint.
    """
    policy: str
    declaration: AcDeclaration
    shift: float
    iterations: int

    @property
    def response_description(self):
        return ('ALDA response using orbitals from the declared MULTPOLE/'
                f'{self.declaration.join.upper()} asymptotic correction '
                '(Fermi-Amaldi asymptotic branch, Tozer-Handy constant '
                f'{self.shift:+.6f} Eh); no asymptotic-correction kernel derivative')


@dataclasses.dataclass(eq=False, frozen=True)
class DeclaredAcOrbitals:
    """Producer output.  Holds owned arrays; applies nothing by itself."""
    declaration: AcDeclaration
    orbitals: np.ndarray
    energies: np.ndarray
    density: np.ndarray
    fock: np.ndarray
    convergence: AcConvergence
    nocc: int


def multipole_origin(mol, which):
    geom = np.asarray(mol.geometry().to_array(), dtype=float)
    if which == 'atom0':
        return geom[0].copy()
    z = np.array([mol.Z(i) for i in range(mol.natom())], dtype=float)
    if which == 'nuccharge':
        return (z[:, None]*geom).sum(0)/z.sum()
    m = np.array([mol.mass(i) for i in range(mol.natom())], dtype=float)
    return (m[:, None]*geom).sum(0)/m.sum()


def splice_weight(mol, grid, declaration):
    """Per-block (f(x), r-O, |r-O|); f is identically zero inside b1."""
    geom = np.asarray(mol.geometry().to_array(), dtype=float)
    table = BRAGG_SLATER_TABLES[declaration.bragg_table]
    try:
        radii = np.array([table[mol.Z(i)] for i in range(mol.natom())], dtype=float)
    except KeyError:
        raise ValueError('declared Bragg-Slater table has no radius for an element present; '
                         'no fallback radius is substituted')
    origin = multipole_origin(mol, declaration.origin)
    xm = .5*(declaration.b1 + declaration.b2)
    hw = .5*(declaration.b2 - declaration.b1)
    weights, vectors, norms = [], [], []
    for block in grid.blocks():
        n = block.npoints()
        points = np.column_stack((np.asarray(block.x())[:n], np.asarray(block.y())[:n],
                                  np.asarray(block.z())[:n]))
        x = (np.linalg.norm(points[:, None, :] - geom[None, :, :], axis=2)
             / radii[None, :]).min(1)
        if declaration.join == 'none':
            f = np.zeros(n)
        elif declaration.join == 'linear':
            f = np.clip((x - declaration.b1)/(declaration.b2 - declaration.b1), 0., 1.)
        elif declaration.join == 'tanh':
            # tanh rescaled to run exactly 0 -> 1 across [b1, b2]
            t = np.tanh(declaration.tanh_k*(x - xm)/hw)
            lo, hi = np.tanh(-declaration.tanh_k), np.tanh(declaration.tanh_k)
            f = np.clip((t - lo)/(hi - lo), 0., 1.)
        else:
            # the unrescaled tanh, truncated to zero inside b1
            f = .5*(1. + np.tanh(declaration.tanh_k*(x - xm)/hw))
            f = np.where(x < declaration.b1, 0., f)
        vector = points - origin[None, :]
        weights.append(f)
        vectors.append(vector)
        norms.append(np.linalg.norm(vector, axis=1))
    return weights, vectors, norms


def density_moments(mints, density, origin, order):
    """Raw Cartesian moments of the TOTAL electron number density, analytic.

    ``ao_multipoles`` carries the electron charge, so orders >= 1 are negated
    back to number-density moments; order 0 is the overlap trace.
    """
    D = np.asarray(density)
    m0 = 2.*np.einsum('mn,mn->', D, np.asarray(mints.ao_overlap()))
    m1 = np.zeros(3)
    m2 = np.zeros((3, 3))
    m3 = np.zeros((3, 3, 3))
    if order >= 1:
        mats = mints.ao_multipoles(order, list(map(float, origin)))
        values = [-2.*np.einsum('mn,mn->', D, np.asarray(m)) for m in mats]
        m1[:] = values[:3]
        if order >= 2:
            it = iter(values[3:9])
            for a in range(3):
                for b in range(a, 3):
                    v = next(it)
                    m2[a, b] = m2[b, a] = v
        if order >= 3:
            it = iter(values[9:19])
            for a in range(3):
                for b in range(a, 3):
                    for c in range(b, 3):
                        v = next(it)
                        for p in ((a, b, c), (a, c, b), (b, a, c),
                                  (b, c, a), (c, a, b), (c, b, a)):
                            m3[p] = v
    return m0, m1, m2, m3


def multipole_potential(vector, norm, moments, order):
    """V_H(r) from the truncated raw-Cartesian multipole expansion, order <= 3."""
    m0, m1, m2, m3 = moments
    u = vector/norm[:, None]
    v = m0/norm
    if order >= 1:
        v = v + np.einsum('pa,a->p', u, m1)/norm**2
    if order >= 2:
        q = np.einsum('pa,pb,ab->p', u, u, m2)
        v = v + .5*(3.*q - np.trace(m2))/norm**3
    if order >= 3:
        t = np.einsum('pa,pb,pc,abc->p', u, u, u, m3)
        s = np.einsum('pa,a->p', u, np.einsum('abb->a', m3))
        v = v + (15.*t - 9.*s)/(6.*norm**4)
    return v


class _AcKohnSham:
    """Restricted Kohn-Sham with a pointwise-spliced declared AC.

    Runs on the wavefunction's OWN exchange-correlation grid, so that the
    unspliced inner region is the same quadrature Psi4's SCF used.
    """

    def __init__(self, wfn, declaration, shift_damping=.5):
        self.wfn, self.declaration = wfn, declaration
        self.mol = wfn.molecule()
        self.basis = wfn.basisset()
        self.nbf = self.basis.nbf()
        self.V = wfn.V_potential()
        if self.V is None:
            raise ValueError('declared asymptotic correction requires the SCF exchange-correlation '
                             'potential and its grid; no grid is rebuilt here')
        self.grid = self.V.grid()
        self.ax = self.V.functional().x_alpha()
        # A second worker at deriv=2 for the pointwise potential.  The caller's
        # gate has already established an unmodified canonical hybrid, so this
        # is that functional rebuilt, not a substituted one.
        self.sf2 = core.SuperFunctional.XC_build('XC_HYB_GGA_XC_PBEH', True)
        self.sf2.set_max_points(self.grid.max_points())
        self.sf2.set_deriv(2)
        self.sf2.allocate()
        self.bfn = core.BasisFunctions(self.basis, self.grid.max_points(), self.nbf)
        self.bfn.set_deriv(2)
        self.f, self.rvec, self.rnorm = splice_weight(self.mol, self.grid, declaration)
        # Exact screen, not a tolerance: every join form vanishes identically
        # inside b1, so a block with f == 0 everywhere contributes nothing and
        # the pointwise potential is never evaluated there.
        self.active = [i for i, f in enumerate(self.f) if np.any(f > 0.)]
        self.mints = core.MintsHelper(self.basis)
        self.S = np.asarray(self.mints.ao_overlap())
        self.H = np.asarray(wfn.H())
        self.enuc = self.mol.nuclear_repulsion_energy()
        self.nocc = wfn.nalpha()
        self.nelec = wfn.nalpha() + wfn.nbeta()
        self.origin = multipole_origin(self.mol, declaration.origin)
        self.jk = core.JK.build(self.basis)
        self.jk.set_do_K(True)
        self.jk.initialize()
        s, U = np.linalg.eigh(self.S)
        self.X = U @ np.diag(s**-.5) @ U.T
        # Damping on the Tozer-Handy fixed point.  A numerical control of the
        # iteration, not a parameter of the declared potential: at the fixed
        # point shift == I + eps_HOMO for any damping in (0, 1].
        self.shift_damping = float(shift_damping)

    def ac_correction(self, Da, shift):
        """Local matrix of f*[c_FA v^FA + Delta - v_xc^{DFA,pw}]."""
        declaration = self.declaration
        if declaration.join == 'none' or not self.active:
            return np.zeros((self.nbf, self.nbf))
        blocks = self.grid.blocks()
        records = []
        for index in self.active:
            block = blocks[index]
            n = block.npoints()
            mask = self.f[index] > 0.
            self.bfn.compute_functions(block)
            values = self.bfn.basis_values()
            local = np.asarray(block.functions_local_to_global())
            nl = len(local)
            P = {k: np.asarray(values[k])[:n, :nl][mask] for k in _KEYS}
            m = int(mask.sum())
            D = Da[np.ix_(local, local)]
            t = P['PHI'] @ D
            rho = np.einsum('pm,pm->p', t, P['PHI'])
            grad = np.empty((m, 3))
            for i, key in enumerate(_D1):
                grad[:, i] = 2.*np.einsum('pm,pm->p', t, P[key])
            gamma = (grad*grad).sum(1)
            hess = np.empty((m, 3, 3))
            for (a, b), key in _HESS.items():
                ta = P[_D1[a]] @ D
                v = (2.*np.einsum('pm,pm->p', t, P[key])
                     + 2.*np.einsum('pm,pm->p', ta, P[_D1[b]]))
                hess[:, a, b] = v
                hess[:, b, a] = v
            out = self.sf2.compute_functional(
                {'RHO_A': core.Vector.from_array(rho),
                 'GAMMA_AA': core.Vector.from_array(gamma)}, m)
            v_rho = np.asarray(out['V_RHO_A'])[:m]
            v_gamma = np.asarray(out['V_GAMMA_AA'])[:m]
            v_rho_gamma = np.asarray(out['V_RHO_A_GAMMA_AA'])[:m]
            v_gamma_gamma = np.asarray(out['V_GAMMA_AA_GAMMA_AA'])[:m]
            grad_gamma = 2.*np.einsum('pab,pb->pa', hess, grad)
            grad_v_gamma = v_rho_gamma[:, None]*grad + v_gamma_gamma[:, None]*grad_gamma
            pointwise = v_rho - 2.*(np.einsum('pa,pa->p', grad_v_gamma, grad)
                                    + v_gamma*np.einsum('paa->p', hess))
            w = np.asarray(block.w())[:n][mask]
            records.append((index, mask, local, w, P['PHI'], pointwise))
        moments = density_moments(self.mints, Da, self.origin, declaration.multipole_order)
        Vm = np.zeros((self.nbf, self.nbf))
        for index, mask, local, w, phi, pointwise in records:
            f = self.f[index][mask]
            asymptotic = -declaration.fa_scale*multipole_potential(
                self.rvec[index][mask], self.rnorm[index][mask], moments,
                declaration.multipole_order)/self.nelec
            correction = f*(asymptotic + shift - pointwise)
            Vm[np.ix_(local, local)] += phi.T @ ((correction*w)[:, None]*phi)
        return Vm

    def run(self, guess_C, maxiter, energy_threshold, gradient_threshold, diis_subspace):
        declaration = self.declaration
        nbf, nocc = self.nbf, self.nocc
        C = np.array(guess_C, dtype=float)
        Da = C[:, :nocc] @ C[:, :nocc].T
        previous, focks, errors = 0., [], []
        shift, spectrum, clamped, delta = 0., None, 0, float('inf')
        gradient, energy, F = float('inf'), float('nan'), None
        converged, iteration = False, 0
        for iteration in range(maxiter):
            occupied = core.Matrix.from_array(C[:, :nocc])
            self.jk.C_left_add(occupied)
            self.jk.C_right_add(occupied)
            self.jk.compute()
            J = np.asarray(self.jk.J()[0])
            K = np.asarray(self.jk.K()[0])
            self.jk.C_clear()
            Vm = core.Matrix(nbf, nbf)
            self.V.set_D([core.Matrix.from_array(Da)])
            self.V.compute_V([Vm])
            exc = self.V.quadrature_values()['FUNCTIONAL']
            F0 = self.H + 2.*J - self.ax*K + np.asarray(Vm)
            # The Tozer-Handy constant is a fixed-point condition on the
            # CORRECTED spectrum.  It is read from the previous iteration's
            # un-extrapolated Fock, never from a DIIS extrapolate (whose
            # spectrum belongs to no density), seeded from the uncorrected Fock
            # at the guess density, damped, and clamped to the interval the
            # condition itself admits for a bound HOMO.  A clamp hit is
            # reported, and refused by the validator, not absorbed.
            if declaration.shift_mode == 'fixed':
                shift = declaration.shift_value
            elif declaration.shift_mode == 'variational':
                ip = declaration.ionization_potential
                if spectrum is None:
                    spectrum = np.linalg.eigvalsh(self.X.T @ F0 @ self.X)
                    shift = ip + spectrum[nocc-1]
                else:
                    shift += self.shift_damping*(ip + spectrum[nocc-1] - shift)
                if not -abs(ip) <= shift <= 2.*abs(ip):
                    clamped += 1
                    shift = min(max(shift, -abs(ip)), 2.*abs(ip))
            else:
                shift = 0.
            F = F0 + self.ac_correction(Da, shift)
            spectrum = np.linalg.eigvalsh(self.X.T @ F @ self.X)
            energy = (2.*np.einsum('mn,mn->', Da, self.H) + 2.*np.einsum('mn,mn->', Da, J)
                      - self.ax*np.einsum('mn,mn->', Da, K) + exc + self.enuc)
            error = self.X.T @ (F @ Da @ self.S - self.S @ Da @ F) @ self.X
            gradient = float(np.abs(error).max())
            delta = energy - previous
            if iteration and abs(delta) < energy_threshold and gradient < gradient_threshold:
                converged = True
                break
            previous = energy
            focks.append(F)
            errors.append(error)
            if len(focks) > diis_subspace:
                focks.pop(0)
                errors.pop(0)
            nd = len(focks)
            B = np.empty((nd+1, nd+1))
            B[-1, :] = -1.
            B[:, -1] = -1.
            B[-1, -1] = 0.
            for i in range(nd):
                for j in range(nd):
                    B[i, j] = np.einsum('mn,mn->', errors[i], errors[j])
            rhs = np.zeros(nd+1)
            rhs[-1] = -1.
            try:
                c = np.linalg.solve(B, rhs)[:-1]
                Fd = sum(ci*fi for ci, fi in zip(c, focks))
            except np.linalg.LinAlgError:
                Fd = F
            _, Cp = np.linalg.eigh(self.X.T @ Fd @ self.X)
            C = self.X @ Cp
            Da = C[:, :nocc] @ C[:, :nocc].T
        # A final un-extrapolated diagonalisation, so the reported orbitals and
        # eigenvalues belong to F[D] rather than to an extrapolate.
        spectrum, Cp = np.linalg.eigh(self.X.T @ F @ self.X)
        C = self.X @ Cp
        # The reported density is rebuilt from the REPORTED orbitals, which is
        # Psi4's own SCF convention and the native response contract's exact
        # requirement (D == C_occ C_occ^T, not merely within the convergence
        # threshold).  F and the energy remain those of the immediately
        # preceding density; the difference between the two densities is bounded
        # by the reported delta_energy/orbital_gradient and is not absorbed into
        # any tolerance -- it is the same one-step lag Psi4's SCF reports.
        Da = C[:, :nocc] @ C[:, :nocc].T
        return C, spectrum, Da, F, AcConvergence(
            iterations=int(iteration), delta_energy=float(delta),
            orbital_gradient=gradient, energy_threshold=float(energy_threshold),
            gradient_threshold=float(gradient_threshold), shift=float(shift),
            shift_clamped=int(clamped), homo=float(spectrum[nocc-1]),
            lumo=float(spectrum[nocc]), energy=float(energy),
            reference_energy=float(self.wfn.energy()),
            grid_points=int(self.grid.npoints())), converged


def declared_ac_orbitals(wfn, declaration, *, maxiter=200, energy_threshold=1.e-10,
                         gradient_threshold=1.e-8, diis_subspace=10, shift_damping=.5):
    """Iterate the declared asymptotic correction from a sealed SCF wavefunction.

    Returns a producer record; nothing is applied to ``wfn`` and no seal is
    created.  The starting wavefunction must carry a current successful SCF seal
    and an unmodified canonical hybrid with no GRAC attachment, because the
    correction is spliced onto that functional's own potential and grid.
    """
    from .isapol_native_correction import validate_correction
    if not isinstance(declaration, AcDeclaration):
        raise TypeError('explicit AcDeclaration required')
    if declaration.join == 'none':
        raise ValueError('join=none declares no asymptotic branch; that is the NONE policy')
    if declaration.shift_mode == 'variational' and not 0. < shift_damping <= 1.:
        raise ValueError('shift damping must lie in (0, 1]')
    if type(maxiter) is not int or maxiter < 1 or type(diis_subspace) is not int or diis_subspace < 1:
        raise ValueError('maxiter and diis_subspace must be positive integers')
    if not (_finite(energy_threshold) and 0 < energy_threshold <= MAX_ENERGY_THRESHOLD):
        raise ValueError(f'AC energy threshold must be positive and <= {MAX_ENERGY_THRESHOLD}')
    if not (_finite(gradient_threshold) and 0 < gradient_threshold <= MAX_GRADIENT_THRESHOLD):
        raise ValueError(f'AC gradient threshold must be positive and <= {MAX_GRADIENT_THRESHOLD}')
    # The uncorrected state must itself be a sealed canonical SCF.  This is the
    # only seal in the AC path, it is verified rather than manufactured, and
    # applying the result deliberately invalidates it.
    validate_correction(wfn, scf_correction='NONE', require_canonical=True)
    require_scf_seal(wfn)
    if wfn.nalpha() != wfn.nbeta() or wfn.nirrep() != 1:
        raise ValueError('declared asymptotic correction supports restricted C1 only')
    driver = _AcKohnSham(wfn, declaration, shift_damping=shift_damping)
    C, spectrum, Da, F, convergence, converged = driver.run(
        np.asarray(wfn.Ca()), maxiter, energy_threshold, gradient_threshold, diis_subspace)
    if not converged:
        raise ValueError('declared asymptotic-correction iteration did not converge; '
                         'no unconverged orbitals are returned')
    if convergence.energy < convergence.reference_energy:
        raise ValueError('declared asymptotic correction has no energy functional, so its energy '
                         'must exceed the plain SCF minimum; a lower value means a mislabel')
    return DeclaredAcOrbitals(declaration, np.ascontiguousarray(C),
                              np.ascontiguousarray(spectrum), np.ascontiguousarray(Da),
                              np.ascontiguousarray(F), convergence, int(driver.nocc))


def apply_declared_ac(wfn, record):
    """Replace ``wfn``'s orbitals with the declared-AC ones and record the fact.

    This is an explicit, named mutation.  It invalidates the SCF seal, which is
    correct: the state is no longer the one Psi4's SCF converged.  The energy is
    replaced by the plain functional evaluated at the corrected density, which is
    not a variational minimum and is labelled as such in the record.
    """
    from .scf_proc.scf_iterator import _scf_state_signature
    if not isinstance(record, DeclaredAcOrbitals):
        raise TypeError('explicit DeclaredAcOrbitals record required')
    if getattr(wfn, '_declared_ac_evidence', None) is not None:
        raise ValueError('wavefunction already carries a declared asymptotic correction; '
                         'corrections are not composed')
    require_scf_seal(wfn)
    if (wfn.basisset().nbf(), wfn.nalpha()) != (record.orbitals.shape[0], record.nocc):
        raise ValueError('declared-AC record does not match this wavefunction basis/occupation')
    if record.convergence.reference_energy != wfn.energy():
        raise ValueError('declared-AC record was produced from a different SCF state')
    for getter, value in ((wfn.Ca, record.orbitals), (wfn.Cb, record.orbitals),
                          (wfn.Da, record.density), (wfn.Db, record.density),
                          (wfn.Fa, record.fock), (wfn.Fb, record.fock),
                          (wfn.epsilon_a, record.energies), (wfn.epsilon_b, record.energies)):
        np.asarray(getter())[:] = value
    wfn.set_energy(record.convergence.energy)
    if wfn.V_potential() is not None:
        # Leave the potential's density pointer consistent with the new state.
        wfn.V_potential().set_D([wfn.Da()])
    wfn._declared_ac_evidence = (record.declaration, record.convergence,
                                 _scf_state_signature(wfn), wfn.basisset())
    return wfn


def validate_declared_ac(wfn, declaration):
    """Verify the declared-AC record against the caller's declaration.

    The caller must declare the same model that actually produced the orbitals;
    the record is never used as the declaration, and there is no SCF seal to
    check because the AC iteration is not Psi4's SCF.
    """
    from .scf_proc.scf_iterator import _scf_state_signature
    if not isinstance(declaration, AcDeclaration):
        raise TypeError(f'{AC_POLICY} requires an explicit AcDeclaration')
    if declaration.join == 'none':
        raise ValueError(f'{AC_POLICY} requires an actual asymptotic branch; join=none is NONE')
    evidence = getattr(wfn, '_declared_ac_evidence', None)
    if evidence is None:
        raise ValueError('Declared asymptotic-correction evidence is required; '
                         'native properties never run the AC iteration')
    recorded, convergence, signature, basis = evidence
    if not isinstance(recorded, AcDeclaration) or not isinstance(convergence, AcConvergence):
        raise ValueError('declared asymptotic-correction evidence is malformed')
    if recorded != declaration:
        raise ValueError('declared asymptotic correction does not match the orbitals: '
                         f'recorded {recorded.label()}, declared {declaration.label()}')
    if basis != wfn.basisset() or signature != _scf_state_signature(wfn):
        raise ValueError('declared asymptotic-correction evidence is stale: '
                         'wavefunction state has changed')
    if (convergence.iterations < 1 or convergence.shift_clamped != 0
            or not all(_finite(v) for v in (convergence.delta_energy, convergence.orbital_gradient,
                                            convergence.energy_threshold, convergence.gradient_threshold,
                                            convergence.shift, convergence.energy,
                                            convergence.reference_energy))
            or not abs(convergence.delta_energy) < convergence.energy_threshold
            or not 0 <= convergence.orbital_gradient < convergence.gradient_threshold):
        raise ValueError('declared asymptotic-correction stopping diagnostics are inconsistent')
    if not (0 < convergence.energy_threshold <= MAX_ENERGY_THRESHOLD
            and 0 < convergence.gradient_threshold <= MAX_GRADIENT_THRESHOLD):
        raise ValueError('declared asymptotic correction was converged to a looser threshold '
                         'than admission allows')
    if convergence.energy < convergence.reference_energy:
        raise ValueError('declared asymptotic correction reports an energy below the plain SCF '
                         'minimum; the correction has no energy functional')
    if declaration.shift_mode == 'variational':
        target = declaration.ionization_potential + convergence.homo
        if abs(convergence.shift - target) > 1.e-6:
            raise ValueError('variational Tozer-Handy constant is not at its fixed point: '
                             f'shift {convergence.shift!r} vs I + eps_HOMO {target!r}')
    elif declaration.shift_mode == 'fixed' and convergence.shift != declaration.shift_value:
        raise ValueError('fixed Tozer-Handy constant does not match the declaration')
    return DeclaredAcProvenance(AC_POLICY, declaration, float(convergence.shift),
                                int(convergence.iterations))
