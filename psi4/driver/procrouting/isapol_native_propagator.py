# Psi4: Copyright (c) 2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""Declared density-fitted propagator and fitted-density ALDA kernel.

The shipped native response builds H1/H2 from EXACT orbital-product two-electron
integrals and an EXACT-density ALDA kernel accumulated over orbital products.
The reference protocol declares something else, and says so in its own input:

    SET PROPAGATOR  Type CKS  Hessians Internal  DF with constraints  DF-integrals
    SET DF-INTEGRALS  DF-TYPE-MONOMER NN
    SET NEW-PROP ... KERNEL-INTEGRAL-PARAMETERS (INFINITY-CONTROL-METHOD FD,
        RHO-EPS=1e-8, F-MAX=1000.0, FD-DELTA=0.01, FD-ALPHA=1.0)

Three separate things are declared there, and this module keeps them separate
because each one is a DIFFERENT MODEL, never a refinement or a tolerance:

1. ``two_electron``: the (ia|jb)/(ij|ab)/(ib|aj) operators are density fitted in
   the molecular AUX basis rather than assembled from exact orbital products.
2. ``kernel_projection``: the ALDA kernel is built as an naux x naux matrix over
   the molecular AUX basis and then projected through the fitted transition
   density (CamCASP ``num_integrals.F90``: ``S_kl = sum_p w_p fxc(p) chi_k(p)
   chi_l(p)``), rather than accumulated over exact orbital products.
3. ``kernel_density``: ``fxc`` is evaluated at the CONSTRAINED DF monomer density
   (``evaluate_FuncExpansion(Rho, ...)``, the same ``Drho-C`` fit the partition
   already owns) rather than at the exact orbital density ``2 sum_i phi_i^2``.
   ``smoothing`` is a fourth, independent declaration: the floor at
   ``Kernel_Rho_Epsilon`` and the Fermi-Dirac cap of ``F-MAX``.

A number produced under any one combination may never be quoted as agreeing with
a number produced under another, and none of them may be quoted beside a number
recorded under the shipped exact-orbital propagator.

Not replicated, and deliberately labelled rather than approximated: the AUX
shell-pair overlap screen at ``KERNEL-INTEGRAL-CUTOFF`` (``num_integrals.F90``
line 205). ``IsaExplicitBasis`` exposes no shell partition to Python, so the
screen cannot be reproduced here; this module integrates every row of the
supplied quadrature instead, which is a superset of the screened sum, not an
approximation of it. The two-electron fit uses the plain Coulomb metric; the
reference's constrained propagator metric is a further declaration that is NOT
implemented here.

The AUX-space kernel is a DIFFERENT ALGORITHM from the shipped orbital-product
accumulator -- ``grid_rows*naux**2`` instead of ``grid_rows*nov**2``, then a
separate ``nov**2*naux`` projection -- so it carries its OWN measured work gate
(:data:`PROPAGATOR_WORK_LIMITS`). That gate never raises, relaxes or replaces
``isapol_response_preflight.ALDA_WORK_LIMITS``, which stays in force in full for
the accumulator it was calibrated against.

Each limit is that same wall-clock budget at the MEASURED rate of its own
primitive, rounded down to two significant figures. The budget was measured, not
assumed: the shipped ``ordered_pairwise`` accumulation of a water cc-pVDZ
response on the reference 100/400 grid (128898 rows, nov=95, 1.163e+09 units)
ran in 0.275 s, so the shipped 2e+09-unit limit is a 0.443 s budget. The five
primitives here are timed separately and gated separately, and their costs are
never quoted as one number: the per-point sampling of the kernel loop is charged
on its own because at small ``naux`` it, not the ``naux**2`` accumulation,
is what the loop actually spends its time on.

No SCF, no partition, no fit, no quadrature and no localization happens here.
"""
from dataclasses import dataclass, field
from numbers import Integral

import numpy as np

from psi4 import core

#: Named ``Kernel_Smooth_Method`` values of CamCASP ``prop_parser.F90:127-160``.
SMOOTHING_METHODS = {'ZERO': 1, 'CONSTANT': 2, 'FD': 3}

#: Exponent above which ``1/(1+exp(z))`` is replaced by ``exp(-z)``; CamCASP
#: ``dft_Sx_PW92c.F90`` ``FD_z``. Part of the declared functional form.
FD_EXPONENT_CUTOFF = 40.0


@dataclass(frozen=True)
class KernelSmoothing:
    """CamCASP's declared ALDA kernel floor and cap: a model, not a tolerance.

    ``rho_epsilon`` is ``Kernel_Rho_Epsilon``: the density handed to the
    functional is raised to it (``rs_and_fix_n_simple``), and no row is dropped
    for being below it. ``f_max``/``fd_delta``/``fd_alpha``/``method`` are
    ``Kernel_F_Max``, ``Kernel_Smooth_FD_Delta``, ``Kernel_Smooth_FD_Alpha`` and
    ``Kernel_Smooth_Method`` of ``subroutine limit``:

        ZERO      |Q| >= Qmax          -> 0
        CONSTANT  Q clamped to +-Qmax
        FD        z = (|Q|/Qmax - 1)/Delta,  FD = (1/(1+exp z))**Alpha
                  Q <= Qmax ? Q*FD : Qmax*FD

    Changing any field names a different model. There is no default instance and
    no inference from the functional, the grid or the basis.
    """
    rho_epsilon: float
    f_max: float
    fd_delta: float
    fd_alpha: float
    method: str

    def __post_init__(self):
        for name in ('rho_epsilon', 'f_max', 'fd_delta', 'fd_alpha'):
            value = getattr(self, name)
            if type(value) is not float or not np.isfinite(value) or value <= 0.:
                raise ValueError(f'KernelSmoothing.{name} must be an explicit finite positive float')
        if self.method not in SMOOTHING_METHODS:
            raise ValueError('KernelSmoothing.method must be an explicit ZERO, CONSTANT or FD')

    @property
    def method_code(self):
        """The reference's integer ``Kernel_Smooth_Method``, for the record only."""
        return SMOOTHING_METHODS[self.method]

    @property
    def declaration(self):
        return (f'RHO-EPS={self.rho_epsilon!r}; F-MAX={self.f_max!r}; '
                f'FD-DELTA={self.fd_delta!r}; FD-ALPHA={self.fd_alpha!r}; '
                f'METHOD {self.method}')

    def floor(self, density):
        """Raise the density to ``rho_epsilon``; never renormalize the grid."""
        return np.maximum(np.asarray(density, dtype=float), self.rho_epsilon)

    def limit(self, values):
        """Apply ``subroutine limit`` elementwise to a real kernel array."""
        q = np.asarray(values, dtype=float)
        if self.method == 'ZERO':
            return np.where(np.abs(q) >= self.f_max, 0., q)
        if self.method == 'CONSTANT':
            return np.clip(q, -self.f_max, self.f_max)
        z = (np.abs(q)/self.f_max - 1.)/self.fd_delta
        # Both branches of the reference's FD_z split, evaluated without overflow.
        low = (1./(1. + np.exp(np.minimum(z, FD_EXPONENT_CUTOFF))))**self.fd_alpha
        high = np.exp(-self.fd_alpha*np.maximum(z, FD_EXPONENT_CUTOFF))
        fd = np.where(z <= FD_EXPONENT_CUTOFF, low, high)
        return np.where(q <= self.f_max, q*fd, self.f_max*fd)


#: The smoothing the reference run declares in its own output header. It is one
#: named model parameter set among the possible ones, not a recommended default.
CAMCASP_ALDA_SMOOTHING = KernelSmoothing(1.e-8, 1000.0, 0.01, 1.0, 'FD')


@dataclass(frozen=True)
class PropagatorDeclaration:
    """The four independent propagator declarations, all explicit.

    ``two_electron``      'exact_orbital' | 'density_fitted'
    ``kernel_projection`` 'orbital_product' | 'auxiliary_metric'
    ``kernel_density``    'exact_orbital' | 'fitted_auxiliary'
    ``smoothing``         ``None`` or a :class:`KernelSmoothing`

    Nothing is inferred from anything else: a density-fitted two-electron
    operator does not imply an AUX-projected kernel, an AUX-projected kernel does
    not imply a fitted kernel density, and neither implies smoothing. All
    sixteen combinations are distinct declared models.
    """
    two_electron: str
    kernel_projection: str
    kernel_density: str
    smoothing: object = None

    def __post_init__(self):
        if self.two_electron not in ('exact_orbital', 'density_fitted'):
            raise ValueError('two_electron must be an explicit exact_orbital or density_fitted')
        if self.kernel_projection not in ('orbital_product', 'auxiliary_metric'):
            raise ValueError('kernel_projection must be an explicit orbital_product or auxiliary_metric')
        if self.kernel_density not in ('exact_orbital', 'fitted_auxiliary'):
            raise ValueError('kernel_density must be an explicit exact_orbital or fitted_auxiliary')
        if self.smoothing is not None and not isinstance(self.smoothing, KernelSmoothing):
            raise TypeError('smoothing must be None or an explicit KernelSmoothing')
        if self.kernel_density == 'fitted_auxiliary' and self.kernel_projection != 'auxiliary_metric':
            # The fitted density is an AUX expansion; there is no orbital-product
            # accumulator that reads it, and inventing one would be a fifth model.
            raise ValueError('fitted_auxiliary kernel density requires the auxiliary_metric projection')
        if self.smoothing is not None and self.kernel_projection != 'auxiliary_metric':
            # The shipped orbital-product accumulator lives in native_response.cc and
            # applies no cap; smoothing it would mean reimplementing that nov**2 sweep
            # here, under the gate it was calibrated against. This module declines to
            # declare a model it does not build.
            raise ValueError('smoothing requires the auxiliary_metric projection; this module '
                             'has no orbital-product kernel accumulator to smooth')

    @property
    def rebuilds_kernel(self):
        return self.kernel_projection == 'auxiliary_metric'

    @property
    def is_exact_orbital(self):
        """True only for the shipped model: nothing is rebuilt, nothing changes."""
        return (self.two_electron == 'exact_orbital' and self.kernel_projection == 'orbital_product'
                and self.kernel_density == 'exact_orbital' and self.smoothing is None)

    @property
    def name(self):
        return ('2e=' + self.two_electron + '; kernel=' + self.kernel_projection
                + '; rho=' + self.kernel_density
                + ('; smoothing none' if self.smoothing is None
                   else '; smoothing ' + self.smoothing.declaration))


#: The shipped model, stated rather than implied by an absent argument.
EXACT_ORBITAL_PROPAGATOR = PropagatorDeclaration('exact_orbital', 'orbital_product', 'exact_orbital')

#: The reference protocol's declared propagator. Named once, here, so that no
#: caller has to reassemble it from four arguments and get one of them wrong.
CAMCASP_DF_PROPAGATOR = PropagatorDeclaration(
    'density_fitted', 'auxiliary_metric', 'fitted_auxiliary', CAMCASP_ALDA_SMOOTHING)


#: Measured budgets of THIS module's primitives, at the same wall-clock cost as
#: the shipped ``ordered_pairwise`` ALDA gate. They gate different products of
#: different dimensions and authorize nothing in ``ALDA_WORK_LIMITS``, which
#: remains in force unchanged for the orbital-product accumulator.
PROPAGATOR_WORK_LIMITS = {
    #: grid_rows*(naux + nbf*nocc), the per-point sampling of the kernel loop:
    #: AUX and AO basis evaluation, the occupied contraction, the functional call
    #: and the smoothing cap. Measured 7.608e+07 units/s on water/128898 rows.
    'kernel_sampling': 33_000_000,
    #: grid_rows*naux**2, one blocked DGEMM per block into an naux x naux matrix.
    #: Measured 1.559e+10 units/s on the same real loop and 2.438e+10 units/s on a
    #: synthetic naux=900 block; the SLOWER of the two sets the limit.
    'auxiliary_metric_kernel': 6_900_000_000,
    #: nov*naux**2 + nov**2*naux, the projection of that matrix onto OV space.
    #: Measured 2.536e+10 units/s.
    'kernel_projection': 11_000_000_000,
    #: naux*nbf**2*nmo, the two AO->MO half transforms of the 3-index integrals.
    #: Measured 9.069e+09 units/s.
    'df_transform': 4_000_000_000,
    #: naux*nov**2 + naux**2*nov, the 4-index DF assembly of V, X and Y.
    #: Measured 7.425e+09 units/s.
    'df_assembly': 3_200_000_000,
}


@dataclass(frozen=True)
class PropagatorWorkEstimate:
    """Pure dimension guard for this module's primitives; no allocation happens.

    Exact Python integer arithmetic, so a rejected case never wraps. This guard
    is additional to, never instead of, the native response guards: whatever is
    handed to ``native_response_from_wavefunction`` still faces those in full.
    """
    nbf: int
    nmo: int
    nocc: int
    nov: int
    naux: int
    grid_rows: int
    df_transform_work: int
    df_assembly_work: int
    kernel_sampling_work: int
    kernel_work: int
    projection_work: int
    bytes_required: int
    max_bytes: int
    failures: tuple
    limits: tuple
    provenance: str

    @property
    def passes(self):
        return not self.failures

    def require_pass(self):
        if self.failures:
            raise ValueError('NativePropagator: ' + self.failures[0])


def estimate_propagator_work(nbf, nmo, nocc, naux, grid_rows, *, declaration,
                             max_bytes=512*1024**2, block_rows=4096):
    """Assess explicit dimensions of the declared propagator rebuild.

    Only the primitives the declaration actually runs are charged: an
    ``exact_orbital`` two-electron declaration pays no transform or assembly
    work, and an ``orbital_product`` unsmoothed kernel pays no kernel work.
    """
    if not isinstance(declaration, PropagatorDeclaration):
        raise TypeError('explicit PropagatorDeclaration required')
    values = []
    for name, value in (('nbf', nbf), ('nmo', nmo), ('nocc', nocc), ('naux', naux),
                        ('grid_rows', grid_rows)):
        if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value <= 2**31-1:
            raise ValueError(f'{name} must be a nonnegative native-int dimension (not bool)')
        values.append(int(value))
    nbf, nmo, nocc, naux, grid_rows = values
    if not 0 < nocc < nmo <= nbf:
        raise ValueError('NativePropagator: invalid integer Aufbau occupations or empty OV space')
    for name, value in (('max_bytes', max_bytes), ('block_rows', block_rows)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    nov = nocc*(nmo-nocc)
    fitted = declaration.two_electron == 'density_fitted'
    aux_kernel = declaration.kernel_projection == 'auxiliary_metric'
    if fitted and not naux:
        raise ValueError('NativePropagator: density_fitted requires a molecular AUX basis')
    if aux_kernel and not naux:
        raise ValueError('NativePropagator: auxiliary_metric requires a molecular AUX basis')
    df_transform = naux*nbf**2*nmo if fitted else 0
    df_assembly = (naux*nov**2 + naux**2*nov) if fitted else 0
    sampling = grid_rows*(naux + nbf*nocc) if aux_kernel else 0
    kernel = grid_rows*naux**2 if aux_kernel else 0
    projection = (nov*naux**2 + nov**2*naux) if aux_kernel else 0
    need = 8*((naux*nbf**2 + naux*nmo**2 if fitted else 0)
              + (3*nov**2 if fitted else 0)
              + (naux**2 + int(block_rows)*(naux + nbf + nmo) if aux_kernel else 0))
    failures, limits = [], (
        ('df_transform', df_transform, PROPAGATOR_WORK_LIMITS['df_transform']),
        ('df_assembly', df_assembly, PROPAGATOR_WORK_LIMITS['df_assembly']),
        ('kernel_sampling', sampling, PROPAGATOR_WORK_LIMITS['kernel_sampling']),
        ('auxiliary_metric_kernel', kernel, PROPAGATOR_WORK_LIMITS['auxiliary_metric_kernel']),
        ('kernel_projection', projection, PROPAGATOR_WORK_LIMITS['kernel_projection']))
    for name, work, limit in limits:
        if work > limit:
            failures.append(f'{name} work resource limit')
    if need > int(max_bytes):
        failures.append('propagator workspace resource limit')
    return PropagatorWorkEstimate(nbf, nmo, nocc, nov, naux, grid_rows, df_transform,
        df_assembly, sampling, kernel, projection, need, int(max_bytes), tuple(failures), limits,
        'explicit caller dimensions and supplied row count; exact integer arithmetic; '
        'only the primitives the declaration runs are charged; separate from and '
        'additional to the native response ALDA and direct-JK gates')


@dataclass(frozen=True)
class PropagatorOperators:
    """Owned H1/H2 under one declaration, with the deltas from the shipped model.

    ``h1``/``h2`` are independent arrays. ``diagnostics`` records the measured
    departure of every rebuilt operator from the exact one it replaced; those are
    diagnostics of the substitution, never operands and never a parity claim.
    """
    h1: np.ndarray
    h2: np.ndarray
    declaration: PropagatorDeclaration
    exact_exchange: float
    local_scale: float
    kernel: str
    work: PropagatorWorkEstimate
    diagnostics: dict = field(repr=False)
    provenance: str


def _superfunctional(kernel, density_cutoff, block_rows):
    """The functional of ``native_response.cc:173-220``, setter for setter."""
    f = core.SuperFunctional.blank()
    f.set_density_tolerance(float(density_cutoff))
    x = core.LibXCFunctional('XC_LDA_X', True)
    x.set_alpha(1.)
    x.set_density_cutoff(float(np.nextafter(.5*density_cutoff, 0.)))
    f.add_x_functional(x)
    name = {'alda_slater_pw92': 'XC_LDA_C_PW', 'alda_slater_vwn': 'XC_LDA_C_VWN'}.get(kernel)
    if name is not None:
        c = core.LibXCFunctional(name, True)
        c.set_alpha(1.)
        c.set_density_cutoff(float(np.nextafter(.5*density_cutoff, 0.)))
        f.add_c_functional(c)
    f.set_max_points(int(block_rows))
    f.set_deriv(2)
    f.allocate()
    return f


def _df_two_electron(partition, full, nocc, nvir):
    """DF (ia|jb), (ij|ab) and (ib|aj) in occupied-fast OV order t = a*nocc+i.

    ``full`` must already be the ISA-MAIN adapted FULL orbital matrix; the raw
    Psi4 coefficients are in a different AO order and are not accepted here.
    """
    naux = partition.auxiliary.nfunction
    nao = partition.main.basis.nfunction
    three = np.asarray(partition.coulomb.three_center(partition.main.basis))
    if three.shape != (naux, nao*nao):
        raise ValueError('native three-centre integrals have unexpected AUX/MAIN dimensions')
    three = three.reshape(naux, nao, nao)
    mo = np.einsum('Pmn,mp,nq->Ppq', three, full, full, optimize=True)
    metric = np.asarray(partition.coulomb.metric())
    occ, vir = slice(0, nocc), slice(nocc, nocc+nvir)
    b_ov = np.ascontiguousarray(mo[:, occ, vir])
    b_oo = np.ascontiguousarray(mo[:, occ, occ])
    b_vv = np.ascontiguousarray(mo[:, vir, vir])
    # Solve against the Coulomb metric rather than inverting it; the reference's
    # further CONSTRAINED propagator metric is a separate undeclared model here.
    c_ov = np.linalg.solve(metric, b_ov.reshape(naux, -1)).reshape(naux, nocc, nvir)
    c_vv = np.linalg.solve(metric, b_vv.reshape(naux, -1)).reshape(naux, nvir, nvir)
    nov = nocc*nvir
    v = np.einsum('Pia,Pjb->iajb', b_ov, c_ov, optimize=True).transpose(1, 0, 3, 2).reshape(nov, nov)
    x = np.einsum('Pij,Pab->ijab', b_oo, c_vv, optimize=True).transpose(2, 0, 3, 1).reshape(nov, nov)
    y = np.einsum('Pib,Pja->ibaj', b_ov, c_ov, optimize=True).transpose(2, 0, 1, 3).reshape(nov, nov)
    return (np.ascontiguousarray(v), np.ascontiguousarray(x), np.ascontiguousarray(y))


def _aux_kernel_matrix(partition, full, nocc, grid, *, kernel, declaration,
                       density_cutoff, block_rows):
    """``S_kl = sum_p w_p fxc(rho(p)) chi_k(p) chi_l(p)`` over the molecular AUX.

    This is CamCASP ``num_integrals.F90``'s kernel matrix, WITHOUT its AUX
    shell-pair overlap screen: every supplied row is integrated. Returns the
    matrix and the counted rows, never a renormalized or reduced grid.
    """
    naux = partition.auxiliary.nfunction
    smoothing = declaration.smoothing
    fitted = declaration.kernel_density == 'fitted_auxiliary'
    coefficients = np.asarray(partition.drho.coefficients, dtype=float).ravel() if fitted else None
    if fitted and coefficients.shape != (naux,):
        raise ValueError('partition Drho-C coefficients do not span the molecular AUX basis')
    functional = _superfunctional(kernel, density_cutoff, block_rows)
    s = np.zeros((naux, naux))
    rows = int(grid.shape[0])
    floored = capped = dropped = 0
    deviation_max, deviation_l1 = 0., 0.
    for start in range(0, rows, int(block_rows)):
        stop = min(start+int(block_rows), rows)
        points = grid[start:stop, :3].tolist()
        weights = np.asarray(grid[start:stop, 3], dtype=float)
        chi = np.asarray(partition.auxiliary.evaluate(points))
        orbital = np.asarray(partition.main.basis.evaluate(points)) @ full[:, :nocc]
        exact = 2.*np.einsum('pi,pi->p', orbital, orbital)
        if fitted:
            density = chi @ coefficients
            deviation_max = max(deviation_max, float(np.max(np.abs(density-exact))))
            deviation_l1 += float(np.dot(weights, np.abs(density-exact)))
        else:
            density = exact
        floor = smoothing.rho_epsilon if smoothing is not None else float(density_cutoff)
        floored += int(np.count_nonzero(density < floor))
        values = functional.compute_functional(
            {'RHO_A': core.Vector.from_array(np.maximum(density, floor))}, stop-start, True)
        fxc = np.asarray(values['V_RHO_A_RHO_A'])[:stop-start].copy()
        if smoothing is not None:
            capped += int(np.count_nonzero(np.abs(fxc) > smoothing.f_max))
            fxc = smoothing.limit(fxc)
            # CamCASP floors the density; it does not discard the row.
            dead = weights == 0.
        else:
            dead = (density < float(density_cutoff)) | (weights == 0.)
        factor = weights*fxc
        factor[dead] = 0.
        dropped += int(np.count_nonzero(dead))
        s += chi.T @ (factor[:, None]*chi)
    counts = dict(kernel_rows=rows, kernel_floored_rows=floored, kernel_capped_rows=capped,
                  kernel_skipped_rows=dropped)
    if fitted:
        counts.update(fitted_density_deviation_maxabs=deviation_max,
                      fitted_density_deviation_weighted_l1=deviation_l1,
                      fitted_density_electrons=float(partition.drho.fitted_electrons),
                      fitted_density_relative_residual=float(partition.drho.relative_residual))
    return s, counts


def propagator_operators(partition, provider, full, legs, *, declaration, kernel,
                         exact_exchange, local_scale, grid, density_cutoff=1.e-10,
                         max_bytes=512*1024**2, block_rows=4096):
    """Rebuild H1/H2 under an explicit declaration, from owned native objects.

    ``H1 = Delta + 4V - a(X+Y) + 4bL`` and ``H2 = Delta - a(X-Y)`` exactly as
    ``native_response.cc:517-523`` assembles them, with ``Delta`` taken from the
    provider's own orbital energies. Whichever operators the declaration does not
    replace are taken verbatim from the provider, so an
    :data:`EXACT_ORBITAL_PROPAGATOR` declaration reproduces the provider's own
    H1/H2 rather than a re-derivation of them; that identity is checked, not
    assumed, for every declaration.

    ``legs`` is the (nov, naux) fitted transition density. The AUX kernel matrix
    is projected through it, which is what makes ``auxiliary_metric`` a different
    kernel from the orbital-product one even before ``kernel_density`` or
    ``smoothing`` are considered. A direct-OV identity leg is not accepted:
    there is no AUX expansion of an orbital product to project.
    """
    if not isinstance(declaration, PropagatorDeclaration):
        raise TypeError('explicit PropagatorDeclaration required')
    for name, value in (('exact_exchange', exact_exchange), ('local_scale', local_scale),
                        ('density_cutoff', density_cutoff)):
        if type(value) is not float or not np.isfinite(value):
            raise ValueError(f'{name} must be an explicit finite float')
    if not 0. <= exact_exchange <= 1. or not 0. <= local_scale <= 1. or density_cutoff <= 0.:
        raise ValueError('exchange/local scales must be in [0,1] and density cutoff positive')
    if kernel not in ('no_local', 'alda_slater', 'alda_slater_pw92', 'alda_slater_vwn'):
        raise ValueError('unsupported explicit native kernel')
    if kernel == 'no_local' and declaration.rebuilds_kernel:
        raise ValueError('no_local has no local kernel to project, smooth or refit')
    nocc, nvir = provider.nocc, provider.nvir
    nov = nocc*nvir
    naux = partition.auxiliary.nfunction
    full = np.asarray(full, dtype=float)
    if full.shape != (partition.main.basis.nfunction, nocc+nvir) or not np.isfinite(full).all():
        raise ValueError('full must be the finite ISA-MAIN adapted (nao, nmo) orbital matrix')
    legs = np.asarray(legs, dtype=float)
    if declaration.kernel_projection == 'auxiliary_metric':
        if legs.shape != (nov, naux) or not np.isfinite(legs).all():
            raise ValueError('auxiliary_metric projection requires the finite (nov, naux) fitted D')
    rows = 0
    if declaration.rebuilds_kernel:
        grid = np.asarray(grid, dtype=float)
        if (grid.ndim != 2 or grid.shape[1] != 4 or not grid.shape[0]
                or not np.isfinite(grid).all() or np.any(grid[:, 3] < 0)):
            raise ValueError('grid must be finite real nonempty [x,y,z,nonnegative weight] rows')
        rows = int(grid.shape[0])
    work = estimate_propagator_work(partition.main.basis.nfunction, nocc+nvir, nocc, naux, rows,
                                    declaration=declaration, max_bytes=max_bytes,
                                    block_rows=block_rows)
    work.require_pass()
    exact = dict(coulomb=np.asarray(provider.coulomb()).copy(),
                 exchange_direct=np.asarray(provider.exchange_direct()).copy(),
                 exchange_transpose=np.asarray(provider.exchange_transpose()).copy(),
                 local=np.asarray(provider.local_primitive()).copy())
    energies = np.asarray(provider.energies(), dtype=float)
    index = np.arange(nov)
    delta = np.zeros((nov, nov))
    delta[index, index] = energies[nocc + index//nocc] - energies[index % nocc]
    a, b = float(exact_exchange), float(local_scale)

    def assemble(v, x, y, local):
        return (delta + 4.*v - a*(x+y) + 4.*b*local, delta - a*(x-y))

    # The assembly formula is verified against the provider's OWN H1/H2 before
    # anything is substituted into it, so a substitution is never credited with
    # a discrepancy that belongs to the rebuild.
    control = assemble(exact['coulomb'], exact['exchange_direct'],
                       exact['exchange_transpose'], exact['local'])
    reference = (np.asarray(provider.h1()), np.asarray(provider.h2()))
    scale = max(1., float(np.max(np.abs(reference[0]))), float(np.max(np.abs(reference[1]))))
    residual = max(float(np.max(np.abs(control[0]-reference[0]))),
                   float(np.max(np.abs(control[1]-reference[1]))))
    if residual > 1.e-11*scale:
        raise ValueError(f'native H1/H2 assembly identity failed: {residual}')
    diagnostics = dict(assembly_identity_residual_maxabs=residual,
                       declaration=declaration.name,
                       aux_shell_pair_screen='not replicated; every supplied row integrated')
    v, x, y, local = (exact['coulomb'], exact['exchange_direct'],
                      exact['exchange_transpose'], exact['local'])
    if declaration.two_electron == 'density_fitted':
        v, x, y = _df_two_electron(partition, full, nocc, nvir)
        for name, fitted, reference_matrix in (('coulomb', v, exact['coulomb']),
                                               ('exchange_direct', x, exact['exchange_direct']),
                                               ('exchange_transpose', y, exact['exchange_transpose'])):
            if not np.allclose(fitted, fitted.T, rtol=0, atol=1.e-9*max(1., np.abs(fitted).max())):
                raise ValueError(f'density-fitted {name} operator is not symmetric')
            diagnostics[f'df_{name}_relative_maxabs_deviation'] = float(
                np.max(np.abs(fitted-reference_matrix))/max(np.abs(reference_matrix).max(), 1.e-300))
        diagnostics['two_electron_metric'] = 'plain Coulomb metric (no constrained propagator metric)'
    if declaration.rebuilds_kernel:
        s, counts = _aux_kernel_matrix(partition, full, nocc, grid, kernel=kernel,
                                       declaration=declaration, density_cutoff=density_cutoff,
                                       block_rows=block_rows)
        local = legs @ s @ legs.T
        diagnostics.update(counts)
        diagnostics['aux_kernel_relative_maxabs_deviation'] = float(
            np.max(np.abs(local-exact['local']))/max(np.abs(exact['local']).max(), 1.e-300))
    h1, h2 = assemble(v, x, y, local)
    if declaration.is_exact_orbital and max(float(np.max(np.abs(h1-reference[0]))),
                                            float(np.max(np.abs(h2-reference[1])))) > 0.:
        raise ValueError('exact_orbital declaration did not reproduce the provider operators')
    return PropagatorOperators(h1, h2, declaration, a, b, kernel, work, diagnostics,
        'native provider orbital energies and unreplaced operators; ' + declaration.name
        + '; Drho-C monomer density from the partition, not refitted'
        + '; AUX shell-pair screen not replicated (all rows integrated)'
        + '; H1=Delta+4V-a(X+Y)+4bL, H2=Delta-a(X-Y) verified against the provider')
