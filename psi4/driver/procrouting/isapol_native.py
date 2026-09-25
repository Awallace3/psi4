# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Explicit wavefunction -> Drho-C ISA-A -> fitted native response -> strict LW.

Expert API, not an SCF/oeprop method or CamCASP PBE0/GRAC parity claim. No I/O,
reference data, hidden quadrature, charge repair, PFIT or anisotropic conversion.
Native records are owned snapshots (some core getters remain caller-mutable).
Failures retain completed stages; ``require_local`` prevents mistaking raw pair
blocks for atomic properties. Callers must not concurrently mutate the wavefunction.
"""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from psi4 import core
from .isapol_native_partition import PartitionRecipe, NativePartitionResult, native_partition
from .isapol_native_response import NativeWavefunctionResponse, native_response_from_wavefunction
from .isapol_native_propagator import PropagatorDeclaration, propagator_operators
from .sapt.fdds_response import FDDSFullOVResponse
from .isapol_native_correction import validate_correction, functional_definition
from . import isapol_lw as lw
from . import isapol_df_multipoles as dfm
from . import isapol_logging as lg


@dataclass(frozen=True)
class Quadrature:
    """Authoritative complete node list; CP weights ALREADY include 1/(2*pi)."""
    frequencies: tuple
    cp_weights: tuple
    provenance: lw.Provenance

    def __post_init__(self):
        f = _frequencies(self.frequencies)
        w = np.asarray(self.cp_weights)
        if (w.dtype.kind not in 'fiu' or w.shape != (len(f),) or not np.isfinite(w).all()
                or np.any(w < 0) or not np.any(w > 0)
                or any(x == 0 and y != 0 for x, y in zip(f, w))
                or any(x > 0 and y <= 0 for x, y in zip(f, w))):
            raise ValueError('complete quadrature requires positive weight at every dynamic node and zero static weight')
        if not isinstance(self.provenance, lw.Provenance):
            raise TypeError('explicit quadrature provenance required')
        object.__setattr__(self, 'frequencies', f)
        object.__setattr__(self, 'cp_weights', tuple(map(float, w)))

    @classmethod
    def from_casimir(cls, grid):
        if not isinstance(grid, core.CasimirGrid):
            raise TypeError('actual native CasimirGrid required')
        f = tuple(grid.omega(i) for i in range(grid.n_freq()+1))
        w = tuple(grid.cp_weight(i) for i in range(grid.n_freq()+1))
        # Actual core nodes, not rounded headers or a reimplemented rule.
        order = np.argsort(f)
        f, w = tuple(f[i] for i in order), tuple(w[i] for i in order)
        digest = hashlib.sha256(np.asarray([f, w], dtype='<f8').tobytes()).hexdigest()
        return cls(f, w, lw.Provenance('native CasimirGrid nodes/CP weights', digest,
                   'Psi4 core.CasimirGrid', f'n={grid.n_freq()}, beta={grid.omega0()}; CP prefactor included once'))


def _frequencies(values):
    a = np.asarray(values)
    if (a.dtype.kind not in 'fiu' or a.ndim != 1 or not 0 < len(a) <= 4096
            or not np.isfinite(a).all() or np.any(a < 0) or np.any(np.diff(a) <= 0)):
        raise ValueError('frequencies require finite strictly increasing nonnegative nodes')
    return tuple(map(float, a))


def _rank4_truncation(rank, localization_rank_limit):
    """The rank4 rows either reach LW and are localized, or they do not.

    Rank3 input carries no rank4 rows and declares nothing. Rank4 input must say
    which of the two models it is: discarded (ranks1..3 local, C12 structurally
    partial) or retained (ranks1..4 local, C12 complete). The two are different
    models and must never be quoted as agreeing.
    """
    if rank != 4:
        return None
    return lw.RETAIN_RANK4 if localization_rank_limit == 4 else lw.TRUNCATE_RANK4


def _context(wfn):
    """Exact context fingerprint, including effective basis and actual full state."""
    if not isinstance(wfn, core.Wavefunction) or wfn.nirrep() != 1:
        raise ValueError('actual C1 wavefunction required')
    b, m = wfn.basisset(), wfn.molecule()
    shells = []
    for i in range(b.nshell()):
        s = b.shell(i)
        shells.append((b.shell_to_center(i), b.shell_to_basis_function(i), s.am, s.is_pure(),
                       tuple(s.exp(k) for k in range(s.nprimitive)),
                       tuple(s.coef(k) for k in range(s.nprimitive))))
    meta = (b.nbf(), wfn.nalpha(), wfn.nbeta(), tuple(wfn.doccpi()), tuple(wfn.soccpi()),
            tuple((m.Z(i), m.x(i), m.y(i), m.z(i)) for i in range(m.natom())), shells)
    h = hashlib.sha256(json.dumps(meta).encode())
    h.update(repr(functional_definition(wfn.functional(), include_cutoffs=True)).encode())
    for obj in (wfn.Ca(), wfn.Cb(), wfn.epsilon_a(), wfn.epsilon_b(), wfn.Da(), wfn.Db()):
        a = np.asarray(obj, dtype='<f8')
        if not np.isfinite(a).all():
            raise ValueError('nonfinite wavefunction context')
        h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def _policy(kernel, exact_exchange, local_scale, grid, density_cutoff, correction=None,
            algorithm='ordered_pairwise', propagator=None):
    # The named response algorithm is part of the policy: the two arrangements
    # are separately gated, so a context built under one is not reusable under
    # the other even though their operators agree. The propagator declaration is
    # likewise part of it: a context built under the exact-orbital propagator
    # holds different H1/H2 than one built under a density-fitted, AUX-projected
    # or smoothed declaration, and the two are different models, not variants.
    h = hashlib.sha256(repr((kernel, exact_exchange, local_scale, density_cutoff, correction,
                             algorithm,
                             None if propagator is None else propagator.name)).encode())
    if grid is not None:
        a = np.asarray(grid)
        if a.dtype.kind not in 'fiu' or not np.isfinite(a).all():
            raise ValueError('invalid explicit response grid')
        h.update(str(a.shape).encode()); h.update(np.asarray(a, dtype='<f8').tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class NativeContext:
    """Factory-owned native provider; reusable for declared quadrature refinements.

    Trusted factory record, not a deserialization boundary. No wavefunction is held.
    Reuse checks exact state/basis/geometry and policy before any fit.
    """
    response: NativeWavefunctionResponse
    wavefunction_sha256: str
    policy_sha256: str

    @property
    def correction_provenance(self):
        return self.response.correction_provenance


@dataclass(frozen=True)
class StageFailure:
    stage: str
    frequency: object
    exception_type: str
    message: str


@dataclass(frozen=True)
class NativeProperties:
    partition: NativePartitionResult
    context: object
    ov_fit: object
    full_adapted_orbitals: object
    frequencies: tuple
    quadrature: object
    coefficient_responses: tuple
    distributed: object
    pair_tensors: object
    local: object
    dispersion: object
    failures: tuple
    diagnostics: dict
    model: str
    correction_provenance: object
    comparison_status: str = 'not_measured_no_reference'

    def require_local(self):
        if self.local is None:
            raise RuntimeError('No accepted localized atomic tensors: ' + '; '.join(f.message for f in self.failures))
        return self.local

    @property
    def atomic_tensors(self):
        return self.require_local().raw_local

    @property
    def atomic_scalars(self):
        return self.require_local().atomic_scalars


def native_properties(wfn, recipe, *, bonds, frames, caller_converged, kernel,
                      exact_exchange, local_scale, response_grid, frequencies=(0.,),
                      quadrature=None, partner=None, pair_self=False, max_order=12,
                      density_cutoff=1.e-10, max_bytes=512*1024**2, max_nov=512,
                      response_context=None, response_basis='fitted_auxiliary',
                      scf_correction='NONE', expected_grac_shift=None, ac_declaration=None,
                      response_algorithm='ordered_pairwise', ov_charge_penalty=1.,
                      ov_metric_damping=0., localization_rank_limit=3, propagator=None,
                      distribution='isa_a', log=None, local_frames=None):
    """Return all owned stages, with strict production LW (1e-6) or failures.

    Explicit ``response_basis='direct_ov'`` integrates actual occupied/virtual
    orbital products against the SAME converged Drho-C ISA-A shapes. It avoids
    transition-density fitting altogether, not by neutralizing fitted D. The
    density partition is unchanged. Its analytic OV charges are zero by the
    independently checked MO orthonormality; measured grid charges remain raw.
    The default fitted_auxiliary route and its lambda1 failures are unchanged.

    ``ov_charge_penalty`` is the finite rank-1 penalty ``A += lambda*q q^T`` of
    the transition fit, and it stays at the archived **lambda1** unless the
    caller explicitly declares otherwise. It is not a tolerance: an OV
    transition density has exactly zero charge by MO orthonormality, so the
    penalty only enforces something the exact answer already satisfies, and the
    fitted charge it leaves behind falls exactly as 1/lambda. Raising it
    therefore converges the constraint rather than loosening a gate -- but it
    also changes the fitted D, so a chain run at any other lambda is a
    differently declared model and must never be compared against a recorded
    lambda1 reference number. The traced constrained-NN route declares
    ``lambda=1000``, and strict production LW accepts the water fitted chain at
    every declared lambda >= 1e3 (SPEC section 6 has the measured table).
    ``direct_ov`` forms no fit and accepts only the default. Recorded in the fit and result provenance, deliberately **not** in
    the response policy hash: H1/H2 come from the orbitals and are independent
    of it, so one native context serves every lambda.

    ``ov_metric_damping`` is the SECOND declared parameter of the same constrained
    fit, the reference protocol's ``Eta``: every Coulomb-metric element whose two
    AUX functions sit on different centres is scaled by ``1-eta`` before the charge
    penalty is added, which is exactly ``ConstraintType = 1`` of the reference's
    constrained density fitting. Unlike lambda it does **not** converge to
    something the exact answer already satisfies -- it changes the fitted
    transition density at every eta, deliberately, by making inter-site fitted
    density more expensive. It is therefore a model declaration and never a
    tolerance, a conditioning repair or a preconditioner: a chain run at any eta
    other than a recorded one is a differently declared model and must never be
    compared against a number recorded at a different eta, exactly as for lambda.
    The default 0. is the undamped fit every committed number was measured at; the
    traced reference polarizability step declares ``Eta = 0.0005, Lambda = 1000``.
    ``direct_ov`` forms no fit, so it accepts only the default here too.

    ``distribution`` names WHICH DISTRIBUTED-MULTIPOLE MODEL the site response is
    formed in, and it is a model declaration, never a variant of one model. The
    default ``isa_a`` is the converged Drho-C stockholder shape partition, whose
    Q assigns each grid point to sites by the ratio ``shape_a/sum_b shape_b``.
    ``df_centre_analytic`` and ``df_centre_grid`` are instead CamCASP's
    ``DistPolAlgorithm = 'DF'`` rule, which charges each auxiliary function
    WHOLLY to the centre it sits on and forms no stockholder weight at all (see
    ``isapol_df_multipoles``). The two rules assign the same molecular density to
    sites by different rules, so their site multipoles, site polarizabilities and
    dispersion coefficients may never be quoted as agreeing or disagreeing: they
    are answers to different questions. Because the DF rule has no denominator it
    needs no ISA-A fixed point, so a DF-centre request does not demand
    ``partition.require_q()`` here -- the partition is still run and narrated, but
    its convergence is not made a precondition of a model that never uses it. The
    DF rule does need AUX columns for a function to have a centre at all, so it
    refuses ``direct_ov``; and because it is basis-sensitive in a way the
    stockholder rule is not, a negative static site isotropic response is refused
    outright rather than propagated into a polarizability or a C6.

    ``recipe.grid`` is the explicit ISA/Q grid policy; ``response_grid`` is None
    for no_local or explicit [x,y,z,w] for ALDA. Full H1/H2 already include all
    native interactions. Finite lambda1 fitted D uses T @ FULL native Psi4 C,
    occupied-fast OV, no added spin/metric/local-kernel factors. All requested
    CDF/distributed nodes are formed BEFORE LW attempts. Partial LW success never
    becomes a local model or a dispersion quadrature. Static needs no partner,
    weights or PFIT. Explicit A/B pairing accepts a successful NativeProperties
    or LW LocalProperties; pair_self=True explicitly chooses this same model B.
    ``response_context`` permits reuse only under exact same state and policy.
    ``response_algorithm`` names the native arrangement (see
    ``isapol_native_response``); it is part of the policy hash and selects which
    calibrated ALDA work limit applies. It changes no other limit and no
    partition, fit, LW or PFIT stage.

    ``localization_rank_limit`` declares the uniform rank the LW stage localizes
    at, in 1..4 (default 3). It is the reference protocol's single ``Limit``, and it
    is a SEPARATE declaration from ``recipe.sites[*].rank``: the site rank is the
    rank of the distributed response fed in and still has to be a uniform explicit 3
    or 4, which this does not relax. A limit of 4 additionally requires site rank 4,
    since rank-4 local tensors cannot be localized out of a rank-3 distributed
    response; that pairing is what makes the C12 (1,4)/(4,1) rank pairs available,
    and it is a different model from the rank-3 one rather than a refinement of it.
    Below the site rank the restriction is exact rather than approximate and gates
    nothing differently; it also cannot change a rank <= limit number, since it
    yields the higher-limit result restricted to the declared space (see
    :func:`isapol_lw.supplied_nonlocal_properties`).

    ``propagator`` is an explicit
    :class:`isapol_native_propagator.PropagatorDeclaration` naming how H1/H2 are
    built, or None. None is not a default declaration: it means the propagator
    module is not entered at all and the provider's own exact-orbital H1/H2 are
    used unchanged, so every number recorded without it stays bitwise what it was.
    Supplying one declares a DIFFERENT MODEL along up to four independent axes
    (density-fitted two-electron operators, an AUX-metric kernel projection, a
    fitted-density kernel argument, and CamCASP's declared kernel smoothing); its
    numbers may never be quoted as agreeing with the exact-orbital ones or with
    each other across declarations. An AUX-metric projection needs the fitted
    transition density and therefore ``response_basis='fitted_auxiliary'``, and a
    rebuilt kernel needs the explicit ``response_grid`` it is integrated on rather
    than an inferred one.

    ``local_frames`` is an alternative to ``frames``: one explicit
    :class:`isapol_geometry.LocalFrame` per site (axis-pair or Tinker-style
    atom-defined recipe), resolved against the partition's site origins at the
    LW stage. Declaring both is refused before any work is done.
    """
    if frames is not None and local_frames is not None:
        raise ValueError('declare frames or local_frames, not both')
    if response_basis not in ('fitted_auxiliary', 'direct_ov'):
        raise ValueError('unsupported response_basis')
    if (type(ov_charge_penalty) is not float or not np.isfinite(ov_charge_penalty)
            or ov_charge_penalty <= 0.):
        raise ValueError('ov_charge_penalty must be an explicit finite positive float')
    if response_basis == 'direct_ov' and ov_charge_penalty != 1.:
        raise ValueError('direct_ov forms no transition fit; no charge penalty applies')
    if (type(ov_metric_damping) is not float or not np.isfinite(ov_metric_damping)
            or not 0. <= ov_metric_damping < 1.):
        raise ValueError('ov_metric_damping must be an explicit finite float in [0,1)')
    if response_basis == 'direct_ov' and ov_metric_damping != 0.:
        raise ValueError('direct_ov forms no transition fit; no metric damping applies')
    if propagator is not None:
        if not isinstance(propagator, PropagatorDeclaration):
            raise TypeError('propagator must be None or an explicit PropagatorDeclaration')
        if propagator.kernel_projection == 'auxiliary_metric' and response_basis != 'fitted_auxiliary':
            raise ValueError('auxiliary_metric kernel projection requires the fitted_auxiliary '
                             'response basis; direct_ov forms no AUX transition density')
        if propagator.rebuilds_kernel and response_grid is None:
            raise ValueError('a rebuilt ALDA kernel requires the explicit response_grid it is '
                             'integrated on; none is inferred')
    if caller_converged is not True:
        raise ValueError('caller_converged must explicitly be True')
    if not isinstance(recipe, PartitionRecipe):
        raise TypeError('explicit PartitionRecipe required')
    if len(set(s.rank for s in recipe.sites)) != 1 or recipe.sites[0].rank not in (3, 4):
        raise ValueError('LW pipeline requires uniform explicit rank3 or rank4')
    if type(localization_rank_limit) is not int or localization_rank_limit not in (1, 2, 3, 4):
        raise ValueError('localization_rank_limit must be explicit integer1,2,3 or4')
    if localization_rank_limit > recipe.sites[0].rank:
        raise ValueError('localization_rank_limit exceeds the site rank of the distributed response')
    if distribution not in dfm.DISTRIBUTIONS:
        raise ValueError(f'distribution must be one of {dfm.DISTRIBUTIONS}, not {distribution!r}')
    if distribution in dfm.DF_CENTRE_DISTRIBUTIONS and response_basis == 'direct_ov':
        raise ValueError('The DF-centre rule charges each auxiliary function to its own centre, '
                         'so it needs AUX columns; direct_ov has occupied-virtual product '
                         'columns and no centre for a function to sit on. Declare '
                         "response_basis='fitted_auxiliary'.")
    freq = _frequencies(frequencies)
    if quadrature is not None and (not isinstance(quadrature, Quadrature) or freq != quadrature.frequencies):
        raise ValueError('requested nodes must exactly match complete authoritative quadrature')
    if type(pair_self) is not bool or (pair_self and partner is not None):
        raise ValueError('choose explicit partner OR pair_self')
    if (pair_self or partner is not None) and quadrature is None:
        raise ValueError('dispersion requires complete authoritative quadrature')
    if partner is not None:
        partner = partner.require_local() if isinstance(partner, NativeProperties) else partner
        if not isinstance(partner, lw.LocalProperties) or partner.frequencies != freq:
            raise ValueError('partner must be accepted LW model with identical nodes')
        if not partner.metadata.production_postcondition_passed:
            raise ValueError('partner must pass production LW')
    correction = validate_correction(wfn, scf_correction=scf_correction,
                                     expected_grac_shift=expected_grac_shift,
                                     ac_declaration=ac_declaration)
    if correction.policy != 'NONE' and kernel == 'no_local':
        raise ValueError(f'{correction.policy} admission requires an explicit ALDA response '
                         'policy; no asymptotic-correction kernel derivative')
    context_hash = _context(wfn)
    policy_hash = _policy(kernel, exact_exchange, local_scale, response_grid, density_cutoff,
                          correction, response_algorithm, propagator)
    if response_context is not None:
        if (not isinstance(response_context, NativeContext)
                or response_context.wavefunction_sha256 != context_hash
                or response_context.policy_sha256 != policy_hash
                or response_context.correction_provenance != correction):
            raise ValueError('native response context mismatch (state/basis/geometry/policy)')
    log = lg.silent() if log is None else log
    log.stage('density partition (Drho-C ISA-A)', lg.partition_parameters(recipe))
    partition = native_partition(wfn, recipe, caller_converged=caller_converged)
    context, fit, adapted, distributed, tensors, local, dispersion = response_context, None, None, None, None, None, None
    responses, failures, diagnostics = [], [], {}
    stage = 'partition'
    def result():
        return NativeProperties(partition, context, fit, adapted, freq, quadrature, tuple(responses),
            distributed, tensors, local, dispersion, tuple(failures), diagnostics,
            f'native {kernel}; exact_exchange={exact_exchange}; local_scale={local_scale}; '
            f'{response_algorithm}; Drho-C ISA-A[{recipe.auxiliary.name}]; '
            f'distribution={distribution}; {response_basis}'
            + ('' if response_basis == 'direct_ov'
               else f' lambda={ov_charge_penalty!r}; eta={ov_metric_damping!r}')
            + ('' if propagator is None else '; propagator ' + propagator.name)
            + '; no PFIT'
            + ('; ' + correction.response_description if correction.policy != 'NONE' else ''),
            correction)
    try:
        if distribution == 'isa_a':
            q = partition.require_q()
        else:
            # The DF rule forms no stockholder denominator, so the ISA-A fixed
            # point is not a precondition of it. The partition still ran and is
            # still reported; it is simply not this model's operand.
            stage = 'DF-centre distributed multipoles'
            declared = dfm.df_centre_multipoles(distribution, recipe.auxiliary, recipe.sites,
                recipe.sites[0].rank,
                **(dict(points=partition.grid_points, weights=partition.grid_weights)
                   if distribution == 'df_centre_grid' else {}))
            q = declared.partition()
            log.stage(stage, (('rule', "CamCASP DistPolAlgorithm='DF'"),
                              ('form', declared.form), ('AUX', recipe.auxiliary.name),
                              ('site rank', recipe.sites[0].rank),
                              ('ISA-A fixed point used', False))
                      + tuple(sorted(declared.diagnostics.items(), key=lambda kv: kv[0])))
            diagnostics.update(('distribution_' + k, v) for k, v in declared.diagnostics.items())
            diagnostics.update(distribution_provenance=declared.provenance,
                               distribution_isa_a_converged=partition.converged)
        diagnostics.update(distribution=distribution)
        lg.report_partition(log, None, partition)
        stage = 'context'
        log.stage('native response context', lg.response_parameters(
            kernel=kernel, exact_exchange=exact_exchange, local_scale=local_scale,
            density_cutoff=density_cutoff, max_bytes=max_bytes, max_nov=max_nov,
            response_algorithm=response_algorithm, response_basis=response_basis,
            response_grid=response_grid, correction=correction,
            ov_charge_penalty=ov_charge_penalty, ov_metric_damping=ov_metric_damping))
        if _context(wfn) != context_hash:
            raise ValueError('wavefunction context changed during partition')
        if context is None:
            response = native_response_from_wavefunction(wfn, caller_converged=True, kernel=kernel,
                exact_exchange=exact_exchange, local_scale=local_scale, grid=response_grid,
                density_cutoff=density_cutoff, max_bytes=max_bytes, max_nov=max_nov,
                scf_correction=scf_correction, expected_grac_shift=expected_grac_shift,
                ac_declaration=ac_declaration, algorithm=response_algorithm)
            context = NativeContext(response, context_hash, policy_hash)
        provider = context.response.provider
        c = np.asarray(provider.orbitals())
        if (_context(wfn) != context_hash or not np.array_equal(c, np.asarray(wfn.Ca()))
                or not np.array_equal(np.asarray(provider.energies()), np.asarray(wfn.epsilon_a()))
                or provider.nocc != wfn.nalpha()):
            raise ValueError('provider/wavefunction full orbital/energy context mismatch')
        full = partition.main.transform @ c
        if not np.array_equal(full[:, :provider.nocc], partition.main.occupied):
            # Matrix-multiplication blocking may differ, but no representation repair.
            error = float(np.max(np.abs(full[:, :provider.nocc]-partition.main.occupied)))
            if error > 2.e-13:
                raise ValueError(f'partition occupied adaptation mismatch: {error}')
        adapted = lw.ArraySnapshot.of(full)
        lg.report_context(log, None, context, provider)
        stage = 'OV moments' if response_basis == 'direct_ov' else 'OV fit'
        log.stage(stage, (('response_basis', response_basis),
                          ('site multipole rank', recipe.sites[0].rank),
                          ('density_cutoff', recipe.controller.density_cutoff))
                  + ((('ov_charge_penalty (lambda)', ov_charge_penalty),
                      ('ov_metric_damping (eta)', ov_metric_damping))
                     if response_basis != 'direct_ov' else ()))
        if response_basis == 'direct_ov':
            total = np.sum(partition.shape_samples, axis=0)
            sites = []
            for i, site in enumerate(recipe.sites):
                samples = core.IsaMultipoleSamples()
                samples.points = partition.grid_points.tolist()
                samples.weights = partition.grid_weights.tolist()
                samples.shape = partition.shape_samples[i].tolist()
                samples.shape_sum = total.tolist()
                samples.auxiliary_sites = list(range(len(recipe.sites)))
                item = core.IsaMultipoleSite()
                item.label, item.origin, item.rank, item.samples = site.label, site.origin, site.rank, samples
                sites.append(item)
            q = core.IsaPartitionedMultipoles(partition.main.basis, sites,
                partition.provenance + '; direct occupied-fast OV integration (no transition fit)',
                recipe.controller.density_cutoff, core.Matrix.from_array(full), provider.nocc)
            d = np.eye(provider.nocc * provider.nvir)
        else:
            fit = partition.coulomb.fit_ov(partition.main.basis,
                core.Matrix.from_array(full[:, :provider.nocc].copy()),
                core.Matrix.from_array(full[:, provider.nocc:].copy()),
                f'native full-C verified AO-to-DALTON; occupied-fast; lambda={ov_charge_penalty!r}; '
                f'eta={ov_metric_damping!r}; ' + context_hash, ov_charge_penalty, ov_metric_damping)
            d = np.asarray(fit.coefficients)
            if d.shape != (provider.nocc*provider.nvir, partition.auxiliary.nfunction):
                raise ValueError('native OV fit dimensions/order mismatch')
        if _context(wfn) != context_hash:
            raise ValueError('wavefunction changed during native fit')
        lg.report_ov_fit(log, fit, response_basis)
        if propagator is None:
            h1_baseline, h2_operator = np.asarray(provider.h1()), np.asarray(provider.h2())
        else:
            stage = 'propagator'
            operators = propagator_operators(partition, provider, full, d,
                declaration=propagator, kernel=kernel, exact_exchange=exact_exchange,
                local_scale=local_scale, grid=response_grid, density_cutoff=density_cutoff,
                max_bytes=max_bytes)
            h1_baseline, h2_operator = operators.h1, operators.h2
            log.stage(stage, lg.propagator_parameters(propagator, operators.work))
            lg.report_propagator(log, operators)
            diagnostics.update(('propagator_' + k, v) for k, v in operators.diagnostics.items())
        solver = FDDSFullOVResponse(h1_baseline=h1_baseline, h2=h2_operator,
            transition_legs=d, coupling=np.zeros((d.shape[1], d.shape[1])),
            representation=('supplied_transition_leg_coordinates' if response_basis == 'direct_ov'
                            else 'fitted_density_coefficients'))
        stage = 'frequency response'
        log.stage(stage, (('solver', 'FDDSFullOVResponse'),
                          ('propagator', 'provider exact-orbital H1/H2' if propagator is None
                           else propagator.name),
                          ('representation', solver.representation),
                          ('nodes', len(freq)), ('frequencies [Eh]', freq)))
        for xi in freq:
            responses.append(solver.at_frequency(xi))
        stage = 'distributed response'
        log.stage(stage, (('sites', len(recipe.sites)),
                          ('site rank', recipe.sites[0].rank),
                          ('components per site', (recipe.sites[0].rank+1)**2),
                          ('transition coordinates',
                           'direct_ov' if response_basis == 'direct_ov'
                           else 'fitted_density_coefficients')))
        distributed = core.IsaDistributedResponse(q, freq,
            [core.Matrix.from_array(r.raw_coupled) for r in responses],
            'direct_ov' if response_basis == 'direct_ov' else 'fitted_density_coefficients',
            'native full H1/H2; '+response_basis+'; '+context_hash)
        n, rank = len(recipe.sites), recipe.sites[0].rank
        m = (rank+1)**2
        raw = np.array([np.asarray(distributed.at_index(k)).reshape(n,m,n,m).transpose(0,2,1,3)
                        for k in range(len(freq))])
        tensors = lw.ArraySnapshot.of(raw)
        if distribution in dfm.DF_CENTRE_DISTRIBUTIONS:
            # alpha^(aa) = Qa (-C) Qa^T with -C(i xi) positive semidefinite, so a
            # negative site isotropic scalar cannot happen for a sound (rule, AUX)
            # pair. When it does, the declared auxiliary set is wrong for this
            # rule, and the number must not reach a polarizability or a C_n.
            for k, xi in enumerate(freq):
                scalars = dfm.site_isotropic_gate(raw[k], rank, labels=tuple(q.labels))
                diagnostics.update((f'distribution_site_isotropic[xi={xi!r}][{a} rank{l}]', v)
                                   for (a, l), v in scalars.items())
        qsum = np.asarray(q.values)[list(q.offsets[:-1])].sum(axis=0)
        analytic_q = (np.zeros(d.shape[1]) if response_basis == 'direct_ov' else np.asarray(fit.charges))
        # Charge origin diagnostics ONLY, never replacement operands for output.
        diagnostics.update(q_sum_minus_analytic_maxabs=float(np.max(np.abs(qsum-analytic_q))),
            fitted_transition_charge_maxabs=float(np.max(np.abs(d @ analytic_q))),
            quadrature_transition_charge_error_maxabs=float(np.max(np.abs(d @ (qsum-analytic_q)))),
            grid_transition_charge_maxabs=float(np.max(np.abs(d @ qsum))),
            charge_reference=('orthogonal direct OV (zero analytically)' if fit is None else 'native AUX integrals'),
            ov_backward_residual=None if fit is None else fit.relative_backward_residual,
            response_basis=response_basis,
            raw_charge_sum_maxabs=[float(np.max(np.abs(a[:,: ,0,:m].sum(axis=0)))) for a in raw],
            analytic_charge_response_maxabs=[float(np.max(np.abs(analytic_q @ r.raw_coupled @ np.asarray(q.values).T))) for r in responses],
            quadrature_charge_response_maxabs=[float(np.max(np.abs((qsum-analytic_q) @ r.raw_coupled @ np.asarray(q.values).T))) for r in responses])
        lg.report_response_diagnostics(log, diagnostics)
        provenance = lw.Provenance('fresh native distributed tensors', tensors.canonical_array_sha256,
            'Psi4 native fitted response and IsaDistributedResponse', partition.provenance+'; '+context_hash)
        args = dict(labels=q.labels, origins=q.origins, bonds=bonds, frames=frames,
                    local_frames=local_frames,
                    input_rank=rank, truncation=_rank4_truncation(rank, localization_rank_limit),
                    provenance=provenance, residual_policy='production',
                    localization_rank_limit=localization_rank_limit)
        stage = 'LW'
        log.stage(stage, lg.localization_parameters(
            input_rank=rank, truncation=args['truncation'],
            localization_rank_limit=localization_rank_limit,
            residual_policy='production', bonds=bonds,
            frames=frames if local_frames is None else local_frames, frequencies=freq))
        # Attempt EVERY node independently; never relax or hide a failed frequency.
        for k, xi in enumerate(freq):
            try:
                lw.supplied_nonlocal_properties(frequencies=(xi,), tensors=raw[k:k+1], **args)
            except Exception as exc:
                failures.append(StageFailure(stage, xi, type(exc).__name__, str(exc)))
        if not failures:
            local = lw.supplied_nonlocal_properties(frequencies=freq, tensors=raw, **args)
            lg.report_localization(log, None, local)
            lg.report_atomic_polarizabilities(log, None, local)
            if pair_self or partner is not None:
                # The producer owns this stage's banner, its node table and its
                # coefficient tables: it is the only place that knows which
                # per-site ranks it resolved off each model. `stage` is still
                # named here for StageFailure attribution.
                stage = 'dispersion'
                # No site_ranks_* here: each side's rank set is read off that
                # side's own model. `partner` is a separately declared model and
                # may carry a different localization limit than this call's.
                dispersion = lw.isotropic_dispersion(local, local if pair_self else partner,
                    cp_weights=quadrature.cp_weights, quadrature_provenance=quadrature.provenance,
                    max_order=max_order, log=log)
    except Exception as exc:
        failures.append(StageFailure(stage, None, type(exc).__name__, str(exc)))
    log.stage_end()
    log.failures(failures)
    return result()
