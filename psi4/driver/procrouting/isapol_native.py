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
from .sapt.fdds_response import FDDSFullOVResponse
from .isapol_native_correction import validate_correction, functional_definition
from . import isapol_lw as lw


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
            algorithm='ordered_pairwise'):
    # The named response algorithm is part of the policy: the two arrangements
    # are separately gated, so a context built under one is not reusable under
    # the other even though their operators agree.
    h = hashlib.sha256(repr((kernel, exact_exchange, local_scale, density_cutoff, correction,
                             algorithm)).encode())
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
                      scf_correction='NONE', expected_grac_shift=None,
                      response_algorithm='ordered_pairwise', ov_charge_penalty=1.):
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
    """
    if response_basis not in ('fitted_auxiliary', 'direct_ov'):
        raise ValueError('unsupported response_basis')
    if (type(ov_charge_penalty) is not float or not np.isfinite(ov_charge_penalty)
            or ov_charge_penalty <= 0.):
        raise ValueError('ov_charge_penalty must be an explicit finite positive float')
    if response_basis == 'direct_ov' and ov_charge_penalty != 1.:
        raise ValueError('direct_ov forms no transition fit; no charge penalty applies')
    if caller_converged is not True:
        raise ValueError('caller_converged must explicitly be True')
    if not isinstance(recipe, PartitionRecipe):
        raise TypeError('explicit PartitionRecipe required')
    if len(set(s.rank for s in recipe.sites)) != 1 or recipe.sites[0].rank not in (3, 4):
        raise ValueError('LW pipeline requires uniform explicit rank3 or rank4')
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
                                     expected_grac_shift=expected_grac_shift)
    if correction.policy == 'FIXED_GRAC' and kernel == 'no_local':
        raise ValueError('FIXED_GRAC admission requires an explicit ALDA response policy; no GRAC kernel derivative')
    context_hash = _context(wfn)
    policy_hash = _policy(kernel, exact_exchange, local_scale, response_grid, density_cutoff,
                          correction, response_algorithm)
    if response_context is not None:
        if (not isinstance(response_context, NativeContext)
                or response_context.wavefunction_sha256 != context_hash
                or response_context.policy_sha256 != policy_hash
                or response_context.correction_provenance != correction):
            raise ValueError('native response context mismatch (state/basis/geometry/policy)')
    partition = native_partition(wfn, recipe, caller_converged=caller_converged)
    context, fit, adapted, distributed, tensors, local, dispersion = response_context, None, None, None, None, None, None
    responses, failures, diagnostics = [], [], {}
    stage = 'partition'
    def result():
        return NativeProperties(partition, context, fit, adapted, freq, quadrature, tuple(responses),
            distributed, tensors, local, dispersion, tuple(failures), diagnostics,
            f'native {kernel}; exact_exchange={exact_exchange}; local_scale={local_scale}; '
            f'{response_algorithm}; Drho-C ISA-A[{recipe.auxiliary.name}]; {response_basis}'
            + ('' if response_basis == 'direct_ov' else f' lambda={ov_charge_penalty!r}')
            + '; no PFIT'
            + ('; ' + correction.response_description if correction.policy == 'FIXED_GRAC' else ''),
            correction)
    try:
        q = partition.require_q()
        stage = 'context'
        if _context(wfn) != context_hash:
            raise ValueError('wavefunction context changed during partition')
        if context is None:
            response = native_response_from_wavefunction(wfn, caller_converged=True, kernel=kernel,
                exact_exchange=exact_exchange, local_scale=local_scale, grid=response_grid,
                density_cutoff=density_cutoff, max_bytes=max_bytes, max_nov=max_nov,
                scf_correction=scf_correction, expected_grac_shift=expected_grac_shift,
                algorithm=response_algorithm)
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
        stage = 'OV moments' if response_basis == 'direct_ov' else 'OV fit'
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
                + context_hash, ov_charge_penalty)
            d = np.asarray(fit.coefficients)
            if d.shape != (provider.nocc*provider.nvir, partition.auxiliary.nfunction):
                raise ValueError('native OV fit dimensions/order mismatch')
        if _context(wfn) != context_hash:
            raise ValueError('wavefunction changed during native fit')
        solver = FDDSFullOVResponse(h1_baseline=np.asarray(provider.h1()), h2=np.asarray(provider.h2()),
            transition_legs=d, coupling=np.zeros((d.shape[1], d.shape[1])),
            representation=('supplied_transition_leg_coordinates' if response_basis == 'direct_ov'
                            else 'fitted_density_coefficients'))
        stage = 'frequency response'
        for xi in freq:
            responses.append(solver.at_frequency(xi))
        stage = 'distributed response'
        distributed = core.IsaDistributedResponse(q, freq,
            [core.Matrix.from_array(r.raw_coupled) for r in responses],
            'direct_ov' if response_basis == 'direct_ov' else 'fitted_density_coefficients',
            'native full H1/H2; '+response_basis+'; '+context_hash)
        n, rank = len(recipe.sites), recipe.sites[0].rank
        m = (rank+1)**2
        raw = np.array([np.asarray(distributed.at_index(k)).reshape(n,m,n,m).transpose(0,2,1,3)
                        for k in range(len(freq))])
        tensors = lw.ArraySnapshot.of(raw)
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
            raw_charge_sum_maxabs=[float(np.max(np.abs(a[:,: ,0,:16].sum(axis=0)))) for a in raw],
            analytic_charge_response_maxabs=[float(np.max(np.abs(analytic_q @ r.raw_coupled @ np.asarray(q.values).T))) for r in responses],
            quadrature_charge_response_maxabs=[float(np.max(np.abs((qsum-analytic_q) @ r.raw_coupled @ np.asarray(q.values).T))) for r in responses])
        provenance = lw.Provenance('fresh native distributed tensors', tensors.canonical_array_sha256,
            'Psi4 native fitted response and IsaDistributedResponse', partition.provenance+'; '+context_hash)
        args = dict(labels=q.labels, origins=q.origins, bonds=bonds, frames=frames,
                    input_rank=rank, truncation=lw.TRUNCATE_RANK4 if rank == 4 else None,
                    provenance=provenance, residual_policy='production')
        stage = 'LW'
        # Attempt EVERY node independently; never relax or hide a failed frequency.
        for k, xi in enumerate(freq):
            try:
                lw.supplied_nonlocal_properties(frequencies=(xi,), tensors=raw[k:k+1], **args)
            except Exception as exc:
                failures.append(StageFailure(stage, xi, type(exc).__name__, str(exc)))
        if not failures:
            local = lw.supplied_nonlocal_properties(frequencies=freq, tensors=raw, **args)
            if pair_self or partner is not None:
                stage = 'dispersion'
                dispersion = lw.isotropic_dispersion(local, local if pair_self else partner,
                    cp_weights=quadrature.cp_weights, quadrature_provenance=quadrature.provenance,
                    max_order=max_order)
    except Exception as exc:
        failures.append(StageFailure(stage, None, type(exc).__name__, str(exc)))
    return result()
