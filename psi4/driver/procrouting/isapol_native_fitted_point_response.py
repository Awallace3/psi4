# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Exact point potentials contracted with an explicitly selected native AUX response.

This does not run SCF, fit, partition, or solve response. The caller selects a
target calculation independently of the localization/anchor calculation.
In particular an eta=0 target must not silently reuse eta=0.0005 anchors.
No CamCASP parity is implied by the fitted representation.
"""
from dataclasses import dataclass
import hashlib
from numbers import Real

import numpy as np
from psi4 import core

from . import isapol_native as native
from .isapol_native_point_response import (
    CONVENTION, MAXIMUM_POINTS, NativePointChargeResponse,
    _validated_frequencies, _validated_points,
)


@dataclass(frozen=True)
class NativeFittedPointResponse(NativePointChargeResponse):
    """Owned fitted targets; ``operators`` holds the AUX-by-point potential matrix.

    Only batch assembly and packing semantics are shared with the direct result.
    The origin is native-fitted, never native-direct or supplied-fitted.
    """
    auxiliary_basis_id: str
    charge_penalty: float
    metric_damping: float

    def target_provenance(self, source_id):
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError('explicit nonempty source_id required')
        provenance = core.IsaPfitTargetProvenance()
        provenance.origin = core.IsaPfitTargetOrigin.NativeFittedPointResponse
        provenance.convention = (
            core.IsaPfitTargetConvention.NegativeInducedPotentialPerUnitSourceChargeAtomicUnits)
        provenance.source_id = source_id
        provenance.response_representation = self.representation
        provenance.auxiliary_basis_id = self.auxiliary_basis_id
        provenance.generation_record = self.generation_record
        return provenance


def native_fitted_point_response(properties, wavefunction, points_bohr, *,
                                 charge_penalty, metric_damping,
                                 max_bytes=512 * 1024**2):
    """Contract v(iw) = -B.T C_aux(iw) B on the selected target's complete grid.

    B[k,p] is integral chi_k(r)/|r-R_p| dr in the target's explicit AUX.
    Retained coefficient responses carry the *effective* propagator, including
    any declared rebuild; the original context provider's H1/H2 are not reused.
    Required lambda/eta arguments must exactly match the selected target fit.
    They do not request a refit. No rank truncation, symmetrization, nuclear
    term, energy half factor or quadrature weight enters this contraction.

    NativeProperties is a trusted factory record, not a serialization/security
    boundary. Context is checked against the live wavefunction; forwarded SCF
    convergence/correction declarations are not independently recertified.
    max_bytes bounds new dense numerical buffers (including returned arrays),
    not process peak, existing properties, hashing overhead or BLAS workspace.
    """
    if (isinstance(max_bytes, (bool, np.bool_))
            or not isinstance(max_bytes, (int, np.integer)) or max_bytes <= 0):
        raise ValueError('max_bytes must be a positive integer')
    if not isinstance(properties, native.NativeProperties):
        raise TypeError('native target NativeProperties required')
    if properties.ov_fit is None or properties.context is None:
        raise ValueError('target requires a native fitted AUX response')
    for name, value in (('charge_penalty', charge_penalty), ('metric_damping', metric_damping)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError(f'{name} must be a finite real declaration')
    fit = properties.ov_fit
    if (charge_penalty != fit.charge_penalty or
            metric_damping != fit.offsite_metric_damping):
        raise ValueError('target fit declaration mismatch; select a separately computed target fit')
    context = properties.context
    if native._context(wavefunction) != context.wavefunction_sha256:
        raise ValueError('target wavefunction context mismatch')
    response = context.response
    if not response.caller_converged:
        raise ValueError('target requires explicit caller convergence')
    points = _validated_points(points_bohr, MAXIMUM_POINTS)
    frequencies = _validated_frequencies(properties.frequencies)
    coefficient_responses = properties.coefficient_responses
    if len(coefficient_responses) != len(frequencies):
        raise ValueError('target requires the complete coefficient-response frequency grid')
    naux, npoint, nfreq = properties.partition.auxiliary.nfunction, len(points), len(frequencies)
    packed_size = npoint*(npoint+1)//2
    # B plus its core/NumPy copy, B^T C workspace, stored full/packed responses,
    # three point matrices for contraction/diagnostics, owned points, the
    # fit.coefficients getter's owned copy, and the AUX^2 finite-check mask.
    nov = response.provider.nocc * response.provider.nvir
    needed = 8*(3*naux*npoint + nfreq*(npoint*npoint+packed_size)
                + 3*npoint*npoint + 3*npoint + nov*naux) + naux*naux
    if needed > max_bytes:
        raise ValueError('fitted point response byte resource limit')
    for omega, item in zip(frequencies, coefficient_responses):
        matrix = np.asarray(item.raw_coupled)
        if (item.omega != omega or item.representation != 'fitted_density_coefficients'
                or matrix.shape != (naux, naux) or np.iscomplexobj(matrix)
                or not np.isfinite(matrix).all()):
            raise ValueError('target coefficient-response frequency/representation/shape mismatch')
    basis_record = repr(properties.partition.recipe.auxiliary)
    basis_id = hashlib.sha256(basis_record.encode()).hexdigest()
    operators = properties.partition.coulomb.point_potentials(
        core.Matrix.from_array(points), int(max_bytes))
    potentials = np.asarray(operators)
    digest = hashlib.sha256()
    for text in (basis_record, context.wavefunction_sha256, context.policy_sha256,
                 properties.model, str(properties.correction_provenance),
                 str(response.convergence_evidence), repr((charge_penalty, metric_damping)),
                 CONVENTION):
        digest.update(text.encode())
    for block in (points, frequencies, potentials, np.asarray(fit.coefficients)):
        array = np.asarray(block, dtype='<f8')
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    values, packed, defects, diagonals, magnitudes = [], [], [], [], []
    for item in coefficient_responses:
        raw = np.asarray(item.raw_coupled)
        digest.update(raw.astype('<f8', copy=False).tobytes())
        value = -(potentials.T @ raw @ potentials)
        if not np.isfinite(value).all():
            raise ValueError('nonfinite fitted point response')
        target = np.concatenate([value[i, :i+1] for i in range(npoint)])
        defects.append(float(np.max(np.abs(value-value.T))))
        diagonals.append(float(np.min(np.diag(value))))
        magnitudes.append(float(np.max(np.abs(value))))
        value.flags.writeable = target.flags.writeable = False
        values.append(value)
        packed.append(target)
    points.flags.writeable = False
    record = (f'native fitted AUX point response; v=-B^T C_aux B; '
              f'B=integral chi_k(r)/|r-R_p| dr; AUX sha256={basis_id}; '
              f'lambda={float(charge_penalty)!r}; eta={float(metric_damping)!r}; '
              f'retained coefficient responses; target model={properties.model}; '
              'no multipole truncation; no refit; no CamCASP parity claim')
    return NativeFittedPointResponse(
        operators=operators, points_bohr=points, frequencies_au=tuple(map(float, frequencies)),
        responses=tuple(values), packed_targets=tuple(packed),
        reciprocity_defects=tuple(defects), minimum_diagonals=tuple(diagonals),
        maximum_absolute_values=tuple(magnitudes), convention=CONVENTION,
        representation='fitted_density_coefficients', generation_record=record,
        context_sha256=digest.hexdigest(), correction_provenance=properties.correction_provenance,
        caller_converged=bool(response.caller_converged),
        convergence_evidence=response.convergence_evidence,
        auxiliary_basis_id=basis_id, charge_penalty=float(charge_penalty),
        metric_damping=float(metric_damping))
