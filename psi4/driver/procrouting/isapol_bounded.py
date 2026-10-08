# Copyright (c) 2007-2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""Explicit bounded response -> distributed moments -> LW -> PFIT -> isotropic dispersion.

The entry point takes a live converged wavefunction and scientific
declarations, never reference output or a legacy NativeProperties record.
The response is the stage-04 ``BoundedResponse`` (reference H2H1 by default,
explicit native FDDS on request), the targets and refinement are the stage-05
``fitted_point_targets``/``refine_streamed`` and the contraction is the
stage-06 ``refined_isotropic_dispersion``. This module owns only their order,
the shared resource ledger and the result record.

Distributions are the analytic DF-centre producer (default) and supplied
``DistributedMoments``. No ISA or MBIS partition is performed or inferred.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib

import numpy as np
from psi4 import core

from . import isapol_logging as _lg, isapol_lw as lw, isapol_refine as refine
from .isapol_bounded_response import BoundedResources, BoundedResponse
from .isapol_distribution import DistributedMoments
from .isapol_pfit_stream import PackedDesignRows, fitted_point_targets, refine_streamed

#: Distributions this orchestrator dispatches; every other name is refused.
DISTRIBUTIONS = ('df_centre_analytic', 'supplied')
#: Bounded lattice and refinement sizes (complete cloud, every point pair).
MAX_FIT_POINTS = 2000
MAX_PARAMETERS = 64
#: Rows per PFIT design block.
BLOCK_ROWS = 4096


@dataclass(frozen=True)
class BoundedProperties:
    """Completed campaign; local and refined quantities remain distinct.

    The ledger reports explicit numeric plans, not measured resident memory.
    Arrays retained here are small local tensors; AUX and full OV intermediates
    are private ephemeral disk checkpoints, removed when the call exits.
    """
    frequencies: tuple
    local_tensors: tuple
    refinements: tuple
    dispersion: object
    diagnostics: tuple
    resources: dict
    provenance: dict

    @property
    def refined_tensors(self):
        return tuple(r.refined_tensors for r in self.refinements)


@contextmanager
def _closes_failed_stage(log):
    """Close the open stage as FAILED on any error, so it is never reported complete."""
    try:
        yield
    except Exception as error:
        log.stage_failed(error)
        raise


def _scalar(name, value, upper=float('inf')):
    if (isinstance(value, (bool, np.bool_)) or not np.isscalar(value)
            or not np.isfinite(value) or not 0 <= value < upper):
        raise ValueError(f'invalid {name}')


def _target_bytes(naux, npoint):
    """``fitted_point_targets`` plan for one response, plus the live coefficient response."""
    row_count, block = npoint*(npoint+1)//2, min(32, npoint)
    return 8*(3*npoint+3*block+2*naux*npoint+npoint*npoint+naux*block+row_count) + 8*naux*naux


def _pfit_plan(npoint, channels, count, nt):
    """Planned bytes and work of the complete-cloud refinement.

    Copied fields, row producer and C++ solver workspaces on top of the
    retained packed targets; both design passes are charged up front.
    """
    row_count, chunk = npoint*(npoint+1)//2, PackedDesignRows.MAX_CHUNK
    plan = (8*(5*npoint*channels+npoint*channels*count+2*chunk*npoint*count
               +16*count*count*nt+BLOCK_ROWS*(8*count+32*nt)) + 8*1024**2)
    work = (2*((row_count+chunk*npoint)*2*channels*count+row_count*(4*count+8))
            +2*row_count*count*count+4*row_count*count*nt)
    return plan, work


def bounded_properties(wfn, auxiliary_recipe, *, caller_converged, distribution='df_centre_analytic',
                       sites, bonds, quadrature, response_grid, smoothing,
                       shell_cutoff, charge_penalty, anchor_metric_damping,
                       lattice_options, localization_rank_limit,
                       weight_type, weight_coefficient, cutoff, resources,
                       scf_correction, expected_grac_shift=None,
                       declared_variables=None, max_order=10, scratch_directory=None, log=None,
                       distributed_moments=None, response='reference_h2h1', fdds=None,
                       publish_qcvariables=False):
    """Run the complete native chain under one explicit resource contract.

    Scientific scope: canonical restricted C1 PBE0, NONE or FIXED_GRAC, plain
    Coulomb-DF operators, Slater/PW92 ALDA of the plain fitted density, and
    rank-4 distributed moments. ``response='reference_h2h1'`` (default) solves
    the eta=0 target/kernel and separately damped anchor legs and localizes
    under the strict production LW gate. ``response='native_fdds'`` (with
    ``fdds=NativeFDDSOptions``) uses the native declared-AUX FDDS response and
    the ``reported_input_sum_rule`` LW policy that runner declares: local
    charge is report-only and the rank-0 remainder is not carried into the
    local tensors, PFIT or C_n. All sites/bonds/frames/ranks, the full
    response grid, quadrature, lattice and fit constraints are caller inputs.
    No ambient options, inferred carbon policy or external reference readers.

    ``distribution='df_centre_analytic'`` charges each AUX function to its own
    centre; ``distribution='supplied'`` takes caller ``distributed_moments``
    under the same identity checks. Neither changes a response or LW gate.

    A successful live SCF seal is required even for NONE; SCF is never run here.
    ``lattice_options`` is an explicit core.FitPointsOptions (at most 2000
    points); all pairs and both PFIT passes are used, with at most 64 fit
    variables. ``scratch_directory`` is the parent of a fresh private temporary
    directory, never overwritten or resumed. Caller-owned wavefunction/basis
    metadata and BLAS/Libint workspace are outside the numeric cap. ``log`` is
    an optional isapol_logging.StageLog for timings, diagnostics and the
    dispersion tables. ``publish_qcvariables`` publishes the refined-dispersion
    QCVariables on ``wfn`` after a successful contraction (the oeprop route);
    by default nothing is written to the wavefunction.
    """
    if log is None:
        log = _lg.StageLog(1)
    if not isinstance(log, _lg.StageLog):
        raise TypeError('log must be a StageLog')
    with _closes_failed_stage(log):
        log.stage('Bounded input validation')
        if not isinstance(resources, BoundedResources):
            raise TypeError('explicit BoundedResources required')
        if distribution not in DISTRIBUTIONS:
            raise ValueError(f'distribution must be one of {DISTRIBUTIONS}; '
                             'no ISA or MBIS partition is available here')
        if distribution == 'supplied':
            if not isinstance(distributed_moments, DistributedMoments):
                raise TypeError('supplied distribution requires DistributedMoments')
        elif distributed_moments is not None:
            raise ValueError('distributed_moments requires distribution=supplied')
        if not isinstance(lattice_options, core.FitPointsOptions):
            raise TypeError('explicit FitPointsOptions required')
        if not 1 <= lattice_options.npoints <= MAX_FIT_POINTS:
            raise ValueError(f'bounded lattice requires 1..{MAX_FIT_POINTS} points')
        if type(localization_rank_limit) is not int or not 1 <= localization_rank_limit <= 3:
            raise ValueError('localization_rank_limit must be 1..3')
        sites, bonds = tuple(sites), tuple(tuple(b) for b in bonds)
        if not 1 <= len(sites) <= refine.MAX_SITES or any(not isinstance(s, refine.RefinementSite) for s in sites):
            raise TypeError('explicit RefinementSite sequence required')
        if len({s.label for s in sites}) != len(sites):
            raise ValueError('site labels must be unique')
        if publish_qcvariables:
            # Refuse unpublishable labels now, not after the whole chain.
            _lg.check_publication_labels(tuple(s.label for s in sites))
        seen = set()
        for edge in bonds:
            if (len(edge) != 2 or any(type(i) is not int or not 0 <= i < len(sites) for i in edge)
                    or edge[0] == edge[1] or tuple(sorted(edge)) in seen):
                raise ValueError('invalid self/duplicate/out-of-range bond')
            seen.add(tuple(sorted(edge)))
        if type(max_order) is not int or max_order not in refine.DISPERSION_ORDERS:
            raise ValueError('max_order must be 6, 8, 10 or 12')
        if type(weight_type) is not int or weight_type not in refine.WEIGHT_TYPES:
            raise ValueError('invalid weight_type')
        for name, value in (('weight_coefficient', weight_coefficient), ('cutoff', cutoff)):
            _scalar(name, value)
        declared_variables = None if declared_variables is None else tuple(declared_variables)
        if declared_variables is not None and not 1 <= len(declared_variables) <= MAX_PARAMETERS:
            raise ValueError(f'bounded refinement requires 1..{MAX_PARAMETERS} declared variables')
        if any(s.rank_limit > localization_rank_limit for s in sites):
            raise ValueError('refinement rank exceeds localization rank')
        if (not isinstance(response_grid, np.ndarray) or response_grid.dtype != np.float64
                or response_grid.ndim != 2 or response_grid.shape[1] != 4
                or not len(response_grid)):
            raise ValueError('finite float64 full response grid (rows,4) required')
        # The runner holds the caller grid, its copy and a validation overlap.
        if 3*response_grid.nbytes > resources.max_bytes:
            raise ValueError('full response grid numeric byte resource limit')
        multipole_sites = []
        for site in sites:
            value = core.IsaMultipoleSite()
            value.label, value.origin, value.rank = site.label, list(site.origin_bohr), 4
            multipole_sites.append(value)
        origins = np.asarray([s.origin_bohr for s in sites])
        input_bytes = distributed_moments.values.nbytes if distribution == 'supplied' else 0
        # The runner validates the shared declarations (AUX/site/geometry centres,
        # grid finiteness, correction, smoothing, fit scalars, response selection).
        runner = BoundedResponse(wfn, auxiliary_recipe, multipole_sites, caller_converged=caller_converged,
            quadrature=quadrature, response_grid=response_grid, smoothing=smoothing, shell_cutoff=shell_cutoff,
            charge_penalty=charge_penalty, anchor_metric_damping=anchor_metric_damping, resources=resources,
            scf_correction=scf_correction, expected_grac_shift=expected_grac_shift,
            scratch_directory=scratch_directory, log=log, label=distribution, input_bytes=input_bytes,
            response=response, fdds=fdds)
        with runner as solver:
            ledger, auxiliary = solver.ledger, solver.auxiliary
            p = auxiliary.nfunction
            q, nf = 25*len(sites), len(quadrature.frequencies)
            source = dict(state_sha256=solver.state_sha256, auxiliary_sha256=solver.auxiliary_sha256,
                          correction=repr(solver.correction), distribution=distribution,
                          grid_sha256=solver.grid_sha256,
                          kernel_smoothing=repr(smoothing), shell_cutoff=float(shell_cutoff),
                          sites=repr(sites), bonds=bonds, frequencies=quadrature.frequencies,
                          cp_weights=quadrature.cp_weights, localization_rank_limit=localization_rank_limit,
                          weight_type=weight_type, weight_coefficient=weight_coefficient, cutoff=cutoff,
                          declared_variables=declared_variables, model=solver.model,
                          response=response, lw_residual_policy=solver.lw_residual_policy,
                          lw_disclosure=solver.lw_disclosure)
            solver.prepare(None if distribution == 'df_centre_analytic' else (lambda ledger: distributed_moments))
            source['partition'] = solver.partition
            if solver.provenance is not None:
                source['response_provenance'] = solver.provenance
            stability = solver.stability
            npoint = lattice_options.npoints
            log.stage('Native point cloud',
                      (('points', npoint), ('seed', lattice_options.seed),
                       ('inner / outer radii', (lattice_options.lolim, lattice_options.hilim))))
            ledger.admit('native fit cloud', 8*12*npoint+8*1024**2)
            cloud = core.FitPoints(wfn.molecule().clone(), lattice_options)
            points = np.column_stack((cloud.x(), cloud.y(), cloud.z()))
            del cloud
            source['lattice_sha256'] = hashlib.sha256(points.tobytes()).hexdigest()
            # The points and every node's packed targets are retained for one
            # shared-design PFIT after the sweep.
            row_count = npoint*(npoint+1)//2
            retained = 8*(row_count*nf+3*npoint)
            ledger.reserved += retained
            ledger.admit('retained point-response targets', 0)
            packed = np.empty((row_count, nf))
            local_tensors, models, diagnostics = [], [], []
            frames = [s.frame for s in sites]
            if response == 'reference_h2h1':
                solve_record, localization = 'native Psi4 original H2H1 solve', 'Production'
            else:
                solve_record, localization = 'native Psi4 declared-AUX FDDS solve', 'Reported-input'
            for node, omega in enumerate(quadrature.frequencies, 1):
                tag = f'node {node}/{nf}, omega={omega:.12g} au'
                solved = solver.solve(omega)
                coefficient, raw, residual = solved.target_response, solved.nonlocal_response, solved.residual
                response_diagnostics = solved.diagnostics
                del solved
                log.stage(f'{localization} LW localization: ' + tag,
                          (('rank limit', localization_rank_limit), ('bonds', len(bonds)),
                           ('residual policy', solver.lw_residual_policy)))
                ns = len(sites)
                # Mirror LW's conservative native workspace reservation, plus live
                # coefficient/raw matrices and Python snapshot/copy overlap.
                local_plan = (3*ns*ns*5000+(2*len(bonds)+ns)*5000+ns*(4608+24)
                              +16*ns*ns*8+4*1000000*48+8*p*p+32*q*q)
                ledger.admit(f'{localization.lower()} localization', local_plan, 8*q**3)
                tensors = raw.reshape(ns, 25, ns, 25).transpose(0, 2, 1, 3)[None]
                provenance = lw.Provenance('bounded native response', hashlib.sha256(raw.tobytes()).hexdigest(),
                                           solve_record, source['model'])
                local = lw.supplied_nonlocal_properties(labels=[s.label for s in sites], origins=origins,
                    bonds=bonds, frames=frames, frequencies=[omega], tensors=tensors, input_rank=4,
                    provenance=provenance, truncation=lw.TRUNCATE_RANK4,
                    residual_policy=solver.lw_residual_policy, localization_rank_limit=localization_rank_limit)
                anchors = []
                for i, site in enumerate(sites):
                    a = np.zeros((site.component_count, site.component_count))
                    a[1:, 1:] = local.raw_local.array[0, i, :site.component_count-1, :site.component_count-1]
                    anchors.append(a)
                model = refine.refinement_model(sites, anchors, frequency_au=omega, cutoff=cutoff,
                    weight_type=weight_type, weight_coefficient=weight_coefficient,
                    declared_variables=declared_variables, provenance=source['model'])
                if declared_variables is None:
                    # localize.py runs process on the first (static) node while no
                    # .pdef exists; process writes it (process_data.F90:2046-2058)
                    # and every later node reads that file, not its own cutoff scan.
                    # Every node then carries that one declared list, so stage06
                    # sees a single declared model; on this node the explicit
                    # list reproduces the cutoff scan exactly.
                    declared_variables = model.parameter_labels
                    derived = model
                    model = refine.refinement_model(sites, anchors, frequency_au=omega, cutoff=cutoff,
                        weight_type=weight_type, weight_coefficient=weight_coefficient,
                        declared_variables=declared_variables, provenance=source['model'])
                    if (model.parameter_entries, model.anchors, model.strengths) != (
                            derived.parameter_entries, derived.anchors, derived.strengths):
                        raise RuntimeError('cutoff-derived variable list does not reproduce its own model')
                    del derived
                    log.items((('cutoff-derived variables from this node', len(declared_variables)),))
                local_tensors.append(local.raw_local)
                diagnostics.append(dict(frequency=omega, response_residual=residual,
                    localization_residual=local.frequency_diagnostics[0].residuals.maximum,
                    response_diagnostics=response_diagnostics))
                log.items((('maximum localization residual', diagnostics[-1]['localization_residual']),))
                # The fitting phase owns only the AUX coefficient response and its
                # small model; do not carry full nonlocal/LW input snapshots into
                # that phase's independently planned workspace.
                del raw, tensors, local, anchors
                if node == 1:
                    # The variable set is fixed here for every node, so the
                    # parameter cap is admitted before any targets or later solves.
                    log.stage('Refinement model admission', (('parameters', model.parameter_count),
                              ('limit', MAX_PARAMETERS), ('declared variables', len(model.declared_variables))))
                    if model.parameter_count > MAX_PARAMETERS:
                        raise ValueError(f'bounded refinement supports at most {MAX_PARAMETERS} parameters')
                log.stage(f'Point-response targets: {tag}', (('points', npoint),))
                ledger.admit('point-response targets', _target_bytes(p, npoint),
                             2*p*p*npoint+2*p*npoint*npoint)
                packed[:, node-1] = fitted_point_targets(auxiliary, points, [coefficient],
                                                         max_bytes=resources.max_bytes-ledger.reserved)[:, 0]
                models.append(model)
                del coefficient, model
            solver.release()
            count, channels = models[0].parameter_count, models[0].channel_count
            log.stage('Complete-cloud PFIT',
                      (('parameters', count), ('rows per pass', row_count),
                       ('frequencies', nf), ('passes', 2), ('weight type', weight_type),
                       ('weight coefficient', weight_coefficient), ('cutoff', cutoff)))
            ledger.admit('complete-cloud refinement', *_pfit_plan(npoint, channels, count, nf))
            results = refine_streamed(models, points, packed, source_id=source['state_sha256'],
                generation_record=source['model'], target_origin=core.IsaPfitTargetOrigin.NativeFittedPointResponse,
                response_representation='fitted_density_coefficients',
                auxiliary_basis_id=source['auxiliary_sha256'], block_rows=BLOCK_ROWS,
                max_bytes=resources.max_bytes-ledger.reserved)
            del packed
            ledger.reserved -= retained
            for node, result in enumerate(results, 1):
                log.items(((f'node {node} solver status', str(result.status)),
                           (f'node {node} maximum parameter change from anchor', result.anchor_shift_maxabs)))
            dispersion = refine.refined_isotropic_dispersion(results, cp_weights=quadrature.cp_weights,
                quadrature_provenance=quadrature.provenance, max_order=max_order, log=log,
                wfn=wfn if publish_qcvariables else None)
            log.stage('Bounded resource totals')
            log.items((('site pairs', len(dispersion.pairs)), ('peak planned numeric bytes', ledger.peak),
                       ('charged work', ledger.work), ('checkpoint I/O bytes', ledger.io)))
            log.stage_end()
        rhs = nf*(p+q) if response == 'reference_h2h1' else None
        return BoundedProperties(quadrature.frequencies, tuple(local_tensors), tuple(results), dispersion,
            tuple(diagnostics), dict(maximum_numeric_plan=ledger.peak, charged_work=ledger.work,
            charged_io_bytes=ledger.io, frequency_rhs=rhs, fit_rows=nf*npoint*(npoint+1),
            stages=ledger.stages), dict(source, stability=stability))
