# Copyright (c) 2007-2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""Explicit bounded PBE0/DF-centre response → LW → PFIT → dispersion.

The entry point takes a live converged wavefunction and scientific declarations,
never reference output or a legacy NativeProperties record. It uses plain-DF
two-electron operators and a screened fitted-density ALDA kernel with eta-zero
kernel/target legs; separately fitted anchor legs enter production LW. This
declared DF-centre model does NOT perform or infer an ISA-A partition.

Existing native_properties/ATOMIC_REFINED defaults and resource limits are
unchanged. This opt-in Python route has its own cumulative resource admission.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import tempfile

import numpy as np
from psi4 import core

from . import isapol_lw as lw, isapol_refine as refine
from .isapol_bounded_response import (
    BoundedResources, _Ledger, _Store, _assemble, _frequency, _kernel, _stability,
)
from .isapol_df_multipoles import df_centre_multipoles
from .isapol_native import Quadrature, _context
from .isapol_native_correction import validate_correction, require_scf_seal
from .isapol_native_factors import native_plain_df_operators, native_constrained_ov
from .isapol_native_partition import BasisRecipe, adapt_main
from .isapol_native_propagator import KernelSmoothing
from .isapol_native_response import native_restricted_state_from_wavefunction
from .isapol_pfit_stream import PackedDesignRows


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


def _matrix(a):
    a = np.asarray(a, dtype=float)
    result = core.IsaPfitMatrix()
    result.rows, result.cols, result.values = a.shape[0], a.shape[1], a.ravel().tolist()
    return result


def _stream_fit(model, points, coefficient, potentials, ledger, source):
    npoint, naux = len(points), len(coefficient)
    channels, count = model.channel_count, model.parameter_count
    if count > 64:
        raise ValueError('bounded refinement supports at most 64 parameters')
    row_count = npoint*(npoint+1)//2
    # Full target/packed overlap, AUX operands, copied fields, row producer and
    # C++ solver workspaces. No per-block or per-frequency allowance reset.
    chunk = PackedDesignRows.MAX_CHUNK
    plan = (8*(3*npoint*npoint+row_count+3*naux*npoint+naux*naux
               +5*npoint*channels+npoint*channels*count+2*chunk*npoint*count
               +16*count*count+4096*(8*count+32)) + 8*1024**2)
    work = (2*naux*naux*npoint+2*naux*npoint*npoint
            +2*((row_count+chunk*npoint)*2*channels*count+row_count*(4*count+8))
            +2*row_count*count*count)
    ledger.admit('complete-cloud refinement', plan, work)
    fields = np.empty((npoint, channels))
    for start in range(0, npoint, 256):
        fields[start:start+256] = refine.channel_fields(points[start:start+256], model)
    targets = -(potentials.T @ coefficient) @ potentials
    packed = np.concatenate([targets[i, :i+1] for i in range(npoint)])
    del targets
    rows = PackedDesignRows(model, fields, block_rows=4096, max_passes=2,
                            max_bytes=ledger.resources.max_bytes-ledger.reserved)
    problem = core.IsaPfitRowProblem()
    problem.frequency_au = model.frequency_au
    declaration = core.IsaPfitRowModel()
    declaration.channel_labels = list(model.channel_labels)
    declaration.parameter_labels = list(model.parameter_labels)
    declaration.parameter_units = ['atomic_units']*count
    declaration.fixed, declaration.fixed_values = [False]*count, [0.]*count
    declaration.provenance = model.provenance
    problem.model = declaration
    cloud = core.IsaPfitCloudRows()
    cloud.label, cloud.points, cloud.full_row_count = 'complete native cloud', npoint, row_count
    cloud.maximum_block_rows = 4096
    problem.cloud = cloud
    penalty = core.IsaPfitMatrixPenalty()
    penalty.matrix, penalty.anchor = _matrix(np.diag(model.strengths)), list(model.anchors)
    problem.penalty = penalty
    provenance = core.IsaPfitTargetProvenance()
    provenance.origin = core.IsaPfitTargetOrigin.NativeFittedPointResponse
    provenance.convention = core.IsaPfitTargetConvention.NegativeInducedPotentialPerUnitSourceChargeAtomicUnits
    provenance.source_id = source['state_sha256']
    provenance.response_representation = 'fitted_density_coefficients'
    provenance.auxiliary_basis_id = source['auxiliary_sha256']
    provenance.generation_record = source['model']
    problem.target_provenance = provenance
    def blocks():
        for start, design in rows.blocks():
            yield start, design, packed[start:start+len(design)].copy()
    options = core.IsaPfitOptions()
    options.solver = core.IsaPfitSolver.NormalEquationsDSYSV
    result = core.isa_pfit_solve_rows(problem, blocks, options)
    if result.status != core.IsaPfitStatus.Solved:
        raise ValueError(f'PFIT did not solve: {result.status}')
    parameters = tuple(result.parameters)
    tensors = [np.zeros((s.component_count, s.component_count)) for s in model.sites]
    for value, entries in zip(parameters, model.parameter_entries):
        for site, i, j in entries:
            tensors[site][i, j] = tensors[site][j, i] = value
    tensors = tuple(np.frombuffer(t.tobytes(), dtype=float).reshape(t.shape) for t in tensors)
    return refine.RefinementResult(
        model, parameters, tensors, max(abs(a-b) for a, b in zip(parameters, model.anchors)),
        result.status, result.diagnostics, result, 'PFIT_refined_against_point_to_point_response')


def bounded_properties(wfn, auxiliary_recipe, *, caller_converged, distribution,
                       sites, bonds, quadrature, response_grid, smoothing,
                       shell_cutoff, charge_penalty, anchor_metric_damping,
                       lattice_options, localization_rank_limit,
                       weight_type, weight_coefficient, cutoff, resources,
                       scf_correction, expected_grac_shift=None,
                       declared_variables=None, max_order=10, scratch_directory=None, log=None):
    """Run the complete native chain under one explicit resource contract.

    Scientific scope: canonical restricted C1 PBE0, NONE or FIXED_GRAC, plain
    Coulomb-DF operators, Slater/PW92 ALDA of the plain fitted density,
    df_centre_analytic rank-4 distribution, eta=0 target/kernel fit, and the
    separately declared anchor damping. All sites/bonds/frames/ranks, full
    response grid, quadrature, lattice and fit constraints are caller inputs.
    No ambient options, inferred carbon policy or external reference readers.

    A successful live SCF seal is required even for NONE. This function neither
    runs SCF nor manufactures convergence evidence for a serialized state.
    ``lattice_options`` is an explicit core.FitPointsOptions (at most 2000
    points). All pairs and both PFIT passes are included; no truncated cloud.
    ``scratch_directory`` is a parent for a fresh private temporary directory,
    never a directory to overwrite or resume. Caller-owned wavefunction/basis
    metadata and implementation-specific BLAS/Libint workspace are outside the
    explicit numeric cap. Return values make no automatic CamCASP parity claim.
    ``log`` optionally supplies an isapol_logging.StageLog for stage timings,
    input dimensions and numerical diagnostics; no numerical policy is changed.
    """
    from .isapol_logging import StageLog
    if log is None:
        log = StageLog(1)
    if not isinstance(log, StageLog):
        raise TypeError('log must be a StageLog')
    log.stage('Bounded input validation')
    if not isinstance(resources, BoundedResources):
        raise TypeError('explicit BoundedResources required')
    if not isinstance(auxiliary_recipe, BasisRecipe) or distribution != 'df_centre_analytic':
        raise ValueError('explicit AUX recipe and df_centre_analytic distribution required')
    if not isinstance(quadrature, Quadrature) or not isinstance(smoothing, KernelSmoothing):
        raise TypeError('explicit Quadrature and KernelSmoothing required')
    if scf_correction not in ('NONE', 'FIXED_GRAC'):
        raise ValueError('bounded route supports NONE or FIXED_GRAC only')
    if not isinstance(lattice_options, core.FitPointsOptions):
        raise TypeError('explicit FitPointsOptions required')
    if not 1 <= lattice_options.npoints <= 2000:
        raise ValueError('bounded lattice requires 1..2000 points')
    for name, value, upper in (('shell_cutoff', shell_cutoff, float('inf')),
                               ('charge_penalty', charge_penalty, float('inf')),
                               ('anchor_metric_damping', anchor_metric_damping, 1.)):
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or not 0 <= value < upper:
            raise ValueError(f'invalid {name}')
    if type(localization_rank_limit) is not int or not 1 <= localization_rank_limit <= 3:
        raise ValueError('localization_rank_limit must be 1..3')
    sites, bonds = tuple(sites), tuple(tuple(b) for b in bonds)
    if not 1 <= len(sites) <= 64 or any(not isinstance(s, refine.RefinementSite) for s in sites):
        raise TypeError('explicit RefinementSite sequence required')
    if len({s.label for s in sites}) != len(sites):
        raise ValueError('site labels must be unique')
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
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or value < 0:
            raise ValueError(f'invalid {name}')
    declared_variables = None if declared_variables is None else tuple(declared_variables)
    if declared_variables is not None and not 1 <= len(declared_variables) <= 64:
        raise ValueError('bounded refinement requires 1..64 declared variables')
    if any(s.rank_limit > localization_rank_limit for s in sites):
        raise ValueError('refinement rank exceeds localization rank')
    origins = np.asarray([s.origin_bohr for s in sites])
    geometry = np.asarray(wfn.molecule().geometry())
    if (origins.shape != geometry.shape or not np.array_equal(origins, geometry)
            or not np.array_equal(auxiliary_recipe.centres, geometry)):
        raise ValueError('site, AUX and wavefunction centres must agree in order and bohr coordinates')
    if (not isinstance(response_grid, np.ndarray) or response_grid.dtype != np.float64
            or response_grid.ndim != 2 or response_grid.shape[1] != 4
            or not len(response_grid)):
        raise ValueError('finite float64 full response grid (rows,4) required')
    if 3*response_grid.nbytes > resources.max_bytes:
        raise ValueError('full response grid numeric byte resource limit')
    if not np.isfinite(response_grid).all():
        raise ValueError('finite response grid required')
    correction = validate_correction(wfn, scf_correction=scf_correction,
                                     expected_grac_shift=expected_grac_shift, require_canonical=True)
    require_scf_seal(wfn)
    state = native_restricted_state_from_wavefunction(wfn, caller_converged=caller_converged,
                                                      max_bytes=resources.max_bytes)
    grid = np.array(response_grid, copy=True)
    # Reserve caller input/copy overlap, state and MAIN transformations together.
    reserved = 2*grid.nbytes+state.planned_bytes+32*state.nbf**2
    ledger = _Ledger(resources, reserved)
    ledger.admit('state and grid', 0)
    main = adapt_main(wfn, caller_converged=True)
    coefficients = main.transform @ np.asarray(state.orbitals())
    energies = np.asarray(state.energies()).copy()
    auxiliary = auxiliary_recipe.build('MolecularAux')
    p, no, nv = auxiliary.nfunction, state.nocc, state.nvir
    n, q, nf = no*nv, 25*len(sites), len(quadrature.frequencies)
    log.items((('AUX / occupied / virtual / OV', (p, no, nv, n)),
               ('sites / frequencies / grid points', (len(sites), nf, len(grid))),
               ('distribution', distribution), ('correction', repr(correction)),
               ('resource ceilings', repr(resources))))
    retained = response_grid.nbytes+nf*(len(sites)*25*25*8*4+4*64*64*8)
    frequency_plan = 8*(3*n*n+4*n*(p+q)+p*p+q*q+16*n)+8*1024**2
    if frequency_plan+retained > resources.max_bytes:
        raise ValueError('complete frequency plan exceeds shared numeric byte resource limit')
    frequency_work = nf*int(2*n**3+2*n**3/3+8*n*n*(p+q))
    if frequency_work > resources.max_work:
        raise ValueError('complete quadrature exceeds cumulative work resource limit')
    source = dict(state_sha256=_context(wfn), auxiliary_sha256=hashlib.sha256(repr(auxiliary_recipe).encode()).hexdigest(),
                  correction=repr(correction), distribution=distribution,
                  grid_sha256=hashlib.sha256(grid.tobytes()).hexdigest(),
                  kernel_smoothing=repr(smoothing), shell_cutoff=float(shell_cutoff),
                  sites=repr(sites), bonds=bonds, frequencies=quadrature.frequencies,
                  cp_weights=quadrature.cp_weights, localization_rank_limit=localization_rank_limit,
                  weight_type=weight_type, weight_coefficient=weight_coefficient, cutoff=cutoff,
                  declared_variables=None if declared_variables is None else tuple(declared_variables),
                  model=f'plain DF PBE0; fitted-density ALDA Slater/PW92; lambda={charge_penalty}; '
                        f'target/kernel eta=0; anchor eta={anchor_metric_damping}; original H2H1; -B.T C0 B')
    with tempfile.TemporaryDirectory(prefix='psi4-bounded-', dir=scratch_directory) as temporary:
        store = _Store(Path(temporary), ledger)
        log.stage('Native plain-DF factors', (('exact exchange', .25), ('tile columns', 512)))
        ledger.admit('native factor work reservation', 0,
                     8*p*(no+nv)**3+4*p*p*(no+nv)**2+2*p**3)
        ops = native_plain_df_operators(auxiliary, main.basis, coefficients, energies,
            nocc=no, shell_count=len(auxiliary_recipe.shells), exact_exchange=.25,
            tile_columns=512,
            max_bytes=resources.max_bytes-ledger.reserved)
        ledger.admit('native factors', ops.construction_planned_bytes)
        for name, value in (('gaps', ops._gaps), ('oo', ops._oo), ('ov', ops._ov),
                            ('density', ops.plain_density_coefficients)):
            store.save(name, value)
        tiles = (len(ops._dual_ov.tiles), len(ops._dual_vv.tiles))
        for kind in ('ov', 'vv'):
            for i, tile in enumerate(getattr(ops, '_dual_'+kind).tiles):
                store.save(f'dual{kind}{i}', tile)
        del tile, value
        multipole_sites = []
        for site in sites:
            value = core.IsaMultipoleSite()
            value.label, value.origin, value.rank = site.label, list(site.origin_bohr), 4
            multipole_sites.append(value)
        for name, eta in (('anchor', anchor_metric_damping), ('target', 0.)):
            log.stage(name + ' constrained OV fit', (('metric damping', eta), ('charge penalty', charge_penalty)))
            ledger.admit(name+' fit work reservation', 0, 2*p**3+4*p*p*n)
            fit = native_constrained_ov(ops, charge_penalty=charge_penalty,
                offsite_metric_damping=eta, tile_columns=32,
                max_bytes=resources.max_bytes-ledger.reserved-16*p*q-16*n*q)
            ledger.admit(name+' fit', fit.planned_bytes+16*p*q+16*n*q)
            if name == 'anchor':
                moments = np.asarray(df_centre_multipoles(distribution, auxiliary_recipe, multipole_sites, 4).values)
                store.save(name, fit.coefficients @ moments.T)
                del moments
            else:
                store.save(name, fit.coefficients)
            del fit
        del ops, main, coefficients, energies, state
        ledger.reserved = 2*grid.nbytes
        log.stage('Full-grid ALDA kernel', (('smoothing', repr(smoothing)), ('shell cutoff', shell_cutoff)))
        kernel = _kernel(auxiliary, store.load('density'), grid, smoothing, shell_cutoff, ledger)
        store.save('kernel', kernel)
        del kernel, grid
        # Input response_grid remains caller-owned; retained output allowance is
        # charged across the entire frequency sweep rather than node by node.
        ledger.reserved = retained
        log.stage('H1/H2 assembly')
        _assemble(store, (p, no, nv), tiles, .25, .75)
        log.stage('Response stability checks')
        stability = _stability(store, n)
        log.items(stability.items())
        log.stage('Native point cloud and Coulomb potentials',
                  (('points', lattice_options.npoints), ('seed', lattice_options.seed),
                   ('inner / outer radii', (lattice_options.lolim, lattice_options.hilim))))
        ledger.admit('native fit cloud', 8*(3*p*lattice_options.npoints+12*lattice_options.npoints)+8*1024**2)
        cloud = core.FitPoints(wfn.molecule().clone(), lattice_options)
        points = np.column_stack((cloud.x(), cloud.y(), cloud.z()))
        del cloud
        b = np.empty((p, len(points)))
        provider = core.IsaAuxCoulomb(auxiliary)
        for begin in range(0, len(points), 32):
            b[:, begin:begin+32] = np.asarray(provider.point_potentials(
                core.Matrix.from_array(points[begin:begin+32]), max_bytes=resources.max_bytes-ledger.reserved))
        store.save('potentials', b)
        del b, provider
        source['lattice_sha256'] = hashlib.sha256(points.tobytes()).hexdigest()
        local_tensors, results, diagnostics = [], [], []
        frames = [s.frame for s in sites]
        for node, omega in enumerate(quadrature.frequencies, 1):
            tag = f'node {node}/{nf}, omega={omega:.12g} au'
            log.stage('Original H2H1 response: ' + tag, (('RHS', p+q),))
            coefficient, raw, residual = _frequency(store, (p, no, nv), q, omega)
            log.items((('relative response residual', residual),))
            log.stage('Production LW localization: ' + tag,
                      (('rank limit', localization_rank_limit), ('bonds', len(bonds))))
            ns = len(sites)
            # Mirror LW's conservative native workspace reservation, plus live
            # coefficient/raw matrices and Python snapshot/copy overlap.
            local_plan = (3*ns*ns*5000+(2*len(bonds)+ns)*5000+ns*(4608+24)
                          +16*ns*ns*8+4*1000000*48+8*p*p+32*q*q)
            ledger.admit('production localization', local_plan, 8*q**3)
            tensors = raw.reshape(len(sites), 25, len(sites), 25).transpose(0, 2, 1, 3)[None]
            provenance = lw.Provenance('bounded native response', hashlib.sha256(raw.tobytes()).hexdigest(),
                                       'native Psi4 original H2H1 solve', source['model'])
            local = lw.supplied_nonlocal_properties(labels=[s.label for s in sites], origins=origins,
                bonds=bonds, frames=frames, frequencies=[omega], tensors=tensors, input_rank=4,
                provenance=provenance, truncation=lw.TRUNCATE_RANK4, residual_policy='production',
                localization_rank_limit=localization_rank_limit)
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
                declared_variables = model.parameter_labels
                log.items((('cutoff-derived variables from this node', len(declared_variables)),))
            local_tensors.append(local.raw_local)
            diagnostics.append(dict(frequency=omega, response_residual=residual,
                localization_residual=local.frequency_diagnostics[0].residuals.maximum))
            log.items((('maximum localization residual', diagnostics[-1]['localization_residual']),))
            # The fitting phase owns only the AUX coefficient response and its
            # small model; do not carry full nonlocal/LW input snapshots into
            # that phase's independently planned workspace.
            del raw, tensors, local, anchors
            log.stage('Complete-cloud PFIT: ' + tag,
                      (('parameters', model.parameter_count), ('rows per pass', len(points)*(len(points)+1)//2),
                       ('passes', 2), ('weight type', weight_type),
                       ('weight coefficient', weight_coefficient), ('cutoff', cutoff)))
            results.append(_stream_fit(model, points, coefficient, store.load('potentials'), ledger, source))
            log.items((('solver status', str(results[-1].status)),
                       ('maximum parameter change from anchor', max(abs(a-b) for a, b in zip(results[-1].parameters, model.anchors)))))
            del coefficient, model
        log.stage('Casimir-Polder dispersion', (('maximum order', max_order), ('nodes', nf)))
        dispersion = refine.refined_isotropic_dispersion(results, cp_weights=quadrature.cp_weights,
            quadrature_provenance=quadrature.provenance, max_order=max_order)
        log.items((('site pairs', len(dispersion.pairs)), ('peak planned numeric bytes', ledger.peak),
                   ('charged work', ledger.work), ('checkpoint I/O bytes', ledger.io)))
        log.stage_end()
    return BoundedProperties(quadrature.frequencies, tuple(local_tensors), tuple(results), dispersion,
        tuple(diagnostics), dict(maximum_numeric_plan=ledger.peak, charged_work=ledger.work,
        charged_io_bytes=ledger.io, frequency_rhs=nf*(p+q), fit_rows=nf*len(points)*(len(points)+1),
        stages=ledger.stages), dict(source, stability=stability))
