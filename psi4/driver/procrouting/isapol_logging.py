# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Narration and QCVariable reporting for the native atomic-property pipeline.

Reporting only: this module computes no science.  Everything it prints or stores
is read off the owned stage records the pipeline already returns, so enabling it
cannot change a number, a convergence decision or a gate.  It never mutates a
record, never reruns a stage, never fills in a value a stage declined to produce
and never reformats a refusal into a softer one.

Two deliberately separate surfaces:

* :class:`StageLog` writes the human-readable narrative through
  ``core.print_out``.  Verbosity is a declared integer, not a debug switch:

  - 0  silent.  This is the default everywhere, so an expert caller that passes
       no log behaves exactly as it did before this module existed.
  - 1  stage banners carrying every tweakable parameter of that stage, the
       stage-exit summary, and the final property tables.
  - 2  adds the iteration tables and the per-stage numerical diagnostics.
  - 3  adds per-frequency detail (all nodes, not just the static one).

* The ``report_*`` functions set QCVariables on the wavefunction.  Arrays ALWAYS
  go through ``core.Matrix.from_array``, because ``p4util.python_helpers``
  reshapes a bare ``ndarray`` QCVariable by looking at the variable NAME (a key
  ending in ``DIPOLE`` is forced to shape ``(1, 3)``); a labeled table must
  never be re-interpreted by its own name.  Setting QCVariables is safe with
  respect to every seal in this pipeline: neither ``isapol_native._context`` nor
  ``scf_proc.scf_iterator._scf_state_signature`` hashes variables.

Large intermediates are not printable from here.  :meth:`StageLog.table` elides
long tables (and says so) and refuses a row wider than :data:`MAX_ROW_CELLS`, so
raw response tensors, grids, transition-fit coefficients, orbital matrices and
density samples cannot reach the output file by accident.  Those remain
available in full through ``isapol_oeprop.atomic_property_result(wfn)``, which
is where this pipeline has always said large labeled tensors live.

Every quantity printed here is either a field of a stage record or an explicitly
labeled sum over the printed rows.  Where a record marks a number incomparable
-- an incomplete dispersion order, a failed production postcondition, a declared
rank limit -- the label carries that mark too, both in the table and in the
QCVariable name, rather than being dropped or silently promoted.
"""
import dataclasses
import time

import numpy as np

from psi4 import core
from . import isapol_native_partition as _partition

#: Longest table body printed in full; longer tables are elided in the middle
#: (both ends are kept) with an explicit count of the omitted rows.
MAX_TABLE_ROWS = 60
#: A row wider than this is bulk data, not a property table. Refused outright:
#: reaching it means a caller handed a raw intermediate to the formatter.
MAX_ROW_CELLS = 24


def _fmt(value):
    """Render one cell. Wide sequences are summarized, never expanded."""
    if value is None:
        return 'none'
    if isinstance(value, (bool, np.bool_)):
        return 'true' if value else 'false'
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if not np.isfinite(v):
            return repr(v)
        if v == 0.:
            return '0'
        return ('%.6f' % v) if 1.e-4 <= abs(v) < 1.e6 else ('%.6e' % v)
    if isinstance(value, np.ndarray):
        return f'<array{tuple(value.shape)} not printed>'
    if isinstance(value, (tuple, list)):
        if len(value) > MAX_ROW_CELLS:
            return f'<{len(value)} entries not printed>'
        return '(' + ', '.join(_fmt(v) for v in value) + ')'
    return str(value)


class StageLog:
    """Verbosity-gated writer with stage banners, wall clocks and small tables.

    Holds no wavefunction and no scientific record. ``writer`` exists so tests
    can capture the text without touching the Psi4 output file; the production
    writer is ``core.print_out``.
    """

    def __init__(self, verbosity=0, writer=None, indent=2):
        if type(verbosity) is not int or isinstance(verbosity, bool) or verbosity < 0:
            raise ValueError('logging verbosity must be an explicit nonnegative integer')
        if type(indent) is not int or indent < 0:
            raise ValueError('indent must be a nonnegative integer')
        self._level = int(verbosity)
        self._writer = core.print_out if writer is None else writer
        if not callable(self._writer):
            raise TypeError('writer must be callable')
        self._pad = ' ' * int(indent)
        self._open = None
        self._started = None
        self._stages = []

    @property
    def verbosity(self):
        return self._level

    @property
    def stages(self):
        """``(name, seconds)`` for every stage closed so far; timings, not results."""
        return tuple(self._stages)

    def enabled(self, level=1):
        return self._level >= int(level)

    def line(self, text='', level=1, indent=0):
        if self.enabled(level):
            self._writer((self._pad + ' ' * int(indent) + str(text) + '\n') if text != '' else '\n')

    def banner(self, title, level=1):
        if self.enabled(level):
            self._writer('\n' + self._pad + '==> ' + str(title) + ' <==\n\n')

    def items(self, pairs, level=1, indent=4):
        """Aligned ``name  value`` block; the stage-parameter rendering."""
        if not self.enabled(level):
            return
        rendered = [(str(k), _fmt(v)) for k, v in pairs]
        if not rendered:
            return
        width = max(len(k) for k, _ in rendered)
        for k, v in rendered:
            self._writer(self._pad + ' ' * int(indent) + k.ljust(width) + '   ' + v + '\n')

    def table(self, title, headers, rows, level=1, indent=4, note=None):
        """Print a small aligned table, eliding the middle of a long body.

        Truncation is announced with the omitted row count; a row wider than
        :data:`MAX_ROW_CELLS` is refused, because that is a raw intermediate.
        """
        headers = [str(h) for h in headers]
        if len(headers) > MAX_ROW_CELLS:
            raise ValueError('refusing to print a wide intermediate as a property table')
        rows = list(rows)
        for row in rows:
            if len(row) != len(headers):
                raise ValueError('table rows must match the declared header count')
        if not self.enabled(level):
            return
        omitted = 0
        if len(rows) > MAX_TABLE_ROWS:
            head = MAX_TABLE_ROWS // 2
            omitted = len(rows) - MAX_TABLE_ROWS
            rows = rows[:head] + rows[len(rows) - (MAX_TABLE_ROWS - head):]
        body = [[_fmt(c) for c in row] for row in rows]
        widths = [max([len(h)] + [len(r[i]) for r in body]) for i, h in enumerate(headers)]
        pad = self._pad + ' ' * int(indent)
        if title:
            self._writer(pad + str(title) + '\n')
        rule = '  '.join('-' * w for w in widths)
        self._writer(pad + '  '.join(h.rjust(w) for h, w in zip(headers, widths)) + '\n')
        self._writer(pad + rule + '\n')
        for i, row in enumerate(body):
            if omitted and i == MAX_TABLE_ROWS // 2:
                self._writer(pad + f'... {omitted} intermediate rows not printed; the full record is '
                             'available through atomic_property_result(wfn)\n')
            self._writer(pad + '  '.join(c.rjust(w) for c, w in zip(row, widths)) + '\n')
        self._writer(pad + rule + '\n')
        if note:
            self._writer(pad + str(note) + '\n')

    def stage(self, name, parameters=(), level=1):
        """Close any open stage, then announce this one with all its parameters."""
        self.stage_end(level=level)
        self._open, self._started = str(name), time.time()
        self.banner('Stage: ' + str(name), level=level)
        if parameters:
            self.line('Parameters (every tweakable input of this stage):', level=level)
            self.items(parameters, level=level)
            self.line(level=level)

    def stage_end(self, level=1):
        """Close the open stage, recording and printing its wall clock."""
        if self._open is None:
            return
        name, seconds = self._open, time.time() - self._started
        self._stages.append((name, float(seconds)))
        self._open, self._started = None, None
        self.line('Stage complete: %s (%.2f s)' % (name, seconds), level=level)

    def failures(self, records, level=1):
        """Print recorded stage failures verbatim; never summarize them away."""
        records = tuple(records)
        if not records:
            return
        self.table('Recorded stage failures (no stage was retried or relaxed):',
                   ('stage', 'frequency', 'exception', 'message'),
                   [(f.stage, f.frequency, f.exception_type, f.message) for f in records],
                   level=level)


def silent():
    """The default log: accepts every call, writes nothing, records timings."""
    return StageLog(0, writer=lambda text: None)


def dataclass_parameters(obj, *, prefix='', skip=()):
    """Every declared field of a recipe dataclass, in declaration order.

    Enumerating the dataclass rather than a hand-written list is the point: a
    stage banner that claims to list all tweakable parameters cannot silently
    omit one that was added later. Bulk fields are reported by size.
    """
    out = []
    for field in dataclasses.fields(obj):
        value = getattr(obj, field.name)
        if isinstance(value, (tuple, list)) and len(value) > MAX_ROW_CELLS:
            value = f'<{len(value)} entries>'
        if field.name in skip:
            continue
        out.append((prefix + field.name, value))
    return tuple(out)


def _matrix(array):
    """Wrap for ``set_variable``: bypasses name-based ndarray reshaping."""
    return core.Matrix.from_array(np.ascontiguousarray(np.asarray(array, dtype=float)))


def _set(wfn, key, value):
    if wfn is None:
        return
    wfn.set_variable(key, value)


# ---------------------------------------------------------------- request ----

def report_request(log, wfn, *, tasks, options, recipe, correction_options, scf_residual):
    """Announce the request itself: tasks, ambient options, recipe and correction.

    The declared correction is reported from the caller's own request options,
    not by admitting it here: admission belongs to ``validate_correction`` at
    the point the pipeline already performs it, and narration must not move a
    refusal earlier or later than the stage that owns it.
    """
    log.banner('Native atomic properties')
    log.items((('recipe', recipe.name), ('recipe origin', recipe.origin),
               ('partition track', recipe.track), ('Drho profile', recipe.drho_profile),
               ('molecular AUX', recipe.auxiliary.name),
               ('sites', len(recipe.sites)),
               ('site labels', tuple(s.label for s in recipe.sites)),
               ('distributed site rank', recipe.sites[0].rank),
               ('requested tasks', tuple(tasks)))
              + tuple((k, v) for k, v in sorted(dict(correction_options).items()))
              + (('maxabs(FDS-SDF) prerequisite', scf_residual),))
    log.line()
    log.line('Requested options (ambient globals; this is the only stage that reads them):')
    log.items(options)
    _set(wfn, 'ATOMIC PROPERTY SCF COMMUTATOR MAXABS', float(scf_residual))


def report_work_estimate(log, estimate):
    """Print the dimension guard's actual numbers and the limits they were compared to."""
    log.line('Response dimension guard (native_response.cc order; no limit is relaxed here):')
    log.items((('nbf / nmo / nocc', (estimate.nbf, estimate.nmo, estimate.nocc)),
               ('nov', estimate.nov), ('grid rows', estimate.grid_rows),
               ('named algorithm', estimate.algorithm),
               ('nov limit', min(estimate.max_nov, estimate.native_nov_limit)),
               ('direct-JK work / limit', (estimate.ao_work, estimate.ao_work_limit)),
               ('ALDA work / limit', (estimate.alda_work, estimate.alda_work_limit)),
               ('unchecked here', estimate.unchecked),
               ('passes', estimate.passes)))
    log.line()


def report_quadrature(log, quadrature, level=2):
    """Casimir-Polder nodes and weights: the complete authoritative node list."""
    if quadrature is None:
        return
    log.table('Casimir-Polder quadrature (CP weights already include 1/(2*pi)):',
              ('node', 'xi [Eh]', 'CP weight'),
              [(i, f, w) for i, (f, w) in enumerate(zip(quadrature.frequencies, quadrature.cp_weights))],
              level=level, note='provenance: ' + quadrature.provenance.description)


# -------------------------------------------------------------- partition ----

def partition_parameters(recipe):
    """Grid, controller and Drho parameters of the density-partition stage."""
    return (dataclass_parameters(recipe.grid, prefix='grid.')
            + dataclass_parameters(recipe.controller, prefix='controller.')
            + (('drho_profile', recipe.drho_profile),
               ('drho charge penalty', 1000.),
               ('site tail cutoffs [bohr]', tuple(s.tail_cutoff for s in recipe.sites)),
               ('site tails allowed', tuple(s.tail_allowed for s in recipe.sites))))


def report_partition(log, wfn, partition):
    """Stage exit for ISA-A: convergence, Drho-C fit quality and shape charges.

    Only the owned ``NativePartitionResult`` is narrated. An object that merely
    imitates part of that record is declared unnarratable and nothing is
    published from it: publishing the few fields such an object happens to carry
    would put a number under a name whose companions are silently missing, which
    is the mislabel this module exists to avoid. On the owned record every field
    below is required, so a record that loses one still raises here.
    """
    if not isinstance(partition, _partition.NativePartitionResult):
        log.line('Partition record is not an owned NativePartitionResult; not narrated.')
        return
    state, trajectory = partition.trajectory.state, partition.trajectory
    log.items((('converged', bool(state.converged)),
               ('iterations', int(state.iteration)),
               ('termination', trajectory.termination),
               ('max |delta w|', float(state.max_delta)),
               ('active w_eps', float(state.active_w_eps)),
               ('active positive lambda', float(state.active_positive_lambda)),
               ('tails applied', bool(state.apply_tails)),
               ('grid points', int(partition.grid_weights.shape[0])),
               ('Drho-C fitted electrons', float(partition.drho.fitted_electrons)),
               ('Drho-C charge penalty', float(partition.drho.charge_penalty)),
               ('Drho-C relative residual', float(partition.drho.relative_residual)),
               ('Drho-C metric condition', float(partition.drho_metric_condition)),
               ('MAIN orthonormality residual', float(partition.main.orthonormality_residual)),
               ('MAIN global overlap residual', float(partition.main.global_overlap_residual)),
               ('MAIN transformed orthonormality', float(partition.main.transformed_orthonormality_residual)),
               ('MAIN adaptation method', partition.main.method)))
    report_partition_iterations(log, partition)
    report_shape_charges(log, partition)
    _set(wfn, 'ISA ITERATIONS', float(state.iteration))
    _set(wfn, 'ISA CONVERGED', 1. if state.converged else 0.)
    _set(wfn, 'ISA MAX DELTA', float(state.max_delta))
    _set(wfn, 'ISA GRID POINTS', float(partition.grid_weights.shape[0]))
    _set(wfn, 'ISA DRHO FITTED ELECTRONS', float(partition.drho.fitted_electrons))
    _set(wfn, 'ISA DRHO RELATIVE RESIDUAL', float(partition.drho.relative_residual))
    _set(wfn, 'ISA DRHO METRIC CONDITION', float(partition.drho_metric_condition))
    charges = np.asarray(state.saved_shape_charges, dtype=float)
    if charges.shape == (len(partition.recipe.sites),):
        # Per-site populations are a property, not an intermediate: they are
        # published even at verbosity 0, where the printed table is suppressed.
        _set(wfn, 'ISA SHAPE CHARGES', _matrix(charges.reshape(1, -1)))


def report_partition_iterations(log, partition, level=2):
    """One row per ISA-A sweep: the actual per-iteration convergence metrics."""
    history = list(partition.trajectory.history)
    if not history:
        return
    rows = []
    for i, step in enumerate(history):
        deltas = np.asarray(step.deltas, dtype=float)
        charges = np.asarray(step.shape_charges, dtype=float)
        rows.append((i + 1,
                     float(np.max(np.abs(deltas))) if deltas.size else float('nan'),
                     float(np.mean(np.abs(deltas))) if deltas.size else float('nan'),
                     int(sum(bool(c) for c in step.atom_converged)),
                     len(step.atom_converged),
                     float(charges.sum()) if charges.size else float('nan'),
                     float(step.next.active_w_eps),
                     float(step.next.active_positive_lambda),
                     bool(step.next.apply_tails)))
    log.table('ISA-A iterations (W-convergence controller; no DIIS):',
              ('iter', 'max|dW|', 'mean|dW|', 'conv', 'sites', 'sum q_shape',
               'w_eps', 'pos lambda', 'tails'), rows, level=level)


def report_shape_charges(log, partition, level=2):
    """Per-site converged shape charges; the partition's own population numbers."""
    state = partition.trajectory.state
    charges = np.asarray(state.saved_shape_charges, dtype=float)
    labels = [s.label for s in partition.recipe.sites]
    if charges.shape != (len(labels),):
        return
    log.table('ISA-A shape charges (electrons; reference pre-mixing bookkeeping):',
              ('site', 'q_shape'), list(zip(labels, charges.tolist())), level=level,
              note='sum %s over %d sites' % (_fmt(float(charges.sum())), len(labels)))


# --------------------------------------------------------------- response ----

def response_parameters(*, kernel, exact_exchange, local_scale, density_cutoff, max_bytes,
                        max_nov, response_algorithm, response_basis, response_grid,
                        correction, ov_charge_penalty, ov_metric_damping):
    """Every declared knob of the native response context and transition basis."""
    rows = np.asarray(response_grid).shape[0] if response_grid is not None else 0
    items = [('kernel', kernel), ('exact_exchange', exact_exchange),
             ('local_scale', local_scale), ('density_cutoff', density_cutoff),
             ('max_bytes', max_bytes), ('max_nov', max_nov),
             ('response_algorithm', response_algorithm),
             ('response_basis', response_basis),
             ('response grid rows', rows),
             ('SCF correction policy', correction.policy)]
    if response_basis == 'direct_ov':
        items.append(('transition fit', 'none (direct occupied-fast OV integration)'))
    else:
        items += [('ov_charge_penalty (lambda)', ov_charge_penalty),
                  ('ov_metric_damping (eta)', ov_metric_damping)]
    return tuple(items)


def report_context(log, wfn, context, provider):
    log.items((('nocc / nvir / nov', (provider.nocc, provider.nvir, provider.nocc * provider.nvir)),
               ('wavefunction context sha256', context.wavefunction_sha256[:16] + '...'),
               ('response policy sha256', context.policy_sha256[:16] + '...')))
    _set(wfn, 'ATOMIC RESPONSE NOV', float(provider.nocc * provider.nvir))


def report_ov_fit(log, fit, response_basis):
    if response_basis == 'direct_ov' or fit is None:
        log.items((('transition density', 'none formed; analytic OV charge is zero'),))
        return
    log.items((('fit relative backward residual', float(fit.relative_backward_residual)),
               ('fitted AUX functions', int(np.asarray(fit.coefficients).shape[1]))))


def report_response_diagnostics(log, diagnostics, level=2):
    """The distributed-response charge diagnostics; diagnostics only, never operands."""
    log.items(tuple((k, diagnostics[k]) for k in sorted(diagnostics)), level=level)


# --------------------------------------------------------------------- LW ----

def localization_parameters(*, input_rank, truncation, localization_rank_limit,
                            residual_policy, bonds, frames, frequencies):
    return (('input_rank', input_rank), ('truncation', truncation),
            ('localization_rank_limit', localization_rank_limit),
            ('residual_policy', residual_policy),
            ('bond graph', tuple(tuple(b) for b in bonds)),
            ('frames', 'global Cartesian (none declared)' if frames is None else 'declared'),
            ('frequency nodes', len(frequencies)))


def report_localization(log, wfn, local):
    """Stage exit for LW: postcondition, declared limit, residuals and warnings."""
    meta = local.metadata
    log.items((('mode', meta.mode), ('tensor origin', meta.tensor_origin),
               ('residual policy', meta.residual_policy),
               ('residual tolerance', float(meta.residual_tolerance)),
               ('production postcondition passed', bool(meta.production_postcondition_passed)),
               ('localization rank limit', int(meta.localization_rank_limit)),
               ('truncated input maxabs', float(meta.localization_truncated_input_maxabs)),
               ('output ranks', meta.output_ranks),
               ('refinement status', meta.refinement_status),
               ('anisotropic status', meta.anisotropic_status),
               ('coverage', meta.coverage), ('units', meta.units)))
    report_lw_residuals(log, local)
    report_tensor_diagnostics(log, local)
    for text in local.warnings:
        log.line('WARNING: ' + text)
    _set(wfn, 'ATOMIC RESPONSE LOCALIZATION RANK LIMIT', float(meta.localization_rank_limit))
    _set(wfn, 'ATOMIC RESPONSE LW PRODUCTION POSTCONDITION',
         1. if meta.production_postcondition_passed else 0.)
    worst = max((d.residuals.maximum for d in local.frequency_diagnostics), default=0.)
    _set(wfn, 'ATOMIC RESPONSE LW MAX RESIDUAL', float(worst))


def report_lw_residuals(log, local, level=2):
    """Per-node LW residuals, each named component kept separate."""
    fields = tuple(type(local.frequency_diagnostics[0].residuals).__dataclass_fields__) \
        if local.frequency_diagnostics else ()
    if not fields:
        return
    rows = []
    for d in local.frequency_diagnostics:
        rows.append((d.frequency,)
                    + tuple(float(getattr(d.residuals, name)) for name in fields)
                    + (d.transfer_count, d.omitted_transfer_count,
                       bool(d.production_postcondition_passed),
                       bool(d.algorithm_postcondition_passed)))
    log.table('LW localization residuals per node (production gate %s):'
              % _fmt(float(local.metadata.residual_tolerance)),
              ('xi [Eh]',) + fields + ('transfers', 'omitted', 'production', 'algorithm'),
              rows, level=level)


def report_tensor_diagnostics(log, local, level=3):
    """Per-site asymmetry and passivity; no symmetrization or clipping is applied."""
    if not local.tensor_diagnostics:
        return
    log.table('Localized tensor diagnostics (reported, never repaired):',
              ('node', 'site', 'axes', 'max asymmetry', 'min sym eigenvalue'),
              [(d.frequency_index, d.label, d.axes, d.max_asymmetry, d.minimum_symmetric_eigenvalue)
               for d in local.tensor_diagnostics], level=level)


# --------------------------------------------------- atomic polarizability ----

RANK_UNITS = {1: 'bohr^3', 2: 'bohr^5', 3: 'bohr^7'}


def report_atomic_polarizabilities(log, wfn, local):
    """Formatted isotropic atomic polarizabilities plus the site-sum dipole tensor.

    ``atomic_scalars`` is ``trace(alpha_ll)/(2l+1)`` at ranks 1..3 on the site
    axes; ``global_dipoles`` is the rank-1 block on global Cartesian axes. Ranks
    above the model's declared localization limit are identically zero BY
    DECLARATION, so they are labeled rather than printed as physics.
    """
    scalars = local.atomic_scalars.array
    labels, freq = local.labels, local.frequencies
    limit = int(local.metadata.localization_rank_limit)
    ranks = tuple(r for r in (1, 2, 3) if r <= limit)
    log.table('Static atomic isotropic polarizabilities, trace(alpha_ll)/(2l+1):',
              ('site',) + tuple('alpha_%d [%s]' % (r, RANK_UNITS[r]) for r in ranks),
              [(labels[i],) + tuple(float(scalars[0, i, r - 1]) for r in ranks)
               for i in range(len(labels))],
              note=('ranks %d..3 absent by declaration (localization_rank_limit=%d)'
                    % (limit + 1, limit)) if limit < 3 else None)
    if len(freq) > 1:
        log.table('Frequency-dependent atomic isotropic polarizabilities:',
                  ('xi [Eh]', 'site') + tuple('alpha_%d' % r for r in ranks),
                  [(freq[k], labels[i]) + tuple(float(scalars[k, i, r - 1]) for r in ranks)
                   for k in range(len(freq)) for i in range(len(labels))], level=3)
    dipoles = local.global_dipoles.array
    cart = ('xx', 'yy', 'zz', 'xy', 'xz', 'yz')
    index = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
    rows = [(labels[i],) + tuple(float(dipoles[0, i, a, b]) for a, b in index)
            for i in range(len(labels))]
    total = dipoles[0].sum(axis=0)
    rows.append(('site sum',) + tuple(float(total[a, b]) for a, b in index))
    log.table('Static atomic dipole polarizability tensors on global Cartesian axes (bohr^3):',
              ('site',) + cart, rows,
              note='site sum isotropic mean %s; LW molecular-sum residual %s'
                   % (_fmt(float(np.trace(total) / 3.)),
                      _fmt(max((d.residuals.molecular_sum for d in local.frequency_diagnostics),
                               default=0.))))
    if wfn is None:
        return
    _set(wfn, 'ATOMIC POLARIZABILITY FREQUENCIES', _matrix(np.asarray(freq).reshape(1, -1)))
    for r in ranks:
        _set(wfn, f'ATOMIC ISOTROPIC POLARIZABILITIES RANK {r}',
             _matrix(scalars[:, :, r - 1]))
    for i, label in enumerate(labels):
        # Retained name: the static rank-1 isotropic value this pipeline has
        # always published under it.
        _set(wfn, f'ATOM {label} DIPOLE POLARIZABILITY', float(scalars[0, i, 0]))
        for r in ranks:
            _set(wfn, f'ATOM {label} ISOTROPIC POLARIZABILITY RANK {r}',
                 float(scalars[0, i, r - 1]))
        _set(wfn, f'ATOM {label} DIPOLE POLARIZABILITY TENSOR', _matrix(dipoles[0, i]))
    _set(wfn, 'ATOMIC POLARIZABILITY SITE SUM TENSOR', _matrix(total))
    _set(wfn, 'ATOMIC POLARIZABILITY SITE SUM ISOTROPIC', float(np.trace(total) / 3.))


# --------------------------------------------------------------- dispersion ----

def dispersion_parameters(*, max_order, pair_self, partner, quadrature):
    return (('max_order', max_order),
            ('model B', 'this same model (pair_self)' if pair_self
             else ('declared partner' if partner is not None else 'none')),
            ('quadrature nodes', 0 if quadrature is None else len(quadrature.frequencies)),
            ('CP weight convention', 'includes the Jacobian and 1/(2*pi) exactly once'))


def report_dispersion(log, wfn, dispersion):
    """Atomic (same-site) and pairwise isotropic C_n tables, completeness kept visible.

    An order with a missing rank pair is NOT comparable with a complete one; the
    ``complete`` column carries that, the total is withheld unless every
    contributing coefficient is complete, and an incomplete coefficient's
    QCVariable name says so.
    """
    labels_a = dispersion.model_a.labels
    labels_b = dispersion.model_b.labels
    orders = tuple(c.order for c in dispersion.pairs[0].coefficients) if dispersion.pairs else ()
    log.items((('units', dispersion.units), ('origin', dispersion.origin),
               ('anisotropic status', dispersion.anisotropic_status),
               ('site pairs', len(dispersion.pairs)), ('orders', orders)))
    same = [p for p in dispersion.pairs
            if p.site_a == p.site_b and labels_a[p.site_a] == labels_b[p.site_b]]
    if same:
        log.table('Atomic (same-site) isotropic dispersion coefficients:',
                  ('site',) + tuple('C%d' % n for n in orders) + ('complete',),
                  [(labels_a[p.site_a],) + tuple(c.value for c in p.coefficients)
                   + (all(c.unrestricted_complete for c in p.coefficients),) for p in same])
    log.table('Pairwise isotropic dispersion coefficients (ordered A x B pairs):',
              ('A', 'B') + tuple('C%d' % n for n in orders) + ('complete',),
              [(labels_a[p.site_a], labels_b[p.site_b])
               + tuple(c.value for c in p.coefficients)
               + (all(c.unrestricted_complete for c in p.coefficients),) for p in dispersion.pairs])
    missing = {}
    for p in dispersion.pairs:
        for c in p.coefficients:
            if c.missing_rank_pairs:
                missing.setdefault(c.order, set()).update(c.missing_rank_pairs)
    if missing:
        log.table('Rank pairs absent from each order (these orders are not comparable '
                  'with complete ones):', ('order', 'missing (la, lb)'),
                  [(n, tuple(sorted(missing[n]))) for n in sorted(missing)])
    totals = _dispersion_totals(dispersion, orders)
    log.table('Sum over all ordered site pairs:', ('order', 'sum', 'complete'),
              [(n, totals[n][0], totals[n][1]) for n in orders],
              note='an incomplete sum is not comparable with a complete reference and is '
                   'published only under an INCOMPLETE-marked variable name')
    if wfn is None:
        return
    for p in dispersion.pairs:
        a, b = labels_a[p.site_a], labels_b[p.site_b]
        for c in p.coefficients:
            key = f'ATOMIC DISPERSION C{c.order} {a} {b}'
            _set(wfn, key if c.unrestricted_complete else key + ' INCOMPLETE', float(c.value))
    for i, p in enumerate(same):
        label = labels_a[p.site_a]
        for c in p.coefficients:
            key = f'ATOM {label} C{c.order} DISPERSION COEFFICIENT'
            _set(wfn, key if c.unrestricted_complete else key + ' INCOMPLETE', float(c.value))
    for n in orders:
        value, complete = totals[n]
        key = f'ATOMIC DISPERSION C{n} TOTAL'
        _set(wfn, key if complete else key + ' INCOMPLETE', float(value))


def _dispersion_totals(dispersion, orders):
    """Sum each order over the complete ordered pair set the record itself holds."""
    totals = {}
    for n in orders:
        value, complete = 0., True
        for p in dispersion.pairs:
            for c in p.coefficients:
                if c.order == n:
                    value += float(c.value)
                    complete = complete and bool(c.unrestricted_complete)
        totals[n] = (value, complete)
    return totals


# ------------------------------------------------------------------- PFIT ----

def refine_parameters(*, model, points, fields, damping, options, source_id):
    return (('model provenance', model.provenance),
            ('frequency [Eh]', model.frequency_au),
            ('parameters', model.parameter_count),
            ('component cutoff', model.cutoff),
            ('weight_type', model.weight_type),
            ('weight_coefficient', model.weight_coefficient),
            ('nonsymmetric parameters', model.nonsymmetric_parameter_count),
            ('COPY anchor discrepancy', model.copy_anchor_discrepancy),
            ('fit points', 0 if points is None else len(points)),
            ('fields', 'all' if fields is None else tuple(fields)),
            ('damping', damping), ('source_id', source_id),
            ('solver', getattr(options, 'solver', None)),
            ('qr_chunk_rows', getattr(options, 'qr_chunk_rows', None)),
            ('maximum_work_bytes', getattr(options, 'maximum_work_bytes', None)),
            ('rank_relative_tolerance', getattr(options, 'rank_relative_tolerance', None)),
            ('minimum_solver_rcond', getattr(options, 'minimum_solver_rcond', None)),
            ('penalty convention', 'strengths[k]*(z[k]-anchors[k])**2, CamCASP read_penalties'))


PFIT_DIAGNOSTIC_KEYS = ('data_rows', 'augmented_rows', 'numerical_rank', 'free_indices',
                        'batches', 'data_sse', 'data_rms', 'data_max_residual',
                        'matrix_objective', 'lc_objective', 'total_objective',
                        'stationarity_inf', 'backward_residual', 'normal_h_rcond',
                        'qr_r_rcond', 'rank_smallest', 'rank_largest',
                        'penalty_min_eigenvalue', 'penalty_asymmetry',
                        'rank_method', 'psd_policy', 'lapack_info', 'native_verified')


def report_refinement(log, wfn, refinement, *, frequency=None):
    """PFIT status, objective breakdown and the refined parameters by label."""
    diagnostics = refinement.diagnostics
    log.items((('status', str(refinement.status)),
               ('refinement status', refinement.refinement_status),
               ('anchor shift maxabs', float(refinement.anchor_shift_maxabs)),
               ('penalty convention', refinement.penalty_convention))
              + tuple((k, getattr(diagnostics, k)) for k in PFIT_DIAGNOSTIC_KEYS
                      if hasattr(diagnostics, k)))
    result = refinement.result
    labels = tuple(result.parameter_labels)
    values = np.asarray(refinement.parameters, dtype=float).ravel()
    if len(labels) == len(values):
        log.table('Refined parameters:', ('parameter', 'value', 'unit'),
                  list(zip(labels, values.tolist(), tuple(result.parameter_units))))
    if wfn is None:
        return
    tag = '' if frequency is None else ' XI %.8f' % float(frequency)
    _set(wfn, 'ATOMIC REFINEMENT STATUS' + tag,
         1. if str(refinement.status).endswith('Solved') else 0.)
    _set(wfn, 'ATOMIC REFINEMENT DATA RMS' + tag, float(diagnostics.data_rms))
    _set(wfn, 'ATOMIC REFINEMENT TOTAL OBJECTIVE' + tag, float(diagnostics.total_objective))
    if len(labels) == len(values):
        _set(wfn, 'ATOMIC REFINEMENT PARAMETERS' + tag, _matrix(values.reshape(1, -1)))


# --------------------------------------------------- declared AC iterations ----

def ac_parameters(declaration, *, maxiter, energy_threshold, gradient_threshold,
                  diis_subspace, shift_damping):
    return (dataclass_parameters(declaration)
            + (('label', declaration.label()), ('maxiter', maxiter),
               ('energy_threshold', energy_threshold),
               ('gradient_threshold', gradient_threshold),
               ('diis_subspace', diis_subspace), ('shift_damping', shift_damping)))


AC_ITERATION_HEADERS = ('iter', 'energy [Eh]', 'dE', 'max|[F,D]|', 'shift [Eh]',
                        'HOMO', 'LUMO', 'clamps')


class AcIterationLog:
    """In-loop row writer for the one iteration that keeps no history record.

    The declared-AC Kohn-Sham loop reports a final ``AcConvergence`` only; it
    deliberately stores no per-iteration trajectory, and adding one would change
    a record that the response policy hash covers. So the rows are written as
    they happen instead. Buffering nothing and storing nothing, this cannot
    influence the loop's own convergence test.
    """

    def __init__(self, log, level=2):
        self._log, self._level = log, int(level)
        self._open = False

    def row(self, *, iteration, energy, delta, gradient, shift, homo, lumo, clamped):
        if not self._log.enabled(self._level):
            return
        if not self._open:
            self._log.line('Declared asymptotic-correction SCF iterations '
                           '(Tozer-Handy shift; DIIS):', level=self._level)
            self._open = True
        cells = (int(iteration), float(energy),
                 None if delta is None else float(delta), float(gradient),
                 float(shift), float(homo), float(lumo), int(clamped))
        if not hasattr(self, '_widths'):
            self._widths = [max(len(h), 14) for h in AC_ITERATION_HEADERS]
            self._log.line('  '.join(h.rjust(w) for h, w in
                                     zip(AC_ITERATION_HEADERS, self._widths)),
                           level=self._level, indent=4)
        self._log.line('  '.join(_fmt(c).rjust(w) for c, w in zip(cells, self._widths)),
                       level=self._level, indent=4)


def report_ac_convergence(log, wfn, convergence, declaration, *, converged):
    log.items((('converged', bool(converged)),
               ('iterations', int(convergence.iterations)),
               ('delta energy', float(convergence.delta_energy)),
               ('orbital gradient', float(convergence.orbital_gradient)),
               ('energy threshold', float(convergence.energy_threshold)),
               ('gradient threshold', float(convergence.gradient_threshold)),
               ('Tozer-Handy shift', float(convergence.shift)),
               ('shift clamp hits', int(convergence.shift_clamped)),
               ('HOMO', float(convergence.homo)), ('LUMO', float(convergence.lumo)),
               ('gap', float(convergence.lumo - convergence.homo)),
               ('declared IP', float(declaration.ionization_potential)),
               ('energy (not variational)', float(convergence.energy)),
               ('plain SCF reference energy', float(convergence.reference_energy)),
               ('grid points', int(convergence.grid_points))))
    if wfn is None:
        return
    _set(wfn, 'ATOMIC DECLARED AC ITERATIONS', float(convergence.iterations))
    _set(wfn, 'ATOMIC DECLARED AC SHIFT', float(convergence.shift))
    _set(wfn, 'ATOMIC DECLARED AC HOMO', float(convergence.homo))
    _set(wfn, 'ATOMIC DECLARED AC LUMO', float(convergence.lumo))
