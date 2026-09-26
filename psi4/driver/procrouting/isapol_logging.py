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
available in full through ``psi4.atomic_property_result(wfn)``, which
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


def report_quadrature(log, frequencies, cp_weights, provenance=None, level=2):
    """Casimir-Polder nodes and weights: the complete authoritative node list.

    The two sequences are taken directly rather than a ``Quadrature`` object so
    that a stage narrates the grid its own record holds -- the nodes that were
    actually contracted -- instead of a caller-side object that need not be the
    one the record was built from.  A record may carry no provenance; that is
    reported as the absence it is, not omitted.
    """
    if frequencies is None or cp_weights is None:
        return
    log.table('Casimir-Polder quadrature (CP weights already include 1/(2*pi)):',
              ('node', 'xi [Eh]', 'CP weight'),
              [(i, f, w) for i, (f, w) in enumerate(zip(frequencies, cp_weights))],
              level=level,
              note=('provenance: ' + provenance.description) if provenance is not None
              else 'no quadrature provenance is declared on this record')


# -------------------------------------------------------------- partition ----


# --------------------------------------------------------------- response ----


# --------------------------------------------------------------------- LW ----


# --------------------------------------------------- atomic polarizability ----


# --------------------------------------------------------------- dispersion ----

#: ``IsotropicDispersion``/``AnisotropicDispersion`` fields that ARE the result
#: or one of its two source models rather than a knob of the stage.  They are
#: narrated as counts, hashes and tables instead of being rendered into a
#: parameter block, where a whole ``LocalProperties`` or the pair list would be
#: a raw intermediate.  Every other declared field of either record is
#: enumerated from the dataclass, so a field added to them later appears in the
#: stage report without touching this module.
DISPERSION_RECORD_FIELDS = ('model_a', 'model_b', 'pairs', 'frequencies', 'cp_weights',
                            'quadrature_provenance', 'placement_a', 'placement_b',
                            'placed_a', 'placed_b', 'truncated_energy')


def provenance_parameters(provenance, *, prefix):
    """A declared ``Provenance``, or the explicit fact that none was declared."""
    if provenance is None:
        return ((prefix + 'provenance', 'none declared with this record'),)
    return dataclass_parameters(provenance, prefix=prefix)


def dispersion_model_parameters(model, *, prefix):
    """Declared identity of one localized model entering the contraction.

    A C_n is only as declared as the two models it contracts, so each side's
    rank limit, residual policy, postcondition verdict, refinement status and
    input hashes belong in the banner.  The tensors themselves are not printed
    here: the localization stage owns them.
    """
    m = model.metadata
    return ((prefix + 'sites', len(model.labels)),
            (prefix + 'labels', model.labels),
            (prefix + 'frequency nodes', len(model.frequencies)),
            (prefix + 'localization_rank_limit', m.localization_rank_limit),
            (prefix + 'mode', m.mode),
            (prefix + 'tensor_origin', m.tensor_origin),
            (prefix + 'residual_policy', m.residual_policy),
            (prefix + 'residual_tolerance', m.residual_tolerance),
            (prefix + 'production_postcondition_passed', m.production_postcondition_passed),
            (prefix + 'refinement_status', m.refinement_status),
            (prefix + 'wavefunction_status', m.wavefunction_status),
            (prefix + 'native_verified', m.native_verified),
            (prefix + 'source_sha256', model.provenance.source_sha256),
            (prefix + 'canonical_input_array_sha256', m.canonical_input_array_sha256))


def declared_site_ranks(model, declared, *, prefix):
    """The per-site rank declaration exactly as the producer resolves it.

    ``None`` does not mean ``(1, 2, 3)``: it means every rank the model was
    actually localized at, read off that model's own limit.  Which of the two
    happened decides whether an order comes out complete, so the resolved
    tuples are printed rather than the argument as it was passed.
    """
    limit = int(model.metadata.localization_rank_limit)
    if declared is None:
        return ((prefix + 'site_ranks', 'not declared: ranks 1..%d on every site, read off '
                 "this model's own localization rank limit" % limit),
                (prefix + 'resolved site ranks',
                 (tuple(range(1, limit + 1)),) * len(model.labels)))
    return ((prefix + 'site_ranks', 'declared explicitly, per site'),
            (prefix + 'resolved site ranks', tuple(tuple(site) for site in declared)))


def cp_weight_parameters(cp_weights):
    """The quadrature knob as a count and a sum; the node list is tabulated."""
    weights = () if cp_weights is None else tuple(float(w) for w in cp_weights)
    return (('quadrature nodes', len(weights)),
            ('CP weight sum', float(np.sum(weights)) if weights else 0.),
            ('CP weight convention', 'includes the Jacobian and 1/(2*pi) exactly once'),
            ('static node weight', 'zero by construction; a nonzero one is refused'))


def dispersion_parameters(*, model_a, model_b, max_order, cp_weights, quadrature_provenance,
                          site_ranks_a=None, site_ranks_b=None):
    """Every tweakable input of the isotropic C_n contraction.

    The knobs are ``max_order``, the CP weights and the two per-site rank
    declarations; the rest of the block is the declared identity of the two
    models and of the quadrature, without which a C_n is not interpretable.
    """
    return ((('max_order', max_order),
             ('admitted max_order values', '6, 8, 10, 12; odd orders are refused here'),
             ('model B', 'this same model (pair_self)' if model_b is model_a
              else 'a separately declared partner model'))
            + cp_weight_parameters(cp_weights)
            + declared_site_ranks(model_a, site_ranks_a, prefix='A.')
            + declared_site_ranks(model_b, site_ranks_b, prefix='B.')
            + dispersion_model_parameters(model_a, prefix='A.')
            + dispersion_model_parameters(model_b, prefix='B.')
            + provenance_parameters(quadrature_provenance, prefix='quadrature.'))


def _dispersion_tables(log, dispersion, *, labels_a, labels_b, frequencies, orders):
    """The tables every isotropic C_n record shares, refined or not.

    Both producers contract the same owned C++ kernel against the same
    quadrature and differ only in where their per-site scalars came from, so the
    node list, the two coefficient tables and the completeness evidence are
    written once, here.  What must NOT be shared is the variable *name*: an
    unrefined and a refined C_n are numbers about two different models, and one
    name for both would let a later reader mistake one for the other.  Returns
    the same-site pair list and the per-order totals, which the publishing half
    needs and would otherwise recompute.
    """
    report_quadrature(log, frequencies, dispersion.cp_weights,
                      dispersion.quadrature_provenance)
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
    report_rank_pair_inventory(log, dispersion)
    totals = _dispersion_totals(dispersion, orders)
    log.table('Sum over all ordered site pairs:', ('order', 'sum', 'complete'),
              [(n, totals[n][0], totals[n][1]) for n in orders],
              note='an incomplete sum is not comparable with a complete reference and is '
                   'published only under an INCOMPLETE-marked variable name')
    return same, totals


def _publish_dispersion(wfn, dispersion, *, labels_a, labels_b, frequencies, orders,
                        same, totals, stem, atom_stem):
    """Machine-readable C_n under the naming stem its producer owns.

    ``stem`` is a property of the producing model, not a formatting choice.  The
    refined and the unrefined coefficient of the same order on the same site
    pair are different numbers about different models, so they are published
    under different names and never overwrite one another.
    """
    if wfn is None:
        return
    for p in dispersion.pairs:
        a, b = labels_a[p.site_a], labels_b[p.site_b]
        for c in p.coefficients:
            key = f'{stem} C{c.order} {a} {b}'
            _set(wfn, key if c.unrestricted_complete else key + ' INCOMPLETE', float(c.value))
    for p in same:
        label = labels_a[p.site_a]
        for c in p.coefficients:
            key = f'ATOM {label} C{c.order} {atom_stem}'
            _set(wfn, key if c.unrestricted_complete else key + ' INCOMPLETE', float(c.value))
    for n in orders:
        value, complete = totals[n]
        key = f'{stem} C{n} TOTAL'
        _set(wfn, key if complete else key + ' INCOMPLETE', float(value))
    _set(wfn, f'{stem} SITE PAIRS', float(len(dispersion.pairs)))
    _set(wfn, f'{stem} MAX ORDER', float(max(orders)) if orders else 0.)
    _set(wfn, f'{stem} QUADRATURE NODES', float(len(dispersion.cp_weights)))
    if dispersion.cp_weights:
        _set(wfn, f'{stem} QUADRATURE FREQUENCIES',
             _matrix(np.asarray(frequencies, dtype=float).reshape(1, -1)))
        _set(wfn, f'{stem} CP WEIGHTS',
             _matrix(np.asarray(dispersion.cp_weights, dtype=float).reshape(1, -1)))


def report_dispersion(log, wfn, dispersion):
    """Atomic (same-site) and pairwise isotropic C_n tables, completeness kept visible.

    An order with a missing rank pair is NOT comparable with a complete one; the
    ``complete`` column carries that, the total is withheld unless every
    contributing coefficient is complete, and an incomplete coefficient's
    QCVariable name says so.
    """
    labels_a, labels_b = dispersion.model_a.labels, dispersion.model_b.labels
    frequencies = dispersion.model_a.frequencies
    orders = tuple(c.order for c in dispersion.pairs[0].coefficients) if dispersion.pairs else ()
    log.items(dataclass_parameters(dispersion, skip=DISPERSION_RECORD_FIELDS)
              + (('site pairs', len(dispersion.pairs)), ('orders', orders),
                 ('quadrature nodes', len(dispersion.cp_weights))))
    same, totals = _dispersion_tables(log, dispersion, labels_a=labels_a, labels_b=labels_b,
                                      frequencies=frequencies, orders=orders)
    _publish_dispersion(wfn, dispersion, labels_a=labels_a, labels_b=labels_b,
                        frequencies=frequencies, orders=orders, same=same, totals=totals,
                        stem='ATOMIC DISPERSION', atom_stem='DISPERSION COEFFICIENT')


def report_rank_pair_inventory(log, dispersion, level=2):
    """Which ``(la, lb)`` rank pairs each order actually contracted, and which are absent.

    The union over the ordered site pairs, so a rank limited on one site alone
    still shows up.  This is the evidence behind the ``complete`` column: an
    order missing a rank pair is a different sum from the complete one, not a
    noisier estimate of it.
    """
    included, missing = {}, {}
    for p in dispersion.pairs:
        for c in p.coefficients:
            included.setdefault(c.order, set()).update(c.included_rank_pairs)
            missing.setdefault(c.order, set()).update(c.missing_rank_pairs)
    orders = sorted(set(included) | set(missing))
    if not orders:
        return
    log.table('Rank pairs entering each order (union over the ordered site pairs):',
              ('order', 'included (la, lb)', 'missing (la, lb)'),
              [(n, tuple(sorted(included.get(n, ()))), tuple(sorted(missing.get(n, ()))))
               for n in orders], level=level,
              note='a rank pair is absent because a site was localized below that rank, '
                   'not because its contribution was found small')


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


#: Fields of a refined C_n record that are reported as counts, resolved tuples
#: or tables rather than in the parameter block: the pair list and the two grids
#: are bulk, and the site inventory has its own row.  Every other declared field
#: is enumerated from the dataclass, so one added later appears in the stage
#: report without touching this module.
REFINED_DISPERSION_RECORD_FIELDS = ('pairs', 'frequencies', 'cp_weights',
                                    'quadrature_provenance', 'labels', 'origins_bohr',
                                    'site_ranks')


def refined_dispersion_parameters(*, refinements, max_order, cp_weights,
                                  quadrature_provenance, site_ranks, resolved_site_ranks,
                                  anchor_sha256):
    """Every tweakable input of the refined isotropic C_n contraction.

    The knobs are ``max_order``, the CP weights and the per-site rank
    declaration.  The rest of the block is the declared identity of the
    refinement the scalars were read off -- its penalty scheme, its variable
    count, its anchor hash and its solver verdict at every node -- because a
    refined C_n is a number about that refinement and not about the
    localization the unrefined one comes from.  A non-``Solved`` node still
    returns numbers, so its status is named here rather than inferred from the
    coefficients.
    """
    first = refinements[0].model
    return ((('max_order', max_order),
             ('admitted max_order values', '6, 8, 10, 12; odd orders are refused here'),
             ('model B', 'this same refined model (pair_self)'),
             ('scalar origin', 'PFIT-refined local tensors, trace(alpha_ll)/(2l+1); rank 0 '
                               'is a refinement variable (charge flow), not a dispersion rank'),
             ('refinement nodes', len(refinements)),
             ('sites', len(first.sites)),
             ('labels', tuple(s.label for s in first.sites)),
             ('site types', first.site_types),
             ('declared rank limits', tuple(s.rank_limit for s in first.sites)),
             ('site_ranks', 'not declared: ranks 1..rank_limit on every site, read off '
                            "each site's own declared refinement rank limit"
                            if site_ranks is None else 'declared explicitly, per site'),
             ('resolved site ranks', tuple(tuple(s) for s in resolved_site_ranks)),
             ('weight_type', first.weight_type),
             ('weight_coefficient', first.weight_coefficient),
             ('cutoff', first.cutoff),
             ('variables per node', first.parameter_count),
             ('anchor sha256 over all nodes', anchor_sha256),
             ('solver status per node',
              tuple(str(r.status).rsplit('.', 1)[-1] for r in refinements)),
             ('anchor shift maxabs', max(float(r.anchor_shift_maxabs) for r in refinements)),
             ('refinement provenance', first.provenance))
            + cp_weight_parameters(cp_weights)
            + provenance_parameters(quadrature_provenance, prefix='quadrature.'))


def report_refined_dispersion(log, wfn, dispersion):
    """Refined isotropic C_n, under names that cannot collide with the unrefined ones.

    Same kernel, same quadrature, different model: the per-site scalars here
    come from the PFIT refinement rather than from the localization, so every
    variable name carries ``REFINED``.  Comparing the two is comparing two
    models, which is exactly why they are not permitted to share a name.
    """
    labels = tuple(dispersion.labels)
    orders = tuple(c.order for c in dispersion.pairs[0].coefficients) if dispersion.pairs else ()
    log.items(dataclass_parameters(dispersion, skip=REFINED_DISPERSION_RECORD_FIELDS)
              + (('sites', len(labels)), ('labels', labels),
                 ('resolved site ranks', tuple(tuple(s) for s in dispersion.site_ranks)),
                 ('site pairs', len(dispersion.pairs)), ('orders', orders),
                 ('quadrature nodes', len(dispersion.cp_weights))))
    same, totals = _dispersion_tables(log, dispersion, labels_a=labels, labels_b=labels,
                                      frequencies=dispersion.frequencies, orders=orders)
    _publish_dispersion(wfn, dispersion, labels_a=labels, labels_b=labels,
                        frequencies=dispersion.frequencies, orders=orders, same=same,
                        totals=totals, stem='ATOMIC REFINED DISPERSION',
                        atom_stem='REFINED DISPERSION COEFFICIENT')
    _set(wfn, 'ATOMIC REFINED DISPERSION ANCHOR SHIFT MAXABS',
         float(dispersion.anchor_shift_maxabs))


# --------------------------------------------- oriented (anisotropic) C_n ----

def placement_parameters(placement, *, prefix):
    """One whole-model placement, by its declared hashes rather than its matrix.

    A placement is an explicit input even when it is the identity, so it is
    enumerated either way; the rotation is identified by hash because printing
    a 3x3 per model in the banner is noise, and the hash is what the core
    provenance string carries.
    """
    rotation = placement.rotation.array
    return ((prefix + 'translation [bohr]', tuple(map(float, placement.translation.array))),
            (prefix + 'rotation', 'explicit identity' if np.array_equal(rotation, np.eye(3))
             else 'explicit proper rotation'),
            (prefix + 'rotation trace', float(np.trace(rotation))),
            (prefix + 'rotation_sha256', placement.rotation.canonical_array_sha256),
            (prefix + 'translation_sha256', placement.translation.canonical_array_sha256))


def anisotropic_dispersion_parameters(*, model_a, model_b, placement_a, placement_b,
                                      max_order, cp_weights, quadrature_provenance):
    """Every tweakable input of the orientation-resolved C_n contraction.

    Ranks are not declarable at this adapter -- it contracts ranks 1..3 of
    ``raw_global`` on every site -- and odd orders are admitted here, unlike the
    isotropic stage, so both facts are stated rather than left to be inferred
    from an absent knob.  The reciprocity requirement is a parameter of the
    stage in the sense that matters: it is exact, and there is no policy that
    loosens it.
    """
    return ((('max_order', max_order),
             ('admitted max_order range', '6..12 inclusive, odd orders included'),
             ('site ranks', 'not declarable: ranks 1..3 of raw_global on every site'),
             ('reciprocity policy', 'exact equality required; no repair, no tolerance, '
                                    'no isotropic fallback'),
             ('model B', 'this same model' if model_b is model_a
              else 'a separately declared partner model'))
            + cp_weight_parameters(cp_weights)
            + placement_parameters(placement_a, prefix='A.placement.')
            + placement_parameters(placement_b, prefix='B.placement.')
            + dispersion_model_parameters(model_a, prefix='A.')
            + dispersion_model_parameters(model_b, prefix='B.')
            + provenance_parameters(quadrature_provenance, prefix='quadrature.'))


def report_anisotropic_dispersion(log, wfn, dispersion):
    """Stage exit for the oriented C_n adapter: geometry, C_n, energies, totals.

    These are orientation-resolved *scalars*, not recoupled ``C_n(t, u, J)``
    components; the record says so and the variable names carry the distinction,
    so an oriented scalar can never be read back as an isotropic C_n.  Each
    coefficient carries two completeness flags -- complete within the declared
    rank-3 model, and complete against the unrestricted rank sum -- and both are
    printed.  The ``INCOMPLETE`` name mark follows ``unrestricted_complete``,
    the same convention the isotropic stage uses, so the two publications mean
    the same thing by the same word.
    """
    orders = tuple(c.order for c in dispersion.pairs[0].coefficients) if dispersion.pairs else ()
    log.items(dataclass_parameters(dispersion, skip=DISPERSION_RECORD_FIELDS)
              + (('site pairs', len(dispersion.pairs)), ('orders', orders),
                 ('quadrature nodes', len(dispersion.cp_weights)),
                 ('truncated interaction energy [Eh]', float(dispersion.truncated_energy)),
                 ('energy convention', 'sum of -C_n/R^n through max_order; no damping, no '
                                       'retardation, and no positivity guarantee')))
    report_quadrature(log, dispersion.frequencies, dispersion.cp_weights,
                      dispersion.quadrature_provenance)
    log.table('Placed site-pair geometry (R = B - A; direction is the unit vector along it):',
              ('A', 'B', 'R [bohr]', 'ex', 'ey', 'ez', 'pair energy [Eh]'),
              [(p.label_a, p.label_b, p.distance) + tuple(p.direction) + (p.truncated_energy,)
               for p in dispersion.pairs])
    if orders:
        log.table('Orientation-resolved dispersion coefficients (ordered A x B pairs):',
                  ('A', 'B') + tuple('C%d' % n for n in orders)
                  + ('declared complete', 'unrestricted complete'),
                  [(p.label_a, p.label_b) + tuple(c.value for c in p.coefficients)
                   + (all(c.declared_model_complete for c in p.coefficients),
                      all(c.unrestricted_complete for c in p.coefficients))
                   for p in dispersion.pairs])
        log.table('Orientation-resolved -C_n/R^n contributions [Eh]:',
                  ('A', 'B') + tuple('E%d' % n for n in orders) + ('pair total',),
                  [(p.label_a, p.label_b) + tuple(c.energy for c in p.coefficients)
                   + (p.truncated_energy,) for p in dispersion.pairs], level=2)
    report_rank_quadruple_inventory(log, dispersion)
    totals = _anisotropic_totals(dispersion, orders)
    log.table('Sum over all ordered site pairs:',
              ('order', 'sum C_n', 'sum -C_n/R^n [Eh]', 'declared complete',
               'unrestricted complete'),
              [(n,) + totals[n] for n in orders],
              note='an incomplete sum is not comparable with a complete reference and is '
                   'published only under an INCOMPLETE-marked variable name')
    if wfn is None:
        return
    for p in dispersion.pairs:
        for c in p.coefficients:
            mark = '' if c.unrestricted_complete else ' INCOMPLETE'
            _set(wfn, f'ATOMIC ANISOTROPIC DISPERSION C{c.order} '
                      f'{p.label_a} {p.label_b}' + mark, float(c.value))
            _set(wfn, f'ATOMIC ANISOTROPIC DISPERSION C{c.order} ENERGY '
                      f'{p.label_a} {p.label_b}' + mark, float(c.energy))
        _set(wfn, f'ATOMIC ANISOTROPIC DISPERSION PAIR ENERGY '
                  f'{p.label_a} {p.label_b}', float(p.truncated_energy))
    for n in orders:
        value, energy, _declared, unrestricted = totals[n]
        mark = '' if unrestricted else ' INCOMPLETE'
        _set(wfn, f'ATOMIC ANISOTROPIC DISPERSION C{n} TOTAL' + mark, float(value))
        _set(wfn, f'ATOMIC ANISOTROPIC DISPERSION C{n} TOTAL ENERGY' + mark, float(energy))
    _set(wfn, 'ATOMIC ANISOTROPIC DISPERSION TRUNCATED ENERGY',
         float(dispersion.truncated_energy))
    _set(wfn, 'ATOMIC ANISOTROPIC DISPERSION MAX ORDER', float(dispersion.max_order))
    _set(wfn, 'ATOMIC ANISOTROPIC DISPERSION SITE PAIRS', float(len(dispersion.pairs)))
    _set(wfn, 'ATOMIC ANISOTROPIC DISPERSION QUADRATURE NODES',
         float(len(dispersion.cp_weights)))
    if dispersion.cp_weights:
        _set(wfn, 'ATOMIC ANISOTROPIC DISPERSION QUADRATURE FREQUENCIES',
             _matrix(np.asarray(dispersion.frequencies, dtype=float).reshape(1, -1)))
        _set(wfn, 'ATOMIC ANISOTROPIC DISPERSION CP WEIGHTS',
             _matrix(np.asarray(dispersion.cp_weights, dtype=float).reshape(1, -1)))
    if dispersion.pairs:
        _set(wfn, 'ATOMIC ANISOTROPIC DISPERSION PAIR DISTANCES',
             _matrix(np.asarray([p.distance for p in dispersion.pairs],
                                dtype=float).reshape(1, -1)))


def report_rank_quadruple_inventory(log, dispersion, level=2):
    """How many ordered ``(la, la', lb, lb')`` quadruples each order contracted.

    The quadruple lists are bulk rather than a property -- the unrestricted set
    reaches theoretical ranks 5..7, so a single order can name hundreds -- and
    they are reported as counts.  What matters for comparability is already in
    the two completeness columns beside them.
    """
    included, missing = {}, {}
    for p in dispersion.pairs:
        for c in p.coefficients:
            included.setdefault(c.order, set()).update(c.included_rank_quadruples)
            missing.setdefault(c.order, set()).update(c.missing_rank_quadruples)
    orders = sorted(set(included) | set(missing))
    if not orders:
        return
    log.table('Rank quadruples entering each order (union over the ordered site pairs):',
              ('order', 'included', 'missing'),
              [(n, len(included.get(n, ())), len(missing.get(n, ()))) for n in orders],
              level=level,
              note='the missing set counts unrestricted ranks, including the theoretical '
                   'ranks 5..7 no rank-3 model can carry; an explicit zero block is included, '
                   'not missing')


def _anisotropic_totals(dispersion, orders):
    """Per-order sums of C_n and of -C_n/R^n, with both completeness flags kept."""
    totals = {}
    for n in orders:
        value, energy, declared, unrestricted = 0., 0., True, True
        for p in dispersion.pairs:
            for c in p.coefficients:
                if c.order == n:
                    value += float(c.value)
                    energy += float(c.energy)
                    declared = declared and bool(c.declared_model_complete)
                    unrestricted = unrestricted and bool(c.unrestricted_complete)
        totals[n] = (value, energy, declared, unrestricted)
    return totals


# ------------------------------------------- refinement lattice and targets ----


# ------------------------------------------------------------------- PFIT ----


# --------------------------------------------------- declared AC iterations ----

def ac_parameters(declaration, *, maxiter, energy_threshold, gradient_threshold,
                  diis_subspace, shift_damping, max_energy_threshold=None,
                  max_gradient_threshold=None):
    """The declared AC model plus the iteration controls, against their limits.

    Every field of the declaration is part of the model's identity, so the block
    is enumerated from its dataclass rather than written out.  The thresholds
    are printed beside the loosest values admission accepts, so that a reader
    can see the declared ones were not relaxed to reach convergence.
    """
    return (dataclass_parameters(declaration)
            + (('label', declaration.label()), ('maxiter', maxiter),
               ('energy_threshold', energy_threshold),
               ('gradient_threshold', gradient_threshold),
               ('loosest admitted energy_threshold', max_energy_threshold),
               ('loosest admitted gradient_threshold', max_gradient_threshold),
               ('diis_subspace', diis_subspace), ('shift_damping', shift_damping)))


def report_ac_splice(log, *, exact_exchange, fermi_amaldi_scale, electrons, occupied,
                     basis_functions, origin_bohr, bragg_radii, grid_points, grid_blocks,
                     active_blocks, active_points, level=1):
    """Where the declared correction actually acts, in the stage's own numbers.

    Each of these is read off the constructed driver.  The grid is the SCF's own
    exchange-correlation grid, unchanged and not rebuilt, and the active counts
    come from the exact ``f > 0`` screen rather than from a tolerance.
    """
    log.line('Splice geometry (the SCF exchange-correlation grid, unchanged):', level=level)
    log.items((('exact exchange a_x', exact_exchange),
               ('Fermi-Amaldi coefficient c_FA/N', fermi_amaldi_scale),
               ('electrons N', electrons), ('occupied orbitals', occupied),
               ('basis functions', basis_functions),
               ('multipole origin [bohr]', tuple(origin_bohr)),
               ('Bragg-Slater radii [bohr]', tuple(bragg_radii)),
               ('grid points', grid_points), ('grid blocks', grid_blocks),
               ('blocks with f > 0', active_blocks), ('points with f > 0', active_points),
               ('fraction of grid corrected',
                0. if not grid_points else active_points / float(grid_points))), level=level)
    log.line(level=level)


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


#: Virtual orbitals printed above the LUMO.  The rest of the virtual spectrum is
#: bulk; the full vector is published as one array instead.
AC_SPECTRUM_VIRTUALS = 10


def report_ac_spectrum(log, wfn, energies, nocc, *, mark='', level=1):
    """Orbital energies of the corrected potential: the occupied set and the gap.

    The eigenvalues are what an asymptotic correction is for, so they are this
    stage's property rather than an intermediate.  Only the low-lying virtuals
    are printed; the full vector is published once as a machine-readable array.
    """
    energies = np.asarray(energies, dtype=float).ravel()
    nocc = int(nocc)
    if nocc < 1 or energies.size <= nocc:
        raise ValueError('orbital energies must cover the occupied set and one virtual')
    stop = min(energies.size, nocc + AC_SPECTRUM_VIRTUALS)
    log.table('Corrected orbital energies (HOMO %s, LUMO %s, gap %s Eh):'
              % (_fmt(float(energies[nocc - 1])), _fmt(float(energies[nocc])),
                 _fmt(float(energies[nocc] - energies[nocc - 1]))),
              ('orbital', 'occupation', 'energy [Eh]'),
              [(i + 1, 2. if i < nocc else 0., float(energies[i])) for i in range(stop)],
              level=level,
              note=('%d higher virtual orbitals not printed' % (energies.size - stop))
              if energies.size > stop else None)
    _set(wfn, 'ATOMIC DECLARED AC ORBITAL ENERGIES' + mark, _matrix(energies.reshape(1, -1)))


def report_ac_convergence(log, wfn, convergence, declaration, *, converged):
    """Stage exit for the declared AC: what the iteration actually achieved.

    A refused run is narrated too -- the producer reports before it refuses --
    so an unconverged record's variables carry an ``UNCONVERGED`` mark rather
    than the names a converged run publishes.  Nothing here softens a refusal:
    this reports what the loop did and decides no admission.
    """
    log.items((('declaration', declaration.label()),
               ('converged', bool(converged)),
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
               ('I + eps_HOMO', float(declaration.ionization_potential + convergence.homo)),
               ('energy (not variational)', float(convergence.energy)),
               ('plain SCF reference energy', float(convergence.reference_energy)),
               ('energy above plain SCF',
                float(convergence.energy - convergence.reference_energy)),
               ('grid points', int(convergence.grid_points))))
    if wfn is None:
        return
    mark = '' if converged else ' UNCONVERGED'
    _set(wfn, 'ATOMIC DECLARED AC CONVERGED', 1. if converged else 0.)
    for key, value in (('ITERATIONS', float(convergence.iterations)),
                       ('SHIFT', float(convergence.shift)),
                       ('SHIFT CLAMP HITS', float(convergence.shift_clamped)),
                       ('HOMO', float(convergence.homo)),
                       ('LUMO', float(convergence.lumo)),
                       ('GAP', float(convergence.lumo - convergence.homo)),
                       ('DELTA ENERGY', float(convergence.delta_energy)),
                       ('ORBITAL GRADIENT', float(convergence.orbital_gradient)),
                       ('DECLARED IP', float(declaration.ionization_potential)),
                       ('ENERGY NOT VARIATIONAL', float(convergence.energy)),
                       ('REFERENCE SCF ENERGY', float(convergence.reference_energy)),
                       ('GRID POINTS', float(convergence.grid_points))):
        _set(wfn, 'ATOMIC DECLARED AC ' + key + mark, value)


def ac_application_parameters(record):
    """What applying a declared-AC record replaces, and what that costs.

    The application has no tweakable input of its own: its parameters are the
    identity of the record being applied, and the two consequences the module
    docstring is explicit about -- a non-variational energy and an invalidated
    SCF seal -- are stated here rather than left for a reader to infer.
    """
    return (('declaration', record.declaration.label()),
            ('basis functions', int(np.asarray(record.orbitals).shape[0])),
            ('occupied orbitals', int(record.nocc)),
            ('Tozer-Handy shift', float(record.convergence.shift)),
            ('iterations', int(record.convergence.iterations)),
            ('replaced state', 'Ca, Cb, Da, Db, Fa, Fb, epsilon_a, epsilon_b, energy'),
            ('energy replaced by', 'the plain functional at the corrected density, '
                                   'which is not a variational minimum'),
            ('SCF seal', 'verified, then deliberately invalidated by this mutation'))
