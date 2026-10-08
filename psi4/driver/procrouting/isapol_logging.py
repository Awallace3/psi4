# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Stage-record reporting; never reruns or mutates a scientific stage.

StageLog verbosity: 0 silent, 1 banners/results, 2 diagnostics, 3 node detail.
Long tables are elided and wide rows refused. Incomplete dispersion orders
remain marked in both tables and optional QCVariables.

QCVariable arrays use core.Matrix to avoid name-driven ndarray reshaping.
QCVariables are not part of the wavefunction's SCF/state seals.
"""
import dataclasses
import time

import numpy as np

from psi4 import core

#: Longest table body printed in full; longer tables are elided in the middle
#: (both ends are kept) with an explicit count of the omitted rows.
MAX_TABLE_ROWS = 60
#: A sequence longer than this is bulk data and is summarized, never printed.
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
    """Verbosity-gated writer with stage banners, wall clocks and aligned items.

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
                             'in the returned result\n')
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

    def stage_failed(self, error, level=1):
        """Close the open stage as FAILED; it is printed, never added to :attr:`stages`."""
        if self._open is None:
            return
        name, seconds = self._open, time.time() - self._started
        self._open, self._started = None, None
        self.line('Stage FAILED: %s (%.2f s): %s: %s' % (name, seconds, type(error).__name__, error),
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


def check_publication_labels(labels):
    """Refuse site labels that cannot name distinct QCVariables.

    QCVariable keys are upper-cased and space-separated, so labels whose
    upper-cased forms collide, that contain whitespace, or that spell the
    reserved key words ``TOTAL`` or ``INCOMPLETE`` would alias other keys.
    Only publication is restricted; the model keeps its labels.
    """
    for label in labels:
        if any(ch.isspace() for ch in label):
            raise ValueError(f'site label {label!r} contains whitespace and cannot name a QCVariable')
        if label.upper() in ('TOTAL', 'INCOMPLETE'):
            raise ValueError(f'site label {label!r} is a reserved QCVariable key word')
    folded = [label.upper() for label in labels]
    if len(set(folded)) != len(folded):
        raise ValueError(f'site labels {tuple(labels)!r} collide when upper-cased as QCVariable names')


# ---------------------------------------------------------------- request ----


def report_quadrature(log, frequencies, cp_weights, provenance=None, level=2):
    """Report the contracted nodes/weights, including absent provenance."""
    if frequencies is None or cp_weights is None:
        return
    log.table('Casimir-Polder quadrature (CP weights already include 1/(2*pi)):',
              ('node', 'xi [Eh]', 'CP weight'),
              [(i, f, w) for i, (f, w) in enumerate(zip(frequencies, cp_weights))],
              level=level,
              note=('provenance: ' + provenance.description) if provenance is not None
              else 'no quadrature provenance is declared on this record')


def provenance_parameters(provenance, *, prefix):
    """A declared ``Provenance``, or the explicit fact that none was declared."""
    if provenance is None:
        return ((prefix + 'provenance', 'none declared with this record'),)
    return dataclass_parameters(provenance, prefix=prefix)


def cp_weight_parameters(cp_weights):
    """The quadrature knob as a count and a sum; the node list is tabulated."""
    weights = () if cp_weights is None else tuple(float(w) for w in cp_weights)
    return (('quadrature nodes', len(weights)),
            ('CP weight sum', float(np.sum(weights)) if weights else 0.),
            ('CP weight convention', 'includes the Jacobian and 1/(2*pi) exactly once'),
            ('static node weight', 'zero by construction; a nonzero one is refused'))


def _dispersion_tables(log, dispersion, *, labels_a, labels_b, frequencies, orders):
    """Report isotropic C_n and completeness; return same-site pairs and totals."""
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


def _report_owned(key, stem, atom_stem):
    """Whether ``key`` is a report-owned name.

    That is ``<stem> ...`` or ``ATOM <label> C<n> <atom_stem>[ INCOMPLETE]``.
    """
    key = key.upper()
    if key.startswith(stem.upper() + ' '):
        return True
    parts = key.split(' ')
    tail = ' '.join(parts[3:])
    return (len(parts) > 3 and parts[0] == 'ATOM' and parts[2][:1] == 'C' and parts[2][1:].isdigit()
            and tail in (atom_stem.upper(), atom_stem.upper() + ' INCOMPLETE'))


def _publish_dispersion(wfn, dispersion, *, labels_a, labels_b, frequencies, orders,
                        same, totals, stem, atom_stem, extra=()):
    """Optional QCVariables, with incomplete orders explicitly marked.

    The report owns every ``<stem> ...`` and ``ATOM <label> C<n> <atom_stem>``
    name: they describe the latest successful publication only.  The full set
    is planned first; owned names outside it (omitted orders, former sites,
    the opposite complete/INCOMPLETE variant) are deleted, then the plan is
    written.  Every other variable is left alone.
    """
    if wfn is None:
        return
    check_publication_labels(labels_a)
    check_publication_labels(labels_b)

    def marked(key, complete):
        return key if complete else key + ' INCOMPLETE'

    plan = []
    for p in dispersion.pairs:
        a, b = labels_a[p.site_a], labels_b[p.site_b]
        for c in p.coefficients:
            plan.append((marked(f'{stem} C{c.order} {a} {b}', c.unrestricted_complete), float(c.value)))
    for p in same:
        label = labels_a[p.site_a]
        for c in p.coefficients:
            plan.append((marked(f'ATOM {label} C{c.order} {atom_stem}', c.unrestricted_complete),
                         float(c.value)))
    for n in orders:
        value, complete = totals[n]
        plan.append((marked(f'{stem} C{n} TOTAL', complete), float(value)))
    plan.append((f'{stem} SITE PAIRS', float(len(dispersion.pairs))))
    plan.append((f'{stem} MAX ORDER', float(max(orders)) if orders else 0.))
    plan.append((f'{stem} QUADRATURE NODES', float(len(dispersion.cp_weights))))
    if dispersion.cp_weights:
        plan.append((f'{stem} QUADRATURE FREQUENCIES',
                     _matrix(np.asarray(frequencies, dtype=float).reshape(1, -1))))
        plan.append((f'{stem} CP WEIGHTS',
                     _matrix(np.asarray(dispersion.cp_weights, dtype=float).reshape(1, -1))))
    plan.extend(extra)
    current = {key.upper() for key, _ in plan}
    for key in list(wfn.scalar_variables()) + list(wfn.array_variables()):
        if _report_owned(key, stem, atom_stem) and key.upper() not in current:
            wfn.del_variable(key)
    for key, value in plan:
        wfn.set_variable(key, value)


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
    """Report contraction controls, refinement identity and each solver verdict."""
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
    """Report PFIT-derived C_n, using REFINED names for optional QCVariables."""
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
                        atom_stem='REFINED DISPERSION COEFFICIENT',
                        extra=(('ATOMIC REFINED DISPERSION ANCHOR SHIFT MAXABS',
                                float(dispersion.anchor_shift_maxabs)),))
