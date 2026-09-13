#!/usr/bin/env python3
"""Decode a CamCASP CASIMIR stage: the printed `Cn` table and the `.pol` it read.

Pure text decoding of reference *output data*, like read_cn_pot.py and
read_local_pol.py next to it.  No CamCASP source is read, run or transcribed,
and the reference tree is never written to.

This covers the PRIMARY acceptance case -- the shipped `tests/H2O_props/psi4`
declaration (`SCFcode psi4`, cc-pVDZ `Type MC`, PBE0, `AC NONE`, ALDA+CHF,
constrained-NN DF, 100x400 ISA/response grid) -- whose CASIMIR stage prints

    Dispersion coefficients for water (hartree bohr^n)
      O  O            C6        C7        C8   ...
        00   00   0   8.367451  0.0  81.67585  ...

One block per site TYPE pair.  The `00 00 0` row is the ISOTROPIC site-site
C_n, which is exactly the quantity isotropic_dispersion.cc assembles; every
other row is a recoupled component in the model's local axes and is reduced to
a census here, for the representation reason read_cn_pot.py documents.

The same stage's INPUT, `<prefix>_0f10.pol`, is decoded alongside and reduced
to its rank isotropics `abar_l = tr(alpha_ll)/(2l+1)`.  Carrying both lets a
test close the loop inside the repository: CamCASP's printed row must be what
Psi4's own Casimir assembly returns when fed CamCASP's own localized
polarizabilities.  Nothing here merges the two -- they stay separate fixture
entries, and their agreement is a measurement the test makes, not a definition.

    ./read_casimir_out.py <run-dir> [prefix]              # decode to stdout
    ./read_casimir_out.py --fixture <run-dir> [prefix]    # rewrite the fixture

`prefix` selects the localization: `water_L3` (default) or `water_L4`.  These
name two DIFFERENT declared models and are written to two different fixtures.
"""
import hashlib
import json
import pathlib
import re
import sys

#: Printed positional order labels of a CASIMIR dispersion block.
ORDERS = (6, 7, 8, 9, 10, 11, 12)

ISOTROPIC_ROW = ('00', '00', '0')

#: Racah component widths by rank; the localized file drops rank 0 entirely.
#: A `Limit all rank n` localization writes ranks 1..n, so the matrix width is
#: sum_{l=1..n} (2l+1) -- 15 at n=3, 24 at n=4.  A rank-4 localization is a
#: DIFFERENT declared model from the rank-3 one; the two fixtures are separate
#: files and their numbers are never quoted as agreeing, even where the decoded
#: bytes coincide.  That coincidence is a measurement about the files.
RANK_SLICE = {1: slice(0, 3), 2: slice(3, 8), 3: slice(8, 15), 4: slice(15, 24)}

#: Localized matrix width by highest written rank.
RANK_WIDTH = {1: 3, 2: 8, 3: 15, 4: 24}

#: `LK(la lb)` header of a single-site recoupled polarizability component.
RECOUPLED_HEADER = re.compile(r'^(\d\d[cs]?)\((\d)(\d)\)\s*(all zero)?\s*$')

NOTICE = (
    'Decoded from printed CamCASP output data of a run of the shipped '
    'tests/H2O_props/psi4 declaration (SCFcode psi4, cc-pVDZ Type MC, PBE0, AC NONE, '
    'ALDA+CHF, DF-TYPE-MONOMER NN, Angular 400 Radial 100). CamCASP: Anthony Stone and '
    'Alston Misquitta, MIT License, Copyright (c) 2019 Anthony Stone; see '
    'h2o_props_psi4_basis/NOTICE for the retained notice in full. Only the isotropic '
    '00 00 0 rows, a census of the recoupled rows, the rank isotropics of the localized '
    'polarizability file and the declared stage settings are extracted; no CamCASP '
    'source is read, run or transcribed, and no claim is made about ORIENT, RRF or '
    'bundled third-party basis databases.')


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def read_dispersion(path):
    """`{('O','O'): {'isotropic': {6: x, ...}, 'census': {...}}}` from `<p>_casimir.out`."""
    blocks, pair, orders = {}, None, ()
    for line in open(path):
        fields = line.split()
        if len(fields) == 2 + len(ORDERS) and fields[2] == 'C6':
            pair, orders = (fields[0], fields[1]), tuple(int(f[1:]) for f in fields[2:])
            blocks['%s %s' % pair] = {'isotropic': {}, 'census': {
                'rows': 0, 'printed_values': 0, 'exact_zeros': 0, 'j_values': [],
                'orders': list(orders)}}
            continue
        if pair is None or len(fields) < 4 or not re.fullmatch(r'\d\d[cs]?', fields[0]):
            continue
        block = blocks['%s %s' % pair]
        values = [float(f) for f in fields[3:]]
        if tuple(fields[:3]) == ISOTROPIC_ROW:
            # Positional over the full declared order list; only this row is.
            assert len(values) == len(orders), (path, fields[:3], len(values))
            block['isotropic'] = {str(n): v for n, v in zip(orders, values)}
        block['census']['rows'] += 1
        block['census']['printed_values'] += len(values)
        block['census']['exact_zeros'] += sum(v == 0.0 for v in values)
        if int(fields[2]) not in block['census']['j_values']:
            block['census']['j_values'].append(int(fields[2]))
    for name, block in blocks.items():
        if not block['isotropic']:
            raise ValueError('%s: block %r printed no %r row' % (path, name, ISOTROPIC_ROW))
    return blocks


def read_recoupled_inventory(path):
    """`{label: {'type': t, 'pairs': [[la, lb], ...], 'isotropic_pairs': [...]}}`.

    The `Recoupled polarizabilities` section of `<prefix>_casimir.out` prints one
    single-site component per `LK(la lb)` header.  Which `(la, lb)` are present is
    a structural property of the run, not a numerical one: CASIMIR builds the
    single-site table only where the recoupling it needs for the declared
    `Dispersion` order exists, so a pair absent here contributes nothing to any
    printed coefficient no matter what the localized file holds at that rank.
    `isotropic_pairs` keeps the `L=0` (`00(la lb)`) subset, which is the only part
    the isotropic `00 00 0` row can draw on.
    """
    sites, site = {}, None
    for line in open(path):
        fields = line.split()
        if len(fields) == 4 and fields[0] == 'Site' and fields[2] == 'type':
            site = sites.setdefault(fields[1], {'type': fields[3], 'pairs': [], 'isotropic_pairs': []})
            continue
        match = RECOUPLED_HEADER.match(line.rstrip('\n'))
        if site is None or match is None:
            continue
        lk, pair = match.group(1), [int(match.group(2)), int(match.group(3))]
        if pair not in site['pairs']:
            site['pairs'].append(pair)
        if lk == '00' and pair not in site['isotropic_pairs']:
            site['isotropic_pairs'].append(pair)
    for site in sites.values():
        site['pairs'].sort()
        site['isotropic_pairs'].sort()
    return sites


def read_local_pol(path):
    """`(labels, freqsq, {label: {rank: [abar per frequency]}})` from `<p>_0f10.pol`."""
    labels, freqsq, blocks, rows, ranks = [], [], {}, None, set()
    for line in open(path):
        if line.startswith('ALPHA'):
            fields = line.split()
            label = fields[3]
            ranks.add(int(fields[fields.index('TO') + 1]))
            index, fsq = int(fields[fields.index('INDEX') + 1]), float(fields[-1])
            if label not in labels:
                labels.append(label)
            if index - 1 == len(freqsq):
                freqsq.append(fsq)
            rows = blocks.setdefault(label, [])
            rows.append([])
            continue
        if line.startswith('ENDFILE'):
            rows = None
            continue
        if rows is not None and line.strip():
            rows[-1].append([float(f) for f in line.split()])
    if len(ranks) != 1:
        raise ValueError('%s: mixed `RANK 1 TO n` declarations %r' % (path, sorted(ranks)))
    top = ranks.pop()
    if top not in RANK_WIDTH:
        raise ValueError('%s: unsupported localization rank %d' % (path, top))
    width, present = RANK_WIDTH[top], [l for l in sorted(RANK_SLICE) if l <= top]
    isotropic = {}
    for label in labels:
        per_rank = isotropic.setdefault(label, {str(l): [] for l in present})
        for matrix in blocks[label]:
            assert len(matrix) == width and all(len(r) == width for r in matrix), label
            for rank in present:
                cut = RANK_SLICE[rank]
                rows_l = matrix[cut]
                trace = sum(row[cut][i] for i, row in enumerate(rows_l))
                per_rank[str(rank)].append(trace / (2 * rank + 1))
    return labels, freqsq, isotropic, top


def read_declared(path):
    """Declared CASIMIR-stage settings from `<prefix>_casimir.data`."""
    declared, sites = {}, []
    for line in open(path):
        fields = line.split()
        if len(fields) >= 2 and fields[0] in ('Frequencies', 'Skip', 'Print', 'Units'):
            declared[fields[0]] = ' '.join(fields[1:])
        elif len(fields) == 4 and fields[0] == 'Site':
            sites.append({'label': fields[1], 'type': fields[3]})
    declared['sites'] = sites
    return declared


def build(rundir, prefix='water_L3'):
    rundir = pathlib.Path(rundir)
    out, pol, data = (rundir / ('%s_casimir.out' % prefix), rundir / ('%s_0f10.pol' % prefix),
                      rundir / ('%s_casimir.data' % prefix))
    labels, freqsq, isotropic, top = read_local_pol(pol)
    declared = read_declared(data)
    beta, nfreq = declared['Frequencies'].split()
    return {
        'case': 'tests/H2O_props/psi4 declaration (primary acceptance target)',
        'prefix': prefix,
        'notice': NOTICE,
        'grid': {'beta': float(beta), 'frequency_count': int(nfreq), 'static_index': 0,
                 'printed_dynamic_count': len(freqsq) - 1,
                 'frequencies_squared': freqsq},
        'localized': {'source': pol.name, 'sha256': sha256(pol),
                      'site_labels': labels,
                      'ranks': [l for l in sorted(RANK_SLICE) if l <= top],
                      'isotropic_by_rank': isotropic},
        'recoupled': {'source': out.name, 'sites': read_recoupled_inventory(out)},
        'dispersion': {'source': out.name, 'sha256': sha256(out),
                       'declared': declared, 'data_sha256': sha256(data),
                       'blocks': read_dispersion(out)},
    }


def main(argv):
    fixture = '--fixture' in argv
    positional = [a for a in argv[1:] if not a.startswith('--')]
    rundir = positional[0]
    prefix = positional[1] if len(positional) > 1 else 'water_L3'
    payload = build(rundir, prefix)
    if not fixture:
        print(json.dumps(payload, indent=1, sort_keys=True))
        return 0
    target = (pathlib.Path(__file__).resolve().parent.parent
              / ('camcasp_casimir_h2o_vdz_%s.json' % prefix.rsplit('_', 1)[-1].lower()))
    target.write_text(json.dumps(payload, indent=1, sort_keys=True) + '\n')
    print('wrote %s (%d bytes)' % (target, target.stat().st_size))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
