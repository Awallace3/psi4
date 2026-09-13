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

    ./read_casimir_out.py <run-dir>              # decode to stdout
    ./read_casimir_out.py --fixture <run-dir>    # rewrite the committed fixture
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
RANK_SLICE = {1: slice(0, 3), 2: slice(3, 8), 3: slice(8, 15)}

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


def read_local_pol(path):
    """`(labels, freqsq, {label: {rank: [abar per frequency]}})` from `<p>_0f10.pol`."""
    labels, freqsq, blocks, rows = [], [], {}, None
    for line in open(path):
        if line.startswith('ALPHA'):
            fields = line.split()
            label = fields[3]
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
    isotropic = {}
    for label in labels:
        per_rank = isotropic.setdefault(label, {str(l): [] for l in RANK_SLICE})
        for matrix in blocks[label]:
            assert len(matrix) == 15 and all(len(r) == 15 for r in matrix), label
            for rank, cut in RANK_SLICE.items():
                rows_l = matrix[cut]
                trace = sum(row[cut][i] for i, row in enumerate(rows_l))
                per_rank[str(rank)].append(trace / (2 * rank + 1))
    return labels, freqsq, isotropic


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
    labels, freqsq, isotropic = read_local_pol(pol)
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
                      'site_labels': labels, 'ranks': sorted(RANK_SLICE),
                      'isotropic_by_rank': isotropic},
        'dispersion': {'source': out.name, 'sha256': sha256(out),
                       'declared': declared, 'data_sha256': sha256(data),
                       'blocks': read_dispersion(out)},
    }


def main(argv):
    fixture = '--fixture' in argv
    rundir = [a for a in argv[1:] if not a.startswith('--')][0]
    payload = build(rundir)
    if not fixture:
        print(json.dumps(payload, indent=1, sort_keys=True))
        return 0
    target = pathlib.Path(__file__).resolve().parent.parent / 'camcasp_casimir_h2o_vdz_l3.json'
    target.write_text(json.dumps(payload, indent=1, sort_keys=True) + '\n')
    print('wrote %s (%d bytes)' % (target, target.stat().st_size))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
