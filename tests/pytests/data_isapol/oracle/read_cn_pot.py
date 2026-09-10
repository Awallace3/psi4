#!/usr/bin/env python3
"""Read CamCASP localized `<name>_Cn.pot` potentials and build the fixture.

Pure text decoding of reference *output data*.  No CamCASP source is read, run
or transcribed here, and the reference tree is never written to: this is the
same "results only" discipline as the rest of this directory, applied to a file
that is already a printed result.

File shape.  A run of `!` header records, then one block per site-TYPE pair,

    <A> <B>  C6 C8 C10 ...
    <index_a> <index_b> <j>   <value_for_C6> [<value_for_C8> ...]
    ...
    End

with values positional in the block's declared order list.  The row
`00 00 0` is the **isotropic** site-site C_n; every other row is a recoupled
Stone component `C_n^{l_a k_a, l_b k_b, j}` in the model's local axes.  One
printed block stands for every ordered site pair of those two types, because
the reference model declares one variable set per type and `COPY`s the rest.

What the fixture carries, and what it deliberately does not.  The isotropic
rows are extracted in full, because those are what a native isotropic
dispersion chain can be compared against at all.  The recoupled rows are
reduced to a **census** -- counts, indices, j values, how many printed entries
are exactly zero -- and their values are not extracted, because the native
anisotropic product is `orientation_resolved_scalars_not_recoupled_components`,
a different representation; extracting the numbers would invite a comparison
that is not yet defined.  Re-run this script if that changes.

The `.clt` inputs are decoded alongside, because the three back-end directories
of `tests/H2O_props` differ in **two** declarations, not one: `SCFcode` and
`HOMO`.  With CamCASP's own 27.21136 eV/Eh those `HOMO` values against the
shared `I.P. 12.62063 eV` are three different GRAC shifts, so the family's
spread is not purely SCF-code noise and the fixture records the shift each row
was produced with.

    ./read_cn_pot.py <one.pot>                     # decode one file to stdout
    ./read_cn_pot.py --fixture <camcasp-tests-dir> # rewrite ../camcasp_cn_pot_h2o_l2h1.json
"""
import hashlib
import json
import pathlib
import re
import sys

#: CamCASP's own eV -> Hartree divisor, recovered exactly from the declared
#: `I.P. 12.62063 eV` / `HOMO -0.3989` pair of the Psi4 back-end row.
EV_PER_HARTREE = 27.21136

#: Positional order labels a `Cn.pot` block may declare.  Odd orders are printed
#: by CamCASP; the *isotropic* row is identically zero at every one of them for
#: this model, but the recoupled rows are not (O-O prints 46 nonzero components
#: at n=7 and 118 at n=9), so the census records both rather than assuming.
ORDERS = (6, 7, 8, 9, 10, 11, 12)

ISOTROPIC_ROW = ('00', '00', 0)

#: Travels with the fixture, because the numbers in it are somebody else's result.
NOTICE = (
    'Decoded from printed CamCASP output data of the shipped tests/H2O_props case. '
    'CamCASP: Anthony Stone and Alston Misquitta, MIT License, Copyright (c) 2019 '
    'Anthony Stone; see h2o_props_psi4_basis/NOTICE for the retained notice in full. '
    'Only the isotropic 00 00 0 rows, the localization header records, the declared '
    'GRAC inputs and a census of the recoupled rows are extracted; no CamCASP source '
    'is read, run or transcribed, and no claim is made about ORIENT, RRF or bundled '
    'third-party basis databases.')


def read_cn_pot(path):
    """`(header, blocks)`; `blocks[(A, B)] = {'orders': (...), 'rows': {...}}`."""
    header, blocks, block, orders = {}, {}, None, ()
    for line in open(path):
        text = line.rstrip('\n')
        if text.startswith('!'):
            m = re.match(r'!\s+([^:]+):\s*(.*?)\s*$', text)
            if m:
                header[m.group(1).strip()] = m.group(2)
            continue
        fields = text.split()
        if not fields:
            continue
        if fields[0] == 'End':
            block = None
            continue
        if fields[2:3] and fields[2].startswith('C') and fields[2][1:].isdigit():
            block = (fields[0], fields[1])
            orders = tuple(int(f[1:]) for f in fields[2:])
            assert set(orders) <= set(ORDERS), orders
            assert block not in blocks, block
            blocks[block] = dict(orders=orders, rows={})
            continue
        assert block is not None, text
        key = (fields[0], fields[1], int(fields[2]))
        values = [float(v) for v in fields[3:]]
        assert len(values) <= len(orders), text
        assert key not in blocks[block]['rows'], key
        blocks[block]['rows'][key] = dict(zip(orders, values))
    assert blocks, path
    for pair, b in blocks.items():
        assert ISOTROPIC_ROW in b['rows'], pair
    return header, blocks


def isotropic(blocks):
    """The `00 00 0` row of every type pair: the isotropic site-site C_n."""
    return {pair: b['rows'][ISOTROPIC_ROW] for pair, b in blocks.items()}


def census(blocks):
    """Row/component census of the recoupled rows, i.e. everything but `00 00 0`.

    `printed_zero` counts entries printed as exactly 0.0, which is most of the
    grid of (row, order) pairs because a block declares one order list for all of
    its rows; a census that did not separate them would badly overstate the
    uncompared track.  `isotropic_orders_printed_zero` is reported per block for
    the same reason, and it is the isotropic row alone that vanishes at the odd
    orders here.
    """
    out = {}
    for pair, b in blocks.items():
        rows = {k: v for k, v in b['rows'].items() if k != ISOTROPIC_ROW}
        printed = zero = 0
        per_order = {}
        for key, values in rows.items():
            for order, value in values.items():
                printed += 1
                zero += (value == 0.)
                if value != 0.:
                    per_order[order] = per_order.get(order, 0) + 1
        nonzero_rows = [k for k, v in rows.items() if any(x != 0. for x in v.values())]
        out[pair] = dict(orders=list(b['orders']), recoupled_rows=len(rows),
                         recoupled_rows_nonzero=len(nonzero_rows),
                         printed_entries=printed, printed_zero=zero,
                         nonzero_rows_per_order={str(k): v for k, v in sorted(per_order.items())},
                         indices=sorted({i for k in rows for i in k[:2]}),
                         j_values=sorted({k[2] for k in rows}),
                         isotropic_orders_printed_zero=sorted(
                             o for o, v in b['rows'][ISOTROPIC_ROW].items() if v == 0.))
    return out


def read_clt(path):
    """The few declarations of a `<name>.clt` this comparison depends on."""
    out = {'sites': []}
    for line in open(path):
        fields = line.split('!')[0].split()
        if not fields:
            continue
        head = fields[0].lower()
        if head == 'i.p.' and len(fields) >= 3 and fields[2].lower() == 'ev':
            out['ip_ev'] = float(fields[1])
        elif head == 'homo':
            out['homo'] = float(fields[1])
        elif head in ('basis', 'scfcode'):
            out[head] = fields[1]
        elif len(fields) == 7 and fields[5].lower() == 'type':
            out['sites'].append(dict(label=fields[0], charge=float(fields[1]),
                                     xyz=[float(v) for v in fields[2:5]], type=fields[6]))
    if 'ip_ev' in out and 'homo' in out:
        out['grac_shift'] = out['ip_ev']/EV_PER_HARTREE + out['homo']
    return out


def _digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def decode(path):
    header, blocks = read_cn_pot(path)
    return dict(header=header, sha256=_digest(path),
                isotropic={' '.join(k): {str(o): v for o, v in row.items()}
                           for k, row in isotropic(blocks).items()},
                census={' '.join(k): v for k, v in census(blocks).items()})


def build_fixture(tests_dir, relative='check/L2H1/H2O_ref_wt3_L2_Cn.pot',
                  clt='H2O-avtz.clt', axes='H2O.axes'):
    """Decode every back-end row of a `tests/H2O_props`-shaped directory."""
    tests_dir = pathlib.Path(tests_dir)
    rows, axes_text = {}, None
    for backend in sorted(p.name for p in tests_dir.iterdir() if (p/relative).is_file()):
        base = tests_dir/backend
        row = decode(base/relative)
        row['input'] = read_clt(base/clt)
        row['input']['sha256'] = _digest(base/clt)
        row['axes_sha256'] = _digest(base/axes)
        text = (base/axes).read_text()
        assert axes_text is None or text == axes_text, backend
        axes_text = text
        rows[backend] = row
    assert rows, tests_dir
    shared = {k: v for k, v in next(iter(rows.values()))['header'].items()
              if all(r['header'].get(k) == v for r in rows.values())}
    return dict(source=str(relative), backends=rows, shared_header=shared,
                axes=axes_text, ev_per_hartree=EV_PER_HARTREE, notice=NOTICE)


if __name__ == '__main__':
    if sys.argv[1] == '--fixture':
        out = pathlib.Path(__file__).resolve().parent.parent/'camcasp_cn_pot_h2o_l2h1.json'
        out.write_text(json.dumps(build_fixture(sys.argv[2]), indent=1, sort_keys=True) + '\n')
        print(out)
    else:
        print(json.dumps(decode(sys.argv[1]), indent=1, sort_keys=True))
