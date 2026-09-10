#!/usr/bin/env python3
"""Read the CamCASP `examples/properties/H2O` polarizability outputs.

Pure text decoding of reference *output data*, plus the case's own *input*
declarations.  No CamCASP source is read, run or transcribed here, and the
reference tree is never written to: the same "results only" discipline as
`read_cn_pot.py`, applied to a second, independently declared case.

Why a second case at all.  `read_cn_pot.py` decodes `tests/H2O_props`, which is
weight type 3, `H2O-avtz.clt`, and -- decisively -- the `H2O.axes` declaration
`H1  z global Z x from H2 to H1`.  This case is weight type **4**, prefix
`H2O_aTZ`, and declares *bond* axes, `H1  z from O to H1   x from H2 to H1`.
The two must never be conflated; their molecular isotropic C6 differs by 2.727138
(`dispersion['molecular_isotropic']['6']` here, 43.890270, against the other
fixture's psi4 row, 46.617408 -- 6.21% of the first, 5.85% of the second), which
is the reference pipeline's own spread across protocol choices and therefore a
bound on what "matching the reference" can mean.

What this case adds that the other cannot.
  * A **rank-4** static distributed polarizability at full double precision
    (`_NL4_static.pol`), which is the only rank-4 reference quantity available.
  * **Refined local** tensors at all 11 Casimir frequencies (`_0f10.pol`),
    i.e. the output of the reference's own refinement step, in local axes.
  * The `.pdef` **model declaration** those tensors were fitted under, so the
    model is read from the input side rather than inferred from the output.
  * A `casimir.out` **dispersion** block computed from those same local tensors,
    which turns the pair into a closed oracle for the Casimir-Polder step alone:
    local polarizabilities in, dispersion coefficients out, both printed.

What the fixture carries, and what it deliberately does not.
  * The refined local tensors are carried as the **`.pdef` variables** they are
    built from -- 13 on O, 4 on H1, and `H2 = COPY H1` -- one value per
    frequency.  `local_tensor` reconstructs the 9x9 blocks from them and
    `decode` asserts the reconstruction reproduces the printed file exactly, so
    this is a lossless representation *and* a check that the declared model
    really describes the printed output.  It is 187 numbers instead of 2673.
  * The isotropic dispersion rows are carried in full, because those are what a
    native isotropic chain can be compared against.  The recoupled rows are
    reduced to a census by `read_cn_pot.census`, for the reason given there --
    except the `00(l l)` diagonal, which *is* carried, because it is the
    recoupled image of the per-rank isotropic scalar and nothing else and so
    certifies the `(-1)^l sqrt(2 l + 1)` convention the isotropic reduction
    rests on.  The anisotropic recoupled components stay withheld.
  * The 75x75 rank-4 distributed tensor is **not** carried element by element.
    Nothing in the native chain is compared to it element by element: the
    partition differs, so only its translation-invariant molecular
    polarizability and its constraint residuals are comparable, and those are
    recorded as derived numbers with the file's sha256 beside them.  Carrying
    5625 numbers we do not compare would be the opposite of the discipline in
    `read_cn_pot.py`.  Re-run this script if that changes.

    ./read_local_pol.py <case-dir>              # decode to stdout
    ./read_local_pol.py --fixture <case-dir>    # rewrite ../camcasp_local_pol_h2o_atz_wt4.json
"""
import hashlib
import json
import pathlib
import re
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from read_cn_pot import census, isotropic, read_cn_pot, read_clt

#: Racah component order, `(rank + 1) ** 2` of them; dipoles are z, x, y.
COMPONENTS = ('00', '10', '11c', '11s', '20', '21c', '21s', '22c', '22s',
              '30', '31c', '31s', '32c', '32s', '33c', '33s',
              '40', '41c', '41s', '42c', '42s', '43c', '43s', '44c', '44s')

#: Cartesian index of each rank-1 Racah component, i.e. `10 -> z, 11c -> x, 11s -> y`.
DIPOLE_CARTESIAN = {'10': 2, '11c': 0, '11s': 1}

_RECOUPLED_ISOTROPIC = re.compile(r'^00\((\d)\1\)$')

#: Travels with the fixture, because the numbers in it are somebody else's result.
NOTICE = (
    'Decoded from printed CamCASP output data and input declarations of the shipped '
    'examples/properties/H2O case. CamCASP: Anthony Stone and Alston Misquitta, MIT '
    'License, Copyright (c) 2019 Anthony Stone; see h2o_props_psi4_basis/NOTICE for '
    'the retained notice in full. Only the .pdef model variables, the isotropic '
    'dispersion rows, the isotropic 00(ll) recoupled rows, a census of the remaining '
    'recoupled rows, the declared inputs and derived '
    'invariants of the rank-4 distributed tensor are extracted; no CamCASP source is '
    'read, run or transcribed, and no claim is made about ORIENT, RRF or bundled '
    'third-party basis databases.')


def _digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def read_pdef(path):
    """The `Polarizabilities` model declaration: named variables and COPY rules.

    Returns `{'variables': [(site, ca, cb, name), ...], 'copies': [(to, frm), ...]}`
    in declaration order.  A variable names one *component pair* of one site; the
    fit has exactly this many free numbers per frequency, which is what makes the
    printed 9x9 blocks compressible without loss.
    """
    out = {'variables': [], 'copies': []}
    section = None
    for line in open(path):
        fields = line.split('!')[0].split()
        if not fields:
            continue
        if fields[0].lower() == 'polarizabilities':
            section = 'pol'
            continue
        if fields[0].lower() == 'end':
            section = None
            continue
        if section != 'pol':
            continue
        if len(fields) >= 5 and fields[2].upper() == 'COPY':
            out['copies'].append(((fields[0], fields[1]), (fields[3], fields[4])))
            continue
        assert fields[4] == '=' and len(fields) == 6, line
        assert fields[0] == fields[1], line          # diagonal site blocks only
        assert fields[2] in COMPONENTS and fields[3] in COMPONENTS, line
        out['variables'].append((fields[0], fields[2], fields[3], fields[5]))
    assert out['variables'], path
    return out


def read_local_pol(path, ncomp=9):
    """`{index: {(A, B): (ncomp, ncomp) array}}` from a refined `<name>_0f10.pol`.

    Blocks are introduced by a two-token non-numeric site-pair header and closed
    by a blank line or the next `# INDEX`.
    """
    blocks, index, pair, rows = {}, None, None, []

    def flush():
        if pair is not None:
            array = np.array(rows, dtype=float)
            assert array.shape == (ncomp, ncomp), (index, pair, array.shape)
            assert pair not in blocks.get(index, {}), (index, pair)
            blocks.setdefault(index, {})[pair] = array

    for line in open(path):
        text = line.strip()
        if text.startswith('# INDEX'):
            flush()
            pair, rows = None, []
            index = int(text.split()[-1])
            continue
        if not text:
            flush()
            pair, rows = None, []
            continue
        fields = text.split()
        if len(fields) == 2 and not fields[0][:1].isdigit() and fields[0][:1] not in '-+.':
            flush()
            pair, rows = (fields[0], fields[1]), []
            continue
        rows.append([float(v) for v in fields])
    flush()
    assert blocks, path
    return blocks


def read_square_pol(path):
    """The one square tensor of a `<name>_NL4_static.pol`, plus its printed index.

    The file is a `# INDEX` record followed by free-form values in row-major
    order; the dimension is recovered from the count, which is how the site count
    and rank are cross-checked against the `.cks` declaration rather than assumed.
    """
    index, values = None, []
    for line in open(path):
        text = line.strip()
        if text.startswith('# INDEX'):
            assert index is None, path
            index = text.split()[-1]
            continue
        values.extend(float(v) for v in text.split())
    dimension = int(round(np.sqrt(len(values))))
    assert dimension * dimension == len(values), (path, len(values))
    return index, np.array(values, dtype=float).reshape(dimension, dimension)


def read_recoupled_isotropic(path, marker='Dispersion coefficients'):
    """The `00(l l)` recoupled rows of a `_casimir.out`, per site.

    Only the isotropic diagonal is taken.  Those rows are the recoupled image of
    the per-rank isotropic scalar and nothing else, so recording them stays
    inside the same isotropic track as the dispersion rows; the anisotropic
    recoupled components remain an uncompared track and are not extracted here.

    The printed rows carry the **dynamic** nodes only -- a `Quad n` case prints
    `n` columns, not `n + 1` -- so the returned lists are indexed 1..n of the
    Casimir grid and the static point is absent by construction.
    """
    rows, site, row_name, values = {}, None, None, []

    def flush():
        if site is not None and row_name is not None and values:
            rows.setdefault(site, {})[row_name] = list(values)

    for line in pathlib.Path(path).read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith(marker):
            break
        if stripped.startswith('Site '):
            flush()
            site, row_name, values = stripped.split()[1], None, []
            continue
        if site is None or not stripped:
            continue
        try:
            numbers = [float(z) for z in stripped.split()]
        except ValueError:
            flush()
            row_name, values = (stripped if _RECOUPLED_ISOTROPIC.match(stripped)
                                else None), []
            continue
        if row_name is not None:
            values.extend(numbers)
    flush()
    return rows


def read_dispersion(path, marker='Dispersion coefficients'):
    """The `Cn.pot`-shaped dispersion section of a `<name>_casimir.out`.

    The section has exactly the block grammar `read_cn_pot` already parses, so the
    rows are handed to it rather than re-parsed here.
    """
    lines = open(path).read().splitlines()
    start = next(i for i, t in enumerate(lines) if marker in t)
    text = '\n'.join(lines[start + 1:]) + '\n'
    with tempfile.NamedTemporaryFile('w', suffix='.pot') as handle:
        handle.write(text)
        handle.flush()
        return read_cn_pot(handle.name)[1]


def read_cks(path):
    """The declared protocol, as an ordered list of `(block, [(key, value), ...])`.

    Kept as a list because the case declares **two** `DF` blocks and **two**
    `Polarizability` blocks -- `Eta = 0.0` / `Rank 2` for the total-polarizability
    stage and `Eta = 0.0005` / `Rank 4` for the distributed stage that writes the
    `NL4` file -- and a dict would silently lose one of each.
    """
    blocks, current = [], None
    for line in open(path):
        raw = line.split('!')[0].rstrip()
        fields = raw.split()
        if not fields:
            continue
        head = fields[0].upper()
        if head in ('SET', 'BEGIN') and len(fields) >= 2:
            current = (f'{head} {fields[1]}', [])
            blocks.append(current)
            continue
        if head in ('END', 'FINISH'):
            current = None
            continue
        entry = ' '.join(fields)
        if current is None:
            blocks.append(('', [entry]))
        else:
            current[1].append(entry)
    return [dict(block=name, declarations=list(entries)) for name, entries in blocks]


def read_axes(path):
    """`{site: (z_spec, x_spec)}` from an `<name>.axes`, plus the verbatim text."""
    text = pathlib.Path(path).read_text()
    axes = {}
    for line in text.splitlines():
        fields = line.split('!')[0].split()
        if not fields or fields[0].lower() in ('axes', 'end'):
            continue
        assert fields[1].lower() == 'z' and 'x' in fields, line
        cut = fields.index('x')
        axes[fields[0]] = (' '.join(fields[2:cut]), ' '.join(fields[cut + 1:]))
    return axes, text


def axes_frames(axes, sites):
    """Proper local-to-global frames (columns) built from an `.axes` declaration.

    Only the two spec forms this case declares are honoured -- `global Z` and
    `from <A> to <B>` -- and anything else raises, so a frame is never guessed.
    Undeclared sites keep global axes, which is CamCASP's own default.
    """
    position = {s['label']: np.array(s['xyz'], dtype=float) for s in sites}

    def direction(spec):
        fields = spec.split()
        if len(fields) == 2 and fields[0].lower() == 'global':
            return {'x': np.array([1., 0, 0]), 'y': np.array([0, 1., 0]),
                    'z': np.array([0, 0, 1.])}[fields[1].lower()]
        assert len(fields) == 4 and fields[0].lower() == 'from' and fields[2].lower() == 'to', spec
        return position[fields[3]] - position[fields[1]]

    frames = []
    for site in sites:
        if site['label'] not in axes:
            frames.append(np.eye(3))
            continue
        z_spec, x_spec = axes[site['label']]
        zhat = direction(z_spec)
        zhat = zhat / np.linalg.norm(zhat)
        xhat = direction(x_spec)
        xhat = xhat - zhat * float(xhat @ zhat)
        assert np.linalg.norm(xhat) > 1e-8, (site['label'], x_spec)
        xhat = xhat / np.linalg.norm(xhat)
        frame = np.column_stack([xhat, np.cross(zhat, xhat), zhat])
        assert abs(np.linalg.det(frame) - 1.) < 1e-12, site['label']
        frames.append(frame)
    return np.array(frames)


def local_tensor(model, values, label, ncomp=9):
    """Rebuild one site's symmetric `(ncomp, ncomp)` block from `.pdef` variables.

    `values` maps variable name -> number.  Components absent from the model are
    zero, which is how the reference's rank limits appear in its own output: the
    `00` row and column vanish and, on the hydrogens, everything above rank 1.
    """
    source = dict(model['copies']).get((label, label), (label, label))[0]
    index = {c: i for i, c in enumerate(COMPONENTS)}
    tensor = np.zeros((ncomp, ncomp))
    for site, ca, cb, name in model['variables']:
        if site != source:
            continue
        row, col = index[ca], index[cb]
        if row >= ncomp or col >= ncomp:
            continue
        tensor[row, col] = tensor[col, row] = values[name]
    return tensor


def isotropic_by_rank(tensor, max_rank=None):
    """`{rank: mean diagonal}`, the isotropic scalar of each rank block.

    `alpha_bar_l = tr(A[block_l]) / (2 l + 1)`, the quantity an isotropic
    dispersion model is built from.  A rank the model does not carry comes back
    exactly 0.
    """
    out = {}
    rank = 1
    while (rank + 1) ** 2 <= tensor.shape[0] and (max_rank is None or rank <= max_rank):
        block = slice(rank * rank, (rank + 1) ** 2)
        out[rank] = float(np.trace(tensor[block, block]) / (2 * rank + 1))
        rank += 1
    return out


def molecular_alpha_from_local(tensors, frames):
    """Sum local dipole-dipole blocks into the molecular polarizability, in xyz.

    Local Racah `10, 11c, 11s` are `z, x, y`; each site's block is rotated by its
    own frame and the sum is a genuine molecular property, which is what makes it
    comparable across partitions at all.
    """
    total = np.zeros((3, 3))
    for tensor, frame in zip(tensors, frames):
        cartesian = np.zeros((3, 3))
        for ca, ia in DIPOLE_CARTESIAN.items():
            for cb, ib in DIPOLE_CARTESIAN.items():
                cartesian[ia, ib] = tensor[COMPONENTS.index(ca), COMPONENTS.index(cb)]
        total += frame @ cartesian @ frame.T
    return total


def molecular_alpha_from_distributed(square, origins, ncomp):
    """Translate a distributed polarizability to the molecular one, sign measured.

    `alpha_ij = sum_ab [ a^ab_{ti tj} + r_i(a) a^ab_{00,tj} + r_j(b) a^ab_{ti,00}
    + r_i(a) r_j(b) a^ab_{00,00} ]`.  The relative sign of the charge-flow terms
    is not assumed: both are returned, and the caller picks by which one sends
    the C2v-forbidden xz element to zero.
    """
    nsite = square.shape[0] // ncomp
    assert nsite * ncomp == square.shape[0], (square.shape, ncomp)
    origins = np.asarray(origins, dtype=float)
    out = {}
    for sign in (1., -1.):
        alpha = np.zeros((3, 3))
        for ca, ia in DIPOLE_CARTESIAN.items():
            for cb, ib in DIPOLE_CARTESIAN.items():
                ta, tb = COMPONENTS.index(ca), COMPONENTS.index(cb)
                total = 0.
                for a in range(nsite):
                    for b in range(nsite):
                        block = square[ncomp * a:ncomp * a + ncomp, ncomp * b:ncomp * b + ncomp]
                        ra, rb = origins[a][ia], origins[b][ib]
                        total += (block[ta, tb] + sign * ra * block[0, tb]
                                  + sign * rb * block[ta, 0] + ra * rb * block[0, 0])
                alpha[ia, ib] = total
        out['plus' if sign > 0 else 'minus'] = alpha
    return out


def charge_residuals(square, ncomp):
    """The two charge-flow sum rules and the total charge-charge element."""
    nsite = square.shape[0] // ncomp
    def block(a, b):
        return square[ncomp * a:ncomp * a + ncomp, ncomp * b:ncomp * b + ncomp]
    over_a = np.array([sum(block(a, b)[0, :] for a in range(nsite)) for b in range(nsite)])
    over_b = np.array([sum(block(a, b)[:, 0] for b in range(nsite)) for a in range(nsite)])
    return dict(sum_over_a_maxabs=float(np.max(np.abs(over_a))),
                sum_over_b_maxabs=float(np.max(np.abs(over_b))),
                charge_charge_total=float(sum(block(a, b)[0, 0]
                                              for a in range(nsite) for b in range(nsite))),
                asymmetry_maxabs=float(np.max(np.abs(square - square.T))))


def _isotropic_pairs(blocks):
    return {' '.join(k): {str(o): v for o, v in row.items()}
            for k, row in isotropic(blocks).items()}


def type_multiplicity(rows, sites):
    """Ordered site pairs each printed type-pair block stands for.

    One printed block per site TYPE pair, because the `.pdef` declares one
    variable set per type and `COPY`s the rest.  Same rule as
    `read_cn_pot`-driven comparisons use, derived here from the `.clt` `Type`
    declarations rather than written out, so a differently typed case cannot pick
    up water's counts.
    """
    counts = {}
    for site in sites:
        counts[site['type']] = counts.get(site['type'], 0) + 1
    out = {}
    for pair in rows:
        a, b = pair.split()
        out[pair] = counts[a]*counts[b]*(1 if a == b else 2)
    return out


def pdef_rank_limits(model):
    """`{site: highest rank the .pdef declares}`, the model's own rank limits.

    Read from the variable names, not from a separate `Limit rank` declaration,
    so the limits are the ones the printed tensors were actually fitted under.
    """
    out = {}
    for site, ca, cb, _ in model['variables']:
        for component in (ca, cb):
            out[site] = max(out.get(site, 0), int(component[0]))
    for (to, _), (frm, _) in model['copies']:
        out[to] = out[frm]
    return out


def admissible_orders(rows, sites, rank_limits, orders):
    """Which even isotropic orders each type pair can carry: `n = 2(l_a + l_b + 1)`.

    An order outside this set is structurally absent from the model rather than
    small, so a molecular total that sums over pairs is **partial** at every order
    some pair cannot reach.  Recording it here keeps that from being read as
    agreement later.
    """
    per_type = {}
    for site in sites:
        per_type[site['type']] = max(per_type.get(site['type'], 0), rank_limits[site['label']])
    out = {}
    for pair in rows:
        a, b = pair.split()
        reachable = {2*(la + lb + 1)
                     for la in range(1, per_type[a] + 1) for lb in range(1, per_type[b] + 1)}
        out[pair] = sorted(n for n in orders if n in reachable)
    return out


def molecular_isotropic(rows, sites, order):
    """The molecular isotropic `C_n`, summed over ordered site pairs.

    The only partition-invariant number in the dispersion comparison.  For this
    case at `n = 6` it is `C6(OO) + 4 C6(HO) + 4 C6(HH)`; the same combination
    reproduces the `tests/H2O_props` psi4 row's recorded 46.617408 exactly.
    """
    multiplicity = type_multiplicity(rows, sites)
    return sum(multiplicity[pair]*values.get(str(order), 0.)
               for pair, values in rows.items())


def decode(case_dir, prefix='H2O_aTZ', wt='wt4_L2'):
    case = pathlib.Path(case_dir)
    out_1, out_2 = case/'output_1', case/'output_2'
    clt = case/f'{case.name}.clt'
    input_declarations = read_clt(clt)
    axes, axes_text = read_axes(case/f'{case.name}.axes')
    model = read_pdef(out_2/f'{prefix}.pdef')

    refined_path = out_2/f'{prefix}_ref_{wt}_0f10.pol'
    refined = read_local_pol(refined_path)
    indices = sorted(refined)
    labels = [s['label'] for s in input_declarations['sites']]
    assert all(sorted(refined[k]) == sorted((l, l) for l in labels) for k in indices), refined_path

    # The .pdef variables, read off the printed blocks, then verified to
    # reproduce them exactly -- that is the check that the declared model is the
    # model the printed tensors were fitted under.
    component = {c: i for i, c in enumerate(COMPONENTS)}
    variables = {}
    for site, ca, cb, name in model['variables']:
        variables[name] = [float(refined[k][(site, site)][component[ca], component[cb]])
                           for k in indices]
    frames = axes_frames(axes, input_declarations['sites'])
    rebuilt, reconstruction_error = [], 0.
    for position, k in enumerate(indices):
        per_site = []
        values = {n: v[position] for n, v in variables.items()}
        for label in labels:
            tensor = local_tensor(model, values, label)
            reconstruction_error = max(reconstruction_error,
                                       float(np.max(np.abs(tensor - refined[k][(label, label)]))))
            per_site.append(tensor)
        rebuilt.append(per_site)
    assert reconstruction_error == 0., reconstruction_error

    refined_alpha = [molecular_alpha_from_local(per_site, frames) for per_site in rebuilt]
    copy_pairs = [((a, a), (b, b)) for (a, _), (b, _) in model['copies']]
    copy_discrepancy = max(
        [float(np.max(np.abs(refined[k][to] - refined[k][frm])))
         for k in indices for to, frm in copy_pairs] or [0.])

    dispersion_path = out_2/f'{prefix}_ref_{wt}_casimir.out'
    dispersion = read_dispersion(dispersion_path)
    rows = _isotropic_pairs(dispersion)
    recoupled = read_recoupled_isotropic(dispersion_path)

    square_path = out_2/f'{prefix}_NL4_static.pol'
    square_index, square = read_square_pol(square_path)
    ncomp = square.shape[0] // len(labels)
    rank = int(round(np.sqrt(ncomp))) - 1
    assert (rank + 1) ** 2 == ncomp, (square.shape, len(labels))
    origins = [s['xyz'] for s in input_declarations['sites']]
    translated = molecular_alpha_from_distributed(square, origins, ncomp)
    forbidden = {k: float(abs(v[0, 2])) for k, v in translated.items()}
    convention = min(forbidden, key=forbidden.get)

    return dict(
        case=f'{case.parent.name}/{case.name}',
        prefix=prefix, weight_label=wt, notice=NOTICE,
        input=dict(clt=input_declarations, clt_sha256=_digest(clt),
                   axes=axes_text, axes_specs={k: list(v) for k, v in axes.items()},
                   axes_sha256=_digest(case/f'{case.name}.axes'),
                   frames=[f.tolist() for f in frames]),
        protocol=dict(cks=read_cks(out_1/f'{prefix}.cks'),
                      cks_sha256=_digest(out_1/f'{prefix}.cks')),
        grid=dict(beta=.5, dynamic_nodes=len(indices) - 1, frequency_count=len(indices),
                  static_index=indices[0], indices=indices),
        model=dict(pdef_sha256=_digest(out_2/f'{prefix}.pdef'),
                   variables=[list(v) for v in model['variables']],
                   copies=[[list(a), list(b)] for a, b in model['copies']],
                   parameter_count=len(model['variables']),
                   components=list(COMPONENTS[:9])),
        refined_local=dict(
            source=refined_path.name, sha256=_digest(refined_path),
            site_labels=labels, values=variables,
            reconstruction_error=reconstruction_error,
            copy_local_discrepancy=copy_discrepancy,
            isotropic_by_rank={label: {str(r): [isotropic_by_rank(per_site[i])[r]
                                                for per_site in rebuilt]
                                       for r in (1, 2)}
                               for i, label in enumerate(labels)},
            molecular_alpha_static=refined_alpha[0].tolist(),
            molecular_alpha_isotropic=[float(np.trace(a)/3.) for a in refined_alpha]),
        recoupled_isotropic=dict(
            source=dispersion_path.name, sha256=_digest(dispersion_path),
            dynamic_indices=indices[1:], rows=recoupled),
        dispersion=dict(
            source=dispersion_path.name, sha256=_digest(dispersion_path),
            isotropic=rows, census={' '.join(k): v for k, v in census(dispersion).items()},
            type_multiplicity=type_multiplicity(rows, input_declarations['sites']),
            site_rank_limits=pdef_rank_limits(model),
            admissible_orders=admissible_orders(rows, input_declarations['sites'],
                                                pdef_rank_limits(model), (6, 8, 10, 12)),
            molecular_isotropic_complete_orders=[
                order for order in (6, 8, 10, 12)
                if all(order in v for v in admissible_orders(
                    rows, input_declarations['sites'], pdef_rank_limits(model),
                    (6, 8, 10, 12)).values())],
            molecular_isotropic={str(order): molecular_isotropic(
                rows, input_declarations['sites'], order) for order in (6, 8, 10)}),
        distributed_static=dict(
            source=square_path.name, sha256=_digest(square_path),
            printed_index=square_index, dimension=int(square.shape[0]),
            site_count=len(labels), rank=rank, components_per_site=ncomp,
            residuals=charge_residuals(square, ncomp),
            translation_convention=convention,
            forbidden_xz_by_convention=forbidden,
            molecular_alpha=translated[convention].tolist(),
            molecular_alpha_isotropic=float(np.trace(translated[convention])/3.)))


def internal_inconsistency(fixture):
    """How far the reference's refinement moves its own molecular polarizability.

    Both routes are the reference's own: the rank-4 distributed tensor translated
    to the origin, and the refined local tensors rotated by the declared axes and
    summed.  The molecular polarizability is partition-independent, so the gap is
    the refinement's own distortion and nothing else -- it bounds how exactly any
    chain can be said to reproduce "the" reference number.
    """
    a = fixture['distributed_static']['molecular_alpha_isotropic']
    b = fixture['refined_local']['molecular_alpha_isotropic'][0]
    return dict(distributed=a, refined=b, absolute=abs(b - a), relative=abs(b - a)/a)


if __name__ == '__main__':
    args = sys.argv[1:]
    fixture = decode(args[-1])
    fixture['internal_inconsistency'] = internal_inconsistency(fixture)
    if args[0] == '--fixture':
        out = (pathlib.Path(__file__).resolve().parent.parent
               / 'camcasp_local_pol_h2o_atz_wt4.json')
        out.write_text(json.dumps(fixture, indent=1, sort_keys=True) + '\n')
        print(out)
    else:
        print(json.dumps(fixture, indent=1, sort_keys=True))
