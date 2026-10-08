# Copyright (c) 2007-2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""Regenerate or check the 100-digit reference solution of the frozen water Drho-C system.

    python make_drhoc_water_reference.py --write   # (re)write the solution in the JSON
    python make_drhoc_water_reference.py --check   # recompute and compare (exit 1 on mismatch)

Standard library only. Reads A and b from drhoc_water_operands.npz exactly
(binary64 bits), converts them exactly to Decimal and solves A c = b by
partially pivoted (|.|, as DGETRF) Gaussian elimination at 100 and 120
significant digits, with no symmetrization or truncation. The stored digits
are the 100-digit solution; the 120-digit solve bounds its change.
"""
import ast
import decimal
import hashlib
import json
import os
import struct
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
OPERANDS = os.path.join(HERE, 'drhoc_water_operands.npz')
REFERENCE = os.path.join(HERE, 'drhoc_water_reference.json')


def read_npy(blob):
    """(shape, list of floats) from a little-endian float64 C-order .npy blob."""
    if blob[:6] != b'\x93NUMPY':
        raise ValueError('not an .npy member')
    major = blob[6]
    if major == 1:
        size, start = struct.unpack('<H', blob[8:10])[0], 10
    else:
        size, start = struct.unpack('<I', blob[8:12])[0], 12
    header = ast.literal_eval(blob[start:start + size].decode('latin1'))
    if header['descr'] != '<f8' or header['fortran_order']:
        raise ValueError('expected little-endian float64 C order')
    data = blob[start + size:]
    count = 1
    for extent in header['shape']:
        count *= extent
    return tuple(header['shape']), list(struct.unpack(f'<{count}d', data[:8 * count])), data[:8 * count]


def operands():
    with zipfile.ZipFile(OPERANDS) as archive:
        (n, m), a, a_bytes = read_npy(archive.read('A.npy'))
        (nb,), b, b_bytes = read_npy(archive.read('b.npy'))
    if n != m or nb != n:
        raise ValueError('A must be n x n and b of length n')
    return n, a, b, hashlib.sha256(a_bytes).hexdigest(), hashlib.sha256(b_bytes).hexdigest()


def solve(n, a, b, digits):
    context = decimal.Context(prec=digits)
    rows = [[decimal.Decimal(a[i * n + j]) for j in range(n)] + [decimal.Decimal(b[i])] for i in range(n)]
    for k in range(n):
        pivot = max(range(k, n), key=lambda i: abs(rows[i][k]))
        if rows[pivot][k] == 0:
            raise ZeroDivisionError('singular stored matrix')
        rows[k], rows[pivot] = rows[pivot], rows[k]
        head = rows[k]
        for i in range(k + 1, n):
            factor = context.divide(rows[i][k], head[k])
            if factor:
                row = rows[i]
                for j in range(k + 1, n + 1):
                    row[j] = context.subtract(row[j], context.multiply(factor, head[j]))
    x = [decimal.Decimal(0)] * n
    for i in reversed(range(n)):
        total = rows[i][n]
        for j in range(i + 1, n):
            total = context.subtract(total, context.multiply(rows[i][j], x[j]))
        x[i] = context.divide(total, rows[i][i])
    return x


def main(mode):
    n, a, b, a_sha, b_sha = operands()
    x100, x120 = solve(n, a, b, 100), solve(n, a, b, 120)
    norm = max(abs(v) for v in x120)
    change = max(abs(p - q) for p, q in zip(x100, x120)) / norm
    digits = [format(v, '.99e') for v in x100]
    record = json.load(open(REFERENCE))
    found = dict(operand_sha256={'A': a_sha, 'b': b_sha}, n=n, solution_100_digits=digits,
                 relative_change_100_to_120_digits=format(change, '.3e'))
    if mode == '--write':
        record.update(found)
        record['generator_sha256'] = hashlib.sha256(open(__file__, 'rb').read()).hexdigest()
        with open(REFERENCE, 'w') as handle:
            json.dump(record, handle, indent=1)
            handle.write('\n')
        print('wrote', REFERENCE, found['relative_change_100_to_120_digits'])
        return 0
    same = all(record[key] == value for key, value in found.items())
    print('match' if same else 'MISMATCH', found['relative_change_100_to_120_digits'])
    return 0 if same else 1


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ('--write', '--check'):
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
