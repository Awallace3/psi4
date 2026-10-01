#!/usr/bin/env python3
# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Strict parser for the historical dynamic (imaginary-frequency) .pol dialect.

Independent of Psi4; consumes the complete bounded dense dialect. The original
capture CLI read a private, unshipped development archive and hash inventory and
has been removed; the parser and printed-frequency check are unchanged.
"""
from decimal import Decimal, localcontext
import re

LABELS = ('O', 'H1', 'H2')
NUMBER = re.compile(r'[+-]?\d+\.\d+(?:E[+-]\d+)?')


def printed_matches(omega, token):
    """Compare -omega**2 to the half-unit interval of the printed decimal."""
    with localcontext() as ctx:
        ctx.prec = 50
        value = Decimal(token)
        square = -Decimal.from_float(float(omega)) ** 2
        return abs(square - value) <= Decimal(5).scaleb(value.as_tuple().exponent - 1)


def parse_pol(text, distributed, index, frequency_squared):
    """Preserve raw headers and every decimal token; reject any extra content.

    index is the positive-node filename index (1..10); raw INDEX is index+1.
    L3 has no numeric site indices or CARTSPHER header: these are declared
    metadata, not invented source fields. Repeated frequency tokens must agree
    literally with the independently pinned header excerpt supplied by caller.
    """
    if type(index) is not int or not 1 <= index <= 10:
        raise ValueError('positive node index must be 1..10')
    if not isinstance(frequency_squared, str) or not NUMBER.fullmatch(frequency_squared):
        raise ValueError('invalid frequency token')
    if Decimal(frequency_squared) >= 0:
        raise ValueError('dynamic imaginary frequency requires negative FREQSQ')
    if len(text) > 100000:
        raise ValueError('input exceeds bounded file size')
    lines = text.splitlines()
    cursor = 0
    sections = []

    def take():
        nonlocal cursor
        if cursor >= len(lines):
            raise ValueError(f'premature EOF at line {cursor + 1}')
        line = lines[cursor]
        cursor += 1
        return line

    pairs = [(a, b) for a in range(3) for b in range(3)] if distributed else [(a, a) for a in range(3)]
    size = 25 if distributed else 15
    for a, b in pairs:
        line_number = cursor + 1
        header = take()
        if distributed:
            expected = (f'ALPHA INDEX {index+1:03} SITE-LABELS {LABELS[a]} {LABELS[b]} '
                        f'SITE-INDICES {a+1} {b+1} RANK 0 : 4 BY 0 : 4 '
                        f'FREQ2 {frequency_squared} CARTSPHER S')
        else:
            expected = (f'ALPHA H2O SITE-NAMES {LABELS[a]} {LABELS[b]} RANK 1 TO 3 '
                        f'INDEX {index+1} FREQSQ {frequency_squared}')
        if header.split() != expected.split():
            raise ValueError(f'invalid header/order/identity at line {line_number}: {header!r}')
        values = []
        for _ in range(size):
            tokens = take().split()
            if len(tokens) != size or any(len(t) > 40 or not NUMBER.fullmatch(t) for t in tokens):
                raise ValueError(f'invalid dense numerical row at line {cursor}')
            values.append(tokens)
        if distributed and take() != 'END':
            raise ValueError(f'missing END at line {cursor}')
        sections.append({'labels': [LABELS[a], LABELS[b]], 'site_indices': [a+1, b+1],
                         'site_indices_authority': 'raw_header' if distributed else 'ordered_SITE-NAMES',
                         'header_line': line_number, 'header': header, 'values': values})
    if take() != 'ENDFILE' or cursor != len(lines):
        raise ValueError('missing ENDFILE or trailing content')
    return sections
