# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Portable read-only printed-grid parser. No engine, recipe or scratch paths."""
from dataclasses import dataclass
from decimal import Decimal
import re

from rounding import e15_8_interval


@dataclass(frozen=True)
class PrintedGrid:
    headers: tuple
    sites: tuple
    tokens: tuple
    intervals: tuple


def read_printed_grid(path):
    """Read explicit SITE blocks and four canonical e15.8 fields per point.

    Preserve token ordering and closed print intervals. Do not infer point
    correspondence, units, radial/angular counts, engine exit or weight policy.
    Those declarations and file authentication belong to the caller.
    """
    headers, sites, rows, intervals = [], [], [], []
    counts, labels = [], set()
    with open(path) as stream:
        for line in stream:
            text = line.strip()
            if not text:
                continue
            if text.startswith("#"):
                header = text[1:].strip()
                match = re.fullmatch(r"SITE ([1-9][0-9]*) NAME (\S+)", header)
                if header.startswith("SITE"):
                    if (match is None or int(match[1]) != len(sites)+1
                            or match[2] in labels or counts and counts[-1] == 0):
                        raise ValueError("invalid, duplicate or empty SITE block")
                    sites.append(match[2])
                    labels.add(match[2])
                    counts.append(0)
                else:
                    headers.append(header)
                continue
            fields = tuple(text.split())
            if not sites or len(fields) != 4:
                raise ValueError("four fields inside a SITE block required")
            bounds = tuple(e15_8_interval(token) for token in fields)
            if not all(Decimal(token.replace("D", "E")).is_finite() for token in fields):
                raise ValueError("finite fields required")
            rows.append(fields)
            intervals.append(bounds)
            counts[-1] += 1
    if not sites or not counts[-1]:
        raise ValueError("nonempty SITE blocks required")
    return PrintedGrid(tuple(headers), tuple(zip(sites, counts)), tuple(rows), tuple(intervals))
