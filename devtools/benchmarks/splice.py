#!/usr/bin/env python3
"""Fill `<!-- NAME -->` markers in README.template.md from generated tables.

README.md is a narrative with generated tables in it. Keeping the tables by hand
guarantees they drift from the JSON the moment anything is re-run, so the
narrative lives in the template and the numbers are substituted here. Every
marker must be filled: an unfilled one is an error, not an empty section, so a
table that failed to generate cannot silently disappear from the report.
"""
import argparse
from pathlib import Path
import re

MARKER = re.compile(r"^<!-- ([A-Z0-9]+) -->$", re.M)


def splice(template, blocks):
    missing = {m.group(1) for m in MARKER.finditer(template)} - set(blocks)
    if missing:
        raise SystemExit(f"no content for marker(s): {', '.join(sorted(missing))}")
    unused = set(blocks) - {m.group(1) for m in MARKER.finditer(template)}
    if unused:
        raise SystemExit(f"content given for absent marker(s): {', '.join(sorted(unused))}")
    return MARKER.sub(lambda m: blocks[m.group(1)].strip(), template)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--block", action="append", default=[], metavar="NAME=FILE",
                        help="fill <!-- NAME --> with FILE's contents; repeat per marker")
    args = parser.parse_args()
    blocks = {}
    for spec in args.block:
        name, _, path = spec.partition("=")
        if not name or not path:
            raise SystemExit(f"expected NAME=FILE, got {spec!r}")
        blocks[name] = Path(path).read_text()
    args.output.write_text(splice(args.template.read_text(), blocks))
    print(f"wrote {args.output} ({len(blocks)} blocks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
