#!/usr/bin/env python3
"""Present several campaign job trees as one directory of case results.

A preempted job leaves its remaining cases to a follow-up job, which writes its
own run directory. The analysis tools each take a single results directory, so
the two trees have to be presented as one. Copying them together would duplicate
hundreds of megabytes and, worse, would make the merged tree the artifact rather
than the jobs; this builds a directory of symlinks instead, so every case still
points back at the job that produced it.

A case left behind by a killed run is a directory with a `psi4.out` and no
`result.json`. That stub must lose to the completed rerun, and it is the only
kind of collision that resolves silently: two directories both claiming a
finished measurement of the same case is a mistake somewhere upstream, and
picking one would hide it.
"""
import argparse
import os
from pathlib import Path


def complete(directory):
    return (directory / "result.json").exists()


def resolve(trees):
    """Map case name -> chosen source directory, later trees taking precedence.

    Precedence applies only between a completed case and a stub. Two completed
    directories for the same case raise, rather than one silently winning.
    """
    chosen = {}
    for tree in trees:
        for directory in sorted(Path(tree).iterdir()):
            if not directory.is_dir():
                continue
            previous = chosen.get(directory.name)
            if previous is None:
                chosen[directory.name] = directory
            elif complete(previous) and complete(directory):
                raise SystemExit(f"{directory.name}: completed in both {previous.parent} "
                                 f"and {directory.parent}; refusing to pick one")
            elif complete(directory):
                chosen[directory.name] = directory
    return chosen


def link(chosen, target):
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    for existing in target.iterdir():
        if existing.is_symlink():
            existing.unlink()
        elif existing.is_dir():
            # A real case directory here means the target is somebody's job tree.
            raise SystemExit(f"{existing} is a real directory; {target} is not a merged tree")
        # Plain files are left alone: the analysis tools write campaign.json and
        # COMPLETE.json into the results directory and overwrite them themselves.
    for name, source in sorted(chosen.items()):
        os.symlink(source.resolve(), target / name)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trees", nargs="+", type=Path,
                        help="results directories, in precedence order (last wins)")
    parser.add_argument("--output", required=True, type=Path,
                        help="directory of symlinks to create; existing links are replaced")
    args = parser.parse_args()
    chosen = resolve(args.trees)
    link(chosen, args.output)
    print(f"{len(chosen)} cases linked into {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
