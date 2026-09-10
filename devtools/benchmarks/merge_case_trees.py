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

Pooling also assumes the trees ran at the same host speed, and that assumption
has already failed once: two gpu-h200 allocations of the same CPU model, same
`core.so`, same geometries differed by 3.2x on identical phases. Merging such
trees produces a results directory whose medians mix two machines. So the merge
consults each tree's host canary and refuses by default; `--allow-host-mismatch`
takes a reason and records it into the merged tree, so the caveat travels with
the artifact rather than living in a shell flag nobody reads later.
"""
import argparse
import os
from pathlib import Path

import host_speed

MISMATCH_NOTE = "HOST_MISMATCH.txt"


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


def check_hosts(trees, allow):
    """Refuse to pool trees whose hosts are not known to have matched.

    An uncertified tree is refused for the same reason a mismatched one is: the
    deficit this guards against was invisible in exactly that state, and
    treating "unmeasured" as "fine" is what let it into a published table.
    """
    payload = host_speed.compare(trees)
    if payload["verdict"] == "matched" or allow:
        return payload
    detail = ", ".join(f"{key} {value:.2f}x"
                       for key, value in sorted(payload["spread"].items()))
    raise SystemExit(
        f"host speed across these trees is {payload['verdict']}"
        + (f" ({detail})" if detail else "")
        + "; pooling them mixes machines into one median. Re-run on one host, or "
          "pass --allow-host-mismatch with a reason that will be recorded in the "
          "merged tree.")


def link(chosen, target, host_note=None):
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    stale = target / MISMATCH_NOTE
    if stale.exists():
        stale.unlink()
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
    if host_note:
        (target / MISMATCH_NOTE).write_text(host_note.rstrip() + "\n")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trees", nargs="+", type=Path,
                        help="results directories, in precedence order (last wins)")
    parser.add_argument("--output", required=True, type=Path,
                        help="directory of symlinks to create; existing links are replaced")
    parser.add_argument("--allow-host-mismatch", metavar="REASON",
                        help="pool trees whose host speeds differ or are unmeasured, "
                             "recording REASON in the merged tree")
    args = parser.parse_args()
    payload = check_hosts(args.trees, args.allow_host_mismatch)
    chosen = resolve(args.trees)
    note = None
    if args.allow_host_mismatch and payload["verdict"] != "matched":
        note = (f"Host speed across the merged trees: {payload['verdict']} "
                f"(worst pairwise ratio {payload['worst_ratio']:.2f}x).\n"
                f"Pooled anyway because: {args.allow_host_mismatch}\n"
                "Timings in this directory may mix machines; a median across them "
                "is not a single-host measurement.")
    link(chosen, args.output, note)
    print(f"{len(chosen)} cases linked into {args.output} "
          f"(host speed: {payload['verdict']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
