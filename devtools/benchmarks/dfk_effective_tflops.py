#!/usr/bin/env python3
"""Effective DF-K TFLOP/s for the SAPT(DFT) J/K work, NVIDIA-style.

Vendor DF-K speedup claims are quoted as achieved FLOP rate: the dense
rectangular-DGEMM flop count of the exchange build divided by the exchange
wall time. This reproduces that metric from artifacts the benchmark already
keeps, so the PR's end-to-end SAPT speedups can be compared against a rate
rather than against another code's end-to-end time.

Per J/K call on a basis of nbf functions with naux fitting functions and
n_occ occupied orbitals summed over spin:

    K:  half transform  B[Q,mu,nu] -> B[Q,mu,i]     2 * naux * nbf^2 * n_occ
        assembly        K[mu,nu] = B[Q,mu,i] B[Q,nu,i]  2 * naux * nbf^2 * n_occ
    J:  d[Q] = B[Q,mu,nu] D[mu,nu] and back         2 * n_dens * 2 * naux * nbf^2

Only SCF-iteration calls are modeled; the covered fraction of the timer's
total J/K call count is reported so the omission is visible rather than
silent. cuEST prints per-call J and K kernel milliseconds, giving a
kernel-resolved rate; the CPU arm only has the aggregate `JK: JK` timer, so
its rate covers J and K together and is labeled accordingly.
"""
import argparse
import json
from pathlib import Path
import re
import statistics

from analyze_gpu_profile import flat_timers

NALPHA = re.compile(r"^\s*Nalpha\s+=\s+(\d+)\s*$")
NBETA = re.compile(r"^\s*Nbeta\s+=\s+(\d+)\s*$")
NBF = re.compile(r"^\s*Number of basis functions:\s+(\d+)\s*$")
ITER = re.compile(r"^\s*@\S*(RHF|RKS|UHF|UKS|ROHF) iter\b")
CUEST = re.compile(r"cuESTJK compute_JK:.*?J=\s*([\d.]+)ms\s+K=\s*([\d.]+)ms")


def scf_blocks(path):
    """Yield one record per SCF in a Psi4 output, in file order."""
    blocks, current, section = [], None, None
    for line in Path(path).read_text(errors="replace").splitlines():
        if match := NALPHA.match(line):
            current = {"nalpha": int(match[1]), "nbeta": None, "nbf": None,
                       "naux": None, "iterations": 0, "j_ms": 0.0, "k_ms": 0.0,
                       "jk_calls": 0}
            blocks.append(current)
            section = None
            continue
        if current is None:
            continue
        if match := NBETA.match(line):
            current["nbeta"] = int(match[1])
        elif "Primary Basis" in line:
            section = "primary"
        elif "Auxiliary Basis Set" in line:
            section = "auxiliary"
        elif match := NBF.match(line):
            if section == "primary" and current["nbf"] is None:
                current["nbf"] = int(match[1])
            elif section == "auxiliary" and current["naux"] is None:
                current["naux"] = int(match[1])
            section = None
        elif match := CUEST.search(line):
            current["j_ms"] += float(match[1])
            current["k_ms"] += float(match[2])
            current["jk_calls"] += 1
        elif ITER.match(line):
            current["iterations"] += 1
    return fill_aux(blocks)


def fill_aux(blocks):
    """Psi4 reuses a J/K object across SCFs and reprints the fitting basis only
    when it rebuilds one. The fitting basis is fixed by the primary basis and
    the atom set, so an unprinted size is the last one printed at the same nbf."""
    for index, block in enumerate(blocks):
        if block["naux"] is not None:
            continue
        neighbors = [other for other in blocks[:index][::-1] + blocks[index + 1:]
                     if other["nbf"] == block["nbf"] and other["naux"] is not None]
        if neighbors:
            block["naux"] = neighbors[0]["naux"]
            block["naux_inherited"] = True
    return blocks


def flops(block):
    """Dense-DGEMM DF J and K flop counts for one SCF's iteration calls."""
    nalpha, nbeta = block["nalpha"], block["nbeta"]
    unrestricted = nbeta is not None and nbeta != nalpha
    n_occ = nalpha + nbeta if unrestricted else nalpha
    n_dens = 2 if unrestricted else 1
    base = block["naux"] * block["nbf"] ** 2
    calls = block["jk_calls"] or block["iterations"]
    return {"k_flops": 4.0 * base * n_occ * calls,
            "j_flops": 4.0 * base * n_dens * calls,
            "calls": calls, "n_occ": n_occ, "n_dens": n_dens}


def merge_aux(target, donor):
    """cuEST does not print the fitting-basis size; take it from the paired
    CPU arm, which runs the identical SCF sequence on identical bases."""
    if len(target) != len(donor):
        raise ValueError(f"SCF count mismatch: {len(target)} vs {len(donor)}")
    for block, other in zip(target, donor):
        if (block["nbf"], block["nalpha"], block["nbeta"]) != (
                other["nbf"], other["nalpha"], other["nbeta"]):
            raise ValueError("Paired SCF blocks differ in dimension")
        if block["naux"] is None:
            block["naux"] = other["naux"]
            block["naux_inherited"] = True
    return target


def case_rate(directory, donor=None):
    blocks = scf_blocks(directory / "psi4.out")
    if donor is not None:
        blocks = merge_aux(blocks, donor)
    missing = [i for i, block in enumerate(blocks)
               if None in (block["nbf"], block["naux"])]
    if missing:
        raise ValueError(f"Incomplete SCF dimensions in {directory}: {missing}")
    counts = [flops(block) for block in blocks]
    timers = flat_timers(directory / "timer.dat")
    jk = timers.get("JK: JK", {"wall_s": 0.0, "calls": 0})
    modeled_calls = sum(count["calls"] for count in counts)
    k_flops = sum(count["k_flops"] for count in counts)
    j_flops = sum(count["j_flops"] for count in counts)
    k_ms = sum(block["k_ms"] for block in blocks)
    j_ms = sum(block["j_ms"] for block in blocks)
    record = {"case": directory.name, "scf_count": len(blocks),
              "modeled_jk_calls": modeled_calls, "timer_jk_calls": jk["calls"],
              "call_coverage": modeled_calls / jk["calls"] if jk["calls"] else None,
              "k_gflop": k_flops / 1e9, "j_gflop": j_flops / 1e9,
              "jk_timer_wall_s": jk["wall_s"],
              "jk_timer_tflops": (k_flops + j_flops) / jk["wall_s"] / 1e12
              if jk["wall_s"] else None}
    if k_ms:
        record.update({"k_kernel_s": k_ms / 1e3, "j_kernel_s": j_ms / 1e3,
                       "k_kernel_tflops": k_flops / (k_ms / 1e3) / 1e12,
                       "j_kernel_tflops": j_flops / (j_ms / 1e3) / 1e12})
    result = directory / "result.json"
    if result.exists():
        payload = json.loads(result.read_text())
        record.update({key: payload.get(key) for key in
                       ("mode", "threads", "wall_s", "grac_compute", "ok")})
        record["jk_fraction_of_total"] = (
            jk["wall_s"] / payload["wall_s"] if payload.get("wall_s") else None)
    return record


def pair_donor(directory):
    """The matching CPU case directory, which carries the fitting-basis size."""
    name = directory.name
    if "-gpu-" not in name:
        return None
    for candidate in sorted(directory.parent.glob(name.replace("-gpu-", "-cpu-")[:-1] + "*")):
        if (candidate / "psi4.out").exists():
            return scf_blocks(candidate / "psi4.out")
    for candidate in sorted(directory.parent.glob("*-cpu*")):
        if (candidate / "psi4.out").exists():
            try:
                blocks = scf_blocks(candidate / "psi4.out")
                merge_aux([dict(b) for b in scf_blocks(directory / "psi4.out")], blocks)
                return blocks
            except ValueError:
                continue
    return None


def summarize(records):
    """Median rate per (case family, mode, threads) across repeats."""
    groups = {}
    for record in records:
        family = re.sub(r"-\d+$", "", record["case"])
        groups.setdefault(family, []).append(record)
    rows = []
    for family, entries in sorted(groups.items()):
        rate = [e["k_kernel_tflops"] for e in entries if e.get("k_kernel_tflops")]
        aggregate = [e["jk_timer_tflops"] for e in entries if e.get("jk_timer_tflops")]
        rows.append({"family": family, "repeats": len(entries),
                     "k_gflop": entries[0]["k_gflop"],
                     "median_k_kernel_tflops": statistics.median(rate) if rate else None,
                     "median_jk_timer_tflops": statistics.median(aggregate) if aggregate else None,
                     "median_jk_wall_s": statistics.median(
                         [e["jk_timer_wall_s"] for e in entries])})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", help="directory of per-case subdirectories")
    parser.add_argument("--output", help="write JSON here")
    args = parser.parse_args()
    root = Path(args.results)
    records = []
    for directory in sorted(root.iterdir()):
        if not (directory / "psi4.out").exists() or not (directory / "timer.dat").exists():
            continue
        try:
            records.append(case_rate(directory, pair_donor(directory)))
        except (ValueError, KeyError) as error:
            records.append({"case": directory.name, "error": f"{type(error).__name__}: {error}"})
    payload = {"cases": records, "summary": summarize(
        [r for r in records if "error" not in r])}
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
