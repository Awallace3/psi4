#!/usr/bin/env python3
"""Protein157 cuEST DF-JK pilot: query capacity first; never fall back to DIRECT.

Workspace queries allocate only basis/pair metadata, not the huge DF plan.
Sampler-free timings and sampled-memory diagnostics are separate fresh processes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import grid_evidence as grid

CANDIDATES = ["6-31+g**", "jun-cc-pvdz", "heavy-aug-cc-pvdz", "aug-cc-pvdz", "def2-tzvp"]
GEOMETRIES = Path(__file__).with_name("protein157_geometries.json")
# Hash the actual geometry text, rather than reusing the old campaign-level
# digest (which is not the digest of its archived result["geometry"] bytes).
GEOMETRY_SHA256 = "a1e9c8949bb5950fdb17c07ab718cead826835f042a6b6e94c5cb7dcf75b6b60"
ARMS = ["fixed99", "fixed75", "coarse75"]


def geometry():
    text = json.loads(GEOMETRIES.read_text())["protein157"]
    assert hashlib.sha256(text.encode()).hexdigest() == GEOMETRY_SHA256
    return text


def export_basis(mol, basis, auxiliary, path):
    """Identical shell ordering, normalization and Bohr coordinates to BasisSet::cuest_initialize."""
    with Path(path).open("w") as stream:
        stream.write(f"{mol.natom()}\n")
        for atom in range(mol.natom()):
            stream.write(" ".join(format(mol.geometry().get(atom, j), ".17g")
                                  for j in range(3)) + "\n")
        for b in (basis, auxiliary):
            for atom in range(mol.natom()):
                count = b.nshell_on_center(atom)
                stream.write(f"{count}\n")
                for shell in range(count):
                    s = b.shell(b.shell_on_center(atom, shell))
                    stream.write(f"{int(s.is_pure())} {s.am} {s.nprimitive}\n")
                    stream.write(" ".join(format(s.exp(k), ".17g")
                                          for k in range(s.nprimitive)) + "\n")
                    stream.write(" ".join(format(s.coef(k), ".17g")
                                          for k in range(s.nprimitive)) + "\n")


def select_candidate(rows, headroom_gib=16.):
    eligible = []
    for r in rows:
        if not r.get("ok"):
            continue
        q = r["query"]
        required = q["df_persistent_bytes"] + q["df_construction_temporary_bytes"]
        available = q["device_free_bytes"] - headroom_gib * 2**30
        if q["df_plan_allocated"] is not False:
            raise ValueError("Capacity query allocated a DF plan")
        if required <= available:
            eligible.append((required, r))
    # Prefer the most memory-demanding candidate that leaves explicit headroom.
    return max(eligible, key=lambda item: item[0])[1] if eligible else None


def query(args):
    import psi4
    assert Path(psi4.__file__).resolve().parent == Path(args.expected_package).resolve()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    psi4.core.set_output_file(str(out / "basis-build.out"), False)
    psi4.set_num_threads(args.threads)
    mol = psi4.geometry(geometry())
    rows = []
    for name in args.candidates:
        row = dict(basis=name, auxiliary="def2-universal-jkfit", ok=False)
        try:
            b = psi4.core.BasisSet.build(mol, "ORBITAL", name, puream=True)
            a = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", row["auxiliary"],
                                       "JKFIT", name, puream=True)
            row.update(nbf=b.nbf(), naux=a.nbf())
            data = out / (name.replace("*", "s") + ".basis.txt")
            export_basis(mol, b, a, data)
            row["basis_input_sha256"] = hashlib.sha256(data.read_bytes()).hexdigest()
            process = subprocess.run([args.query_executable, str(data)], capture_output=True,
                                     text=True, timeout=300)
            (out / (data.stem + ".query.log")).write_text(process.stdout + process.stderr)
            if process.returncode:
                raise RuntimeError(f"Query exit {process.returncode}: {process.stderr[-1500:]}")
            row["query"] = json.loads(process.stdout.strip().splitlines()[-1])
            row["ok"] = True
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        grid.atomic(out / "queries.json", dict(rows=rows, complete=False))
    selected = select_candidate(rows, args.headroom_gib)
    result = dict(rows=rows, complete=True, selected=selected,
                  headroom_gib=args.headroom_gib, geometry_sha256=GEOMETRY_SHA256,
                  scientific_commit=args.scientific_commit,
                  query_executable_sha256=hashlib.sha256(
                      Path(args.query_executable).read_bytes()).hexdigest(),
                  job_id=os.getenv("SLURM_JOB_ID"), diagnostic_only=True,
                  eligible_for_speedup_bars=False,
                  caveat="Query excludes XC and other simultaneous lifetimes; actual job must be sampled.")
    grid.atomic(out / "queries.json", result)
    psi4.core.close_outfile()
    return 0 if selected else 1


def worker(args):
    grid.SYSTEMS["protein157"] = args.basis
    args.system = "protein157"
    args.geometry_file = GEOMETRIES
    args.ints_tolerance = 1.e-12
    result = grid.worker(args)
    path = Path(args.output) / "result.json"
    row = json.loads(path.read_text())
    row.update(auxiliary="def2-universal-jkfit", eligible_for_speedup_bars=not args.diagnostic,
               diagnostic_only=args.diagnostic)
    grid.atomic(path, row)
    return result


def campaign(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    selection = json.loads(Path(args.selection).read_text())
    assert selection["complete"] and selection["scientific_commit"] == args.scientific_commit
    assert selection["geometry_sha256"] == GEOMETRY_SHA256
    selected = selection["selected"]
    if not selected or not selected.get("ok"):
        raise ValueError("No capacity-checked basis")
    args.basis, args.expected_nbf = selected["basis"], selected["nbf"]
    grid.SYSTEMS["protein157"] = args.basis
    manifest = dict(job_id=os.getenv("SLURM_JOB_ID"), selected=selected,
                    source_selection_sha256=hashlib.sha256(Path(args.selection).read_bytes()).hexdigest(),
                    scientific_commit=args.scientific_commit, expected_package=args.expected_package,
                    geometry_sha256=GEOMETRY_SHA256, threads=args.threads, memory=args.memory,
                    variant=args.variant, arms=ARMS, repeats=args.repeats,
                    memory_sampler_for_timing=False, records=[],
                    harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    grid_worker_sha256=hashlib.sha256(Path(grid.__file__).read_bytes()).hexdigest())
    grid.atomic(out / "campaign.json", manifest)
    rows = []
    for repeat in range(args.repeats):
        order = ARMS[repeat % len(ARMS):] + ARMS[:repeat % len(ARMS)]
        for arm in order:
            directory = out / f"protein157-{arm}-{repeat + 1}"
            directory.mkdir()
            with tempfile.TemporaryDirectory(prefix="protein157-grid-", dir=os.environ["TMPDIR"]) as scratch:
                env = dict(os.environ, TMPDIR=scratch, SCRATCH=scratch, PSI_SCRATCH=scratch)
                command = [sys.executable, str(Path(__file__).resolve()), "worker",
                           "--output", str(directory), "--basis", args.basis,
                           "--expected-nbf", str(args.expected_nbf), "--arm", arm,
                           "--repeat", str(repeat + 1), "--variant", args.variant,
                           "--threads", str(args.threads), "--memory", args.memory,
                           "--expected-package", args.expected_package,
                           "--scientific-commit", args.scientific_commit]
                with (directory / "console.log").open("w") as log:
                    process = subprocess.run(command, cwd=directory, env=env, stdout=log,
                                             stderr=subprocess.STDOUT, timeout=3600)
            result_path = directory / "result.json"
            row = json.loads(result_path.read_text()) if result_path.exists() else dict(
                ok=False, system="protein157", arm=arm, repeat=repeat + 1,
                error="No worker result", exit_code=process.returncode)
            if process.returncode:
                row.update(ok=False, exit_code=process.returncode)
            rows.append(row)
            manifest["records"].append(str(result_path))
            grid.atomic(out / "campaign.json", manifest)
            grid.atomic(out / "evidence.json",
                        grid.summarize(rows, repeats=args.repeats, arms=ARMS))
            if not row.get("ok"):
                return 1
    evidence = grid.summarize(rows, repeats=args.repeats, arms=ARMS)
    assert len(evidence["groups"]) == 1 and len(evidence["groups"][0]["arms"]) == 3
    grid.atomic(out / "COMPLETE.json",
                dict(records=len(rows), completed=True,
                     accuracy_pass=all(a["agreement_pass"] for a in evidence["groups"][0]["arms"])))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["query", "worker", "campaign"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-package", required=True)
    parser.add_argument("--scientific-commit", required=True)
    parser.add_argument("--query-executable")
    parser.add_argument("--candidates", nargs="+", default=CANDIDATES)
    parser.add_argument("--headroom-gib", type=float, default=16.)
    parser.add_argument("--selection")
    parser.add_argument("--basis")
    parser.add_argument("--expected-nbf", type=int)
    parser.add_argument("--arm", choices=ARMS, default="coarse75")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory", default="112 GiB")
    parser.add_argument("--variant", choices=["host", "cuda"], default="cuda")
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()
    geometry()
    return dict(query=query, worker=worker, campaign=campaign)[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
