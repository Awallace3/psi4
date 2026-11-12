"""Paired protein157 resource measurement using the exact PR44 base/head."""
import argparse
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
import traceback

HARNESS = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS.parent / "pr44"))
from case import cgroup_dir, dump, monitor, setup_scratch, sha


def git(*args, directory=HARNESS):
    return subprocess.check_output(["git", "-C", str(directory), *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=["base", "head"])
    parser.add_argument("output", type=Path)
    parser.add_argument("harness_commit")
    args = parser.parse_args()
    config = json.loads((HARNESS / "config.json").read_text())
    assert git("rev-parse", "HEAD") == args.harness_commit
    assert not git("status", "--porcelain")
    build_root = Path(config["build_campaign"])
    source = build_root / "sources" / args.arm
    stage = build_root / "builds" / args.arm / "stage"
    assert git("rev-parse", "HEAD", directory=source) == config[args.arm]
    assert not git("diff", "HEAD", directory=source)
    assert int(os.environ["SLURM_CPUS_PER_TASK"]) == config["threads"]
    scratch, incoming = setup_scratch(f"protein157-{args.arm}")
    disk = shutil.disk_usage(scratch)
    assert disk.free >= config["minimum_free_scratch_bytes"], disk
    import psi4
    assert Path(psi4.core.__file__).resolve().is_relative_to(stage.resolve())
    validated = json.loads((build_root / "builds" / args.arm / "validated.json").read_text())
    native_hash = sha(psi4.core.__file__)
    assert validated["commit"] == config[args.arm]
    assert native_hash == validated["core_sha256"]
    psi4.core.IOManager.shared_object().set_default_path(str(scratch) + "/")
    geometry_path = HARNESS / "protein157.xyz"
    lines = geometry_path.read_text().splitlines()
    assert int(lines[0]) == len(lines[2:]) == 157
    atoms, split = lines[2:], config["fragment_a_atoms"]
    geometry = ("0 1\n" + "\n".join(atoms[:split]) + "\n--\n0 1\n"
                + "\n".join(atoms[split:])
                + "\nunits angstrom\nsymmetry c1\nno_reorient\nno_com\n")
    options = dict(config["options"])
    if args.arm == "head":
        options.update(sapt_fdds_disp_df_algorithm="AUTO", sapt_fdds_disp_ao_storage="AUTO",
                       sapt_fdds_disp_split_monomers=True)
    metadata = {
        "arm": args.arm, "source_commit": config[args.arm], "harness_commit": args.harness_commit,
        "core": psi4.core.__file__, "core_sha256": native_hash,
        "config": config, "options": options, "config_sha256": sha(HARNESS / "config.json"),
        "geometry_sha256": sha(geometry_path), "driver_sha256": sha(__file__),
        "monitor_sha256": sha(HARNESS.parent / "pr44" / "case.py"),
        "scratch": str(scratch), "incoming_scratch": incoming, "scratch_free_bytes": disk.free,
        "host": platform.node(), "python": sys.executable, "python_version": platform.python_version(),
        "slurm": {k: v for k, v in os.environ.items() if k.startswith("SLURM_")},
        "packages": sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions()),
    }
    dump(args.output / "metadata.json", metadata)
    (args.output / "geometry.psi4").write_text(geometry)
    psi4.core.set_output_file(str(args.output / "psi4.out"), False)
    psi4.set_num_threads(config["threads"])
    psi4.set_memory(config["memory_bytes"])
    psi4.set_options(options)
    molecule = psi4.geometry(geometry)
    molecule.update_geometry()
    metadata["nbf"] = psi4.core.BasisSet.build(molecule, "ORBITAL", options["basis"]).nbf()
    for field, option, fit in [("naux_scf", "df_basis_scf", "JKFIT"),
                               ("naux_fdds", "df_basis_mp2", "RIFIT")]:
        metadata[field] = psi4.core.BasisSet.build(
            molecule, option.upper(), options[option], fit, options["basis"]).nbf()
    for field in ["nbf", "naux_scf", "naux_fdds"]:
        assert metadata[field] == config[field], (field, metadata[field])
    dump(args.output / "metadata.json", metadata)
    print(json.dumps({k: metadata[k] for k in ["arm", "nbf", "naux_scf", "naux_fdds", "scratch"]}), flush=True)
    context = multiprocessing.get_context("spawn")
    stop = context.Event()
    sampler = context.Process(target=monitor,
                              args=(args.output / "samples.jsonl", scratch, stop, os.getpid()),
                              daemon=True)
    sampler.start()
    start = time.perf_counter()
    result = {"completed": False, "arm": args.arm, "source_commit": config[args.arm]}
    try:
        energy = psi4.energy("sapt(dft)", molecule=molecule)
        values = {k: psi4.variable(k) for k in [
            "SAPT ELST ENERGY", "SAPT EXCH ENERGY", "SAPT IND ENERGY", "SAPT DISP ENERGY", "SAPT TOTAL ENERGY"]}
        assert all(math.isfinite(v) for v in [energy, *values.values()])
        result.update(completed=True, energy_Eh=energy, variables=values)
    except BaseException:
        result["exception"] = traceback.format_exc()
        raise
    finally:
        result["seconds"] = time.perf_counter() - start
        result["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        psi4.core.flush_outfile()
        stop.set()
        sampler.join(timeout=10)
        if sampler.is_alive():
            sampler.terminate()
            sampler.join()
        result["sampler_exitcode"] = sampler.exitcode
        cg, version = cgroup_dir()
        names = (["memory.current", "memory.peak", "memory.max", "memory.events", "memory.stat"]
                 if version == 2 else
                 ["memory.usage_in_bytes", "memory.max_usage_in_bytes", "memory.limit_in_bytes",
                  "memory.failcnt", "memory.stat"])
        result["final_cgroup"] = {"cgroup": str(cg), "cgroup_version": version}
        if cg is not None:
            result["final_cgroup"].update({name: (cg / name).read_text().strip()
                                          for name in names if (cg / name).exists()})
        dump(args.output / "result.json", result)
        print(json.dumps(result), flush=True)
    assert sampler.exitcode == 0, "Resource sampler failed; do not accept this measurement"
    assert (args.output / "samples.jsonl").stat().st_size > 0
    psi4.core.clean()


if __name__ == "__main__":
    main()
