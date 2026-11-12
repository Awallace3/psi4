"""Instrumented full SAPT run; change only the declared Psi4 memory budget."""
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pr44"))
from case import cgroup_dir, dump, monitor, setup_scratch, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    parser.add_argument("memory_gb", type=int, choices=[800, 1200])
    parser.add_argument("output", type=Path)
    parser.add_argument("commit")
    args = parser.parse_args()
    root = HERE.parent.parent
    assert subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip() == args.commit
    assert not subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"])
    config_path = HERE.parent / "pr44-protein157/config.json"
    config = json.loads(config_path.read_text())
    config["memory_bytes"] = args.memory_gb * 10**9
    scratch, incoming = setup_scratch(f"memory-{args.memory_gb}")
    assert shutil.disk_usage(scratch).free >= config["minimum_free_scratch_bytes"]
    import numpy as np
    import psi4
    from psi4.driver.procrouting.sapt import sapt_mp2_terms
    stage = args.campaign / "build/stage"
    validated = json.loads((args.campaign / "build/validated.json").read_text())
    assert Path(psi4.core.__file__).resolve().is_relative_to(stage.resolve())
    assert validated["commit"] == args.commit
    assert sha(psi4.core.__file__) == validated["core_sha256"]
    assert int(os.environ["SLURM_CPUS_PER_TASK"]) == 192
    assert os.environ.get("PR44_MEMORY_TRACE") == "1"
    psi4.core.IOManager.shared_object().set_default_path(str(scratch) + "/")
    psi4.core.set_output_file(str(args.output / "psi4.out"), False)
    psi4.set_num_threads(192)
    psi4.set_memory(config["memory_bytes"])
    options = dict(config["options"], sapt_fdds_disp_df_algorithm="AUTO",
                   sapt_fdds_disp_ao_storage="AUTO", sapt_fdds_disp_split_monomers=True)
    psi4.set_options(options)
    xyz = HERE.parent / "pr44-protein157/protein157.xyz"
    atoms = xyz.read_text().splitlines()[2:]
    assert len(atoms) == 157
    geometry = ("0 1\n" + "\n".join(atoms[:118]) + "\n--\n0 1\n" + "\n".join(atoms[118:])
                + "\nunits angstrom\nsymmetry c1\nno_reorient\nno_com\n")
    (args.output / "geometry.psi4").write_text(geometry)
    molecule = psi4.geometry(geometry)
    molecule.update_geometry()
    assert psi4.core.BasisSet.build(molecule, "ORBITAL", options["basis"]).nbf() == 2491
    metadata = {
        "source_commit": args.commit, "production_parent": config["head"],
        "core": psi4.core.__file__, "core_sha256": validated["core_sha256"],
        "driver_sha256": sha(__file__), "config_sha256": sha(config_path),
        "geometry_sha256": sha(xyz), "config": config, "options": options,
        "scratch": str(scratch), "incoming_scratch": incoming, "host": platform.node(),
        "python": sys.executable, "job_id": os.environ["SLURM_JOB_ID"],
        "slurm": {k: v for k, v in os.environ.items() if k.startswith("SLURM_")},
        "packages": sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions()),
    }
    dump(args.output / "metadata.json", metadata)
    original = sapt_mp2_terms.df_fdds_dispersion

    def checkpoint(primary, auxiliary, cache, is_hybrid, x_alpha, *rest, **kwargs):
        assert primary.nbf() == 2491 and auxiliary.nbf() == 7384
        keys = ["Cocc_A", "Cvir_A", "Cocc_B", "Cvir_B", "D_A", "D_B",
                "eps_occ_A", "eps_vir_A", "eps_occ_B", "eps_vir_B"]
        temporary = args.output / "fdds-cache.tmp.npz"
        np.savez(temporary, **{k: cache[k].to_array() for k in keys})
        temporary.replace(args.output / "fdds-cache.npz")
        dump(args.output / "fdds-cache.json", {
            "time": time.time(), "source_commit": args.commit, "is_hybrid": is_hybrid,
            "x_alpha": x_alpha, "rest": rest, "kwargs": kwargs,
            "sha256": sha(args.output / "fdds-cache.npz"),
        })
        print("PR44_PHASE " + json.dumps({"phase": "FDDS", "event": "begin", "time": time.time()}), flush=True)
        values = original(primary, auxiliary, cache, is_hybrid, x_alpha, *rest, **kwargs)
        dump(args.output / "fdds-values.json", values)
        print("PR44_PHASE " + json.dumps({"phase": "FDDS", "event": "end", "time": time.time()}), flush=True)
        return values

    sapt_mp2_terms.df_fdds_dispersion = checkpoint
    context = multiprocessing.get_context("spawn")
    stop = context.Event()
    sampler = context.Process(target=monitor,
                              args=(args.output / "samples.jsonl", scratch, stop, os.getpid()), daemon=True)
    sampler.start()
    start = time.perf_counter()
    result = {"completed": False, "memory_gb": args.memory_gb, "source_commit": args.commit}
    try:
        energy = psi4.energy("sapt(dft)", molecule=molecule)
        reference = {
            "SAPT ELST ENERGY": -0.0075476756846910575,
            "SAPT EXCH ENERGY": 0.021148227122191283,
            "SAPT IND ENERGY": -0.0030370768130164056,
            "SAPT DISP ENERGY": -0.024184984850164085,
            "SAPT TOTAL ENERGY": -0.013621510225680264,
        }
        values = {k: psi4.variable(k) for k in reference}
        assert all(math.isfinite(v) for v in [energy, *values.values()])
        deltas = {k: values[k] - v for k, v in reference.items()}
        result.update(completed=True, energy_Eh=energy, variables=values,
                      delta_original_head_Eh=deltas,
                      parity_passed=all(abs(v) <= config["energy_tolerance_Eh"] for v in deltas.values()))
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
        result["final_cgroup"] = {"path": str(cg), "version": version}
        names = ["memory.current", "memory.peak", "memory.max", "memory.events", "memory.stat"]
        if cg is not None:
            result["final_cgroup"].update({name: (cg / name).read_text().strip()
                                          for name in names if (cg / name).exists()})
        dump(args.output / "result.json", result)
        print(json.dumps(result), flush=True)
    assert sampler.exitcode == 0 and (args.output / "samples.jsonl").stat().st_size > 0
    assert result["parity_passed"], "Numerical parity failed; resource artifacts preserved"
    psi4.core.clean()


if __name__ == "__main__":
    main()
