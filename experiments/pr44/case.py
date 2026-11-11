"""One attributable SAPT dimer run; never turn an exception into success."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import resource
import subprocess
import time
import traceback


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def cgroup_dir():
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        _, controllers, relative = line.split(":", 2)
        if not controllers:
            return Path("/sys/fs/cgroup") / relative.lstrip("/"), 2
        if "memory" in controllers.split(","):
            return Path("/sys/fs/cgroup/memory") / relative.lstrip("/"), 1
    return None, None


def monitor(output, scratch, stop, pid):
    cg, version = cgroup_dir()
    names = (["memory.current", "memory.peak", "memory.max", "memory.events", "memory.stat"]
             if version == 2 else
             ["memory.usage_in_bytes", "memory.max_usage_in_bytes", "memory.limit_in_bytes",
              "memory.failcnt", "memory.stat"])
    with output.open("x") as handle:
        while not stop.is_set():
            sample = {"time": time.time(), "cgroup": str(cg), "cgroup_version": version}
            try:
                sample["rss_bytes"] = int(Path(f"/proc/{pid}/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
                sizes = [p.stat() for p in scratch.rglob("*") if p.is_file()]
                sample["scratch_logical_bytes"] = sum(s.st_size for s in sizes)
                sample["scratch_allocated_bytes"] = sum(s.st_blocks * 512 for s in sizes)
                if cg is not None:
                    for name in names:
                        path = cg / name
                        if path.exists():
                            sample[name] = path.read_text().strip()
            except (FileNotFoundError, PermissionError, ProcessLookupError) as error:
                sample["sampling_error"] = repr(error)
            handle.write(json.dumps(sample) + "\n")
            handle.flush()
            stop.wait(2.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    parser.add_argument("index", type=int)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    harness = Path(__file__).resolve().parent
    config = json.loads((harness / "config.json").read_text())
    case = config["cases"][args.index]
    arm = "base" if case["arm"] == "base" else "head"
    source = args.campaign / "sources" / arm
    stage = args.campaign / "builds" / arm / "stage"
    expected = config[arm]
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    assert actual == expected, (actual, expected)
    assert not subprocess.check_output(["git", "-C", str(source), "diff", "HEAD"])
    import psi4
    assert Path(psi4.core.__file__).resolve().is_relative_to(stage.resolve()), psi4.core.__file__
    build = json.loads((args.campaign / "builds" / arm / "validated.json").read_text())
    assert build["commit"] == expected
    native_hash = sha(psi4.core.__file__)
    assert native_hash == build["core_sha256"]
    scratch = Path(os.environ["PSI_SCRATCH"]).resolve()
    assert scratch.is_dir() and str(scratch) == os.environ["TMPDIR"] == os.environ["SCRATCH"]
    geometry_path = harness / f'{case["system"]}.xyz'
    lines = geometry_path.read_text().splitlines()
    atoms = lines[2:]
    assert len(atoms) == int(lines[0])
    split = {"peptide": 12, "nanotube": 6}[case["system"]]
    geometry = ("0 1\n" + "\n".join(atoms[:split]) + "\n--\n0 1\n" +
                "\n".join(atoms[split:]) + "\nunits angstrom\nsymmetry c1\nno_reorient\nno_com\n")
    options = dict(config["options"])
    if arm == "head":
        options.update(sapt_fdds_disp_df_algorithm="AUTO",
                       sapt_fdds_disp_ao_storage="OUT_OF_CORE" if case["arm"] == "disk" else "AUTO",
                       sapt_fdds_disp_split_monomers=True)
    metadata = {
        "case": case, "options": options, "source_commit": expected,
        "core": psi4.core.__file__, "core_sha256": native_hash,
        "config_sha256": sha(harness / "config.json"), "input_sha256": sha(geometry_path),
        "driver_sha256": sha(__file__), "host": platform.node(), "python": platform.python_version(),
        "harness_commit": subprocess.check_output(["git", "-C", str(harness), "rev-parse", "HEAD"], text=True).strip(),
        "threads": config["threads"], "scratch": str(scratch),
        "slurm": {k: v for k, v in os.environ.items() if k.startswith("SLURM_")},
        "packages": sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions()),
    }
    dump(args.output / "metadata.json", metadata)
    (args.output / "geometry.psi4").write_text(geometry)
    psi4.core.set_output_file(str(args.output / "psi4.out"), False)
    psi4.set_num_threads(config["threads"])
    psi4.set_memory(case["memory_gib"] * 1024**3)
    psi4.set_options(options)
    molecule = psi4.geometry(geometry)
    molecule.update_geometry()
    basis = psi4.core.BasisSet.build(molecule, "ORBITAL", options["basis"])
    assert basis.nbf() == {"peptide": 260, "nanotube": 574}[case["system"]]
    metadata["nbf"] = basis.nbf()
    metadata["naux_scf"] = psi4.core.BasisSet.build(
        molecule, "DF_BASIS_SCF", options["df_basis_scf"], "JKFIT", options["basis"]).nbf()
    metadata["naux_sapt"] = psi4.core.BasisSet.build(
        molecule, "DF_BASIS_SAPT", options["df_basis_sapt"], "RIFIT", options["basis"]).nbf()
    dump(args.output / "metadata.json", metadata)
    # FDDS's native bindings retain the GIL: a Python thread would miss exactly
    # the long transform/QR peaks this experiment needs to observe.
    context = multiprocessing.get_context("spawn")
    stop = context.Event()
    sampler = context.Process(target=monitor, args=(args.output / "samples.jsonl", scratch, stop, os.getpid()), daemon=True)
    sampler.start()
    start = time.perf_counter()
    result = {"completed": False, "case": case, "source_commit": expected}
    try:
        energy = psi4.energy("sapt(dft)", molecule=molecule)
        values = {key: psi4.variable(key) for key in
                  ["SAPT ELST ENERGY", "SAPT EXCH ENERGY", "SAPT IND ENERGY", "SAPT DISP ENERGY", "SAPT TOTAL ENERGY"]}
        assert all(math.isfinite(value) for value in [energy, *values.values()])
        result.update(completed=True, energy_Eh=energy, variables=values)
    except BaseException:
        result["exception"] = traceback.format_exc()
        raise
    finally:
        result["seconds"] = time.perf_counter() - start
        result["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        psi4.core.flush_outfile()
        stop.set()
        sampler.join(timeout=5)
        if sampler.is_alive():
            sampler.terminate()
            sampler.join()
        result["sampler_exitcode"] = sampler.exitcode
        dump(args.output / "result.json", result)
        print(json.dumps(result), flush=True)
    psi4.core.clean()


if __name__ == "__main__":
    main()
