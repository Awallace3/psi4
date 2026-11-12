"""Same-SCF-cache FDDS replay to separate algorithm effects from run variation."""
import gc
import json
import os
from pathlib import Path
import sys
import time

from case import dump, setup_scratch, sha

campaign = Path(sys.argv[1])
output = Path(sys.argv[3])
harness = Path(__file__).resolve().parent
scratch, incoming = setup_scratch("replay")

import psi4
from psi4.driver.procrouting.sapt import sapt_mp2_terms

build = json.loads((campaign / "builds/head/validated.json").read_text())
assert Path(psi4.core.__file__).resolve().is_relative_to((campaign / "builds/head/stage").resolve())
assert sha(psi4.core.__file__) == build["core_sha256"]
psi4.core.IOManager.shared_object().set_default_path(str(scratch) + "/")
psi4.core.set_output_file(str(output / "psi4.out"), False)
psi4.set_num_threads(24)
psi4.set_memory(16 * 1024**3)
config = json.loads((harness / "config.json").read_text())
psi4.set_options(config["options"])
atoms = (harness / "nanotube.xyz").read_text().splitlines()[2:]
molecule = psi4.geometry("0 1\n" + "\n".join(atoms[:6]) + "\n--\n0 1\n" +
                         "\n".join(atoms[6:]) + "\nunits angstrom\nsymmetry c1\nno_reorient\nno_com")
original = sapt_mp2_terms.df_fdds_dispersion
results = []
metadata = {"build": build, "incoming_scratch": incoming, "scratch": str(scratch),
            "config_sha256": sha(harness / "config.json"), "driver_sha256": sha(__file__),
            "job_id": os.environ["SLURM_JOB_ID"]}
dump(output / "metadata.json", metadata)


def replay(*args, **kwargs):
    metadata["actual_fdds_naux"] = args[1].nbf()
    dump(output / "metadata.json", metadata)
    first = None
    cases = [("STORE", True, "AUTO"), ("STORE", True, "AUTO"),
             ("LEGACY", True, "AUTO"), ("LEGACY", False, "AUTO"),
             ("STORE", True, "OUT_OF_CORE")]
    for algorithm, split, storage in cases:
        psi4.set_options({"sapt_fdds_disp_df_algorithm": algorithm,
                          "sapt_fdds_disp_split_monomers": split,
                          "sapt_fdds_disp_ao_storage": storage})
        start = time.perf_counter()
        values = original(*args, **kwargs)
        if first is None:
            first = dict(values)
        result = {"algorithm": algorithm, "split": split, "storage": storage,
                  "values": values, "delta_first": {k: v - first[k] for k, v in values.items()},
                  "seconds": time.perf_counter() - start}
        results.append(result)
        dump(output / "replay.json", results)
        print(json.dumps(result), flush=True)
        gc.collect()
    return first


sapt_mp2_terms.df_fdds_dispersion = replay
energy = psi4.energy("sapt(dft)", molecule=molecule)
dump(output / "result.json", {"completed": True, "replays": len(results),
                             "total_Eh": energy, "core": psi4.core.__file__})
psi4.core.clean()
