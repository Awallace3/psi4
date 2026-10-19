"""Allocation-only one-time factor OR one fresh JK process. Never SCF/energy."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import numpy as np

import campaign as c
import fixed_jk as f
import observe as o
import policy as p
import probe
import repeatability as r


def stop(signum, _):
    raise RuntimeError("signal interrupts fixed JK process: "+str(signum))


def imported(arm, before):
    sys.argv = [__file__, str(before)]
    probe.main()  # exact same runtime maps/import guards, no device construction
    value = p.read_json(before)
    p.require(value["core_sha256"] == r.ALLOWLIST[arm]["core_sha256"], "imported exact arm")
    import psi4
    return psi4, value


def factor(root, before):
    psi4, observed = imported("baseline", before)
    psi4.set_num_threads(8)
    psi4.set_memory("100 GiB")
    da = f.load_Da(f.INPUT)
    root.mkdir(exist_ok=False)
    cmat, dmat, reconstruction = f.factor_once(psi4.core, da)
    value = dict(rank=cmat.shape[1], tolerance=f.TOL, input_path=f.INPUT,
        input_sha256=f.INPUT_SHA, Da=f.DA, C=f.save_array(root/"C.npy", cmat),
        D=f.save_array(root/"D.npy", dmat), reconstruction=reconstruction,
        baseline_core=observed["core"], baseline_core_sha256=observed["core_sha256"],
        baseline_version=observed["psi4_version"], baseline_pin=r.ALLOWLIST["baseline"]["pin"],
        baseline_origin_job="14165998", compiled_in_this_job=False, pid=os.getpid(),
        source="SADGuess::form_C -> Matrix::partial_cholesky_factorize",
        input_unchanged=(o.array_info(da) == f.DA and p.sha(f.INPUT) == f.INPUT_SHA))
    p.require(value["input_unchanged"], "factor input mutation")
    o.atomic(root/"FACTOR.json", value)
    f.load_factor(root, value)


def gpu_pid(uuid):
    text = subprocess.check_output(["nvidia-smi", "--id="+uuid,
        "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True, timeout=20)
    rows = [line.strip().split(",") for line in text.splitlines() if line.strip()]
    matches = [row for row in rows if len(row) == 2 and row[1].strip() == str(os.getpid())]
    p.require(len(matches) == 1 and matches[0][0].strip() == uuid, "actual PID/physical GPU UUID")
    return dict(pid=os.getpid(), gpu_uuid=uuid)


def jk_process(m, row, root, factor_root, before, uuid):
    psi4, observed = imported(row["arm"], before)
    psi4.core.clean_options()
    psi4.set_num_threads(8)
    psi4.set_memory("100 GiB")
    p.require(psi4.core.get_num_threads() == 8 and psi4.core.get_memory() == f.SETTINGS["memory_bytes"],
              "threads/memory")
    psi4.set_options(f.OPTIONS)
    options = {key: psi4.core.get_option("SCF", key) for key in f.OPTIONS}
    p.require({k:str(v).lower() if isinstance(v,str) else v for k,v in options.items()} ==
              {k:str(v).lower() if isinstance(v,str) else v for k,v in f.OPTIONS.items()}, "effective options")
    geometry = c.literal_assignments(Path(m["suite"])/"systems.py", ["SYSTEMS"])["SYSTEMS"]["protein157"]
    import hashlib
    p.require(hashlib.sha256(geometry.encode()).hexdigest() == p.SYSTEMS["protein157"][6], "geometry input")
    mol = psi4.geometry(geometry)
    primary = psi4.core.BasisSet.build(mol, "ORBITAL", "6-31+g**", puream=1)
    auxiliary = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", "def2-universal-jkfit",
                                        "JKFIT", "6-31+g**", puream=1)
    p.require((mol.natom(), primary.nbf(), auxiliary.nbf()) == (157,1786,7265) and
              primary.has_puream() and auxiliary.has_puream() and mol.schoenflies_symbol() == "c1",
              "exact spherical basis dimensions/frame")
    geometry_array = np.array(mol.geometry().np, order="C", copy=True)
    receipt = p.read_json(factor_root/"FACTOR.json")
    receipt_sha = p.sha(factor_root/"FACTOR.json")
    coefficients, density = f.load_factor(factor_root, receipt)
    coeff = psi4.core.Matrix.from_array(coefficients)
    p.require(o.array_info(coeff.np) == receipt["C"]["array"], "actual C_left_add coefficient bytes/column count")
    store = f.Store(root)
    trace_path = root/"plan.out"
    psi4.core.set_output_file(str(trace_path), False)
    jk = psi4.core.JK.build_JK(primary, auxiliary)
    # Set everything before first initialize, so setters cannot create extra plans.
    try:
        p.require(jk.name() == "cuESTJK", "no CPU/direct fallback")
        jk.set_print(3)
        jk.set_memory(f.SETTINGS["memory_bytes"]//8)  # JK public API uses doubles
        jk.set_cutoff(f.OPTIONS["INTS_TOLERANCE"])
        jk.set_do_J(True)
        jk.set_do_K(True)
        jk.set_do_wK(False)
        jk.set_omega_alpha(1.0)
        jk.set_omega_beta(0.0)
        jk.set_omega(0.0)
    except BaseException:
        jk.finalize()
        raise
    def trace():
        psi4.core.flush_outfile()
        p.require(trace_path.stat().st_size < 2*1024**2, "bounded trace log")
        return trace_path.read_text()
    def snapshot(state, before_trace, after_trace, calls):
        p.require(o.array_info(coeff.np) == receipt["C"]["array"], "native coefficient mutation")
        p.require(o.array_info(np.array(mol.geometry().np, order="C", copy=True)) == o.array_info(geometry_array),
                  "live molecular frame mutation")
        p.require(len(jk.J()) == len(jk.K()) == 1 and len(jk.wK()) == 0, "one symmetric J/K, no wK")
        arrays = {k:np.array(getattr(jk,k)()[0].np, order="C", copy=True) for k in ("J","K")}
        meta = store.save(state, arrays)  # read-only copies only; no state writeback
        symmetry = {k:float(np.max(np.abs(v-v.T))) for k,v in arrays.items()}
        value = dict(state=state, before=before_trace, after=after_trace, calls=calls,
            C_left_add=o.array_info(coeff.np), rank=coeff.np.shape[1],
            gpu=gpu_pid(uuid), scalar_eh=f.scalar(density, arrays["J"], arrays["K"]),
            symmetry_maxabs=symmetry,
            symmetry_findings={k:v>f.BOUNDS["symmetry_maxabs"] for k,v in symmetry.items()}, **meta)
        o.atomic(root/(state+".json"), value)
        return value
    def input_guard():
        p.require(o.array_info(coeff.np) == receipt["C"]["array"], "exact native coefficient bytes before/after dispatch")
    records, calls = f.dispatch(jk, coeff, row["arm"], trace, snapshot,input_guard)
    p.require(calls == dict(initialize=3, compute=4, finalize=2, C_left_add=1, C_right_add=0), "exact dispatch count")
    p.require(f.traces(trace(), row["arm"]) == f.expected_trace(row["arm"], "D"), "post-cleanup trace")
    f.load_factor(factor_root, receipt)
    p.require(p.sha(factor_root/"FACTOR.json") == receipt_sha and
              o.array_info(coeff.np) == receipt["C"]["array"], "factor/receipt mutation")
    mapped = probe.maps()
    p.check_runtime_maps(mapped)
    proof = p.check_required_runtime_maps(mapped)
    o.atomic(root/"COMPLETE.json", dict(label=row["label"], arm=row["arm"], pid=os.getpid(),
        job=os.environ["SLURM_JOB_ID"], gpu_uuid=uuid, settings=f.SETTINGS, effective_options=options,
        molecule_geometry=o.array_info(geometry_array), input_geometry_sha256=p.SYSTEMS["protein157"][6],
        factor_receipt_sha256=receipt_sha, factor_C=receipt["C"], calls=calls, records=records,
        stored_bytes=store.used, trace_sha256=p.sha(trace_path), trace=f.traces(trace(), row["arm"]),
        core_sha256=observed["core_sha256"], runtime_maps=mapped, runtime_proof=proof,
        compiled_in_this_job=False, origin_job=r.ALLOWLIST[row["arm"]]["origin_job"]))


def main():
    f.allocated()
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, stop)
    mode, manifest, manifest_sha, *args = sys.argv[1:]
    p.require(p.sha(manifest) == manifest_sha, "child manifest bytes")
    m = p.read_json(manifest)
    f.validate(m)
    raw = Path(f.RUN)/"raw"
    if mode == "factor":
        factor(raw/"factor", raw/"factor-import.json")
    elif mode == "jk":
        label, uuid = args
        matches = [row for row in f.sequence() if row["label"] == label]
        p.require(len(matches) == 1, "fresh process label")
        jk_process(m, matches[0], raw/label, raw/"factor", raw/(label+"-import.json"), uuid)
    else:
        raise RuntimeError("unknown fixed-input runner mode")


if __name__ == "__main__":
    main()
