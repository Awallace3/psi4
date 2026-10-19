"""One real allocation-only JK process; no energy/SCF/RHF finalization or CPU ref."""
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import config as s


def stop(signum, _):
    raise RuntimeError("signal interrupted GPU process: "+str(signum))


def early(argv):
    """Called before NumPy, Psi4, Einsums, or ctypes.CDLL."""
    s.allocated()
    _, manifest_path, mh, label, uuid = argv
    s.require(s.sha(manifest_path) == mh and label in {r["label"] for r in s.SEQUENCE},
              "pre-import manifest and fresh process label")
    receipt = dict(pid=os.getpid(), job=os.environ["SLURM_JOB_ID"], label=label,
        manifest_sha256=mh, gpu_uuid=uuid, numerical_overrides=s.n.check_env(os.environ),
        locators={k:os.getenv(k) for k in ("CUDA_ROOT", "CUDA_HOME", "CUDA_PATH", "PATH")},
        effective_runtime={k:os.getenv(k) for k in ("LD_LIBRARY_PATH", "PYTHONPATH",
            "MAJOR2_EXPECTED_STAGE", "MAJOR2_HELPER_RUNTIME_ENV", "OMP_NUM_THREADS",
            "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES")})
    s.durable(Path(s.RUN)/"raw"/(label+".effective-env.json"), receipt)
    return receipt


def exclusive_pid(uuid):
    text = subprocess.check_output(["nvidia-smi", "--id="+uuid,
        "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True, timeout=20)
    rows = [[v.strip() for v in row.split(",")] for row in text.splitlines() if row.strip()]
    s.require(rows == [[uuid, str(os.getpid())]], "GPU must be exclusively held by actual own PID")
    return dict(pid=os.getpid(), uuid=uuid)


def execute(m, row, uuid, receipt, started):
    # All scientific imports are deliberately after durable early ENV proof.
    import numpy as np
    import campaign as c
    import fixed_jk as f
    import observe as o
    import policy as p
    import probe
    import repeatability as r
    from numerics import Analysis, Store
    from protocol import Timer, dispatch, expected_trace
    from runtime import Cuda, validate_maps
    root = Path(s.RUN)/"raw"/row["label"]
    root.mkdir()
    output = root/"import.json"
    sys.argv = [__file__, str(output)]
    probe.main()
    imported = p.read_json(output)
    s.require(imported["core_sha256"] == r.ALLOWLIST[row["arm"]]["core_sha256"], "exact imported arm")
    import psi4
    s.require(hasattr(psi4.core.JK, "D"), "reviewed public JK.D binding absent")
    psi4.core.clean_options()
    psi4.set_num_threads(8)
    psi4.set_memory("100 GiB")
    psi4.set_options(f.OPTIONS)
    options = {k:psi4.core.get_option("SCF", k) for k in f.OPTIONS}
    norm = lambda v:{k:x.lower() if isinstance(x, str) else x for k, x in v.items()}
    s.require(norm(options) == norm(f.OPTIONS) and psi4.core.get_num_threads() == 8 and
              psi4.core.get_memory() == f.SETTINGS["memory_bytes"], "effective options/host resources")
    s.require(p.sha(Path(m["suite"])/"systems.py") == p.SUITE_SHA, "suite bytes")
    geometry = c.literal_assignments(Path(m["suite"])/"systems.py", ["SYSTEMS"])["SYSTEMS"]["protein157"]
    import hashlib
    s.require(hashlib.sha256(geometry.encode()).hexdigest() == p.SYSTEMS["protein157"][6], "geometry")
    mol = psi4.geometry(geometry)
    primary = psi4.core.BasisSet.build(mol, "ORBITAL", "6-31+g**", puream=1)
    aux = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", "def2-universal-jkfit",
                                 "JKFIT", "6-31+g**", puream=1)
    s.require((mol.natom(), primary.nbf(), aux.nbf()) == (157, 1786, 7265) and
              primary.has_puream() and aux.has_puream() and mol.schoenflies_symbol() == "c1",
              "exact basis/frame")
    frame = o.array_info(np.array(mol.geometry().np, order="C", copy=True))
    factor, (coefficients, diagnostic_D_only) = f.load_frozen_factor(f.FROZEN_FACTOR["root"])
    # D.npy is NEVER passed to JK or used in scalar contraction. Actual jk.D()
    # below is independently reconstructed by the real wrapper's compute_D.
    del diagnostic_D_only
    coeff = psi4.core.Matrix.from_array(coefficients)
    cuda = Cuda()
    store = Store(root)
    analysis = Analysis(store)
    trace_path = root/"plan.out"
    psi4.core.set_output_file(str(trace_path), False)
    jk = psi4.core.JK.build_JK(primary, aux)
    try:
        s.require(jk.name() == "cuESTJK", "no backend fallback")
        jk.set_print(3)
        jk.set_memory(f.SETTINGS["memory_bytes"]//8)
        jk.set_cutoff(1e-12)
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
        s.require(trace_path.stat().st_size < 2*1024**2, "bounded trace")
        text = trace_path.read_text()
        counts = f.traces(text, row["arm"])  # reviewed actual BASE/CAND parser
        s.require(all(v == "1" for v in re.findall(r"cuESTJK plan reused #(\d+)", text)),
                  "reuse must refer to creation #1")
        return counts

    def guard():
        s.n.check_env(os.environ)
        s.require(jk.name() == "cuESTJK" and o.array_info(coeff.np) == factor["C"]["array"],
                  "live backend/exact frozen C bytes")
        s.require(o.array_info(np.array(mol.geometry().np, order="C", copy=True)) == frame,
                  "molecular frame mutation")

    def checkpoint(phase, round_id, timing, calls):
        counts = trace()
        s.require(counts == expected_trace(row["arm"], calls["initialize"]), "actual creation/reuse counts")
        s.require(len(jk.J()) == len(jk.K()) == len(jk.D()) == 1 and not jk.wK(),
                  "one symmetric actual D/J/K, no wK")
        arrays = {k:np.array(getattr(jk, k)()[0].np, order="C", copy=True) for k in ("J", "K")}
        density = np.array(jk.D()[0].np, order="C", copy=True)
        numeric = analysis.snapshot(phase, round_id, arrays, density)
        value = dict(phase=phase, round=round_id, timing=timing, calls=calls,
                     trace=counts, gpu=exclusive_pid(uuid), C=o.array_info(coeff.np), **numeric)
        name = "warmup" if phase == "warmup" else f"round{round_id}-{phase}"
        s.durable(root/(name+".json"), value)
        s.require(numeric["matched_D"], "actual reconstructed D changed; STOP performance acceptance")
        return value

    setup_ns = time.perf_counter_ns()-started
    records, calls = dispatch(jk, coeff, Timer(cuda.synchronize), checkpoint, guard)
    cuda.synchronize()  # cleanup status outside all primary windows
    guard()
    s.require(trace() == expected_trace(row["arm"], 6), "post-finalize counts")
    f.frozen_factor_files(f.FROZEN_FACTOR["root"])  # no chmod of historical receipt
    mapped = validate_maps()
    s.n.check_env(os.environ)
    s.require(s.load(Path(s.RUN)/"raw"/(row["label"]+".effective-env.json")) == receipt,
              "early ENV custody")
    s.require(store.saved == ["warmup", "round5-control"] and
              len(list(root.glob("*.npy"))) == 4, "exact four stored arrays")
    s.durable(root/"COMPLETE.json", dict(label=row["label"], arm=row["arm"],
        pid=os.getpid(), job=os.environ["SLURM_JOB_ID"], gpu_uuid=uuid,
        setup_ns=setup_ns, process_until_complete_ns=time.perf_counter_ns()-started,
        calls=calls, records=records, trace=trace(), trace_sha256=s.sha(trace_path),
        factor_C=factor["C"], actual_geometry=frame, effective_options=options,
        settings=m["settings"], backend=jk.name(), early_env=receipt,
        core_sha256=imported["core_sha256"], cudart=cuda.pin,
        maps_after_CDLL=cuda.maps, final_runtime_maps=mapped, stored_bytes=store.used,
        numerical_overrides={}, actual_gpu_execution=True))


def main():
    started = time.perf_counter_ns()
    receipt = early(sys.argv)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, stop)
    _, path, mh, label, uuid = sys.argv
    m = s.manifest(path, mh)
    row = next(v for v in s.SEQUENCE if v["label"] == label)
    execute(m, row, uuid, receipt, started)


if __name__ == "__main__":
    main()
