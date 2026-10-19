"""One fresh child: early ENV, original import probe, one owned runner, post proof."""
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import config as s


def exclusive_pid(uuid):
    text = subprocess.check_output(["nvidia-smi","--id="+uuid,
        "--query-compute-apps=gpu_uuid,pid","--format=csv,noheader,nounits"],
        text=True, timeout=20)
    rows = [[v.strip() for v in row.split(",")] for row in text.splitlines() if row.strip()]
    s.require(rows == [[uuid,str(os.getpid())]], "exclusive own PID on allocated UUID")
    return dict(pid=os.getpid(), uuid=uuid)


def early(path, mh, label, uuid):
    s.allocated()
    s.require(s.sha(path) == mh and label in {v["label"] for v in s.SEQUENCE}, "early identity")
    receipt = dict(pid=os.getpid(), job=os.environ["SLURM_JOB_ID"], label=label,
        manifest_sha256=mh, gpu_uuid=uuid, numerical_overrides=s.n.check_env(os.environ),
        environment={k:os.getenv(k) for k in ("CUDA_ROOT","CUDA_HOME","CUDA_PATH","PATH",
            "LD_LIBRARY_PATH","PYTHONPATH","MAJOR2_EXPECTED_STAGE","MAJOR2_HELPER_RUNTIME_ENV",
            "OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","CUDA_VISIBLE_DEVICES",
            "TMPDIR","SCRATCH","PSI_SCRATCH")})
    s.durable(Path(s.RUN)/"raw"/(label+".effective-env.json"), receipt)
    return receipt


def stop(signum, _):
    raise RuntimeError("whole child interrupted by signal "+str(signum))


def execute(path, mh, label, uuid):
    receipt = early(path, mh, label, uuid)
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
        signal.signal(sig, stop)
    m = s.manifest(path, mh)
    row = next(v for v in s.SEQUENCE if v["label"] == label)
    import probe
    import repeatability as r
    before = Path(s.RUN)/"raw"/(label+".before.json")
    sys.argv = [__file__,str(before)]
    probe.main()  # import/interface only, no energy/JK/XC or warmup
    imported = s.load(before)
    s.require(imported["core_sha256"] == r.ALLOWLIST[row["arm"]]["core_sha256"], "actual core")
    runtime = s.inherited("planreuse_speed", "runtime")
    cuda = runtime.Cuda(m["native_libraries"])
    cuda.synchronize()
    pre_gpu = exclusive_pid(uuid)
    output = Path(s.RUN)/"raw/sessions"/label
    sys.argv = [str(s.HERE/"runner.py"), "--arm","gpu","--suite",m["suite"],
        "--system",row["system"],"--output",str(output),"--threads","8","--memory","100 GiB",
        "--initial","35x194","--final","75x302","--factor","1","--minimum","2",
        "--source-commit",m["arms"][row["arm"]]["pin"]]
    try:
        runpy.run_path(str(s.HERE/"runner.py"), run_name="__main__")
    except SystemExit as exc:
        s.require(exc.code == 0, "whole runner failed")
    cuda.synchronize()  # outside unchanged primary wall; checked completion, not measured as energy
    post_gpu = exclusive_pid(uuid)
    mapped = runtime.validate_maps(m["native_libraries"])
    s.n.check_env(os.environ)
    s.require(all(os.getenv(k) == v for k,v in receipt["environment"].items()), "effective ENV changed")
    s.require(s.load(Path(s.RUN)/"raw"/(label+".effective-env.json")) == receipt and
              s.sha(path) == mh, "early receipt/manifest custody")
    data = s.load(output/"result.json")
    s.require(data["ok"] is True and data["trace_print_options"] ==
              {"global":3,"scf":3,"sapt":3}, "whole endpoint/PRINT options")
    # Ordinary output files must be synchronous and durable before COMPLETE.
    files = {}
    for name in ("result.json","psi4.out","timer.dat","profile.txt"):
        file = output/name
        s.require(file.is_file() and not file.is_symlink() and
                  file.stat().st_size <= s.POLICY["output_cap_bytes"], "missing/capped output")
        fd = os.open(file, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        files[name] = s.sha(file)
    s.n.fsync_dir(output)
    s.durable(Path(s.RUN)/"raw"/(label+".maps.json"), mapped)
    s.durable(Path(s.RUN)/"raw"/(label+".COMPLETE.json"), dict(
        label=label, pid=os.getpid(), uuid=uuid, job=os.environ["SLURM_JOB_ID"],
        early=receipt, pre_gpu=pre_gpu, post_gpu=post_gpu,
        maps_after_CDLL=cuda.maps, final_maps=mapped, output_hashes=files,
        actual_gpu_execution=True, energy_calls=1, synchronization_outside_primary=True))


if __name__ == "__main__":
    execute(*sys.argv[1:])
