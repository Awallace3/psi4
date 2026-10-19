"""Documented child-only cuBLAS control; stdlib ONLY, safe before native imports."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys

PARENT = "b9abe3e38fd194524185648d9e15807a8d66d4c9"
ROOT = "/storage/project/r-cs207-0/awallace43"
RUN = ROOT+"/runs/saptdft-cuest-cpks/major2-workspace-20261019"
OLD_RUN = ROOT+"/runs/saptdft-cuest-cpks/major2-fixed-jk-20261019-v2"
CONTROL = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
POLICY = dict(jk_children=CONTROL, controller={}, identity={}, probe_only={},
              source="https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility",
              documentation_version="13.4", installed_cuda_unchanged="13.3",
              numerical_only=True, guarantees_cuest_determinism=False)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def overrides(env):
    return {k:v for k,v in env.items() if
            k.startswith(("CUBLAS_", "CUBLASLT_", "CUDNN_", "CUSOLVER_", "CUEST_",
                          "NVIDIA_TF32", "TORCH_ALLOW_TF32", "CUDA_MATH_")) or
            k in ("CUDA_LAUNCH_BLOCKING", "CUDA_DEVICE_MAX_CONNECTIONS",
                  "CUDA_DEVICE_MAX_COPY_CONNECTIONS")}


def check_env(env, controlled=False):
    observed = overrides(env)
    require(observed == (CONTROL if controlled else {}),
            "unexpected/missing numerical environment override; STOP, never remove it")
    return observed


def child_env(parent):
    check_env(parent)
    env = dict(parent)
    env.update(CONTROL)
    check_env(env, True)
    return env


def preimport(argv):
    """First runner action, before numpy, policy, probe or Psi4/native loading."""
    mode, manifest, expected, *extra = argv[1:]
    check_env(os.environ, mode == "jk")
    data = Path(manifest).read_bytes()
    require(hashlib.sha256(data).hexdigest() == expected, "preimport manifest bytes")
    m = json.loads(data)
    require(m["numerical_control"] == POLICY and m["run_root"] == RUN, "preimport policy")
    if mode != "jk":
        require(mode == "reuse-factor", "unknown runner mode")
        return None
    require(len(extra) == 2, "JK child arguments")
    label, uuid = extra
    matches = [row for row in m["sequence"] if row["label"] == label]
    require(len(matches) == 1 and Path(label).name == label, "preimport child label")
    require(not any(k == "numpy" or k == "psi4" or k.startswith("psi4.") for k in sys.modules),
            "receipt must precede numpy/Psi4 import")
    require(os.environ.get("SLURM_JOB_ID", "").isdigit(), "allocated preimport child")
    value = dict(label=label, arm=matches[0]["arm"], pid=os.getpid(),
                 job=os.environ["SLURM_JOB_ID"], gpu_uuid=uuid, observed=check_env(os.environ, True),
                 before_numpy_psi4_import=True, manifest_sha256=expected)
    path = Path(RUN)/"raw"/(label+".effective-env.json")
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return value


def verify_receipt(value, row, pid, job, uuid, manifest_sha):
    require(value == dict(label=row["label"], arm=row["arm"], pid=pid, job=job,
        gpu_uuid=uuid, observed=CONTROL, before_numpy_psi4_import=True,
        manifest_sha256=manifest_sha), "effective ENV receipt identity/integrity")


def summary(report):
    """Only numerical report fields; no timing/log data, no old array reads."""
    require(len(report["outputs"]) == 24 and len(report["differences"]) == 276,
            "complete 24-output/276-pair numerical report")
    outputs = {v["label"]+"/"+v["state"]:v for v in report["outputs"]}
    require(len(outputs) == 24, "unique outputs")
    groups = {}
    for arm in ("baseline", "candidate"):
        for group in ("same-plan", "unchanged-initialize", "finalize-newplan", "fresh-A-A"):
            rows = []
            for d in report["differences"]:
                a, b = outputs[d["a"]], outputs[d["b"]]
                if a["arm"] != arm or b["arm"] != arm:
                    continue
                same = a["label"] == b["label"]
                states = {a["state"], b["state"]}
                selected = ((group == "same-plan" and same and states == {"A","B"}) or
                    (group == "unchanged-initialize" and same and states == {"A","C"}) or
                    (group == "finalize-newplan" and same and states == {"A","D"}) or
                    (group == "fresh-A-A" and not same and states == {"A"}))
                if selected:
                    rows.append(d)
            require(len(rows) == 3, "three explicit anchored comparisons per arm/group")
            vals = [abs(d["scalar_delta_eh"]) for d in rows]
            matrices = {k:dict(maxabs=max(d["matrices"][k]["maxabs"] for d in rows),
                relative_frobenius=max(d["matrices"][k]["relative_frobenius"] for d in rows),
                bitwise_equal_pairs=sum(d["matrices"][k]["bitwise_equal"] for d in rows))
                for k in ("J","K")}
            require(all(math.isfinite(x) for x in vals+
                [v[key] for v in matrices.values() for key in ("maxabs","relative_frobenius")]),
                "finite numerical summary")
            groups[arm+"/"+group] = dict(pairs=3, scalar_maxabs_eh=max(vals),
                scalar_nonzero_pairs=sum(x != 0 for x in vals), matrices=matrices)
    return dict(findings_bounds=report["findings_bounds"], groups=groups)


if __name__ == "__main__":
    # Used by child-arm.sh before inherited arm.sh can execute --psiapi.
    require(sys.argv[1:] == ["--child-shell-check"], "shell gate arguments")
    check_env(os.environ, True)
