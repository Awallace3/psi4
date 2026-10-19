"""Fixed policy and small stdlib utilities. No scientific imports or scheduler writes."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
P = "/storage/project/r-cs207-0/awallace43"
E = P+"/miniconda/envs/p4cuest_sapt"
SOURCE = P+"/gits/psi4.major2-native-plan-20261019"
RUN = P+"/runs/saptdft-cuest-cpks/major2-native-plan-20261019-v3"
OLD = P+"/runs/saptdft-cuest-cpks/major2-workspace-20261019"
BASE = P+"/gits/psi4.saptdft-cuest-major2-jkexception-baseline-20261018"
BASE_PIN = "c56708fc6f6e8fcf2801868f0d3cb49bdb1bb0a2"
STAGE = BASE+"/build_major2_cuda_jkexception/stage"
CORE_SHA = "67c1e9021cc2fcaaf3b4ad3f98f6750ee39049c1aafe8cdfbe75b46c8335f224"
FACTOR = P+"/runs/saptdft-cuest-cpks/major2-fixed-jk-20261019/raw/factor"
FACTOR_FILES = {
    "FACTOR.json":"b6777acd5ec6098c0a81bcf08af74591b8c17edb511c4bd04c5aee9be3c9121e",
    "C.npy":"76b2aa312d8b6de1568224a793f33c634630b8d7f418fbd4a2b10522220a92a0",
    "D.npy":"76cef9607770669ffd08b5f5599b385bb2f1e42ed57254261c5209dfb3025f92"}
D_SHA = "ba006a181c7ddfb151eef54bd98dcc01bd500f7c50da9c4a02ac7b0f5cf2a1a3"
SDK = P+"/software/cuest-0.2.50-a06e8148"
SDK_SHA = "65c9e559201ee589cb742b919cc1c5d80c45aa480b64aede17ac96aff9e4963c"
CUDA = P+"/software/toolchains/cuda13"
CXX = P+"/software/toolchains/conda-g++"
CC1PLUS = P+"/software/toolchains/conda-gcc14-libexec/cc1plus"
CC1PLUS_SHA = "b798f20b3b14756f20957c1a8535da2b6266919833f6aedad61cd30c1a841b83"
CXX_REAL = E+"/bin/x86_64-conda-linux-gnu-c++"
CXX_REAL_SHA = "ecf6b7dc10f7cb21933e7f51c8c64c0127679cf912dbf571afb4511bf19e5116"
CAP = 2*1024**3
STATES = [p+str(k) for p in "ABC" for k in (1,2)]
SCRATCH = "/storage/scratch1/3/awallace43/cpks-major2-native-plan"
POLICY = dict(processes=3, handles_per_process=1, upstream_builds_per_process=2,
    plans_per_process=3, outputs=18, pairs=153, states=STATES, cap=CAP,
    numerical_environment={}, native_fp64=True, alpha=1., beta=0., omega=0.,
    threshold_pq=1.e-12, metric_parameters="SDK defaults; not Psi4 condition_",
    workspace_initialization="malloc/cudaMalloc, no zero/poison; J overwritten by API",
    density="original frozen D.npy, NOT JK::compute_D reconstructed C density",
    statistic="Tr(DJ), NOT prior 2Tr(DJ)-Tr(DK)", k=False,
    timing_summaries=False, performance_acceptance=False, auto_followup=False)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def check_factor(root, files):
    """Validate original factor bytes and permissions without mutating inputs."""
    for name, digest in files.items():
        p = Path(root)/name
        # The historical receipt is 0600. Hash-pin every file, but only the
        # frozen arrays have a readonly contract. Preservation checks retain
        # the receipt's original stat/mode; never chmod historical inputs.
        require(sha(p) == digest and
                (name == "FACTOR.json" or not p.stat().st_mode & 0o222),
                "frozen original factor: "+name)


def check_env(env):
    # ROOT/HOME/PATH locate toolchains, not numerical modes. Keep inherited
    # values as evidence; explicit compiler/header/runtime checks remain binding.
    allowed_cuda = {"CUDA_VISIBLE_DEVICES","CUDA_ROOT","CUDA_HOME","CUDA_PATH","CUDA_CACHE_PATH",
                    "CUDA_CACHE_DISABLE","CUDA_CACHE_MAXSIZE"}
    found = {k:v for k,v in env.items() if
        k.startswith(("CUBLAS","CUDNN","CUSOLVER","CUSPARSE","CUEST","NVIDIA_TF32",
                      "TORCH_ALLOW_TF32","CUDA_MATH")) or
        (k.startswith("CUDA_") and k not in allowed_cuda) or
        k in {"OMP_PROC_BIND","OMP_PLACES","MKL_CBWR","MKL_DEBUG_CPU_TYPE"}}
    require(not found, "inherited numerical override: STOP, never unset: "+str(found))
    return found


def output(*args, env=None, cwd=None, timeout=35):
    return subprocess.check_output(list(map(str,args)), text=True, env=env, cwd=cwd,
                                   timeout=timeout).strip()


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY|os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def durable(path, value, binary=False):
    path = Path(path)
    data = value if binary else (value if isinstance(value,str) else
                json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    with path.open("xb") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    path.chmod(0o444)
    fsync_dir(path.parent)


def load(path):
    return json.loads(Path(path).read_text())


def manifest(path, digest):
    require(sha(path) == digest, "manifest bytes")
    m = load(path)
    require(m["policy"] == POLICY and m["source"] == SOURCE and m["run"] == RUN and
            m["base_pin"] == BASE_PIN and m["base_stage"] == STAGE and
            re.fullmatch("[a-f0-9]{40}",m["pin"]), "fixed native policy")
    return m


def source_check(m, root=SOURCE):
    require(output("git","-C",root,"rev-parse","HEAD") == m["pin"], "source pin")
    require(not output("git","-C",root,"status","--porcelain","--untracked-files=no"),
            "tracked source changes")
    # Git verifies all tracked bytes against index; SHA256 additionally checks every owned helper.
    for rel, digest in m["helpers"].items():
        require(sha(Path(root)/rel) == digest, "helper tracked bytes: "+rel)
    require(output("git","-C",root,"diff","--name-only",m["parent"],m["pin"]).splitlines() ==
            sorted(m["helpers"]), "add-only owned change set")


def allocation_only():
    require(os.environ.get("SLURM_JOB_ID","").isdigit() and
            not os.uname().nodename.startswith("login"), "allocation only")
    check_env(os.environ)
