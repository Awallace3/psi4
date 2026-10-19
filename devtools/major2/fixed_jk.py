"""Fixed-identical-input endpoint policy. No molecular energy or primary acceptance API."""
import itertools
import os
from pathlib import Path
import re
import tempfile

import numpy as np

import observe as o
import policy as p
import repeatability as r

PARENT_PIN = "c1bccd2215a2bf02a6a91fa1723f8d1729f2cd2e"
RUN = p.P + "/runs/saptdft-cuest-cpks/major2-fixed-jk-20261019"
HARNESS = p.P + "/gits/psi4.major2-fixed-jk-20261019"
MODE = "p157-fixed-identical-factor-jk-endpoint"
REVIEW_ID = "parent-fixed-identical-jk-review-required"
COMPILED_IN_NEW_JOB = False
REPLACEMENT_FOR = "none-exploratory-endpoint-only"
SUBMISSION_BUILD_ORIGINS = r.SUBMISSION_BUILD_ORIGINS
IDENTITY = r.IDENTITY
OLD = r.RUN
INPUT = OLD + "/raw/p157-baseline-r1.before.matrices/stage-0.npz"
INPUT_SHA = "dbe9d5ccbab8ebdb8299c6a8ba6463997ba3aca24fb828f124edf1c0f7b0c32e"
DA = dict(shape=[1786, 1786], dtype="<f8", c_contiguous=True, bytes=25518368,
          sha256="4a17f42cba7778bad576751b2cad0a066111c9231dd4641c347fb79c75b6a410")
RANK = 465
TOL = 1e-7
CAP = 2 * 1024**3
PROCESS_CAP = 300 * 1024**2
STATES = ("A", "B", "C", "D")
# Findings only, never stopping criteria for finite endpoint differences.
BOUNDS = dict(matrix_maxabs=1e-8, matrix_relative_frobenius=5e-12,
              scalar_abs_eh=1e-8, symmetry_maxabs=1e-10)
OPTIONS = dict(BASIS="6-31+g**", DF_BASIS_SCF="def2-universal-jkfit", PUREAM=True,
               SCF_TYPE="DF", SCF_SUBTYPE="AUTO", REFERENCE="RHF", USE_CUEST=True,
               CUEST_MIXED_PRECISION=False, INTS_TOLERANCE=1e-12,
               DF_FITTING_CONDITION=1e-10, CUEST_DFK_SLICES=5, CUEST_DFK_MODULI=8)
SETTINGS = dict(options=OPTIONS, nbf=1786, naux=7265, natom=157, rank=RANK,
                threads=8, memory_bytes=100*1024**3, print_level=3,
                alpha=1.0, beta=0.0, omega=0.0, do_J=True, do_K=True, do_wK=False,
                right_coefficients=0, computes_per_process=4, process_count=6)


def sequence():
    return [dict(row, label=row["label"].replace("p157-", "fixed-")) for row in r.sequence()]


def validate(m):
    r.validate_pair(m["diagnostic_pair"])
    expected = dict(mode=MODE, approval=REVIEW_ID, run_root=RUN, route=p.ROUTE,
        gates=p.GATES, suite=p.H0+"/geometries", suite_sha256=p.SUITE_SHA,
        sequence=sequence(), matrix_cap_bytes=CAP, settings=SETTINGS,
        findings_bounds=BOUNDS, input_path=INPUT, input_sha256=INPUT_SHA, input_Da=DA,
        factor_tolerance=TOL, per_process_timeout_seconds=1800, factor_budget_seconds=600,
        analysis_budget_seconds=1200, teardown_reserve_seconds=300,
        build_budget_seconds=[0, 0], accepted_speedup_data=False,
        identity_helper=IDENTITY, scratch_parent="/storage/scratch1/3/awallace43/cpks-major2-fixed-jk")
    for key, value in expected.items():
        p.require(m.get(key) == value, "fixed policy mismatch: "+key)
    p.require(m["harness"]["path"] == HARNESS, "fixed harness path")
    p.full_pin(m["harness"]["pin"])
    p.require(p.digest(m["harness"]["files"]) == m["harness"]["files_digest"], "harness files")
    p.require(set(m["arms"]) == set(r.ALLOWLIST), "two exact arms")
    for arm, item in m["arms"].items():
        p.require(all(item[k] == r.ALLOWLIST[arm][k] for k in ("source", "build", "pin")), "scientific arm")
        p.require(p.digest(item["files"]) == m["diagnostic_pair"]["arms"][arm]["source_files_digest"],
                  "scientific source digest")
    p.require(len(m["preservation"]["failed_root_stats"]) == 11, "eleven failed roots")
    p.require(set(m["prior_staging"]) == {"v1", "v2"}, "retain both old staging roots")
    p.require(48 * 1786**2 * 8 + 6*4*1024**2 < CAP, "validated worst-case payload reserve")


def allocated():
    p.require(os.environ.get("SLURM_JOB_ID", "").isdigit() and
              os.environ.get("SLURM_JOB_QOS") == "embers" and
              os.environ.get("SLURM_CPUS_PER_TASK") == "8" and
              bool(os.environ.get("CUDA_VISIBLE_DEVICES")), "allocated GPU host only")


def finite_array(value, shape):
    p.require(isinstance(value, np.ndarray) and value.shape == tuple(shape) and
              value.dtype.str == "<f8" and value.flags.c_contiguous and
              np.isfinite(value).all(), "finite exact FP64 shape/dtype/layout")
    return value


def load_Da(path):
    p.require(p.sha(path) == INPUT_SHA, "frozen original archive")
    with np.load(path, allow_pickle=False) as archive:
        p.require(set(archive.files) == {"Da", "J", "K"}, "original archive arrays")
        value = np.array(archive["Da"], order="C", copy=True)
    finite_array(value, DA["shape"])
    p.require(o.array_info(value) == DA, "frozen initial SAD Da")
    value.flags.writeable = False
    return value


def save_array(path, value):
    """Exclusive durable NPY; coefficient bytes cannot be replaced by a repeat."""
    path = Path(path)
    with path.open("xb") as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)
    return dict(file=path.name, file_sha256=p.sha(path), array=o.array_info(value))


def factor_once(core, da):
    """Exactly SADGuess::form_C: one baseline partial pivoted Cholesky, not eig/sqrt."""
    before = o.array_info(da)
    matrix = core.Matrix.from_array(da)
    factor = matrix.partial_cholesky_factorize(TOL)
    c = np.array(factor.np, dtype=np.float64, order="C", copy=True)
    p.require(c.ndim == 2 and c.shape[1] == RANK,
              "STOP for parent review: actual SAD factor rank="+str(c.shape))
    finite_array(c, (da.shape[0], RANK))
    p.require(o.array_info(da) == before, "input Da mutated by decomposition")
    # Single common D for every later contraction; never regenerated in JK children.
    d = c @ c.T
    finite_array(d, da.shape)
    diff = d-da
    reconstruction = dict(maxabs=float(np.max(np.abs(diff))),
        frobenius=float(np.linalg.norm(diff)), relative_frobenius=float(np.linalg.norm(diff)/np.linalg.norm(da)),
        trace_difference=float(np.trace(diff)), trace_Da=float(np.trace(da)), trace_D=float(np.trace(d)))
    c.flags.writeable = d.flags.writeable = False
    return c, d, reconstruction


def load_factor(root, receipt):
    root = Path(root)
    result = []
    for name, shape in (("C", (1786, RANK)), ("D", (1786, 1786))):
        meta = receipt[name]
        file = root/(name+".npy")
        p.require(meta["file"] == file.name and p.sha(file) == meta["file_sha256"], "fixed factor file bytes")
        value = np.load(file, allow_pickle=False)
        finite_array(value, shape)
        p.require(o.array_info(value) == meta["array"], "fixed factor array bytes")
        value.flags.writeable = False
        result.append(value)
    p.require(receipt["rank"] == RANK and receipt["tolerance"] == TOL and
              receipt["input_sha256"] == INPUT_SHA and receipt["Da"] == DA and
              receipt["baseline_core_sha256"] == r.ALLOWLIST["baseline"]["core_sha256"],
              "factor receipt provenance")
    return result


def traces(text, arm):
    created = re.findall(r"cuESTJK plan created #(\d+)", text)
    reused = re.findall(r"cuESTJK plan reused #(\d+)", text)
    banners = text.count("Device memory, DF int plan:")
    p.require(not any(x in text for x in ("DirectJK", "DiskDFJK", "MemDFJK", "status code", "CUDA error")),
              "backend fallback or SDK error")
    if arm == "candidate":
        p.require(banners == len(created), "candidate creation trace/banner agreement")
        return dict(created=len(created), reused=len(reused), counters=list(map(int, created)))
    p.require(arm == "baseline" and not created and not reused, "baseline trace identity")
    return dict(created=banners, reused=0, counters=[])


def expected_trace(arm, state):
    creates = ([1, 1, 2, 3] if arm == "baseline" else [1, 1, 1, 2])[STATES.index(state)]
    reuses = int(arm == "candidate" and state in ("C", "D"))
    return dict(created=creates, reused=reuses,
                counters=list(range(1, creates+1)) if arm == "candidate" else [])


def dispatch(jk, coefficients, arm, trace, snapshot, input_guard=lambda:None):
    """Only four computes. No writes through JK-owned arrays, no independent refactor."""
    p.require(jk.name() == "cuESTJK", "cuEST backend mandatory")
    calls = dict(initialize=0, compute=0, finalize=0, C_left_add=0, C_right_add=0)
    records = []
    def checkpoint(state, phase):
        actual = traces(trace(), arm)
        if phase == "after":
            p.require(actual == expected_trace(arm, state), "exact plan creation/reuse count")
        return actual
    try:
        input_guard()
        jk.C_clear()
        jk.C_left_add(coefficients)
        calls["C_left_add"] += 1
        for index, state in enumerate(STATES):
            before = checkpoint(state, "before")
            expected_before = expected_trace(arm, STATES[index-1]) if index else dict(created=0,reused=0,counters=[])
            p.require(before == expected_before, "before-state plan trace count")
            if state == "D":
                jk.finalize()
                calls["finalize"] += 1
                p.require(traces(trace(), arm) == before, "finalize cannot create a plan")
            if state != "B":
                jk.initialize()
                calls["initialize"] += 1
            p.require(traces(trace(), arm) == expected_trace(arm, state), "initialize trace count")
            input_guard()
            jk.compute()
            calls["compute"] += 1
            after = checkpoint(state, "after")
            records.append(snapshot(state, before, after, calls.copy()))
            input_guard()
        return records, calls
    finally:
        # Includes native SDK/trace/input/persistence errors and TERM exceptions.
        jk.finalize()
        calls["finalize"] += 1


class Store:
    """Each fresh process is capped; total cap is rechecked at every parent boundary."""
    def __init__(self, root, cap=PROCESS_CAP):
        self.root = Path(root)
        self.root.mkdir(exist_ok=False)
        self.used, self.cap = 0, cap

    def save(self, state, arrays):
        p.require(state in STATES and set(arrays) == {"J", "K"}, "exact snapshot set")
        for value in arrays.values():
            finite_array(value, (1786, 1786))
        # Conservative ZIP/deflate overhead allowance before opening any output.
        p.require(self.used + sum(v.nbytes for v in arrays.values()) + 1024**2 <= self.cap,
                  "persistent cap pre-reservation")
        file = self.root/(state+".npz")
        with file.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        self.used += file.stat().st_size
        p.require(self.used <= self.cap, "actual compressed storage cap")
        with np.load(file, allow_pickle=False) as reread:
            p.require(set(reread.files) == set(arrays) and all(
                o.array_info(reread[k]) == o.array_info(v) for k, v in arrays.items()), "NPZ readback")
        file.chmod(0o444)
        return dict(file=file.name, file_sha256=p.sha(file), file_bytes=file.stat().st_size,
                    arrays={k:o.array_info(v) for k,v in arrays.items()})


def scalar(d, j, k):
    # RHF::compute_E: coulomb_E=2*Da:J, exchange_E=-alpha*Da:K.
    value = float(2*np.einsum("ij,ji->", d, j)-np.einsum("ij,ji->", d, k))
    p.require(np.isfinite(value), "nonfinite fixed-D two-electron contraction")
    return value


def difference(a, b):
    p.require(np.isfinite(a).all() and np.isfinite(b).all(), "nonfinite difference input")
    delta = b-a
    norm = float(np.linalg.norm(a))
    result = dict(bitwise_equal=a.tobytes() == b.tobytes(), maxabs=float(np.max(np.abs(delta))),
                  relative_frobenius=float(np.linalg.norm(delta)/max(norm, np.finfo(float).tiny)))
    p.require(np.isfinite(list(result.values())).all(), "nonfinite difference")
    return dict(result, exceeds_maxabs=result["maxabs"] > BOUNDS["matrix_maxabs"],
                exceeds_relative_frobenius=result["relative_frobenius"] > BOUNDS["matrix_relative_frobenius"])


def no_timing(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "accepted_speedup_data":
                p.require(item is False, "no acceptance claim in endpoint report")
                continue
            p.require(not re.search(r"time|timing|wall|elapsed|ratio|speed|median", key, re.I),
                      "forbidden report field: "+key)
            no_timing(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            no_timing(item)
