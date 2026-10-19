"""Temporary RHF hooks: read-only first actual Python-dispatched form_G per SAPT stage."""
from contextlib import contextmanager
import hashlib
import inspect
import os
from pathlib import Path
import tempfile

import numpy as np

import policy as p
import repeatability as r
import science


def atomic(path, value):
    """Exclusive atomic publication + directory fsync; retain prior evidence."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix="."+path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            import json
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(tmp, path)  # atomic and fails on any existing receipt
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(tmp)


def array(matrix):
    view = matrix.np
    p.require(isinstance(view, np.ndarray) and view.dtype == np.dtype("float64") and
              view.ndim == 2 and view.shape == (1786, 1786), "actual C1 FP64 density/J/K required")
    result = np.array(view, dtype=np.float64, order="C", copy=True)
    p.require(np.isfinite(result).all(), "nonfinite snapshot")
    result.flags.writeable = False
    return result


def array_info(value):
    return dict(shape=list(value.shape), dtype=value.dtype.str, c_contiguous=value.flags.c_contiguous,
                sha256=hashlib.sha256(memoryview(value).cast("B")).hexdigest(), bytes=value.nbytes)


def stage_stamp(wfn, stage):
    """Read the actual staged scf_helper's banner; never replace its function."""
    iterator = stage/"lib/psi4/driver/procrouting/scf_proc/scf_iterator.py"
    helper = stage/"lib/psi4/driver/procrouting/proc.py"
    frame = inspect.currentframe()
    seen_iterator = None
    banner = None
    try:
        while frame is not None:
            if frame.f_code.co_name == "scf_iterate" and Path(frame.f_code.co_filename) == iterator:
                p.require(frame.f_locals.get("self") is wfn, "iterator object mismatch")
                seen_iterator = dict(path=str(iterator), function="scf_iterate", line=frame.f_lineno)
            if frame.f_code.co_name == "scf_helper" and Path(frame.f_code.co_filename) == helper:
                banner = frame.f_locals.get("banner")  # scf_helper already popped kwargs['banner']
                break
            frame = frame.f_back
    finally:
        del frame
    p.require(seen_iterator is not None and banner in ["SAPT(DFT): "+x for x in science.STAGES],
              "not actual staged SAPT first form_G seam")
    return dict(banner=banner, caller=seen_iterator)


class Capture:
    def __init__(self, cls, stage, directory, provenance, stamp=stage_stamp, cap=r.PROCESS_CAP):
        self.cls, self.stage, self.directory = cls, Path(stage), Path(directory)
        self.provenance, self.stamp, self.cap = provenance, stamp, cap
        self.objects, self.rows, self.finalized = [], [], []
        self.used = 0
        self.directory.mkdir(exist_ok=False)

    def flush(self, suffix):
        atomic(self.directory/(suffix+".json"),
               dict(provenance=self.provenance, pid=os.getpid(), stages=self.rows,
                    finalized=self.finalized, stored_bytes=self.used, cap_bytes=self.cap))

    def form(self, original, wfn, *args, **kwargs):
        p.require(type(wfn) is self.cls, "unexpected non-RHF class")
        stamp = self.stamp(wfn, self.stage)
        if wfn in self.objects:
            index = self.objects.index(wfn)
            row = self.rows[index]
            p.require(stamp == row["stamp"], "stage/callsite changed during SCF")
            row["calls"] += 1
            row["native_calls"] += 1
            result = original(wfn, *args, **kwargs)
            row["native_returns"] += 1
            return result
        index = len(self.objects)
        p.require(index < 5 and stamp["banner"] == "SAPT(DFT): "+science.STAGES[index], "first stage order")
        p.require(wfn.iteration_ == 0 and wfn.sad_, "first actual SAD form_G was missed")
        p.require(not self.finalized if index == 0 else True, "HF dimer after finalize")
        p.require(wfn.basisset().nbf() == 1786 and wfn.nalpha() == wfn.nbeta() > 0, "nbf/closed-shell")
        functional = wfn.functional().name()
        p.require(functional == ("HF" if index < 3 else "PBE0"), "stage functional")
        molecule = wfn.molecule()
        p.require(molecule.natom() == 157, "Protein157 identity")
        if index == 0:
            p.require(molecule.nfragments() == 2 and all(molecule.Z(i) > 0 for i in range(157)),
                      "not physical HF dimer")
        density = array(wfn.Da())
        p.require(np.array_equal(density, wfn.Db().np), "RHF Db differs; cannot omit")
        row = dict(stage=science.STAGES[index], stamp=stamp, python_class=self.cls.__module__+"."+self.cls.__name__,
                   object_id=id(wfn), pid=os.getpid(), nalpha=wfn.nalpha(), nbeta=wfn.nbeta(), nbf=1786,
                   functional=functional, iteration=wfn.iteration_, sad=bool(wfn.sad_),
                   calls=1, native_calls=0, native_returns=0, db_equals_da=True,
                   initial_dimer_before_finalize=index == 0)
        self.objects.append(wfn)  # hold reference; cannot recycle identity
        self.rows.append(row)
        self.flush(f"stage-{index}-before")
        row["native_calls"] = 1
        result = original(wfn, *args, **kwargs)  # EXACTLY ONE native dispatch; no retries
        row["native_returns"] = 1
        jk = wfn.jk()
        js, ks = jk.J(), jk.K()
        p.require(len(js) == len(ks) == 1, "one actual J/K output required")
        values = dict(Da=density, J=array(js[0]), K=array(ks[0]))
        p.require(np.array_equal(density, wfn.Da().np), "native form_G unexpectedly altered density")
        row["arrays"] = {key: array_info(value) for key, value in values.items()}
        path = self.directory/f"stage-{index}.npz"
        # np.savez_compressed uses no matrix state writes; temporary output is private.
        fd, tmp = tempfile.mkstemp(prefix=".arrays-", dir=self.directory)
        try:
            with os.fdopen(fd, "wb") as stream:
                np.savez_compressed(stream, **values)
                stream.flush()
                os.fsync(stream.fileno())
            size = Path(tmp).stat().st_size
            p.require(self.used+size <= self.cap, "persistent matrix budget exceeded")
            os.link(tmp, path)
        finally:
            os.unlink(tmp)
        self.used += size
        row.update(file=path.name, file_sha256=p.sha(path), stored_bytes=size)
        self.flush(f"stage-{index}-captured")
        return result

    def finalize(self, original, wfn, *args, **kwargs):
        p.require(wfn in self.objects, "finalize before observed form_G")
        self.finalized.append(self.objects.index(wfn))
        return original(wfn, *args, **kwargs)

    @contextmanager
    def installed(self):
        cls = self.cls
        p.require("form_G" not in cls.__dict__ and "finalize" not in cls.__dict__, "unexpected binding owner")
        form, finalize = cls.form_G, cls.finalize
        try:
            cls.form_G = lambda wfn, *a, **k: self.form(form, wfn, *a, **k)
            cls.finalize = lambda wfn, *a, **k: self.finalize(finalize, wfn, *a, **k)
            yield self
        finally:
            # Delete shadow attributes to restore descriptor ownership, not only callable equality.
            if "form_G" in cls.__dict__:
                del cls.form_G
            if "finalize" in cls.__dict__:
                del cls.finalize
            p.require(cls.form_G is form and cls.finalize is finalize, "hook restore failed")
            self.flush("partial-restored")

    def finish(self):
        p.require(len(self.rows) == 5 and self.finalized == list(range(5)), "five stages/finalizes required")
        p.require(all(x["calls"] == x["native_calls"] == x["native_returns"] and x.get("arrays") and x["calls"] > 1
                      for x in self.rows), "native count/capture")
        self.flush("COMPLETE")


def matrix_difference(a, b):
    p.require(a.dtype == b.dtype == np.dtype("float64") and a.shape == b.shape, "matrix identity")
    diff = a-b
    denominator = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)))
    numerator = float(np.linalg.norm(diff))
    return dict(exact_equal=bool(np.array_equal(a, b)),
                bitwise_equal=bool(np.array_equal(a.view(np.uint64), b.view(np.uint64))),
                maxabs=float(np.max(np.abs(diff))),
                normrel=numerator/denominator if denominator else 0.0)
