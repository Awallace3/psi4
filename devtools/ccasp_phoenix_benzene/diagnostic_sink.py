# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Explicit bounded live-operand diagnostics. Stdlib only; no solve or tensor repair."""
from array import array
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import sys


def capture_tools():
    name = "_ccasp_integrated_owned_lw"
    path = Path(__file__).with_name("lw_capture.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def reduce_pair(terms):
    """Same ordered products, ordinary sum AND fsum; neither changes science."""
    products, ordinary = array("d"), 0.
    for term in terms:
        value = float(term)
        ordinary += value
        if not math.isfinite(value) or not math.isfinite(ordinary):
            raise ValueError("nonfinite diagnostic reduction")
        products.append(value)
    return ordinary, math.fsum(products)


def view(value, shape):
    if value.shape != shape or value.dtype.str != "<f8":
        raise ValueError("diagnostics require exact live float64 shape " + repr(shape))
    return value


def operand_sha(value):
    """Exact C-order little-endian row hashing; never a full F/Q/B/X copy."""
    result = hashlib.sha256()
    for i in range(value.shape[0]):
        row = array("d",(float(value[i,j]) for j in range(value.shape[1])))
        if any(not math.isfinite(x) for x in row):
            raise ValueError("nonfinite diagnostic operand")
        result.update(row.tobytes())
    return result.hexdigest()


class DiagnosticSink:
    """One explicit run-owned sink, reference-H2H1 only, at most 12 rank4 sites.

    The ledger reserves 32MiB throughout and charges a conservative whole-run
    work/I/O allowance. Only O(nOV+nAUX+ncomponents) vectors survive a hook;
    no reference/copy of whole F/Q/B/X is stored. No inputs are ever assigned.
    """
    memory_bytes = 32*1024**2
    output_bytes = 64*1024**2
    io_bytes = 4*output_bytes  # writes plus checksum/validation reads, charged up front

    def __init__(self, root, identities, *, kind="native_owned", context=None):
        if sys.byteorder != "little":
            raise ValueError("explicit little-endian diagnostic runtime required")
        self.tools = capture_tools()
        self.tools._identities(identities)
        self.identities = json.loads(encoded(identities))
        if kind not in ("synthetic", "native_owned"):
            raise ValueError("explicit diagnostic kind required")
        self.kind, self.context = kind, json.loads(encoded(context or {}))
        self.root = Path(root)
        self.root.mkdir(mode=0o700, exist_ok=False)
        self.written, self.phase, self.node = 0, "new", 0
        self.vectors, self.node_vectors = {}, {}
        self._json(self.root/"context.json", dict(identities=self.identities,
            kind=kind, context=self.context, properties_accepted=False,
            semantics="diagnostic only; no change to production inputs, solver or LW gate"))

    def _write(self, path, data):
        if self.written+len(data) > self.output_bytes:
            raise ValueError("diagnostic output byte allowance")
        self.tools._write(path, data)  # exclusive0600, flush/fsync; never overwrite
        self.written += len(data)

    def _json(self, path, value):
        data = encoded(value)
        if len(data) > 1024**2:
            raise ValueError("diagnostic metadata limit")
        self._write(path, data)

    def _bundle(self, directory, vectors, metadata):
        directory.mkdir(mode=0o700, exist_ok=False)
        records = {}
        for name, values in vectors.items():
            data = values.tobytes()
            if any(not math.isfinite(x) for x in values):
                raise ValueError("nonfinite diagnostic vector")
            self._write(directory/(name+".f64"), data)
            records[name] = dict(file=name+".f64", dtype="<f8", shape=[len(values)],
                nbytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        manifest = dict(schema="ccasp-live-charge-attribution/1", arrays=records,
            metadata=metadata, properties_accepted=False,
            units="atomic units; moment components carry their declared rank-dependent units",
            convention="global real Racah/no-CS, site-major; charge row25*a; F row a*nocc+i",
            reduction="ordinary ordered and math.fsum of same products; no repair or gate",
            grouping="regrouped contractions need not agree bitwise; signed differences are diagnostics")
        self._json(directory/"manifest.json", manifest)
        return hashlib.sha256(encoded(manifest)).hexdigest()

    def plan(self, p, n, m, nf):
        if (self.phase != "new" or any(type(x) is not int for x in (p,n,m,nf))
                or not 1 <= p <= 2048 or not 1 <= n <= 16384
                or not 25 <= m <= 300 or m % 25 or not 1 <= nf <= 11):
            raise ValueError("diagnostic dimensions/one-use allowance")
        self.dimensions = p,n,m,nf
        self.phase = "planned"
        # Conservative scalar product/reduction/byte-snapshot estimate, not CPU time.
        work = 128*n*(p+nf*m)+128*nf*m*m+128*p*m
        result = dict(memory_bytes=self.memory_bytes, work=work, io_bytes=self.io_bytes,
            output_bytes=self.output_bytes, dimensions=self.dimensions,
            lifetime="only small vectors retained; no F/Q/B/X checkpoints")
        self._json(self.root/"plan.json", result)
        return result

    def auxiliary_charges(self, charges):
        p,n,m,nf = self.dimensions
        if self.phase != "planned":
            raise ValueError("observe anchor AUX charges exactly once")
        view(charges, (p,))
        q = array("d", (float(charges[k]) for k in range(p)))
        if any(not math.isfinite(x) for x in q):
            raise ValueError("nonfinite actual AUX charge")
        self.vectors["q"] = q
        self.phase = "charges"

    def anchor(self, F, Q, B, *, charge_penalty=None, offsite_metric_damping=None):
        p,n,m,nf = self.dimensions
        if self.phase != "charges":
            raise ValueError("anchor diagnostics require actual AUX charges")
        if self.kind == "native_owned" and (
                charge_penalty is None or offsite_metric_damping is None
                or not math.isfinite(charge_penalty) or charge_penalty < 0
                or not math.isfinite(offsite_metric_damping) or not 0 <= offsite_metric_damping < 1):
            raise ValueError("actual native anchor fit parameters required")
        view(F,(n,p)); view(Q,(m,p)); view(B,(n,m))
        q = self.vectors["q"]
        qc, qcf = array("d"), array("d")
        for k in range(p):
            ordinary, compensated = reduce_pair(Q[t,k] for t in range(0,m,25))
            qc.append(ordinary); qcf.append(compensated)
        delta = array("d", (qc[k]-q[k] for k in range(p)))
        self.vectors.update(QT_c=qc, QT_c_fsum=qcf, deltaq=delta,
            deltaq_fsum=array("d",(qcf[k]-q[k] for k in range(p))))
        for name in ("Fq","Fdeltaq","Bc"):
            self.vectors[name], self.vectors[name+"_fsum"] = array("d"), array("d")
        for i in range(n):
            for name, terms in (
                ("Fq", (float(F[i,k])*q[k] for k in range(p))),
                ("Fdeltaq", (float(F[i,k])*delta[k] for k in range(p))),
                ("Bc", (B[i,t] for t in range(0,m,25)))):
                ordinary, compensated = reduce_pair(terms)
                self.vectors[name].append(ordinary); self.vectors[name+"_fsum"].append(compensated)
        self.vectors["Bc_minus_Fq_minus_Fdeltaq"] = array("d",
            (self.vectors["Bc"][i]-self.vectors["Fq"][i]-self.vectors["Fdeltaq"][i] for i in range(n)))
        self.anchor_sha = self._bundle(self.root/"anchor", self.vectors, dict(
            q_source="existing core.IsaAuxCoulomb(auxiliary).charges() in native_constrained_ov",
            charge_constraint="finite lambda; q from exact same native effective AUX, not guessed normalization",
            fit="anchor fit only; not eta=0 target", Q_shape=[m,p], F_shape=[n,p], B_shape=[n,m],
            actual_charge_penalty=charge_penalty, actual_offsite_metric_damping=offsite_metric_damping,
            Q_sha256=operand_sha(Q), F_sha256=operand_sha(F), B_sha256=operand_sha(B)))
        self.phase = "anchor"

    def solved(self, omega, B, X):
        p,n,m,nf = self.dimensions
        if self.phase not in ("anchor","captured") or self.node >= nf:
            raise ValueError("one existing anchor solve per diagnostic node")
        if not math.isfinite(omega) or omega < 0 or (self.node and omega <= self.omega):
            raise ValueError("diagnostic node frequency/order")
        view(B,(n,m)); view(X,(n,m))
        self.omega, self.node = float(omega), self.node+1
        self.node_vectors = {}
        xc, xcf = array("d"), array("d")
        for i in range(n):
            a,b = reduce_pair(X[i,t] for t in range(0,m,25))
            xc.append(a); xcf.append(b)
        self.node_vectors.update(Xc=xc, Xc_fsum=xcf)
        for name in ("sfit","sQ","minus_BcT_X","minus_BT_Xc"):
            self.node_vectors[name], self.node_vectors[name+"_fsum"] = array("d"), array("d")
        for j in range(m):
            for name, terms in (
                ("sfit", (-self.vectors["Fq"][i]*float(X[i,j]) for i in range(n))),
                ("sQ", (-self.vectors["Fdeltaq"][i]*float(X[i,j]) for i in range(n))),
                ("minus_BcT_X", (-self.vectors["Bc"][i]*float(X[i,j]) for i in range(n))),
                ("minus_BT_Xc", (-float(B[i,j])*xc[i] for i in range(n)))):
                a,b = reduce_pair(terms)
                self.node_vectors[name].append(a); self.node_vectors[name+"_fsum"].append(b)
        self.node_vectors["sfit_plus_sQ_minus_direct_B"] = array("d",
            (self.node_vectors["sfit"][j]+self.node_vectors["sQ"][j]
             -self.node_vectors["minus_BcT_X"][j] for j in range(m)))
        self.node_root = self.root/f"node-{self.node:02d}"
        self.solved_sha = self._bundle(self.node_root, self.node_vectors, dict(
            frequency_au_hex=self.omega.hex(), anchor_manifest_sha256=self.anchor_sha,
            X_sha256=operand_sha(X), B_sha256=operand_sha(B), X_shape=[n,m],
            X_source="existing solved[:,p:] after unchanged H2H1 residual gate; NOT target_response",
            B_source="existing self.legs[:,p:]; actual contracted anchor, not reconstructed",
            column_axis="alpha.T @ c = -X.T @ Bc; attributed by sfit+sQ",
            row_axis="alpha @ c = -B.T @ Xc; do not assume reciprocity"))
        self.phase = "solved"

    def before_lw(self, *, raw, tensors, labels, origins, frames, bonds, frequencies,
                  input_rank, truncation, residual_policy, localization_rank_limit,
                  provenance, source):
        p,n,m,nf = self.dimensions
        if (self.phase != "solved" or tuple(frequencies) != (self.omega,)
                or input_rank != 4 or truncation != self.tools.TRUNCATION
                or residual_policy != "production" or localization_rank_limit != 2):
            raise ValueError("capture requires original strict rank4-to-rank2 node")
        view(raw,(m,m)); view(tensors,(1,m//25,m//25,25,25))
        view(origins,(m//25,3))
        def packed(value):
            return self.tools.Float64Array(value.shape, value.tobytes(order="C"))
        pair = packed(tensors)
        if len(labels) != m//25:
            raise ValueError("diagnostic site order/size differs")
        frame_bytes = b"".join(struct.pack("<d",float(x)) for f in frames for row in f for x in row)
        # Reserve maximum helper output before its exclusive write. Actual bytes
        # are bounded again by the helper; subsequent difference bundles count too.
        estimate = 16*m*m+1024**2
        if self.written+estimate > self.output_bytes:
            raise ValueError("diagnostic capture output allowance")
        receipt = self.tools.snapshot(self.node_root/"lw-input", tensors=pair,
            raw_site_major=packed(raw), origins=packed(origins),
            frames=self.tools.Float64Array((m//25,3,3),frame_bytes),
            labels=labels, bonds=bonds, frequency=self.omega, identities=self.identities, kind=self.kind)
        capture = self.tools.load_capture(receipt.path, receipt.manifest_sha256)
        self.written += sum(len(a.data) for _,a in capture.arrays) + len(capture.manifest_bytes)
        differences = {}
        for name, raw_name, direct in (
            ("column_axis_minus_direct_B","sum_response_site","minus_BcT_X"),
            ("row_axis_minus_direct_B","sum_potential_site","minus_BT_Xc")):
            values = struct.unpack("<"+"d"*m,capture.array(raw_name).data)
            differences[name] = array("d",(values[j]-self.node_vectors[direct][j] for j in range(m)))
        difference_sha = self._bundle(self.node_root/"raw-axis-comparisons",differences,
            dict(lw_manifest_sha256=receipt.manifest_sha256, solved_manifest_sha256=self.solved_sha))
        self._json(self.node_root/"pre-lw-complete.json", dict(
            status="captured_before_unchanged_strict_LW", properties_accepted=False,
            capture=str(receipt.path), lw_manifest_sha256=receipt.manifest_sha256,
            solved_manifest_sha256=self.solved_sha, anchor_manifest_sha256=self.anchor_sha,
            axis_comparison_manifest_sha256=difference_sha,
            provenance_repr=repr(provenance), source_repr=repr(source),
            actual_written_bytes=self.written))
        self.phase = "captured"
