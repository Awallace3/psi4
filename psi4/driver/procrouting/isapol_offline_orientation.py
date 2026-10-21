"""Focused offline supplied-local adapter; no SCF/response/fit entry points.

Real Racah/noCS z,x,y. Frames have local-to-global columns. Changing a frame
does not change local tensors (nor local recoupled coefficients). This module
delegates all orientation-resolved contractions to the existing actual core.
"""
from decimal import Decimal
import hashlib
import struct
import numpy as np

COMPONENTS = ["10", "11c", "11s", "20", "21c", "21s", "22c", "22s"]


def require_reciprocal(array, context):
    """Admit a full declared-rank2 matrix, never average or zero-fill."""
    a = np.asarray(array, dtype=float)
    if a.shape != (8, 8) or not np.isfinite(a).all():
        raise ValueError("full finite rank2 8x8 response required: " + context)
    if not np.array_equal(a, a.T):
        raise ValueError("exact reciprocity required: " + context)


def _matrix(core, array):
    a = np.asarray(array, dtype=float)
    if a.ndim != 2:
        raise ValueError("response must be a matrix")
    m = core.Matrix("offline full local response", *a.shape)
    for i, j in np.ndindex(a.shape):
        m.set(i, j, float(a[i, j]))
    return m


def placed_model(core, sites, frequencies, rotation, translation, provenance):
    """Explicit rigid placement: o'=Q o+t, F'=Q F, alpha_local unchanged."""
    q, t = np.asarray(rotation, dtype=float), np.asarray(translation, dtype=float)
    if (q.shape != (3, 3) or t.shape != (3,) or not np.isfinite(q).all()
            or not np.isfinite(t).all()
            or not np.allclose(q.T @ q, np.eye(3), rtol=0, atol=1e-12)
            or abs(np.linalg.det(q) - 1) > 1e-12):
        raise ValueError("explicit proper rigid placement required")
    supplied = []
    for source in sites:
        site = core.IsaAnisotropicSite()
        site.label, site.ranks = source["label"], list(source["ranks"])
        site.origin = (q @ np.asarray(source["origin"]) + t).tolist()
        site.frame = q @ np.asarray(source["frame"])
        site.responses = [_matrix(core, a) for a in source["responses"]]
        supplied.append(site)
    return core.IsaAnisotropicModel(list(frequencies), supplied,
                                    "supplied_local_response", provenance)


def contract(core, model_a, model_b, weights, max_order=10):
    """Native orientation-resolved scalars/energies, not angular-table energies."""
    result = core.isa_anisotropic_dispersion(model_a, model_b, list(weights), max_order)
    pairs = []
    for pair in result.pairs:
        pairs.append(dict(
            site_a=pair.site_a, site_b=pair.site_b, displacement=list(pair.displacement),
            distance=pair.distance, total_energy=pair.truncated_energy,
            coefficients=[dict(order=c.order, value=c.value, energy=c.energy,
                               declared_model_complete=c.declared_model_complete,
                               unrestricted_complete=c.unrestricted_complete,
                               included_rank_quadruples=c.included_rank_quadruples,
                               missing_rank_quadruples=c.missing_rank_quadruples)
                          for c in pair.coefficients]))
    return dict(pairs=pairs, total_energy=result.truncated_energy,
                label="undamped/nonretarded/truncated-rank2 diagnostic energy"
                if all(s.ranks == [1, 2] for m in (model_a, model_b) for s in m.sites)
                else "undamped/nonretarded/declared-rank diagnostic energy")


def refined_records(properties, reference):
    """Admit the accepted full-export dialect, without recoupling or core work.

    The caller must authenticate the source records/files. No reconstruction
    from traces, local symmetry repair, rank promotion, or missing-site COPY.
    """
    def require(ok, message):
        if not ok:
            raise ValueError(message)
    labels = ["O", "H1", "H2"]
    sites = properties["sites"]
    require([s["label"] for s in sites] == labels, "native sites")
    require(properties["components"] == COMPONENTS, "native components")
    require(all(s["rank_limit"] == 2 for s in sites), "declared rank2 required")
    f, w = properties["frequencies"], properties["cp_weights"]
    require(len(f) == len(w) == 11 and f[0] == w[0] == 0, "11-node grid/static weight")
    require(all(np.isfinite(f)) and all(np.isfinite(w)) and
            all(b > a for a, b in zip(f, f[1:])) and all(x > 0 for x in w[1:]), "grid")
    checksum = hashlib.sha256(struct.pack("<22d", *(list(f)+list(w)))).hexdigest()
    require(checksum == properties["quadrature_provenance"]["source_sha256"], "native CP checksum")
    cf, cw = reference["grid"]["frequencies"], reference["grid"]["cp_weights"]
    require(cf == f and len(cw) == 11 and cw[0] == 0 and
            all(np.isfinite(cw)) and all(x > 0 for x in cw[1:]), "C source grid")
    require([x["node"] for x in reference["frequency_associations"]] == list(range(11)),
            "C node association")
    require([x["node"] for x in properties["refined"]] == list(range(11)), "P refined node association")
    require(len(reference["refined"]) == 11, "C refined nodes")
    require([s["label"] for s in properties["core_input"]] == labels, "P core sites")
    require([s["label"] for s in reference["public_frames"]] == labels, "C public sites")
    psites, csites = [], []
    for i, (public, supplied, cpublic) in enumerate(zip(
            sites, properties["core_input"], reference["public_frames"], strict=True)):
        label = public["label"]
        require(supplied["ranks"] == [1, 2] and supplied["components"] == COMPONENTS and
                supplied["shape"] == [11, 8, 8] and len(supplied["responses"]) == 11 and
                len(supplied["response_sha256"]) == 11, "P full rank2 core input")
        require(public["origin_bohr"] == supplied["origin_bohr"] == cpublic["xyz_bohr"] and
                public["frame"] == supplied["frame"] == cpublic["frame"], "public origin/frame mismatch")
        pa, ca = [], []
        for node in range(11):
            a = np.array(supplied["responses"][node], dtype=float, copy=True)
            require_reciprocal(a, f"P/refined/node{node}/{label}")
            require(hashlib.sha256(a.astype("<f8").tobytes()).hexdigest() ==
                    supplied["response_sha256"][node], "P tensor hash")
            refined = np.asarray(properties["refined"][node]["values"][i])
            require(refined.shape == (9, 9) and np.array_equal(refined[1:, 1:], a),
                    "P full refined/core input disagree")
            blocks = reference["refined"][node]["blocks"]
            require([b["site"] for b in blocks] == labels, "all three actual C site tensors required")
            b = blocks[i]
            require(b["rank"] == 2 and b["components"] == COMPONENTS, "C declared rank/components")
            tokens = b["tokens"]
            require(len(tokens) == 8 and all(len(row) == 8 for row in tokens), "C full matrix shape")
            decimals = [[Decimal(v) for v in row] for row in tokens]
            require(all(x.is_finite() for row in decimals for x in row), "C nonfinite token")
            require(all(decimals[j][k] == decimals[k][j] for j in range(8) for k in range(8)),
                    f"exact reciprocity required: C/refined/node{node}/{label}; no reconstruction admitted")
            tensor = np.array(tokens, dtype=float)
            require_reciprocal(tensor, f"C/refined/node{node}/{label}")
            pa.append(a)
            ca.append(tensor)
        common = dict(label=label, ranks=[1, 2], origin=list(public["origin_bohr"]),
                      frame=np.array(public["frame"], dtype=float))
        psites.append(dict(**common, responses=pa))
        csites.append(dict(**common, responses=ca))
    return {
        "P": dict(frequencies=list(f), weights=list(w), sites=psites,
                  qualification="sealed native refined full rank2; actual native CP weights"),
        "C": dict(frequencies=list(cf), weights=list(cw), sites=csites,
                  qualification="sealed C printed refined full rank2; source-reconstructed CP weights; "
                                "NOT observed engine weights or exact grid-equivalence proof"),
    }
