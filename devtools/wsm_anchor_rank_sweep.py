"""Sweep the WSM refinement's anchor penalty over both of its axes.

Analysis-time devtool.  It reads ``.camcasp-reference/`` artifacts, so it is *not*
importable from production code or from pytest; nothing under ``psi4/`` or
``tests/`` may depend on it.

The questions it answers. The WSM refinement anchors only the rank-1 blocks by
default, and Stone's g_pp' penalty (eqn 9.3.13) is introduced in Sec. 9.3.4 for
exactly the buried-atom conditioning failure that our rank-2/rank-3 blocks show.

1. Does extending the anchor over the higher blocks recover the missing rank-2 and
   rank-3 polarizability that the low C8/C10/C12 dispersion coefficients point at?
   (It does not -- it pins them to the anchor, which is why the rank axis is swept
   under both weight conventions rather than only the default one.)
2. The ISA-Pol paper's eqn (22) replaces the flat unit weight with a self-scaling
   one, ``g_kk' = delta_kk' w0 / (1 + (p0_k)^2)``.  Dropped in as published that
   changes two things at once -- the rescaling, and the loss of the rank gate,
   because the published sum runs over every fitted parameter.  Which of the two
   is responsible for what?  The ``gated`` arms apply the rescaling behind the gate
   and separate them.

What it does, for each arm:

* rebuilds the active-variable mask and the equality (COPY-class) rows from
  ``H2O.pdef`` -- 360 variables, 170 active, 66 equality rows, 104 independent;
* takes the anchor from the CamCASP localized model ``H2O_L3_0f10.pol``, rotated
  out of the per-site local axes into the molecular frame;
* drives the shipped C++ refinement over the 500-point reference grid and
  point-to-point response in ``H2O_000.p2p``;
* reports the rank invariants a_l = Tr(alpha^ll)/(2l+1) per site, against both the
  localized anchor and CamCASP's own refined ``H2O_ref_wt4_L3`` model.

The ``gated`` arm at limit 3 admits every block of a rank-3 model, so it must
reproduce the ``isapol`` arm exactly; a divergence there means the gate leaks.

Usage::

    PYTHONPATH=<repo>/build_camcasp/stage/lib python wsm_anchor_rank_sweep.py out.json

Run it from outside the repository root, or the source ``psi4/`` package shadows
the built extension.  It needs a checkout that carries both ``.camcasp-reference/``
and the ``parse_p2p`` reader in ``devtools/camcasp_reference.py``; set
``CAMCASP_PSI4_ROOT`` when that checkout is not this file's own repository.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("CAMCASP_PSI4_ROOT", Path(__file__).resolve().parent.parent))

# Import psi4 from the staged build *before* the repository root joins sys.path,
# or the source psi4/ package shadows the extension and the failure surfaces as a
# misleading "cannot import name 'core' ... circular import".
import psi4  # noqa: E402

sys.path.append(str(REPO))
from devtools import camcasp_reference as cr  # noqa: E402

for _symbol in ("parse_p2p", "parse_pdef", "parse_alpha_sections",
                "build_local_frames", "l3_local_to_molecular", "CANONICAL_ATOMS"):
    if not hasattr(cr, _symbol):
        raise SystemExit(
            f"{REPO}/devtools/camcasp_reference.py has no {_symbol}; point "
            "CAMCASP_PSI4_ROOT at a checkout of the camcasp branch"
        )

WORK = REPO / ".camcasp-reference/work/H2O-isagrid"
SITES = ("O", "H1", "H2")
COMPONENT_INDEX = {name: i for i, name in enumerate(cr.REAL_REFINED_COMPONENTS_L3)}
# The physical multipole ranks of an L3 model.  Used both for the a_l invariants and
# as the range of the anchor gate.
RANKS = (1, 2, 3)

# (label, anchor_scaling, anchor_rank_limit).  The rank axis alone cannot separate the
# two things the published ISA-Pol weight changes at once -- it rescales the penalty by
# 1/(1 + p0^2) AND, because the published sum runs over every fitted parameter, it drops
# the rank gate.  The gated rows apply the rescaling behind the gate so the two can be
# attributed separately; the gated row at limit 3 must reproduce the ungated row exactly,
# which is the self-consistency check on the gate.
ARMS = (
    ("unit  r1", "unit", 1),
    ("unit  r2", "unit", 2),
    ("unit  r3", "unit", 3),
    ("gated r1", "inverse_reference_norm_gated", 1),
    ("gated r2", "inverse_reference_norm_gated", 2),
    ("gated r3", "inverse_reference_norm_gated", 3),
    ("isapol --", "inverse_reference_norm", 1),
)

# CamCASP's own refined wt4 L3 model, the target the refinement is trying to hit.
# Recorded in devtools/wsm_hermetic_parity.hermetic.json -> rank_invariants.
REFERENCE_REFINED = {
    "O": (7.214899017611334, 25.3108017309632, 157.0931137808487),
    "H1": (1.1962588355516666, 3.1002525289965996, 19.239002862478998),
    "H2": (1.1962588355516666, 3.1002525289965996, 19.239002862478998),
}

INPUTS = ("H2O.pdef", "H2O.axes", "H2O_L3_0f10.pol", "H2O_000.p2p")


def upper_index(first: int, second: int) -> int:
    """Index of the upper-triangle entry ``(first, second)`` of a 15x15 block."""
    if not 0 <= first <= second < 15:
        raise ValueError(f"({first}, {second}) is not an upper-triangle pair")
    return first * 15 - first * (first - 1) // 2 + (second - first)


def rank_slice(rank: int) -> slice:
    """Diagonal block of rank ``rank``; the L3 offsets are ``l*l - 1``."""
    return slice(rank * rank - 1, (rank + 1) * (rank + 1) - 1)


def rank_invariants(tensor: np.ndarray) -> tuple[float, ...]:
    return tuple(
        float(np.trace(tensor[rank_slice(rank), rank_slice(rank)]) / (2 * rank + 1))
        for rank in RANKS
    )


def build_anchor() -> tuple[dict[str, tuple[float, float, float]], list[np.ndarray]]:
    """Molecular-frame 15x15 anchors from the CamCASP localized L3 model."""
    geometry = {atom["label"]: tuple(atom["xyz"]) for atom in cr.CANONICAL_ATOMS}
    frames = cr.build_local_frames(geometry, (WORK / "H2O.axes").read_text())
    block = cr.parse_alpha_sections(WORK / "H2O_L3_0f10.pol", atom_labels=SITES)[0]
    anchors = []
    for site in SITES:
        molecular = np.array(cr.l3_local_to_molecular(block.atoms[site], frames[site]))
        # The .pol carries seven significant figures, so the written block is
        # asymmetric at 5e-12 on entries of order 200.  That is ASCII round-off,
        # not physics, but the refinement requires an exactly symmetric reference.
        asymmetry = float(np.abs(molecular - molecular.T).max())
        if asymmetry > 1.0e-9:
            raise SystemExit(
                f"{site}: localized anchor asymmetry {asymmetry:.3e} exceeds round-off"
            )
        anchors.append(0.5 * (molecular + molecular.T))
    return geometry, anchors


def build_constraints() -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Active mask and equality rows from the pdef's parameter/COPY classes.

    Every parameter name that appears more than once names a set of variables the
    pdef declares equal, so a class of ``k`` entries contributes ``k - 1`` rows.
    """
    pdef = cr.parse_pdef(WORK / "H2O.pdef")
    variables = 120 * len(SITES)
    active = np.zeros(variables, dtype=bool)
    rows = []
    for entries in pdef.values():
        members = []
        for site, first, second in entries:
            left, right = sorted((COMPONENT_INDEX[first], COMPONENT_INDEX[second]))
            index = SITES.index(site) * 120 + upper_index(left, right)
            active[index] = True
            members.append(index)
        for other in members[1:]:
            row = np.zeros(variables)
            row[members[0]] = 1.0
            row[other] = -1.0
            rows.append(row)
    equality = np.asarray(rows) if rows else np.empty((0, variables))
    return active, equality, [0.0] * len(rows)


def as_matrix(values, columns: int | None = None):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1 and values.size == 0:
        values = np.empty((0, columns), dtype=float)
    return psi4.core.Matrix.from_array(values)


def main() -> None:
    # 500 fit points give 125250 pair rows; the economy SVD over the active
    # columns does not fit in the 500 MB default reservation.
    psi4.set_memory("32 GB")

    geometry, anchors = build_anchor()
    sites = np.array([geometry[site] for site in SITES])
    active, equality, targets = build_constraints()
    variables = active.size
    independent = int(active.sum()) - len(targets)
    print(f"mask: variables {variables}  active {int(active.sum())}  "
          f"equality rows {len(targets)}  independent {independent}")
    print("per-site active:",
          {site: int(active[k * 120:(k + 1) * 120].sum()) for k, site in enumerate(SITES)})

    points, response = cr.parse_p2p(WORK / "H2O_000.p2p")
    points = np.asarray(points)
    response = np.asarray(response)
    print(f"p2p: {len(points)} points, response {response.shape}")

    anchor_invariants = {site: rank_invariants(anchors[k]) for k, site in enumerate(SITES)}

    sweep = {}
    for label, scaling, limit in ARMS:
        block = psi4.core._atomic_polarizability_test_refine_wsm(
            as_matrix(points), [0.0], [as_matrix(response)], as_matrix(sites),
            [as_matrix(anchor) for anchor in anchors], [0.0], list(active),
            as_matrix(equality, variables), targets,
            {"cutoff": 0.0, "anchor_rank_limit": limit,
             "anchor_scaling": scaling},
        )[0]
        tensors = [np.array(tensor) for tensor in block["tensors"]]
        sweep[label] = {
            "anchor_scaling": scaling,
            "anchor_rank_limit": limit,
            "anchor_variable_count": block["anchor_variable_count"],
            "active_variable_count": block["active_variable_count"],
            "condition_number": block["condition_number"],
            "weighted_residual_norm": block["weighted_residual_norm"],
            "anchor_residual_norm": block["anchor_residual_norm"],
            "constraint_residual_norm": block["constraint_residual_norm"],
            "objective_residual_norm": block["objective_residual_norm"],
            "max_point_residual": block["max_point_residual"],
            "max_output_asymmetry": block["max_output_asymmetry"],
            "invariants": {site: rank_invariants(tensors[k])
                           for k, site in enumerate(SITES)},
        }

    header = ("%-10s %8s %10s %11s %10s | %-23s | %-23s"
              % ("arm", "anchored", "cond", "maxPtResid", "anchResid",
                 "sum a1 / a2 / a3", "ratio to refined"))
    print()
    print(header)
    print("-" * len(header))
    reference_sum = [sum(REFERENCE_REFINED[s][l] for s in SITES) for l in range(3)]
    for label, record in sweep.items():
        total = [sum(record["invariants"][s][l] for s in SITES) for l in range(3)]
        ratio = [total[l] / reference_sum[l] for l in range(3)]
        print("%-10s %8d %10.4g %11.3e %10.3e | %6.3f %7.3f %8.2f | %6.4f %7.4f %7.4f"
              % (label, record["anchor_variable_count"], record["condition_number"],
                 record["max_point_residual"], record["anchor_residual_norm"],
                 total[0], total[1], total[2], ratio[0], ratio[1], ratio[2]))

    print()
    print("per-site a_l, with the ratio to CamCASP's refined wt4 L3 in parentheses:")
    for label, record in sweep.items():
        for site in SITES:
            got = record["invariants"][site]
            want = REFERENCE_REFINED[site]
            print("  %-10s %-3s a1 %8.4f (%.4f)  a2 %8.4f (%.4f)  a3 %9.4f (%.4f)"
                  % (label, site, got[0], got[0] / want[0], got[1], got[1] / want[1],
                     got[2], got[2] / want[2]))

    payload = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "manifest": {
            name: hashlib.sha256((WORK / name).read_bytes()).hexdigest()
            for name in INPUTS
        },
        "site_order": list(SITES),
        "mask": {
            "variable_count": variables,
            "active_variable_count": int(active.sum()),
            "equality_row_count": len(targets),
            "independent_variable_count": independent,
        },
        "point_count": len(points),
        "localized_anchor_invariants": {s: list(v) for s, v in anchor_invariants.items()},
        "reference_refined_invariants": {s: list(v) for s, v in REFERENCE_REFINED.items()},
        "sweep": {
            label: {**record,
                    "invariants": {s: list(v)
                                   for s, v in record["invariants"].items()}}
            for label, record in sweep.items()
        },
    }
    if len(sys.argv) > 1:
        destination = Path(sys.argv[1])
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print("\nwrote", destination)


if __name__ == "__main__":
    main()
