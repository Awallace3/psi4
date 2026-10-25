"""Independent frozen large-case dimensions, not an AO==MO rank assumption."""
import hashlib
import json
from prepare_inputs import EXPECTED,SYSTEMS,geometry_metadata


def load_point(path,checksum,system):
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=checksum:
        raise RuntimeError("Frozen input manifest changed")
    value=json.loads(raw)
    if set(value["systems"])!=set(SYSTEMS) or system not in SYSTEMS:
        raise ValueError("Exactly the three requested systems required")
    point=value["systems"][system]
    if hashlib.sha256(point["geometry"].encode()).hexdigest()!=point["geometry_sha256"]:
        raise RuntimeError("Frozen geometry changed")
    fragments=geometry_metadata(point["geometry"])
    if fragments!=point["fragments"]:
        raise RuntimeError("Geometry/fragment metadata differ")
    counts=point["spherical_basis_counts"]
    actual=(sum(f["atoms"] for f in fragments),counts["primary"],counts["jkfit"],
            counts["rifit"],tuple(f["occupied"] for f in fragments))
    if actual!=EXPECTED[system]:
        raise RuntimeError("Full-system input dimensions changed")
    occupied=[f["occupied"] for f in fragments]
    frozen=[f["frozen_occupied"] for f in fragments]
    dimensions={"atoms":actual[0],"ao":actual[1],"occupied":occupied,"frozen":frozen,
                "active":[o-f for o,f in zip(occupied,frozen)]}
    return point,dimensions
