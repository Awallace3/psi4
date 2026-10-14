"""Private quadrature artifacts; numpy is loaded only inside audit operations.

No point cloud or raw geometry from these files belongs in the served dashboard.
Keep duplicates, negative weights and zero-weight entries in the original export.
"""
import hashlib
import json
from pathlib import Path

IDENTITY_FIELDS = ("engine", "system", "functional", "geometry_sha256", "basis",
                   "grid_mode", "config_sha256", "build_sha256")


def validate_identity(identity):
    for key in IDENTITY_FIELDS:
        if not isinstance(identity.get(key), str) or not identity[key]:
            raise ValueError(f"Missing/invalid grid provenance: {key}")
    if identity["grid_mode"] not in ("nominal", "native-default"):
        raise ValueError("Unknown grid provenance mode")


def grid_arrays(coords, weights):
    import numpy as np
    coords = np.asarray(coords, dtype="<f8")
    weights = np.asarray(weights, dtype="<f8").reshape(-1)
    if coords.ndim != 2 or coords.shape != (weights.size, 3) or not weights.size:
        raise ValueError("Expected nonempty (N, 3) coordinates and N weights")
    if not np.isfinite(coords).all() or not np.isfinite(weights).all():
        raise ValueError("Grid contains nonfinite coordinates or weights")
    if not np.count_nonzero(weights):
        raise ValueError("Grid has no nonzero integration weights")
    return coords, weights


def grid_fingerprint(coords, weights):
    """Exact, permutation-independent multiset, excluding zero-weight entries.

    Excluding zero weights removes padding's contribution, not its stored count.
    It does not equate nonzero count with physical/unpadded generated point count.
    """
    import numpy as np
    coords, weights = grid_arrays(coords, weights)
    pairs = np.column_stack((coords, weights)).astype("<f8", copy=False)
    # Signed zero has identical quadrature meaning; normalize it for hashing.
    pairs[pairs == 0] = 0
    ordered = hashlib.sha256(pairs.tobytes()).hexdigest()
    active = pairs[weights != 0]
    indices = np.lexsort(tuple(active[:, col] for col in (3, 2, 1, 0)))
    canonical = np.ascontiguousarray(active[indices], dtype="<f8")
    return dict(
        stored_points=int(weights.size),
        nonzero_weight_points=int(np.count_nonzero(weights)),
        zero_weight_points=int(np.count_nonzero(weights == 0)),
        negative_weight_points=int(np.count_nonzero(weights < 0)),
        ordered_pairs_sha256=ordered,
        nonzero_pairs_multiset_sha256=hashlib.sha256(canonical.tobytes()).hexdigest(),
        coordinate_unit="bohr", weight_unit="bohr^3",
        fingerprint_rule="float64-le; signed zeros normalized; nonzero pair multiset; duplicates retained",
    )


def export_grid(path, coords, weights, identity, *, phase, settings, known_padding_points=None):
    """Save immutable raw arrays and a content fingerprint, outside headline runs."""
    import numpy as np
    path = Path(path)
    validate_identity(identity)
    if not isinstance(settings, dict) or not settings:
        raise ValueError("Effective grid settings must be recorded")
    coords, weights = grid_arrays(coords, weights)
    info = grid_fingerprint(coords, weights)
    if known_padding_points is not None:
        if not 0 <= known_padding_points <= info["zero_weight_points"]:
            raise ValueError("Padding count cannot exceed zero-weight count")
        info["known_padding_points"] = int(known_padding_points)
    # Only whitelist provenance: never copy the whole worker record/config.
    info["identity"] = {key: identity[key] for key in IDENTITY_FIELDS}
    info["effective_settings"] = settings
    info["phase"] = phase
    info["diagnostic_only"] = True
    # Exclusive creation protects earlier exports, even on a partial audit retry.
    with path.open("xb") as stream:
        np.savez(stream, coords_bohr=coords, total_weights_bohr3=weights)
    with path.with_suffix(".json").open("x") as stream:
        json.dump(info, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return info


def compare_exports(left, right):
    """Do not infer equality from equal settings, counts, file names or headers."""
    import numpy as np
    arrays, metadata = [], []
    for path in (Path(left), Path(right)):
        with np.load(path, allow_pickle=False) as data:
            coords, weights = grid_arrays(data["coords_bohr"], data["total_weights_bohr3"])
        info = json.loads(path.with_suffix(".json").read_text())
        validate_identity(info["identity"])
        if not isinstance(info.get("effective_settings"), dict) or not info["effective_settings"]:
            raise ValueError("Missing effective grid settings")
        fingerprint = grid_fingerprint(coords, weights)
        if any(info.get(key) != value for key, value in fingerprint.items()):
            raise ValueError("Grid artifact fingerprint mismatch")
        arrays.append(fingerprint)
        metadata.append(info)
    for key in ("system", "functional", "geometry_sha256", "basis"):
        if metadata[0]["identity"][key] != metadata[1]["identity"][key]:
            raise ValueError(f"Grid identity mismatch: {key}")
    same = arrays[0]["nonzero_pairs_multiset_sha256"] == arrays[1]["nonzero_pairs_multiset_sha256"]
    counts = [info["nonzero_weight_points"] for info in arrays]
    smaller = None if counts[0] == counts[1] else metadata[counts.index(min(counts))]["identity"]["engine"]
    native_defaults = all(info["identity"]["grid_mode"] == "native-default" for info in metadata)
    return dict(
        identical_nonzero_quadrature=same,
        comparison_rule="exact float64 coordinate/total-weight multiset; order and zero-weight padding ignored",
        native_default_comparison=native_defaults,
        smaller_nonzero_grid=smaller,
        smaller_native_default_grid=smaller if native_defaults else None,
        left_provenance=metadata[0]["identity"], right_provenance=metadata[1]["identity"],
        left=arrays[0], right=arrays[1],
        note="Smaller active grid is not necessarily more accurate. This audit alone does not certify completed SCF runs.",
    )


def apply_pyscf_grid_settings(mf, config):
    mode = config.get("grid_mode", "nominal")
    if mode == "native-default":
        # In particular, retain the native density-dependent cutoff.
        return
    if mode != "nominal":
        raise ValueError(f"Unsupported grid mode: {mode}")
    mf.grids.atom_grid = (config["radial_points"], config["angular_points"])
    mf.grids.prune = None
    mf.small_rho_cutoff = 0.0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare private exported XC quadratures, not nominal grid settings")
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare_exports(args.left, args.right), indent=2, sort_keys=True))
