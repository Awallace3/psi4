#!/usr/bin/env python3
"""Opt-in supplied-local atomic-alpha/C_n artifact (Python-only staged Psi4).

--model-a is a directory holding manifest-listed response files. With
--placement-b, B is explicitly a placed copy of A unless --model-b and
--manifest-b select another imported model. Placement JSON requires translation
(bohr) and rotation (proper Cartesian local-to-global, 3x3). Example translation
[0,0,10] with identity rotation is a demonstration, NOT archive pair geometry.
Output uses exclusive creation and finite-only JSON. Numerical comparison failures
remain in the artifact and do not change the successful computation exit status.
"""
import argparse
import hashlib
from dataclasses import replace
from pathlib import Path

from psi4.driver.procrouting import isapol_supplied as b


def pot_comparisons(result):
    """Only rank-zero 00/00/J=0 isotropic rows, never directional vs recoupled.

    The excerpt uses type H; require exact H1/H2 rank-trace equivalence before
    selecting the representative. Archive C12 remains rank-3 partial.
    """
    sources = [s for s in result.model_a.sources if s.role == "independent_pot_excerpt"]
    if not sources:
        return ()
    source, = sources
    data = b._json(source.text)
    expected_convention = "00 00 0 rotational scalar; compare only isotropic even orders; C12 rank3 partial"
    if data["convention"] != expected_convention:
        raise ValueError("unverified pot comparison convention")
    for row in data["rows"]:
        tokens = row["text"].split()
        if len(row["pair"]) != 2 or any(t not in {"O", "H"} for t in row["pair"]):
            raise ValueError("unsupported pot type pair")
        if tokens[:3] != ["00", "00", "0"] or len(tokens) != 10:
            raise ValueError("unsupported pot scalar row")
        for token in tokens[3:]:
            b._number(token)
    sites = result.model_a.sites
    scalars = result.model_a.atomic_scalars
    labels = {s.label: i for i, s in enumerate(sites)}
    import numpy as np
    if set(labels) != {"O", "H1", "H2"} or sites[labels["H1"]].ranks != sites[labels["H2"]].ranks or not np.array_equal(scalars[labels["H1"]], scalars[labels["H2"]]):
        return (b.Comparison("pot:type-H-mapping", None, None, None,
                             expected_convention, availability="unavailable",
                             reason="pot type-H mapping requires exact H1/H2 scalar equivalence; no representative selected"),)
    types = {"O": labels["O"], "H": labels["H1"]}
    pairs = {(p.site_a, p.site_b): p for p in result.isotropic}
    comparisons = []
    for row in data["rows"]:
        a, c = row["pair"]
        p = pairs[types[a], types[c]]
        tokens = row["text"].split()
        if tokens[:3] != ["00", "00", "0"] or len(tokens) != 10:
            raise ValueError("unsupported pot scalar row")
        for coef in p.coefficients:
            token = tokens[3+coef.order-6]
            expected = b._number(token)
            error = abs(coef.value-expected)
            tolerance = b.printed_tolerance(token)
            comparisons.append(b.Comparison(f"pot:{a}-{c}:C{coef.order}", error, tolerance,
                                             error <= tolerance, expected_convention, 1, error/tolerance))
    return tuple(comparisons)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--manifest-sha256", help="otherwise required adjacent .sha256")
    p.add_argument("--model-a", required=True, type=Path)
    p.add_argument("--model-b", type=Path)
    p.add_argument("--manifest-b", type=Path)
    p.add_argument("--manifest-b-sha256")
    p.add_argument("--placement-b", type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    if args.output.exists():
        p.error("output exists; exclusive creation required")
    if (args.model_b is None) != (args.manifest_b is None):
        p.error("--model-b and --manifest-b must be paired")
    a = b.read_orient_local_response(args.model_a, args.manifest, manifest_sha256=args.manifest_sha256)
    other = a if args.model_b is None else b.read_orient_local_response(
        args.model_b, args.manifest_b, manifest_sha256=args.manifest_b_sha256)
    placement, placement_source = None, None
    if args.placement_b:
        raw_bytes = args.placement_b.read_bytes()
        text = raw_bytes.decode("utf-8")
        placement_source = b.Source(args.placement_b.name, hashlib.sha256(raw_bytes).hexdigest(), text,
                                    "placement_b", str(args.placement_b.resolve()))
        raw = b._json(text)
        if set(raw) != {"translation", "rotation"}:
            p.error("placement requires exactly translation and rotation")
        placement = b.Placement(raw["translation"], raw["rotation"])
    result = b.supplied_local_properties(a, other, placement_b=placement, anisotropic=placement is not None)
    comparisons = b.compare_casimir_data(a) if any(s.role == "independent_casimir_data" for s in a.sources) else ()
    comparisons += b.compare_frequency_headers(a)
    # The archived pot is A/A, never apply it to a different B model.
    if args.model_b is None:
        comparisons += pot_comparisons(result)
    result = replace(result, comparisons=comparisons, placement_source=placement_source)
    payload = result.to_json()
    with args.output.open("x") as f:
        f.write(payload)
    print(f"computed={result.computed_status}; anisotropic={result.anisotropic_status}; "
          f"numerical_agreement={result.numerical_agreement}; output={args.output}")


if __name__ == "__main__":
    main()
