# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Explicit full pdef adapter; shared own-SCF driver and strict physical gates."""
import importlib.util
from pathlib import Path


def _local(file):
    spec = importlib.util.spec_from_file_location("_pdef_" + file[:-3], Path(__file__).with_name(file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


declared_variables = _local("pdef_profile.py").declared_variables
validate_angular_complete = _local("angular_driver.py").validate_complete


def validate_complete(result, quadrature, cfg):
    """Check actual labels AND COPY entries, plus one invariant static mask."""
    validate_angular_complete(result, quadrature, cfg)
    labels = declared_variables()
    if tuple(cfg["refinement"]["declared_variables"]) != labels:
        raise ValueError("exact 72-variable declaration required")
    first = result.refinements[0].model
    sites = first.sites
    if (tuple(s.label for s in sites) != tuple(f"{t}{n}" for t in ("C", "H") for n in range(1, 7))
            or any(s.rank_limit != 2 or s.site_type != sites[0 if i < 6 else 6].site_type
                   for i, s in enumerate(sites))
            or sites[0].site_type == sites[6].site_type):
        raise ValueError("exact benzene rank2 COPY ownership required")
    entries = tuple(tuple((s, i, j) for s in members)
                    for members in (range(6), range(6, 12))
                    for i in range(1, 9) for j in range(i, 9))
    mask = tuple(name not in first.unpenalized_variables for name in labels)
    for fit in result.refinements:
        m = fit.model
        if (m.parameter_count != 72 or m.parameter_labels != labels
                or tuple(m.declared_variables) != labels or m.parameter_entries != entries
                or m.sites != sites or m.channel_count != 108
                or m.reference_sites != (0, 6) or m.equivalent_sites != (tuple(range(6)), tuple(range(6, 12)))
                or len(m.anchors) != 72 or len(m.strengths) != 72
                or len(set(m.unpenalized_variables)) != len(m.unpenalized_variables)
                or not set(m.unpenalized_variables) <= set(labels)
                or tuple(name not in m.unpenalized_variables for name in labels) != mask):
            raise ValueError("all 72 actual labels/entries and same static mask required")
        for i, selected in enumerate(mask):
            if not selected and (m.anchors[i] != 0. or m.strengths[i] != 0.):
                raise ValueError("static-unselected variables must remain fitted free")


def capture_science(runtime, psi4, binding, runtime_sha, derivative_sha):
    runtime.compute_guard()
    profile = runtime.module("_ccasp_pdef_profile", "pdef_profile.py")
    cfg, identity = profile.configuration(runtime)
    if binding.get("effective_profile") != identity:
        raise ValueError("effective full pdef profile binding differs")
    from psi4.driver.procrouting import isapol_lw
    if isapol_lw.PRODUCTION_TOLERANCE != cfg["response"]["production_tolerance"]:
        raise ValueError("unchanged production LW tolerance required")
    driver = runtime.module("_ccasp_shared_science", "integrated_driver.py")
    runtime.write(Path(runtime.PLAN["run"])/"results/effective-profile.json", identity)
    result = driver.capture_science(runtime, psi4, binding, runtime_sha, derivative_sha,
        configuration=cfg, profile_identity=identity, validate_result=validate_complete)
    return dict(result, effective_profile=identity)
