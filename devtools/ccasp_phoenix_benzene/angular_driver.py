# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Explicit angular-full adapter to the same own-SCF physical pipeline."""
import math
from pathlib import Path


def validate_complete(result, quadrature, cfg):
    """Additional completeness checks, never a replacement for native physical gates."""
    frequencies = tuple(quadrature.frequencies)
    if (len(frequencies) != 11 or frequencies[0] != 0.
            or tuple(result.frequencies) != frequencies
            or any(len(items) != 11 for items in (
                result.local_tensors,result.refinements,result.refined_tensors,result.diagnostics))):
        raise ValueError("all eleven own response/LW/PFIT nodes required")
    hashes = {r.model.static_reference_sha256 for r in result.refinements}
    if (len(hashes) != 1 or not next(iter(hashes))
            or any(r.model.penalty_selection != "static_reference" for r in result.refinements)):
        raise ValueError("unchanged static PFIT selection required")
    for node in result.diagnostics:
        residual = node["localization_residual"]
        if not math.isfinite(residual) or residual > cfg["response"]["production_tolerance"]:
            raise ValueError("strict physical LW diagnostics required")
    if len(result.dispersion.pairs) != 144 or result.recoupled_dispersion is None:
        raise ValueError("isotropic and recoupled anisotropic properties required")
    if len(result.recoupled_dispersion.pairs) != 144:
        raise ValueError("complete ordered recoupled site pairs required")
    res = result.resources
    limits = cfg["resources"]
    for key, limit in (("maximum_numeric_plan",limits["psi4_memory_bytes"]),
                       ("charged_work",limits["max_work"]),("charged_io_bytes",limits["max_io_bytes"])):
        if not 0 <= res[key] <= limit:
            raise ValueError("complete-chain numerical resource receipt differs")


def capture_science(runtime, psi4, binding, runtime_sha, derivative_sha):
    runtime.compute_guard()
    profile = runtime.module("_ccasp_angular_profile","angular_profile.py")
    cfg,identity = profile.configuration(runtime)
    if binding.get("effective_profile") != identity:
        raise ValueError("effective angular profile binding differs")
    from psi4.driver.procrouting import isapol_lw
    if isapol_lw.PRODUCTION_TOLERANCE != cfg["response"]["production_tolerance"]:
        raise ValueError("unchanged production LW tolerance required")
    driver = runtime.module("_ccasp_shared_science","integrated_driver.py")
    runtime.write(Path(runtime.PLAN["run"])/"results/effective-profile.json",identity)
    result = driver.capture_science(runtime,psi4,binding,runtime_sha,derivative_sha,
        configuration=cfg,profile_identity=identity,validate_result=validate_complete)
    return dict(result,effective_profile=identity)
