# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Explicit paid profile into the unchanged shared own-SCF science pipeline."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("_ccasp_inferno_complete_validator",
    Path(__file__).with_name("pdef_driver.py"))
parent_driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent_driver)
validate_complete = parent_driver.validate_complete


def capture_science(runtime, psi4, binding, runtime_sha, derivative_sha):
    runtime.compute_guard()
    profile = runtime.module("_ccasp_inferno_profile", "pdef_inferno_profile.py")
    cfg, identity = profile.configuration(runtime)
    if binding.get("effective_profile") != identity:
        raise ValueError("effective inferno profile binding differs")
    from psi4.driver.procrouting import isapol_lw
    if isapol_lw.PRODUCTION_TOLERANCE != cfg["response"]["production_tolerance"]:
        raise ValueError("unchanged production LW tolerance required")
    driver = runtime.module("_ccasp_inferno_shared_science", "integrated_driver.py")
    runtime.write(Path(runtime.PLAN["run"])/"results/effective-profile.json", identity)
    result = driver.capture_science(runtime, psi4, binding, runtime_sha, derivative_sha,
        configuration=cfg, profile_identity=identity, validate_result=validate_complete)
    return dict(result, effective_profile=identity)
