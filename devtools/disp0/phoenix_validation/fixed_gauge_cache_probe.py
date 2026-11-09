"""Fixed-snapshot blocked-dispersion diagnostic; never a full F-SAPT speedup."""
import json
import os
from pathlib import Path
import time

import numpy as np

from stage_scaling import digest, write

SCOPE = "FIXED_GAUGE_DISPERSION_CACHE_ONLY_NOT_SCF_FSAPT_OR_SPEEDUP_QUALIFICATION"
CAPS = (64*1024**2, 4*1024**3)
ORDER = (0, 1, 1, 0)
PAIR_FIELDS = ("Disp20 occupied pairs", "Exch-Disp20 occupied pairs", "Disp0 occupied pairs")
SCALARS = ("Disp20,u", "Exch-Disp20,u", "Disp0,u")


def cache_plan(q, a, b, nr, ns, block, budget):
    if (any(type(n) is not int or n < 1 for n in (q,a,b,nr,ns,block))
            or type(budget) is not int or budget < 0):
        raise ValueError("Positive dimensions and explicit nonnegative cache cap required")
    r, s = (nr+block-1)//block, (ns+block-1)//block
    used = selected = 0
    if r > 1:
        for start in range(0, ns, block):
            size = 8*(2*q+2)*min(block,ns-start)*(a+b)
            if size <= budget-used:
                used += size
                selected += 1
    return {"allocated_bytes":used, "cached_slabs":selected,
            "hits":(r-1)*selected, "r_factor_batches":r,
            "s_factor_batches":r*s-(r-1)*selected,
            "native_batches":r+r*s-(r-1)*selected, "compute_blocks":r*s}


def compare_components(left, right, shape):
    errors = {}
    for name in PAIR_FIELDS+SCALARS:
        a, b = np.asarray(left[name]), np.asarray(right[name])
        expected = shape if name in PAIR_FIELDS else ()
        if a.shape != expected or b.shape != expected or not (
                np.isfinite(a).all() and np.isfinite(b).all()):
            raise ValueError("Incomplete/nonfinite blocked output: "+name)
        # Unchanged within-plan reordering/cache check, not an SCF-repeat waiver.
        np.testing.assert_allclose(b,a,rtol=1e-12,atol=1e-12,err_msg=name)
        errors[name] = float(np.max(np.abs(b-a),initial=0.))
    return errors


def run(runtime, manifest, output):
    from scf_repeat_probe import require_reference
    from full_inputs import load_point
    from psi4.driver.procrouting.sapt.sapt_fisapt_workflow_cuda import HF_FIELDS
    from psi4.driver.procrouting.sapt.sapt_dispersion_cuda_setup import prepare_hf
    from psi4.driver.procrouting.sapt.sapt_dispersion_cuda import blocked_device_dispersion

    doc = require_reference(manifest)
    reference = Path(manifest["reference_result"]).parent
    loaded = {}
    for name in ("raw-hf.npz", "fragment-matrices.npz"):
        path = reference/name
        if digest(path) != doc["artifact_sha256"][name]:
            raise RuntimeError("Captured fixed-wavefunction payload changed")
        with np.load(path,allow_pickle=False) as arrays:
            loaded[name] = dict(arrays)
    raw, fields = loaded["raw-hf.npz"], loaded["fragment-matrices.npz"]
    dims = doc["orbital_dimensions"]
    point, expected = load_point(Path(manifest["inputs"]),manifest["inputs_sha256"],"protein83")
    if expected != dims or point["geometry_sha256"] != doc["geometry_sha256"]:
        raise RuntimeError("Frozen input differs from fixed-wavefunction capture")
    core = runtime.psi4.core
    core.set_output_file(str(output/"psi4.out"),False)
    runtime.psi4.set_memory(runtime.config["host_memory"])
    runtime.psi4.set_num_threads(runtime.config["threads"])
    options = dict(doc["options"])
    options["fisapt_fsapt_filepath"] = str(output/"unused-fsapt")
    runtime.psi4.set_options(options)
    molecule = runtime.psi4.geometry(doc["geometry"])
    molecule.update_geometry()
    inputs = json.loads(Path(manifest["inputs"]).read_text())
    for item in inputs["basis_files"].values():
        if digest(Path(core.get_datadir())/"basis"/item["filename"]) != item["sha256"]:
            raise RuntimeError("Frozen installed basis changed")
    primary = core.BasisSet.build(molecule,"BASIS",options["basis"],puream=1)
    rifit = core.BasisSet.build(molecule,"DF_BASIS_MP2",options["df_basis_mp2"],
                               "RIFIT",options["basis"],puream=1)
    if (primary.nbf(),rifit.nbf()) != (dims["ao"],point["spherical_basis_counts"]["rifit"]):
        raise RuntimeError("Fixed AO/RIFIT dimensions changed")
    matrices = {k:np.ascontiguousarray(raw[k]) for k in HF_FIELDS}
    energies = {k:np.ascontiguousarray(raw[name]) for k,name in zip(
        "abrs",("eps_occ_A","eps_occ_B","eps_vir_A","eps_vir_B"))}
    rotations = []
    for side,count,nf in zip("AB",dims["occupied"],dims["frozen"]):
        u, co, locc = fields["Uocc0"+side], raw["Cocc_"+side], fields["Locc_"+side]
        if u.shape != (count,count) or locc.shape != co.shape:
            raise ValueError("Complete captured IBO gauge required")
        np.testing.assert_allclose(u.T@u,np.eye(count),rtol=1e-8,atol=1e-8)
        np.testing.assert_allclose(co@u,locc,rtol=1e-9,atol=1e-9)
        np.testing.assert_allclose(u[:nf,nf:],0.,rtol=0,atol=1e-8)
        np.testing.assert_allclose(u[nf:,:nf],0.,rtol=0,atol=1e-8)
        rotations.append(np.ascontiguousarray(u[nf:,nf:]))
    report = {"status":"STARTED", "scope":SCOPE, "accepted_speedup_count":0,
              "scf_executed":False, "fisapt_executed":False,
              "science_pin":manifest["science_pin"], "pid":os.getpid(),
              "reference_result_sha256":manifest["reference_result_sha256"],
              "snapshot_artifact_sha256":doc["artifact_sha256"],
              "pair_rtol":1e-12, "pair_atol":1e-12, "runs":[],
              "cache_caps":list(CAPS), "order":list(ORDER),
              "gauge_scope":"Captured IBO gauge is diagnostic input only, not production"}
    try:
        retained = {}
        captured = {}
        with runtime.provider(primary,rifit) as provider, provider:
            provider.prepare_device_compute()
            session = provider.session
            report["device"] = runtime.device_identity(session)
            native = provider.report()
            report["native_policy"] = {k:native[k] for k in (
                "producer","native_version","declared_plan_policy","native_math_mode")}
            report["fixed_dimensions"] = dims
            report["rifit_nbf"] = rifit.nbf()
            with session.scope():
                spaces, corrections, eps = prepare_hf(session,matrices,energies,
                    frozen_a=dims["frozen"][0],frozen_b=dims["frozen"][1])
                dr = tuple(session.upload(v) for v in rotations)
                shape = tuple(dims["active"])
                a,nr,b,ns = (eps[k].size for k in "arbs")
                for index, arm in enumerate(ORDER):
                    budget = CAPS[arm]
                    plan = cache_plan(provider.naux,a,b,nr,ns,24,budget)
                    before, live = provider.calls, session.live_bytes
                    producer_before = provider.elapsed
                    session.synchronize_device()
                    started = time.perf_counter()
                    upload = session.upload
                    def forbidden(*_,**__):
                        raise RuntimeError("Blocked path attempted host numerical upload")
                    session.upload = forbidden
                    try:
                        result = blocked_device_dispersion(provider,spaces,corrections,eps,
                            block_size=24,rotations=dr,factor_cache_bytes=budget)
                        session.synchronize_device()
                    finally:
                        session.upload = upload
                    elapsed = time.perf_counter()-started
                    execution = result["Disp0 execution"]
                    if (session.live_bytes != live
                            or provider.calls-before != plan["native_batches"]
                            or execution["compute_blocks"] != plan["compute_blocks"]
                            or any(execution["factor_cache"][k] != plan[k] for k in (
                                "allocated_bytes","cached_slabs","hits",
                                "r_factor_batches","s_factor_batches"))
                            or session.peak_live_bytes+session.native_bytes >
                               runtime.config["device_memory_limit_bytes"]):
                        raise RuntimeError("Changed work count, residency, budget or incomplete cleanup")
                    row = {"index":index, "arm":arm, "cache_cap_bytes":budget,
                           "wall_seconds":elapsed, "work":plan,
                           "owned_peak_device_bytes_session_lifetime":session.peak_live_bytes,
                           "live_bytes_before_and_after":live,
                           "native_persistent_bytes":session.native_bytes,
                           "producer_wall_seconds":provider.elapsed-producer_before,
                           "signed_components":{k:result[k] for k in SCALARS}}
                    if retained:
                        row["maximum_abs_errors_from_first"] = compare_components(
                            retained,result,shape)
                    else:
                        compare_components(result,result,shape)
                        if not np.any(result[PAIR_FIELDS[0]]):
                            raise RuntimeError("Vacuous zero dispersion fixture")
                        retained = result
                    report["runs"].append(row)
                    captured.update({f"r{index}_{k}":np.array(result[k],copy=True)
                                     for k in PAIR_FIELDS+SCALARS})
                    write(output/"probe-result.json",report)
                np.savez(output/"dispersion-pairs.npz", **captured)
                report["matrices_sha256"] = digest(output/"dispersion-pairs.npz")
        runtime.verify_optional_modules()
        report["provider_cleanup"] = runtime.report["plans"][-1]["session_cleanup"]
        report["status"] = "PASS_FIXED_GAUGE_CACHE_DIAGNOSTIC"
    except BaseException as error:
        report.update(status="FAILED",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write(output/"probe-result.json",report)


def verify(output, manifest):
    """Replay every paired output, never trust a timing-only success label."""
    from scf_repeat_probe import require_reference
    reference = require_reference(manifest)
    doc = json.loads((output/"probe-result.json").read_text())
    if (doc["status"] != "PASS_FIXED_GAUGE_CACHE_DIAGNOSTIC"
            or doc["scope"] != SCOPE or doc["accepted_speedup_count"] != 0
            or doc["science_pin"] != manifest["science_pin"]
            or doc["reference_result_sha256"] != manifest["reference_result_sha256"]
            or doc["cache_caps"] != list(CAPS) or doc["order"] != list(ORDER)
            or len(doc["runs"]) != len(ORDER)
            or doc["scf_executed"] or doc["fisapt_executed"]
            or doc["matrices_sha256"] != digest(output/"dispersion-pairs.npz")
            or "H200" not in doc["device"]["name"]
            or not doc["provider_cleanup"]["complete"]
            or doc["provider_cleanup"]["live_bytes"] != 0
            or doc["provider_cleanup"]["native_owners"] != 0):
        raise RuntimeError("Incomplete/mismatched fixed-gauge cache diagnostic")
    with np.load(output/"dispersion-pairs.npz",allow_pickle=False) as data:
        if set(data) != {f"r{i}_{k}" for i in range(len(ORDER)) for k in PAIR_FIELDS+SCALARS}:
            raise ValueError("Every paired result must be retained")
        first = {k:data[f"r0_{k}"] for k in PAIR_FIELDS+SCALARS}
        for i, arm in enumerate(ORDER):
            row = doc["runs"][i]
            if (row["arm"] != arm or row["cache_cap_bytes"] != CAPS[arm]
                    or not 0 < row["wall_seconds"] < float("inf")):
                raise RuntimeError("Changed paired execution order/timing")
            current = {k:data[f"r{i}_{k}"] for k in PAIR_FIELDS+SCALARS}
            errors = compare_components(first,current,tuple(reference["orbital_dimensions"]["active"]))
            if i and errors != row["maximum_abs_errors_from_first"]:
                raise RuntimeError("Paired output comparison seal changed")
    return doc
