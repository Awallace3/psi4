"""Full-fragment cache coexistence on fixed HF inputs; no SCF/speedup claim."""
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

from stage_scaling import digest, write
from fixed_gauge_cache_probe import CAPS, cache_plan

SCOPE = "FIXED_HF_FULL_FRAGMENT_CACHE_INTEGRATION_NOT_ENERGY_BENCHMARK"
STATUS = "PASS_FIXED_HF_CACHE_INTEGRATION_ONLY"


def require_minimal_basis(selected, expected):
    if (not isinstance(selected,str) or not isinstance(expected,str) or not expected
            or selected.casefold() != expected.casefold()):
        raise ValueError(f"Original minimal basis required: selected={selected!r}, expected={expected!r}")


def expected_minimal_basis(source, options):
    explicit = [v for k,v in options.items() if k.casefold()=="minao_basis"]
    if len(explicit) > 1:
        raise ValueError("Ambiguous original minimal-basis option")
    if explicit:
        if not isinstance(explicit[0],str) or not explicit[0]:
            raise ValueError("Explicit original minimal basis must be nonempty")
        return explicit[0]
    # The source is already frozen by the controller and bound to the native build.
    text = (source/"psi4/src/read_options.cc").read_text()
    matches = re.findall(r'options\.add_str\(\s*"MINAO_BASIS"\s*,\s*"([^"]+)"\s*\)',text)
    if len(matches) != 1:
        raise RuntimeError("One original registered minimal-basis default required")
    return matches[0]


def validate_trace(trace):
    row = trace["workflow"]
    windows = row["windows"]
    if (row["status"] != "COMPLETED"
            or set(windows) != {"localization","elst","exch","ind","disp"}
            or any(not 0 < v["wall_seconds"] < float("inf") for v in windows.values())
            or row["workflow_other_seconds"] < 0
            or abs(sum(v["wall_seconds"] for v in windows.values())+
                   row["workflow_other_seconds"]-row["wall_seconds"]) > 1e-7):
        raise RuntimeError("Incomplete/overlapping full-fragment term windows")


def validate_work(execution, dimensions, q, nr, ns):
    a,b = dimensions["active"]
    expected = cache_plan(q,a,b,nr,ns,24,CAPS[1])
    dispersion = execution["dispersion"]
    if (execution["same_session"] is not True
            or execution["backend"] != "CUDA_CUEST_SHARED_IBO_EXPERIMENTAL"
            or execution["jkfit_producer"]["calls"] != 5
            or execution["rifit_producer"]["calls"] != expected["native_batches"]
            or dispersion["compute_blocks"] != expected["compute_blocks"]
            or dispersion["factor_cache"]["budget_bytes"] != CAPS[1]
            or any(dispersion["factor_cache"][k] != expected[k] for k in (
                "allocated_bytes","cached_slabs","hits","r_factor_batches","s_factor_batches"))):
        raise RuntimeError("Full workflow changed native work, cache policy or session ownership")
    return expected


def run(runtime, manifest, output, *, doc, raw, fields, primary, rifit, molecule,
        basis_counts):
    from psi4.driver.procrouting.sapt.sapt_fisapt_workflow_cuda import fragment_terms, HF_FIELDS
    from fragment_gate import FIELDS
    from repeat_gauge import compare_arrays, close
    core = runtime.psi4.core
    dims = doc["orbital_dimensions"]
    options = doc["options"]
    jkfit = core.BasisSet.build(molecule,"DF_BASIS_SCF",options["df_basis_scf"],
                               "JKFIT",options["basis"],puream=1)
    if jkfit.nbf() != basis_counts["jkfit"]:
        raise RuntimeError("Frozen JKFIT dimension changed")
    minao_name = core.get_global_option("MINAO_BASIS")
    minao_expected = expected_minimal_basis(Path(manifest["source"]),options)
    require_minimal_basis(minao_name,minao_expected)
    minimal = core.BasisSet.build(molecule,"BASIS",minao_name,puream=1)
    fragments = [list(range(first,last)) for first,last in molecule.get_fragments()]
    if (len(fragments) != 2 or sorted(i for f in fragments for i in f) !=
            list(range(dims["atoms"]))):
        raise ValueError("Exactly the original two unlinked fragments required")
    charges = [np.zeros(dims["atoms"],dtype=np.float64) for _ in fragments]
    for value, fragment in zip(charges,fragments):
        for i in fragment:
            value[i] = molecule.Z(i)
    for side, value in zip("AB",charges):
        np.testing.assert_array_equal(fields["Z"+side],value)
    xyz = np.array([[molecule.x(i),molecule.y(i),molecule.z(i)]
                    for i in range(dims["atoms"])],dtype=np.float64)
    matrices = {k:np.ascontiguousarray(raw[k]) for k in (*HF_FIELDS,"J_O")}
    energies = {k:np.ascontiguousarray(raw[name]) for k,name in zip(
        "abrs",("eps_occ_A","eps_occ_B","eps_vir_A","eps_vir_B"))}
    scaling = {k:np.asarray(value["raw_final_scalar_inputs"],dtype=np.float64)
               for k,value in doc["fragment_execution"]["scaling"].items()}
    if set(scaling) != {"exchange","induction"}:
        raise ValueError("Original scaled fragment controls required")
    report = {"status":"STARTED", "scope":SCOPE, "accepted_speedup_count":0,
              "science_pin":manifest["science_pin"], "pid":os.getpid(),
              "reference_result_sha256":manifest["reference_result_sha256"],
              "snapshot_artifact_sha256":doc["artifact_sha256"],
              "scf_executed":False, "fisapt_executed":True, "full_energy_executed":False,
              "cache_cap_bytes":CAPS[1], "fixed_dimensions":dims,
              "input_scope":"Fixed raw HF matrices/energies/scaling, not saved localized orbitals",
              "rifit_nbf":rifit.nbf(), "jkfit_nbf":jkfit.nbf(),
              "minimal_basis":{"name":minao_name,"nbf":minimal.nbf(),
                               "original_selection":minao_expected}}
    try:
        # Reuse the already tested boundary observer, never replace science functions.
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"cachy"))
        from fragment_term_trace import TermTrace
        import importlib
        workflow = Path(importlib.import_module(
            "psi4.driver.procrouting.sapt.sapt_fisapt_workflow_cuda").__file__).resolve()
        relative = "driver/procrouting/sapt/sapt_fisapt_workflow_cuda.py"
        if digest(workflow) != runtime.build["scientific_python_sha256"][relative]:
            raise RuntimeError("Installed full workflow changed")
        tracer = TermTrace(workflow,workflow.read_text(),include_localization=True)
        started = time.perf_counter()
        with tracer:
            with runtime.provider(primary,jkfit) as provider, provider:
                provider.validate_basis(primary,jkfit)
                report["device"] = runtime.device_identity(provider.session)
                arrays, data = fragment_terms(provider,rifit,minimal,matrices,energies,
                    xyz,*charges,frozen=tuple(dims["frozen"]),block_size=24,scaling=scaling,
                    factor_cache_bytes=CAPS[1])
        report["fragment_call_seconds_including_factory_cleanup"] = time.perf_counter()-started
        report["term_trace"] = tracer.report()
        validate_trace(report["term_trace"])
        report["timing_scope"] = "Synchronized instrumented fixed-HF fragment, not baseline energy"
        execution = data["FISAPT execution"]
        report["work"] = validate_work(execution,dims,rifit.nbf(),
            raw["Cvir_A"].shape[1],raw["Cvir_B"].shape[1])
        actual = {k:np.asarray(arrays[k]) for k in FIELDS if k not in ("ZA","ZB")}
        actual.update(ZA=charges[0],ZB=charges[1])
        # Original strict full-field criterion; only labels/signs may change.
        report["original_array_check"] = compare_arrays(fields,actual,raw,raw,dims)
        report["original_scalar_checks"] = {k:close(np.asarray(doc[k]),np.asarray(data[k]),k)
                                             for k in ("Disp20,u","Exch-Disp20,u")}
        report["signed_components"] = {k:float(data[k]) for k in ("Disp20,u","Exch-Disp20,u")}
        np.savez(output/"fragment-matrices.npz",**actual)
        write(output/"full-fragment-execution.json",execution)
        report["matrices_sha256"] = digest(output/"fragment-matrices.npz")
        report["execution_sha256"] = digest(output/"full-fragment-execution.json")
        report["provider_cleanup"] = runtime.report["plans"][-1]["session_cleanup"]
        if (not report["provider_cleanup"]["complete"]
                or report["provider_cleanup"]["live_bytes"]
                or report["provider_cleanup"]["native_owners"]
                or report["provider_cleanup"]["peak_explicit_bytes"]+
                   runtime.config["native_persistent_bytes"] >
                   runtime.config["device_memory_limit_bytes"]):
            raise RuntimeError("Full workflow cleanup or declared device budget failed")
        runtime.verify_optional_modules()
        report["status"] = STATUS
    except BaseException as error:
        report.update(status="FAILED",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write(output/"probe-result.json",report)


def verify(output, manifest):
    from scf_repeat_probe import require_reference
    from repeat_gauge import compare_arrays, close
    from fragment_gate import FIELDS
    from full_inputs import load_point
    reference = require_reference(manifest)
    minao_expected = expected_minimal_basis(Path(manifest["source"]),reference["options"])
    point, dimensions = load_point(Path(manifest["inputs"]),manifest["inputs_sha256"],"protein83")
    doc = json.loads((output/"probe-result.json").read_text())
    require_minimal_basis(doc["minimal_basis"]["name"],minao_expected)
    if doc["minimal_basis"]["original_selection"] != minao_expected:
        raise RuntimeError("Original minimal-basis selection seal changed")
    if (doc["status"] != STATUS or doc["scope"] != SCOPE or doc["accepted_speedup_count"] != 0
            or doc["science_pin"] != manifest["science_pin"]
            or doc["reference_result_sha256"] != manifest["reference_result_sha256"]
            or doc["cache_cap_bytes"] != CAPS[1] or doc["scf_executed"]
            or doc["fixed_dimensions"] != dimensions
            or doc["rifit_nbf"] != point["spherical_basis_counts"]["rifit"]
            or doc["jkfit_nbf"] != point["spherical_basis_counts"]["jkfit"]
            or doc["full_energy_executed"] or not doc["fisapt_executed"]
            or "H200" not in doc["device"]["name"]
            or not doc["provider_cleanup"]["complete"] or doc["provider_cleanup"]["live_bytes"]
            or doc["provider_cleanup"]["native_owners"]
            or doc["matrices_sha256"] != digest(output/"fragment-matrices.npz")
            or doc["execution_sha256"] != digest(output/"full-fragment-execution.json")):
        raise RuntimeError("Incomplete/changed fixed-HF cache integration")
    source = Path(manifest["reference_result"]).parent
    values = []
    for path in (source/"raw-hf.npz",source/"fragment-matrices.npz",
                 output/"fragment-matrices.npz"):
        if path.parent == source and digest(path) != reference["artifact_sha256"][path.name]:
            raise RuntimeError("Original HF/gauge snapshot changed")
        with np.load(path,allow_pickle=False) as data:
            values.append(dict(data))
    raw, left, right = values
    if set(left) != set(FIELDS) or set(right) != set(FIELDS):
        raise ValueError("Every original fragment field must be retained")
    if compare_arrays(left,right,raw,raw,reference["orbital_dimensions"]) != doc["original_array_check"]:
        raise RuntimeError("Full-field comparison receipt changed")
    validate_trace(doc["term_trace"])
    for key,field in zip(("Disp20,u","Exch-Disp20,u"),
                         ("Disp20 occupied pairs","Exch-Disp20 occupied pairs")):
        np.testing.assert_allclose(right[field].sum(),doc["signed_components"][key],
                                   rtol=1e-11,atol=1e-12)
        error = close(np.asarray(reference[key]),np.asarray(doc["signed_components"][key]),key)
        if error != doc["original_scalar_checks"][key]:
            raise RuntimeError("Original scalar comparison receipt changed")
    execution = json.loads((output/"full-fragment-execution.json").read_text())
    expected = validate_work(execution,reference["orbital_dimensions"],
        point["spherical_basis_counts"]["rifit"],
        raw["Cvir_A"].shape[1],raw["Cvir_B"].shape[1])
    if expected != doc["work"]:
        raise RuntimeError("Full-workflow work-count seal changed")
    return doc
