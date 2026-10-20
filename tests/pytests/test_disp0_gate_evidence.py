"""Parser fixtures only: these tests are not GPU/device-execution evidence."""

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("einsums")

path = Path(__file__).resolve().parents[2] / "devtools/disp0/gpu_gate.py"
spec = importlib.util.spec_from_file_location("disp0_gpu_gate_validation", path)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def record(**updates):
    return {"api": "cublasDgemm_v2", "pid": 123, "status": 0, "m": 72, "n": 48, "k": 257,
            "phase": "physical/parser-fixture/production", **updates}


def products_replay(annotations):
    return {"name": "ComputeGraph::execute(sapt0-dispersion-products)", "children": [
        {"name": "Einsum", "annotations": annotations, "children": []}]}


def test_cublas_trace_parser_fixture(tmp_path):
    p = tmp_path / "parser-fixture.jsonl"
    p.write_text(json.dumps(record()) + "\n")
    result = gate.trace_evidence(p, 123)
    assert result["actual_physical_cublas_calls"] == result["actual_production_cublas_calls"] == 1


@pytest.mark.parametrize("changes", [
    {"status": 1}, {"pid": 321}, {"m": 0}, {"api": "gpu-labelled-wrapper"},
    {"phase": "synthetic/production"}, {"phase": "physical/parser-fixture/edge-1"},
])
def test_trace_parser_rejects_insufficient_or_failed_evidence(tmp_path, changes):
    p = tmp_path / "invalid-parser-fixture.jsonl"
    p.write_text(json.dumps(record(**changes)) + "\n")
    with pytest.raises(AssertionError):
        gate.trace_evidence(p, 123)


@pytest.mark.parametrize("phase,dispatch,accepted", [
    ("physical/parser-fixture/production", "gemm", True),
    ("physical/parser-fixture/production", "cpu_fallback", False),
    ("not-physical", "gemm", False),
])
def test_profile_parser_requires_observed_physical_dispatch(tmp_path, phase, dispatch, accepted):
    p = tmp_path / "profile-parser-fixture.json"
    p.write_text(json.dumps({"thread": {"children": [
        {"name": phase, "children": [products_replay({"gpu_dispatch": dispatch})]}]}}))
    if accepted:
        assert len(gate.profile_evidence(p)["gpu_dispatches"]) == 1
    else:
        with pytest.raises(AssertionError):
            gate.profile_evidence(p)


def test_profile_parser_rejects_placement_only(tmp_path):
    p = tmp_path / "placement-only-fixture.json"
    p.write_text(json.dumps({"nodes": [{"target": "GPU", "kind": "Einsum"}]}))
    with pytest.raises(AssertionError):
        gate.profile_evidence(p)


@pytest.mark.parametrize("edge_annotations,accepted", [
    ({"op_kind": "Einsum", "device": "CPU"}, True),
    ({"op_kind": "Einsum", "device": "GPU", "gpu_dispatch": "cpu_fallback"}, True),
    ({"op_kind": "Einsum", "device": "GPU", "gpu_dispatch": "gemm"}, False),
    ({"target": "CPU", "kind": "Einsum"}, False),
    ({}, False),
])
def test_profile_parser_requires_executed_cpu_edge(tmp_path, edge_annotations, accepted):
    p = tmp_path / "edge-parser-fixture.json"
    p.write_text(json.dumps({"thread": {"children": [
        {"name": "physical/parser-fixture/production", "children": [products_replay({"gpu_dispatch": "gemm"})]},
        {"name": "physical/parser-fixture/edge-1", "children": [products_replay(edge_annotations)]},
    ]}}))
    if accepted:
        assert gate.profile_evidence(p, require_cpu_edge=True)["observed_cpu_edges"]
    else:
        with pytest.raises(AssertionError):
            gate.profile_evidence(p, require_cpu_edge=True)


def test_cpu_setup_execution_does_not_prove_cpu_block_fallback(tmp_path):
    p = tmp_path / "unrelated-setup-parser-fixture.json"
    unrelated = products_replay({"op_kind": "Einsum", "device": "CPU"})
    unrelated["name"] = "ComputeGraph::execute(fdisp0_setup)"
    p.write_text(json.dumps({"thread": {"children": [
        {"name": "physical/parser-fixture/production", "children": [products_replay({"gpu_dispatch": "gemm"})]},
        {"name": "physical/parser-fixture/edge-1", "children": [unrelated]},
    ]}}))
    with pytest.raises(AssertionError):
        gate.profile_evidence(p, require_cpu_edge=True)


def test_gate_driver_qcvariable_names_and_sum_rules_on_cpu():
    """Validate published-variable wiring without any CUDA request."""
    original = gate.mp2.df_mp2_einsums_dispersion
    retained = {}

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        retained.update(result)
        return result

    gate.psi4.set_options({
        "basis": "cc-pvdz", "scf_type": "df", "SAPT_DFT_FUNCTIONAL": "HF",
        "SAPT_DFT_MP2_DISP_ALG": "EINSUMS", "SAPT_DFT_EINSUMS_GPU": False,
        "SAPT_DFT_DO_DHF": False, "SAPT_DFT_INDUCTION_TYPE": "NONE",
        "SAPT_DFT_USE_EINSUMS": True, "ORBITAL_OPTIMIZER_PACKAGE": "INTERNAL",
    })
    with patch.object(gate.mp2, "df_mp2_einsums_dispersion", capture):
        energy, wfn = gate.psi4.energy("sapt(dft)", molecule=gate.psi4.geometry(gate.WATER_NE), return_wfn=True)
    for label in gate.LABELS:
        np.testing.assert_allclose([gate.core.variable(label), wfn.variable(label)], retained[label],
                                   rtol=gate.KERNEL_RTOL, atol=gate.KERNEL_ATOL)
    np.testing.assert_allclose([gate.core.variable("SAPT DISP ENERGY"), wfn.variable("SAPT DISP ENERGY")],
                               sum(retained.values()), rtol=gate.KERNEL_RTOL, atol=gate.KERNEL_ATOL)
    total = sum(float(gate.core.variable(label)) for label in
                ("SAPT ELST ENERGY", "SAPT EXCH ENERGY", "SAPT IND ENERGY", "SAPT DISP ENERGY"))
    np.testing.assert_allclose([energy, wfn.variable("SAPT TOTAL ENERGY")], total,
                               rtol=gate.KERNEL_RTOL, atol=gate.KERNEL_ATOL)
