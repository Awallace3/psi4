"""Unqualified, versioned ISA accounting seam; NOT a production policy.

Reuse the byte-pinned historical grid400 ledger unchanged, replace ONLY its
ISA stage accounting, then recompute whole-chain work/peak/admission/refusals.
No descriptor, installed build, receipt, dataset or dispatch is authorized.
"""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
POLICY = "qomp-isa-grid100-200-accounting-v1-unqualified"
PARENT_SCIENTIFIC_COMMIT = "2fe7a711dd915e102fe6976f44d22571064aa31e"
LEGACY_FILES = {
    "resource_plan_grid400.py": "cb6629c50b2c9e7ad154969675b29bb7d8c973346a5df55d429cd979b71daf57",
    "resource_plan_ed561666.py": "42dec3a372e326ef45119b158084e0cb3a6bd21131dc8db14075dde2636b75ba",
    "resource_plan_cb63305.py": "089aa033255af054906dcaffd8f83c6e16b66f5f7819d38742432c41640ef3ac",
}
HELPER = "psi4/driver/procrouting/isapol_isa_grid_plan.py"
RUNTIME = "psi4/driver/procrouting/isapol_distribution.py"


def _verified(path, digest):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"accounting sources differ: {path.name}")
    return raw


def _load(path, name, raw):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


for _name, _digest in LEGACY_FILES.items():
    _verified(Path(__file__).with_name(_name), _digest)
_path = Path(__file__).with_name("resource_plan_grid400.py")
_legacy = _load(_path, "_isa_grid_v1_legacy", _path.read_bytes())
Dimensions = _legacy.Dimensions
FILES = dict(_legacy.FILES)
FILES.update({
    RUNTIME: "ded958694b96a76db14e5c50f32d7703999946cfd2117afb81c828cc81efeaea",
    HELPER: "7e9a0dc54570aff10458b006985cac79eb093e06c339af758b1d43078e2a7971",
    "psi4/src/psi4/libisapol/partitioned_response.cc":
        "72a8a8bab52bc74ba416d0fd237398bd1bcb4ab601b3a145cd0f7f196ed5f4ec",
})


def source_binding(root):
    observed = {name: hashlib.sha256((Path(root)/name).read_bytes()).hexdigest()
                for name in FILES}
    if observed != FILES:
        raise ValueError("accounting scientific sources differ from ISA grid v1")
    for name, digest in LEGACY_FILES.items():
        _verified(Path(root)/"devtools/isapol_s66_campaign"/name, digest)
    return dict(policy_id=POLICY, parent_scientific_commit=PARENT_SCIENTIFIC_COMMIT,
                files_sha256=observed, qualified=False)


def _accounting():
    helper = _load(ROOT/HELPER, "_isa_grid_v1_pure",
                   _verified(ROOT/HELPER, FILES[HELPER]))
    # Execute ONLY the existing scalar workspace function from verified bytes,
    # never import distribution.py (which would import NumPy and native Psi4).
    tree = ast.parse(_verified(ROOT/RUNTIME, FILES[RUNTIME]))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == "_q_accumulation_workspace_bytes"]
    if len(functions) != 1:
        raise ValueError("one native Q workspace accounting function required")
    namespace = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(ROOT/RUNTIME), "exec"), namespace)
    return helper, namespace["_q_accumulation_workspace_bytes"]


def plan(dimensions, resources, *, max_replay_work, pfit_maximum_work_bytes):
    """Unsigned arithmetic prediction, retaining every independent old gate."""
    result = _legacy.plan(dimensions, resources, max_replay_work=max_replay_work,
                          pfit_maximum_work_bytes=pfit_maximum_work_bytes)
    helper, workspace = _accounting()
    d = result["dimensions"]
    s, p, h = d["sites"], d["response_aux_width"], d["integration_grid_rows"]
    g = helper.isa_grid_rows(
        s, radial_points=100, spherical_points=200, becke_smoothing=3,
        radius_scaling=1., radius_policy="native_tabulated_bragg_slater",
        neighbour_policy="all_sites_unscreened_full_molecular_grid")
    removed_rows = d["isa_grid_accounting_rows"]-g
    a, a2 = sum(d["atomic_widths"]), sum(w*w for w in d["atomic_widths"])
    shape = sum(d["shape_widths"])
    scratch = workspace(p, 4, h)
    preparation = helper.q_preparation_work(s, p, 4, h)
    byte_delta = -8*removed_rows*(a+40*s+40)+scratch
    work_delta = -d["isa_iterations"]*8*removed_rows*(a2+s*shape)+preparation
    matches = [t for t in result["stages"] if t["name"] == "ISA iteration and response-AUX Q"]
    if len(matches) != 1 or matches[0]["calls"] != 1:
        raise ValueError("one complete historical ISA stage required")
    stage = matches[0]
    stage["transient_bytes"] += byte_delta
    stage["numeric_bytes"] += byte_delta
    stage["work_per_call"] += work_delta
    stage["charged_work"] += work_delta
    d["isa_grid_accounting_rows"] = g
    result["charged_work"] = sum(t["charged_work"] for t in result["stages"])
    result["peak_numeric_bytes"] = max(t["numeric_bytes"] for t in result["stages"])
    result["admission_numeric_bytes"] = result["peak_numeric_bytes"]
    result["peak_stage"] = max(result["stages"], key=lambda t: t["numeric_bytes"])["name"]
    result["work_by_stage"] = {t["name"]: t["charged_work"] for t in result["stages"]}
    refusals = set(result["refusals"])-{"cumulative_work", "numeric_bytes"}
    if result["charged_work"] > resources["max_work"]:
        refusals.add("cumulative_work")
    if result["peak_numeric_bytes"] > resources["max_bytes"]:
        refusals.add("numeric_bytes")
    order = ("cumulative_work", "checkpoint_io", "numeric_bytes", "packed_replay_bytes",
             "quadrature_bytes", "quadrature_work", "initial_grid_bytes")
    if refusals-set(order):
        raise ValueError("unknown historical refusal")
    result["refusals"] = [key for key in order if key in refusals]
    result.update(
        policy_id=POLICY, policy_scientific_commit=None,
        status="REFUSED" if refusals else "WITHIN_PREDICTED_LEDGER_LIMITS",
        qualification="Unqualified source arithmetic only; a new build, installed qualification, "
                      "closed production profile and genuine descriptors are still required.",
        isa_accounting=dict(version=1, grid_rows_removed_from_envelope=removed_rows,
                            q_workspace_bytes=scratch, q_preparation_work=preparation,
                            numeric_delta=byte_delta, work_delta=work_delta))
    return result
