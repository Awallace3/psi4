# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""One isolated paid execution; parent-owned price approval and activation only."""
import argparse
from decimal import Decimal
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import traceback

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_ccasp_inferno_bootstrap", HERE/"integrated_runtime.py")
bootstrap = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bootstrap
spec.loader.exec_module(bootstrap)
R = bootstrap.for_plan(HERE/"pdef_inferno_plan.json")


def argv(commit, tree, runtime_sha, derivative_sha, *, test_only=True):
    R.effective_profile()
    for value, length in ((commit, 40), (tree, 40), (runtime_sha, 64), (derivative_sha, 64)):
        if not isinstance(value, str) or re.fullmatch("[0-9a-f]{"+str(length)+"}", value) is None:
            raise ValueError("concrete reviewed commit/tree/runtime/derivative pins required")
    return ["/opt/slurm/current/bin/sbatch", "--test-only" if test_only else "--parsable",
        "--export=NONE", "--chdir="+R.PLAN["run"],
        str(Path(R.PLAN["source"])/"devtools/ccasp_phoenix_benzene/pdef_inferno_full.sbatch"),
        commit, tree, runtime_sha, derivative_sha]


def price_approval(binding, runtime_sha, derivative_sha):
    """External parent record, never synthesized by prepare or a price query here.

    This is a fail-closed payload admission, NOT a spending lock: the scheduler
    charges at allocation start. Parent must verify freshness/price/route and
    consume the one grant externally BEFORE submitting, including on ambiguity.
    """
    path = Path(R.PLAN["run"])/"metadata/paid-price-approval.json"
    pin = R.small_regular(path, 64*1024)["sha256"]
    actual = R.pinned_json(path, pin)
    if not isinstance(actual, dict):
        raise ValueError("external parent price/route approval record required")
    quote, checked, evidence = (actual.get(k) for k in
        ("quoted_maximum_usd", "checked_at_utc", "price_route_evidence"))
    if (not isinstance(quote, str) or re.fullmatch(r"0(?:\.[0-9]{1,6})?", quote) is None
            or not Decimal("0") <= Decimal(quote) <= Decimal(R.PLAN["paid_grant"]["maximum_usd"])
            or not isinstance(checked, str)
            or re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", checked) is None
            or not isinstance(evidence, str) or not evidence.strip()):
        raise ValueError("external bounded price/fresh route evidence required")
    expected = dict(schema="ccasp-benzene-pdef-inferno-price-approval/1",
        grant_identity=R.PLAN["paid_grant"]["identity"],
        approved_by="parent", approved_for_single_submission=True,
        automatic_further_retry=False, commit=binding["commit"], tree=binding["tree"],
        execution_identity_sha256=binding["effective_profile"]["identity_sha256"],
        runtime_manifest_sha256=runtime_sha, derivative_sha256=derivative_sha,
        allocation=R.PLAN["allocation"], quoted_maximum_usd=quote,
        checked_at_utc=checked, price_route_evidence=evidence)
    profile = R.module("_ccasp_inferno_approval_hash", "pdef_profile.py")
    if profile.canonical_sha(actual) != profile.canonical_sha(expected):
        raise ValueError("external parent single paid grant/route/pins differ")
    return dict(path=str(path), sha256=pin, approval=actual)


def prepare(commit, tree):
    R.effective_profile()
    binding = R.source_binding(commit, tree)
    p = R.PLAN
    cfg = binding["effective_profile"]["effective_config"]
    if R.sha(p["fixture"]) != cfg["fixture_sha256"]:
        raise ValueError("original private geometry hash differs")
    R.prior_probe()
    root, runtime, run, scratch = (Path(p[name]) for name in ("root", "runtime", "run", "scratch"))
    if any(path.exists() or path.is_symlink() for path in (root, runtime, run, scratch)):
        raise ValueError("root/runtime/run/scratch destinations must all be new")
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    runtime.mkdir(mode=0o700, exist_ok=False)
    run.mkdir(mode=0o700, exist_ok=False)
    scratch.mkdir(mode=0o700, exist_ok=False)
    for name in ("logs", "metadata"):
        (run/name).mkdir(mode=0o700)
    manifest = R.compose(R.ROOT, R.OLD.STAGE, runtime/"stage", binding)
    runtime_sha = R.write(runtime/"runtime-manifest.json", manifest)
    R.verify_runtime(runtime/"runtime-manifest.json", runtime_sha, binding)
    derivative_sha = R.write(runtime/"basis-derivative-admission.json", R.derived_document(binding, runtime_sha))
    receipt = dict(binding=binding, runtime_manifest_sha256=runtime_sha,
        derivative_sha256=derivative_sha, script_sha256=R.sha(HERE/"pdef_inferno_full.sbatch"),
        test_only_argv=argv(commit, tree, runtime_sha, derivative_sha),
        future_submit_argv_not_executed=argv(commit, tree, runtime_sha, derivative_sha, test_only=False),
        activation_authority=p["activation_authority"], paid_grant=p["paid_grant"],
        required_external_price_approval=str(run/"metadata/paid-price-approval.json"),
        parent_must_approve_price_route_and_consume_one_grant_before_submission=True,
        no_native_import_build_or_submission=True)
    R.write(run/"metadata/preparation.json", receipt)
    print(json.dumps(receipt, indent=2))


def run(commit, tree, runtime_sha, derivative_sha):
    R.compute_guard()
    R.effective_profile()
    p = R.PLAN
    root = Path(p["run"])
    out = root/"results"
    out.mkdir(mode=0o700, exist_ok=False)
    record = dict(status="failed", science_pass=False, commit=commit, tree=tree,
        native_commit=R.OLD.BUILD_COMMIT, native_core_sha256=R.OLD.CORE_SHA,
        runtime_manifest_sha256=runtime_sha, derivative_admission_sha256=derivative_sha,
        original_probe_sha256=p["prior_probe_sha256"], job=os.environ["SLURM_JOB_ID"],
        paid_grant=p["paid_grant"])
    handles = []
    try:
        binding = R.source_binding(commit, tree)
        record["effective_profile"] = binding["effective_profile"]
        preparation = json.loads((root/"metadata/preparation.json").read_text())
        if (preparation["binding"] != binding or preparation["runtime_manifest_sha256"] != runtime_sha
                or preparation["derivative_sha256"] != derivative_sha
                or preparation["script_sha256"] != R.sha(HERE/"pdef_inferno_full.sbatch")
                or preparation["paid_grant"] != p["paid_grant"]
                or preparation["activation_authority"] != p["activation_authority"]
                or preparation["required_external_price_approval"] != str(root/"metadata/paid-price-approval.json")
                or preparation["parent_must_approve_price_route_and_consume_one_grant_before_submission"] is not True):
            raise ValueError("reviewed preparation/script/effective inferno profile pins differ")
        record["external_price_approval"] = price_approval(binding, runtime_sha, derivative_sha)
        if R.pinned_json(Path(p["runtime"])/"basis-derivative-admission.json", derivative_sha) != R.derived_document(binding, runtime_sha):
            raise ValueError("derivative admission pin/binding differs before imports")
        psi4, _ = R.initialize(binding, Path(p["runtime"])/"runtime-manifest.json", runtime_sha, root, handles)
        driver = R.module("_ccasp_inferno_science", "pdef_inferno_driver.py")
        record.update(driver.capture_science(R, psi4, binding, runtime_sha, derivative_sha))
    except BaseException as error:
        record.update(status="failed", science_pass=False,
            error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    finally:
        if handles:
            try:
                R.OLD.snapshot("after-capture", receipt_run=root)
                R.verify_runtime(Path(p["runtime"])/"runtime-manifest.json", runtime_sha, binding, hash_core=True)
                if R.source_binding(commit, tree) != binding:
                    raise ValueError("final source/effective inferno profile differs")
                approval = record["external_price_approval"]
                if R.pinned_json(approval["path"], approval["sha256"]) != approval["approval"]:
                    raise ValueError("external price approval changed during execution")
                record["after_runtime_closure"] = "passed"
            except BaseException as error:
                record.update(status="failed", science_pass=False, after_runtime_closure="failed",
                    closure_error=f"{type(error).__name__}: {error}", closure_traceback=traceback.format_exc())
        else:
            record["after_runtime_closure"] = "not initialized; no scientific import attempted"
    captured = sorted(str(f) for f in (out/"diagnostic").glob("node-*/pre-lw-complete.json"))
    record.update(complete_pre_lw_capture_receipts=captured, capture_is_not_scientific_acceptance=True)
    R.write(out/"result.json", record)
    code = 0 if record["science_pass"] else 1
    R.OLD.check_driver_result(code, record)
    R.write(root/"metadata/payload-terminal.json", dict(exit_code=code, science_pass=record["science_pass"],
        complete=code == 0, capture_receipts=captured, after_runtime_closure=record["after_runtime_closure"]))
    return code  # retained Einsums handle lives through final closure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--commit", required=True)
    parser.add_argument("--tree", required=True)
    parser.add_argument("--runtime-sha256")
    parser.add_argument("--derivative-sha256")
    a = parser.parse_args()
    if a.action == "prepare":
        prepare(a.commit, a.tree)
        return 0
    if not a.runtime_sha256 or not a.derivative_sha256:
        parser.error("run requires reviewed runtime and derivative SHA256 pins")
    argv(a.commit, a.tree, a.runtime_sha256, a.derivative_sha256)
    return run(a.commit, a.tree, a.runtime_sha256, a.derivative_sha256)


if __name__ == "__main__":
    try:
        code = main()
    except BaseException as error:
        # Rejected operational plans cannot direct even an error receipt.
        try:
            R.effective_profile()
        except BaseException:
            pass
        else:
            terminal = Path(R.PLAN["run"])/"metadata/payload-terminal.json"
            if (os.environ.get("SLURM_JOB_ID") and os.environ.get("USER") == "awallace43"
                    and os.environ.get("HOME", "").startswith("/storage/home/")
                    and terminal.parent.is_dir() and not terminal.exists()):
                R.write(terminal, dict(exit_code=1, complete=False, science_pass=False,
                    error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()))
        raise
    sys.exit(code)
