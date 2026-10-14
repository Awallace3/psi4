#!/usr/bin/env python3
"""Validate induction-NONE evidence, never relabel a failed full-SAPT job.

Read-only with respect to the original validation and scientific build. Writes
only a new scoped receipt. CPKS/FDDS support is explicitly not certified.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from xml.etree import ElementTree

PIN = "c28950041bbe8bb10617264a7b6ddc56a436cda1"
UNSUPPORTED = "The RKS XC response kernel (compute_Vx; used by TDDFT and CPHF response properties) has no cuEST implementation."
COMPONENTS = {"SAPT ELST ENERGY", "SAPT EXCH ENERGY", "SAPT IND ENERGY", "SAPT TOTAL ENERGY"}
CONFIGS = {
    ((99, 590), None), ((99, 590), (65, 302)), ((99, 590), (50, 302)),
    ((75, 302), None), ((75, 302), (35, 194)),
    ((50, 194), None), ((50, 194), (25, 110)),
    ((25, 110), None), ((25, 110), (15, 50)),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(validation, science, pin):
    validation, science = Path(validation).resolve(), Path(science).resolve()
    require(pin == PIN, "This scoped gate is pinned to the inspected scientific checkpoint")
    hashes = {}

    def read(relative):
        data = (validation / relative).read_bytes()
        hashes[relative] = hashlib.sha256(data).hexdigest()
        return data

    runtime = dict(line.split("=", 1) for line in
                   read("metadata/runtime.env").decode().splitlines() if "=" in line)
    require(runtime.get("scientific_commit") == pin, "Wrong validation science pin")
    require(runtime.get("exit_code") == "1" and runtime.get("term_received") == "0",
            "Expected the inspected, unsignalled full-SAPT failure")
    reports = {}
    for backend in ("host", "cuda"):
        require(read(f"{backend}/commit.txt").decode().strip() == pin, "Wrong backend pin")
        imported = json.loads(read(f"{backend}/import.json"))
        package = science / f"build_grid_{backend}/stage/lib/psi4"
        core = Path(imported["core"]).resolve()
        require(Path(imported["package"]).resolve() == package, "Wrong package")
        require(core.parent == package and imported["variant"] == backend, "Wrong core/backend")
        require(hashlib.sha256(core.read_bytes()).hexdigest() == imported["core_sha256"],
                "Core changed after validation")
        suite = ElementTree.fromstring(read(f"{backend}/regressions.xml")).find("testsuite")
        require(suite is not None and int(suite.get("tests", 0)) == 36, "Expected 36 GPU tests")
        require(all(int(suite.get(k, 0)) == 0 for k in ("errors", "failures", "skipped")),
                "GPU suite failed or skipped")
        require(len(suite.findall("testcase")) == 36, "Incomplete GPU test inventory")
        require(not any(case.find(k) is not None for case in suite.findall("testcase")
                        for k in ("failure", "error", "skipped")), "Nonpassing GPU testcase")
        profile = json.loads(read(f"{backend}/profile/timers.json"))
        require(any("cuestXCPotentialCompute" in k for k in profile["microtimers"]),
                "Missing cuEST integration microtimers")
        require(any("xc_gga_exc_vxc" in k for k in profile["microtimers"]),
                "Missing LibXC microtimers")
        cuda = any("cuEST XC: CUDA LibXC Functional" in k for k in profile["timers"])
        require(cuda == (backend == "cuda"), "Wrong profiling LibXC backend")
        errors = {}
        for stage, iterative in (("water", False), ("water-iterative", True)):
            rows = json.loads(read(f"{backend}/{stage}/summary.json"))
            cases = {(tuple(r["final"]), tuple(r["initial"]) if r["initial"] else None): r
                     for r in rows}
            require(len(rows) == len(cases) == 9 and set(cases) == CONFIGS,
                    "Incomplete supported grid sweep")
            worst_switch = worst_75 = 0.0
            for (final, initial), row in cases.items():
                opts = row["options"]
                require(row["status"] == "ok" and not row["full_sapt"],
                        "Supported-workload calculation failed")
                require(opts["use_cuest"] and opts["scf_type"].lower() == "df" and
                        opts["sapt_dft_induction_type"].lower() == "none" and
                        opts["sapt_dft_grac_compute"].lower() == ("iterative" if iterative else "none"),
                        "Wrong supported workload")
                values = row["components"]
                require(set(values) == COMPONENTS and all(math.isfinite(v) for v in values.values()),
                        "Invalid component values")
                if initial:
                    require(row["final_grid_iterations"] >= 1, "No final-grid evaluation")
                    control = cases[(final, None)]["components"]
                    error = max(abs(values[k] - control[k]) for k in COMPONENTS)
                    require(error <= 2e-9, "Same-final switching exceeds 2e-9 Eh")
                    worst_switch = max(worst_switch, error)
                if final == (75, 302):
                    control = cases[((99, 590), None)]["components"]
                    error = max(abs(values[k] - control[k]) for k in COMPONENTS)
                    require(error <= 1e-5, "75x302 exceeds user 1e-5 Eh target")
                    worst_75 = max(worst_75, error)
            errors[stage] = dict(worst_switch_error_eh=worst_switch,
                                 worst_75x302_component_error_eh=worst_75)
        full = json.loads(read(f"{backend}/water-full/summary.json"))
        expected = {((99, 590), None), ((99, 590), (50, 302)),
                    ((75, 302), None), ((75, 302), (35, 194))}
        require(len(full) == 4 and
                {(tuple(r["final"]), tuple(r["initial"]) if r["initial"] else None)
                 for r in full} == expected, "Incomplete unsupported controls")
        require(all(r["status"] == "failed" and r["full_sapt"] and
                    r["options"]["sapt_dft_induction_type"].lower() == "cpks" and
                    UNSUPPORTED in r.get("error", "") for r in full),
                "Unexpected full-SAPT failure; do not bypass it")
        reports[backend] = dict(regressions_passed=36, core_sha256=imported["core_sha256"],
                                accuracy=errors)
    return dict(schema=1, eligible_workload="PBE0 SAPT(DFT)-D4(I), induction NONE",
                scoped_validation_passed=True, full_sapt_validation_passed=False,
                original_job_status="FAILED", original_job_id=runtime["job_id"],
                validation=str(validation), scientific_commit=pin, backends=reports,
                exclusions=["cuEST XC CPKS response", "FDDS/full SAPT GPU support",
                            "performance or memory savings (not measured here)"],
                unsupported_error=UNSUPPORTED, input_sha256=hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--science", required=True)
    parser.add_argument("--pin", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = validate(args.validation, args.science, args.pin)
    output = Path(args.output)
    if output.exists():
        raise SystemExit("Refusing to overwrite a scoped validation receipt")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("Supported induction-NONE workload validated; original job remains FAILED.")


if __name__ == "__main__":
    main()
