#!/usr/bin/env python3
"""CPU-only strict PW6B95 controls; diagnostic endpoints never become IE inputs."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import mgga_ie

PROTOCOLS = {
    "baseline": {},
    "damped-20": {"damping_percentage": 20.0, "damping_convergence": 1e-12},
    "timeout-only": {},
}
CASES = [(system, protocol, fragment)
         for system, protocol in (("water", "baseline"), ("water", "damped-20"),
                                  ("benzene", "timeout-only"))
         for fragment in ("A", "B")]
ITERATION = re.compile(
    r"@DF-RKS iter\s+(\d+):\s+([-+.\deE]+)\s+([-+.\deE]+)\s+([-+.\deE]+)")


def history(path):
    iterations = []
    if path.exists():
        for number, line in enumerate(path.read_text().splitlines(), 1):
            match = ITERATION.search(line)
            if match:
                i, e, de, residual = match.groups()
                iterations.append(dict(iteration=int(i), energy_hartree=float(e),
                                       delta_energy_hartree=float(de), residual=float(residual),
                                       output_line=number))
    return dict(count=len(iterations), last=iterations[-1] if iterations else None,
                min_residual_last20=min((r["residual"] for r in iterations[-20:]), default=None),
                iterations=iterations,
                note="RMS commutator residual, not a successive-density difference. "
                     "Iteration energies from failed workers are not converged references.")


def worker(a):
    # Reuse the established fresh-process worker and route assertions. The
    # option delta is injected only within this isolated diagnostic process.
    original = mgga_ie.options
    mgga_ie.options = lambda basis, route: dict(original(basis, route), **PROTOCOLS[a.protocol])
    return mgga_ie.worker(a)


def campaign(a):
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).resolve()
    package = a.package.resolve()
    env = mgga_ie.build_environment("host", package, None)
    pin = mgga_ie.provenance(package, env)
    manifest = dict(job=os.getenv("SLURM_JOB_ID"), package=str(package), provenance=pin,
                    harness_commit=subprocess.check_output(
                        ["git", "-C", str(script.parent), "rev-parse", "HEAD"], text=True).strip(),
                    script_sha256=mgga_ie.digest(script), protocols=PROTOCOLS, records=[],
                    protocol="Native CPU XC/J/K only. Same original strict SCF thresholds, "
                             "200 iterations, 99x590 robust grid, full precision. "
                             "Water timeout 600s, benzene timeout 1200s. No complete IE implied.",
                    solver_note="SOSCF is not attempted: current source RV::compute_Vx_full "
                                "rejects MGGA rotated builds. Damping is a labeled solver control.")
    mgga_ie.atomic_json(out / "manifest.json", manifest)
    for system, protocol, fragment in CASES:
        name = f"{system}-pw6b95-host-cpu-{protocol}-{fragment}"
        directory = out / name
        timeout = 1200 if system == "benzene" else 600
        cmd = [sys.executable, str(script), "--worker", "--output", str(directory),
               "--package", str(package), "--system", system, "--basis", "aug-cc-pvdz",
               "--functional", "pw6b95", "--route", "cpu", "--fragment", fragment,
               "--expected-xc", "native-host", "--protocol", protocol]
        with (out / f"{name}.log").open("w") as log:
            try:
                code = subprocess.run(cmd, env=env, stdout=log,
                                      stderr=subprocess.STDOUT, timeout=timeout).returncode
            except subprocess.TimeoutExpired:
                code = 124
        file = directory / "result.json"
        result = mgga_ie.process_result(json.loads(file.read_text()) if file.exists()
                                       else dict(ok=False, error="Missing result.json"), code)
        item = dict(name=name, system=system, protocol=protocol, fragment=fragment,
                    timeout_s=timeout, returncode=code, result=result,
                    history=history(directory / "psi4.out"))
        manifest["records"].append(item)
        mgga_ie.atomic_json(out / "manifest.json", manifest)
        print(name, "exit", code, "endpoint", item["history"]["last"], flush=True)
    assert all(mgga_ie.digest(Path(p)) == h for p, h in pin["sha256"].items()), "Build changed"
    # A diagnosis can finish with reproducible SCF failures. Keep execution
    # completeness separate from scientific convergence and no COMPLETE IE claim.
    def attributable(r):
        if r["returncode"] == 0:
            return r["result"]["ok"]
        if r["returncode"] == 1:
            return r["result"].get("error", "").startswith("SCFConvergenceError:")
        return r["returncode"] == 124 and r["history"]["count"] > 0
    diagnosed = all(attributable(r) for r in manifest["records"])
    mgga_ie.atomic_json(out / "DIAGNOSIS.json", dict(
        workers_recorded=len(manifest["records"]), expected=len(CASES),
        all_endpoints_attributable=diagnosed,
        all_converged=all(r["result"]["ok"] for r in manifest["records"]),
        final_provenance_verified=True,
        note="No valid interaction energies are produced by this diagnostic."))
    return 0 if diagnosed else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--protocol", choices=PROTOCOLS)
    for key in ("system", "basis", "functional", "route", "fragment", "expected-xc"):
        p.add_argument("--"+key)
    a = p.parse_args()
    raise SystemExit(worker(a) if a.worker else campaign(a))
