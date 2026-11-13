"""Protein83 beta-only HF control on the existing native build.

No iteration observer, native overlay, threshold relaxation, or retry.
Preserve initialization receipts and the native trajectory on failure too.
"""
import json

from dimer_energy_probe import run as energy_run
from peptide_plan_probe import hf_beta_control
from phoenix_timing import digest, save
from small_dimer_probe import trajectory


def run(psi4, inputs, output, backend, threads, beta):
    if backend != "gpu" or beta not in (0, 1):
        raise ValueError("Explicit GPU beta0/beta1 diagnostic required")
    records = []
    try:
        with hf_beta_control(psi4.core, beta, records):
            energy_run(psi4, inputs, output, backend, threads)
        if len(records) != 1 or records[0]["after"]["beta"] != beta:
            raise RuntimeError("Single controlled SCF initialization not confirmed")
    finally:
        result = output / "result.json"
        if result.is_file():
            report = json.loads(result.read_text())
            report.update(beta_control=beta, beta_initializations=records,
                          beta_control_verified=(len(records) == 1
                              and records[0]["after"]["beta"] == beta),
                          per_iteration_observer=False,
                          scope="PROTEIN83_ZERO_OMEGA_BETA_ONLY_SCF_CONTROL")
            log = output / "psi4.out"
            if log.is_file():
                report["log_sha256"] = digest(log)
                try:
                    report["trajectory"] = trajectory(log.read_text())
                except RuntimeError as error:
                    report["trajectory_error"] = str(error)
            save(result, report)
