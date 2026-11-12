"""Small-system SCF entrypoint screen; diagnostics, never speedup evidence.

One calculation per fresh process. SAPT stops immediately after its first
dimer SCF, before monomer/fragment work. No per-iteration Python tracing.
"""
import hashlib
import importlib
import json
from pathlib import Path
import re
from unittest.mock import patch

from dimer_energy_probe import scf_options
from peptide_scf import INPUT_SHA

HCL_GEOMETRY = """0 1
H 0.0 0.0 0.0
Cl 0.0 0.0 1.27
--
0 1
H 0.0 0.0 3.6
Cl 0.0 0.0 4.87
units angstrom
symmetry c1
no_com
no_reorient
"""


def trajectory(text):
    rows = []
    pattern = re.compile(
        r"@(?:DF-)?RHF\s+iter\s+(SAD|\d+):\s+([\d.eE+-]+)"
        r"\s+([\d.eE+-]+)\s+([\d.eE+-]+)(.*)")
    for line in text.splitlines():
        match = pattern.search(line)
        if not match:
            continue
        it, energy, delta, residual, tags = match.groups()
        it = 0 if it == "SAD" else int(it)
        if rows and it <= rows[-1]["iteration"]:
            raise RuntimeError("Unexpected second SCF block in dimer-only diagnostic")
        rows.append(dict(iteration=it, energy_Eh=energy, delta_Eh=delta,
                         residual=residual, tags=tags.strip()))
    if not rows:
        raise RuntimeError("No native RHF trajectory recorded")
    return rows


def first_dimer_only(psi4, molecule, sapt_module):
    """Run the actual SAPT preamble; never substitute a hand-built SCF helper."""
    class DimerFinished(BaseException):
        pass

    original = sapt_module.scf_helper
    captured = []

    def stop_after_dimer(*args, **kwargs):
        if captured:
            raise RuntimeError("More than one SAPT SCF requested")
        if kwargs.get("molecule") is not molecule:
            # prepare_sapt_molecule normally returns this exact dimer.
            raise RuntimeError("Unexpected molecule at the SAPT dimer boundary")
        wfn = original(*args, **kwargs)
        captured.append(wfn)
        raise DimerFinished()

    with patch.object(sapt_module, "scf_helper", stop_after_dimer):
        try:
            psi4.energy("sapt(dft)", molecule=molecule)
        except DimerFinished:
            return captured[0]
    raise RuntimeError("SAPT did not reach its dimer SCF")


def basis_receipt(basis):
    """Exact hex floats; normalized coefficients match the adapter's input."""
    shells = []
    for i in range(basis.nshell()):
        s = basis.shell(i)
        shells.append(dict(center=s.ncenter, am=s.am, pure=s.is_pure(),
                           xyz=[float(s.coord(k)).hex() for k in range(3)],
                           exponents=[float(s.exp(k)).hex() for k in range(s.nprimitive)],
                           coefficients=[float(s.coef(k)).hex() for k in range(s.nprimitive)]))
    payload = dict(nbf=basis.nbf(), nshell=basis.nshell(), shells=shells)
    payload["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def run(psi4, inputs_path, output, backend, entrypoint, system, threads=4):
    import numpy as np
    if entrypoint not in ("energy", "sapt") or system not in ("hcl", "peptide"):
        raise ValueError("Explicit small system and entrypoint required")
    if not 1 <= threads <= 4:
        raise ValueError("Use at most four cachy threads")
    raw = inputs_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != INPUT_SHA:
        raise RuntimeError("Frozen timing-suite manifest differs")
    inputs = json.loads(raw)
    if system == "peptide":
        point = inputs["systems"][system]
        geometry = point["geometry"]
        if hashlib.sha256(geometry.encode()).hexdigest() != point["geometry_sha256"]:
            raise RuntimeError("Peptide geometry differs")
    else:
        geometry = HCL_GEOMETRY
    output.mkdir(parents=True, exist_ok=False)
    output = output.resolve()
    psi4.core.set_output_file(str(output / "psi4.out"), False)
    psi4.set_memory("6 GiB")
    psi4.set_num_threads(threads)
    options = {**scf_options(backend), "freeze_core": True}
    # Match scientific options in both arms; only the public entrypoint varies.
    options.update(df_basis_sapt="aug-cc-pvdz-jkfit", df_basis_mp2="aug-cc-pvdz-ri",
                   sapt_dft_functional="HF", sapt_dft_do_fsapt="SAPTDFT",
                   sapt_dft_do_dhf=True, sapt_dft_use_einsums=True)
    psi4.set_options(options)
    mol = psi4.geometry(geometry)
    report = dict(status="STARTED", system=system, entrypoint=entrypoint,
                  backend=backend, options=options, threads=threads,
                  geometry=geometry, input_sha256=INPUT_SHA,
                  psi4_file=psi4.__file__, core_file=psi4.core.__file__,
                  version=psi4.__version__, per_iteration_observer=False,
                  primitive_preamble=False, accepted_speedups=0)
    wfn = None
    try:
        if entrypoint == "energy":
            energy, wfn = psi4.energy("hf", molecule=mol, return_wfn=True)
        else:
            module = importlib.import_module("psi4.driver.procrouting.sapt.sapt_proc")
            wfn = first_dimer_only(psi4, mol, module)
            energy = wfn.energy()
        report.update(status="CONVERGED_DIAGNOSTIC_ONLY", energy_Eh=float(energy))
    except psi4.SCFConvergenceError as error:
        # Nonconvergence is a preserved diagnostic outcome, not a passing gate.
        report.update(status="SCF_NONCONVERGENCE", error=str(error))
        wfn = error.wfn
    finally:
        psi4.core.flush_outfile()
        log = output / "psi4.out"
        report["log_sha256"] = hashlib.sha256(log.read_bytes()).hexdigest()
        report["trajectory"] = trajectory(log.read_text())
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    if wfn is not None:
        expected = "cuESTJK" if backend == "gpu" else "MemDFJK"
        report["jk_name"] = wfn.jk().name()
        if report["jk_name"] != expected:
            raise RuntimeError("Unexpected JK backend: " + report["jk_name"])
        report["primary"] = basis_receipt(wfn.basisset())
        report["auxiliary"] = basis_receipt(wfn.get_basisset("DF_BASIS_SCF"))
        # Post-SCF only: do not perturb the trajectory with snapshot work.
        np.savez(output / "endpoint.npz",
                 Cocc=np.array(wfn.Ca_subset("AO", "OCC").np, copy=True),
                 D=np.array(wfn.Da().np, copy=True), S=np.array(wfn.S().np, copy=True),
                 H=np.array(wfn.H().np, copy=True))
        report["endpoint_sha256"] = hashlib.sha256(
            (output / "endpoint.npz").read_bytes()).hexdigest()
        report["endpoint_note"] = (
            "Cocc and D are terminal objects, not guaranteed to be from the "
            "same energy-evaluation boundary; reconstruct D from Cocc for replay.")
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
