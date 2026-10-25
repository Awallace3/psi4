# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Synthetic ACTUAL driver-kwargs regression. No native/real engine imports."""
import argparse
import copy
from pathlib import Path
import os
import sys
import tempfile
from types import SimpleNamespace as NS
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]/"tests/pytests"))
from isapol_source_test_support import load_source
from test_angular_full import load
import numpy as np

B = load_source("isapol_bounded")
F = load_source("isapol_refine")
BR = load_source("isapol_bounded_response")
S = load("pdef_boundary_shared", "integrated_driver.py")


def anchors(crossing=4e-5):
    sites = tuple(F.RefinementSite(f"{t}{n}", t, (float(i), 0., 0.), np.eye(3), 2)
                  for i, (t, n) in enumerate((t, n) for t in ("C", "H") for n in range(1, 7)))
    static = [np.zeros((9, 9)) for _ in sites]
    for a in static:
        a[1:, 1:] = .02
    static[6][3, 5] = static[6][5, 3] = crossing
    # COPY members cannot select variables or replace reference anchors.
    static[1][1, 1] = 9.
    static[7][3, 5] = static[7][5, 3] = 7.
    nodes = []
    for node in range(11):
        current = [a.copy() for a in static]
        if node:
            current[0][1, 1] = 3e-5  # downward crossing stays penalized
        if node >= 3:
            current[6][3, 5] = current[6][5, 3] = 2e-4
        nodes.append((node * .03, current))
    return sites, nodes


def drive(which, *, crossing=4e-5, config_mutator=None, observed=None):
    """Run wrapper AND shared driver; oeprop adapter forwards only captured kwargs."""
    job = load("boundary_"+which+"_job", which+"_job.py")
    profile = load("boundary_"+which+"_profile", which+"_profile.py")
    driver = load("boundary_"+which+"_driver", which+"_driver.py")
    cfg, identity = profile.configuration(job.R)
    if config_mutator:
        config_mutator(cfg)
        identity = dict(identity, effective_config=cfg)
    sites, nodes = anchors(crossing)
    frequencies = tuple(w for w, _ in nodes)
    observed = {} if observed is None else observed
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root/"results").mkdir()
        scratch = root/"scratch"
        scratch.mkdir()
        fixture = root/"toy-private.json"
        fixture.write_text(__import__("json").dumps(dict(source_sha256=cfg["source_dataframe_sha256"],
            system_id="24_Benzene-Benzene_pi-pi_1.00", dataframe_index=3806,
            fragment="A", charge=0, multiplicity=1)))
        grid_options = NS()
        core = NS(__file__=str(root/"NOT-ELF"), IsaGridOptions=lambda: grid_options,
            IsaGrid=lambda *a: NS(x=lambda: [1.], y=lambda: [2.], z=lambda: [3.], w=lambda: [4.]),
            CasimirGrid=lambda *a: NS(frequencies=frequencies))
        modules = {
            "psi4": NS(core=core),
            "psi4.driver.p4util": NS(OptionsState=object()),
            "psi4.driver.procrouting": NS(isapol_lw=NS(PRODUCTION_TOLERANCE=1e-6)),
            "psi4.driver.procrouting.isapol_bounded_response": NS(BoundedResources=BR.BoundedResources),
            "psi4.driver.procrouting.isapol_native": NS(Quadrature=NS(from_casimir=lambda q: q)),
            "psi4.driver.procrouting.isapol_native_propagator": NS(KernelSmoothing=lambda *a: a)}
        wfn = NS(functional=lambda: NS(grac_shift=lambda: 0.), basisset=lambda: NS(nbf=lambda: 414))

        def oeprop(*args, **kwargs):
            observed["kwargs"] = kwargs
            # NEVER insert a declaration here. Missing kwargs remain None.
            models = B.refinement_models(sites, nodes, **{k: kwargs[k] for k in
                ("cutoff", "weight_type", "weight_coefficient", "declared_variables") if k in kwargs})
            observed.update(models=models, sites=sites, nodes=nodes)
            observed["result"] = NS(frequencies=frequencies, local_tensors=[None]*11,
                refinements=[NS(model=m) for m in models], refined_tensors=[None]*11,
                diagnostics=[dict(localization_residual=1e-8)]*11,
                dispersion=NS(pairs=[None]*144), recoupled_dispersion=NS(pairs=[None]*144),
                resources=dict(maximum_numeric_plan=26570915696, charged_work=0, charged_io_bytes=0))

        engine = NS(set_output_file=mock.Mock(), set_memory=mock.Mock(), set_num_threads=mock.Mock(),
            set_options=mock.Mock(), energy=mock.Mock(return_value=(-1., wfn)), oeprop=oeprop,
            atomic_property_result=lambda w: observed["result"])
        prior = NS(make_molecule=lambda *a: (NS(clone=lambda: None), []),
            recipe=mock.Mock(), export=mock.Mock())
        sink_module = NS(DiagnosticSink=lambda *a, **kw: NS(written=0))
        def admit(core, options, molecule, audit, *args):
            audit.update(basis_files={}, main={"fingerprint_sha256": "main"},
                         auxiliary={"fingerprint_sha256": "aux"})
            return None, NS(auxiliary="synthetic AUX")
        profile_stub = NS(configuration=lambda r: (copy.deepcopy(cfg), copy.deepcopy(identity)))
        def module(name, file):
            return {which+"_profile.py": profile_stub, "integrated_driver.py": S,
                    "pilot.py": prior, "diagnostic_sink.py": sink_module}[file]
        rt = NS(HERE=HERE, ROOT=root, PLAN=dict(run=str(root), scratch=str(scratch),
            fixture=str(fixture), runtime=str(root/"runtime"), allocation=job.R.PLAN["allocation"],
            prior_probe_sha256="probe"), OLD=job.R.OLD, compute_guard=mock.Mock(), module=module,
            admit_basis=admit, write=job.R.write, sha=lambda p: (
                cfg["fixture_sha256"] if Path(p) == fixture else job.R.sha(p) if Path(p).exists() else "e"*64),
            BASIS=NS(compare_scf_main=mock.Mock()))
        with mock.patch.dict(sys.modules, modules), mock.patch.dict(os.environ, SLURM_JOB_ID="SYNTHETIC"), \
             mock.patch.object(S.os, "statvfs", return_value=NS(f_bavail=401*1024**3, f_frsize=1)):
            observed["completion"] = driver.capture_science(rt, engine,
                {"effective_profile": identity, "commit": "a"*40, "tree": "b"*40}, "r", "d")
        observed.update(memory=engine.set_memory.call_args.args[0],
            grid=(grid_options.radial_points, grid_options.spherical_points),
            exported=prior.export.call_count)
    return observed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", choices=("angular", "pdef"), required=True)
    args = parser.parse_args()
    observed = {}
    try:
        out = drive(args.driver, observed=observed)
    except ValueError as error:
        print("ACTUAL DRIVER REFUSAL:", error)
        print("driver:", args.driver, "declaration supplied:", "declared_variables" in observed["kwargs"])
        return 1
    print("SYNTHETIC ACTUAL DRIVER GREEN:", args.driver,
          "nodes", len(out["models"]), "parameters", out["models"][0].parameter_count,
          "static penalties", 72-len(out["models"][0].unpenalized_variables))
    print("No native solve/LW, actual failing-anchor replay, or property acceptance established.")
    assert not any(n == "adcc" or n == "einsums" or n.startswith("psi4.") for n in sys.modules)
    assert getattr(sys.modules.get("psi4"), "_source_only_fake", False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
