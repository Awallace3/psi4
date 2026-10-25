# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""One own SCF/ISA/property pipeline with explicit diagnostics; no local activation."""
import json
import os
from pathlib import Path


def capture_science(runtime, psi4, binding, runtime_sha, derivative_sha, *,
                    configuration=None, profile_identity=None, validate_result=None):
    """Called only by the reviewed same-interpreter allocated job; no loader fallback."""
    runtime.compute_guard()
    prior = runtime.module("_ccasp_frozen_recipe","pilot.py")
    sink_module = runtime.module("_ccasp_capture_sink","diagnostic_sink.py")
    plan, old = runtime.PLAN, runtime.OLD
    root = Path(plan["run"])
    out = root/"results"
    scratch = Path(plan["scratch"])/("payload-"+os.environ["SLURM_JOB_ID"])
    cfg = (json.loads((runtime.HERE/"pilot.json").read_text())
           if configuration is None else configuration)
    if cfg["resources"]["psi4_memory_bytes"] != plan["allocation"]["psi4_memory_bytes"]:
        raise ValueError("distinct approved remote numerical memory profile differs")
    free = os.statvfs(scratch.parent)
    if free.f_bavail*free.f_frsize < cfg["resources"]["required_free_scratch_gib"]*1024**3:
        raise ValueError("less than unchanged400GiB required free scratch;200GiB is not a hard quota")
    scratch.mkdir(mode=0o700,exist_ok=False)
    for name in ("TMPDIR","SCRATCH","PSI_SCRATCH"):
        os.environ[name] = str(scratch)
    fixture = Path(plan["fixture"])
    if runtime.sha(fixture) != cfg["fixture_sha256"]:
        raise ValueError("original private fixture hash differs")
    raw = json.loads(fixture.read_text())
    if (raw["source_sha256"] != cfg["source_dataframe_sha256"]
            or raw["system_id"] != "24_Benzene-Benzene_pi-pi_1.00" or raw["dataframe_index"] != 3806
            or raw["fragment"] != "A" or raw["charge"] != 0 or raw["multiplicity"] != 1):
        raise ValueError("original geometry attribution differs")
    import numpy as np
    from psi4 import core
    from psi4.driver.p4util import OptionsState
    from psi4.driver.procrouting.isapol_bounded_response import BoundedResources
    from psi4.driver.procrouting.isapol_native import Quadrature
    from psi4.driver.procrouting.isapol_native_propagator import KernelSmoothing
    psi4.set_output_file(str(out/"psi4.out"),False)
    psi4.set_memory(cfg["resources"]["psi4_memory_bytes"])
    psi4.set_num_threads(1)
    molecule,xyz = prior.make_molecule(psi4,cfg,raw)
    psi4.set_options(cfg["scf"]["options"])
    admission = {}
    try:
        _,part = runtime.admit_basis(core,OptionsState,molecule,admission,binding,runtime_sha,
            Path(plan["runtime"])/"basis-derivative-admission.json",derivative_sha,
            lambda auxiliary: prior.recipe(molecule,cfg["isa"],auxiliary_native=auxiliary))
    finally:
        runtime.write(out/"basis-identities-before-scf.json",admission)
    energy,wfn = psi4.energy(cfg["scf"]["method"],molecule=molecule,return_wfn=True)
    if wfn.functional().grac_shift() != 0:
        raise ValueError("unexpected GRAC shift")
    runtime.write(out/"scf.json",dict(energy_Eh=float(energy),own_scf=True,geometry_bohr=xyz,nbf=wfn.basisset().nbf()))
    try:
        runtime.BASIS.compare_scf_main(wfn.basisset(),admission["main"],admission)
    finally:
        runtime.write(out/"basis-identities-after-scf.json",admission.get("actual_scf_main"))
    grid_options = core.IsaGridOptions()
    grid_options.radial_points,grid_options.spherical_points = cfg["isa"]["q_grid"]
    grid = core.IsaGrid(molecule.clone(),grid_options)
    q_grid = np.column_stack((grid.x(),grid.y(),grid.z(),grid.w()))
    del grid
    quadrature = Quadrature.from_casimir(core.CasimirGrid(cfg["quadrature"]["n_freq"],cfg["quadrature"]["beta"]))
    stage = Path(plan["runtime"])/"stage"
    identities = dict(
        native_core=dict(source_root=str(old.SOURCE),source_commit=old.BUILD_COMMIT,
            source_tree=old.BUILD_TREE,path=str(Path(core.__file__).resolve()),
            sha256=old.CORE_SHA,build_node=old.BUILD_NODE),
        python_boundary=dict(source_root=str(runtime.ROOT),source_commit=binding["commit"],
            source_tree=binding["tree"],path=str(stage/"lib/psi4/driver/procrouting/isapol_lw.py"),
            package_root=str(stage/"lib/psi4"),sha256=runtime.sha(stage/"lib/psi4/driver/procrouting/isapol_lw.py")),
        driver=dict(source_root=str(runtime.ROOT),source_commit=binding["commit"],source_tree=binding["tree"]),
        config_sha256=(runtime.sha(runtime.HERE/"pilot.json") if profile_identity is None
                       else profile_identity["effective_config_sha256"]),
        fixture_sha256=runtime.sha(fixture),basis_manifest_sha256=derivative_sha)
    sink = sink_module.DiagnosticSink(out/"diagnostic",identities,context=dict(
        runtime_manifest_sha256=runtime_sha,original_probe_sha256=plan["prior_probe_sha256"],
        derivative_admission_sha256=derivative_sha,basis_files=admission["basis_files"],
        main_fingerprint=admission["main"]["fingerprint_sha256"],
        auxiliary_fingerprint=admission["auxiliary"]["fingerprint_sha256"],
        basis_receipt_sha256=runtime.sha(out/"basis-identities-before-scf.json"),
        source_binding=binding,frames="original preset Cn->Hn z, Cn->Cn+1 x; absolute dimer frame",
        **({} if profile_identity is None else dict(effective_profile=profile_identity))))
    r,res = cfg["response"],cfg["resources"]
    refinement = {k:v for k,v in cfg["refinement"].items() if k!="penalty_selection"}
    psi4.oeprop(wfn,"ATOMIC_REFINED_POLARIZABILITIES","ATOMIC_REFINED_DISPERSION",
        atomic_backend="BOUNDED_DF",preset="benzene",distribution="isa",
        partition_recipe=part,partition_grid=q_grid,auxiliary_recipe=part.auxiliary,
        auxiliary_basis=r["auxiliary_basis"],response=r["model"],
        radial_points=r["grid"][0],spherical_points=r["grid"][1],
        smoothing=KernelSmoothing(r["rho_eps"],r["f_max"],r["fd_delta"],r["fd_alpha"],"FD"),
        shell_cutoff=r["shell_cutoff"],charge_penalty=r["charge_penalty"],
        anchor_metric_damping=r["anchor_eta"],localization_rank_limit=r["lw_rank"],
        quadrature=quadrature,scf_correction="NONE",expected_grac_shift=None,
        resources=BoundedResources(res["psi4_memory_bytes"],res["max_work"],res["max_io_bytes"],capacity=res["capacity"]),
        scratch_directory=str(scratch),verbosity=1,diagnostic_sink=sink,**refinement,**cfg["dispersion"])
    result = psi4.atomic_property_result(wfn)
    if validate_result is not None:
        validate_result(result,quadrature,cfg)
    prior.export(result,quadrature,out)  # only after full unchanged gates
    return dict(status="complete",science_pass=True,energy_Eh=float(energy),diagnostic_written_bytes=sink.written)
