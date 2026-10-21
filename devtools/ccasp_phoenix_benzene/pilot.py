# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""P-only Phoenix pilot adapter. Not a launcher or an authorization to run.

Science is bound to pilot.json and the externally staged, hash-pinned fixture.
No historical wrapper, overlay, reference output, subprocess launcher, signal,
watchdog, private monkeypatch, second partition or external comparator is used.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    # Run-owned files only. A failed/partial write is NOT a completion marker.
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def recipe(molecule, cfg):
    """Definitions from recipe56o (b6d0bb1c...), not archived run state.

    Even-tempered C 2**(8..-3), H 2**(5..-2), normalized primitive s shells:
    published CamCASP ISA/set2 definition plus MIN-S-EXP H=0.2 rule.
    Angular shells and molecular AUX come from this build's Psi4 basis library.
    """
    import numpy as np
    from psi4 import core
    from psi4.driver.procrouting import isapol_native_partition as isa
    from psi4.driver.procrouting.isapol_basis import BasisRecipe, ShellRecipe

    if isa.DRHO_REFINEMENT_ITERATIONS != cfg["drho_refinement_iterations"]:
        raise ValueError("Drho refinement semantics differ from the bound profile")
    xyz = molecule.geometry().np
    symbols = [molecule.symbol(i) for i in range(12)]
    if symbols != ["C"] * 6 + ["H"] * 6:
        raise ValueError("requires cyclic C1..C6 then H1..H6")
    for i in range(6):
        if np.argmin(np.linalg.norm(xyz[:6] - xyz[i + 6], axis=1)) != i:
            raise ValueError("C/H topology mismatch")
        if not 2.4 < np.linalg.norm(xyz[i] - xyz[(i + 1) % 6]) < 2.9:
            raise ValueError("noncyclic ring")
    native = core.BasisSet.build(molecule, "DF_BASIS_SCF", "aug-cc-pVTZ-RI", puream=0)
    if native.nbf() != cfg["auxiliary_nbf"]:
        raise ValueError("Cartesian response AUX must have 1146 functions")
    centres = tuple(map(tuple, xyz))
    shells = tuple(ShellRecipe(int(native.shell_to_center(i)), int(native.shell(i).am),
        tuple(native.shell(i).exp(k) for k in range(native.shell(i).nprimitive)),
        tuple(native.shell(i).coef(k) for k in range(native.shell(i).nprimitive)))
        for i in range(native.nshell()))
    origin = "recipe56o mathematical declarations; current Psi4 basis data"
    auxiliary = BasisRecipe("aug-cc-pVTZ-RI", origin, "Cartesian", centres, shells)
    sites = []
    for i, symbol in enumerate(symbols):
        label = f"{symbol}{i % 6 + 1}"
        powers = range(8, -4, -1) if symbol == "C" else range(5, -3, -1)
        ss = tuple(ShellRecipe(0, 0, (2.**p,), ((2. * 2.**p / np.pi)**.75,)) for p in powers)
        angular = tuple(ShellRecipe(0, s.l, s.exponents, s.coefficients)
                        for s in shells if s.centre == i and s.l > 0)
        shape = BasisRecipe(label + " shape", origin, "Spherical", (centres[i],), ss)
        atomic = BasisRecipe(label + " atomic", origin, "Spherical", (centres[i],), ss + angular)
        sites.append(isa.SiteRecipe(label, centres[i], atomic, shape, tuple(range(len(ss))), 4,
            cfg["tail_r1_multiplier"] * core.isapol_slater_radius(int(molecule.Z(i))), True))
    controller = isa.ControllerRecipe(
        convergence=1e-9, max_iterations=120, w_eps=.17, positive_lambda=.001,
        positive_max_alpha=.2, positive_auto=True, damping=0., s_block_only=True,
        density_cutoff=1e-36, w_eps_activation=1e-5, positive_activation=1e-5,
        tail_activation=cfg["tail_activation"], mixing=0., mixing_skip=20,
        tail_iteration_limit=20, fix_tails=True, tail_fit_policy=cfg["tail_fit_policy"])
    return isa.PartitionRecipe("Phoenix benzene recipe56o adaptation", origin,
        "explicit_cartesian_drho_c_isa_a", auxiliary, tuple(sites),
        isa.GridRecipe(100, 200, 3, 1., "native_tabulated_bragg_slater",
                       "all_sites_unscreened_full_molecular_grid"),
        controller, "Drho1e-2", atomic_initialization="zero_atomic_D0")


def export(result, quadrature, out):
    """Serialize existing successful public results; no new contraction algebra."""
    import numpy as np
    from psi4.driver.procrouting.isapol_refine import COMPONENT_NAMES

    sites = result.refinements[0].model.sites
    arrays = {"omega_au": np.asarray(result.frequencies),
              "cp_weights": np.asarray(quadrature.cp_weights)}
    for n, (local, refined) in enumerate(zip(result.local_tensors, result.refined_tensors)):
        # Public LW snapshots retain a zero-padded rank-3 block (15x15).
        # Export exactly the declared rank<=2 positive-rank block, not padding.
        arrays[f"lw_{n:02d}"] = local.array[..., :8, :8]
        arrays[f"refined_{n:02d}"] = np.asarray(refined)
    np.savez(out / "local-response.npz", **arrays)
    pairs = []
    for p in result.dispersion.pairs:
        for c in p.coefficients:
            pairs.append(dict(site_a=sites[p.site_a].label, site_b=sites[p.site_b].label,
                order=c.order, value=float(c.value),
                unrestricted_complete=bool(c.unrestricted_complete),
                included_rank_pairs=list(c.included_rank_pairs),
                missing_rank_pairs=list(c.missing_rank_pairs)))
    if any(sum(p["order"] == order for p in pairs) != 144 for order in (6, 8, 10)):
        raise ValueError("expected 144 ordered site-pair coefficients per order")
    angular = result.recoupled_dispersion
    if angular is None:
        raise ValueError("requested recoupled result missing")
    angular_pairs = []
    for p in angular.pairs:
        angular_pairs.append(dict(site_a=p.site_a, site_b=p.site_b,
            coefficients=[dict(order=c.order, t=c.t, u=c.u, J=c.J, value=c.value)
                          for c in p.coefficients],
            coverage=[dict(order=c.order, table_complete=c.table_complete,
                unrestricted_complete=c.unrestricted_complete,
                included_rank_quadruples=list(c.included_rank_quadruples),
                missing_table_rank_quadruples=list(c.missing_table_rank_quadruples),
                missing_unrestricted_rank_quadruples=list(c.missing_unrestricted_rank_quadruples))
                for c in p.coverage]))
    write_json(out / "properties.json", dict(
        convention="real Racah/no-CS, local-to-global frame columns; imaginary frequencies in au",
        units="atomic units; alpha_(l,l') in a0**(l+l'+1); Cn in Eh*a0**n",
        component_names=COMPONENT_NAMES[:9],
        lw_components=COMPONENT_NAMES[1:9], lw_axes="frequency=singleton,site,component,component",
        refined_axes="site,component,component, including zero 00 row/column",
        sites=[dict(label=s.label, copy_type=s.site_type, origin_bohr=s.origin_bohr,
                    frame=s.frame, rank_limit=s.rank_limit) for s in sites],
        isotropic_pairs=pairs, c10="partial (2,2) only; not unrestricted complete C10",
        recoupled_convention=angular.convention, recoupled_pairs=angular_pairs,
        declared_model_coverage=[vars(c) for c in angular.declared_model_coverage],
        penalty_schemes=[dict(frequency=r.model.frequency_au,
            selection=r.model.penalty_selection, static_reference_sha256=r.model.static_reference_sha256,
            anchor_sha256=r.model.anchor_sha256, labels=r.model.parameter_labels)
            for r in result.refinements],
        provenance_repr=repr(result.provenance), resources=result.resources,
        diagnostics_repr=repr(result.diagnostics),
        parity="not evaluated; historical benzene comparator is scalar/isotropic"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for arg in ("source", "commit", "stage", "core-sha256", "fixture", "out", "scratch"):
        ap.add_argument("--" + arg, required=True)
    args = ap.parse_args()
    # Refuse before scientific imports, directory creation or any numerical work.
    host = socket.gethostname().split(".")[0]
    allocated_node = os.environ.get("SLURMD_NODENAME", "").split(".")[0]
    if (not os.environ.get("SLURM_JOB_ID") or os.environ.get("USER") != "awallace43"
            or not os.environ.get("HOME", "").startswith("/storage/home/")
            or host.startswith("login") or not allocated_node or host != allocated_node):
        raise SystemExit("Phoenix compute allocation only; no local or login-node science")
    source, stage, fixture = map(lambda p: Path(p).resolve(),
                                (args.source, args.stage, args.fixture))
    cfg_path = Path(__file__).with_name("pilot.json")
    cfg = json.loads(cfg_path.read_text())
    if git(source, "rev-parse", "HEAD") != args.commit or git(source, "status", "--porcelain"):
        raise SystemExit("source commit/clean-state mismatch")
    git(source, "merge-base", "--is-ancestor", cfg["candidate_base"], args.commit)
    changed = git(source, "diff", "--name-only", cfg["candidate_base"], args.commit).splitlines()
    if any(not p.startswith("devtools/ccasp_phoenix_benzene/") for p in changed):
        raise SystemExit("inherited scientific source changed")
    if not Path(__file__).resolve().is_relative_to(source) or not stage.is_relative_to(source):
        raise SystemExit("driver and private stage must belong to pinned source")
    if sha(fixture) != cfg["fixture_sha256"]:
        raise SystemExit("geometry fixture hash mismatch")
    raw = json.loads(fixture.read_text())
    if (raw["source_sha256"] != cfg["source_dataframe_sha256"]
            or raw["system_id"] != "24_Benzene-Benzene_pi-pi_1.00"
            or raw["dataframe_index"] != 3806 or raw["fragment"] != "A"
            or raw["charge"] != 0 or raw["multiplicity"] != 1):
        raise SystemExit("geometry attribution mismatch")
    out, scratch = Path(args.out).resolve(), Path(args.scratch).resolve()
    if out.exists() or scratch.exists():
        raise SystemExit("output and job-owned scratch must both be new")
    if not str(scratch).startswith(("/storage/scratch1/3/awallace43/", "/scratch/")):
        raise SystemExit("job scratch outside approved Phoenix roots")
    free = os.statvfs(scratch.parent)
    if free.f_bavail * free.f_frsize < cfg["resources"]["required_free_scratch_gib"] * 1024**3:
        raise SystemExit("less than 400 GiB scratch free")
    out.mkdir(parents=True, exist_ok=False)
    scratch.mkdir(exist_ok=False)
    for name in ("TMPDIR", "SCRATCH", "PSI_SCRATCH"):
        os.environ[name] = str(scratch)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    record = dict(status="started", science_pass=False, commit=args.commit,
        job_id=os.environ["SLURM_JOB_ID"], host=host, allocated_node=allocated_node,
        config_sha256=sha(cfg_path),
        fixture_sha256=sha(fixture), started_unix=time.time(),
        failure_capture="SCF record, log and traceback only; no pre-LW tensor capture adapter",
        orbital_source="own native PBE0 SCF; no C state or historical overlay")
    write_json(out / "started.json", record)
    try:
        import numpy as np
        import psi4
        from psi4 import core
        from psi4.driver.procrouting.isapol_bounded_response import BoundedResources
        from psi4.driver.procrouting.isapol_native import Quadrature
        from psi4.driver.procrouting.isapol_native_propagator import KernelSmoothing
        from psi4.driver.procrouting import isapol_lw
        if (not Path(psi4.__file__).resolve().is_relative_to(stage)
                or not Path(core.__file__).resolve().is_relative_to(stage)
                or sha(core.__file__) != args.core_sha256):
            raise ValueError("package/core is not the smoke-identified private stage")
        if isapol_lw.PRODUCTION_TOLERANCE != cfg["response"]["production_tolerance"]:
            raise ValueError("production tolerance changed")
        record.update(package=psi4.__file__, core=core.__file__, core_sha256=sha(core.__file__),
                      numpy_version=np.__version__, python=sys.executable)
        psi4.set_output_file(str(out / "psi4.out"), False)
        psi4.set_memory(cfg["resources"]["psi4_memory_bytes"])
        psi4.set_num_threads(1)
        xyz = [raw["qcel_geometry_bohr_float64"][i] for i in cfg["permutation"]]
        geom = ["0 1", "units bohr", "no_com", "no_reorient", "symmetry c1"]
        geom += [f"{'C' if i < 6 else 'H'} {x:.8f} {y:.8f} {z:.8f}"
                 for i, (x, y, z) in enumerate(xyz)]
        molecule = psi4.geometry("\n".join(geom))
        if not np.array_equal(molecule.geometry().np, np.asarray(xyz)):
            raise ValueError("supplied dimer-frame coordinates changed")
        psi4.set_options(cfg["scf"]["options"])
        energy, wfn = psi4.energy(cfg["scf"]["method"], molecule=molecule, return_wfn=True)
        if wfn.functional().grac_shift() != 0:
            raise ValueError("unexpected GRAC shift")
        write_json(out / "scf.json", dict(energy_Eh=float(energy), own_scf=True,
                   geometry_bohr=xyz, nbf=wfn.basisset().nbf(), commit=args.commit))
        part = recipe(molecule, cfg["isa"])
        grid_options = core.IsaGridOptions()
        grid_options.radial_points, grid_options.spherical_points = cfg["isa"]["q_grid"]
        grid = core.IsaGrid(molecule.clone(), grid_options)
        q_grid = np.column_stack((grid.x(), grid.y(), grid.z(), grid.w()))
        del grid
        quadrature = Quadrature.from_casimir(
            core.CasimirGrid(cfg["quadrature"]["n_freq"], cfg["quadrature"]["beta"]))
        r, res = cfg["response"], cfg["resources"]
        ref = {k: v for k, v in cfg["refinement"].items() if k != "penalty_selection"}
        psi4.oeprop(wfn, "ATOMIC_REFINED_POLARIZABILITIES", "ATOMIC_REFINED_DISPERSION",
            atomic_backend="BOUNDED_DF", preset="benzene", distribution="isa",
            partition_recipe=part, partition_grid=q_grid, auxiliary_recipe=part.auxiliary,
            auxiliary_basis=r["auxiliary_basis"], response=r["model"],
            radial_points=r["grid"][0], spherical_points=r["grid"][1],
            smoothing=KernelSmoothing(r["rho_eps"], r["f_max"], r["fd_delta"], r["fd_alpha"], "FD"),
            shell_cutoff=r["shell_cutoff"], charge_penalty=r["charge_penalty"],
            anchor_metric_damping=r["anchor_eta"], localization_rank_limit=r["lw_rank"],
            quadrature=quadrature, scf_correction="NONE", expected_grac_shift=None,
            resources=BoundedResources(res["psi4_memory_bytes"], res["max_work"],
                                       res["max_io_bytes"], capacity=res["capacity"]),
            scratch_directory=str(scratch), verbosity=1, **ref, **cfg["dispersion"])
        export(psi4.atomic_property_result(wfn), quadrature, out)
        record.update(status="complete", science_pass=True)
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}",
                      traceback=traceback.format_exc())
    record["finished_unix"] = time.time()
    write_json(out / "result.json", record)
    return 0 if record["science_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
