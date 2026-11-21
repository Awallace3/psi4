"""Authenticated sealed worker for fsapt1, fixed-HF fdisp0, and protein157 GPU probes.

Runs inside a fresh process after calling the build's authenticated
sealed_bootstrap.bootstrap(config, build) before importing Psi4 or Einsums.
"""
import argparse
from contextlib import ExitStack
import hashlib
import importlib
import json
import os
from pathlib import Path
import resource
import sys
import time
from unittest.mock import patch

import numpy as np


# Exact literal S22 phenol dimer geometry from tests/fsapt1/input.dat.
PHENOL_DIMER = """0 1
O    -1.3885044    1.9298523   -0.4431206
H    -0.5238121    1.9646519   -0.0064609
C    -2.0071056    0.7638459   -0.1083509
C    -1.4630807   -0.1519120    0.7949930
C    -2.1475789   -1.3295094    1.0883677
C    -3.3743208   -1.6031427    0.4895864
C    -3.9143727   -0.6838545   -0.4091028
C    -3.2370496    0.4929609   -0.7096126
H    -0.5106510    0.0566569    1.2642563
H    -1.7151135   -2.0321452    1.7878417
H    -3.9024664   -2.5173865    0.7197947
H    -4.8670730   -0.8822939   -0.8811319
H    -3.6431662    1.2134345   -1.4057590
--
0 1
O     1.3531168    1.9382724    0.4723133
H     1.7842846    2.3487495    1.2297110
C     2.0369747    0.7865043    0.1495491
C     1.5904026    0.0696860   -0.9574153
C     2.2417367   -1.1069765   -1.3128110
C     3.3315674   -1.5665603   -0.5748636
C     3.7696838   -0.8396901    0.5286439
C     3.1224836    0.3383498    0.8960491
H     0.7445512    0.4367983   -1.5218583
H     1.8921463   -1.6649726   -2.1701843
H     3.8330227   -2.4811537   -0.8566666
H     4.6137632   -1.1850101    1.1092635
H     3.4598854    0.9030376    1.7569489
units angstrom
symmetry c1
no_reorient
no_com
"""

# Exact functional-group partition from tests/fsapt1/fsapt-ref/{fA,fB}.dat.
PHENOL_FA = "OH 1 2\nPH 3 4 5 6 7 8 9 10 11 12 13\n"
PHENOL_FB = "OH 14 15\nPH 16 17 18 19 20 21 22 23 24 25 26\n"
PHENOL_DIMENSIONS = {
    "atoms": 26, "ao": 312, "occupied": [25, 25], "frozen": [7, 7], "active": [18, 18],
}

PAIR_FIELDS = ("Disp20 occupied pairs", "Exch-Disp20 occupied pairs")
DISP_SCALARS = ("Disp20,u", "Exch-Disp20,u")


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def serialize_order2(order2):
    out = {}
    for term, rows in order2.items():
        out[term] = {
            fa: {fb: float(val) for fb, val in cols.items()}
            for fa, cols in rows.items()
        }
    return out


def timer_delta_optional(before, after, name):
    keys = [k for k in after if k == name or k.endswith(";" + name)]
    if not keys:
        return None
    seconds = sum(after[k]["wall_time"] - before.get(k, {}).get("wall_time", 0.) for k in keys)
    calls = sum(after[k]["n_calls"] - before.get(k, {}).get("n_calls", 0) for k in keys)
    if calls < 1:
        return None
    return float(seconds)


def extract_step_timers(before, after):
    names = {
        "scf_dimer": "SAPT(DFT):Dimer SCF",
        "scf_A": "SAPT(DFT):Monomer A SCF",
        "scf_B": "SAPT(DFT):Monomer B SCF",
        "canonical_prepare_JK": "SAPT(DFT):Build JK",
        "canonical_elst": "SAPT(DFT):elst",
        "canonical_exch": "SAPT(DFT):exch",
        "canonical_ind": "SAPT(DFT):ind",
        "canonical_disp": "SAPT(DFT):disp",
        "delta_hf_assembly": "SAPT(DFT):delta HF assembly",
        "gpu_fsapt_total": "SAPT(DFT): GPU F-SAPT",
        "fragment_elst": "SAPT(DFT): F-SAPT Electrostatics",
        "fragment_exch": "SAPT(DFT): F-SAPT Exchange",
        "fragment_ind": "SAPT(DFT): F-SAPT Induction",
        "fragment_disp": "SAPT(DFT): F-SAPT Dispersion",
    }
    return {k: timer_delta_optional(before, after, v) for k, v in names.items()}


def run_fsapt1(args, config, build):
    """Run S22 phenol dimer F-SAPT + fsapt.py functional-group analysis."""
    import psi4
    from psi4 import core
    from psi4.driver.procrouting.sapt import fsapt as fsapt_mod
    from gpu_runtime import AuditedRuntime
    import fragment_gate as fg

    if args.oracle_from:
        args.oracle_from = args.oracle_from.resolve(strict=True)
        if args.arm != "cpu":
            raise ValueError("Same-gauge oracle requires CPU arm")
    args.output.mkdir()
    os.chdir(args.output)
    result = {
        "status": "STARTED",
        "mode": "fsapt1",
        "system": "phenol_dimer_fsapt1",
        "arm": args.arm,
        "scaled": args.scaled,
        "basis_profile": "jun-cc-pvdz",
        "block_size": args.block_size,
        "factor_cache_bytes": args.factor_cache_bytes,
        "source_pin": config["source_pin"],
        "runtime_config_sha256": args.config_sha256,
        "pid": os.getpid(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": os.uname().nodename,
    }
    runtime = None
    captured, phases = {}, []
    try:
        core.set_output_file(str(args.output / "psi4.out"), False)
        psi4.set_memory(config["host_memory"])
        psi4.set_num_threads(config["threads"])
        selected = fg.options(
            args.arm == "gpu",
            args.scaled,
            "jun-cc-pvdz",
            scf_convergence=args.scf_convergence,
            scf_density_convergence=args.scf_density_convergence,
        )
        fsapt_dir = args.output / "fsapt"
        selected["fisapt_fsapt_filepath"] = str(fsapt_dir)
        psi4.set_options(selected)
        molecule = psi4.geometry(PHENOL_DIMER)
        result["options"] = selected
        result["geometry"] = PHENOL_DIMER
        result["geometry_sha256"] = hashlib.sha256(PHENOL_DIMER.encode()).hexdigest()

        terms = importlib.import_module("psi4.driver.procrouting.sapt.sapt_jk_terms_ein")
        from psi4.driver.procrouting.sapt.sapt_fisapt_workflow_cuda import HF_FIELDS

        def capture(cache):
            wfns = [cache["wfn_" + side] for side in "AB"]
            occupied = [int(w.nalpha()) for w in wfns]
            frozen = [
                int(w.basisset().n_frozen_core(core.get_global_option("FREEZE_CORE"), w.molecule()))
                for w in wfns
            ]
            dims = {
                "atoms": int(molecule.natom()),
                "ao": int(wfns[0].basisset().nbf()),
                "occupied": occupied,
                "frozen": frozen,
                "active": [n - nf for n, nf in zip(occupied, frozen)],
            }
            if dims != PHENOL_DIMENSIONS:
                raise RuntimeError(f"Unexpected fsapt1 dimensions: {dims} != {PHENOL_DIMENSIONS}")
            result["orbital_dimensions"] = dims
            for name in fg.FIELDS:
                key = {"Uocc0A": "Uocc_A", "Uocc0B": "Uocc_B"}.get(name, name)
                if name in cache or key in cache:
                    item = cache[name] if name in cache else cache[key]
                    captured[name] = np.array(getattr(item, "np", item), dtype=np.float64, copy=True)
            for name in DISP_SCALARS:
                if name in cache:
                    result[name] = float(cache[name])

        def forbidden(*_, **__):
            raise RuntimeError("Forbidden CPU fragment fallback in GPU fsapt1")

        raw = gpu = None
        iao_results = []
        deferred_raw = []
        with ExitStack() as stack:
            if args.oracle_from:
                prev = json.loads((args.oracle_from / "result.json").read_text())
                if prev["status"] != "PASS_FSAPT1_WORKER" or prev["arm"] != "gpu":
                    raise RuntimeError("Oracle requires completed GPU fsapt1 worker")
                for fname in ("raw-hf.npz", "fragment-matrices.npz"):
                    if sha256_file(args.oracle_from / fname) != prev["artifact_sha256"][fname]:
                        raise RuntimeError("GPU fsapt1 artifact changed: " + fname)
                raw = stack.enter_context(np.load(args.oracle_from / "raw-hf.npz", allow_pickle=False))
                gpu = stack.enter_context(np.load(args.oracle_from / "fragment-matrices.npz", allow_pickle=False))
                original_loc = core.IBOLocalizer2

                class CaptureIAO:
                    def __init__(self, *a, **kw):
                        self.native = original_loc(*a, **kw)

                    def __getattr__(self, name):
                        return getattr(self.native, name)

                    def localize(self, *a, **kw):
                        out = self.native.localize(*a, **kw)
                        iao_results.append(np.array(out["A"].np, copy=True))
                        return out

                stack.enter_context(patch.object(core, "IBOLocalizer2", CaptureIAO))

            if args.arm == "gpu":
                runtime = AuditedRuntime(args.config, args.config_sha256)
                from full_observe import observed_factory
                factory = observed_factory(runtime)
                original = terms.fisapt_cuda

                def fragment(cache, *positional, **kwargs):
                    deferred_raw.append({
                        name: cache[name]
                        for name in (*HF_FIELDS, "J_O", "eps_occ_A", "eps_occ_B", "eps_vir_A", "eps_vir_B")
                    })
                    with ExitStack() as guard:
                        for name in ("DFHelper", "FISAPT", "MintsHelper", "IBOLocalizer2", "ExternalPotential"):
                            guard.enter_context(patch.object(core, name, forbidden))
                        for name in ("localization", "flocalization", "felst", "fexch", "find", "fdisp0"):
                            guard.enter_context(patch.object(terms, name, forbidden))
                        if kwargs.get("factor_cache_bytes") != args.factor_cache_bytes:
                            raise RuntimeError("Driver did not forward factor_cache_bytes")
                        t0 = time.perf_counter()
                        out, data = original(cache, *positional, **kwargs)
                        phases.append({"name": "GPU full fragment continuation", "seconds": time.perf_counter() - t0})
                    capture(out)
                    result["Disp20,u"] = float(data["Disp20,u"])
                    result["Exch-Disp20,u"] = float(data["Exch-Disp20,u"])
                    result["fragment_execution"] = data["FISAPT execution"]
                    return out, data

                stack.enter_context(patch.object(terms, "fisapt_cuda", fragment))
            else:
                store_comp = terms._store_dispersion0_components

                def components(cache, disp, exch, *positional, **kwargs):
                    out = store_comp(cache, disp, exch, *positional, **kwargs)
                    cache["Disp20 occupied pairs"] = np.array(disp, copy=True)
                    cache["Exch-Disp20 occupied pairs"] = np.array(exch, copy=True)
                    return out

                stack.enter_context(patch.object(terms, "_store_dispersion0_components", components))
                for name in ("localization", "partition", "flocalization", "felst", "fexch", "find", "fdisp0"):
                    orig_fn = getattr(terms, name)

                    def timed(*positional, _name=name, _orig=orig_fn, **kwargs):
                        t0 = time.perf_counter()
                        if _name == "localization":
                            deferred_raw.append({
                                k: positional[0][k]
                                for k in (*HF_FIELDS, "J_O", "eps_occ_A", "eps_occ_B", "eps_vir_A", "eps_vir_B")
                            })
                        if raw is not None and _name == "localization":
                            fg.replay_hf(core, positional[0], positional[1], raw)
                        out = _orig(*positional, **kwargs)
                        if gpu is not None and _name == "flocalization":
                            result["independent_ibo"] = fg.replay_localization(
                                core, positional[0], positional[1], gpu, iao_results
                            )
                        phases.append({"name": _name, "seconds": time.perf_counter() - t0})
                        if _name == "fdisp0":
                            capture(out)
                        return out

                    stack.enter_context(patch.object(terms, name, timed))

            before_timers = dict(core.get_timer_records(False))
            t_start = time.perf_counter()
            invoke = runtime.guarded_call if runtime else (lambda fn, *a, **kw: fn(*a, **kw))
            energy, wfn = invoke(
                psi4.energy,
                "sapt(dft)",
                molecule=molecule,
                return_wfn=True,
                **(
                    {
                        "cuda_provider_factory": factory,
                        "cuda_dispersion_block_size": args.block_size,
                        "cuda_dispersion_factor_cache_bytes": args.factor_cache_bytes,
                    }
                    if runtime
                    else {}
                ),
            )
            elapsed = time.perf_counter() - t_start
            after_timers = dict(core.get_timer_records(False))

        result["benchmark_whole_energy_seconds"] = elapsed
        result["step_timers"] = extract_step_timers(before_timers, after_timers)
        result["phases"] = phases
        result["energy_Eh"] = float(energy)
        result["variables"] = {k: float(v) for k, v in core.scalar_variables().items()}

        np.savez(
            args.output / "raw-hf.npz",
            **{k: np.array(v.np, copy=True) for k, v in deferred_raw[0].items()},
        )
        np.savez(args.output / "fragment-matrices.npz", **captured)
        missing = set(fg.FIELDS) - captured.keys()
        if missing:
            raise RuntimeError("Missing F-SAPT fields: " + repr(sorted(missing)))
        for name, key in (("Disp20 occupied pairs", "Disp20,u"), ("Exch-Disp20 occupied pairs", "Exch-Disp20,u")):
            np.testing.assert_allclose(np.sum(captured[name]), result[key], rtol=1e-11, atol=1e-12)

        if args.oracle_from:
            with np.load(args.oracle_from / "fragment-matrices.npz", allow_pickle=False) as target:
                errors = {}
                for name in fg.FIELDS:
                    np.testing.assert_allclose(
                        captured[name], target[name], rtol=2e-5, atol=2e-7,
                        err_msg="Same-gauge fsapt1 " + name,
                    )
                    diff = captured[name] - target[name]
                    errors[name] = {
                        "maximum_absolute_difference": float(np.max(np.abs(diff))),
                        "signed_sum_difference": float(np.sum(diff)),
                    }
                result["same_gauge_errors"] = errors

        result["signed_matrix_sums"] = {
            k: float(np.sum(v)) for k, v in captured.items() if k.endswith(("_AB", "_BA"))
        }

        # Run real fsapt.py functional-group partition (OH / PH covalent split).
        (fsapt_dir / "fA.dat").write_text(PHENOL_FA)
        (fsapt_dir / "fB.dat").write_text(PHENOL_FB)
        charge_stuff = fsapt_mod.compute_fsapt(str(fsapt_dir), False, print_output=False)
        half_stuff = fsapt_mod.compute_fsapt(str(fsapt_dir), True, print_output=False)
        fsapt_mod.run_from_output(str(fsapt_dir), return_data="reduced_analysis")
        result["fsapt_analysis_kcal_mol"] = {
            "charge": {
                "full": serialize_order2(charge_stuff["order2"]),
                "reduced": serialize_order2(charge_stuff["order2r"]),
            },
            "links5050": {
                "full": serialize_order2(half_stuff["order2"]),
                "reduced": serialize_order2(half_stuff["order2r"]),
            },
        }
        result["fsapt_dat_sha256"] = sha256_file(fsapt_dir / "fsapt.dat")
        result["artifact_sha256"] = {p.name: sha256_file(p) for p in args.output.glob("*.npz")}
        result["max_rss_kib"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        if runtime:
            runtime.verify_optional_modules()
            psi_out = (args.output / "psi4.out").read_text()
            if "cuESTJK: GPU-Accelerated Density-Fitted" not in psi_out:
                raise RuntimeError("Missing cuESTJK banner in fsapt1 GPU run")
            if "MemDFJK" in psi_out or "DiskDFJK" in psi_out:
                raise RuntimeError("CPU DFJK fallback found in fsapt1 GPU run")
            for plan in runtime.report["plans"]:
                if (
                    not plan["session_cleanup"]["complete"]
                    or plan["session_cleanup"]["live_bytes"]
                    or plan["session_cleanup"]["native_owners"]
                ):
                    raise RuntimeError("Incomplete GPU cleanup in fsapt1")
            result["runtime"] = runtime.report

        result["status"] = "PASS_FSAPT1_WORKER"
    except BaseException as error:
        result["status"] = "FAILED"
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        core.close_outfile()
        save_json(args.output / "result.json", result)


def run_fixed_fdisp0(args, config, build):
    """Benchmark optimized GPU fdisp0 on a frozen SHA-verified Protein83 snapshot."""
    import psi4
    from psi4 import core
    from gpu_runtime import AuditedRuntime
    from full_inputs import load_point
    from psi4.driver.procrouting.sapt.sapt_fisapt_workflow_cuda import HF_FIELDS
    from psi4.driver.procrouting.sapt.sapt_dispersion_cuda_setup import prepare_hf
    from psi4.driver.procrouting.sapt.sapt_dispersion_cuda import blocked_device_dispersion

    args.output.mkdir()
    os.chdir(args.output)
    ref_path = args.reference_result.resolve(strict=True)
    if sha256_file(ref_path) != args.reference_result_sha256:
        raise RuntimeError("Frozen reference result.json SHA256 mismatch")
    doc = json.loads(ref_path.read_text())
    ref_dir = ref_path.parent
    loaded = {}
    for name in ("raw-hf.npz", "fragment-matrices.npz"):
        p = ref_dir / name
        if sha256_file(p) != doc["artifact_sha256"][name]:
            raise RuntimeError("Snapshot artifact changed: " + name)
        with np.load(p, allow_pickle=False) as arrays:
            loaded[name] = {k: np.array(arrays[k], copy=True) for k in arrays.files}
    raw, fields = loaded["raw-hf.npz"], loaded["fragment-matrices.npz"]
    dims = doc["orbital_dimensions"]
    point, expected = load_point(args.inputs, args.inputs_sha256, args.system)
    if expected != dims or point["geometry_sha256"] != doc["geometry_sha256"]:
        raise RuntimeError("Frozen input manifest differs from snapshot")

    caps = [int(x.strip()) for x in args.cache_caps.split(",") if x.strip()]
    if not caps or any(c < 0 for c in caps):
        raise ValueError("Nonnegative integer cache caps required")

    ref_fdisp_sec = None
    for phase in doc.get("phases", []):
        if phase.get("name") == "fdisp0":
            ref_fdisp_sec = float(phase["seconds"])

    report = {
        "status": "STARTED",
        "mode": "fixed-fdisp0",
        "system": args.system,
        "basis_profile": doc.get("basis_profile"),
        "reference_arm": doc.get("arm"),
        "reference_result": str(ref_path),
        "reference_result_sha256": args.reference_result_sha256,
        "reference_fdisp0_seconds": ref_fdisp_sec,
        "source_pin": config["source_pin"],
        "runtime_config_sha256": args.config_sha256,
        "fixed_dimensions": dims,
        "block_size": args.block_size,
        "cache_caps": caps,
        "runs": [],
    }
    runtime = None
    try:
        core.set_output_file(str(args.output / "psi4.out"), False)
        psi4.set_memory(config["host_memory"])
        psi4.set_num_threads(config["threads"])
        options = dict(doc["options"])
        options["use_cuest"] = True
        options["fisapt_fsapt_filepath"] = str(args.output / "unused-fsapt")
        psi4.set_options(options)
        molecule = psi4.geometry(doc["geometry"])
        molecule.update_geometry()
        primary = core.BasisSet.build(molecule, "BASIS", options["basis"], puream=1)
        rifit = core.BasisSet.build(
            molecule, "DF_BASIS_MP2", options["df_basis_mp2"], "RIFIT", options["basis"], puream=1
        )
        if (primary.nbf(), rifit.nbf()) != (dims["ao"], point["spherical_basis_counts"]["rifit"]):
            raise RuntimeError("Fixed AO/RIFIT dimensions mismatch")
        report["rifit_nbf"] = int(rifit.nbf())

        matrices = {k: np.ascontiguousarray(raw[k]) for k in HF_FIELDS}
        energies = {
            k: np.ascontiguousarray(raw[name])
            for k, name in zip("abrs", ("eps_occ_A", "eps_occ_B", "eps_vir_A", "eps_vir_B"))
        }
        rotations = []
        for side, count, nf in zip("AB", dims["occupied"], dims["frozen"]):
            u, co, locc = fields["Uocc0" + side], raw["Cocc_" + side], fields["Locc_" + side]
            np.testing.assert_allclose(u.T @ u, np.eye(count), rtol=1e-8, atol=1e-8)
            np.testing.assert_allclose(co @ u, locc, rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(u[:nf, nf:], 0., rtol=0, atol=1e-8)
            np.testing.assert_allclose(u[nf:, :nf], 0., rtol=0, atol=1e-8)
            rotations.append(np.ascontiguousarray(u[nf:, nf:]))

        ref_pairs = {
            "Disp20 occupied pairs": fields["Disp20 occupied pairs"],
            "Exch-Disp20 occupied pairs": fields["Exch-Disp20 occupied pairs"],
            "Disp20,u": float(doc["Disp20,u"]),
            "Exch-Disp20,u": float(doc["Exch-Disp20,u"]),
        }

        runtime = AuditedRuntime(args.config, args.config_sha256)
        first_out = None
        t_provider = time.perf_counter()
        with runtime.provider(primary, rifit) as provider, provider:
            provider.prepare_device_compute()
            report["provider_setup_seconds"] = time.perf_counter() - t_provider
            session = provider.session
            report["device"] = runtime.device_identity(session)
            native = provider.report()
            report["native_policy"] = {
                k: native[k] for k in ("producer", "native_version", "declared_plan_policy", "native_math_mode")
            }
            with session.scope():
                spaces, corrections, eps = prepare_hf(
                    session, matrices, energies, frozen_a=dims["frozen"][0], frozen_b=dims["frozen"][1]
                )
                dr = tuple(session.upload(v) for v in rotations)
                for idx, cap in enumerate(caps):
                    calls_before = provider.calls
                    prod_before = provider.elapsed
                    live_before = session.live_bytes
                    session.synchronize_device()
                    t0 = time.perf_counter()
                    orig_upload = session.upload

                    def no_upload(*_, **__):
                        raise RuntimeError("Dispersion path attempted host numerical upload")

                    session.upload = no_upload
                    try:
                        out = blocked_device_dispersion(
                            provider, spaces, corrections, eps,
                            block_size=args.block_size, rotations=dr, factor_cache_bytes=cap,
                        )
                        session.synchronize_device()
                    finally:
                        session.upload = orig_upload
                    elapsed = time.perf_counter() - t0
                    if session.live_bytes != live_before:
                        raise RuntimeError("Device memory leaked across fixed-fdisp0 call")

                    # Compare against frozen snapshot reference (CPU48 or historical GPU).
                    ref_errors = {}
                    for k in (*PAIR_FIELDS, *DISP_SCALARS):
                        a, b = np.asarray(ref_pairs[k]), np.asarray(out[k])
                        np.testing.assert_allclose(b, a, rtol=1e-10, atol=1e-11, err_msg="Snapshot parity " + k)
                        ref_errors[k] = float(np.max(np.abs(b - a)))

                    repeat_errors = {}
                    if first_out is None:
                        first_out = out
                    else:
                        for k in (*PAIR_FIELDS, *DISP_SCALARS):
                            a, b = np.asarray(first_out[k]), np.asarray(out[k])
                            np.testing.assert_allclose(b, a, rtol=1e-12, atol=1e-12, err_msg="Repeat parity " + k)
                            repeat_errors[k] = float(np.max(np.abs(b - a)))

                    exec_meta = dict(out["Disp0 execution"])
                    exec_meta.pop("producer", None)
                    row = {
                        "index": idx,
                        "cache_cap_bytes": cap,
                        "wall_seconds": elapsed,
                        "producer_wall_seconds": provider.elapsed - prod_before,
                        "dfmo_calls": provider.calls - calls_before,
                        "signed_components": {k: float(out[k]) for k in DISP_SCALARS},
                        "max_abs_err_vs_snapshot": ref_errors,
                        "max_abs_err_vs_first": repeat_errors,
                        "disp0_execution": exec_meta,
                        "peak_live_bytes": session.peak_live_bytes,
                        "native_persistent_bytes": session.native_bytes,
                    }
                    if ref_fdisp_sec is not None and elapsed > 0:
                        row["speedup_vs_snapshot_fdisp0"] = ref_fdisp_sec / elapsed
                    report["runs"].append(row)

        runtime.verify_optional_modules()
        cleanup = runtime.report["plans"][-1]["session_cleanup"]
        report["provider_cleanup"] = cleanup
        if not cleanup["complete"] or cleanup["live_bytes"] or cleanup["native_owners"]:
            raise RuntimeError("Incomplete GPU cleanup in fixed-fdisp0")
        report["status"] = "PASS_FIXED_FDISP0"
    except BaseException as error:
        report["status"] = "FAILED"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        core.close_outfile()
        save_json(args.output / "result.json", report)


def run_custom_fsapt(args, config, build):
    """GPU-only bounded F-SAPT probe on a frozen case JSON (e.g. protein157)."""
    import psi4
    from psi4 import core
    from gpu_runtime import AuditedRuntime
    from full_observe import observed_factory
    import fragment_gate as fg

    args.output.mkdir()
    os.chdir(args.output)
    case_path = args.case.resolve(strict=True)
    if sha256_file(case_path) != args.case_sha256:
        raise RuntimeError("Frozen case JSON changed")
    case = json.loads(case_path.read_text())
    if hashlib.sha256(case["geometry"].encode()).hexdigest() != case["geometry_sha256"]:
        raise RuntimeError("Frozen case geometry SHA256 mismatch")

    report = {
        "status": "STARTED",
        "mode": "custom-fsapt",
        "system": case["system"],
        "arm": "gpu",
        "basis_profile": "jun-cc-pvdz",
        "block_size": args.block_size,
        "factor_cache_bytes": args.factor_cache_bytes,
        "source_pin": config["source_pin"],
        "runtime_config_sha256": args.config_sha256,
        "case_sha256": args.case_sha256,
    }
    runtime = None
    try:
        core.set_output_file(str(args.output / "psi4.out"), False)
        psi4.set_memory(config["host_memory"])
        psi4.set_num_threads(config["threads"])
        selected = fg.options(
            True, True, "jun-cc-pvdz",
            scf_convergence=args.scf_convergence,
            scf_density_convergence=args.scf_density_convergence,
        )
        selected["fisapt_fsapt_filepath"] = str(args.output / "fsapt")
        psi4.set_options(selected)
        molecule = psi4.geometry(case["geometry"])
        runtime = AuditedRuntime(args.config, args.config_sha256)
        factory = observed_factory(runtime)
        terms = importlib.import_module("psi4.driver.procrouting.sapt.sapt_jk_terms_ein")
        original = terms.fisapt_cuda

        def capture_exec(cache, *pos, **kw):
            out, data = original(cache, *pos, **kw)
            report["Disp20,u"] = float(data["Disp20,u"])
            report["Exch-Disp20,u"] = float(data["Exch-Disp20,u"])
            report["fragment_execution"] = data["FISAPT execution"]
            wfns = [out["wfn_" + s] for s in "AB"]
            occ = [int(w.nalpha()) for w in wfns]
            frz = [
                int(w.basisset().n_frozen_core(core.get_global_option("FREEZE_CORE"), w.molecule()))
                for w in wfns
            ]
            report["orbital_dimensions"] = {
                "atoms": int(molecule.natom()),
                "ao": int(wfns[0].basisset().nbf()),
                "occupied": occ,
                "frozen": frz,
                "active": [o - f for o, f in zip(occ, frz)],
            }
            return out, data

        before = dict(core.get_timer_records(False))
        t0 = time.perf_counter()
        with patch.object(terms, "fisapt_cuda", capture_exec):
            energy, _ = runtime.guarded_call(
                psi4.energy,
                "sapt(dft)",
                molecule=molecule,
                return_wfn=True,
                cuda_provider_factory=factory,
                cuda_dispersion_block_size=args.block_size,
                cuda_dispersion_factor_cache_bytes=args.factor_cache_bytes,
            )
        elapsed = time.perf_counter() - t0
        after = dict(core.get_timer_records(False))
        runtime.verify_optional_modules()
        report["benchmark_whole_energy_seconds"] = elapsed
        report["step_timers"] = extract_step_timers(before, after)
        report["energy_Eh"] = float(energy)
        report["max_rss_kib"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        report["runtime"] = runtime.report
        report["status"] = "PASS_CUSTOM_FSAPT_GPU"
    except BaseException as error:
        report["status"] = "FAILED"
        report["error"] = f"{type(error).__name__}: {error}"
        report["max_rss_kib"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if runtime is not None:
            report["runtime"] = runtime.report
        raise
    finally:
        core.close_outfile()
        save_json(args.output / "result.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--mode", choices=("fsapt1", "fixed-fdisp0", "custom-fsapt"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--scaled", action="store_true", default=True)
    parser.add_argument("--oracle-from", type=Path)
    parser.add_argument("--block-size", type=int, default=24)
    parser.add_argument("--factor-cache-bytes", type=int, default=0)
    parser.add_argument("--scf-convergence", type=int, choices=(10, 12), default=10)
    parser.add_argument("--scf-density-convergence", type=int, choices=(10, 12), default=12)
    parser.add_argument("--reference-result", type=Path)
    parser.add_argument("--reference-result-sha256")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--inputs-sha256")
    parser.add_argument("--system")
    parser.add_argument("--cache-caps", default="0,4294967296,0")
    parser.add_argument("--case", type=Path)
    parser.add_argument("--case-sha256")
    args = parser.parse_args()

    bootstrap_path = args.bootstrap.resolve(strict=True)
    sys.path.insert(0, str(bootstrap_path.parent))
    from gpu_runtime import read_config, digest
    import sealed_bootstrap

    config, build = read_config(args.config, args.config_sha256)
    if build["provenance_sha256"].get(str(bootstrap_path)) != digest(bootstrap_path):
        raise RuntimeError("Authenticated original sealed_bootstrap.py required")
    _owner = sealed_bootstrap.bootstrap(config, build)  # Retain through scientific cleanup.

    if args.mode == "fsapt1":
        run_fsapt1(args, config, build)
    elif args.mode == "fixed-fdisp0":
        run_fixed_fdisp0(args, config, build)
    else:
        run_custom_fsapt(args, config, build)


if __name__ == "__main__":
    main()
