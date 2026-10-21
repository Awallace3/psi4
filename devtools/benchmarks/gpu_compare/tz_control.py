#!/usr/bin/env python3
"""Augmentation control on one frozen, exported cuEST quadrature per case.

Run inside a GPU allocation. This is not the smaller-native-default experiment.
Audit failures gate timing runs; matching counts alone is never sufficient.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import itertools
import math

from compare import (HERE, ENGINES, atomic_json, digest, engine_env,
                     expected_nbf, observed_worker, summarize, xyz)
from grids import compare_exports
from protocol import compare_overlap_exports
from direct import is_direct, jk_mode


def strategy_modes(config, requested=None):
    if is_direct(config):
        if requested:
            raise ValueError("--df-modes cannot be combined with conventional-direct")
        return [None]
    return requested or ["native-strategy", "stored-df"]


def matching_audit(source, candidate):
    proof = compare_exports(source, candidate, allow_basis_difference=True)
    if not proof["identical_nonzero_quadrature"]:
        raise RuntimeError("Actual coordinates/total weights differ; timing comparison refused")
    return proof


def validate_source_record(record):
    if not record.get("ok") or record.get("process_returncode") != 0:
        raise RuntimeError("cuEST native source-grid SCF failed")
    info = record.get("grid_exports", {}).get("live-plan-before-scf", {})
    if not info.get("stored_points") or not info.get("nonzero_weight_points"):
        raise RuntimeError("Actual cuEST source-grid export is missing")
    if info["stored_points"] != record.get("grid_points"):
        raise RuntimeError("Actual source-grid count disagrees with the SCF record")
    if info.get("identity", {}).get("grid_mode") != "native-default":
        raise RuntimeError("Source quadrature is not a cuEST native-default export")
    return info


def timing_gate(proofs):
    """Require both bases and both engines; a partial audit cannot pass."""
    return (set(proofs) == {"cc-pvtz", "aug-cc-pvtz"} and
            all(set(engines) == set(ENGINES) and
                all(proof.get("verified") and proof.get("identical_nonzero_quadrature")
                    and proof.get("overlap_rank_spectrum_verified")
                    for proof in engines.values())
                for engines in proofs.values()))


def energy_audit(records, config):
    """Require a successful finite energy pair before spending timed repeats."""
    pair = {r["engine"]: r for r in records}
    if set(pair) != set(ENGINES) or not all(r.get("ok") for r in pair.values()):
        return dict(verified=False, error="Both diagnostic SCFs must succeed")
    energies = [pair[e].get("energy_hartree") for e in ENGINES]
    if not all(isinstance(e, (int, float)) and math.isfinite(e) for e in energies):
        return dict(verified=False, error="Missing/nonfinite diagnostic energy")
    delta = energies[1] - energies[0]
    threshold = config["energy_review_threshold_eh"]
    return dict(verified=abs(delta) <= threshold, delta_pyscf_minus_psi4_eh=delta,
                threshold_eh=threshold)


def worker_cleanup(base, label, uuid):
    """Subreaper catches detached descendants; NVML checks selected GPU only."""
    from containment import tree
    usage = json.loads((base / f"{label}.usage.json").read_text())
    descendants = sorted(tree())
    text = subprocess.check_output(
        ["nvidia-smi", "-i", uuid, "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
         "--format=csv,noheader,nounits"], text=True, timeout=10)
    with (base / f"{label}.gpu-after.csv").open("x") as stream:
        stream.write(text)
    return dict(verified=(usage.get("state") == "EXITED" and
                          usage.get("child_returncode") == 0 and
                          not descendants and not text.strip()),
                descendants=descendants, selected_gpu_empty=not text.strip(),
                supervisor_state=usage.get("state"),
                worker_returncode=usage.get("child_returncode"))


def require_worker_success(record, strict, label):
    if strict and (not record.get("ok") or
                   record.get("worker_cleanup", {}).get("verified") is not True):
        raise RuntimeError(f"Worker {label} failed/cleanup unknown; subsequent workers withheld")


def shared_config(config, basis, source, info):
    result = dict(config, basis=basis, grid_mode="shared-cuest",
                  shared_grid=str(source.resolve()),
                  shared_grid_file_sha256=digest(source),
                  shared_grid_metadata_sha256=digest(source.with_suffix(".json")),
                  shared_grid_source_points=info["stored_points"],
                  shared_grid_fingerprint=info["nonzero_pairs_multiset_sha256"],
                  shared_grid_nonzero_weight_points=info["nonzero_weight_points"])
    return result


def attempt(a, base, config_path, engine, system, functional, label, profile, repeat, gpu):
    output = base / "results" / label
    python = (a.psi_env if engine == "psi4" else a.gpu_env) / "bin/python"
    cmd = [str(python), str(HERE / "compare.py"), "worker", "--engine", engine,
           "--system", system, "--functional", functional, "--repeat", str(repeat),
           "--config", str(config_path), "--xyz", str(a.output / "inputs/geometries" / f"{system}.xyz"),
           "--output", str(output), "--psi-package", str(a.psi_package), "--gpu-env", str(a.gpu_env)]
    if profile:
        cmd += ["--profile", "--export-grid"]
    config = json.loads(config_path.read_text())
    expected = dict(engine=engine, system=system, functional=functional, repeat=repeat, profile=profile,
                    jk_mode=jk_mode(config),
                    df_mode=None if is_direct(config) else config.get("df_mode", "native-strategy"),
                    config_sha256=a._config_pins[config_path],
                    geometry_sha256=a._geometry_pins[system])
    if digest(config_path) != expected["config_sha256"]:
        raise RuntimeError("Configuration changed after creation")
    audit = config.get("audited_overlap")
    if not profile and audit:
        if not all(digest(Path(path)) == sha for path, sha in audit["references"].items()):
            raise RuntimeError("Diagnostic audit artifact changed before timing")
    record, interrupted = observed_worker(cmd, engine_env(engine, a), base, label, expected,
        gpu["uuid"], config["worker_timeout_s"], a.gpu_env / "bin/python")
    strict = getattr(a, "stop_on_worker_failure", False)
    if strict:
        try:
            record["worker_cleanup"] = worker_cleanup(base, label, gpu["uuid"])
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            record["worker_cleanup"] = dict(verified=False, error=str(exc))
    if record.get("ok") and record.get("gpu", {}).get("uuid") != gpu["uuid"]:
        record.update(ok=False, error="Different GPU selected")
    if record.get("ok") and not profile and audit and record.get("nmo") != audit["nmo"]:
        record.update(ok=False, error="Timed retained orbital dimension differs from audited rank")
    atomic_json(base / f"{label}.record.json", record)
    if interrupted:
        raise interrupted
    print(label, "OK" if record["ok"] else "FAILED",
          "actual grid points:", record.get("grid_points"),
          "nonzero:", record.get("grid_nonzero_weight_points"),
          record.get("error", ""), flush=True)
    require_worker_success(record, strict, label)
    return record, output


def control(a):
    if not os.getenv("SLURM_JOB_ID"):
        raise RuntimeError("Use an allocated compute-node GPU, not a login node")
    a.output.mkdir(parents=True, exist_ok=False)
    a._output_created = True
    if getattr(a, "stop_on_worker_failure", False):
        from containment import subreaper
        subreaper()
    (a.output / "inputs/geometries").mkdir(parents=True)
    (a.output / "harness").mkdir()
    for path in HERE.iterdir():
        if path.is_file():
            shutil.copy2(path, a.output / "harness" / path.name)
    config = json.loads(a.config.read_text())
    config["jk_mode"] = getattr(a, "jk_mode", None) or config.get("jk_mode", "density-fitted")
    if is_direct(config):
        config.pop("df_mode", None)
    modes = strategy_modes(config, a.df_modes)
    a._config_pins = {}
    a._geometry_pins = {}
    config.update(grid_mode="native-default", threads=a.threads, repeats=a.repeats)
    config.update(overlap_eigenvalue_cutoff=1e-7, headline_timing="full_process_wall_s",
                  require_memory_observations=True,
                  geometry_angstrom_per_bohr=0.52917721067)
    if a.preflight:
        config["functionals"] = {name: config["functionals"][name] for name in ("pbe", "wb97x")}
    for system in a.systems:
        source = a.data / "geometries" / f"{system}.xyz"
        xyz(source, system)
        shutil.copy2(source, a.output / "inputs/geometries" / source.name)
        a._geometry_pins[system] = digest(a.output / "inputs/geometries" / source.name)
        for basis in ("cc-pvtz", "aug-cc-pvtz"):
            expected_nbf(dict(config, basis=basis), system)
    gpu = json.loads(subprocess.check_output(
        [str(a.gpu_env / "bin/python"), str(HERE / "compare.py"), "device"],
        env=engine_env("pyscf", a), text=True))
    core = next(a.psi_package.glob("core*.so"))
    pins = {core: digest(core)}
    pins.update({path: digest(path) for path in a.psi_package.rglob("*.py")})
    basis_root = a.psi_package.parents[1] / "share/psi4/basis"
    if not basis_root.is_dir():
        raise RuntimeError("Staged Psi4 basis-data directory is missing")
    pins.update({path: digest(path) for path in basis_root.rglob("*") if path.is_file()})
    for package in ("gpu4pyscf", "pyscf", "cupy", "cupy_backends"):
        for root in a.gpu_env.glob(f"lib/python*/site-packages/{package}"):
            pins.update({path: digest(path) for path in root.rglob("*")
                         if path.is_file() and path.suffix in (".py", ".so")})
    pins.update({path: digest(path) for path in HERE.iterdir() if path.is_file()})
    linkage = subprocess.check_output(["ldd", str(core)], env=engine_env("psi4", a), text=True)
    if "not found" in linkage:
        raise RuntimeError(linkage)
    for line in linkage.splitlines():
        words = line.split()
        if "=>" in words and any(token in words[0].lower() for token in ("libcuest", "libxc")):
            path = Path(words[words.index("=>") + 1]).resolve()
            pins[path] = digest(path)
    for engine in ENGINES:
        python = (a.psi_env if engine == "psi4" else a.gpu_env) / "bin/python"
        lock = subprocess.check_output([str(python), "-m", "pip", "freeze", "--all"],
                                       env=engine_env(engine, a), text=True)
        (a.output / f"{engine}-pip-freeze.txt").write_text(lock)
    atomic_json(a.output / "manifest.json", dict(
        config=config, systems=a.systems, gpu=gpu, job_id=os.environ["SLURM_JOB_ID"],
        jk_mode=jk_mode(config), df_modes=[] if is_direct(config) else modes,
        command=sys.argv, linkage=linkage, pins={str(path): sha for path, sha in pins.items()},
        geometry_sha256={path.name: digest(path) for path in (a.output / "inputs/geometries").glob("*.xyz")},
        grid_protocol="shared actual cuEST native-default mesh, not necessarily smaller native default",
        account=os.getenv("SLURM_JOB_ACCOUNT"), qos=os.getenv("SLURM_JOB_QOS"),
        partition=os.getenv("SLURM_JOB_PARTITION"),
        timing_note="Headline is fresh worker launch-to-reap, including interpreter startup, imports, grid and DF construction, SCF, validation and teardown. Internal clocks are diagnostics only." ))
    results = []
    for system in a.systems:
        for functional, df_mode in itertools.product(config["functionals"], modes):
            case = a.output / system / functional / (df_mode or "conventional-direct")
            source_base = case / "native-source"
            source_base.mkdir(parents=True)
            native = dict(config, basis="cc-pvtz", df_mode=df_mode,
                          functionals={functional: config["functionals"][functional]})
            native_path = source_base / "config.json"
            atomic_json(native_path, native)
            a._config_pins[native_path] = digest(native_path)
            source_record, source_output = attempt(a, source_base, native_path, "psi4", system,
                                                   functional, "source", True, 0, gpu)
            item = dict(system=system, functional=functional, df_mode=df_mode, jk_mode=jk_mode(config),
                        headline_timing="full_process_wall_s", complete=False, basis_runs={})
            try:
                source_info = validate_source_record(source_record)
            except RuntimeError as exc:
                item["error"] = str(exc)
                results.append(item)
                atomic_json(a.output / "control-results.json", results)
                continue
            source_mesh = source_output / "xc-grid.npz"
            pins[source_mesh] = digest(source_mesh)
            pins[source_mesh.with_suffix(".json")] = digest(source_mesh.with_suffix(".json"))
            item["source_grid"] = source_info
            native_proofs = {}
            for basis in ("cc-pvtz", "aug-cc-pvtz"):
                if basis == "cc-pvtz":
                    native_psi, psi_output = source_record, source_output
                    path = native_path
                else:
                    path = source_base / "config-native-aug.json"
                    atomic_json(path, dict(native, basis=basis))
                    a._config_pins[path] = digest(path)
                    native_psi, psi_output = attempt(a, source_base, path, "psi4", system,
                        functional, "psi4-native-aug", True, 0, gpu)
                native_py, py_output = attempt(a, source_base, path, "pyscf", system,
                    functional, f"pyscf-native-{basis}", True, 0, gpu)
                proof = dict(verified=False, error=native_psi.get("error") or native_py.get("error"))
                if native_psi.get("ok") and native_py.get("ok"):
                    try:
                        proof = dict(verified=True, **compare_exports(
                            psi_output / "xc-grid.npz", py_output / "xc-entry-grid.npz"))
                    except (OSError, ValueError, RuntimeError) as exc:
                        proof["error"] = str(exc)
                native_proofs[basis] = proof
            item["native_default_grid_comparison"] = native_proofs
            atomic_json(source_base / "native-default-grid-comparison.json", native_proofs)
            all_proofs = {}
            energy_proofs = {}
            cohorts = []
            for basis in ("cc-pvtz", "aug-cc-pvtz"):
                base = case / basis / "campaign"
                base.mkdir(parents=True)
                shared = shared_config(native, basis, source_mesh, source_info)
                config_path = base / "config.json"
                atomic_json(config_path, shared)
                a._config_pins[config_path] = digest(config_path)
                records, proofs = [], {}
                for engine in ENGINES:
                    record, output = attempt(a, base, config_path, engine, system, functional,
                                              f"{engine}-audit", True, 0, gpu)
                    records.append(record)
                    if not record.get("ok"):
                        proofs[engine] = dict(verified=False, error=record.get("error"))
                    else:
                        candidate = output / ("xc-grid.npz" if engine == "psi4" else "xc-entry-grid.npz")
                        try:
                            proofs[engine] = dict(verified=True, **matching_audit(source_mesh, candidate))
                        except (OSError, ValueError, RuntimeError) as exc:
                            proofs[engine] = dict(verified=False, error=str(exc))
                overlap_proof = dict(verified=False, error="Both audit SCFs must succeed")
                if all(record.get("ok") for record in records):
                    try:
                        overlap_proof = compare_overlap_exports(
                            base / "results/psi4-audit/overlap-spectrum.npz",
                            base / "results/pyscf-audit/overlap-spectrum.npz")
                    except (OSError, ValueError, RuntimeError) as exc:
                        overlap_proof = dict(verified=False, error=str(exc))
                for proof in proofs.values():
                    proof["overlap_rank_spectrum_verified"] = overlap_proof["verified"]
                atomic_json(base / "overlap-proof.json", overlap_proof)
                energy_proofs[basis] = energy_audit(records, shared)
                atomic_json(base / "energy-proof.json", energy_proofs[basis])
                if overlap_proof["verified"]:
                    audit_paths = [base / "overlap-proof.json", base / "grid-proof.json"]
                    audit_paths += [base / "results" / f"{engine}-audit" / f"overlap-spectrum.{ext}"
                                    for engine in ENGINES for ext in ("npz", "json")]
                all_proofs[basis] = proofs
                atomic_json(base / "grid-proof.json", proofs)
                atomic_json(base / "records.json", records)
                if overlap_proof["verified"]:
                    pins.update({path: digest(path) for path in audit_paths})
                    # Freeze a separate timed config; the diagnostic config and
                    # its spectrum metadata must remain immutable.
                    shared = dict(shared, audited_overlap=dict(
                        nmo=overlap_proof["nmo"], nbf=overlap_proof["nbf"],
                        cutoff=overlap_proof["cutoff"],
                        references={str(path): pins[path] for path in audit_paths}))
                    config_path = base / "config-timed.json"
                    atomic_json(config_path, shared)
                    a._config_pins[config_path] = digest(config_path)
                cohorts.append((basis, base, config_path, shared, records))
            # Four successful actual-mesh audits required BEFORE spending timing repeats.
            audited = timing_gate(all_proofs) and all(p["verified"] for p in energy_proofs.values())
            if audited:
                for basis, base, config_path, shared, records in cohorts:
                    failed = False
                    for repeat in range(a.repeats):
                        for engine in ENGINES[::1 if repeat % 2 == 0 else -1]:
                            record, _ = attempt(a, base, config_path, engine, system, functional,
                                                  f"{engine}-{repeat}", False, repeat, gpu)
                            records.append(record)
                            atomic_json(base / "records.json", records)
                            if not record.get("ok"):
                                failed = True
                                break
                        if failed:
                            break
            for basis, base, config_path, shared, records in cohorts:
                summary = summarize(records, shared, [system])
                summary[0]["actual_grid_audit_verified"] = audited
                if not audited:
                    summary[0]["comparison_eligible"] = False
                    summary[0]["timings_skipped_reason"] = "Four-arm actual-mesh / retained-rank-and-spectrum audit did not pass"
                atomic_json(base / "summary.json", summary)
                item["basis_runs"][basis] = summary[0]
            item["grid_audits_verified"] = audited
            item["diagnostic_energy_audits"] = energy_proofs
            item["complete"] = audited and all(proof["verified"] for proof in native_proofs.values()) and all(row["complete"] and row["comparison_eligible"] and row.get("energy_review") is False
                                               for row in item["basis_runs"].values())
            results.append(item)
            atomic_json(a.output / "control-results.json", results)
    pins.update(a._config_pins)
    pins.update({a.output / "inputs/geometries" / f"{system}.xyz": sha
                 for system, sha in a._geometry_pins.items()})
    atomic_json(a.output / "final-pins.json", {str(path): sha for path, sha in pins.items()})
    unchanged = all(path.exists() and digest(path) == sha for path, sha in pins.items())
    complete = unchanged and bool(results) and all(item["complete"] for item in results)
    atomic_json(a.output / "COMPLETE.json", dict(
        complete=complete, runtime_pins_unchanged=unchanged,
        actual_grid_audit_required=True, numerical_review_is_not_certification=True))
    return 0 if complete else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config-cc-pvtz.json")
    for name in ("psi-env", "psi-package", "psi-cuda-lib", "psi-xc-lib", "gpu-env", "cuda12-lib", "data", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--systems", choices=("peptide", "protein83"), nargs="+")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--stop-on-worker-failure", action="store_true")
    parser.add_argument("--jk-mode", choices=("density-fitted", "conventional-direct"))
    parser.add_argument("--df-modes", choices=("native-strategy", "stored-df"),
                        nargs="+")
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    args.systems = args.systems or (["peptide"] if args.preflight else ["protein83"])
    if args.repeats < 1 or args.threads < 1:
        parser.error("At least one timing repeat and one thread are required")
    # Match the controller's TERM handling: stop workers, preserve observations.
    import signal
    def on_term(*_):
        raise KeyboardInterrupt("SLURM TERM")
    signal.signal(signal.SIGTERM, on_term)
    try:
        return control(args)
    except BaseException as exc:
        # Only write into this invocation's newly created run, never overwrite
        # a completed run when output.mkdir(exist_ok=False) rejects a retry.
        if getattr(args, "_output_created", False) and not (args.output / "COMPLETE.json").exists():
            failure = args.output / "FAILED.json"
            if not failure.exists():
                with failure.open("x") as stream:
                    json.dump(dict(error=f"{type(exc).__name__}: {exc}", complete=False), stream)
        raise


if __name__ == "__main__":
    sys.exit(main())
