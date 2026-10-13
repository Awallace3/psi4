#!/usr/bin/env python3
"""Same-allocation, fresh-process GPU DFT benchmark; stdlib-only controller."""
import argparse
from collections import defaultdict
import contextlib
import ctypes
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import signal
import statistics
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ENGINES = ("psi4", "pyscf")
ATOMS = {"peptide": 24, "nanotube": 42, "protein83": 83, "protein157": 157}
# Spherical cc-pVDZ: H=5, C/N/O=14.
NBF = {"peptide": 210, "nanotube": 444, "protein83": 766, "protein157": 1478}
NBF_BY_BASIS = {
    "cc-pvdz": NBF,
    # Spherical aug-cc-pVTZ: H=23, C/N/O=46.
    "aug-cc-pvtz": {"peptide": 782, "nanotube": 1564, "protein83": 2806, "protein157": 5382},
}


def expected_nbf(c, system):
    """Reject unsupported basis/count assumptions before costly GPU work."""
    return NBF_BY_BASIS[c["basis"].lower()][system]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def xyz(path, system):
    lines = Path(path).read_text().splitlines()
    n = int(lines[0])
    assert n == ATOMS[system] and len(lines[2:]) == n, "Geometry size mismatch"
    assert "charge=0" in lines[1] and "mult=1" in lines[1]
    for line in lines[2:]:
        fields = line.split()
        assert fields[0] in ("H", "C", "N", "O") and len(fields) == 4
        assert all(math.isfinite(float(x)) for x in fields[1:])
    return "\n".join(lines[2:])


def psi_options(c):
    return dict(
        basis=c["basis"], puream=True, reference="rhf", scf_type="df",
        df_basis_scf=c["auxbasis"], use_cuest=True, cuest_xc=True,
        cuest_mixed_precision=c["cuest_mixed_precision"], guess="sad",
        e_convergence=c["energy_convergence"],
        d_convergence=c["orbital_gradient_convergence"],
        maxiter=c["max_iterations"], fail_on_maxiter=True,
        ints_tolerance=c["integral_tolerance"],
        df_fitting_condition=c["df_fitting_condition"],
        dft_radial_points=c["radial_points"],
        dft_spherical_points=c["angular_points"], dft_pruning_scheme="none",
        dft_nuclear_scheme="becke",
    )


def versions():
    names = ("pyscf", "gpu4pyscf-cuda12x", "gpu4pyscf-libxc-cuda12x",
             "cupy-cuda12x", "cutensor-cu12", "numpy")
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def runtime_libraries():
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split()
        if fields and fields[-1].startswith("/") and any(
                term in fields[-1].lower() for term in
                ("libxc", "cuest", "libcuda", "libcudart", "libcublas",
                 "libcutensor", "libstdc++", "libcusolver")):
            paths.add(fields[-1])
    return sorted(paths)


def cuda_identity(runtime=None):
    """Use the process's already loaded CUDA runtime, not NVML's index order."""
    if runtime is not None:
        # CuPy wheels can statically link cudart inside their extension, so a
        # valid runtime need not appear as libcudart.so in /proc/self/maps.
        count, device = runtime.getDeviceCount(), runtime.getDevice()
        assert count == 1 and device == 0, "Exactly one visible CUDA GPU is required"
        raw = bytes(runtime.getDeviceProperties(device)["uuid"])
        assert len(raw) == 16, ("Invalid CUDA device UUID", raw)
        value = raw.hex()
        return dict(uuid="GPU-" + "-".join((value[:8], value[8:12], value[12:16],
                    value[16:20], value[20:])), logical_device=device,
                    visible_count=count, runtime="cupy.cuda.runtime",
                    cuda_runtime=runtime.runtimeGetVersion())
    paths = [p for p in runtime_libraries() if "/libcudart.so" in p]
    assert len(set(paths)) == 1, ("Ambiguous CUDA runtime", paths)
    rt = ctypes.CDLL(paths[0])
    device, count = ctypes.c_int(), ctypes.c_int()
    assert rt.cudaGetDeviceCount(ctypes.byref(count)) == 0 and count.value == 1, \
        "Exactly one visible CUDA GPU is required"
    assert rt.cudaGetDevice(ctypes.byref(device)) == 0 and device.value == 0
    # cudaDeviceGetUuid is not exported by the CUDA13 runtime. Resolve the
    # selected CUDA ordinal to PCI bus first, then query NVML by that bus.
    pci = ctypes.create_string_buffer(64)
    assert rt.cudaDeviceGetPCIBusId(pci, len(pci), device) == 0
    bus = pci.value.decode()
    uuid = subprocess.check_output(["nvidia-smi", "-i", bus, "--query-gpu=uuid",
                                    "--format=csv,noheader"], text=True).strip()
    assert uuid.startswith("GPU-") and "\n" not in uuid
    return dict(uuid=uuid, pci_bus_id=bus, logical_device=device.value,
                visible_count=count.value, runtime=paths[0])


def verify_components(actual, expected):
    assert set(actual) == set(expected), ("XC components differ", actual, expected)
    assert all(math.isclose(actual[k], expected[k], rel_tol=0, abs_tol=1e-12)
               for k in expected), ("XC component weights differ", actual, expected)


def load_result(path, expected):
    """Malformed output and stale/wrong identities are failed measurements."""
    try:
        r = json.loads(path.read_text())
        assert isinstance(r, dict)
        assert all(r.get(k) == v for k, v in expected.items()), "Result identity mismatch"
        if r.get("ok"):
            assert all(math.isfinite(r[k]) for k in
                       ("calculation_wall_s", "scf_wall_s", "energy_hartree"))
            assert all(k in r for k in ("scf_iterations", "nbf", "naux", "functional_metadata"))
        return r
    except (OSError, ValueError, AssertionError, TypeError, KeyError) as exc:
        return dict(expected, ok=False, error=f"Invalid/missing worker result: {exc}")


class Timings:
    """Inclusive, synchronized diagnostic boundaries; NOT headline timings."""
    def __init__(self, sync):
        self.sync = sync
        self.data = defaultdict(lambda: dict(calls=0, wall_s=0.0))

    def wrap(self, obj, name, label):
        original = getattr(obj, name)
        def measured(*args, **kwargs):
            self.sync()
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.sync()
                self.data[label]["calls"] += 1
                self.data[label]["wall_s"] += time.perf_counter() - start
        setattr(obj, name, measured)

    def wrap_class(self, cls, name, label):
        """Copied range-separated DF objects must invoke build on themselves."""
        original = getattr(cls, name)
        def measured(instance, *args, **kwargs):
            self.sync()
            start = time.perf_counter()
            try:
                return original(instance, *args, **kwargs)
            finally:
                self.sync()
                self.data[label]["calls"] += 1
                self.data[label]["wall_s"] += time.perf_counter() - start
        setattr(cls, name, measured)


def run_pyscf(a, c, atom, r):
    import_start = time.perf_counter()
    import warnings
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        import cupy as cp
        import pyscf
        from pyscf import gto, lib
        import gpu4pyscf
        from gpu4pyscf.dft import rks
        # Prefer failure to a CPU or CuPy contraction fallback.
        from gpu4pyscf.lib import cutensor
    r["imports_wall_s"] = time.perf_counter() - import_start
    r["import_warnings"] = [str(w.message) for w in captured]
    assert not any("using cupy as the tensor contraction engine" in w.lower()
                   for w in r["import_warnings"]), "cuTENSOR fallback"
    assert Path(gpu4pyscf.__file__).resolve().is_relative_to(a.gpu_env.resolve())
    r["versions"] = versions()
    calculation_start = time.perf_counter()
    lib.num_threads(c["threads"])
    cp.cuda.Device(0).use()
    sync = cp.cuda.runtime.deviceSynchronize
    sync()
    name = cp.cuda.runtime.getDeviceProperties(0)["name"]
    r.update(module=gpu4pyscf.__file__, pyscf_module=pyscf.__file__,
             versions=versions(), gpu_name=name.decode() if isinstance(name, bytes) else name,
             cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
             cuda_driver=cp.cuda.runtime.driverGetVersion(),
             cutensor_module=cutensor.__file__)
    start = time.perf_counter()
    mol = gto.M(atom=atom, basis=c["basis"], unit="Angstrom", charge=0,
                spin=0, cart=False, verbose=4,
                output=str(Path.cwd() / "pyscf.out"),
                max_memory=c["memory_gib"] * 1024)
    mf = rks.RKS(mol, xc=c["functionals"][a.functional]["pyscf"]).density_fit(
        auxbasis=c["auxbasis"], only_dfj=False)
    assert mf.__class__.__module__.startswith("gpu4pyscf.")
    mf.grids.atom_grid = (c["radial_points"], c["angular_points"])
    mf.grids.prune = None
    mf.small_rho_cutoff = 0.0
    mf.conv_tol = c["energy_convergence"]
    mf.conv_tol_grad = c["orbital_gradient_convergence"]
    mf.max_cycle = c["max_iterations"]
    mf.direct_scf_tol = c["integral_tolerance"]
    mf.init_guess = "atom"
    mf.diis_space = 8
    from gpu4pyscf.df import df
    df.LINEAR_DEP_THR = c["df_fitting_condition"]
    r["df_metric_note"] = "PySCF uses Cholesky then absolute eigenvalue cutoff; Psi4/cuEST fitting is not algebraically identical"
    # Psi4 SVWN and B3LYP use VWN-RPA. Name the exact LibXC components,
    # rather than confuse those with the distinct VWN5 variants.
    omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=0)
    from pyscf.dft import libxc
    _, parsed = libxc.parse_xc(mf.xc)
    expected = c["functionals"][a.functional]["components"]
    ids = {int(libxc.XC_CODES[name]): name for name in expected}
    components = {ids.get(int(fid), f"UNKNOWN_ID_{fid}"): float(weight) for fid, weight in parsed}
    verify_components(components, expected)
    r["functional_metadata"] = dict(xc=mf.xc, omega=float(omega),
                                    components=components,
                                    long_range_exchange=float(alpha if omega else hyb),
                                    short_range_exchange=float(hyb),
                                    xc_type=mf._numint._xc_type(mf.xc))
    r["setup_wall_s"] = time.perf_counter() - start
    history = []
    def callback(env):
        history.append(dict(cycle=int(env["cycle"]), energy_eh=float(env["e_tot"]),
                            elapsed_s=time.perf_counter() - scf_start))
    mf.callback = callback
    timers = Timings(sync)
    if a.profile:
        timers.wrap(mf, "get_jk", "jk")
        timers.wrap(mf, "get_j", "j_inclusive")
        timers.wrap(mf, "get_k", "k_inclusive")
        timers.wrap(mf._numint, "nr_rks", "xc")
        timers.wrap_class(type(mf.with_df), "build", "df_setup")
        timers.wrap(mf, "eig", "diagonalization")
        timers.wrap(mf.grids, "build", "grid_setup")
    sync()
    scf_start = time.perf_counter()
    energy = mf.kernel()
    sync()
    calculation_elapsed = time.perf_counter() - calculation_start
    r.update(scf_wall_s=time.perf_counter() - scf_start,
             calculation_wall_s=calculation_elapsed,
             energy_hartree=float(energy), converged=bool(mf.converged),
             scf_iterations=len(history), iteration_history=history,
             nbf=int(mol.nao_nr()), naux=int(mf.with_df.auxmol.nao_nr()),
             grid_points=int(mf.grids.weights.size),
             diagnostic_timers=dict(timers.data),
             cupy_pool_peak_note="End-of-run pool reservation is NOT peak device memory",
             cupy_pool_reserved_bytes=int(cp.get_default_memory_pool().total_bytes()))
    assert r["converged"], "SCF did not converge"
    r["gpu"] = cuda_identity(cp.cuda.runtime)
    r["effective_grid"] = dict(radial_method=str(mf.grids.radi_method),
        partition=str(mf.grids.becke_scheme), radii_adjust=str(mf.grids.radii_adjust),
        prune=str(mf.grids.prune), small_rho_cutoff=float(mf.small_rho_cutoff))
    r["effective_df"] = dict(linear_dep_threshold=float(df.LINEAR_DEP_THR),
                            direct_scf_tol=float(mf.direct_scf_tol))
    if a.profile:
        assert timers.data["xc"]["calls"] > 0
        assert any(timers.data[k]["calls"] > 0 for k in ("jk", "j_inclusive")), \
            "No J/K calls captured"
    mol.stdout.close()


def run_psi4(a, c, atom, r):
    import_start = time.perf_counter()
    import psi4
    r["imports_wall_s"] = time.perf_counter() - import_start
    assert Path(psi4.__file__).resolve().parent == a.psi_package.resolve()
    calculation_start = time.perf_counter()
    psi4.core.set_output_file(str(Path.cwd() / "psi4.out"), False)
    psi4.set_memory(f'{c["memory_gib"]} GiB')
    psi4.set_num_threads(c["threads"])
    opts = psi_options(c)
    # Historical CUDA-LibXC build has CPU SAD and predates CUEST_SAD.
    try:
        psi4.core.get_global_option("CUEST_SAD")
    except Exception:
        pass
    else:
        opts["cuest_sad"] = False
    psi4.set_options(opts)
    start = time.perf_counter()
    mol = psi4.geometry("0 1\n" + atom + "\nunits angstrom\nsymmetry c1\nno_reorient\nno_com")
    r["setup_wall_s"] = time.perf_counter() - start
    psi4.core.clean_timers()
    start = time.perf_counter()
    energy, wfn = psi4.energy(c["functionals"][a.functional]["psi4"],
                             molecule=mol, return_wfn=True)
    elapsed = time.perf_counter() - start
    calculation_elapsed = time.perf_counter() - calculation_start
    f = wfn.functional()
    components = {component.name().removeprefix("XC_"): float(component.alpha())
                  for component in list(f.x_functionals()) + list(f.c_functionals())}
    verify_components(components, c["functionals"][a.functional]["components"])
    r.update(module=psi4.__file__, version=psi4.__version__, options=opts,
             energy_hartree=float(energy), scf_wall_s=elapsed, converged=True,
             calculation_wall_s=calculation_elapsed,
             nbf=wfn.basisset().nbf(),
             naux=wfn.get_basisset("DF_BASIS_SCF").nbf(),
             scf_iterations=int(wfn.variable("SCF ITERATIONS")),
             timer_records=psi4.core.get_timer_records(),
             functional_metadata=dict(name=f.name(), meta=bool(f.is_meta()),
                 gga=bool(f.is_gga()), omega=float(f.x_omega()),
                 components=components,
                 short_range_exchange=float(f.x_alpha()),
                 long_range_exchange=float(f.x_alpha() + f.x_beta())
                    if f.is_x_lrc() else float(f.x_alpha())))
    psi4.core.close_outfile()
    text = Path("psi4.out").read_text()
    assert "cuESTJK: GPU-Accelerated" in text, "cuEST JK not observed"
    names = ["/".join(t["timer_path"]) for t in r["timer_records"].values()]
    assert any("CUDA LibXC Functional" in n for n in names), \
        "CUDA LibXC timer missing: host-functional build is not this benchmark's GPU arm"
    r["xc_route"] = "cuda-libxc"
    r["gpu"] = cuda_identity()
    r["effective_grid"] = {key: psi4.core.get_option("SCF", key) for key in
        ("DFT_RADIAL_SCHEME", "DFT_NUCLEAR_SCHEME", "DFT_PRUNING_SCHEME",
         "DFT_DENSITY_TOLERANCE", "DFT_BASIS_TOLERANCE")}
    r["effective_grid"]["note"] = "Nominal options may be bypassed by cuEST; actual points are not guaranteed identical"


def worker(a):
    start = time.perf_counter()
    c = json.loads(a.config.read_text())
    a.output.mkdir(parents=True, exist_ok=False)
    os.chdir(a.output)
    r = dict(ok=False, engine=a.engine, system=a.system, functional=a.functional,
             rung=c["functionals"][a.functional]["rung"], profile=a.profile,
             repeat=a.repeat, config_sha256=digest(a.config),
             geometry_sha256=digest(a.xyz), config=c, job_id=os.getenv("SLURM_JOB_ID"),
             hostname=os.uname().nodename, python=sys.executable)
    try:
        expected = expected_nbf(c, a.system)
        atom = xyz(a.xyz, a.system)
        {"psi4": run_psi4, "pyscf": run_pyscf}[a.engine](a, c, atom, r)
        assert r["nbf"] == expected, ("Basis nbf mismatch", c["basis"], r["nbf"], expected)
        assert math.isfinite(r["energy_hartree"])
        r["runtime_libraries"] = runtime_libraries()
        r["ok"] = True
    except Exception as exc:
        r.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    r["worker_wall_s"] = time.perf_counter() - start
    r["maxrss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    atomic_json(a.output / "result.json", r)
    return 0 if r["ok"] else 1


def summarize(records, c, systems):
    rows = []
    for system in systems:
        for functional in c["functionals"]:
            row = dict(system=system, functional=functional,
                       rung=c["functionals"][functional]["rung"])
            groups = {}
            for engine in ENGINES:
                group = [r for r in records if r["system"] == system
                         and r["functional"] == functional and r["engine"] == engine
                         and not r["profile"]]
                good = [r for r in group if r.get("ok") and r.get("process_returncode") == 0]
                complete = len(good) == c["repeats"]
                groups[engine] = good if complete else []
                row[engine] = dict(planned=c["repeats"], succeeded=len(good),
                                   failures=[r.get("error", r.get("process_returncode"))
                                             for r in group if r not in good])
                if complete:
                    for field in ("calculation_wall_s", "scf_wall_s", "process_wall_s", "scf_iterations", "energy_hartree"):
                        values = [r[field] for r in good]
                        row[engine][field] = dict(median=statistics.median(values),
                                                 min=min(values), max=max(values))
            row["complete"] = bool(groups["psi4"] and groups["pyscf"])
            if row["complete"]:
                p, g = row["psi4"], row["pyscf"]
                row["psi4_over_pyscf_scf_time"] = (
                    p["scf_wall_s"]["median"] / g["scf_wall_s"]["median"])
                row["psi4_over_pyscf_calculation_time"] = (
                    p["calculation_wall_s"]["median"] / g["calculation_wall_s"]["median"])
                row["energy_delta_pyscf_minus_psi4_eh"] = (
                    g["energy_hartree"]["median"] - p["energy_hartree"]["median"])
                row["energy_review"] = abs(row["energy_delta_pyscf_minus_psi4_eh"]) > c[
                    "energy_review_threshold_eh"]
                row["nbf_equal"] = {r["nbf"] for r in groups["psi4"] + groups["pyscf"]} == {
                    expected_nbf(c, system)}
                row["naux_equal"] = len({r["naux"] for r in groups["psi4"] + groups["pyscf"]}) == 1
                row["functional_parameters_equal"] = all(math.isclose(
                    groups["psi4"][0]["functional_metadata"][key],
                    groups["pyscf"][0]["functional_metadata"][key],
                    rel_tol=0, abs_tol=1e-9) for key in
                    ("omega", "short_range_exchange", "long_range_exchange"))
                row["functional_components_equal"] = all(r["functional_metadata"]["components"] ==
                    c["functionals"][functional]["components"] for r in groups["psi4"] + groups["pyscf"])
                row["comparison_eligible"] = (row["nbf_equal"] and row["naux_equal"]
                                              and row["functional_parameters_equal"]
                                              and row["functional_components_equal"])
                if not row["comparison_eligible"]:
                    row.pop("psi4_over_pyscf_scf_time")
                    row.pop("psi4_over_pyscf_calculation_time")
            rows.append(row)
    return rows


def engine_env(engine, a):
    env = dict(os.environ, PYTHONNOUSERSITE="1")
    for key in ("LD_PRELOAD", "PSIDATADIR", "PYSCF_CONFIG_FILE", "PYSCF_EXT_PATH"):
        env.pop(key, None)
    # Never allow Psi4's CUDA13 libraries into the CUDA12 PySCF process.
    root = a.psi_env if engine == "psi4" else a.gpu_env
    env["LD_LIBRARY_PATH"] = ":".join(map(str, [root / "lib",
        a.psi_cuda_lib if engine == "psi4" else a.cuda12_lib]))
    env["PYTHONPATH"] = str(a.psi_package.parent) if engine == "psi4" else ""
    if engine == "pyscf":
        env["CUDA_PATH"] = str(a.cuda12_lib.parent)
    env["OMP_NUM_THREADS"] = str(a.threads)
    env["MKL_NUM_THREADS"] = str(a.threads)
    env["OPENBLAS_NUM_THREADS"] = str(a.threads)
    return env


def campaign(a):
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / "results").mkdir()
    (a.output / "harness").mkdir()
    for p in HERE.glob("*"):
        if p.is_file():
            (a.output / "harness" / p.name).write_bytes(p.read_bytes())
    for engine in ENGINES:
        root = a.psi_env if engine == "psi4" else a.gpu_env
        lock = subprocess.check_output([str(root / "bin/python"), "-m", "pip", "freeze", "--all"],
                                       env=engine_env(engine, a), text=True)
        (a.output / f"{engine}-pip-freeze.txt").write_text(lock)
    c = json.loads(a.config.read_text())
    for system in a.systems:
        expected_nbf(c, system)
    if a.functionals:
        assert len(set(a.functionals)) == len(a.functionals), "Duplicate functionals"
        c["functionals"] = {name: c["functionals"][name] for name in a.functionals}
    c["repeats"] = a.repeats
    c["threads"] = a.threads
    atomic_json(a.output / "config.json", c)
    config = a.output / "config.json"
    systems = a.systems
    staged = a.output / "inputs"
    staged.mkdir()
    for system in systems:
        source = a.data / "geometries" / f"{system}.xyz"
        xyz(source, system)
        (staged / source.name).write_bytes(source.read_bytes())
    core = next(a.psi_package.glob("core*.so"))
    ldd = subprocess.check_output(["ldd", str(core)], env=engine_env("psi4", a), text=True)
    assert "not found" not in ldd, ldd
    provenance_files = {core, a.psi_package / "metadata.py"}
    for line in ldd.splitlines():
        words = line.split()
        if "=>" in words and any(k in words[0].lower() for k in ("libxc", "libcuest")):
            provenance_files.add(Path(words[words.index("=>") + 1]).resolve())
    provenance_files.update(a.psi_package.glob("driver/**/*.py"))
    basis_dir = a.psi_package.parents[1] / "share/psi4/basis"
    provenance_files.update(basis_dir.glob("*.gbs"))
    for lib in a.psi_cuda_lib.glob("libxc.so*"):
        if lib.is_file():
            provenance_files.add(lib.resolve())
    for package in ("gpu4pyscf", "pyscf"):
        for root in a.gpu_env.glob(f"lib/python*/site-packages/{package}"):
            provenance_files.update(p for p in root.rglob("*")
                                    if p.is_file() and (p.suffix in (".py", ".so")))
    fingerprints = {str(p): digest(p) for p in sorted(provenance_files)}
    harness_fingerprints = {str(p): digest(p) for p in HERE.iterdir() if p.is_file()}
    device = json.loads(subprocess.check_output(
        [str(a.gpu_env / "bin/python"), str(HERE / "compare.py"), "device"],
        env=engine_env("pyscf", a), text=True))
    manifest = dict(schema=1, config=c, commit=subprocess.check_output(
        ["git", "-C", str(HERE), "rev-parse", "HEAD"], text=True).strip(),
        job_id=os.getenv("SLURM_JOB_ID"), qos=os.getenv("SLURM_JOB_QOS"),
        partition=os.getenv("SLURM_JOB_PARTITION"), command=sys.argv,
        gpu_identity=device, harness_sha256=harness_fingerprints,
        gpu=subprocess.check_output(["nvidia-smi", "-i", device["uuid"], "--query-gpu=name,uuid,driver_version,memory.total",
                                     "--format=csv"], text=True),
        systems=systems, staged_build_sha256=fingerprints,
        psi4_linkage=ldd,
        build_note="Reused CUDA-LibXC build; staged hashes, NOT harness HEAD, identify compiled code",
        inputs_sha256={p.name: digest(p) for p in staged.glob("*.xyz")})
    atomic_json(a.output / "manifest.json", manifest)
    records = []
    monitor_log = (a.output / "gpu-samples.csv").open("w")
    monitor = subprocess.Popen(["nvidia-smi", "-i", device["uuid"], "--query-gpu=timestamp,uuid,utilization.gpu,memory.used,power.draw",
                                "--format=csv", "-lms", "1000"], stdout=monitor_log,
                               stderr=subprocess.STDOUT)
    try:
        for system in systems:
            for functional in c["functionals"]:
                # Profiles are separate, last, and never enter headline medians.
                for repeat in range(c["repeats"] + int(a.profiles)):
                    profile = repeat == c["repeats"]
                    for engine in ENGINES[::1 if repeat % 2 == 0 else -1]:
                        name = f"{system}-{functional}-{engine}-{'profile' if profile else repeat}"
                        out = a.output / "results" / name
                        py = (a.psi_env if engine == "psi4" else a.gpu_env) / "bin/python"
                        cmd = [str(py), str(HERE / "compare.py"), "worker",
                               "--engine", engine, "--system", system, "--functional", functional,
                               "--repeat", str(repeat), "--config", str(config),
                               "--xyz", str(staged / f"{system}.xyz"), "--output", str(out),
                               "--psi-package", str(a.psi_package), "--gpu-env", str(a.gpu_env)]
                        if profile:
                            cmd.append("--profile")
                        env = engine_env(engine, a)
                        start = time.perf_counter()
                        started_unix_s = time.time()
                        with (a.output / f"{name}.log").open("w") as log:
                            proc = subprocess.Popen(cmd, env=env, stdout=log,
                                                    stderr=subprocess.STDOUT, start_new_session=True)
                            try:
                                rc = proc.wait(timeout=c["worker_timeout_s"])
                            except subprocess.TimeoutExpired:
                                os.killpg(proc.pid, signal.SIGTERM)
                                try:
                                    proc.wait(timeout=30)
                                except subprocess.TimeoutExpired:
                                    os.killpg(proc.pid, signal.SIGKILL)
                                    proc.wait()
                                rc = 124
                            except BaseException:
                                os.killpg(proc.pid, signal.SIGTERM)
                                try:
                                    proc.wait(timeout=30)
                                except subprocess.TimeoutExpired:
                                    os.killpg(proc.pid, signal.SIGKILL)
                                    proc.wait()
                                raise
                        r = load_result(out / "result.json", dict(engine=engine,
                            system=system, functional=functional, profile=profile, repeat=repeat))
                        r.update(process_returncode=rc, process_wall_s=time.perf_counter() - start,
                                 started_unix_s=started_unix_s, ended_unix_s=time.time(), output=str(out))
                        if rc:
                            r["ok"] = False
                        if r.get("ok") and r["gpu"]["uuid"] != device["uuid"]:
                            r.update(ok=False, error="Worker ran on a different GPU UUID")
                        records.append(r)
                        atomic_json(a.output / "records.json", records)
                        atomic_json(a.output / "summary.json", summarize(records, c, systems))
                        print(name, "OK" if r["ok"] else "FAILED", r.get("error", ""), flush=True)
                        if a.stop_on_worker_failure and not r["ok"]:
                            raise RuntimeError(f"Stopping after failed worker {name}; records preserved")
    except BaseException as exc:
        atomic_json(a.output / "FAILED.json", dict(error=str(exc), traceback=traceback.format_exc(),
                    interrupted=isinstance(exc, KeyboardInterrupt), completed_workers=len(records)))
        raise
    finally:
        monitor.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            monitor.wait(timeout=10)
        if monitor.poll() is None:
            monitor.kill()
            monitor.wait()
        monitor_log.close()
    unchanged = all(digest(p) == sha for p, sha in {**fingerprints, **harness_fingerprints}.items())
    rows = summarize(records, c, systems)
    complete = (unchanged and all(r["ok"] for r in records)
                and all(r["complete"] and r["comparison_eligible"] for r in rows))
    atomic_json(a.output / "COMPLETE.json", dict(complete=complete,
        final_build_hashes_unchanged=unchanged, energy_review_is_not_accuracy_certification=True))
    with (a.output / "summary.csv").open("w", newline="") as f:
        columns = ["system", "functional", "rung", "complete", "comparison_eligible",
                   "psi4_over_pyscf_calculation_time", "psi4_over_pyscf_scf_time",
                   "energy_delta_pyscf_minus_psi4_eh", "energy_review"]
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return 0 if complete else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    sub.add_parser("device")
    w = sub.add_parser("worker")
    w.add_argument("--engine", choices=ENGINES, required=True)
    w.add_argument("--system", choices=ATOMS, required=True)
    w.add_argument("--functional", required=True)
    w.add_argument("--repeat", type=int, required=True)
    w.add_argument("--profile", action="store_true")
    w.add_argument("--xyz", type=Path, required=True)
    w.add_argument("--psi-package", type=Path, required=True)
    w.add_argument("--gpu-env", type=Path, required=True)
    c = sub.add_parser("campaign")
    c.add_argument("--systems", choices=ATOMS, nargs="+", default=list(ATOMS))
    c.add_argument("--data", type=Path, required=True)
    c.add_argument("--psi-package", type=Path, required=True)
    c.add_argument("--psi-env", type=Path, required=True)
    c.add_argument("--psi-cuda-lib", type=Path, required=True)
    c.add_argument("--gpu-env", type=Path, required=True)
    c.add_argument("--cuda12-lib", type=Path, required=True)
    c.add_argument("--threads", type=int, default=8)
    c.add_argument("--repeats", type=int, default=3)
    c.add_argument("--profiles", action="store_true")
    c.add_argument("--functionals", nargs="+",
                   help="Subset for focused preflight; selection is pinned in campaign config")
    c.add_argument("--stop-on-worker-failure", action="store_true",
                   help="Do not spend repeats on a known failing large-basis route")
    for parser in (w, c):
        parser.add_argument("--config", type=Path, default=HERE / "config.json")
        parser.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.action == "device":
        import cupy
        cupy.cuda.Device(0).use()
        print(json.dumps(cuda_identity(cupy.cuda.runtime)))
        return 0
    for key, value in vars(a).items():
        if isinstance(value, Path):
            setattr(a, key, value.resolve())
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    return worker(a) if a.action == "worker" else campaign(a)


if __name__ == "__main__":
    sys.exit(main())
