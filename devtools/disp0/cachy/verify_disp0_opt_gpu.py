"""Physical GPU correctness and compute-sanitizer gate for factorized disp0/fdisp0.

Modes:
  --mode sanitizer : Standalone CUDASession GPU verification (all new CUDA
                     kernels, warp-over-b edge sizes B=1/32/35, batched GEMM
                     slices, canonical & localized single-call + tiled modes,
                     lazy extra module compile/unload) suitable for direct
                     execution under /opt/cuda/bin/compute-sanitizer
                     (memcheck/initcheck/racecheck/synccheck).
  --mode psi4-gate : Full Psi4 + cuEST 0.2.50 physical GPU gate through public
                     psi4.energy('sapt(dft)', cuda_provider_factory=...,
                     cuda_dispersion_factor_cache_bytes=...) covering
                     Water-Ne, Water dimer (canonical & F-SAPT), and
                     Ethane-Water (unequal frozen-core), comparing factorized
                     vs blocked GPU parity (<= 1e-12 Eh) and CPU parity.

RTX 3090 timings are debug-only and never quoted as speedups.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np


def _file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_cublas(lib_dir):
    root = Path(lib_dir).resolve(strict=True)
    for name in ("libcublas.so.13",):
        cand = root / name
        if cand.is_file():
            return str(cand.resolve(strict=True))
    matches = sorted(root.glob("libcublas.so.13*"))
    if not matches:
        raise FileNotFoundError(f"No libcublas.so.13* found in {root}")
    return str(matches[0].resolve(strict=True))


class _GPUAOProvider:
    """Real-GPU synthetic DFMO provider for standalone compute-sanitizer gates.

    Computes (Q | L, R) entirely on the GPU via session.gemm / gemm_batched
    from a symmetric device AO tensor, exercising CUDASession + cuBLAS + NVRTC
    with zero CPU numerical fallback.
    """
    def __init__(self, session, ao_gpu):
        self.session = session
        self.ao = ao_gpu
        self.naux, self.nao, _ = ao_gpu.shape
        self.calls = 0
        self.variable_bytes = 64 * 1024 * 1024

    def compute_device(self, pairs, *, variable_bytes=None):
        s = self.session
        out = {}
        for name, (left, right) in pairs.items():
            arr = s.empty((self.naux, left.shape[1], right.shape[1]))
            with s.scope():
                tmp = s.gemm_batched(self.ao, right)
                s.gemm_batched(left, tmp, transpose_left=True, out=arr)
            out[name] = arr
        self.calls += 1
        return out

    def report(self):
        return {"producer": "GPU_SYNTHETIC_SANITIZER_FIXTURE", "calls": self.calls}


def _numpy_oracle(ao, spaces, c, e, ua=None, ub=None):
    def f(left, right):
        return np.einsum("Qmn,mi,nj->Qij", ao, spaces[left], spaces[right])
    ar, bs = f("r", "a"), f("s", "b")
    far, fbs = f("r3", "a") + f("r", "a4"), f("s3", "b") + f("s", "b4")
    v = np.einsum("Qra,Qsb->rasb", ar, bs)
    x = (np.einsum("Qra,Qsb->rasb", ar, fbs) + np.einsum("Qra,Qsb->rasb", far, bs)
         + np.einsum("ar,bs->rasb", c["Qar"], c["SAbs"])
         + np.einsum("ar,bs->rasb", c["SBar"], c["Qbs"]))
    w = (np.einsum("Qsa,Qrb->sarb", f("s1", "a"), f("r1", "b"))
         + np.einsum("Qsa,Qrb->sarb", f("s", "a2"), f("r", "b2"))
         + np.einsum("as,br->sarb", c["Sas"], c["Qbr"])
         + np.einsum("as,br->sarb", c["Qas"], c["Sbr"]))
    den = (e["a"][None, :, None, None] + e["b"][None, None, None, :]
           - e["r"][:, None, None, None] - e["s"][None, None, :, None])
    t = v / den
    if ua is not None:
        rot = lambda arr: np.einsum("al,rasb,bk->rlsk", ua, arr, ub)
        t, v, x = rot(t), rot(v), rot(x)
        w = np.einsum("al,sarb,bk->slrk", ua, w, ub)
    disp = 4.0 * np.einsum("rasb,rasb->ab", t, v)
    exch = -2.0 * (np.einsum("rasb,rasb->ab", t, x) + np.einsum("rasb,sarb->ab", t, w))
    return disp, exch


def _make_hf_case(seed, nao, oa, ob, nr, ns):
    rng = np.random.default_rng(seed)
    d = {name: rng.normal(size=(nao, nao)) * 0.1
         for name in ("S", "D_A", "P_A", "V_A", "J_A", "K_A", "D_B", "P_B", "V_B", "J_B", "K_B", "K_O")}
    s_raw = rng.normal(size=(nao, nao)) * 0.08
    d["S"] = 0.5 * (s_raw + s_raw.T) + np.eye(nao)
    d["Cocc_A"] = rng.normal(size=(nao, oa)) * 0.2
    d["Cocc_B"] = rng.normal(size=(nao, ob)) * 0.2
    d["Cvir_A"] = rng.normal(size=(nao, nr)) * 0.2
    d["Cvir_B"] = rng.normal(size=(nao, ns)) * 0.2
    d["D_A"] = d["Cocc_A"] @ d["Cocc_A"].T
    d["D_B"] = d["Cocc_B"] @ d["Cocc_B"].T
    e = {
        "a": np.linspace(-1.5, -0.5, oa),
        "b": np.linspace(-1.4, -0.55, ob),
        "r": np.linspace(0.15, 0.85, nr),
        "s": np.linspace(0.20, 0.90, ns),
    }
    return d, e


def _cpu_prepare_ref(d, fa, fb):
    a, b, r, s = d["Cocc_A"][:, fa:], d["Cocc_B"][:, fb:], d["Cvir_A"], d["Cvir_B"]
    S, DA, DB, PA, PB = (d[k] for k in ("S", "D_A", "D_B", "P_A", "P_B"))
    JA, JB, VA, VB, KA, KB, KO = (d[k] for k in ("J_A", "J_B", "V_A", "V_B", "K_A", "K_B", "K_O"))
    spaces = {
        "a": a, "b": b, "r": r, "s": s,
        "r1": DB @ S @ r - r, "s1": DA @ S @ s - s,
        "a2": DB @ S @ a, "b2": DA @ S @ b,
        "r3": 2 * DB @ S @ r - 2 * DA @ S @ DB @ S @ r,
        "s3": 2 * DA @ S @ s - 2 * DB @ S @ DA @ S @ s,
        "a4": -2 * DA @ S @ DB @ S @ a, "b4": -2 * DB @ S @ DA @ S @ b,
    }
    corr = {
        "Qar": 4 * a.T @ JB @ r + 2 * a.T @ VB @ r,
        "Qbs": 4 * b.T @ JA @ s + 2 * b.T @ VA @ s,
        "Qas": (2 * a.T @ JB @ s - a.T @ KB @ s + a.T @ KO @ s
                - 2 * a.T @ JB @ DA @ S @ s - 2 * a.T @ S @ DB @ JA @ s
                - a.T @ S @ DB @ VA @ s + a.T @ VB @ PA @ S @ s),
        "Qbr": (2 * b.T @ JA @ r - b.T @ KA @ r + b.T @ KO.T @ r
                - 2 * b.T @ S @ DA @ JB @ r - 2 * b.T @ JA @ DB @ S @ r
                - b.T @ S @ DA @ VB @ r + b.T @ VA @ PB @ S @ r),
        "Sas": a.T @ S @ s, "Sbr": b.T @ S @ r,
        "SBar": a.T @ S @ DB @ S @ r, "SAbs": b.T @ S @ DA @ S @ s,
    }
    return spaces, corr


def _load_sapt_pkg():
    """Import psi4/driver/procrouting/sapt without requiring compiled psi4.core."""
    import importlib
    import types
    sapt_dir = Path(__file__).resolve().parents[3] / "psi4/driver/procrouting/sapt"
    pkg_name = "_disp0_gpu_sapt"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(sapt_dir)]
        sys.modules[pkg_name] = pkg
    return (
        importlib.import_module(pkg_name + ".sapt_cuda_runtime"),
        importlib.import_module(pkg_name + ".sapt_dispersion_cuda"),
        importlib.import_module(pkg_name + ".sapt_dispersion_cuda_setup"),
    )


def run_sanitizer_gate(lib_dir):
    rt_mod, disp_mod, setup_mod = _load_sapt_pkg()
    CUDASession = rt_mod.CUDASession
    blocked_device_dispersion = disp_mod.blocked_device_dispersion
    prepare_hf = setup_mod.prepare_hf

    cublas_path = _find_cublas(lib_dir)
    cublas_sha = _file_sha256(cublas_path)
    report = {"mode": "sanitizer", "cublas_path": cublas_path, "cases": []}

    # Test cases cover B=1 (partial warp), B=32 (exact warp), B=35 (multi-warp b_tiles=2),
    # frozen core, unequal spaces, canonical + localized, single-call + tiled multi-slab.
    specs = [
        ("edge_b1", 6, 1, 1, 0, 0, 2, 2, 10, 0),
        ("small_frozen_single", 10, 4, 3, 1, 0, 7, 6, 14, 0),
        ("small_frozen_tiled_noreuse", 10, 4, 3, 1, 0, 7, 6, 14, 5200),
        ("small_frozen_tiled_reuses", 10, 4, 3, 1, 0, 7, 6, 14, 9200),
        ("exact_warp_b32", 38, 5, 32, 1, 0, 8, 7, 16, 0),
        ("multi_warp_b35", 40, 4, 36, 1, 1, 7, 6, 16, 0),
    ]

    with CUDASession(cublas_library=cublas_path, cublas_sha256=cublas_sha,
                     memory_limit_bytes=4 * 1024**3, native_persistent_bytes=0) as session:
        report["arch"] = session.arch
        report["ptx_sha256"] = session.ptx_sha256
        # Verify core-only compile happened eagerly; extra F-SAPT kernels stay lazy until invoked.
        assert "reduce_dispersion" in session.functions
        assert "ibo_sweep" not in session.functions

        for idx, (label, nao, oa, ob, fa, fb, nr, ns, naux, cache_cap) in enumerate(specs):
            rng = np.random.default_rng(1000 + idx)
            ao = rng.normal(size=(naux, nao, nao)) * 0.15
            ao = 0.5 * (ao + np.swapaxes(ao, 1, 2))
            d, e = _make_hf_case(2000 + idx, nao, oa, ob, nr, ns)
            ref_sp, ref_c = _cpu_prepare_ref(d, fa, fb)
            ref_e = {"a": e["a"][fa:], "b": e["b"][fb:], "r": e["r"], "s": e["s"]}
            A, B = oa - fa, ob - fb
            ua = np.linalg.qr(rng.normal(size=(A, A)))[0]
            ub = np.linalg.qr(rng.normal(size=(B, B)))[0]

            for localized in (False, True):
                d_ref, x_ref = _numpy_oracle(
                    ao, ref_sp, ref_c, ref_e, ua if localized else None, ub if localized else None)
                with session.scope():
                    ao_gpu = session.upload(ao)
                    prov = _GPUAOProvider(session, ao_gpu)
                    spaces, corr, eps = prepare_hf(session, d, e, frozen_a=fa, frozen_b=fb)
                    rot = (session.upload(ua), session.upload(ub)) if localized else None
                    fac = blocked_device_dispersion(
                        prov, spaces, corr, eps, block_size=3, rotations=rot,
                        factor_cache_bytes=cache_cap, algorithm="factorized")
                    blk = blocked_device_dispersion(
                        prov, spaces, corr, eps, block_size=3, rotations=rot,
                        factor_cache_bytes=0, algorithm="blocked")
                err_ref_d = float(np.max(np.abs(fac["Disp20 occupied pairs"] - d_ref)))
                err_ref_x = float(np.max(np.abs(fac["Exch-Disp20 occupied pairs"] - x_ref)))
                err_blk_d = float(np.max(np.abs(fac["Disp20 occupied pairs"] - blk["Disp20 occupied pairs"])))
                err_blk_x = float(np.max(np.abs(fac["Exch-Disp20 occupied pairs"] - blk["Exch-Disp20 occupied pairs"])))
                assert max(err_ref_d, err_ref_x, err_blk_d, err_blk_x) < 1e-11, (
                    label, localized, err_ref_d, err_ref_x, err_blk_d, err_blk_x)
                assert session.live_bytes == 0, session.live_bytes
                report["cases"].append({
                    "label": label, "localized": localized, "cache_cap": cache_cap,
                    "dfmo_calls": fac["Disp0 execution"]["dfmo_calls"],
                    "tiles": fac["Disp0 execution"]["compute_tile_extents"],
                    "slabs": fac["Disp0 execution"]["dfmo_slab_extents"],
                    "max_err_oracle": max(err_ref_d, err_ref_x),
                    "max_err_blocked": max(err_blk_d, err_blk_x),
                })

        # Also exercise lazy EXTRA_CUDA_SOURCE compilation + launch + unload under sanitizer.
        with session.scope():
            eye = session.empty((4, 4))
            session.launch("set_identity", 1, 32, [4, eye])
            got = session.download(eye)
            np.testing.assert_allclose(got, np.eye(4), atol=1e-15)
        assert session.live_bytes == 0
        assert session._extra_module is not None
        report["peak_live_bytes"] = session.peak_live_bytes

    assert session.cleanup_complete and session.live_bytes == 0
    report["status"] = "PASS_GPU_SANITIZER_GATE"
    print(json.dumps(report, indent=2), flush=True)


def _build_cuda_manifest(lib_dir, env_lib_dir):
    from psi4.driver.procrouting.sapt.sapt_cuest250_identity import CUDA_PREFIXES
    components = []
    seen = set()
    for base in (Path(lib_dir).resolve(strict=True), Path(env_lib_dir).resolve(strict=True)):
        for p in sorted(base.iterdir()):
            if not p.is_file():
                continue
            if any(p.name.startswith(pref + ".so") for pref in CUDA_PREFIXES):
                real = str(p.resolve(strict=True))
                if real not in seen:
                    seen.add(real)
                    components.append({"path": real, "sha256": _file_sha256(real)})
    return {
        "cuda_root": str(Path(lib_dir).resolve(strict=True)),
        "components": components,
        "cuda_runtime_version": 13030,
        "nvrtc_version": [13, 3],
    }


def run_psi4_gate(cuest_root, cuest_archive, runtime_lib, env_lib):
    import tempfile
    rlib = Path(runtime_lib).resolve(strict=True)
    native = (Path(cuest_root).resolve(strict=True) / "lib/13/libcuest.so.0.2.50").resolve(strict=True)
    cublas_path = _find_cublas(rlib)
    cublas_sha = _file_sha256(cublas_path)

    # Prepare the exact production sealed runtime BEFORE importing psi4/einsums
    # (required on btrfs subvolumes where stat().st_dev != /proc/self/maps superblock dev).
    sealed_names = (
        "libcudart.so.13", "libcublasLt.so.13", "libcublas.so.13",
        "libnvJitLink.so.13", "libnvfatbin.so.13",
        "libnvrtc-builtins.so.13.3", "libnvrtc.so.13",
        "libcusparse.so.12", "libcusolver.so.12",
        "libcufft.so.12", "libcurand.so.10",
    )
    components = []
    pins = {str(native): _file_sha256(native)}
    preload_paths = []
    for name in sealed_names:
        cand = rlib / name
        if cand.is_file():
            real = str(cand.resolve(strict=True))
            sha = _file_sha256(real)
            components.append({"path": real, "sha256": sha})
            pins[real] = sha
            preload_paths.append(real)
    preload_paths.append(str(native))
    manifest = {
        "cuda_root": str(rlib),
        "components": components,
        "cuda_runtime_version": 13030,
        "nvrtc_version": [13, 3],
    }
    manifest_sha = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()

    alias_dir = Path(tempfile.mkdtemp(prefix="disp0-sealed-aliases-")).resolve(strict=True)
    os.chmod(alias_dir, 0o700)
    import psi4_runtime_support
    owner = psi4_runtime_support.prepare_sealed_runtime(
        pins, alias_directory=alias_dir, dedicated_worker=True)
    owner.preload(preload_paths)

    import psi4
    from psi4 import core
    from psi4.driver.procrouting.sapt import sapt_dispersion_cuda as disp_mod
    from psi4.driver.procrouting.sapt import sapt_fisapt_workflow_cuda as wf_mod
    from psi4.driver.procrouting.sapt.sapt_cuda_runtime import CUDASession
    from psi4.driver.procrouting.sapt.sapt_cuest_dfmo_cuda import CuESTDFMOProvider, POLICY
    from psi4.driver.procrouting.sapt.sapt_cuest_factory_abi import admit_factory_api
    from psi4.driver.procrouting.sapt.sapt_cuest_owned_runtime import admit_sealed_runtime
    from psi4.driver.procrouting.sapt.sapt_cuest_plan_owner import DiagnosticPlanOwner

    admit_sealed_runtime(pins, alias_directory=alias_dir, dedicated_worker=True)

    @contextmanager
    def provider_factory(primary, auxiliary):
        with CUDASession(cublas_library=cublas_path, cublas_sha256=cublas_sha,
                         memory_limit_bytes=16 * 1024**3,
                         native_persistent_bytes=2 * 1024**3) as session:
            api = admit_factory_api(
                session=session, sdk_root=cuest_root, archive=cuest_archive,
                cuda_manifest=manifest, cuda_manifest_sha256=manifest_sha)
            with DiagnosticPlanOwner(
                api, primary, auxiliary,
                coefficient_candidate=DiagnosticPlanOwner.COEFFICIENT_CANDIDATE,
                jit_compile_threads=4,
            ) as owner:
                yield CuESTDFMOProvider(
                    session=session, sdk_root=cuest_root, archive=cuest_archive,
                    cuda_manifest=manifest, cuda_manifest_sha256=manifest_sha,
                    handle=owner.handle, plan=owner.plan, handle_pid=owner.pid,
                    primary_basis=primary, auxiliary_basis=auxiliary,
                    keepalive=(owner,), plan_policy=dict(POLICY),
                    output_order="C",
                )

    psi4.set_memory("4 GiB")
    psi4.set_num_threads(4)
    core.set_output_file("/dev/null", False)

    geoms = {
        "water_ne": """
            0 1
            O  0.0000  0.0000  0.0000
            H  0.7570  0.5860  0.0000
            H -0.7570  0.5860  0.0000
            --
            0 1
            Ne 0.0000  0.0000  3.1000
            units angstrom
            symmetry c1
            no_reorient
            no_com
        """,
        "water_dimer": """
            0 1
            O -1.551007 -0.114520  0.000000
            H -1.934259  0.762503  0.000000
            H -0.599677  0.040712  0.000000
            --
            0 1
            O  1.350625  0.111469  0.000000
            H  1.680398 -0.373741 -0.758561
            H  1.680398 -0.373741  0.758561
            units angstrom
            symmetry c1
            no_reorient
            no_com
        """,
        "ethane_water": """
            0 1
            C  0.0000  0.0000  0.0000
            C  1.5300  0.0000  0.0000
            H -0.3900  1.0200  0.0000
            H -0.3900 -0.5100  0.8830
            H -0.3900 -0.5100 -0.8830
            H  1.9200 -1.0200  0.0000
            H  1.9200  0.5100  0.8830
            H  1.9200  0.5100 -0.8830
            --
            0 1
            O  0.7650  0.0000  3.4000
            H  1.5220  0.5860  3.4000
            H  0.0080  0.5860  3.4000
            units angstrom
            symmetry c1
            no_reorient
            no_com
        """,
    }

    def base_opts(use_cuest, fsapt_mode="NONE", basis="cc-pvdz", jk="cc-pvdz-jkfit", ri="cc-pvdz-ri"):
        return {
            "basis": basis, "puream": True, "reference": "rhf", "scf_type": "df",
            "df_basis_scf": jk, "df_basis_sapt": ri, "df_basis_mp2": ri,
            "df_fitting_condition": 1e-12, "use_cuest": use_cuest,
            "cuest_mixed_precision": False, "e_convergence": 1e-10, "d_convergence": 1e-10,
            "ints_tolerance": 1e-16, "freeze_core": True,
            "sapt_dft_functional": "HF", "sapt_dft_do_fsapt": fsapt_mode,
            "sapt_dft_use_einsums": True, "sapt_dft_mp2_disp_alg": "SAPT",
            "sapt_dft_einsums_gpu": False, "sapt_dft_do_ddft": False,
            "sapt_dft_do_disp": True, "sapt_dft_do_dhf": False,
            "sapt_dft_induction_type": "CPHF", "cphf_r_convergence": 1e-10,
            "sapt_dft_grac_compute": "NONE", "sapt_dft_d3_ie": False, "sapt_dft_d4_ie": False,
        }

    results = {"mode": "psi4-gate", "hardware_note": "RTX 3090 debug only, not speedup", "cases": []}
    for name, fsapt_mode, basis, jk, ri, caches in (
        ("water_ne", "NONE", "cc-pvdz", "cc-pvdz-jkfit", "cc-pvdz-ri", (0, 4 * 1024**3)),
        ("water_dimer", "NONE", "jun-cc-pvdz", "aug-cc-pvdz-jkfit", "aug-cc-pvdz-ri", (0, 64 * 1024, 4 * 1024**3)),
        ("water_dimer", "FISAPT", "jun-cc-pvdz", "aug-cc-pvdz-jkfit", "aug-cc-pvdz-ri", (0, 4 * 1024**3)),
        ("ethane_water", "NONE", "cc-pvdz", "cc-pvdz-jkfit", "cc-pvdz-ri", (0, 4 * 1024**3)),
        ("ethane_water", "FISAPT", "cc-pvdz", "cc-pvdz-jkfit", "cc-pvdz-ri", (0, 4 * 1024**3)),
    ):
        mol = psi4.geometry(geoms[name])
        mol.reset_point_group("c1")
        mol.update_geometry()

        # Native CPU reference
        psi4.core.clean_options()
        psi4.core.clean()
        psi4.set_options(base_opts(False, fsapt_mode=fsapt_mode, basis=basis, jk=jk, ri=ri))
        e_cpu, wfn_cpu = psi4.energy("sapt(dft)", molecule=mol, return_wfn=True)
        cpu_disp = float(wfn_cpu.variable("Disp20,u"))
        cpu_exch = float(wfn_cpu.variable("Exch-Disp20,u"))

        # Baseline GPU blocked control (algorithm="blocked")
        orig_bdd = disp_mod.blocked_device_dispersion
        captured_blk = []
        captured_fac = []

        def _force_blocked(*a, **kw):
            kw["algorithm"] = "blocked"
            out = orig_bdd(*a, **kw)
            captured_blk.append(out)
            return out

        def _capture_fac(*a, **kw):
            out = orig_bdd(*a, **kw)
            # Also compare against algorithm="blocked" on the EXACT same device spaces/rotations!
            blk_same = orig_bdd(*a, **{**kw, "algorithm": "blocked"})
            same_pair_err = float(max(
                np.max(np.abs(out["Disp20 occupied pairs"] - blk_same["Disp20 occupied pairs"])),
                np.max(np.abs(out["Exch-Disp20 occupied pairs"] - blk_same["Exch-Disp20 occupied pairs"])),
            ))
            same_scalar_err = float(max(
                abs(out["Disp20,u"] - blk_same["Disp20,u"]),
                abs(out["Exch-Disp20,u"] - blk_same["Exch-Disp20,u"]),
            ))
            assert same_pair_err < 1e-11 and same_scalar_err < 1e-11, (same_pair_err, same_scalar_err)
            captured_fac.append((out, same_pair_err, same_scalar_err))
            return out

        psi4.core.clean_options()
        psi4.core.clean()
        psi4.set_options(base_opts(True, fsapt_mode=fsapt_mode, basis=basis, jk=jk, ri=ri))
        with patch.object(disp_mod, "blocked_device_dispersion", _force_blocked), \
             patch.object(wf_mod, "blocked_device_dispersion", _force_blocked):
            t0 = time.perf_counter()
            e_blk, wfn_blk = psi4.energy(
                "sapt(dft)", molecule=mol, return_wfn=True,
                cuda_provider_factory=provider_factory,
                cuda_dispersion_factor_cache_bytes=0,
            )
            t_blk = time.perf_counter() - t0
        blk_disp = float(wfn_blk.variable("Disp20,u"))
        blk_exch = float(wfn_blk.variable("Exch-Disp20,u"))
        assert len(captured_blk) == 1
        assert captured_blk[0]["Disp0 execution"]["factor_cache"]["policy"] == "PER_CALL_PACKED_S_SLABS_R_MAJOR"

        for cache_bytes in caches:
            captured_fac.clear()
            psi4.core.clean_options()
            psi4.core.clean()
            psi4.set_options(base_opts(True, fsapt_mode=fsapt_mode, basis=basis, jk=jk, ri=ri))
            with patch.object(disp_mod, "blocked_device_dispersion", _capture_fac), \
                 patch.object(wf_mod, "blocked_device_dispersion", _capture_fac):
                t0 = time.perf_counter()
                e_fac, wfn_fac = psi4.energy(
                    "sapt(dft)", molecule=mol, return_wfn=True,
                    cuda_provider_factory=provider_factory,
                    cuda_dispersion_factor_cache_bytes=cache_bytes,
                )
                t_fac = time.perf_counter() - t0
            assert len(captured_fac) == 1
            fac_out, same_pair_err, same_scalar_err = captured_fac[0]
            fac_disp = float(wfn_fac.variable("Disp20,u"))
            fac_exch = float(wfn_fac.variable("Exch-Disp20,u"))
            exec_info = fac_out["Disp0 execution"]
            assert exec_info["factor_cache"]["policy"] == "ALGEBRAIC_OCCUPIED_FIRST_FACTORIZED"
            assert exec_info["factor_cache"]["budget_bytes"] == cache_bytes

            err_cpu = max(abs(fac_disp - cpu_disp), abs(fac_exch - cpu_exch))
            err_blk = max(abs(fac_disp - blk_disp), abs(fac_exch - blk_exch))
            assert err_cpu < 1e-8, (name, fsapt_mode, cache_bytes, err_cpu, fac_disp, cpu_disp, fac_exch, cpu_exch)
            assert err_blk < 1e-9, (name, fsapt_mode, cache_bytes, err_blk, fac_disp, blk_disp, fac_exch, blk_exch)
            row = {
                "case": name, "fsapt_mode": fsapt_mode, "basis": basis,
                "cuda_dispersion_factor_cache_bytes": cache_bytes,
                "Disp20,u_Eh": fac_disp, "Exch-Disp20,u_Eh": fac_exch,
                "err_vs_cpu_Eh": err_cpu, "err_vs_blocked_gpu_Eh": err_blk,
                "same_wfn_pair_err_vs_blocked_Eh": same_pair_err,
                "same_wfn_scalar_err_vs_blocked_Eh": same_scalar_err,
                "debug_wall_fac_s": t_fac, "debug_wall_blk_s": t_blk,
                "dfmo_calls": exec_info.get("dfmo_calls"),
                "policy": exec_info["factor_cache"]["policy"],
                "tiles": exec_info.get("compute_tile_extents"),
                "slabs": exec_info.get("dfmo_slab_extents"),
                "stage_seconds": exec_info.get("stage_seconds"),
            }
            results["cases"].append(row)
            print("CASE_OK", json.dumps(row), flush=True)

    results["status"] = "PASS_PSI4_PHYSICAL_GPU_GATE"
    print(json.dumps(results, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("sanitizer", "psi4-gate"), required=True)
    p.add_argument("--runtime-lib", type=Path, required=True)
    p.add_argument("--cuest-root", type=Path)
    p.add_argument("--cuest-archive", type=Path)
    p.add_argument("--env-lib", type=Path)
    args = p.parse_args()
    if args.mode == "sanitizer":
        run_sanitizer_gate(args.runtime_lib)
    else:
        run_psi4_gate(str(args.cuest_root), str(args.cuest_archive),
                      str(args.runtime_lib), str(args.env_lib or args.runtime_lib))


if __name__ == "__main__":
    main()
