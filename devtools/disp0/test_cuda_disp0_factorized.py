"""CPU fake-C-ABI tests for occupied-first factorized disp0 and batched-GEMM slices.

Not actual CUDA/cuEST GPU execution proof; verifies algebraic equivalence,
tiling, packed-S reuse, non-idempotent fallback, memory cleanup, and BLAS
descriptor contracts on the host before physical GPU gates.
"""
import ctypes as ct
import importlib
import types
import unittest
from unittest.mock import patch

import numpy as np

from test_cuda_backend import PACKAGE, Kernels, host, runtime, oracle, backend
from test_cuda_dfmo_device import DeviceSession
from test_cuda_fisapt import factor_reference, fisapt
from test_cuda_hf_setup import reference as hf_reference
import test_cuda_dfmo_transport as transport

setup = importlib.import_module(PACKAGE + ".sapt_dispersion_cuda_setup")
dfmo = transport.dfmo


def _make_idempotent_hf(seed, *, nao=9, oa=3, ob=2, nr=5, ns=4, symmetric_s=True):
    rng = np.random.default_rng(seed)
    d = {
        name: rng.normal(size=(nao, nao)) * 0.1
        for name in ("S", "D_A", "P_A", "V_A", "J_A", "K_A", "D_B", "P_B", "V_B", "J_B", "K_B", "K_O")
    }
    if symmetric_s:
        s_mat = rng.normal(size=(nao, nao)) * 0.1
        d["S"] = 0.5 * (s_mat + s_mat.T) + np.eye(nao)
    else:
        d["S"] = rng.normal(size=(nao, nao)) * 0.1 + np.eye(nao)
    d["Cocc_A"] = rng.normal(size=(nao, oa)) * 0.25
    d["Cocc_B"] = rng.normal(size=(nao, ob)) * 0.25
    d["Cvir_A"] = rng.normal(size=(nao, nr)) * 0.25
    d["Cvir_B"] = rng.normal(size=(nao, ns)) * 0.25
    d["D_A"] = d["Cocc_A"] @ d["Cocc_A"].T
    d["D_B"] = d["Cocc_B"] @ d["Cocc_B"].T
    e = {
        "a": np.linspace(-1.4, -0.5, oa),
        "b": np.linspace(-1.3, -0.6, ob),
        "r": np.linspace(0.15, 0.75, nr),
        "s": np.linspace(0.20, 0.85, ns),
    }
    return d, e


def _register_factorized_kernels(lib):
    specs = {
        "matrix_window": [ct.c_longlong] * 4 + [ct.c_int] + [ct.c_void_p] * 2,
        "linear_combination": [ct.c_longlong] * 3 + [ct.c_void_p] * 3,
        "reduce_components": [ct.c_longlong] + [ct.c_void_p] * 3,
        "matrix_error": [ct.c_longlong] + [ct.c_void_p] * 2,
        "make_amplitudes": [ct.c_longlong] * 6 + [ct.c_void_p] * 6,
        "rotation_error": [ct.c_longlong] + [ct.c_void_p] * 2,
        "reduce_localized_dispersion": [ct.c_longlong] * 4 + [ct.c_void_p] * 6,
        "transpose_factor_slab": [ct.c_longlong] * 8 + [ct.c_void_p] * 2,
        "pack_factor_tails": [ct.c_longlong] * 5 + [ct.c_void_p] * 3,
        "fill_active_rotation": [ct.c_longlong] * 3 + [ct.c_void_p] * 2,
        "concat_columns": [ct.c_longlong] * 3 + [ct.c_void_p] * 3,
    }
    for name, argtypes in specs.items():
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = None


class FactorizedDisp0Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Kernels.setUpClass.__func__(cls)
        _register_factorized_kernels(cls.lib)
        DeviceSession.kernels = cls.lib

    @classmethod
    def tearDownClass(cls):
        DeviceSession.kernels = None
        Kernels.tearDownClass.__func__(cls)

    def _fixture(self, nao=9, naux=11, seed=31):
        rng = np.random.default_rng(seed)
        ao = rng.normal(size=(naux, nao, nao)) * 0.2
        ao = 0.5 * (ao + np.swapaxes(ao, 1, 2))
        def basis(size):
            return types.SimpleNamespace(
                nbf=lambda: size, nshell=lambda: 1,
                shell=lambda i: types.SimpleNamespace(is_pure=lambda: True, is_cartesian=lambda: False),
            )
        with patch.object(transport, "Session", DeviceSession):
            p, s, n, _, _ = transport.Transport().fixture(
                primary_basis=basis(nao), auxiliary_basis=basis(naux),
            )
        n.ao = ao
        audit = patch.object(
            dfmo.identity, "audit_maps",
            return_value={"cuda_mapped_sha256": {s.blas_path: s.blas_sha256}},
        )
        audit.start()
        self.addCleanup(audit.stop)
        p.prepare_device_compute()
        self.addCleanup(p.close)
        return p, s, n, ao

    def test_factorized_matches_oracle_and_blocked_canonical_and_localized(self):
        p, s, n, ao = self._fixture(nao=9, naux=11, seed=41)
        for symmetric_s in (True, False):
            for fa, fb in ((0, 0), (1, 0), (2, 1)):
                d, e = _make_idempotent_hf(
                    100 + 10 * fa + fb + (0 if symmetric_s else 50),
                    nao=9, oa=4, ob=3, nr=6, ns=5, symmetric_s=symmetric_s,
                )
                ref_sp, ref_c = hf_reference(d, fa, fb)
                ref_e = {"a": e["a"][fa:], "b": e["b"][fb:], "r": e["r"], "s": e["s"]}
                A, B = ref_e["a"].size, ref_e["b"].size
                rng = np.random.default_rng(700 + fa * 7 + fb)
                ua = np.linalg.qr(rng.normal(size=(A, A)))[0]
                ub = np.linalg.qr(rng.normal(size=(B, B)))[0]
                canon_ref = oracle(ao, ref_sp, ref_c, ref_e)
                loc_ref = factor_reference(ao, ref_sp, ref_c, ref_e, ua, ub)

                for localized in (False, True):
                    expected = loc_ref if localized else canon_ref
                    with s.scope():
                        spaces, corr, eps = setup.prepare_hf(s, d, e, frozen_a=fa, frozen_b=fb)
                        rot = (s.upload(ua), s.upload(ub)) if localized else None
                        before = n.computed
                        with patch.object(s, "upload", side_effect=AssertionError("unexpected host upload")):
                            fac = backend.blocked_device_dispersion(
                                p, spaces, corr, eps, block_size=2, rotations=rot, algorithm="factorized",
                            )
                            dfmo_calls_fac = n.computed - before
                            blk = backend.blocked_device_dispersion(
                                p, spaces, corr, eps, block_size=2, rotations=rot, algorithm="blocked",
                            )
                        self.assertEqual(dfmo_calls_fac, 1)
                        self.assertEqual(fac["Disp0 execution"]["dfmo_calls"], 1)
                        self.assertEqual(
                            fac["Disp0 execution"]["factor_cache"]["policy"],
                            "ALGEBRAIC_OCCUPIED_FIRST_FACTORIZED",
                        )
                        for key, target in (
                            ("Disp20 occupied pairs", expected[0]),
                            ("Exch-Disp20 occupied pairs", expected[1]),
                        ):
                            np.testing.assert_allclose(fac[key], target, rtol=1e-11, atol=1e-12)
                            np.testing.assert_allclose(fac[key], blk[key], rtol=1e-11, atol=1e-12)
                        np.testing.assert_allclose(
                            [fac["Disp20,u"], fac["Exch-Disp20,u"], fac["Disp0,u"]],
                            [expected[0].sum(), expected[1].sum(), expected[0].sum() + expected[1].sum()],
                            rtol=1e-11, atol=1e-12,
                        )
                    self.assertEqual(s.live_bytes, 0)

    def test_factorized_edge_size_one_and_tiled_multi_slab_modes(self):
        # Single-orbital edge case (A=1, B=1, nr=1, ns=1)
        p1, s1, _, ao1 = self._fixture(nao=5, naux=7, seed=55)
        d1, e1 = _make_idempotent_hf(56, nao=5, oa=1, ob=1, nr=1, ns=1)
        ref_sp1, ref_c1 = hf_reference(d1, 0, 0)
        expected1 = oracle(ao1, ref_sp1, ref_c1, e1)
        with s1.scope():
            spaces, corr, eps = setup.prepare_hf(s1, d1, e1)
            res = backend.blocked_device_dispersion(p1, spaces, corr, eps, block_size=1, algorithm="factorized")
            np.testing.assert_allclose(res["Disp20 occupied pairs"], expected1[0], rtol=1e-11, atol=1e-12)
            np.testing.assert_allclose(res["Exch-Disp20 occupied pairs"], expected1[1], rtol=1e-11, atol=1e-12)
        self.assertEqual(s1.live_bytes, 0)

        # Multi-slab tiled modes: tight factor_cache_bytes exercises both reuse_s=False and reuse_s=True,
        # plus packed-S tile caching when n_r_tiles > 1.
        p2, s2, n2, ao2 = self._fixture(nao=9, naux=13, seed=61)
        d2, e2 = _make_idempotent_hf(62, nao=9, oa=3, ob=3, nr=7, ns=6)
        ref_sp2, ref_c2 = hf_reference(d2, 1, 0)
        ref_e2 = {"a": e2["a"][1:], "b": e2["b"], "r": e2["r"], "s": e2["s"]}
        rng = np.random.default_rng(63)
        ua = np.linalg.qr(rng.normal(size=(2, 2)))[0]
        ub = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        loc_ref = factor_reference(ao2, ref_sp2, ref_c2, ref_e2, ua, ub)
        canon_ref = oracle(ao2, ref_sp2, ref_c2, ref_e2)

        # bytes_oo = 8 * 13 * 5 * 6 = 3120; per_r = 8*13*6 = 624; per_s = 8*13*5 = 520.
        # Budget 4500 forces tiny slabs (reuse_s=False); budget 7600 fits all S + partial R (reuse_s=True).
        for budget in (4500, 7600, 100000):
            for localized in (False, True):
                expected = loc_ref if localized else canon_ref
                with s2.scope():
                    spaces, corr, eps = setup.prepare_hf(s2, d2, e2, frozen_a=1, frozen_b=0)
                    rot = (s2.upload(ua), s2.upload(ub)) if localized else None
                    # Also force small compute tiles via tight avail in _choose_tile_extents for multi-tile S cache test.
                    blk_sz = 1 if budget == 4500 else 2
                    with patch.object(backend, "_choose_tile_extents", return_value=(2, 2, 1024)):
                        res = backend.blocked_device_dispersion(
                            p2, spaces, corr, eps, block_size=blk_sz, rotations=rot,
                            factor_cache_bytes=budget, algorithm="factorized",
                        )
                    info = res["Disp0 execution"]
                    if budget == 4500:
                        self.assertGreater(info["dfmo_calls"], 2)
                        self.assertEqual(info["factor_cache"]["cached_slabs"], 0)
                    elif budget == 7600:
                        self.assertGreater(info["dfmo_calls"], 1)
                        self.assertEqual(info["factor_cache"]["cached_slabs"], 1)
                        self.assertGreater(info["factor_cache"]["hits"], 0)
                    else:
                        self.assertEqual(info["dfmo_calls"], 1)
                    np.testing.assert_allclose(res["Disp20 occupied pairs"], expected[0], rtol=1e-11, atol=1e-12)
                    np.testing.assert_allclose(res["Exch-Disp20 occupied pairs"], expected[1], rtol=1e-11, atol=1e-12)
                self.assertEqual(s2.live_bytes, 0)

    def test_non_idempotent_hf_falls_back_in_auto_and_rejects_explicit_factorized(self):
        p, s, _, ao = self._fixture(nao=9, naux=11, seed=71)
        d, e = _make_idempotent_hf(72, nao=9, oa=3, ob=2, nr=4, ns=5)
        d["D_A"] = d["D_A"] + 0.05 * np.eye(9)  # Break D_A = Cocc_A @ Cocc_A^T
        ref_sp, ref_c = hf_reference(d, 0, 0)
        expected = oracle(ao, ref_sp, ref_c, e)
        with s.scope():
            spaces, corr, eps = setup.prepare_hf(s, d, e)
            auto_res = backend.blocked_device_dispersion(p, spaces, corr, eps, block_size=2, algorithm="auto")
            self.assertEqual(auto_res["Disp0 execution"]["factor_cache"]["policy"], "PER_CALL_PACKED_S_SLABS_R_MAJOR")
            np.testing.assert_allclose(auto_res["Disp20 occupied pairs"], expected[0], rtol=1e-11, atol=1e-12)
            with self.assertRaisesRegex(ValueError, "idempotent"):
                backend.blocked_device_dispersion(p, spaces, corr, eps, block_size=2, algorithm="factorized")
            with self.assertRaisesRegex(ValueError, "algorithm"):
                backend.blocked_device_dispersion(p, spaces, corr, eps, block_size=2, algorithm="bogus")
        self.assertEqual(s.live_bytes, 0)

    def test_gemm_and_batched_slices_scalars_out_descriptors_and_guards(self):
        p, s, _, _ = self._fixture(nao=6, naux=7, seed=81)
        calls = []
        s._blas = lambda *args: calls.append(args)
        s.handle = ct.c_void_p(456)
        with s.scope():
            # 2D gemm with alpha/beta/out
            left = s.empty((4, 6))
            right = s.empty((6, 5))
            out = s.empty((4, 5))
            res = runtime.CUDASession.gemm(s, left, right, alpha=-2.0, beta=0.5, out=out)
            self.assertIs(res, out)
            args = calls.pop()
            self.assertEqual(args[:7], ("cublasDgemm_v2", s.handle, 0, 0, 5, 4, 6))
            self.assertEqual(ct.cast(args[7], ct.POINTER(ct.c_double))[0], -2.0)
            self.assertEqual(ct.cast(args[12], ct.POINTER(ct.c_double))[0], 0.5)
            self.assertEqual(args[13], out.pointer)

            # Batched gemm with left_cols, right_cols, transpose_left, transpose_right, beta=1.0
            a3 = s.empty((7, 5, 10))
            b3 = s.empty((7, 4, 9))
            c3 = s.empty((7, 3, 4))
            # left_cols=(2,3) -> (5,3)^T = (3,5); right_cols=(1,5) -> (4,5)^T = (5,4); out=(7,3,4)
            res_b = runtime.CUDASession.gemm_batched(
                s, a3, b3, transpose_left=True, transpose_right=True,
                alpha=1.5, beta=1.0, left_cols=(2, 3), right_cols=(1, 5), out=c3,
            )
            self.assertIs(res_b, c3)
            bargs = calls.pop()
            self.assertEqual(bargs[:7], ("cublasDgemmStridedBatched", s.handle, 1, 1, 4, 3, 5))
            self.assertEqual(ct.cast(bargs[7], ct.POINTER(ct.c_double))[0], 1.5)
            self.assertEqual(bargs[8], b3.pointer + 8 * 1)
            self.assertEqual(bargs[9], 9)
            self.assertEqual(bargs[10], 4 * 9)
            self.assertEqual(bargs[11], a3.pointer + 8 * 2)
            self.assertEqual(bargs[12], 10)
            self.assertEqual(bargs[13], 5 * 10)
            self.assertEqual(ct.cast(bargs[14], ct.POINTER(ct.c_double))[0], 1.0)
            self.assertEqual(bargs[15], c3.pointer)
            self.assertEqual(bargs[16], 4)
            self.assertEqual(bargs[17], 3 * 4)
            self.assertEqual(bargs[18], 7)

            # Validation guards: nonfinite/bool scalars, out-of-bounds slices, aliasing output
            for bad_alpha in (float("nan"), float("inf"), True, "1.0"):
                with self.assertRaises(ValueError):
                    runtime.CUDASession.gemm(s, left, right, alpha=bad_alpha)
                with self.assertRaises(ValueError):
                    runtime.CUDASession.gemm_batched(s, a3, b3, alpha=bad_alpha)
            for bad_slice in ((-1, 2), (0, 0), (8, 3), (True, 2), (1,)):
                with self.assertRaises(ValueError):
                    runtime.CUDASession.gemm_batched(s, a3, b3, transpose_left=True, left_cols=bad_slice)
            with self.assertRaisesRegex(ValueError, "overlap"):
                runtime.CUDASession.gemm(s, left, left, transpose_right=True, out=left.view((4, 4)))
        self.assertEqual(s.live_bytes, 0)

    def test_occupied_transform_zero_alloc_workspace_matches_einsum(self):
        rng = np.random.default_rng(99)
        R, A, S, B = 4, 3, 5, 2
        v = rng.normal(size=(R, A, S, B))
        ua = np.linalg.qr(rng.normal(size=(A, A)))[0]
        ub = np.linalg.qr(rng.normal(size=(B, B)))[0]
        expected = np.einsum("al,rasb,bk->rlsk", ua, v, ub)
        _, s, _, _ = self._fixture(nao=6, naux=7, seed=82)
        with s.scope():
            dv = s.upload(v.reshape(R * A, S * B))
            dua, dub = s.upload(ua), s.upload(ub)
            scratch = s.empty((R * A * S * B,))
            out = s.empty((R * A, S * B))
            allocs_before = len(s.allocations)
            res = fisapt.occupied_transform(s, dv, dua, dub, R, S, scratch=scratch, out=out)
            self.assertIs(res, out)
            self.assertEqual(len(s.allocations), allocs_before)
            np.testing.assert_allclose(host(out).reshape(R, A, S, B), expected, rtol=1e-12, atol=1e-13)
        self.assertEqual(s.live_bytes, 0)

    def test_nvrtc_core_lazy_extra_and_in_process_ptx_cache_and_unload(self):
        kernels_mod = importlib.import_module(PACKAGE + ".sapt_dispersion_cuda_kernels")
        self.assertTrue(set(kernels_mod.CORE_KERNEL_NAMES).isdisjoint(kernels_mod.EXTRA_KERNEL_NAMES))
        compile_calls = []
        loaded = []
        unloaded = []

        class FakeNV:
            def nvrtcVersion(self):
                return (0, 13, 3)

            def nvrtcCreateProgram(self, src, name, *_):
                compile_calls.append(name)
                return (0, len(compile_calls) + 100)

            def nvrtcCompileProgram(self, prog, *_):
                return (0,)

            def nvrtcGetPTXSize(self, prog):
                return (0, 16)

            def nvrtcGetPTX(self, prog, buf):
                buf[:] = f"PTX_{prog:012d}".encode()
                return (0,)

            def nvrtcDestroyProgram(self, prog):
                return (0,)

        class FakeDR:
            def cuModuleLoadData(self, ptx):
                mod = len(loaded) + 1
                loaded.append((mod, ptx))
                return (0, mod)

            def cuModuleGetFunction(self, mod, name):
                return (0, (mod, name.decode()))

            def cuModuleUnload(self, mod):
                unloaded.append(mod)
                return (0,)

            def cuLaunchKernel(self, fn, *_):
                return (0,)

        def make_fake_session():
            s = types.SimpleNamespace(
                nvrtc=FakeNV(), driver=FakeDR(),
                runtime=types.SimpleNamespace(cudaStreamDestroy=lambda stream: (0,)),
                arch="compute_90",
                functions={}, module=None, _extra_module=None,
                handle=None, stream=1, allocations=[], native_owners=[],
                kernel_launches={}, closed=False, failed=False, cleanup_complete=False,
                identity_owner=None, rt=runtime.CUDASession.rt,
                check=lambda: None, synchronize_device=lambda: None,
            )
            s._compile_unit = types.MethodType(runtime.CUDASession._compile_unit, s)
            s._compile = types.MethodType(runtime.CUDASession._compile, s)
            s._compile_extra = types.MethodType(runtime.CUDASession._compile_extra, s)
            return s

        with patch.dict(runtime._NVRTC_PTX_CACHE, {}, clear=True):
            s1 = make_fake_session()
            s1._compile()
            self.assertEqual(compile_calls, [b"sapt0_blocks.cu"])
            self.assertIn("reduce_dispersion", s1.functions)
            self.assertNotIn("ibo_sweep", s1.functions)
            # Second session reuses content-derived PTX without invoking NVRTC again.
            s2 = make_fake_session()
            s2._compile()
            self.assertEqual(compile_calls, [b"sapt0_blocks.cu"])
            self.assertEqual(s1.ptx_sha256, s2.ptx_sha256)
            # Launching an F-SAPT extra kernel lazily compiles EXTRA_CUDA_SOURCE once.
            runtime.CUDASession.launch(s1, "ibo_sweep", 1, 128, [])
            self.assertEqual(compile_calls, [b"sapt0_blocks.cu", b"sapt0_fisapt.cu"])
            self.assertIn("ibo_sweep", s1.functions)
            runtime.CUDASession.launch(s2, "ibo_sweep", 1, 128, [])
            self.assertEqual(compile_calls, [b"sapt0_blocks.cu", b"sapt0_fisapt.cu"])
            # Closing unloads both extra and core modules cleanly.
            runtime.CUDASession.close(s1)
            runtime.CUDASession.close(s2)
            self.assertEqual(set(unloaded), {mod for mod, _ in loaded})

    def test_plan_owner_batches_shell_map_admission_per_basis(self):
        import test_cuest_plan_owner as plan_tests
        owner, session, native, primary, auxiliary = plan_tests.PlanOwner().fixture()
        maps = {"cuda_mapped_sha256": {session.blas_path: session.blas_sha256}}
        with patch.object(plan_tests.abi, "audit_factory_maps", return_value=maps) as audit:
            with owner:
                phases = owner.report()["host_phase_timings"]["phases"]
                self.assertEqual(phases["primary shells.admission"]["calls"], 1)
                self.assertEqual(phases["auxiliary shells.admission"]["calls"], 1)
                self.assertEqual(phases["primary shells.native_create"]["calls"], len(owner.primary_snapshot.shells))
                self.assertEqual(phases["auxiliary shells.native_create"]["calls"], len(owner.auxiliary_snapshot.shells))
        self.assertGreater(audit.call_count, 0)
        self.assertEqual(session.live_bytes, 0)


if __name__ == "__main__":
    unittest.main()
