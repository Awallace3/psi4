import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import protocol


class FakeGPU(np.ndarray):
    @property
    def device(self):
        return SimpleNamespace(id=0)


class ProtocolTests(unittest.TestCase):
    def record(self, engine):
        return dict(engine=engine, system="peptide", functional="pbe", nbf=2,
                    geometry_sha256="geometry", config_sha256=engine,
                    config=dict(basis="aug-cc-pvtz", overlap_eigenvalue_cutoff=1e-7))

    def test_spectrum_and_retained_rank_verified_from_arrays(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                for engine in ("psi4", "pyscf"):
                    path = Path(directory) / engine
                    path.mkdir()
                    os.chdir(path)
                    r = self.record(engine)
                    protocol.capture_overlap(np.diag([6e-8, 2.]), 1, r, True)
                    self.assertEqual(r["removed_overlap_directions"], 1)
                os.chdir(previous)
                paths = [Path(directory) / engine / "overlap-spectrum.npz"
                         for engine in ("psi4", "pyscf")]
                self.assertTrue(protocol.compare_overlap_exports(*paths)["verified"])
                with paths[1].open("wb") as stream:
                    np.savez(stream, eigenvalues=np.array([6e-8, 3.]))
                with self.assertRaisesRegex(ValueError, "spectra differ"):
                    protocol.compare_overlap_exports(*paths)
            finally:
                os.chdir(previous)

    def test_same_ao_count_wrong_retained_rank_fails(self):
        with self.assertRaisesRegex(RuntimeError, "prescribed overlap cutoff"):
            protocol.capture_overlap(np.diag([6e-8, 2.]), 2, self.record("pyscf"), True)

    def test_invalid_cutoff_fails(self):
        for cutoff in (float("nan"), -1, 0):
            with self.assertRaises(ValueError):
                protocol.overlap_cutoff({"overlap_eigenvalue_cutoff": cutoff})

    def test_stored_df_prebuilt_and_used_no_direct_or_host_fallback(self):
        built = []
        jk = SimpleNamespace(get_j=lambda *a: "direct", get_jk=lambda *a, **k: "stored")
        original = jk.get_j, jk.get_jk
        obj = SimpleNamespace(_cderi=None, _rsh_df={})
        def build():
            built.append(True)
            obj._cderi = {0: np.zeros((3, 4)).view(FakeGPU)}
        obj.build = build
        modules = {"cupy": SimpleNamespace(ndarray=FakeGPU, float64=np.float64),
                   "gpu4pyscf.df": SimpleNamespace(df_jk=jk)}
        with patch.dict("sys.modules", modules):
            r = {}
            with protocol.pyscf_df_policy(SimpleNamespace(with_df=obj),
                                          {"df_mode": "stored-df"}, r, lambda: None):
                self.assertEqual(built, [True])
                self.assertTrue(obj.use_gpu_memory)
                self.assertEqual(jk.get_jk(obj, None), "stored")
            self.assertTrue(r["stored_df_verified"])
            self.assertEqual(r["df_tensor_at_scf_entry"][0]["bytes"], 3*4*8)
            self.assertEqual((jk.get_j, jk.get_jk), original)
            obj._cderi = {0: np.zeros((3, 4))}
            with self.assertRaisesRegex(RuntimeError, "no host spill"):
                protocol.stored_tensor_info(obj, modules["cupy"])
            with self.assertRaisesRegex(RuntimeError, "Integral-direct"):
                with protocol.pyscf_df_policy(SimpleNamespace(with_df=obj),
                                              {"df_mode": "stored-df"}, {}, lambda: None):
                    jk.get_j(obj, None)

    def test_missing_stored_contractions_refused_and_hooks_restored(self):
        jk = SimpleNamespace(get_j=lambda *a: None, get_jk=lambda *a: None)
        original = jk.get_j, jk.get_jk
        obj = SimpleNamespace(_rsh_df={}, build=lambda: None,
                              _cderi={0: np.zeros((2, 2)).view(FakeGPU)})
        modules = {"cupy": SimpleNamespace(ndarray=FakeGPU, float64=np.float64),
                   "gpu4pyscf.df": SimpleNamespace(df_jk=jk)}
        with patch.dict("sys.modules", modules), self.assertRaisesRegex(RuntimeError, "No stored-tensor"):
            with protocol.pyscf_df_policy(SimpleNamespace(with_df=obj),
                                          {"df_mode": "stored-df"}, {}, lambda: None):
                pass
        self.assertEqual((jk.get_j, jk.get_jk), original)

    def test_stored_tensor_accepts_installed_multi_gpu_list_but_rejects_host_arrays(self):
        # GPU4PySCF1.8.1: _cholesky_eri -> multi_gpu.run -> [func(...)]
        cp = SimpleNamespace(ndarray=FakeGPU, float64=np.float64)
        gpu = np.zeros((3, 4)).view(FakeGPU)
        self.assertEqual(protocol.stored_tensor_info(SimpleNamespace(_cderi=[gpu]), cp),
                         [dict(device=0, shape=[3, 4], bytes=96)])
        with self.assertRaisesRegex(RuntimeError, "no host spill"):
            protocol.stored_tensor_info(SimpleNamespace(_cderi=[np.zeros((3, 4))]), cp)
        with self.assertRaisesRegex(RuntimeError, "missing"):
            protocol.stored_tensor_info(SimpleNamespace(_cderi=[]), cp)

    def test_lazy_rsh_tensor_built_with_requested_omega_before_contraction(self):
        builds = []
        def contract(dfobj, dm, hermi=0, with_j=True, with_k=True, omega=None):
            self.assertIsNotNone(dfobj._cderi)
            return omega
        jk = SimpleNamespace(get_j=lambda *a: None, get_jk=contract)
        base = SimpleNamespace(_cderi={0: np.zeros((2, 2)).view(FakeGPU)},
                               _rsh_df={}, build=lambda: None)
        lazy = SimpleNamespace(_cderi=None)
        def build(omega=None):
            self.assertTrue(lazy.use_gpu_memory)
            builds.append(omega)
            lazy._cderi = {0: np.zeros((2, 2)).view(FakeGPU)}
        lazy.build = build
        base._rsh_df["0.3"] = lazy
        modules = {"cupy": SimpleNamespace(ndarray=FakeGPU, float64=np.float64),
                   "gpu4pyscf.df": SimpleNamespace(df_jk=jk)}
        with patch.dict("sys.modules", modules):
            record = {}
            with protocol.pyscf_df_policy(SimpleNamespace(with_df=base),
                                          {"df_mode": "stored-df"}, record, lambda: None):
                self.assertEqual(jk.get_jk(lazy, None, 1, False, True, .3), .3)
            self.assertEqual(builds, [.3])
            self.assertEqual(record["lazy_df_builds"], [{"omega": .3}])
            self.assertTrue(record["stored_df_verified"])

    def test_native_strategy_is_not_prebuilt(self):
        jk = SimpleNamespace(get_j=lambda *a: "direct", get_jk=lambda *a: "stored")
        modules = {"cupy": SimpleNamespace(), "gpu4pyscf.df": SimpleNamespace(df_jk=jk)}
        obj = SimpleNamespace(build=lambda: self.fail("Native route must not prebuild"))
        with patch.dict("sys.modules", modules):
            r = {}
            with protocol.pyscf_df_policy(SimpleNamespace(with_df=obj),
                                          {"df_mode": "native-strategy"}, r, lambda: None):
                self.assertEqual(jk.get_j(obj, None), "direct")
            self.assertFalse(r["stored_df_verified"])
            self.assertEqual(r["df_route_calls"]["integral_direct_df_j_calls"], 1)


if __name__ == "__main__":
    unittest.main()
