import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import hashlib

import grids


@unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy required for mesh audit")
class GridTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np = np
        self.coords = np.array([[0., 1., 2.], [2., 3., 4.], [2., 3., 4.]])
        self.weights = np.array([1., -.5, -.5])
        self.identity = dict(engine="psi4", system="peptide", functional="pbe",
                             geometry_sha256="geometry", basis="cc-pvdz",
                             grid_mode="native-default", config_sha256="config",
                             build_sha256="build")

    def fingerprint(self, coords=None, weights=None):
        return grids.grid_fingerprint(self.coords if coords is None else coords,
                                      self.weights if weights is None else weights)

    def test_order_and_padding_do_not_change_nonzero_multiset(self):
        coords = self.np.vstack((self.coords[::-1], [999, 999, 999]))
        weights = self.np.append(self.weights[::-1], 0)
        first, second = self.fingerprint(), self.fingerprint(coords, weights)
        self.assertEqual(first["nonzero_pairs_multiset_sha256"], second["nonzero_pairs_multiset_sha256"])
        self.assertNotEqual(first["ordered_pairs_sha256"], second["ordered_pairs_sha256"])
        self.assertEqual(second["stored_points"], 4)
        self.assertEqual(second["nonzero_weight_points"], 3)
        self.assertEqual(second["negative_weight_points"], 2)

    def test_duplicates_retained(self):
        single = self.fingerprint(self.coords[:2], self.weights[:2])
        self.assertNotEqual(single["nonzero_pairs_multiset_sha256"],
                            self.fingerprint()["nonzero_pairs_multiset_sha256"])

    def test_equal_counts_different_weights_are_not_equal(self):
        second = self.fingerprint(weights=self.weights * 2)
        self.assertNotEqual(second["nonzero_pairs_multiset_sha256"],
                            self.fingerprint()["nonzero_pairs_multiset_sha256"])

    def test_signed_zeros_have_equal_quadrature_meaning(self):
        second = self.coords.copy()
        second[0, 0] = -0.0
        self.assertEqual(self.fingerprint(second), self.fingerprint())

    def test_malformed_and_nonfinite_grids_rejected(self):
        with self.assertRaises(ValueError):
            self.fingerprint(coords=self.coords[:, :2])
        with self.assertRaises(ValueError):
            self.fingerprint(weights=[1, float("nan"), 3])
        with self.assertRaises(ValueError):
            self.fingerprint(weights=[0, 0, 0])

    def test_export_validation_and_immutability(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grid.npz"
            info = grids.export_grid(path, self.coords, self.weights, self.identity,
                                     phase="xc-entry", settings={"level": 3})
            self.assertTrue(info["diagnostic_only"])
            self.assertEqual(info["phase"], "xc-entry")
            with self.assertRaises(FileExistsError):
                grids.export_grid(path, self.coords, self.weights, self.identity,
                                  phase="xc-entry", settings={"level": 3})
            self.assertTrue(grids.compare_exports(path, path)["identical_nonzero_quadrature"])
            altered = json.loads(path.with_suffix(".json").read_text())
            altered["stored_points"] = 99
            path.with_suffix(".json").write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                grids.compare_exports(path, path)

    def test_geometry_identity_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            left, right = [Path(directory) / name for name in ("left.npz", "right.npz")]
            grids.export_grid(left, self.coords, self.weights, self.identity,
                              phase="xc-entry", settings={"level": 3})
            grids.export_grid(right, self.coords, self.weights,
                              dict(self.identity, geometry_sha256="another"),
                              phase="xc-entry", settings={"level": 3})
            with self.assertRaisesRegex(ValueError, "geometry_sha256"):
                grids.compare_exports(left, right)

    def test_both_missing_required_identity_fields_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grid.npz"
            grids.export_grid(path, self.coords, self.weights, self.identity,
                              phase="xc-entry", settings={"level": 3})
            info = json.loads(path.with_suffix(".json").read_text())
            del info["identity"]["geometry_sha256"]
            path.with_suffix(".json").write_text(json.dumps(info))
            with self.assertRaisesRegex(ValueError, "geometry_sha256"):
                grids.compare_exports(path, path)

    def test_nominal_grid_cannot_select_smaller_native_default(self):
        with tempfile.TemporaryDirectory() as directory:
            left, right = [Path(directory) / name for name in ("left.npz", "right.npz")]
            grids.export_grid(left, self.coords, self.weights, self.identity,
                              phase="xc-entry", settings={"level": 3})
            grids.export_grid(right, self.coords[:2], self.weights[:2],
                              dict(self.identity, engine="pyscf", grid_mode="nominal"),
                              phase="xc-entry", settings={"level": 3})
            compared = grids.compare_exports(left, right)
            self.assertEqual(compared["smaller_nonzero_grid"], "pyscf")
            self.assertFalse(compared["native_default_comparison"])
            self.assertIsNone(compared["smaller_native_default_grid"])

    def test_basis_control_requires_explicit_basis_difference_permission(self):
        with tempfile.TemporaryDirectory() as directory:
            left, right = [Path(directory) / name for name in ("left.npz", "right.npz")]
            for path, basis in [(left, "cc-pvtz"), (right, "aug-cc-pvtz")]:
                grids.export_grid(path, self.coords, self.weights, dict(self.identity, basis=basis),
                                  phase="xc-entry", settings={"level": 3})
            with self.assertRaisesRegex(ValueError, "basis"):
                grids.compare_exports(left, right)
            comparison = grids.compare_exports(left, right, allow_basis_difference=True)
            self.assertTrue(comparison["identical_nonzero_quadrature"])

    def test_alignment_padding_preserves_weights_and_duplicate_points(self):
        coords, weights, padding = grids.prepare_fixed_arrays(self.coords, self.weights, 128)
        self.assertEqual(padding, 125)
        self.assertEqual(weights.size, 128)
        self.assertTrue(self.np.all(weights[3:] == 0))
        self.assertEqual(self.fingerprint(coords, weights)["nonzero_pairs_multiset_sha256"],
                         self.fingerprint()["nonzero_pairs_multiset_sha256"])

    def test_fixed_installation_disables_pruning_and_prevents_regeneration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grid.npz"
            grids.export_grid(path, self.coords, self.weights, self.identity,
                              phase="xc-entry", settings={"level": 3})
            config = dict(shared_grid=str(path),
                          shared_grid_source_points=3,
                          shared_grid_fingerprint=self.fingerprint()["nonzero_pairs_multiset_sha256"],
                          shared_grid_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          shared_grid_metadata_sha256=hashlib.sha256(path.with_suffix(".json").read_bytes()).hexdigest())
            grid = SimpleNamespace(alignment=128)
            grid.reset = lambda: None
            mf = SimpleNamespace(grids=grid, small_rho_cutoff=1e-7)
            fake_cupy = SimpleNamespace(asarray=self.np.asarray, asnumpy=self.np.asarray, float64=self.np.float64)
            with patch.dict("sys.modules", cupy=fake_cupy), patch.object(
                    grids, "load_grid", side_effect=AssertionError("Do not reopen verified snapshots")):
                grids.install_pyscf_shared_grid(mf, config, self.identity)
            self.assertEqual(mf.small_rho_cutoff, 0)
            self.assertEqual(grid.weights.size, 128)
            self.assertEqual(grid._shared_grid_padding, 125)
            with self.assertRaisesRegex(RuntimeError, "regenerate"):
                grid.build()
            config["shared_grid_file_sha256"] = "changed"
            with patch.dict("sys.modules", cupy=fake_cupy), self.assertRaisesRegex(ValueError, "changed"):
                grids.install_pyscf_shared_grid(mf, config, self.identity)


class GridSettingsTests(unittest.TestCase):
    def test_native_defaults_are_untouched(self):
        mf = SimpleNamespace(grids=SimpleNamespace(atom_grid={}, prune="native"), small_rho_cutoff=1e-7)
        grids.apply_pyscf_grid_settings(mf, {"grid_mode": "native-default"})
        self.assertEqual(mf.grids.prune, "native")
        self.assertEqual(mf.grids.atom_grid, {})
        self.assertEqual(mf.small_rho_cutoff, 1e-7)

    def test_unknown_mode_fails(self):
        with self.assertRaises(ValueError):
            grids.apply_pyscf_grid_settings(None, {"grid_mode": "same-count-is-enough"})
