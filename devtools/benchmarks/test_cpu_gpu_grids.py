import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("paired_grid_campaign", Path(__file__).with_name("cpu_gpu_grids.py"))
paired = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paired)


class PairedTests(unittest.TestCase):
    def rows(self):
        rows = []
        for arm in paired.ARMS:
            final, initial = paired.grid.ALL_ARMS[arm]
            for backend in paired.ROUTES:
                for repeat in range(3):
                    gpu = backend != "cpu"
                    rows.append(dict(system="water", arm=arm, backend=backend, ok=True,
                        scientific_commit="pin", geometry_sha256="same", threads=8, memory="112 GiB",
                        effective_route_options=dict(USE_CUEST=gpu, CUEST_XC=gpu, CUEST_SAD=gpu),
                        effective_scf_route_options=dict(USE_CUEST=gpu, CUEST_XC=gpu, CUEST_SAD=gpu),
                        cuda_visible_devices="0" if gpu else "",
                        wall_s=(20 if not gpu else 5) + repeat * .01,
                        components={k: (2e-6 if arm == "coarse75" else 0) for k in paired.grid.COMPONENTS},
                        options=dict(USE_CUEST=gpu, CUEST_XC=gpu, CUEST_SAD=gpu, CUEST_XC_TIMING=False,
                            DFT_RADIAL_POINTS=final[0], DFT_SPHERICAL_POINTS=final[1],
                            DFT_RADIAL_POINTS_INITIAL=initial[0] if initial else 0,
                            DFT_SPHERICAL_POINTS_INITIAL=initial[1] if initial else 0)))
        return rows

    def test_complete_pairs_use_same_grid(self):
        d = paired.summarize(self.rows())
        self.assertEqual(len(d["groups"]), 1)
        a = d["groups"][0]["arms"][-1]["routes"]
        self.assertAlmostEqual(a["cuda"]["speedup"], 20.01 / 5.01)
        self.assertTrue(a["cuda"]["acceptable_speedup"])
        self.assertAlmostEqual(a["cpu"]["grid_error_eh"], 2e-6)

    def test_incomplete_not_published(self):
        self.assertEqual(paired.summarize(self.rows()[:-1])["groups"], [])

    def test_cpu_route_flags_rejected(self):
        rows = self.rows()
        rows[0]["options"]["USE_CUEST"] = True
        with self.assertRaises(AssertionError):
            paired.summarize(rows)

    def test_cross_route_error_kept_not_silenced(self):
        rows = copy.deepcopy(self.rows())
        for row in rows:
            if row["backend"] == "cuda":
                row["components"][paired.grid.COMPONENTS[0]] += 2e-5
        route = paired.summarize(rows)["groups"][0]["arms"][0]["routes"]["cuda"]
        self.assertFalse(route["acceptable_speedup"])
        self.assertGreater(route["speedup"], 1)

    def test_environment_separates_backends(self):
        science, env = Path("/base/gits/science"), Path("/base/miniconda/envs/p4")
        host, package = paired.route_environment(science, "cpu", env, {"PATH": "/bin"})
        self.assertEqual(package, science / "build_grid_host/stage/lib/psi4")
        self.assertEqual(host["LD_LIBRARY_PATH"], "/base/miniconda/envs/p4/lib")
        self.assertEqual(host["CUDA_VISIBLE_DEVICES"], "")
        cuda, _ = paired.route_environment(science, "cuda", env, {"PATH": "/bin"})
        self.assertTrue(cuda["LD_LIBRARY_PATH"].startswith("/base/software/libxc-7.1.2-cuda/lib64:"))


if __name__ == "__main__":
    unittest.main()
