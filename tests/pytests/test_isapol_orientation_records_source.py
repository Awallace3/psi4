"""Synthetic full sealed-export dialect, including omission/association mutations."""
import copy
import hashlib
import struct
import unittest
import numpy as np
from isapol_source_test_support import load_source

adapter = load_source("isapol_offline_orientation")

COMPONENTS = ["10", "11c", "11s", "20", "21c", "21s", "22c", "22s"]


def records():
    frequencies, weights = [float(i) for i in range(11)], [0.] + [.1]*10
    sites = [dict(label=s, origin_bohr=[float(i), 0., 0.], frame=np.eye(3).tolist(),
                  rank_limit=2) for i, s in enumerate(("O", "H1", "H2"))]
    core_input = []
    for s in sites:
        a = [np.eye(8).tolist() for _ in frequencies]
        core_input.append(dict(label=s["label"], origin_bohr=s["origin_bohr"],
                               frame=s["frame"], ranks=[1, 2], components=COMPONENTS,
                               responses=a, shape=[11, 8, 8],
                               response_sha256=[hashlib.sha256(
                                   np.asarray(x, dtype="<f8").tobytes()).hexdigest() for x in a]))
    p = dict(sites=sites, components=COMPONENTS, frequencies=frequencies,
             cp_weights=weights, core_input=core_input,
             quadrature_provenance=dict(source_sha256=hashlib.sha256(
                 struct.pack("<22d", *(frequencies + weights))).hexdigest()),
             refined=[dict(node=i, values=[np.pad(np.eye(8), ((1, 0), (1, 0))).tolist()
                                           for _ in sites]) for i in range(11)])
    r = dict(public_frames=[dict(label=s["label"], xyz_bohr=s["origin_bohr"], frame=s["frame"])
                            for s in sites],
             grid=dict(frequencies=frequencies, cp_weights=weights, origin="source-reconstructed"),
             frequency_associations=[dict(node=i) for i in range(11)],
             refined=[dict(blocks=[dict(site=s["label"], components=COMPONENTS, rank=2,
                                       tokens=[[f"{v:.12f}" for v in row] for row in np.eye(8)])
                                   for s in sites]) for i in range(11)])
    return p, r


class RecordTests(unittest.TestCase):
    def test_full_refined_records_preserve_rank_nodes_frames_and_tensor_entries(self):
        p, r = records()
        arms = adapter.refined_records(p, r)
        self.assertEqual(set(arms), {"P", "C"})
        for arm in arms.values():
            self.assertEqual(arm["frequencies"], p["frequencies"])
            self.assertEqual(arm["weights"], p["cp_weights"])
            self.assertEqual([s["label"] for s in arm["sites"]], ["O", "H1", "H2"])
            self.assertEqual([s["ranks"] for s in arm["sites"]], [[1, 2]]*3)
            np.testing.assert_array_equal(arm["sites"][1]["responses"][5], np.eye(8))
        self.assertIn("NOT observed", arms["C"]["qualification"])

    def test_no_omission_rank_promotion_node_reassociation_or_symmetry_repair(self):
        for mutation in ("missing_site", "matrix_shape", "asymmetry", "rank",
                         "components", "node", "hash", "native_refined", "frame", "weights"):
            with self.subTest(mutation=mutation):
                p, r = records()
                if mutation == "missing_site": r["refined"][5]["blocks"].pop()
                elif mutation == "matrix_shape": r["refined"][5]["blocks"][0]["tokens"][0].pop()
                elif mutation == "asymmetry": r["refined"][5]["blocks"][0]["tokens"][0][1] = ".000000000001"
                elif mutation == "rank": p["core_input"][0]["ranks"] = [1, 2, 3]
                elif mutation == "components": r["refined"][0]["blocks"][0]["components"].reverse()
                elif mutation == "node": r["frequency_associations"][5]["node"] = 4
                elif mutation == "hash": p["core_input"][0]["response_sha256"][1] = "0"*64
                elif mutation == "native_refined": p["refined"][5]["values"][0][1][1] = 2.
                elif mutation == "frame": r["public_frames"][1]["frame"][0][0] = -1.
                elif mutation == "weights": p["cp_weights"][0] = .1
                with self.assertRaises(ValueError):
                    adapter.refined_records(p, r)
