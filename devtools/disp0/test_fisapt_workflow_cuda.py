"""Unit tests for sapt_fisapt_workflow_cuda lifecycle and deferred RIFIT reporting."""
from contextlib import contextmanager
import importlib
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from test_cuda_backend import PACKAGE

workflow = importlib.import_module(PACKAGE + ".sapt_fisapt_workflow_cuda")
dfmo = importlib.import_module(PACKAGE + ".sapt_cuest_dfmo_cuda")
cachy_dir = Path(__file__).resolve().parent / "cachy"
import sys
if str(cachy_dir) not in sys.path:
    sys.path.insert(0, str(cachy_dir))
import fragment_term_trace


class _Alloc:
    def __init__(self, data):
        self.data = data
        self.freed = False
        self.nbytes = data.nbytes

    def free(self):
        self.freed = True


class _Arr:
    def __init__(self, alloc, data):
        self._alloc = alloc
        self.data = data
        self.shape = data.shape
        self.dtype = data.dtype
        self.nbytes = data.nbytes

    def view(self, shape, offset=0):
        start = offset // 8
        count = int(np.prod(shape))
        return _Arr(self._alloc, self.data.ravel()[start:start + count].reshape(shape))


class _Session:
    def __init__(self):
        self.allocations = []
        self.failed = False

    def upload(self, value):
        arr = np.array(value, dtype=np.float64, copy=True)
        alloc = _Alloc(arr)
        self.allocations.append(alloc)
        return _Arr(alloc, arr)

    def download(self, value):
        return np.array(value.data, dtype=np.float64, copy=True)

    def synchronize_device(self):
        return None

    @contextmanager
    def scope(self):
        mark = len(self.allocations)
        try:
            yield
        finally:
            for item in reversed(self.allocations[mark:]):
                if not item.freed:
                    item.free()
            del self.allocations[mark:]


class _FakeProvider(dfmo.CuESTDFMOProvider):
    def __init__(self, session, events, nao=3, ncenter=2):
        self.session = session
        self.events = events
        self.nao = nao
        self.auxiliary_basis = "jkfit"
        self._owner = SimpleNamespace(primary_snapshot=SimpleNamespace(ncenter=ncenter))
        self.closed = False

    def owned_plan(self):
        return self._owner

    def prepare_device_compute(self):
        self.events.append("prepare_jkfit")

    def report(self):
        if self.closed:
            raise RuntimeError("Native cuEST Major version query failed: 5")
        self.events.append("jkfit_report")
        return {"kind": "jkfit"}

    @contextmanager
    def sibling_provider(self, rifit):
        self.events.append("enter_rifit")
        sibling = SimpleNamespace(session=self.session, closed=False)

        def _ri_report():
            if sibling.closed:
                raise RuntimeError("Native cuEST Major version query failed: 5")
            self.events.append("rifit_report")
            return {"kind": "rifit"}

        sibling.report = _ri_report
        alloc = _Alloc(np.zeros((4,), dtype=np.float64))
        self.session.allocations.append(alloc)
        try:
            yield sibling
        finally:
            sibling.closed = True
            alloc.free()
            self.events.append("exit_rifit")


class WorkflowLifecycleTests(unittest.TestCase):
    def test_rifit_report_runs_before_sibling_exit_and_after_coupled_induction(self):
        session = _Session()
        events = []
        provider = _FakeProvider(session, events, nao=3, ncenter=2)

        matrices = {name: np.eye(3, dtype=np.float64) for name in set(workflow.HF_FIELDS) | {"J_O"}}
        matrices["Cocc_A"] = np.eye(3, 2, dtype=np.float64)
        matrices["Cocc_B"] = np.eye(3, 2, dtype=np.float64)
        matrices["Cvir_A"] = np.eye(3, 1, dtype=np.float64)
        matrices["Cvir_B"] = np.eye(3, 1, dtype=np.float64)
        energies = {
            "a": np.array([-1.0, -0.5]),
            "b": np.array([-1.0, -0.5]),
            "r": np.array([0.5]),
            "s": np.array([0.5]),
        }
        xyz = np.zeros((2, 3), dtype=np.float64)
        za = np.array([1.0, 0.0], dtype=np.float64)
        zb = np.array([0.0, 1.0], dtype=np.float64)

        def fake_localize(iao, ed, atoms, **kwargs):
            return SimpleNamespace(
                l=session.upload(np.eye(3, 2)),
                u=session.upload(np.eye(2)),
                q=session.upload(np.eye(2)),
                jacobi=SimpleNamespace(iterations=1, converged=True, work_plan={}),
                iao=SimpleNamespace(work_plan={}),
            )

        def fake_elst(*args, **kwargs):
            events.append("felst")
            return SimpleNamespace(matrix=session.upload(np.zeros((5, 5))), work_plan={})

        def fake_exch(*args, **kwargs):
            events.append("fexch")
            return SimpleNamespace(matrix=session.upload(np.zeros((5, 5))), total=0.0)

        def fake_ind(*args, **kwargs):
            events.append("coupled_find")
            pair = lambda: SimpleNamespace(ab=session.upload(np.zeros((5, 5))),
                                           ba=session.upload(np.zeros((5, 5))))
            return SimpleNamespace(
                coupled=SimpleNamespace(
                    combined_ab=session.upload(np.zeros((5, 5))),
                    combined_ba=session.upload(np.zeros((5, 5))),
                    induction=pair(),
                    exchange_induction=pair(),
                ),
                uncoupled=SimpleNamespace(ind20u=pair(), exch_ind20u=pair()),
                response_diagnostics={},
            )

        def fake_disp(ri, *args, **kwargs):
            self.assertFalse(ri.closed)
            events.append("dispersion")
            return {
                "Disp20,u": -0.01,
                "Exch-Disp20,u": 0.001,
                "Disp0 occupied pairs": np.zeros((2, 2)),
                "Disp20 occupied pairs": np.zeros((2, 2)),
                "Exch-Disp20 occupied pairs": np.zeros((2, 2)),
                "Disp0 execution": {"ok": True},
            }

        class _Math:
            def __init__(self, sess):
                self.sess = sess

            def window(self, arr, col0, ncols):
                return arr

        with patch.object(workflow, "CuESTPointPotential", return_value=SimpleNamespace()), \
             patch.object(workflow, "CuESTOverlapBlocks",
                          return_value=SimpleNamespace(local_atom_indices=[0, 1], compute=lambda: None)), \
             patch.object(workflow, "CuSolverSymmetricPower", return_value=SimpleNamespace()), \
             patch.object(workflow, "DeviceMath", _Math), \
             patch.object(workflow, "prepare_iao_metrics", return_value=SimpleNamespace()), \
             patch.object(workflow, "build_iao_device", return_value=SimpleNamespace()), \
             patch.object(workflow, "localize_iao_device", side_effect=fake_localize), \
             patch.object(workflow, "felst_device", side_effect=fake_elst), \
             patch.object(workflow, "fexch_device", side_effect=fake_exch), \
             patch.object(workflow, "coupled_find_device", side_effect=fake_ind), \
             patch.object(workflow, "prepare_hf", return_value=({}, {}, {})), \
             patch.object(workflow, "blocked_device_dispersion", side_effect=fake_disp):
            outputs, meta = workflow.fragment_terms(
                provider, "rifit", "minao", matrices, energies, xyz, za, zb,
            )

        self.assertEqual(meta["FISAPT execution"]["rifit_producer"], {"kind": "rifit"})
        self.assertEqual(meta["FISAPT execution"]["jkfit_producer"], {"kind": "jkfit"})
        self.assertEqual(
            events,
            ["prepare_jkfit", "felst", "fexch", "coupled_find",
             "enter_rifit", "dispersion", "rifit_report", "exit_rifit", "jkfit_report"],
        )
        self.assertEqual(session.allocations, [])
        self.assertIn("Disp_AB", outputs)

    def test_fragment_term_trace_boundaries_remain_valid(self):
        src = Path(workflow.__file__).read_text()
        markers = fragment_term_trace.boundaries(src, include_localization=True)
        self.assertEqual(
            tuple(markers.values()),
            ("localization", "elst", "exch", "ind", "disp", "after_terms"),
        )


if __name__ == "__main__":
    unittest.main()
