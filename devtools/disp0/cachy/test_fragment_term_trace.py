"""Real Python tracing and deterministic fake-session controls, no native work."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

from fragment_term_trace import TermTrace, boundaries
from run_fragment_term_profile import decomposition


SOURCE = """\
def fragment_terms(provider):
    session = provider.session
    state = felst_device()
    exch = fexch_device()
    induction = coupled_find_device()
    rotations = []
    prep()
    disp()
    device_outputs = {}
    finish()
    return state, exch, induction, device_outputs
"""


class TermTraceTests(unittest.TestCase):
    def execute(self, source=SOURCE, *, fail=False, include_localization=False):
        filename = str(Path("/private/verified/sapt_fisapt_workflow_cuda.py").resolve())
        counter = [0.]

        def clock():
            return counter[0]

        def advance(n):
            def call():
                counter[0] += n
                if fail:
                    raise ValueError("test failure")
                return n
            return call

        scope = {"__file__": filename, "felst_device": advance(2),
                 "prepare_iao_metrics": advance(17),
                 "fexch_device": advance(3), "coupled_find_device": advance(5),
                 "prep": advance(7), "disp": advance(11), "finish": advance(13)}
        exec(compile(source, filename, "exec"), scope)
        synchronizations = []
        provider = SimpleNamespace(session=SimpleNamespace(
            synchronize_device=lambda: synchronizations.append(clock())))
        tracer = TermTrace(filename, source, clock=clock, include_localization=include_localization)
        with tracer:
            if fail:
                with self.assertRaises(ValueError):
                    scope["fragment_terms"](provider)
            else:
                scope["fragment_terms"](provider)
        return tracer, synchronizations

    def test_four_terms_and_preparation_are_synchronized_and_disjoint(self):
        tracer, syncs = self.execute()
        report = tracer.report()["workflow"]
        self.assertEqual({k: v["wall_seconds"] for k, v in report["windows"].items()},
                         {"elst": 2, "exch": 3, "ind": 5, "disp": 18})
        self.assertEqual(report["workflow_other_seconds"], 13)
        self.assertEqual(report["wall_seconds"], 41)
        self.assertEqual(len(syncs), 7)
        self.assertIsNone(sys.gettrace())

    def test_exception_cannot_be_reported_as_complete_and_trace_is_restored(self):
        tracer, _ = self.execute(fail=True)
        with self.assertRaisesRegex(RuntimeError, "fully observed"):
            tracer.report()
        self.assertIsNone(sys.gettrace())

    def test_source_shape_change_rejects_instead_of_guessing_boundaries(self):
        with self.assertRaises(ValueError):
            boundaries(SOURCE.replace("felst_device()", "foreign_device()"))
        with self.assertRaises(ValueError):
            boundaries(SOURCE.replace("rotations = []", "rotations = get_rotations()"))

    def test_absent_workflow_or_two_workflows_reject(self):
        tracer, _ = self.execute()
        tracer.calls.append(dict(tracer.calls[0]))
        with self.assertRaises(RuntimeError):
            tracer.report()
        empty = TermTrace("/private/workflow.py", SOURCE)
        with self.assertRaises(RuntimeError):
            empty.report()

    def test_existing_debugger_trace_is_not_overwritten(self):
        old = sys.gettrace()
        sentinel = lambda frame, event, arg: None
        sys.settrace(sentinel)
        try:
            with self.assertRaisesRegex(RuntimeError, "another tracer"):
                with TermTrace("/private/workflow.py", SOURCE):
                    pass
            self.assertIs(sys.gettrace(), sentinel)
        finally:
            sys.settrace(old)

    def test_full_fragment_closes_without_folding_overhead_into_terms(self):
        tracer, _ = self.execute()
        result = decomposition(tracer.report(), 51)
        self.assertEqual(result["parts_seconds"], {
            "elst": 2, "exch": 3, "ind": 5, "disp": 18,
            "workflow_other": 13, "outer_factory_other": 10})
        self.assertEqual(sum(result["parts_seconds"].values()), 51)
        with self.assertRaises(RuntimeError):
            decomposition(tracer.report(), 40)

    def test_optional_localization_is_observed_not_guessed_from_residual(self):
        source = SOURCE.replace("    state = felst_device()",
                                "    metrics = prepare_iao_metrics()\n    state = felst_device()")
        tracer, syncs = self.execute(source, include_localization=True)
        workflow = tracer.report()["workflow"]
        self.assertEqual(workflow["windows"]["localization"]["wall_seconds"], 17)
        self.assertEqual(workflow["workflow_other_seconds"], 13)
        self.assertEqual(workflow["wall_seconds"], 58)
        self.assertEqual(len(syncs), 8)
        result = decomposition(tracer.report(), 68)
        self.assertEqual(result["parts_seconds"]["localization"], 17)
        self.assertEqual(sum(result["parts_seconds"].values()), 68)
        with self.assertRaises(ValueError):
            boundaries(SOURCE, include_localization=True)


if __name__ == "__main__":
    unittest.main()
