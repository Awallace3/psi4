"""Host-side cuEST diagnostic timings, not CUDA-event kernel measurements."""
from contextlib import contextmanager
import time


class PhaseTimings:
    """Accumulate inclusive wall/process-CPU intervals, including failed calls.

    Nested intervals overlap and must not be summed. Process CPU includes native
    worker threads, so it can exceed wall time; it is not a GPU utilization metric.
    This adds no synchronization and changes no native lifetime or admission rule.
    """
    def __init__(self):
        self.phases = {}

    @contextmanager
    def measure(self, name):
        wall, cpu = time.perf_counter(), time.process_time()
        failed = True
        try:
            yield
            failed = False
        finally:
            cpu_elapsed = time.process_time() - cpu
            wall_elapsed = time.perf_counter() - wall
            row = self.phases.setdefault(name, {
                "calls": 0, "failures": 0, "wall_seconds": 0.0, "process_cpu_seconds": 0.0})
            row["calls"] += 1
            row["failures"] += int(failed)
            row["wall_seconds"] += wall_elapsed
            row["process_cpu_seconds"] += cpu_elapsed

    def report(self):
        return {
            "scope": "Inclusive host wall/process CPU; nested intervals overlap; not GPU kernel time",
            "phases": {name: dict(row) for name, row in self.phases.items()},
        }
