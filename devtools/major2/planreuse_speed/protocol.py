"""Production dispatch. Only clocks, checked synchronization and JK calls in windows."""
import time

from config import CALLS, require


class Timer:
    def __init__(self, sync, clock=time.perf_counter_ns):
        self.sync, self.clock = sync, clock

    @staticmethod
    def elapsed(start, end):
        require(type(start) is int and type(end) is int and 0 <= start < end,
                "invalid/nonpositive monotonic timing")
        return end-start

    def single(self, operation):
        self.sync()
        start = self.clock()
        operation()
        self.sync()
        end = self.clock()
        return self.elapsed(start, end)

    def initialize_compute(self, jk):
        self.sync()
        start = self.clock()
        jk.initialize()
        self.sync()
        middle = self.clock()
        jk.compute()
        self.sync()
        end = self.clock()
        # Derivation is after BOTH timed sections. No trace, hashing, copies,
        # counters, NVML, environment checks, JSON, or validation inside.
        return dict(initialize_ns=self.elapsed(start, middle),
                    compute_ns=self.elapsed(middle, end),
                    combined_ns=self.elapsed(start, end))


def expected_trace(arm, initializes):
    require(arm in ("baseline", "candidate") and 1 <= initializes <= 6, "trace arguments")
    return (dict(created=initializes, reused=0, counters=[]) if arm == "baseline" else
            dict(created=1, reused=initializes-1, counters=[1]))


def dispatch(jk, coefficients, timer, checkpoint, guard):
    """Warmup initialize, C once, compute; 5 paired rounds; exactly one cleanup."""
    calls = dict.fromkeys(CALLS, 0)
    records = []
    try:
        guard()
        warm_init = timer.single(jk.initialize)
        calls["initialize"] += 1
        jk.C_clear()
        jk.C_left_add(coefficients)
        calls["C_left_add"] += 1
        warm_compute = timer.single(jk.compute)
        calls["compute"] += 1
        records.append(checkpoint("warmup", 0, dict(initialize_ns=warm_init,
            compute_ns=warm_compute), calls.copy()))
        guard()
        for round_id in range(1, 6):
            times = timer.initialize_compute(jk)
            calls["initialize"] += 1
            calls["compute"] += 1
            records.append(checkpoint("initialize-compute", round_id, times, calls.copy()))
            guard()
            control = timer.single(jk.compute)
            calls["compute"] += 1
            records.append(checkpoint("control", round_id, dict(control_ns=control), calls.copy()))
            guard()
    finally:
        # Includes clock/sync/native failure, snapshots, signals and validation.
        jk.finalize()
        calls["finalize"] += 1
    require(calls == CALLS, "exact six/eleven/one native lifecycle")
    return records, calls
