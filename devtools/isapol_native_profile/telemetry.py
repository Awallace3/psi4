"""Nonnumeric, process-local instrumentation. No Psi4 imports at module scope."""
import contextlib
import ast
import functools
import inspect
import json
import os
from pathlib import Path
import resource
import sys
import threading
import time
import textwrap


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


class Recorder:
    """Independent spans and StageLog windows; overlapping durations are NOT additive."""

    def __init__(self, path, clock=time.perf_counter_ns):
        self.stream = Path(path).open("x")
        self.clock = clock
        self.origin = clock()
        self.sequence = 0
        self.stack = []
        self.windows = {}

    def emit(self, event, **fields):
        self.stream.write(json.dumps(dict(event=event, ns=self.clock() - self.origin,
                                          **fields), allow_nan=False) + "\n")
        self.stream.flush()

    @contextlib.contextmanager
    def span(self, name):
        self.sequence += 1
        identifier = self.sequence
        parent = self.stack[-1] if self.stack else None
        start = self.clock()
        self.emit("begin", id=identifier, parent=parent, name=name)
        self.stack.append(identifier)
        try:
            yield
        except BaseException as error:
            # A telemetry write failure must not hide the scientific exception.
            try:
                self.emit("end", id=identifier, name=name, status="failed",
                          exception=type(error).__name__, seconds=(self.clock() - start) / 1e9)
            except Exception:
                pass
            raise
        else:
            self.emit("end", id=identifier, name=name, status="returned",
                      seconds=(self.clock() - start) / 1e9)
        finally:
            self.stack.pop()

    def close_window(self, log, status="complete"):
        active = self.windows.pop(id(log), None)
        if active:
            name, start = active
            self.emit("stage_end", name=name, logger=id(log), status=status,
                      seconds=(self.clock() - start) / 1e9)

    def close(self):
        for logger, (name, start) in self.windows.items():
            self.emit("stage_end", name=name, logger=logger, status="unclosed",
                      seconds=(self.clock() - start) / 1e9)
        self.windows.clear()
        self.stream.close()


class Patches:
    """Restore every Python binding, even after exceptions. Never patch native types."""

    def __init__(self):
        self.saved = []
        self.inventory = []

    def attribute(self, obj, name, value, label):
        self.saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)
        self.inventory.append(label)

    def mapping(self, obj, name, value, label):
        self.saved.append((obj, name, obj[name]))
        obj[name] = value
        self.inventory.append(label)

    def restore(self):
        for obj, name, value in reversed(self.saved):
            if isinstance(obj, dict):
                obj[name] = value
            else:
                setattr(obj, name, value)
        self.saved.clear()

    def wrap(self, obj, name, recorder, label=None):
        original = getattr(obj, name)
        label = label or obj.__name__ + "." + name

        @functools.wraps(original)
        def measured(*args, **kwargs):
            with recorder.span(label):
                return original(*args, **kwargs)

        self.attribute(obj, name, measured, label)


@contextlib.contextmanager
def line_intervals(function, recorder, assignments, observations=None):
    """Observe selected Python statement boundaries, never inner arithmetic.

    assignments maps a local assignment name to an interval label (or None to
    stop). Global tracing dispatch still has overhead; compare the off control.
    No source rewrite, native-type patch, arrays, or integral engine is involved.
    """
    require_trace = sys.gettrace()
    if require_trace is not None:
        raise RuntimeError("line-boundary probes refuse an existing debugger/trace hook")
    source, first = inspect.getsourcelines(function)
    tree = ast.parse(textwrap.dedent("".join(source)))
    boundaries = {}
    observe_lines = {}
    seen = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in assignments:
                    values = assignments[target.id]
                    values = values if isinstance(values, tuple) else (values,)
                    index = seen.get(target.id, 0)
                    if index >= len(values):
                        raise RuntimeError("unexpected repeated line-boundary assignment: " + target.id)
                    boundaries[first + node.lineno - 1] = values[index]
                    if observations and target.id in observations:
                        observe_lines[first + node.lineno - 1] = observations[target.id]
                    seen[target.id] = index + 1
    expected = {key: len(value) if isinstance(value, tuple) else 1
                for key, value in assignments.items()}
    if seen != expected:
        raise RuntimeError("qualified function line-boundary layout differs: " + function.__name__)
    active = None

    def finish(exc=(None, None, None)):
        nonlocal active
        if active is not None:
            closing, active = active, None
            closing.__exit__(*exc)

    def local(frame, event, arg):
        nonlocal active
        if event == "line" and frame.f_lineno in boundaries:
            finish()
            if frame.f_lineno in observe_lines:
                recorder.emit("metadata", function=function.__name__,
                              values=observe_lines[frame.f_lineno](frame.f_locals))
            name = boundaries[frame.f_lineno]
            if name is not None:
                active = recorder.span(name)
                active.__enter__()
        elif event == "exception":
            finish(arg)
        elif event == "return":
            finish()
        return local

    def trace(frame, event, arg):
        if event == "call" and frame.f_code is function.__code__:
            return local
        return None

    sys.settrace(trace)
    try:
        yield
    finally:
        sys.settrace(require_trace)
        finish()


def install_stages(patches, cls, recorder):
    original_stage, original_end, original_failed = cls.stage, cls.stage_end, cls.stage_failed

    @functools.wraps(original_stage)
    def stage(log, name, *args, **kwargs):
        # Original stage calls the patched stage_end, closing the previous window.
        result = original_stage(log, name, *args, **kwargs)
        recorder.windows[id(log)] = (str(name), recorder.clock())
        recorder.emit("stage_begin", name=str(name), logger=id(log))
        return result

    @functools.wraps(original_end)
    def end(log, *args, **kwargs):
        recorder.close_window(log)
        return original_end(log, *args, **kwargs)

    @functools.wraps(original_failed)
    def failed(log, *args, **kwargs):
        recorder.close_window(log, "failed")
        return original_failed(log, *args, **kwargs)

    for name, value in (("stage", stage), ("stage_end", end), ("stage_failed", failed)):
        patches.attribute(cls, name, value, "StageLog." + name)


def scratch_snapshot(root):
    """lstat only a fresh owned root; never follow directory/file symlinks."""
    result = dict(files=0, directories=0, symlinks=0, logical_bytes=0,
                  allocated_bytes=0, races=0)
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            try:
                path = Path(directory) / name
                stat = path.lstat()
                if path.is_symlink():
                    result["symlinks"] += 1
                elif name in dirs:
                    result["directories"] += 1
                else:
                    result["files"] += 1
                    result["logical_bytes"] += stat.st_size
                    result["allocated_bytes"] += stat.st_blocks * 512
            except FileNotFoundError:
                result["races"] += 1
    return result


def process_snapshot():
    result = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in ("VmRSS", "VmHWM", "Threads"):
            result[key] = value.strip()
    # Linux ru_maxrss is KiB, unlike macOS; this harness runs on Linux.
    result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return result


class Sampler:
    """Sampled scratch maxima are lower bounds, not a scratch reservation policy."""

    def __init__(self, root, path, interval):
        self.root, self.path, self.interval = Path(root), Path(path), interval
        self.stop_event = threading.Event()
        self.thread = None
        self.peaks = {}
        self.errors = []
        self.count = 0

    def sample(self, stream):
        try:
            scratch = scratch_snapshot(self.root)
            for key, value in scratch.items():
                self.peaks[key] = max(value, self.peaks.get(key, 0))
            stream.write(json.dumps(dict(monotonic_ns=time.perf_counter_ns(),
                                         scratch=scratch, process=process_snapshot())) + "\n")
            stream.flush()
            self.count += 1
        except Exception as error:
            self.errors.append(repr(error))

    def start(self):
        def worker():
            try:
                with self.path.open("x") as stream:
                    self.sample(stream)
                    while not self.stop_event.wait(self.interval):
                        self.sample(stream)
                    self.sample(stream)
            except Exception as error:
                self.errors.append(repr(error))
        self.thread = threading.Thread(target=worker, name="profile-sampler", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        return dict(sample_interval_seconds=self.interval, samples=self.count,
                    scratch_sampled_maxima=self.peaks, errors=self.errors,
                    process=process_snapshot(),
                    limitation="Scratch maxima are sampled lower bounds; metric peaks need not coincide.")
