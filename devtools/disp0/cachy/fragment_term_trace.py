"""Observe verified workflow source boundaries; never replace science functions.

The four term windows have device synchronization at their boundaries. DISP
includes rotation views, prepare_hf and blocked dispersion. Localization, plans,
scaling, transfers and cleanup outside those windows remain explicit residuals.
This is an instrumented diagnostic, not an unprofiled timing or kernel clock.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time


def boundaries(source):
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == "fragment_terms"]
    if len(functions) != 1:
        raise ValueError("One fragment_terms function required")
    markers = {}
    for name, call in (("state", "felst_device"), ("exch", "fexch_device"),
                       ("induction", "coupled_find_device"), ("rotations", None),
                       ("device_outputs", None)):
        nodes = [n for n in ast.walk(functions[0]) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
        if len(nodes) != 1:
            raise ValueError("Ambiguous workflow boundary: "+name)
        node = nodes[0]
        if call and not (isinstance(node.value, ast.Call)
                         and isinstance(node.value.func, ast.Name)
                         and node.value.func.id == call):
            raise ValueError("Workflow term dispatch changed: "+name)
        if name == "rotations" and not (
                isinstance(node.value, ast.List) and not node.value.elts):
            raise ValueError("Dispersion preparation boundary changed")
        if name == "device_outputs" and not isinstance(node.value, ast.Dict):
            raise ValueError("Post-term output boundary changed")
        markers[name] = node.lineno
    if list(markers.values()) != sorted(set(markers.values())):
        raise ValueError("Expected sequential, distinct workflow boundaries")
    return dict(zip(markers.values(), ("elst", "exch", "ind", "disp", "after_terms")))


class TermTrace:
    def __init__(self, workflow, source, *, clock=time.perf_counter):
        self.workflow = str(Path(workflow).resolve())
        self.markers = boundaries(source)
        self.source_sha256 = hashlib.sha256(source.encode()).hexdigest()
        self.clock = clock
        self.calls = []
        self.previous = None

    def __enter__(self):
        self.previous = sys.gettrace()
        if self.previous is not None:
            raise RuntimeError("Fresh dedicated worker without another tracer required")
        sys.settrace(self.global_trace)
        return self

    def __exit__(self, *exc):
        sys.settrace(self.previous)

    def global_trace(self, frame, event, arg):
        if (event != "call" or frame.f_code.co_name != "fragment_terms"
                or frame.f_globals.get("__file__") != self.workflow):
            return None
        session = frame.f_locals["provider"].session
        session.synchronize_device()
        start = self.clock()
        row = {"status": "STARTED", "windows": {}, "started": start}
        self.calls.append(row)
        active, seen = None, []
        frame.f_trace_opcodes = False

        def local_trace(current, event, arg):
            nonlocal active
            if event == "line":
                label = self.markers.get(current.f_lineno)
                if label is None or label in seen:
                    return local_trace
                expected = ("elst", "exch", "ind", "disp", "after_terms")[len(seen)]
                if label != expected:
                    raise RuntimeError("Unexpected term-boundary execution order")
                # Complete preceding work before closing/starting adjacent terms.
                # Initial preparation waiting remains outside ELST.
                session.synchronize_device()
                now = self.clock()
                if active is not None:
                    row["windows"][active]["finished"] = now
                    row["windows"][active]["wall_seconds"] = (
                        now-row["windows"][active]["started"])
                seen.append(label)
                active = None if label == "after_terms" else label
                if active is not None:
                    row["windows"][active] = {"started": now}
            elif event == "exception":
                row["exception_observed"] = True
            elif event == "return":
                if arg is None or active is not None or len(seen) != 5:
                    row["status"] = "INCOMPLETE_OR_EXCEPTION"
                    return None
                session.synchronize_device()
                row.update(status="COMPLETED", finished=self.clock())
                row["wall_seconds"] = row["finished"]-start
                term_sum = sum(v["wall_seconds"] for v in row["windows"].values())
                row["workflow_other_seconds"] = row["wall_seconds"]-term_sum
                if row["workflow_other_seconds"] < 0:
                    raise RuntimeError("Overlapping or invalid term windows")
                return None
            return local_trace

        return local_trace

    def report(self):
        if len(self.calls) != 1 or self.calls[0]["status"] != "COMPLETED":
            raise RuntimeError("Exactly one fully observed fragment workflow required")
        return {
            "scope": "SYNCHRONIZED_SOURCE_BLOCK_WALL_NOT_KERNEL_OR_BASELINE_TIMING",
            "source": self.workflow, "source_sha256": self.source_sha256,
            "boundaries": {str(k): v for k, v in self.markers.items()},
            "synchronization": "Existing owned session synchronize_device at each boundary",
            "dispersion_scope": "Rotation views + prepare_hf + blocked_device_dispersion",
            "workflow": self.calls[0],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--trace-output", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    bootstrap = args.bootstrap.resolve(strict=True)
    sys.path.insert(0, str(bootstrap.parent.parent/"phoenix_validation"))
    from gpu_runtime import read_config, digest
    _, build = read_config(args.config, args.config_sha256)
    if build["provenance_sha256"].get(str(bootstrap)) != digest(bootstrap):
        raise RuntimeError("Original authenticated bootstrap required")
    workflow = Path(build["python_package"])/"driver/procrouting/sapt/sapt_fisapt_workflow_cuda.py"
    relative = "driver/procrouting/sapt/sapt_fisapt_workflow_cuda.py"
    if digest(workflow) != build["scientific_python_sha256"][relative]:
        raise RuntimeError("Whole-installed workflow changed")
    tracer = TermTrace(workflow, workflow.read_text())
    status = {"status": "STARTED", "scope": "SEPARATE_TRACED_WORKER"}
    if args.trace_output.exists():
        raise RuntimeError("Fresh trace receipt required")
    try:
        sys.argv = [str(bootstrap), "--config", str(args.config),
                    "--config-sha256", args.config_sha256, *remaining]
        with tracer:
            runpy.run_path(str(bootstrap), run_name="__main__")
        status.update(status="PASS_OBSERVED_TERM_BOUNDARIES", trace=tracer.report())
    except BaseException as error:
        status.update(status="FAILED", error=f"{type(error).__name__}: {error}",
                      partial_calls=tracer.calls)
        raise
    finally:
        with args.trace_output.open("x") as stream:
            json.dump(status, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")


if __name__ == "__main__":
    main()
