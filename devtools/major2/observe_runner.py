"""Fresh exact staged package + unchanged runner; no driver overlay or scientific writes."""
import os
from pathlib import Path
import runpy
import signal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import observe
import policy as p
from probe import main as probe, maps


def main():
    def stop(sig, _):
        for value in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(value, signal.SIG_IGN)
        raise SystemExit(128+sig)
    for value in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(value, stop)
    before, after, runner, *args = sys.argv[1:]
    sys.argv = [__file__, before]
    probe()  # exact staged import/interface only; no scientific object/warmup
    import psi4
    stage = Path(os.environ["MAJOR2_EXPECTED_STAGE"])
    p.require(Path(psi4.__file__).resolve().parent == (stage/"lib/psi4").resolve(), "package overlay")
    p.require("--system" in args and args[args.index("--system")+1] == "protein157" and
              "--smoke" not in args, "Protein157 only")
    p.require(p.sha(runner) == p.RUNNER_SHA, "unchanged runner")
    provenance = p.read_json(before)
    label = Path(before).name.removesuffix(".before.json")
    provenance["process_meta"] = p.read_json(Path(before).with_name(label+".meta.json"))
    provenance.update(runner=str(runner), runner_sha256=p.sha(runner), wrapper_sha256=p.sha(__file__),
                      hook_sha256=p.sha(Path(__file__).with_name("observe.py")))
    capture = observe.Capture(psi4.core.RHF, stage, Path(before).with_suffix(".matrices"), provenance)
    iterations = psi4.core.HF.iterations
    sys.argv = [runner, *args]
    success = False
    try:
        with capture.installed():
            try:
                runpy.run_path(runner, run_name="__main__")
                success = True
            except SystemExit as exc:
                if exc.code not in (None, 0):
                    raise
                success = True
        if success:
            capture.finish()
    finally:
        # Unchanged runner installs its proof adapter on HF; restore even on TERM/failure.
        psi4.core.HF.iterations = iterations
        observe.atomic(after, maps())


if __name__ == "__main__":
    main()
