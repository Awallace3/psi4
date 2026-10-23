import os
import subprocess
import sys
import textwrap

import pytest

import psi4

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]


def test_timer_error_releases_lock():
    """A timer call that throws must release the timer lock.

    timer_off of a timer that is not on used to throw while holding the global
    timer lock, so the next timer call (anywhere in psi4) waited forever. Run in a
    subprocess so a regression fails on the timeout instead of hanging the suite.
    """
    script = textwrap.dedent("""
        import psi4
        try:
            psi4.core.timer_off("timer-lock-test: never on")
        except RuntimeError:
            pass
        else:
            raise SystemExit("timer_off of a timer that is not on did not raise")
        # Thrown inside Timer_Structure::turn_on, not by timer_on itself.
        psi4.core.timer_on("timer-lock-test: twice")
        try:
            psi4.core.timer_on("timer-lock-test: twice")
        except RuntimeError:
            pass
        else:
            raise SystemExit("turning on a timer that is already on did not raise")
        psi4.core.timer_off("timer-lock-test: twice")
        psi4.core.timer_on("timer-lock-test: after")
        psi4.core.timer_off("timer-lock-test: after")
        print("released")
    """)
    # The child must import the psi4 under test, not whichever one its own sys.path finds first.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (os.path.dirname(os.path.dirname(psi4.__file__)), env.get("PYTHONPATH")) if p)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120, env=env)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "released" in result.stdout
