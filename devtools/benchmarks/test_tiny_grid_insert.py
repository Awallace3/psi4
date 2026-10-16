from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tiny_grid_insert as insert


def test_only_exact_original_controller_is_selected():
    assert insert.matches_controller(dict(cmd=[b"python", insert.CONTROLLER.encode(), b"campaign"]))
    assert not insert.matches_controller(dict(cmd=[b"python", insert.CONTROLLER.encode(), b"worker"]))
    assert not insert.matches_controller(dict(cmd=[b"python", b"/another/tiny_grid_pilot.py", b"campaign"]))
    assert not insert.matches_controller(None)


def test_new_checks_are_below_ten_percent_with_default_final():
    for final, initial in insert.EXTRAS.values():
        assert final == (75, 302)
        assert .04 < initial[0]*initial[1]/(75*302) < .08


def test_proc_describes_only_owned_real_process():
    import os
    own = insert.proc(os.getpid())
    assert own["pid"] == os.getpid()
    assert own["start"].isdigit()
    assert insert.proc(999999999) is None
