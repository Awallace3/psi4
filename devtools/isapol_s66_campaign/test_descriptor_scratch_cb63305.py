"""Pure quota-format/budget tests; no filesystem or scheduler contact."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'descriptor_scratch_cb63305', Path(__file__).with_name('descriptor_scratch_cb63305.py'))
scratch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scratch)
PATH = '/storage/scratch1/3/example'


def test_actual_unscaled_lustre_layout_and_zero_group_limits():
    user = scratch.parse_quota(
        f'Filesystem kbytes bquota blimit bgrace files iquota ilimit igrace\n'
        f'{PATH} 100 200 300 - 10 20 30 -\n', PATH)
    assert user['remaining_bytes'] == 100*1024
    assert user['remaining_files'] == 10
    group = scratch.parse_quota(f'{PATH} 100 0 0 - 10 0 0 -\n', PATH)
    assert group['remaining_bytes'] is group['remaining_files'] is None


def test_wrapped_path_and_exceeded_marker_do_not_hide_quota_exhaustion():
    quota = scratch.parse_quota(f'{PATH}\n 201* 200 300 - 21* 20 30 -\n', PATH)
    assert quota['remaining_bytes'] == quota['remaining_files'] == 0


@pytest.mark.parametrize('text', ['', 'quota query failed', f'{PATH} 100K 200 300 - 10 20 30 -',
                                 f'{PATH} 1 2 3 -', f'{PATH} 1 2 3 - 1 2 3 -\n{PATH} 1 2 3 - 1 2 3 -'])
def test_failed_unknown_scaled_or_partial_quota_queries_cannot_pass(text):
    with pytest.raises(ValueError):
        scratch.parse_quota(text, PATH)


def test_exact_and_one_short_filesystem_user_group_and_inode_headroom():
    finite = dict(remaining_bytes=1024, remaining_files=8)
    unlimited = dict(remaining_bytes=None, remaining_files=None)
    result = scratch.admit_scratch(free_bytes=1024, free_files=8, user_quota=finite,
        group_quota=unlimited, required_bytes=1024, required_files=8)
    assert 'not a reservation' in result['qualification']
    cases = [
        dict(free_bytes=1023, user_quota=finite, group_quota=unlimited),
        dict(free_bytes=2048, user_quota=finite | {'remaining_bytes': 1023}, group_quota=unlimited),
        dict(free_bytes=2048, user_quota=finite, group_quota=finite | {'remaining_bytes': 1023}),
        dict(free_bytes=2048, user_quota=finite | {'remaining_files': 7}, group_quota=unlimited),
    ]
    for kwargs in cases:
        with pytest.raises(ValueError):
            scratch.admit_scratch(**kwargs, free_files=8, required_bytes=1024, required_files=8)
