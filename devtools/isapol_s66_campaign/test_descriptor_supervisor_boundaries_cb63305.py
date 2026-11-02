"""Effective allocation and filesystem boundaries; no scheduler or science."""
import importlib.util
from pathlib import Path

import pytest


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


supervisor = load('descriptor_supervisor_route', 'phoenix_descriptor_bank_cb63305.py')
scratch = load('descriptor_scratch_boundaries', 'descriptor_scratch_cb63305.py')


def allocation():
    return dict(Partition='cpu-small', NumCPUs='16', NumNodes='1', TimeLimit='00:45:00',
                QOS='embers', Account='gts-cs207-chemx')


def test_filesystem_inodes_not_just_quota_must_have_exact_headroom():
    unlimited = dict(remaining_bytes=None, remaining_files=None)
    args = dict(free_bytes=2*1024**3, user_quota=unlimited, group_quota=unlimited,
                required_bytes=2*1024**3, required_files=4096)
    assert scratch.admit_scratch(**args, free_files=4096)['filesystem_free_files'] == 4096
    with pytest.raises(ValueError, match='filesystem.*inode'):
        scratch.admit_scratch(**args, free_files=4095)


def test_exact_effective_route_is_not_just_a_submission_directive():
    assert supervisor.admit_route(allocation())['cpus'] == 16
    for change in (dict(Partition='cpu-large'), dict(Partition='gpu-v100'),
                   dict(NumCPUs='4'), dict(NumCPUs='32'), dict(NumNodes='2'),
                   dict(TimeLimit='08:00:00'), dict(TimeLimit='00:30:00'),
                   dict(QOS='inferno'), dict(Account='another')):
        with pytest.raises(ValueError, match='effective descriptor route'):
            supervisor.admit_route(allocation() | change)
