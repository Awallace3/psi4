"""Closed engineering inode ESTIMATE; not production acceptance/creation cap.

Conditional native subtotal1671, defensive second-domain2171; launcher/vendor/
failure lifetimes remain incompletely bounded.65536 retains an uncertainty
margin, not a universal limit. No iteration/thread multiplier or quota bypass.
Qualification/runtime evidence belongs to qualified_qomp, not these dicts.
"""
import hashlib
import importlib.util
import math
from pathlib import Path
import time

SCHEMA = 'qomp-bank-execution-profile/1'
PRODUCER = 'qomp'
POLICY = 'qomp-estimated-inodes65536-v1'
FILES_KIND = 'engineering_estimate_not_hard_cap'
FILES_PER_ATTEMPT = 65536
SCRATCH = '/storage/scratch1/3/awallace43'
NAMES = ('s20','s40','s60','s80','s120')
_LEGACY_SHA = '19b4fcff756c31ddca113d12a617181383bfcb4f90064b2693ea13b94d1f6d63'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def helper(name):
    need(name in ('grid400_bank_policy','descriptor_scratch_cb63305'), 'closed quota/policy helper')
    path = Path(__file__).with_name(name+'.py')
    raw = path.read_bytes()
    if name == 'grid400_bank_policy':
        need(hashlib.sha256(raw).hexdigest() == _LEGACY_SHA, 'historical shape policy bytes differ')
    spec = importlib.util.spec_from_file_location('_inode147_'+name,path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(raw,str(path),'exec'),module.__dict__)
    return module


def execution_profile(name):
    need(type(name) is str and name in NAMES, 'closed qomp execution shape required')
    # All science/byte/memory/core/time values remain the exact historical ones.
    return dict(helper('grid400_bank_policy').execution_profile(name),
        schema=SCHEMA, producer=PRODUCER, policy_id=POLICY, name='qomp-bank-'+name,
        scratch_files=FILES_PER_ATTEMPT, scratch_files_kind=FILES_KIND)


def validate_profile(profile):
    need(type(profile) is dict, 'exact typed qomp execution profile required')
    for name in NAMES:
        expected = execution_profile(name)
        if (set(profile)==set(expected) and all(type(profile[k]) is type(v) and profile[k]==v
                                               for k,v in expected.items())):
            return expected
    raise ValueError('exact typed qomp execution profile required')


def admit_scratch(profiles, observe):
    """Immediate, pure admission seam; caller supplies NEW live reads per call.

    Use once for preadmission and AGAIN immediately before release. observe()
    must collect filesystem, same-device and BOTH raw quotas during this call,
    marking completion with this process's monotonic clock. Old serialized
    snapshots/receipts are not inputs. No collector, job control or release
    implementation is supplied here. Ordinary trusted-caller/OS boundary.
    """
    need(type(profiles) in (list,tuple) and 1 <= len(profiles) <= 8,
         'one to eight closed qomp members required')
    profiles = [validate_profile(p) for p in profiles]
    need(callable(observe), 'fresh local observation callback required')
    begin = time.monotonic()
    observation = observe()
    end = time.monotonic()
    need(type(observation) is dict and set(observation)=={
        'filesystem','scratch_device','quota_device','observed_at',
        'free_bytes','free_files','user_quota','group_quota'}, 'exact live scratch observation required')
    stamp = observation['observed_at']
    need(type(stamp) is float and math.isfinite(stamp) and begin <= stamp <= end,
         'stale/future/foreign-clock scratch observation')
    need(type(observation['filesystem']) is str and observation['filesystem']==SCRATCH
         and type(observation['scratch_device']) is int and observation['scratch_device']>0
         and type(observation['quota_device']) is int
         and observation['scratch_device']==observation['quota_device'],
         'owned scratch and quota filesystem devices differ')
    guard = helper('descriptor_scratch_cb63305')
    quotas = {}
    for owner in ('user','group'):
        text = observation[owner+'_quota']
        need(type(text) is str and 0 < len(text) <= 65536, 'bounded raw live quota text required')
        quotas[owner+'_quota'] = guard.parse_quota(text,SCRATCH)
    common = dict(free_bytes=observation['free_bytes'],free_files=observation['free_files'],**quotas)
    for p in profiles:
        guard.admit_scratch(**common,required_bytes=p['scratch_bytes'],required_files=p['scratch_files'])
    result = guard.admit_scratch(**common,
        required_bytes=sum(p['scratch_bytes'] for p in profiles)+max(p['scratch_bytes'] for p in profiles),
        required_files=sum(p['scratch_files'] for p in profiles)+max(p['scratch_files'] for p in profiles))
    return dict(result, schema='qomp-estimated-scratch-snapshot/1', policy_id=POLICY,
        member_count=len(profiles), observed_at=stamp,
        qualification='Live estimated headroom only, not reservation/creation cap/production adoption. '
                      'Repeat fresh reads immediately before release; concurrent usage can change.')
