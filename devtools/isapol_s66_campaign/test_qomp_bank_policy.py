"""Pure execution-policy/role/quota tests; no native imports or runtime hashes."""
import importlib.util
import json
from pathlib import Path
import sys
import time

import pytest


def load(name):
    path = Path(__file__).with_name(name+'.py')
    spec = importlib.util.spec_from_file_location('_inode147_'+name,path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__)
    return module


def expected_profile(name='s20'):
    old = load('grid400_bank_policy').execution_profile(name)
    return dict(old, schema='qomp-bank-execution-profile/1', producer='qomp',
        policy_id='qomp-estimated-inodes65536-v1', name='qomp-bank-'+name,
        scratch_files=65536, scratch_files_kind='engineering_estimate_not_hard_cap')


def execution(env, scratch, suffix):
    argv = [env+'/bin/python','-I','-B','-X','pycache_prefix='+str(scratch/'cache'),
            env+'/bin/psi4',*suffix]
    return dict(syscall='execve('+json.dumps(argv[0])+', '+json.dumps(argv)+', []) = 0')


def test_bank_cli_explicitly_consumes_new_qomp_policy(tmp_path):
    gate = load('bank_runtime_gate')
    assert gate.cli_role(execution('/private/env',tmp_path,['--version']),
        env='/private/env',scratch=tmp_path,execution_profile=expected_profile(),
        producer='qomp') == 'discovery_version'
    with pytest.raises(ValueError):
        gate.cli_role(execution('/private/env',tmp_path,['--version']),
            env='/private/env',scratch=tmp_path,execution_profile=expected_profile())


def test_ordinary_cli_explicitly_selects_qomp_not_grid400(monkeypatch,tmp_path):
    rt = load('runtime_qomp')
    scratch = tmp_path/'ccasp-native-unit'
    scratch.mkdir(mode=0o700)
    for name in ('home','cache','cwd'):
        (scratch/name).mkdir(mode=0o700)
    monkeypatch.setattr(rt,'SCRATCH_ROOT',tmp_path)
    monkeypatch.setattr(rt,'verify_runtime_view',lambda _:None)
    result = rt.ordinary_cli_launch(b'fake',['--version'],scratch=scratch,
        execution_profile=expected_profile(), producer='qomp',
        inherited_threads=dict(OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1'))
    assert result['command'][-1]=='--version'


def test_decoded_fd_role_explicitly_selects_qomp(monkeypatch,tmp_path):
    fixtures = load('test_qualified_qomp')
    run,scratch,_,_,lines = fixtures.trace_fixture(monkeypatch,tmp_path)
    out = fixtures.rt.trace_projection(b'fake',lines,run=run,scratch=scratch,
        allowed_inputs={},forbidden=(),execution_profile=expected_profile(),producer='qomp')
    assert out['status']=='passed' and out['role']=='discovery_version'


@pytest.mark.parametrize('name',['s20','s40','s60','s80','s120'])
def test_new_profile_changes_only_identity_and_estimate_with_exact_types(name):
    policy,old = load('qomp_bank_policy'),load('grid400_bank_policy')
    profile = policy.execution_profile(name)
    assert profile==expected_profile(name)
    assert policy.validate_profile(profile)==profile
    original = old.execution_profile(name)
    changed = {'schema','producer','policy_id','name','scratch_files','scratch_files_kind'}
    assert {k:v for k,v in profile.items() if k not in changed} == {
        k:v for k,v in original.items() if k not in changed}
    assert profile['scratch_bytes']==256*1024**3
    assert original['scratch_files']==131072 and profile['scratch_files']==65536
    with pytest.raises(ValueError):
        policy.validate_profile(original)
    with pytest.raises(ValueError):
        old.validate_profile(profile)
    profile['scratch_files']=1
    assert policy.execution_profile(name)['scratch_files']==65536


@pytest.mark.parametrize('field,value',[
    ('cores',8.0),('cores',True),('science_memory_GiB',20.0),
    ('job_memory_GiB','32'),('scratch_bytes',float(256*1024**3)),
    ('scratch_files',65536.0),('scratch_files',131072),('scratch_files',True),
    ('concurrency',True),('time_limit','08:00:00'),('qos','inferno'),
    ('schema','qomp-bank-execution-profile/2'),('producer','grid400'),
    ('policy_id','future'),('name','grid400-bank-s20'),
    ('scratch_files_kind','hard_cap'),('account',None),('extra','unused')])
def test_modified_or_freeform_profiles_refuse(field,value):
    profile = expected_profile()
    profile[field]=value
    with pytest.raises(ValueError,match='typed qomp'):
        load('qomp_bank_policy').validate_profile(profile)


@pytest.mark.parametrize('value',[None,True,0,{},[],'latest','../grid400_bank_policy'])
def test_unclosed_shape_names_refuse(value):
    with pytest.raises(ValueError):
        load('qomp_bank_policy').execution_profile(value)


def quota(path, remaining_bytes, remaining_files, *, unlimited=False):
    assert remaining_bytes%1024==0
    # Deliberately nonzero usage; zero group limits mean no ceiling reported.
    used_kb,used_files = 123,321
    kb_limit=0 if unlimited else used_kb+remaining_bytes//1024
    file_limit=0 if unlimited else used_files+remaining_files
    return f'{path} {used_kb} {kb_limit} {kb_limit} - {used_files} {file_limit} {file_limit} -\n'


def observation(policy, count, **changes):
    required_bytes=(count+1)*256*1024**3
    required_files=(count+1)*65536
    result=dict(filesystem=policy.SCRATCH,scratch_device=77,quota_device=77,
        observed_at=time.monotonic(),free_bytes=required_bytes,free_files=required_files,
        user_quota=quota(policy.SCRATCH,required_bytes,required_files),
        group_quota=quota(policy.SCRATCH,required_bytes,required_files))
    return result|changes


@pytest.mark.parametrize('count',range(1,9))
def test_all_member_and_sum_largest_checks_heterogeneous_exact_boundary(monkeypatch,count):
    policy = load('qomp_bank_policy')
    profiles=[policy.execution_profile(policy.NAMES[i%5]) for i in range(count)]
    guard=policy.helper('descriptor_scratch_cb63305')
    original=guard.admit_scratch
    calls=[]
    def admit(**kw):
        calls.append((kw['required_bytes'],kw['required_files']))
        return original(**kw)
    guard.admit_scratch=admit
    helper=policy.helper
    monkeypatch.setattr(policy,'helper',lambda name: guard if name=='descriptor_scratch_cb63305'
                        else helper(name))
    result=policy.admit_scratch(profiles,lambda:observation(policy,count))
    assert calls==[(256*1024**3,65536)]*count+[((count+1)*256*1024**3,(count+1)*65536)]
    assert result['member_count']==count
    assert result['required_files']==(count+1)*65536
    assert 'not reservation/creation cap/production adoption' in result['qualification']
    assert 'science_memory_GiB' not in result  # Numeric caps aren't rescaled.


@pytest.mark.parametrize('owner',['filesystem','user','group'])
@pytest.mark.parametrize('resource',['bytes','files'])
def test_one_short_at_every_filesystem_user_group_boundary(owner,resource):
    policy=load('qomp_bank_policy')
    profiles=[policy.execution_profile('s20'),policy.execution_profile('s120')]
    def observe():
        o=observation(policy,2)
        if owner=='filesystem':
            o['free_'+resource]-=1
        else:
            b,f=3*256*1024**3,3*65536
            if resource=='files':
                o[owner+'_quota']=quota(policy.SCRATCH,b,f-1)
            else:
                # Raw Lustre granularity is KiB; use used123/limit so remaining
                # is1024 short, then exercise guard's byte-exact1-short below.
                o[owner+'_quota']=quota(policy.SCRATCH,b-1024,f)
        return o
    with pytest.raises(ValueError,match='insufficient'):
        policy.admit_scratch(profiles,observe)


@pytest.mark.parametrize('owner',['user','group'])
def test_reused_guard_byte_exact_one_short_no_custom_quota_engine(owner):
    guard=load('descriptor_scratch_cb63305')
    b,f=2*256*1024**3,2*65536
    observation=dict(free_bytes=b,free_files=f,
        user_quota=dict(remaining_bytes=b,remaining_files=f),
        group_quota=dict(remaining_bytes=b,remaining_files=f))
    guard.admit_scratch(**observation,required_bytes=b,required_files=f)
    observation[owner+'_quota']['remaining_bytes']-=1
    with pytest.raises(ValueError,match='insufficient'):
        guard.admit_scratch(**observation,required_bytes=b,required_files=f)


def test_zero_unbounded_group_still_requires_user_and_filesystem_headroom():
    policy=load('qomp_bank_policy')
    profiles=[policy.execution_profile('s60')]
    def observe(**changes):
        return observation(policy,1,group_quota=quota(policy.SCRATCH,0,0,unlimited=True),**changes)
    result=policy.admit_scratch(profiles,observe)
    assert result['group_quota']['remaining_bytes'] is None
    assert result['group_quota']['remaining_files'] is None
    with pytest.raises(ValueError):
        policy.admit_scratch(profiles,lambda:observe(free_files=1))
    with pytest.raises(ValueError):
        policy.admit_scratch(profiles,lambda:observe(
            user_quota=quota(policy.SCRATCH,0,0)))


@pytest.mark.parametrize('change',[
    dict(observed_at=-1.0),dict(observed_at=float('inf')),dict(observed_at=float('nan')),
    dict(observed_at=True),dict(observed_at=10**12),
    dict(filesystem='/other'),dict(scratch_device=78),dict(quota_device=True),
    dict(user_quota='stale quota unavailable'),dict(group_quota=''),
    dict(free_bytes=True),dict(free_files=0),dict(extra='not a recognized observation'),
])
def test_stale_future_malformed_or_other_filesystem_observations_refuse(change):
    policy=load('qomp_bank_policy')
    with pytest.raises((ValueError,KeyError)):
        policy.admit_scratch([policy.execution_profile('s20')],lambda:observation(policy,1,**change))


@pytest.mark.parametrize('text',[
    '{p} 1K 2 3 - 1 2 3 -\n', '{p} 1 2 3 -\n',
    '{p} 1 2 3 - 1 2 3 -\n{p} 1 2 3 - 1 2 3 -\n',
    '{p}\n', '{p} -1 0 0 - 1 0 0 -\n'])
def test_malformed_raw_quota_never_reinterpreted_as_unbounded(text):
    policy=load('qomp_bank_policy')
    with pytest.raises(ValueError):
        policy.admit_scratch([policy.execution_profile('s20')],
            lambda:observation(policy,1,group_quota=text.format(p=policy.SCRATCH)))


def test_release_requires_another_collection_not_reuse_of_prior_snapshot(monkeypatch):
    policy=load('qomp_bank_policy')
    clock=[100.0]
    monkeypatch.setattr(policy.time,'monotonic',lambda:clock[0])
    profiles=[policy.execution_profile('s20')]*8
    prior=observation(policy,8)
    assert policy.admit_scratch(profiles,lambda:prior)['required_files']==589824
    clock[0]=101.0
    with pytest.raises(ValueError,match='stale'):
        policy.admit_scratch(profiles,lambda:prior)
    refreshed=policy.admit_scratch(profiles,lambda:observation(policy,8))
    assert refreshed['observed_at']==101.0
    assert refreshed['required_bytes']==9*256*1024**3
    with pytest.raises(ValueError):
        policy.admit_scratch(profiles,prior)  # Not a live callback.


@pytest.mark.parametrize('count',[0,9])
def test_member_count_never_widens_wave(count):
    policy=load('qomp_bank_policy')
    with pytest.raises(ValueError):
        policy.admit_scratch([policy.execution_profile('s20')]*count,
                             lambda:observation(policy,count))


def test_cross_policy_member_cannot_enter_new_scratch_admission():
    policy=load('qomp_bank_policy')
    with pytest.raises(ValueError):
        policy.admit_scratch([load('grid400_bank_policy').execution_profile('s20')],
                             lambda:observation(policy,1))


@pytest.mark.parametrize('name',['s20','s40','s60','s80','s120'])
@pytest.mark.parametrize('role',['--module','--version','molecular'])
def test_all_new_cli_roles_use_new_policy_and_all_old_new_crossings_refuse(name,role,tmp_path):
    gate=load('bank_runtime_gate')
    profile=load('qomp_bank_policy').execution_profile(name)
    old=load('grid400_bank_policy').execution_profile(name)
    suffix=([role] if role!='molecular' else [
        '--scratch',str(tmp_path/'task'),'--nthread',str(profile['cores']),
        '--memory',str(float(profile['science_memory_GiB']))+'GiB','--qcschema','data.msgpack'])
    observed=execution('/unit/env',tmp_path,suffix)
    expected={'--module':'discovery_module','--version':'discovery_version','molecular':'molecular_qcschema'}[role]
    assert gate.cli_role(observed,env='/unit/env',scratch=tmp_path,
        execution_profile=profile,producer='qomp')==expected
    with pytest.raises(ValueError):
        gate.cli_role(observed,env='/unit/env',scratch=tmp_path,execution_profile=profile)
    with pytest.raises(ValueError):
        gate.cli_role(observed,env='/unit/env',scratch=tmp_path,execution_profile=old,producer='qomp')
    with pytest.raises(ValueError):
        gate.cli_role(observed,env='/unit/env',scratch=tmp_path,
            execution_profile=profile,producer='future')


def test_new_role_policy_cannot_hide_malformed_decoded_fd_trace(monkeypatch,tmp_path):
    fixtures=load('test_qualified_qomp')
    run,scratch,_,_,lines=fixtures.trace_fixture(monkeypatch,tmp_path)
    lines+=['100 open("relative", O_RDONLY) = 3']
    with pytest.raises(ValueError,match='decoded fd'):
        fixtures.rt.trace_projection(b'fake',lines,run=run,scratch=scratch,
            allowed_inputs={},forbidden=(),execution_profile=expected_profile(),producer='qomp')


def test_qomp_transitive_inventory_names_new_policy_and_shared_guards_only_explicitly():
    profiles=load('descriptor_profiles')
    selected=profiles.profile('qomp')
    needed={'qomp_bank_policy.py','bank_runtime_gate.py','grid400_bank_policy.py',
            'runtime_audit.py','descriptor_scratch_cb63305.py'}
    assert needed<=set(selected['child_sources'])
    assert needed<=set(selected['parent_extra_sources'])
    assert not needed.intersection(profiles.profile('grid400')['child_sources'])


def test_largest_ordinary_molecular_route_and_crossings_before_runtime_hash(monkeypatch,tmp_path):
    rt=load('runtime_qomp')
    scratch=tmp_path/'ccasp-native-unit'
    scratch.mkdir(mode=0o700)
    for name in ('home','cache','cwd'):
        (scratch/name).mkdir(mode=0o700)
    monkeypatch.setattr(rt,'SCRATCH_ROOT',tmp_path)
    checked=[]
    monkeypatch.setattr(rt,'verify_runtime_view',lambda _:checked.append('finite-check'))
    profile=load('qomp_bank_policy').execution_profile('s120')
    kwargs=dict(scratch=scratch,execution_profile=profile,
        inherited_threads=dict(OMP_NUM_THREADS='24',MKL_NUM_THREADS='24',OPENBLAS_NUM_THREADS='1'))
    suffix=['--scratch',str(scratch/'task'),'--nthread','24',
            '--memory','120.0GiB','--qcschema','data.msgpack']
    out=rt.ordinary_cli_launch(b'fake',suffix,producer='qomp',**kwargs)
    assert out['command'][-8:]==suffix and checked==['finite-check']
    with pytest.raises(ValueError):
        rt.ordinary_cli_launch(b'fake',suffix,**kwargs)
    kwargs['execution_profile']=load('grid400_bank_policy').execution_profile('s120')
    with pytest.raises(ValueError):
        rt.ordinary_cli_launch(b'fake',suffix,producer='qomp',**kwargs)
    assert checked==['finite-check']  # Rejected crossings performed NO runtime hash.


def test_largest_decoded_fd_molecular_role_and_both_policy_crossings(monkeypatch,tmp_path):
    f=load('test_qualified_qomp')
    run,scratch,_,prefix,_=f.trace_fixture(monkeypatch,tmp_path)
    suffix=['--scratch',str(scratch/'task'),'--nthread','24',
            '--memory','120.0GiB','--qcschema','data.msgpack']
    lines=['100 '+execution(str(prefix),scratch,suffix)['syscall']]
    kwargs=dict(run=run,scratch=scratch,allowed_inputs={},forbidden=(),
                execution_profile=load('qomp_bank_policy').execution_profile('s120'))
    out=f.rt.trace_projection(b'fake',lines,producer='qomp',**kwargs)
    assert out['status']=='passed' and out['role']=='molecular_qcschema'
    with pytest.raises(ValueError):
        f.rt.trace_projection(b'fake',lines,**kwargs)
    kwargs['execution_profile']=load('grid400_bank_policy').execution_profile('s120')
    with pytest.raises(ValueError):
        f.rt.trace_projection(b'fake',lines,producer='qomp',**kwargs)
