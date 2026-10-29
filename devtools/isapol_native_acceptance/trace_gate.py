"""Decoded-fd scientific input gate; normal OS trust, not a hostile-host sandbox."""
import hashlib
import os
from pathlib import Path
import re


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def beneath(path, root):
    return path == root or path.startswith(root+'/')


def assembled_records(lines):
    pending, records = {}, []
    for line in lines:
        match = re.match(r'^(\d+)\s+(.*)$', line)
        if not match:
            raise ValueError('trace record has no PID')
        pid, text = match.groups()
        if '<unfinished ...>' in text:
            if pid in pending:
                raise ValueError('overlapping unfinished syscalls')
            pending[pid] = text.split('<unfinished ...>', 1)[0]
            continue
        resumed = re.match(r'<\.\.\. (\w+) resumed>(.*)', text)
        if resumed:
            if pid not in pending or not pending[pid].startswith(resumed[1]+'('):
                raise ValueError('unmatched resumed syscall')
            text = pending.pop(pid)+resumed[2]
        records.append((pid, text))
    if pending:
        raise ValueError('incomplete trace: unfinished syscalls remain')
    return records


def check_trace(lines, *, env, base, manifest_path, manifest, run, forbidden,
                allowed_inputs=None, scratch=None):
    """Use successful-open resolved fd targets, covering relative/dirfd aliases.

    Source/tests/history cannot become inputs merely by changing their path
    spelling. Only sealed installed dependencies, explicit geometry/provenance,
    fresh own-run files and ordinary OS inputs are allowed.
    """
    env, base, run = (os.path.normpath(str(p)) for p in (env, base, run))
    forbidden = tuple(os.path.normpath(str(p)) for p in forbidden)
    runtime_files, runtime_dirs = manifest['runtime_files'], set(manifest['runtime_directories'])
    external = manifest.get('external_runtime_files', {})
    allowed_inputs = {os.path.normpath(str(p)): value for p, value in (allowed_inputs or {}).items()}
    system_dirs = ('/usr/lib', '/usr/lib64', '/lib', '/lib64',
                   '/usr/share/zoneinfo', '/usr/share/locale', '/sys/devices/system/cpu')
    system_files = {
        '/etc/ld.so.cache', '/etc/localtime', '/etc/timezone', '/etc/nsswitch.conf',
        '/etc/passwd', '/etc/group', '/etc/hosts', '/etc/resolv.conf', '/etc/gai.conf',
        '/etc/os-release', '/etc/machine-id', '/proc/cpuinfo', '/proc/meminfo',
        '/proc/stat', '/proc/sys/vm/overcommit_memory', '/dev/null', '/dev/urandom', '/dev/random'}
    manifest_path = os.path.normpath(str(manifest_path))
    scratch = os.path.normpath(str(scratch)) if scratch is not None else None
    accesses, disallowed, executions = [], [], []
    verified, registrations = {}, set()
    for pid, text in assembled_records(lines):
        if re.match(r'execve(?:at)?\(', text):
            executions.append(dict(pid=pid, syscall=text))
            continue
        for quoted in re.findall(r'"((?:\\.|[^"\\])*)"', text):
            if quoted.startswith('/') and '\\' not in quoted:
                candidate = os.path.normpath(quoted)
                if any(beneath(candidate, p) for p in forbidden):
                    disallowed.append(dict(pid=pid, path=candidate, syscall=text))
        if not re.match(r'(?:open|openat|openat2|creat)\(', text):
            continue
        returned = re.search(r'\)\s+=\s+(-?\d+)(?:<([^>]+)>)?', text)
        if not returned:
            raise ValueError('unparseable open return')
        if int(returned[1]) < 0:
            continue
        target = returned[2]
        if not target or not target.startswith('/') or '\\' in target:
            raise ValueError('successful open lacks decoded absolute fd target')
        target = os.path.normpath(target)
        accesses.append(target)
        if any(beneath(target, p) for p in forbidden):
            disallowed.append(dict(pid=pid, path=target, syscall=text))
            continue
        if (beneath(target, run) or target == manifest_path
                or (scratch is not None and beneath(target, scratch))):
            continue
        if target in allowed_inputs:
            if digest(target) != allowed_inputs[target]:
                disallowed.append(dict(pid=pid, path=target, reason='declared scientific input changed'))
            continue
        if target == '/dev/shm' and 'O_DIRECTORY' in text:
            continue
        if re.fullmatch(r'/dev/shm/__KMP_REGISTERED_LIB_'+pid+r'_\d+', target):
            if 'O_CREAT' in text and 'O_EXCL' in text and 'O_NOFOLLOW' in text:
                registrations.add(target)
            if target in registrations and 'O_NOFOLLOW' in text:
                continue
        if target == '/proc/'+pid+'/environ':
            continue
        if beneath(target, env):
            relative = os.path.relpath(target, env)
            if relative in runtime_dirs:
                continue
            if relative not in runtime_files:
                disallowed.append(dict(pid=pid, path=target, reason='unsealed runtime input'))
                continue
            verified.setdefault(target, digest(target))
            if verified[target] != runtime_files[relative]:
                disallowed.append(dict(pid=pid, path=target, reason='runtime file hash changed'))
            continue
        if beneath(target, base+'/lib/python3.13/site-packages'):
            disallowed.append(dict(pid=pid, path=target, reason='base site packages forbidden'))
            continue
        resolved = str(Path(target).resolve())
        if resolved in external:
            verified.setdefault(resolved, digest(resolved))
            if verified[resolved] != external[resolved]:
                disallowed.append(dict(pid=pid, path=target, reason='external runtime hash changed'))
            continue
        if beneath(target, base):
            if 'O_DIRECTORY' in text:
                continue
            disallowed.append(dict(pid=pid, path=target, reason='unsealed external runtime input'))
            continue
        if target in system_files or any(beneath(target, p) for p in system_dirs):
            continue
        if re.fullmatch(r'/proc/(?:self|\d+)/(?:stat|status|maps|smaps|cmdline|statm|task(?:/\d+/(?:stat|status))?)', target):
            continue
        disallowed.append(dict(pid=pid, path=target, reason='outside declared runtime/OS/fresh-run inputs'))
    initial_only = (len(executions) == 1
                    and executions[0]['syscall'].startswith('execve("'+env+'/bin/python",')
                    and re.search(r'\)\s+=\s+0$', executions[0]['syscall']) is not None)
    return dict(
        status='passed' if not disallowed and initial_only else 'failed',
        disallowed_accesses=disallowed, executions=executions,
        only_initial_python_executed=initial_only,
        successful_open_targets=sorted(set(accesses)), verified_runtime_files=verified,
        fresh_openmp_registrations=sorted(registrations),
        trust_boundary='Sealed dependencies, explicit geometry and own fresh files; ordinary OS trust, no hostile-host/TOCTOU proof.')
