"""Eligible-bank CLI role adapter; historical FD/CLI gates remain unchanged."""
import importlib.util
import json
from pathlib import Path
import re


def helper(name):
    if name not in ('runtime_audit', 'grid400_bank_policy'):
        raise ValueError('closed eligible-bank CLI gate helpers required')
    path = Path(__file__).with_name(name+'.py')
    spec = importlib.util.spec_from_file_location('_bank_runtime_gate_'+name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cli_role(execution, *, env, scratch, execution_profile):
    """Classify the actual observed command; never rewrite trace bytes."""
    profile = helper('grid400_bank_policy').validate_profile(execution_profile)
    shape = (profile['cores'], profile['science_memory_GiB'])
    if shape != (24, 120):
        return helper('runtime_audit').cli_role(execution, env=env, scratch=scratch,
                                               worker_shape=shape)
    # The frozen historical classifier deliberately cannot admit 24/120.
    # Keep that source and its historical receipts byte-for-byte intact.
    env, scratch = str(env), str(scratch)
    match = re.fullmatch(r'execve\("([^"]+)", (\[.*\]), .*?\)\s+=\s+0', execution['syscall'])
    if match is None:
        raise ValueError('exact successful initial execve required')
    argv = json.loads(match[2])
    prefix = [env+'/bin/python', '-I', '-B', '-X',
              'pycache_prefix='+scratch+'/cache', env+'/bin/psi4']
    if match[1] != prefix[0] or argv[:6] != prefix:
        raise ValueError('foreign interpreter, CLI or import-cache boundary')
    rest = argv[6:]
    if rest == ['--module']:
        return 'discovery_module'
    if rest == ['--version']:
        return 'discovery_version'
    if (len(rest) == 8 and rest[0] == '--scratch'
            and str(Path(rest[1]).resolve()).startswith(scratch+'/')
            and rest[2:] == ['--nthread', '24', '--memory', '120.0GiB', '--qcschema', 'data.msgpack']):
        return 'molecular_qcschema'
    raise ValueError('foreign CLI execution mode, options or scratch input')
