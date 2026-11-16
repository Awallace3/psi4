"""Loopback-only QCA 0.71 bridge; no credentials or dataset writes on disk."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import subprocess
import time

LOGIN = 'login-phoenix-gnr-2.pace.gatech.edu'
LOGIN_PORT = 27777
CONFIG = Path('/storage/project/r-cs207-0/awallace43/qca/resources_losii_ac.yml')
ORIGINAL_URI = 'http://128.61.254.96:7777'
SERVER_NAME = 'CDS QCFractal Server'
SERVER_VERSION = '0.71'


def server_config(uri):
    """Read the existing private config in place; never serialize its contents."""
    import yaml
    from urllib.parse import urlsplit
    address = urlsplit(uri)
    if (address.scheme != 'http' or address.hostname != '127.0.0.1'
            or address.username or address.password or address.path
            or address.query or address.fragment or address.port is None):
        raise ValueError('explicit loopback-only QCA forwarding address required')
    data = yaml.safe_load(CONFIG.read_text())['server']
    if (data.get('username') != 'awallace43' or data.get('fractal_uri') != ORIGINAL_URI
            or set(data) - {'fractal_uri', 'username', 'password', 'verify'}):
        raise ValueError('existing approved owner/deployment config required')
    return dict(data, fractal_uri=uri)


def validate_information(info):
    if info.get('name') != SERVER_NAME or info.get('version') != SERVER_VERSION:
        raise ValueError('expected CDS QCA 0.71 deployment required')
    # The server, not a guessed equality of package versions, declares support.
    from importlib.metadata import version
    from packaging.version import Version
    for package, role in (('qcportal', 'client'), ('qcfractalcompute', 'manager')):
        actual = Version(version(package))
        lower = Version(info[role + '_version_lower_limit'])
        upper = Version(info[role + '_version_upper_limit'])
        if not lower <= actual <= upper:
            raise ValueError('installed ' + role + ' outside server compatibility range')
    return dict(name=info['name'], version=info['version'],
                qcportal=version('qcportal'), qcfractalcompute=version('qcfractalcompute'))


@contextmanager
def forward(directory):
    """One job-owned SSH child to the already established Hornet reverse hop."""
    directory = Path(directory)
    directory.mkdir(mode=0o700, exist_ok=False)
    if directory.resolve() != directory or directory.stat().st_uid != os.getuid():
        raise ValueError('private canonical owned tunnel directory required')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    command = ['/usr/bin/ssh', '-N', '-T', '-o', 'BatchMode=yes',
               '-o', 'ControlMaster=no', '-o', 'ControlPath=none',
               '-o', 'ExitOnForwardFailure=yes', '-o', 'ConnectTimeout=15',
               '-o', 'StrictHostKeyChecking=yes', '-o', 'ServerAliveInterval=30',
               '-o', 'ServerAliveCountMax=3', '-L',
               f'127.0.0.1:{port}:127.0.0.1:{LOGIN_PORT}', LOGIN]
    with (directory/'ssh.log').open('xb') as log:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            deadline = time.monotonic() + 30
            while True:
                if child.poll() is not None:
                    raise RuntimeError('QCA forwarding SSH exited; inspect owned ssh.log')
                try:
                    with socket.create_connection(('127.0.0.1', port), timeout=.2):
                        break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('QCA forwarding listener did not become ready')
                    time.sleep(.1)
            uri = f'http://127.0.0.1:{port}'
            with (directory/'route.json').open('x') as stream:
                json.dump(dict(schema='qca071-loopback-route/1', uri=uri,
                               login=LOGIN, login_port=LOGIN_PORT,
                               ssh_pid=child.pid, job_id=os.environ.get('SLURM_JOB_ID')),
                          stream, sort_keys=True)
            yield uri
            if child.poll() is not None:
                raise RuntimeError('QCA forwarding SSH died during the operation')
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
