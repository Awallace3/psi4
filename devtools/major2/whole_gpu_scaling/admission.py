"""Read-only, fail-closed admission for THIS campaign, not GPU custody proof."""
import datetime
import hashlib
import os
from pathlib import PurePosixPath
import pwd
import re
import selectors
import subprocess
import time

import config as s

OWNER = "awallace43"
CAMPAIGN = "major2-whole-gpu-scaling"
FORMAT = "%i|%u|%U|%T|%a|%q|%j|%P|%o|%Z"
HEADER = "JOBID|USER|UID|STATE|ACCOUNT|QOS|NAME|PARTITION|COMMAND|WORK_DIR"
MAX_BYTES = 2*1024**2
MAX_ROWS = 4096
STATES = {"PENDING", "RUNNING", "SUSPENDED", "COMPLETING", "CONFIGURING",
          "RESIZING", "REQUEUED", "REQUEUE_FED", "REQUEUE_HOLD", "SPECIAL_EXIT",
          "STOPPED", "STAGE_OUT"}


def query(*args):
    """Bound bytes AND duration while reading, including stderr; no shell/writes."""
    deadline = time.monotonic()+35
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        try:
            with selectors.DefaultSelector() as sel:
                sel.register(proc.stdout, selectors.EVENT_READ, "stdout")
                sel.register(proc.stderr, selectors.EVENT_READ, "stderr")
                while sel.get_map():
                    left = deadline-time.monotonic()
                    s.require(left > 0, "queue query timeout")
                    for key, _ in sel.select(min(left, .25)):
                        block = os.read(key.fileobj.fileno(), 65536)
                        if not block:
                            sel.unregister(key.fileobj)
                        else:
                            buffers[key.data].extend(block)
                            s.require(sum(map(len, buffers.values())) <= MAX_BYTES,
                                      "queue query output cap")
                left = deadline-time.monotonic()
                s.require(left > 0, "queue query timeout")
                code = proc.wait(timeout=left)
            s.require(code == 0 and not buffers["stderr"], "queue query failed/noisy")
            return buffers["stdout"].decode("utf-8", errors="strict")
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)


def duplicate_fields(name, command, workdir):
    """Name/source/request identity, including dated/versioned/partial attempts.

    The unique campaign stem in any identity field catches prior run/source
    namespaces and JobName variants. The owned helper directory also catches
    direct helper commands from differently named checkouts. Generic 'major2',
    partition, GPU type, account and QoS are NOT duplicate identities.
    """
    return [key for key, value in (("JobName", name), ("Command", command),
                                  ("WorkDir", workdir))
            if CAMPAIGN in value.lower() or
            "/devtools/major2/whole_gpu_scaling" in value.lower()]


def parse(text, uid):
    s.require(isinstance(text, str) and len(text.encode()) <= MAX_BYTES,
              "bounded queue text")
    lines = text.splitlines()
    # A successful header-only snapshot is explicit empty evidence. Empty output
    # or a partial final line is ambiguous, NEVER silently treated as no jobs.
    s.require(text.endswith("\n") and lines and lines[0] == HEADER and
              len(lines)-1 <= MAX_ROWS and "\r" not in text,
              "queue header/completeness/row bound")
    jobs, seen = [], set()
    for line in lines[1:]:
        s.require(len(line) <= 16384 and len(line.split("|")) == 10 and
                  not any(ord(c) < 32 for c in line), "malformed queue row")
        job, user, owner_uid, state, account, qos, name, partition, command, workdir = line.split("|")
        s.require(re.fullmatch(r"[1-9][0-9]*(?:_[0-9]+)?(?:\+[0-9]+)?", job) and
                  job not in seen, "queue job identity/expanded arrays")
        seen.add(job)
        s.require(user == OWNER and owner_uid == str(uid), "queue owner identity")
        s.require(state in STATES and all(re.fullmatch(r"[A-Za-z0-9_.-]+", x)
                  for x in (account, qos)) and re.fullmatch(r"[A-Za-z0-9_.,-]+", partition),
                  "queue route/state metadata")
        s.require(name.strip() == name and bool(name) and
                  name.lower() not in {"n/a", "(null)", "unknown"},
                  "queue name metadata")
        for path in (command, workdir):
            s.require(path.startswith("/") and ".." not in PurePosixPath(path).parts and
                      path.strip() == path, "queue command/workdir metadata")
        matches = duplicate_fields(name, command, workdir)
        s.require(not matches, "active whole-GPU campaign duplicate: "+job+" "+",".join(matches))
        jobs.append(dict(job_id=job, user=user, uid=owner_uid, state=state,
                         account=account, qos=qos, job_name=name, partition=partition,
                         command=command, workdir=workdir, classification="unrelated",
                         allocation_exclusivity_proven=False))
    used = sum(v["qos"] == "embers" for v in jobs)
    s.require(used < 50, "free embers headroom")
    return dict(campaign=CAMPAIGN, owner=OWNER, uid=uid, own_queue_rows=len(jobs),
                embers_used=used, embers_limit=50, embers_available=50-used,
                requested_slots=1, observations=jobs, duplicate_jobs=0,
                allocation_exclusivity_proven=False, no_scheduler_writes=True,
                snapshot_sha256=hashlib.sha256(text.encode()).hexdigest())


def queue():
    uid = os.getuid()
    s.require(pwd.getpwuid(uid).pw_name == OWNER, "effective admission owner")
    args = ("squeue", "-r", "-u", OWNER, "-o", FORMAT)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    value = parse(query(*args), uid)
    return dict(value, query=list(args), observed_at_utc=started,
                completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
