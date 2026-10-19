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
CPU_PARTITIONS = {"cpu-small", "cpu-medium", "cpu-large", "cpu-sas", "cpu-gnr", "cpu-amd"}
MAX_CPU_PROOFS = 8
PROOF_BYTES = 65536
PROOF_SECONDS = 5


def query(*args, timeout=35, max_bytes=None):
    """Bound bytes AND duration while reading, including stderr; no shell/writes."""
    deadline = time.monotonic()+timeout
    cap = MAX_BYTES if max_bytes is None else max_bytes
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
                            s.require(sum(map(len, buffers.values())) <= cap,
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


def cpu_tres(value, required):
    """Conservative known CPU-only TRES grammar, not a GPU-name blacklist."""
    fields = {}
    for item in value.split(","):
        match = re.fullmatch(r"(cpu|mem|node|billing)=([1-9][0-9]{0,18})([KMGTPE]?)", item)
        s.require(match is not None, "CPU proof malformed/unknown TRES")
        key, number, unit = match.groups()
        s.require(key not in fields and (key == "mem" or not unit) and
                  int(number)*1024**("KMGTPE".index(unit)+1 if unit else 0) < 2**63,
                  "CPU proof TRES duplicate/overflow/unit")
        fields[key] = item.split("=", 1)[1]
    s.require(not required or {"cpu", "mem", "node"} <= fields.keys(),
              "CPU proof missing CPU/memory/node")
    return fields


def controller_cpu_proof(row):
    args = ("scontrol", "show", "job", "-o", row["job_id"])
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    raw = query(*args, timeout=PROOF_SECONDS, max_bytes=PROOF_BYTES)
    s.require(isinstance(raw, str) and len(raw.encode()) <= PROOF_BYTES and
              raw.endswith("\n") and len(raw.splitlines()) == 1 and
              not any(ord(c) < 32 for c in raw[:-1]), "CPU proof bounded single controller row")
    # Values (notably SubmitLine) can contain spaces. Recognize field boundaries,
    # reject duplicate keys instead of allowing last-value identity substitution.
    matches = list(re.finditer(r"(?:^| )([A-Za-z][A-Za-z0-9_/:]*)=", raw[:-1]))
    s.require(matches and matches[0].start() == 0, "CPU proof controller fields")
    fields = {}
    for i, match in enumerate(matches):
        key = match[1]
        s.require(key not in fields, "CPU proof duplicate controller field")
        fields[key] = raw[match.end():matches[i+1].start() if i+1 < len(matches) else -1].strip()
    expected = dict(JobId=row["job_id"], UserId=f'{row["user"]}({row["uid"]})',
                    JobName=row["job_name"], WorkDir=row["workdir"],
                    Partition=row["partition"], JobState=row["state"],
                    Account=row["account"], QOS=row["qos"], Command="(null)")
    s.require(all(fields.get(k) == v for k, v in expected.items()) and
              fields.get("BatchFlag") in {"0", "1"} and
              fields["JobState"] in {"PENDING", "RUNNING"},
              "CPU proof exact identity/active state/command")
    cpu_tres(fields.get("ReqTRES", ""), True)
    allocated = fields.get("AllocTRES")
    if row["state"] == "RUNNING" or allocated not in {None, "", "(null)"}:
        cpu_tres(allocated or "", True)
    resource_keys = {"ReqTRES", "AllocTRES", "Gres", "GresDetail",
                     "TresPerJob", "TresPerNode", "TresPerSocket", "TresPerTask"}
    for key, value in fields.items():
        if key in {"ReqTRES", "AllocTRES"}:
            continue
        if "gres" in key.lower() or "tres" in key.lower() or "gpu" in key.lower():
            s.require(key in resource_keys, "CPU proof unknown resource field")
            if value not in {"", "(null)", "N/A"}:
                s.require(key.startswith("TresPer") and
                          re.fullmatch(r"cpu=[1-9][0-9]{0,18}", value),
                          "CPU proof GPU/ambiguous GRES/TresPer request")
                cpu_tres(value, False)
    selected = {k: fields[k] for k in sorted(set(expected) | resource_keys | {"BatchFlag"})
                if k in fields}
    return dict(query=list(args), returncode=0, stderr="", raw=raw,
                raw_sha256=hashlib.sha256(raw.encode()).hexdigest(), raw_bytes=len(raw.encode()),
                timeout_seconds=PROOF_SECONDS, max_bytes=PROOF_BYTES, selected=selected,
                missing_fields=sorted(resource_keys-fields.keys()), observed_at_utc=started,
                completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())


def parse(text, uid, cpu_proof=None):
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
        for path in ((workdir,) if command == "(null)" else (command, workdir)):
            s.require(path.startswith("/") and ".." not in PurePosixPath(path).parts and
                      path.strip() == path, "queue command/workdir metadata")
        matches = duplicate_fields(name, command, workdir)
        s.require(not matches, "active whole-GPU campaign duplicate: "+job+" "+",".join(matches))
        if command == "(null)":
            s.require(cpu_proof is not None and partition in CPU_PARTITIONS and
                      state in {"PENDING", "RUNNING"}, "null Command requires targeted CPU proof")
        jobs.append(dict(job_id=job, user=user, uid=owner_uid, state=state,
                         account=account, qos=qos, job_name=name, partition=partition,
                         command=command, workdir=workdir, classification="unrelated",
                         allocation_exclusivity_proven=False))
    used = sum(v["qos"] == "embers" for v in jobs)
    s.require(used < 50, "free embers headroom")
    candidates = [v for v in jobs if v["command"] == "(null)"]
    s.require(len(candidates) <= MAX_CPU_PROOFS, "targeted CPU proof candidate bound")
    # Validate the ENTIRE snapshot/headroom before any targeted controller query.
    for row in candidates:
        row["controller_proof"] = cpu_proof(row)
        row["command_available"] = False
        row["classification"] = "unrelatedCPUproven-no-GPU"
    return dict(campaign=CAMPAIGN, owner=OWNER, uid=uid, own_queue_rows=len(jobs),
                embers_used=used, embers_limit=50, embers_available=50-used,
                requested_slots=1, observations=jobs, duplicate_jobs=0,
                targeted_cpu_queries=len(candidates), targeted_cpu_query_limit=MAX_CPU_PROOFS,
                allocation_exclusivity_proven=False, no_scheduler_writes=True,
                snapshot_sha256=hashlib.sha256(text.encode()).hexdigest())


def queue():
    uid = os.getuid()
    s.require(pwd.getpwuid(uid).pw_name == OWNER, "effective admission owner")
    args = ("squeue", "-r", "-u", OWNER, "-o", FORMAT)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    value = parse(query(*args), uid, cpu_proof=controller_cpu_proof)
    return dict(value, query=list(args), observed_at_utc=started,
                completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
