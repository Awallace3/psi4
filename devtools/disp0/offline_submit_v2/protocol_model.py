"""OFFLINE MODEL ONLY. No CLI, SSH, subprocess, Slurm or live adapter.

The real attempt remains UNKNOWN and its approval RESERVED. This module
cannot settle that history or authorize a replacement.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class Intent:
    """Reviewed fixture bytes, not evidence that a real approval exists."""

    artifact_parent: str
    attempt_name: str
    source_pin: str
    batch_sha256: str
    payload_sha256: str
    preflight_policy_sha256: str
    authorization_decision: str
    ambiguity_handling_decision: str
    owner_turn: str
    argv: tuple[str, ...] = ("OFFLINE_FAKE_RPC",)
    cluster: str = "OFFLINE_CLUSTER"
    cpus: int = 1
    memory_mib: int = 2048
    wall_seconds: int = 600
    gpus: int = 0
    requeue: bool = False

    def bytes(self):
        return encoded(asdict(self))

    def digest(self):
        return sha(self.bytes())

    def validate_fixture(self):
        if not Path(self.artifact_parent).is_absolute():
            raise ValueError("Fixture intent must bind an absolute artifact parent")
        if not re.fullmatch("[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", self.attempt_name):
            raise ValueError("Invalid bound fixture namespace")
        if self.cluster != "OFFLINE_CLUSTER":
            raise ValueError("Only the fixed OFFLINE cluster is accepted")
        for value in (self.authorization_decision, self.ambiguity_handling_decision, self.owner_turn):
            if not value.startswith("OFFLINE:") or len(value) == len("OFFLINE:"):
                raise ValueError("Only explicit OFFLINE fixture references are accepted")
        if not re.fullmatch("[0-9a-f]{40}", self.source_pin):
            raise ValueError("Malformed fixture source pin")
        for value in (self.batch_sha256, self.payload_sha256, self.preflight_policy_sha256):
            if not re.fullmatch("[0-9a-f]{64}", value):
                raise ValueError("Malformed fixture hash")
        if not isinstance(self.argv, tuple) or self.argv != ("OFFLINE_FAKE_RPC",):
            raise ValueError("No live command is accepted by this model")
        if (any(type(value) is not int for value in
                (self.cpus, self.memory_mib, self.wall_seconds, self.gpus))
                or type(self.requeue) is not bool):
            raise ValueError("Resource policy requires exact numeric/boolean types")
        if (self.cpus, self.memory_mib, self.wall_seconds, self.gpus, self.requeue) != (
                1, 2048, 600, 0, False):
            raise ValueError("Resource policy drift")


@dataclass(frozen=True)
class Reply:
    returncode: int = 0
    stdout: bytes = b"123;OFFLINE_CLUSTER\n"
    stderr: bytes = b""


@dataclass(frozen=True)
class Outcome:
    state: str
    candidate_id: str | None = None
    id_durable: bool = False
    error: str | None = None
    scope: str = "OFFLINE_MODEL_ONLY_NOT_REAL_JOB_EVIDENCE"


class OfflineScheduler:
    """In-memory fake. No command vector is ever executed."""

    def __init__(self, *, reply=Reply(), preflight_ok=True, lose_reply=False, query_fails=False):
        self.reply = reply
        self.preflight_ok = preflight_ok
        self.lose_reply = lose_reply
        self.query_fails = query_fails
        self.submit_calls = self.query_calls = self.fake_acceptances = 0

    def preflight(self, intent, event):
        event("preflight.begin")
        result = {"scope": "OFFLINE", "ok": self.preflight_ok, "intent_sha256": intent.digest()}
        event("preflight.done")
        return result

    def submit(self, intent, event):
        self.submit_calls += 1
        event("rpc.begin")
        if self.lose_reply:
            self.fake_acceptances += 1
            event("rpc.fake_acceptance_reply_lost")
            raise TimeoutError("OFFLINE simulated accepted RPC with lost reply")
        event("rpc.return")
        return self.reply

    def query(self, job_id, event):
        self.query_calls += 1
        event("query.begin")
        if self.query_fails:
            raise TimeoutError("OFFLINE simulated query failure")
        return {"scope": "OFFLINE", "job_id": job_id, "state": "SIMULATED_ONLY"}


class _ExistingAttempt(Exception):
    pass


class _Journal:
    def __init__(self, parent, name, event):
        parent = Path(parent)
        if not parent.is_absolute() or parent.resolve(strict=True) != parent:
            raise ValueError("Fixture parent must already exist and be canonical")
        if not re.fullmatch("[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", name):
            raise ValueError("Invalid fixture attempt name")
        self.event = event
        self.fd = None
        parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            info = os.fstat(parent_fd)
            if info.st_uid != os.getuid() or info.st_mode & 0o022:
                raise ValueError("Fixture parent must be private/owned")
            try:
                os.mkdir(name, mode=0o700, dir_fd=parent_fd)
            except FileExistsError as error:
                raise _ExistingAttempt("Refuse existing attempt; NEVER adopt/resume/replay") from error
            event("attempt.mkdir")
            event("attempt.before_parent_fsync")
            os.fsync(parent_fd)
            event("attempt.parent_fsync")
            self.fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
        finally:
            os.close(parent_fd)

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def write_once(self, name, data):
        if not re.fullmatch("[a-z0-9_.-]+", name) or len(data) > 65536:
            raise ValueError("Invalid/oversized model record")
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=self.fd)
        try:
            view = memoryview(data)
            while view:
                count = os.write(fd, view)
                if count <= 0:
                    raise OSError("Short record write")
                view = view[count:]
            os.fchmod(fd, 0o400)
            self.event(name + ".written")
            self.event(name + ".before_file_fsync")
            os.fsync(fd)
            self.event(name + ".file_fsync")
        finally:
            os.close(fd)
        self.event(name + ".before_directory_fsync")
        os.fsync(self.fd)
        self.event(name + ".directory_fsync")

    def read(self, name):
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=self.fd)
        try:
            data = os.read(fd, 65537)
            if len(data) > 65536:
                raise ValueError("Oversized record")
            return data
        finally:
            os.close(fd)


def run_model_once(parent, name, intent, reviewed_digest, scheduler, *, trace=None, checkpoint=None):
    """One synchronous invocation; never asks for input or resumes an old attempt.

    Checkpoints support LOCAL fault/crash injection only. There is deliberately
    no supported live scheduler adapter.
    """
    if type(scheduler) is not OfflineScheduler:
        raise TypeError("Only the in-memory OFFLINE scheduler is supported")
    events = trace if trace is not None else []

    def event(value):
        events.append(value)
        if checkpoint:
            checkpoint(value)

    intent.validate_fixture()
    data = intent.bytes()
    if sha(data) != reviewed_digest:
        raise ValueError("Intent differs from previously reviewed bytes")
    if str(Path(parent)) != intent.artifact_parent or name != intent.attempt_name:
        raise ValueError("Invocation namespace differs from reviewed intent")
    journal = None
    dispatched = False
    candidate = None
    id_durable = False
    try:
        journal = _Journal(parent, name, event)
        journal.write_once("intent.json", data)
        report = scheduler.preflight(intent, event)
        journal.write_once("preflight.json", encoded(report))
        if report != {"scope": "OFFLINE", "ok": True, "intent_sha256": reviewed_digest}:
            return Outcome("PREFLIGHT_STOP")
        if sha(journal.read("intent.json")) != reviewed_digest:
            raise ValueError("Stored reviewed intent changed")
        event("intent.revalidated")
        journal.write_once("rpc-intent.json", encoded({
            "scope": "OFFLINE", "intent_sha256": reviewed_digest,
            "rule": "potentially dispatched after this durable record; NEVER REPLAY"}))
        dispatched = True
        reply = scheduler.submit(intent, event)
        journal.write_once("rpc.stdout", reply.stdout)
        journal.write_once("rpc.stderr", reply.stderr)
        journal.write_once("rpc-result.json", encoded({
            "returncode": reply.returncode, "stdout_sha256": sha(reply.stdout),
            "stderr_sha256": sha(reply.stderr), "intent_sha256": reviewed_digest}))
        match = re.fullmatch(rb"([1-9][0-9]{0,19})(?:;([A-Za-z0-9_-]+))?\n?", reply.stdout)
        if reply.returncode != 0 or not match:
            return Outcome("SUBMISSION_UNRESOLVED")
        cluster = match.group(2)
        if cluster is not None and cluster.decode() != intent.cluster:
            return Outcome("SUBMISSION_UNRESOLVED", error="Unexpected reply cluster")
        candidate = match.group(1).decode()
        journal.write_once("job-id.json", encoded({
            "scope": "OFFLINE", "job_id": candidate, "cluster": intent.cluster,
            "intent_sha256": reviewed_digest, "stdout_sha256": sha(reply.stdout)}))
        id_durable = True
        query = scheduler.query(candidate, event)
        journal.write_once("query.json", encoded(query))
        return Outcome("COMPLETE", candidate, True)
    except _ExistingAttempt as error:
        return Outcome("REFUSED_EXISTING_ATTEMPT", error=str(error))
    except Exception as error:
        state = ("ACCEPTED_QUERY_UNRESOLVED" if id_durable else
                 "ACK_RECEIPT_INCOMPLETE" if candidate else
                 "SUBMISSION_UNRESOLVED" if dispatched else "PREFLIGHT_STOP")
        return Outcome(state, candidate, id_durable, type(error).__name__ + ": " + str(error))
    finally:
        if journal is not None:
            journal.close()
