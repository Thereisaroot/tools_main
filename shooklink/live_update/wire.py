"""Authenticated live probes and a single, release-specific activation operation."""

from __future__ import annotations

import queue
import sys
import threading
import uuid

from shooklink import BUILD_REVISION
from shooklink.protocol.messages import Message, MessageType
from shooklink.transport.multiplexer import Priority
from .release import FEATURE, Release, hex_id


class Unavailable(RuntimeError):
    """No live reply. An earlier apply may already have reached the peer."""


class Refused(RuntimeError):
    """A definite local or peer refusal."""


class UpdateService:
    def __init__(self, core, receiver=None):
        self.core = core
        self.receiver = receiver
        self.revision = BUILD_REVISION
        self.instance = uuid.uuid4().hex
        self._lock = threading.Lock()
        self._pending = {}

    def exchange(self, kind, fields, *, peer=None, peer_id=None, timeout=3.0):
        snapshot = self.core.snapshot
        if (peer and snapshot.fingerprint and peer != snapshot.fingerprint
                or peer_id and snapshot.peer_id and peer_id != snapshot.peer_id):
            raise Refused("connected peer identity changed")
        if snapshot.state.value != "ready":
            raise Unavailable("waiting for the trusted peer to reconnect")
        if FEATURE not in snapshot.features:
            raise Unavailable("peer needs the managed bootstrap ZIP (use send-update first)")
        nonce = uuid.uuid4().hex
        inbox = queue.Queue(1)
        with self._lock:
            self._pending[nonce] = ((snapshot.fingerprint, snapshot.peer_id), inbox)
        try:
            try:
                self.core.send(Message(kind, dict(fields, nonce=nonce)), secure=True,
                               priority=Priority.INTERACTIVE, expected_connection_id=snapshot.connection_id)
            except (RuntimeError, OSError) as error:
                raise Unavailable(str(error)) from error
            try:
                reply = inbox.get(timeout=timeout)
            except queue.Empty as error:
                raise Unavailable("trusted peer did not answer the live update probe") from error
            if reply.get("ok") is not True:
                raise Refused(str(reply.get("error", "peer refused request"))[:300])
            if not isinstance(reply.get("result"), dict):
                raise Refused("invalid update report")
            return dict(reply["result"], fingerprint=snapshot.fingerprint, peer_id=snapshot.peer_id)
        finally:
            with self._lock:
                self._pending.pop(nonce, None)

    def idle_for_update(self):
        return (not self.core.files.busy and self.core.shell.active_session_id is None
                and (self.core.input is None or self.core.input.state.value == "idle"))

    def tick(self) -> bool:
        snapshot = self.core.snapshot
        return self.receiver.tick(
            snapshot.fingerprint if snapshot.state.value == "ready" else None, self.idle_for_update()
        )

    def _report(self):
        details = (self.receiver.report(self.core.snapshot.fingerprint) if self.receiver
                   else {"managed": False, "phase": "unmanaged"})
        return dict(details, revision=self.revision, instance=self.instance,
                    platform=sys.platform, idle=self.idle_for_update())

    def handle_message(self, message):
        if not self.core.trusted or self.core.decrypt_secure(message) != b"":
            return
        snapshot = self.core.snapshot
        fields = message.metadata
        nonce = fields.get("nonce")
        if not hex_id(nonce, 32) or snapshot.connection_id != self.core.chat_connection_id:
            return
        if message.message_type is MessageType.UPDATE_REPORT:
            with self._lock:
                pending = self._pending.get(nonce)
                if pending and pending[0] == (snapshot.fingerprint, snapshot.peer_id):
                    try:
                        pending[1].put_nowait(fields)
                    except queue.Full:
                        pass
            if pending is None and fields.get("ok") is True:
                result = fields.get("result")
                revision = result.get("revision") if isinstance(result, dict) else None
                if revision == "checkout" or hex_id(revision, 40) or hex_id(revision, 64):
                    self.core.record_peer_revision(snapshot.connection_id, nonce, revision)
            return
        try:
            job = fields.get("job")
            if message.message_type is MessageType.UPDATE_PROBE:
                if set(fields) != {"nonce", "job"} or not (job == "" or hex_id(job, 32)):
                    raise ValueError("invalid update probe")
            elif message.message_type is MessageType.UPDATE_CONFIRM:
                if (set(fields) != {"nonce", "job", "instance", "revision"} or not hex_id(job, 32)
                        or self.receiver is None or fields["instance"] != self.instance
                        or fields["revision"] != self.revision):
                    raise ValueError("health confirmation is for a different GUI or release")
                self.receiver.confirm(job, snapshot.fingerprint, self.instance, self.revision)
            elif message.message_type is MessageType.UPDATE_APPLY:
                if (set(fields) != {"nonce", "job", "instance", "transfer_id", "release"}
                        or not hex_id(job, 32) or not hex_id(fields["transfer_id"], 32)):
                    raise ValueError("invalid release apply request")
                if fields["instance"] != self.instance:
                    raise ValueError("the target GUI instance has changed")
                if not self.receiver:
                    raise ValueError("managed updates are not enabled locally")
                if not self.idle_for_update():
                    raise ValueError("busy: finish the other file, shell or input session first")
                release = Release.parse(fields["release"])
                path = self.core.files.completed_path(fields["transfer_id"], release.sha256)
                if path is None:
                    raise ValueError("no matching verified transfer on this connection")
                self.receiver.apply(job, release, path, snapshot.fingerprint, self.instance)
            else:
                return
            report = {"nonce": nonce, "ok": True, "result": self._report()}
        except (ValueError, OSError, RuntimeError) as error:
            report = {"nonce": nonce, "ok": False, "error": str(error)[:240]}
        self.core.send(Message(MessageType.UPDATE_REPORT, report), secure=True,
                       priority=Priority.INTERACTIVE, expected_connection_id=snapshot.connection_id)
