"""Controller-side update transaction over the GUI's existing connection."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

from shooklink.protocol.messages import MessageType
from .manager import PENDING
from .release import MAX_ZIP, Release, hex_id, read_bounded
from .wire import Refused, Unavailable


def deploy(core, send_file, path: Path, release: Release, job: str, timeout: int) -> dict:
    if not hex_id(job, 32) or type(timeout) is not int or not 1 <= timeout <= 3600:
        raise ValueError("invalid live update job or timeout (1 to 3600 seconds)")
    if hashlib.sha256(read_bounded(path, MAX_ZIP)).hexdigest() != release.sha256:
        raise ValueError("local ZIP differs from the signed release")
    deadline = time.monotonic() + timeout
    epoch = core.files.connection_epoch
    query = {"job": job}
    wire = core.updates
    before = wire.exchange(MessageType.UPDATE_PROBE, query, timeout=min(3, timeout))
    if not before.get("managed"):
        raise Refused("on Windows, open the updated .bat and click Allow Peer Updates for this peer")
    release.verify(before["publisher"])
    if before["phase"] in PENDING:
        raise Refused(f"receiver is busy with update {before['job']}")
    if before["revision"] == release.revision:
        return dict(before, outcome="already_running")
    if not before["idle"]:
        raise Refused("receiver is busy with another file, shell or input session")
    if release.sequence <= before["highest"]:
        raise Refused("release sequence was already attempted or superseded")
    peer = before["fingerprint"]
    old = before["instance"]
    transfer = send_file({"path": str(path)}, deadline=deadline, expected_epoch=epoch)
    apply = dict(job=job, release=release.as_dict(), instance=old,
                 transfer_id=transfer["transfer_id"])
    kind = MessageType.UPDATE_APPLY
    fields = apply
    last_status = "archive delivered"
    while (remaining := deadline - time.monotonic()) > 0:
        try:
            outgoing_kind, outgoing_fields = kind, fields
            # After a lost apply/confirm reply, probe before retrying either operation.
            kind, fields = MessageType.UPDATE_PROBE, query
            report = wire.exchange(outgoing_kind, outgoing_fields, peer=peer, peer_id=before["peer_id"],
                                   timeout=min(1.5, remaining))
            last_status = report["phase"]
            if report.get("job") == job:
                if report.get("target") != release.revision:
                    raise Refused("transaction belongs to another release")
                if last_status in {"failed", "rolled_back"}:
                    raise Refused(f"{last_status}: {report.get('reason', '')}")
                if report["instance"] != old and report["revision"] == release.revision:
                    if last_status == "installed":
                        return dict(report, outcome="verified")
                    if last_status == "trial":
                        kind = MessageType.UPDATE_CONFIRM
                        fields = dict(query, revision=release.revision, instance=report["instance"])
            elif report.get("instance") == old and last_status not in PENDING:
                kind, fields = MessageType.UPDATE_APPLY, apply
            else:
                raise Refused("peer changed transaction or GUI instance during activation")
        except Unavailable as error:
            last_status = str(error)
        time.sleep(min(0.35, max(0, deadline - time.monotonic())))
    raise TimeoutError(
        f"update {job} not confirmed ({last_status}). Check peer-update-status --job {job}"
    )
