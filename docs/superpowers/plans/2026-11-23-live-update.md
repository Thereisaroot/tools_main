# Single-port live update

1. **Receive while running.** The controller CLI delegates to the existing GUI
   over loopback. Negotiate `live_update_v1` on the trusted serial link. Use a
   fresh challenge to read the peer's process instance, running build and update
   policy. Transfer the ZIP through the existing verified file service.
2. **Prepare a specific release.** The managed receiver accepts an apply request
   referencing a completed transfer in this connection. The locally pinned
   publisher must have signed the product, release sequence, full Git revision
   and ZIP digest. Stage only the bounded ShookLink runtime layout into a fresh
   slot. Ordinary file reception never requests activation.
3. **Hand over the COM port.** A foreground supervisor, started locally on the
   receiver, waits for its GUI child. Once staging finishes and the link is idle,
   the old GUI exits normally with a dedicated switch code. Only after that
   process exits does the supervisor launch the candidate with the same Python,
   serial settings and identity. The supervisor never opens a serial port.
4. **Verify or restore.** The controller keeps its serial port open, re-pairs
   with the previously trusted identity, and probes for the transaction, new
   process instance and exact running revision. The new GUI acknowledges from
   its event loop. A journal retains the last healthy slot until confirmation.
   A failed launch, failed reconnect or interrupted trial restores that slot.
5. **Deliver.** Keep passive ZIP updates for legacy installations. Test signed
   packaging, refusal and replay, busy sessions, bounded waits, restart on a
   single link, committed activation and rollback. Build the bootstrap ZIP from
   the tested commit. Initial receiver enrollment is a local operation.
