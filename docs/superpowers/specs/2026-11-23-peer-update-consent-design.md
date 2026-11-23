# Local consent for signed peer updates

An ordinary Windows launch starts the single-port supervisor, not an update
authorization. The existing trust button gives way to **Allow Peer Updates**
once the link is ready; it does not consume an extra row in the window.

Each installation advertises its separate Ed25519 release public key as one
parameterized capability in the authenticated TRUST frame. Legacy peers can
ignore the capability. Missing, malformed or ambiguous offers are not eligible.
The private key remains in local app data; the default signer used by
`sign-update` is the one advertised by that installation.

Only the local window submits an enrollment. Recheck the connection ID, peer
installation, fingerprint, current offer and session activity before writing
the pin. A normal launch, file transfer, trust approval or wire message cannot
enroll a publisher. Persist the fingerprint and signer together. Never replace
an existing pin through the button. Legacy offline pins retain their original
semantics and do not silently become desktop-signer pins.

The file verification, signed release sequence, single-port handoff and health
confirmation are unchanged. Integration coverage includes authorizing via the
Qt button, restarting without enrollment flags, installing a matching release
and refusing a different signer or transport peer.
