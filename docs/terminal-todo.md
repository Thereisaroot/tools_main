# ShookLink terminal worklist

Keep the trusted serial PTY/ConPTY and the peer's own shell completion.
Use [xterm.js](https://xtermjs.org/) for the interactive screen; its official
fit and search addons work without a network service. [QTermWidget](https://github.com/lxqt/qtermwidget)
does not list Windows support. Follow the [xterm security](https://xtermjs.org/docs/guides/security/)
and [flow-control](https://xtermjs.org/docs/guides/flowcontrol/) guidance.

- [x] Forward Tab and Shift+Tab without moving Qt focus in live mode; leave local editing optional.
- [x] Allowlist the exact offline resources in release packaging and signed staging; keep legacy archives valid.
- [x] Vendor pinned upstream releases and licenses; record a reproducible refresh procedure.
- [x] Embed an offline xterm with bounded output, VT alternate screen, scrollback, selection and the engine's IME input.
- [x] Add find, next/previous match, fit-on-resize and deliberate copy/paste. No external URLs or remote clipboard commands.
- [x] Keep a working native fallback if the browser engine or resources are unavailable.
- [ ] Test actual key delivery, shell completion, alternate-screen restoration, UTF-8 and packaging on Mac and Windows.
- [ ] Verify live deployment on an idle peer without taking over the user's WSL session.

## Maintainer notes

The approved upstream packages are `@xterm/xterm@6.0.0`, `@xterm/addon-fit@0.11.0`,
and `@xterm/addon-search@0.16.0` (MIT). In `tools/terminal-vendor`, run
`npm ci --ignore-scripts` then `node sync.mjs` to reproduce the checked-in
assets and notices from the integrity-locked archives. No npm install happens
when a user launches ShookLink.

Receivers predating `71e23e40d4a6` intentionally reject JS/CSS in a live
update. First install that Python-only compatibility build, verify its running
revision, then install the asset-bearing release with the next sequence.
Alternatively, extract the full trusted bootstrap ZIP locally. Do not wrap
scripts in Python or relax the archive path allowlist.
