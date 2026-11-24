# ShookLink terminal worklist

Keep the trusted serial PTY/ConPTY and the peer's own shell completion.
Use [xterm.js](https://xtermjs.org/) for the interactive screen; its official
fit and search addons work without a network service. [QTermWidget](https://github.com/lxqt/qtermwidget)
does not list Windows support. Follow the [xterm security](https://xtermjs.org/docs/guides/security/)
and [flow-control](https://xtermjs.org/docs/guides/flowcontrol/) guidance.

- [x] Forward Tab and Shift+Tab without moving Qt focus in live mode; leave local editing optional.
- [x] Allowlist the exact offline resources in release packaging and signed staging; keep legacy archives valid.
- [ ] Vendor pinned upstream releases and licenses; record a reproducible refresh procedure.
- [ ] Embed an offline xterm with bounded output, proper VT alternate screen, scrolling, selection and IME.
- [ ] Add find, next/previous match, fit-on-resize and deliberate copy/paste. No external URLs or remote clipboard commands.
- [ ] Keep a working native fallback if the browser engine or resources are unavailable.
- [ ] Test actual key delivery, shell completion, alternate-screen restoration, UTF-8 and packaging on Mac and Windows.
- [ ] Verify live deployment on an idle peer without taking over the user's WSL session.
