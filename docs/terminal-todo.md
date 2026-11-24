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
- [x] Test actual key delivery, Bash completion in a PTY, alternate-screen restoration, UTF-8 and packaging.
  [The macOS/Windows test matrix](https://github.com/Thereisaroot/tools_main/actions/runs/45375170683)
  passed on Python 3.11 and 3.13 for runtime build `d43d2692283d`.
- [ ] Verify live deployment on an idle peer without taking over the user's WSL session.

## 사용 및 적용

- 파일명이나 명령의 앞부분을 입력하고 Tab을 누르면 원격 셸이 완성 후보를 선택한다. 실시간 입력이 기본이다.
- 출력 검색은 **Find** 또는 Ctrl+Shift+F로 시작한다. macOS에서는 Cmd+F도 지원한다.
- 여러 줄을 붙여넣을 때 원격 프로그램이 bracketed paste 모드를 요청하면 터미널은 붙여넣은 범위를 구분하여 전송한다.
- 창 하단의 `xterm.js` 표시는 새 엔진을, `Native fallback` 표시는 대체 화면을 뜻한다. 설치 후 실행 버전은 상대가 보고한 커밋으로 확인한다.
- 셸 사용 중에는 설치를 요청하지 않는다. 기존 WSL(Windows Subsystem for Linux) 세션을 종료한 뒤 업데이트와 재연결을 검증한다.

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
