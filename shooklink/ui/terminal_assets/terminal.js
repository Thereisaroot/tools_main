'use strict';
new QWebChannel(qt.webChannelTransport, channel => {
  const bridge = channel.objects.terminalIO;
  const term = new Terminal({
    scrollback: 10000, cursorBlink: true, cursorStyle: 'block',
    fontFamily: 'Menlo, Consolas, monospace', fontSize: 15, rightClickSelectsWord: true,
    allowProposedApi: false, allowTransparency: false,
    linkHandler: {activate() {}},
    theme: {background: '#0d1715', foreground: '#d8e7df', cursor: '#b9d8ce',
      cursorAccent: '#15312c', selectionBackground: '#345e55'}
  });
  const fit = new FitAddon.FitAddon();
  const search = new SearchAddon.SearchAddon();
  term.loadAddon(fit);
  term.loadAddon(search);
  const host = document.getElementById('terminal');
  term.open(host);
  const encoder = new TextEncoder();
  function transmit(bytes) {
    for (let i = 0; i < bytes.length; i += 16384) {
      bridge.keys(btoa(String.fromCharCode(...bytes.subarray(i, i + 16384))));
    }
  }
  term.onData(text => transmit(encoder.encode(text)));
  term.onBinary(text => transmit(Uint8Array.from(text, c => c.charCodeAt(0) & 255)));
  term.onResize(({cols, rows}) => bridge.dimensions(cols, rows));
  function refit() {
    const d = fit.proposeDimensions();
    if (d && d.cols >= 2 && d.rows >= 2) term.resize(Math.min(500, d.cols), Math.min(500, d.rows));
  }
  let requested = false;
  new ResizeObserver(() => {
    if (!requested) requestAnimationFrame(() => { requested = false; refit(); });
    requested = true;
  }).observe(host);
  bridge.output.connect(encoded => {
    const bytes = Uint8Array.from(atob(encoded), c => c.charCodeAt(0));
    term.write(bytes, () => bridge.drained());
  });
  bridge.pasteRequested.connect(text => term.paste(text));
  bridge.fontRequested.connect(size => { term.options.fontSize = size; refit(); });
  function find(query, previous) {
    if (!query) { search.clearDecorations(); term.clearSelection(); return false; }
    return previous ? search.findPrevious(query) : search.findNext(query);
  }
  bridge.findRequested.connect((query, previous) => bridge.found(find(query, previous)));
  window.shookTerm = Object.freeze({
    focus: () => term.focus(), selection: () => term.getSelection(), find,
    screenText() {
      const buf = term.buffer.active;
      return Array.from({length: term.rows}, (_, i) =>
        buf.getLine(buf.viewportY + i)?.translateToString(true) || '').join('\n');
    }
  });
  refit();
  bridge.dimensions(term.cols, term.rows);
  bridge.started();
});
