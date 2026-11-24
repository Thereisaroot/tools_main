// Run npm ci --ignore-scripts first; npm verifies the lockfile integrity.
import {copyFileSync, mkdirSync} from 'node:fs';
import {fileURLToPath} from 'node:url';

const modules = new URL('./node_modules/@xterm/', import.meta.url);
const assets = new URL('../../shooklink/ui/terminal_assets/', import.meta.url);
mkdirSync(assets, {recursive: true});
const files = {
  'xterm/lib/xterm.js': 'xterm.js', 'xterm/css/xterm.css': 'xterm.css',
  'addon-fit/lib/addon-fit.js': 'addon-fit.js', 'addon-search/lib/addon-search.js': 'addon-search.js',
  'xterm/LICENSE': 'LICENSE-xterm.txt', 'addon-fit/LICENSE': 'LICENSE-fit.txt',
  'addon-search/LICENSE': 'LICENSE-search.txt'
};
for (const [from, to] of Object.entries(files)) {
  copyFileSync(fileURLToPath(new URL(from, modules)), fileURLToPath(new URL(to, assets)));
}
