// Renders index.html to a 1920x1080 30 fps MP4 using headless Chrome + ffmpeg.
//
//   node render.mjs                 -> card-payment-silent.mp4
//   node render.mjs --stills 2.3 7  -> still-2.3.jpg, still-7.jpg (for checking frames)
//
// Then add the sound:  python3 sfx.py && ffmpeg ... (see README.md)
//
// Needs: playwright (npm i playwright), Chrome or Chromium, ffmpeg on PATH.
// Set CHROME=/path/to/chrome to choose the browser binary.
import { createRequire } from 'node:module';
import { spawn } from 'node:child_process';
import { writeFileSync, renameSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import path from 'node:path';

const require = createRequire(import.meta.url);
let playwright;
try { playwright = require('playwright'); }
catch { playwright = require(process.env.PLAYWRIGHT_PATH || '/opt/npm-tools/node_modules/playwright'); }

const here = path.dirname(fileURLToPath(import.meta.url));
const FPS = 30;
const args = process.argv.slice(2);
const stills = args[0] === '--stills' ? args.slice(1).map(Number) : null;
const out = process.env.OUT || path.join(here, 'card-payment-silent.mp4');

const browser = await playwright.chromium.launch({
  executablePath: process.env.CHROME || undefined,
  args: ['--force-device-scale-factor=1', '--font-render-hinting=none'],
});
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
await page.goto(pathToFileURL(path.join(here, 'index.html')).href + '?render');
await page.evaluate(() => window.ready);

const grab = t => page.evaluate(t => {
  window.renderAt(t);
  return document.getElementById('c').toDataURL('image/jpeg', 0.95).split(',')[1];
}, t);

if (stills) {
  for (const t of stills) writeFileSync(path.join(here, `still-${t}.jpg`), Buffer.from(await grab(t), 'base64'));
} else {
  const dur = await page.evaluate(() => window.DUR);
  const frames = Math.round(dur * FPS);
  const tmp = out.replace(/\.mp4$/, '.part.mp4');
  const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(FPS), '-c:v', 'mjpeg', '-i', '-',
    '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '16', '-pix_fmt', 'yuv420p', '-r', String(FPS), tmp], { stdio: ['pipe', 'inherit', 'inherit'] });
  const done = new Promise((res, rej) => ff.on('close', c => c === 0 ? res() : rej(new Error('ffmpeg exit ' + c))));
  for (let i = 0; i < frames; i++) {
    const buf = Buffer.from(await grab(i / FPS), 'base64');
    if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once('drain', r));
    if (i % 60 === 0) process.stdout.write(`frame ${i}/${frames}\n`);
  }
  ff.stdin.end();
  await done;
  renameSync(tmp, out);
  console.log('wrote', out);
}
await browser.close();
