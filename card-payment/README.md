# Card payment explainer

A 15 second isometric explainer of a contactless card payment, in navy and mint: tap card → terminal sends the request → bank checks it → approved. The camera glides continuously between the steps, and subtle sound effects mark each beat.

- `card-payment.mp4`: the finished video (1920×1080, 30 fps, H.264 + AAC).
- `index.html`: the animation itself, drawn on a canvas. Open it in a browser to watch a silent live version; the Replay button restarts it.
- `render.mjs`: renders `index.html` frame by frame in headless Chrome and pipes the frames to ffmpeg.
- `sfx.py`: synthesizes the sound effects (sine tones and filtered noise, no sample files) to `sfx.wav`.

## Rebuild the video

Needs Node with `playwright`, Chrome or Chromium, Python 3 with `numpy` and `scipy`, and ffmpeg.

```sh
node render.mjs                     # -> card-payment-silent.mp4 (set CHROME=/path/to/chrome if needed)
python3 sfx.py                      # -> sfx.wav
ffmpeg -i card-payment-silent.mp4 -i sfx.wav -map 0:v -map 1:a -c:v copy \
       -c:a aac -b:a 192k -shortest -movflags +faststart card-payment.mp4
```

`node render.mjs --stills 2.3 7.9` writes single frames at those times, which is handy for checking a change before a full render.

## Changing it

Everything is a pure function of time (`renderAt(t)` in `index.html`):

- `T` holds the event times (tap, request out, the three bank checks, response back, approval). The sound effects in `sfx.py` use the same times.
- `CAM_KEYS` lists the camera poses (time, focus point, zoom). The camera follows a monotone cubic spline through them, so it never cuts or overshoots.
- `COL` holds the palette, and `STEPS` holds the caption text.
