# Lumen app promo

A 20 second Keynote-style product promo for Lumen, a made-up banking app. A phone floats in 3D over soft drifting gradients while the screens slide inside it: the balance counts up, a debit card flips over, spending bars grow, and a transfer confirms with a check. It ends on the brand screen. Music is synthesized, with a soft whoosh on every screen change.

- `app-promo.mp4`: the finished video (1920×1080, 30 fps, H.264 + AAC).
- `index.html`: the animation itself, plain HTML and CSS 3D. Open it in a browser to watch a silent live loop.
- `render.py`: renders `index.html` frame by frame in headless Chrome and pipes the frames to ffmpeg.
- `music.py`: synthesizes the score (pad, plucked arpeggio, bass, soft kick, sound effects, no sample files) to `music.wav`.

## Rebuild the video

Needs Python 3 with `numpy`, `scipy` and `playwright`, Chrome or Chromium (set `CHROME=/path/to/chrome` if it is not at `/usr/bin/google-chrome`), and ffmpeg.

```sh
python3 render.py 0 300 a.mp4      # first half  (frames 0-299)
python3 render.py 300 600 b.mp4    # second half (frames 300-599), can run in parallel
printf "file 'a.mp4'\nfile 'b.mp4'\n" > list.txt
ffmpeg -f concat -safe 0 -i list.txt -c copy video.mp4
python3 music.py                   # -> music.wav
ffmpeg -i video.mp4 -i music.wav -map 0:v -map 1:a -c:v copy \
       -c:a aac -b:a 192k -shortest -movflags +faststart app-promo.mp4
```

## Changing it

Everything is a pure function of time (`render(t)` in `index.html`):

- `SLIDES` holds the times the phone moves to the next screen (every 3.75 s, from 6.25 s). `CAPS` holds the caption text and when each one shows. `CH` holds the floating notification cards.
- The music runs at 96 BPM, so one bar is 2.5 s and the slides land on beat boundaries. If you move a slide, move the matching `whoosh(...)` call in `music.py`.
- The brand name, copy and palette are all plain text in `index.html`.
