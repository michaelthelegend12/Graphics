# Graphics

Motion graphics experiments, built as plain HTML, CSS and JavaScript so they run in any browser with no build step.

## Pieces

- [`logo-intro/`](logo-intro/index.html): animated logo and title intro. Open `logo-intro/index.html` in a browser and use the Replay button to run it again.
- [`mascot/`](mascot/index.html): Pip, an original mascot. It wakes up, waves, hops twice and celebrates with confetti, then dozes off in an 8 second loop.
- [`card-payment/`](card-payment/): 15 second isometric explainer of how a card payment works (tap, terminal, bank check, approved), with smooth camera moves and sound. Watch [`card-payment.mp4`](card-payment/card-payment.mp4), or open `card-payment/index.html` for the live version.
- [`app-promo/`](app-promo/): 20 second Keynote-style promo for Lumen, a made-up banking app. A phone floats in 3D while the screens slide inside it, the balance counts up and a card flips over, with music. Watch [`app-promo.mp4`](app-promo/app-promo.mp4), or open `app-promo/index.html` for a silent live loop.
- [`smartphone-data/`](smartphone-data/): 25 second animated data story of worldwide smartphone users from 2010 to 2025. A neon line graph draws itself while the camera follows its tip, with milestone callouts at each billion, on a dark theme. Watch [`smartphone-data.mp4`](smartphone-data/smartphone-data.mp4) (1080p), or run `smartphone-data/render.py` (Python, matplotlib and ffmpeg) to re-render it.
- [`bees-collage/`](bees-collage/): 20 second paper cutout / collage explainer on why bees matter for our food. Textured paper, layered shadows and a 12 fps stop-motion feel, with gentle music. Watch [`bees-collage.mp4`](bees-collage/bees-collage.mp4) (720p).
- [`flight-globe/`](flight-globe/): 15 second 1080p travel-vlog flight route, London to Dubai to Tokyo. A dotted 3D globe rotates while two glowing arcs draw themselves, a plane follows each arc, and city labels pop in, with a progress bar and distance counter, over an original chill lo-fi track. Watch [`flight-globe.mp4`](flight-globe/flight-globe.mp4), or run `flight-globe/render.py` (Python, numpy, OpenCV, Pillow and ffmpeg) to re-render the picture and `flight-globe/music.py` (numpy and scipy) to regenerate the music.
- [`vaccine-whiteboard/`](vaccine-whiteboard/): 30 second hand-drawn whiteboard explainer of how vaccines train your immune system. Lines and handwritten labels draw on stroke by stroke with a marker, simple cell characters act out the story, and an eraser wipes between scenes. Watch [`vaccine-whiteboard.mp4`](vaccine-whiteboard/vaccine-whiteboard.mp4) (720p, silent), or run `vaccine-whiteboard/render.py` (Python, numpy, OpenCV and ffmpeg) to re-render it.
