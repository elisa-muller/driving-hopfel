# HOPJam Quick Play

> Phantom Jam - HOPJam Traffic demo (60s simulation)

A quick guide to run and play the demo in this repo.

What
- Try to keep traffic smooth and avoid phantom jams. Toggle HOPJam (ghost car) to see how one smart agent stabilises flow.

How to play
- Open [hopjam-game.html](hopjam-game.html#L1) in a browser (double-click or serve the folder).
- On the start screen choose: Solo Run (with/without HOPJam) or Comparison mode.
- Adjust `Duration` and `Cars / Lane` sliders to change simulation length and congestion.

Controls
- Accelerate: Arrow Up
- Keep Speed: Space
- Decelerate: Arrow Down
- Induce Jam (seed a phantom jam): `J` button
- Gamepad: D-pad / left stick up-down and common action buttons map to the above

Quick tips
- Use Comparison mode to view your lane (human) vs the HOPJam-controlled lane side-by-side.
- Increase `Cars / Lane` to observe phantom jams forming; add HOPJam cars to see stabilisation.

Run locally (optional)
If you prefer a simple local server, run:

```bash
python -m http.server 8000
# then open http://localhost:8000/hopjam-game.html
```

Credits
- HOPJam demo: Technology for Social Innovation course (2026)

Enjoy, and try seeding a jam with `J` to stress-test the system!
