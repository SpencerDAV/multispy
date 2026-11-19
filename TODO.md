# TODO

Open work, newest first. Each item says what it is for and, where it has been
measured, what it is *not* worth — a proposal with a measured ceiling on it is
worth more than one without.

---

## Done: surfaces drawn on the photographs (Nov 2026)

A reference plane could only be typed: a name, an axis, an offset. That is the
right way to enter a datum off a print and no way at all to capture a fence, a
floor or the face of a tooling plate, which is most of what is actually in a
cell. Those have no stated direction and no stated distance — they are simply
*there*, in the photographs.

So a plane may now name four `corners` instead. They are ordinary tie points,
which is the whole design and the reason this added so little: marked with the
tool that already exists, triangulated by the solve that already runs. Click
four corners in order round the surface, then the same four in a second photo;
`solve.fit_drawn_planes` takes the least-squares plane through where they land
and projects the corners onto it, so the exported `FACE_SURFACE` is exactly
planar and bounded by the real corners rather than squared off around their
middle.

Two things worth keeping:

- **The corners are not evidence.** They are excluded from the bundle and
  triangulated against the finished cameras afterwards, exactly as a muted
  body's anchors are. Measured while building this: letting four
  roughly-clicked corners into the solve took a 7-photo project from
  **1.41 px to 82 px**. With the exclusion the RMS is unchanged to 1e-6, and
  the surface still fits. `ReferencePlane` always said a plane is what the
  measurement is reported *against*; this keeps that true for a kind of plane
  that is measured.
- **`flatness` is the plane's error bar.** Four marked corners are never
  coplanar, and projecting them makes the face exact — so the gap is reported
  instead of being quietly absorbed into the geometry. A carelessly marked
  fence says so in millimetres rather than arriving as a plausible plane.

Not done: no way to say "this corner is also that corner of the next panel", so
two abutting surfaces are marked independently and need not meet. Fine for a
fence and wrong for a tooling plate with pockets in it.

Rejected: fitting the plane from four corners in **one** photograph, via the
quad's two vanishing points. It is four clicks instead of eight and very much
in this tool's spirit, but it is only correct if the quad is a parallelogram in
space, and nothing about a fence promises that. Worth revisiting as an
*option*, never as the only way.

---

## Done: the panels stopped burying their own numbers (Nov 2026)

Twelve panels came to about 9,000 characters, 78% of it prose. The useful half
of that prose is the first sentence — nearly every message here is one
measurement followed by what it means — so the tail now folds, per message or
all at once, and nothing was rewritten to achieve it: the split is made at
render time out of the prose already there, by `explanationStart`, over the
live DOM with a `Range` so the dozen messages carrying markup survive it.

Measured on the 7-photo project: panel prose 3,287 → 1,419 characters, the hint
bar over the photograph 337 → 99, and **fold all** takes the whole panel to 202
— just the headings.

The thing worth recording is what it does *not* touch. On `rearmod3` the same
clamp only folds 29% of the prose, because that panel's length is not
explanation at all: it is twenty-eight bodies each stating four perfectly short
facts. Repetition and prose are different problems. **Fold all** is the answer
to the first one and the sentence clamp is the answer to the second; neither
substitutes for the other, which is why both are in that bar.

---

## Done: the browser build no longer freezes while it works (Nov 2026)

Reported as "the browser is frozen trying to solve without completing", on the
Pages build. It was not the solver: the project in question solves in 2.2s
natively and 3.6s in the tab, to the same RMS. It was that Pyodide ran *in the
page*, so every call into Python was a synchronous call on the main thread —
nothing could repaint for its duration, `setStatus('solving…')` never reached
the screen, and the window sat there looking hung. Measured on `rearmod3`:

| | before | after |
|---|---|---|
| solve | 8911 ms | 8881 ms |
| longest frame gap | **8977 ms** | **148 ms** |
| frames painted during the solve | **0** | **263** |
| reprojection RMS | 9.37 px | 9.37 px |

Same numbers, live window. Python now runs in `web/worker.js`; `backend.js`
keeps its public API and talks to it over postMessage, counts what is in flight,
and the status bar shows what is running and for how long.

Worth saying in that order: this made nothing faster. It made a slow thing
*legible*, which is the whole complaint — the user could not tell "working"
from "hung", and closed the tab, which on a page whose only persistence is a
download costs every mark since the last Save.

The elapsed counter also named a second offender nobody had measured: reading a
43 MB STL is **20+ seconds** in the browser, and it used to be 20 seconds of
dead window with no label on it. Still 20 seconds; now it says so.

Two guards in `tests/test_build.py`, since none of this is visible from a
terminal: `backend.js` may not touch `pyodide.*` or `loadPyodide`, and the
Python bridge may not be written in both files. CI also checks the bundle ships
a *versioned* `worker.js` — it is the one script `index.html` does not name, so
the content-hash rewrite cannot reach it, and a cached old worker beside a fresh
`backend.js` is a page that loads and then does nothing when Solve is pressed.

Not done: cancelling. Pyodide cannot be interrupted without a SharedArrayBuffer,
which needs COOP/COEP headers GitHub Pages does not send, so the only cancel
available is terminating the worker — which throws away the photos it holds.
Worth doing properly if anyone asks for it; the elapsed counter was the part
that mattered.

---

## Done: anchors can sit out the solve (Nov 2026)

`ObjectInstance.anchors_off` per body, `Settings.anchors_in_solve` for the lot.
A muted body keeps every click, stays in the scene, and is still placed — by
`solve.place_muted_instances`, which triangulates its anchors against the
*finished* cameras and rigid-fits the model, so it cannot move them. The CAD
guard's "worsens" verdict now leads somewhere that survives the next Solve;
`restore_without_bodies` was only ever a snapshot.

Measured on RearMod3, muting every body:

| | voting | muted |
|---|---|---|
| reprojection RMS | 9.37 px | **2.99 px** |
| culled marks | 15 | **2** |
| bodies placed | 26 of 28 | 26 of 28 |
| shape error, median¹ | 3.49 mm | 3.74 mm |

¹ one yardstick for both arms: every body's anchors triangulated against *that
arm's* final cameras, model rigid-fitted. Muting was better on 9 bodies of 23.

So it buys back the camera solution and **not** accuracy — the same ceiling the
table below reports, from the other side. Worth saying in that order whenever
this feature is described, because "9.37 → 2.99 px" on its own reads as an
accuracy claim and is not one.

Open question it raises, not scheduled: a muted body's `shape_rms` is measured
post-bundle while a voting body's is measured in `fit_instances`, pre-bundle —
see item 4 below, which this makes more worth doing, since the two columns now
mean different things in the same table.

---

## From the RearMod3 investigation (Nov 2026)

Dataset: `projects/rearmod3/` — 8 photos, 33 tie points, 28 bodies across 7 CAD
templates, 582 anchor marks, 3 nominal'd points. Launch it with the `rearmod3`
entry in `.claude/launch.json` (port 8806).

### Measured ceiling on all of this — read first

Six very different weighting policies were run against a held-out yardstick (the
17 bodies the solve does *not* flag as suspect, so a policy is not scored on the
data it reacted to). Their shape error against CAD:

| policy | culled | RMS | good bodies, shape |
|---|---|---|---|
| as shipped now | 15 | 9.37 px | **2.71 mm** |
| reweight off | 115 | 4.22 px | 2.79 mm |
| reweight off, no cull ceiling | 115 | 4.22 px | 2.79 mm |
| reweight on, no cull ceiling | 13 | 11.28 px | 2.71 mm |
| tighter cull (3 sigma) | 60 | 7.35 px | 2.75 mm |
| reweight off + 3 sigma | 151 | 3.82 px | 2.76 mm |

**0.08 mm of spread.** The remaining error on this dataset is in the marks, not
in the processing, and no weighting change reaches it. Nothing below will move
that number on RearMod3. They are all for finding out *sooner*, and for telling
the user what different data would be worth.

Note also what that table does to RMS as a yardstick: it swings 3.82 → 11.28 px
while the geometry moves 0.08 mm. RMS mostly measures how much data was
down-weighted or deleted. Prefer the held-out shape figure when judging a change.

### 1. Per-body pose uncertainty, decomposed

`uncertainty.py` already Monte Carlos the part origin against click noise, with
the right lever-arm reasoning in its docstring. Point the same machinery at every
placed body and report three numbers instead of one:

- **centroid sigma** — how well the average of its anchors is known
- **orientation sigma**
- **origin sigma** — the pose actually reported, which is the other two combined
  through the lever arm

Measured on RearMod3 (2 px click noise, 300 trials): centroid 0.64 mm median,
origin 2.45 mm median. For bodies whose model origin lies inside their anchor
cloud those two agree (0.62 → 0.67 mm); for the sensor templates, whose origin is
463–685 mm outside it, they do not (0.64 → 3.32 mm).

Replaces the two columns the UI shows today, which are worse than they look —
see item 4.

### 2. A design-time lever warning

`origin_sigma(anchors)` is closed form and needs no photographs: build the
rigid-fit Jacobian from the anchor coordinates, invert, read the translation
block. At template-definition time this can say

> 917-4007's origin is 463 mm outside its anchor cloud, whose thinnest axis is
> 70 mm. Expect roughly 3x the per-anchor noise at the reported pose.

Costs nothing and is knowable before anyone drives to the cell.

### 3. "Where should the next anchor go"

Same covariance, scanned over candidate positions. On RearMod3 it already gives
a clear answer for 917-4007 (predicted origin sigma 5.54 mm as marked):

| change | predicted | better by |
|---|---|---|
| one more anchor on the existing cloud | 5.52 mm | **0%** |
| one anchor at the best free position | 3.31 mm | 40% |
| one anchor at the model origin | 2.30 mm | 58% |
| seen in twice as many photos | 3.92 mm | 29% |

This is the feature that answers "we have lots of data, why is the fit not
better": it says when more of the same is worth nothing, and what is worth
something instead.

### 4. Recompute `shape_rms` and `implied_scale` after the bundle

Both are set in `fit_instances`, which runs *before* `bundle_adjust`, and nothing
recomputes them — so the per-object "shape" and "size" columns describe the
linear initialisation, not the answer. They came out byte-identical across seven
solver variants, which is how the bug shows itself.

Post-bundle they are materially different: shape median 4.29 → 3.48 mm, scale sd
5.26% → 3.40%, and individual bodies move a long way (obj0 3.44 → 0.35 mm,
obj8 27.5 → 4.8 mm, obj18 11.05 → 1.27 mm).

Smallest item here and the most misleading one while it stands.

### 5. A weight-share line in the report

One line: `tie points 84%, anchors 3%, lines 13%`. On the original code 582
anchor marks — 84% of every click in the project — carried 3.3% of the
photogrammetric weight, and nothing said so. A single line would have made that
visible immediately instead of after a day of digging.

### 6. Guard the invariant that failed

`_transform_scene` carried a comment asserting that non-manual poses are
"recomputed from `positions` after this runs". True at the two call sites inside
`fit_instances`, false at the one in `align_to_nominals`. Nothing checked it, and
the anchors started the bundle a median of 5.4 million pixels out.

Add the assertion the comment implies: after `align_to_nominals`, no anchor
should reproject further than some generous bound. Cheap, and it is the check
whose absence cost the most here.

### 7. Possibly: per-axis anchor sigma in the template frame

Measured anchor scatter on RearMod3 is anisotropic — 14.0 / 8.7 / 6.0 mm in X/Y/Z
of the template frame, a factor of 2.3 between axes — so one isotropic number is
not what the data says.

**Untested.** Uniform sigma changes were measured and do *not* help (shape 3.51 →
3.72 mm going from 2 px to 8 px), but a per-axis weighting is a different thing
and has not been tried. Treat as a question, not a plan.

---

## Known, not scheduled

- The three nominal'd points on RearMod3 form a sliver: 2122 mm long, 198 mm tall,
  against a 4148 mm scene, so 1 mm of error at p18 is ~21 mm across the scene.
  Each nominal is a 0.1 mm axis lock exempt from the robust kernel. A datum that
  strong deserves more than three points and more than two rays each.
- `p1` appears to be marked on different physical features in different photos
  (cone-pin tip in img1/img4, a post edge in img7). Worth re-clicking before
  trusting the frame.
