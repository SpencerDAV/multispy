# MultiSpy

fSpy-style vanishing point calibration across a **whole photo set**, with
hand-marked corresponding features tying the images together, resolved in one
joint bundle adjustment.

Built for the case where there is real overlap between photos but not enough for
COLMAP or GLOMAP: wide baselines, few images, unknown camera, scenes with
man-made structure.

Objects whose shape you already know can be marked against their CAD model
instead of as free points — which makes the reconstruction metric without a tape
measure, and turns the layout of a camera array into something you can solve by
walking around it with a camera.

**New here?** [TUTORIAL.md](TUTORIAL.md) walks through both jobs start to finish
on scenes with known ground truth, so you can check your own work. This file is
the reference.

---

## Why this works where SfM does not

Two properties do the heavy lifting.

**Vanishing points give you global rotations for free.** Three orthogonal
vanishing points in one image hand you that camera's orientation in world axes
without reference to any other image. No match graph, no chaining, no drift
accumulation. That is normally the hardest part of structure-from-motion, and
here it falls out of control lines you draw by hand.

**With rotations known, the rest is linear.** Solving for camera translations
and 3D point positions with the rotations fixed is a linear least-squares
problem — no initial pair to choose, no incremental mapper to wander off, no
local minima. It needs roughly **two shared tie points per image pair**, against
the dozens of well-distributed matches an SfM pipeline wants.

And the thing that makes it beat fSpy on its own ground: **a single image's
focal length is determined by about four clicks with zero redundancy**, which is
why fSpy feels so twitchy — a distant vanishing point swings hundreds of pixels
for a two-pixel slip on a control line. Share one focal length across N images
and tie them together with corresponding points, and that estimate becomes
massively over-constrained.

Measured on the synthetic scene in `tests/`, with 1.0 px of click noise on tie
points and 1.5 px on control-line endpoints:

| | focal length error |
|---|---|
| Independent single-image solves (what fSpy gives you) | median **2.5%**, worst **81%** |
| Joint solve, shared focal + tie points | **0.05%** |

(Those single-image figures are already the *improved* ones. Each image's focal
estimate is an inverse-variance weighted blend of its axis pairs rather than a
plain median, which cut the median single-image error from 15.5% to 2.5% —
the weights come from the same error propagation that drives the Focal check
panel. The joint solve still beats it by 50x on the median and 1600x on the
worst case, which is the whole argument for marking a set rather than a photo.)

Rotations land within 0.2°, camera positions within 0.3% of the scene diagonal,
and reprojection RMS settles below the injected click noise.

---

## Install

```bat
cd "C:\Animation Studio\MultiSpy"
python -m venv .venv
.venv\Scripts\python.exe -m pip install numpy scipy pillow
```

That is the whole dependency list. The web layer is stdlib only.

## Quick start

```bat
.venv\Scripts\python.exe -m multispy
```

No arguments needed. It reopens your most recent project, or starts an empty one
if there are none, and opens the UI. Then **drag photos straight onto the
canvas**.

---

## Or run it in a browser, with nothing installed

**[spencerdav.github.io/multispy](https://spencerdav.github.io/multispy/)**

The same solver, in a tab. Drop in photos and a CAD model, mark them, press
Solve, download the exports. There is no install and no sign-up.

It is not a demo or a cut-down version. The page loads CPython — numpy, scipy
and Pillow included — through [Pyodide](https://pyodide.org) and runs the very
same `multispy` package in-process, so it produces identical numbers. On the
synthetic rig scene, the browser and the command line agree to the decimal:
focal 3198.8 px, reprojection RMS 1.209 px, body shape error 0.55–0.72 mm,
baselines 225.2 / 450.0 / 675.0 / 900.3 mm. The solve itself takes about a
quarter of a second.

**Nothing is uploaded.** There is no server to upload to — the whole thing runs
in the tab, so photos of parts and tooling never leave the machine. That is the
main reason to hand someone this link rather than a hosted service.

Three things differ, all of them consequences of a browser having no disk:

- **First load is about 20 MB** and takes a few seconds, then it is cached. It
  waits behind a button rather than downloading on arrival.
- **Saving is a download.** The project comes back as a zip with its photos
  inside, because the marks alone would reopen pointing at files that are not
  there. Reopen it by dropping that zip back in.
- **Close the tab and it is gone.** Download the project to keep it.

Memory is the real limit: a 4000 × 3000 photo is about 36 MB decoded, so thirty
is comfortable and two hundred is not. For large sets, run it locally.

Build it yourself with:

```bat
.venv\Scripts\python.exe tools\build_static.py --serve 8800
```

which assembles `dist/` and serves it, so the static path can be exercised
without deploying anything.

---

## Projects and photos

Everything is managed in the app; no paths on the command line.

**Adding photos.** Drag them onto the canvas, or use **+ add photos** in the
strip on the left. They are copied into the project's own `photos/` folder, so
the project folder stays self-contained and portable. **link a folder…**
instead references photos where they already sit on disk, without copying —
better for a 200-frame set you do not want duplicated.

**Removing photos.** Hover a thumbnail and click its **×**, or use **remove
photo** in the This image panel. You are asked twice, on purpose: first to
confirm the removal (it warns you how many marks will be lost), then separately
whether to delete the file from disk. Cancelling the second keeps the file and
only unlinks it from the set. Photos that were *linked* rather than uploaded are
never deleted from disk — they are yours, sitting somewhere else on the machine,
and the app only ever forgets about them.

**Projects.** The name in the toolbar opens a switcher: every project under
`projects/`, most recent first, plus a field to create a new one. Each project
is a folder holding `project.multispy` and its `photos/`. Put them elsewhere
with `--projects "D:\somewhere"`.

Removing a photo can orphan tie points that were only visible in it. They are
not silently discarded — the points panel marks them **needs 2**, and the solve
report names them.

## Try it on the synthetic scene

```bat
.venv\Scripts\python.exe demo\make_demo.py
.venv\Scripts\python.exe -m multispy demo\project.multispy
```

This renders six synthetic photos of a 10 x 6 x 4 box, opens the UI with
everything already marked, and gives you a known-good solve to press **Solve**
on. Add `--blank` to `make_demo.py` for the same photos with no marks, which is
the one to practise the workflow on — the ground truth is known, so you can tell
how well you did (true focal 3200 px).

---

## The workflow

### 1. Control lines (Lines mode, keys 1 / 2 / 3)

Drag along edges in the photo that are **parallel in 3D**. Keys `1` `2` `3`
choose which world axis you are tracing: X, Y, Z.

- **Two per axis minimum, three is better.** Two lines fix a vanishing point
  exactly, which means nothing is left over to detect a mistake. Three gives the
  solver a disagreement to report.
- **Two axes minimum, three is better.** More axes means more focal estimates
  to cross-check.
- **Direction matters.** Draw from the lower coordinate toward the higher one,
  i.e. along the **positive** axis. The arrowhead shows which way the line
  points. This is what resolves the sign ambiguity — the same vanishing point
  serves both `+d` and `−d`, so only the direction you drew can tell them apart.
  Drew one the wrong way? **Left-click it** to reverse it in place.
- **Longer is better and is weighted accordingly.** A 2 px slip on a 100 px
  segment says far less about a direction than the same slip on a 1000 px one,
  and the solver weights by segment length to match.
- Keep **VP rays** on (it is, by default; it and the other overlays live in the
  **show** menu in the toolbar). Each control line is extended to its vanishing point.
  If one ray does not converge with its neighbours, that line is on an edge that
  is not actually parallel to the others — the single most common mistake.

#### Watch the Focal check panel

As soon as you have two lines on each of two axes — four lines, long before a
full markup — the **Focal check** panel tells you whether the focal length is
determinable at all, and to what precision. It needs no solve.

```
FOCAL CHECK                              POOR
1884 px  ±57%
best pair XxZ, limited by Z

axis  lines  length   VP out   lever
X       3    1140px    15306     13x
Z       3     118px    13988    119x

The Z vanishing point is 13,988 px away while the Z control lines
are only 118 px long, a 119x lever — so one pixel of marking error
moves it about 119 px. Longer Z lines, further apart, are what help.
```

**Lever** is the number that matters: vanishing-point distance divided by
control-line length. It is the factor by which your marking error is magnified.
At 2x you can afford to be sloppy; at 119x you cannot mark accurately enough
with any amount of care, and no effort spent on tie points will rescue it.

The verdicts are **good** (±<2%), **weak** (±2–8%) and **poor** (±>8%), from a
first-order propagation of your assumed marking accuracy through the vanishing
points into the focal length. It is calibrated, not decorative — on the
synthetic scene it predicts ±1.0% and the solve lands within 0.05%; on a real
drone set it predicted ±11% and the solve came out 11.4% off.

If it says **poor**, you have three options: draw longer control lines on the
limiting axis, find a third axis with better convergence, or accept that this
viewpoint cannot determine focal length and lock the focal to a value you trust
from elsewhere.

#### Locking the focal length

The **focal length** box in the This image panel sets it for the whole camera
group, three ways:

- **From EXIF** — the row underneath reads the photos and offers a **use**
  button. Confidence is reported honestly, and there are two routes:
  *focal plane resolution* (`FocalLength` in mm plus the sensor-size tags) is
  exact and marked **good**; *35mm equivalent* is marked **approximate**,
  because cameras round it to a whole millimetre. On the demo set that rounding
  alone costs 1.1% — 27.7 mm stored as 28, read back as 3236 px instead of 3200.
- **Typed** — a number plus a unit, either raw **px** or **mm eq** (35mm
  equivalent, converted through the image **diagonal** — see below).
- **From a solve** — solve once with it free, then tick **lock focal length**
  to pin what came out, before refining anything else.

**Use EXIF as a fallback, not a default.** On a photo set where the Focal check
says *good*, the solve beats EXIF by a wide margin — on the demo it lands at
3201.6 px against a true 3200 (0.05%), where EXIF gives 3236 (1.1%), 22x worse.
EXIF earns its place exactly when the Focal check says *poor*.

**Why the diagonal.** 35mm equivalence is a *diagonal* field of view, so it
converts as `f_px = f35 × hypot(W, H) / hypot(36, 24)`. Dividing by the 36 mm
film width instead only agrees when the frame is 3:2 like the film itself: a 4:3
photo comes out 3.8% low and a 16:9 frame 4.8% high. The diagonal is also the
same after a quarter turn, which matters more than it sounds — a portrait photo
arrives with its width and height already swapped, and the width rule reads the
very same file 28% low purely for having been held upright.

Two readings are refused rather than offered:

- **Cropped images.** If the aspect ratio disagrees with the EXIF dimensions,
  the pixels no longer span the whole sensor and the conversion is void.
  Resizing is fine and handled; so is a quarter turn, which is reported as a
  rotation rather than refused as a crop. Cropping is not.
- **A zoomed lens.** If the EXIF focal lengths across a group disagree by more
  than 2%, one shared focal is the wrong model and you are told to split the
  group rather than handed an average.

Check it without the UI:

```bat
.venv\Scripts\python.exe -m multispy exif demo\project.multispy
```

### 2. Tie points (Points mode)

Click the same physical feature in every photo it appears in.

- Press `N` for a new point, then click it in the first photo. With
  **auto-advance** on, you are moved straight to the next photo that does not
  have it yet. `Tab` does the same manually.
- **Two shared points per image pair** is the hard floor. Four to six
  well-spread points per pair is comfortable.
- Spread them in **depth**, not just across the frame. Points that are nearly
  coplanar as seen from the cameras make the linear stage ill-conditioned, and
  the solve panel will say so.
- Every image needs at least 2 tie points or its camera cannot be placed, and
  every point needs at least 2 images or it cannot be triangulated. Anything
  that fails these gets named in the report rather than silently dropped.

### 3. Constraints (optional)

- **Distance** between two points gives the solve real units. Without one,
  scale is arbitrary — everything is correct up to a similarity.
- **Axis lock** pins one world coordinate of a point. Three of them on one
  point sets the origin; the same value across several points states a plane
  ("these are all on the floor").

After a solve each constraint shows **what it got against what you asked for**,
in world units, with the tolerance it was given. Worth reading every time: a
constraint is the one figure you *assert* rather than click, so it is the only
place "did the reconstruction agree with me?" has a definite answer — and a datum
missed by a millimetre costs almost nothing in pixels, so it is invisible in
every other number here. A constraint naming a tie point that did not get solved
is reported as having had no effect, rather than silently having none.

Two things worth knowing before you lock several of them:

- **Locking several points imposes a scale**, whether or not you meant it to. Two
  points pinned 150 mm apart in X is a distance constraint wearing a different
  hat. Locking them to the *same* value states a plane instead, and says nothing
  about scale.
- **If a known object is also setting the scale**, the two are independent
  measurements of the same thing, and the solve reports how far apart they are.
  See below.

### 4. Solve, then read the residuals

The **Per-image residuals** panel sorts photos worst-first. Click a row to jump
there. That is the feedback loop fSpy cannot give you: with one image there is
nothing to compare against, but with a set, a bad control line shows up as one
photo whose residual stands out.

---

## Known objects, and camera arrays

If something in the scene has a shape you already know, say so and the solve gets
dramatically stronger. Give MultiSpy a CAD model, mark a few of its corners in
the photos, and it places that object as a rigid body.

**The arithmetic is the whole argument.** Twelve clicks on free features add
twelve unknown 3D positions — 36 unknowns the solve has to find. Twelve clicks on
a body whose geometry is known add **six**, because the only things not already
determined are where that body sits and which way it faces. Everything else was
settled by the measurement you already have.

And a known object carries its own size, so **one of them makes the whole
reconstruction metric** with no distance constraint and no tape measure.

### The case this was built for

Solving the layout of a **camera array**. Walk around the rig with a handheld
camera, mark each camera body against its CAD model, and the solve returns where
every camera in the array sits and which way it points.

```bat
.venv\Scripts\python.exe demo\make_rig_demo.py
.venv\Scripts\python.exe -m multispy demo_rig\project.multispy
```

Five identical 100 × 60 × 40 mm bodies on a 1.2 m frame, 225 mm apart, toed in
6° each, photographed six times. Add `--blank` to mark it yourself: the model,
its anchors, its optical frame and the five objects stay set up, and everything
you would have clicked is stripped. Note that control lines go too, and are still
required — known objects do not replace them, because the cameras' orientations
and the focal length still come from vanishing points. Tie points, on the other
hand, you can skip entirely once objects cover the scene.

Measured on that scene with 1 px of click noise:

| | |
|---|---|
| baseline error, 225–900 mm spacings | worst **1.4 mm** |
| body shape error against the model | **0.6 mm** on a 100 mm body |
| solved body size vs the model | within **0.3 %** |
| relative toe-in angle | within **1.2°** |

Note the gap between the position and angle figures, because it is a property of
the geometry and not a defect to be tuned away. A body's **orientation** is read
off a lever only as long as the body itself — 100 mm, seen from 1.3 m away — so
it inherits roughly *(distance ÷ body size)* times the error of a position.
Positions land near a millimetre; angles near a degree. Marking a **bigger**
object, or shooting **closer**, is what moves that number. More photos of the
same geometry does not.

### The workflow

**1. Add the model.** Drag an `.obj`, `.stl` or `.ply` onto the canvas, or use
**+ add model** in the Known objects panel. STEP, IGES and SLDPRT are parametric
formats needing a geometry kernel to read — convert once in Blender or FreeCAD
and the units come across.

**2. Check the units.** This is the one step that will quietly ruin everything if
you skip it. No mesh format records its units, so an STL exported in millimetres
and one in metres are byte-identical apart from the numbers. The panel shows what
the model's size works out to in metres and offers an opinion, but it never
guesses silently — get it wrong and the entire layout is scaled by 1000 while
every residual stays perfect.

**3. Pick anchors.** Press **D** for the 3D view, switch to **model**, and click
the points it offers. The only test that matters is whether you could find that
point again *in a photograph* — the middle of a flat face is unmarkable because
nothing in the picture locates it.

Three kinds are offered, drawn so you can tell them apart, because they are not
equally good and the difference is invisible in the geometry:

| | | |
|---|---|---|
| **ring** | the centre of a round or symmetric feature — a bored hole, a bolt head, a port | **the best kind.** Rotationally symmetric, so it reads identically from every angle, and you place it by eye at the centre of the visible ellipse. Not a vertex at all. |
| **cross** | where two straight edges would meet, if a fillet had not rounded them off | The sharp corner does not exist in the model, but it is well defined, and it is *steadier* than the rounded corner it replaces — whose apparent position slides with viewpoint and lighting. Mark it in the photo by extending the two visible edges by eye. |
| **dot** | a genuinely sharp vertex | Fine where they exist. Shell corners, square cut-outs. |

The panel reports the mix before you open the viewer, so a part with everything
filleted and no round features is obvious early.

**Do not mark the rim of a bolt head**, which is why those vertices are not
offered. Every one of them is a perfectly sharp corner and none is identifiable:
you cannot tell which of six hex corners you clicked, and in the next photo you
will click a different one. That mistake produces a plausible-looking solve whose
shape error never comes down. Its *centre*, by contrast, is one of the best
anchors on the part.

If the point you want is one you measured rather than one the model offers, **type
a position…** takes coordinates in the model's own frame. Worth having: a drawing
sometimes carries a datum the mesh does not, and a CAD export can round off
exactly the corner you were counting on.

**4. Add objects.** **+ object** per physical copy. Ten identical cameras are one
model and ten objects, which is the point: measuring the body once places all
ten, and every anchor marked in every photo refines the same shared geometry.

**5. Mark them.** Press **O**, pick a body, and click its anchors in the photos.
The next unmarked anchor on the same body follows automatically; `Tab` jumps to
the next photo. **Three anchors in two photos each** is the floor for a
placement — two leave the body free to spin about the line joining them, and the
panel says so rather than quietly returning one of the poses.

**6. Solve, then look at it.** Turn on **Models** and the model's wireframe is
drawn over the real object in the photo. That is the check no number gives you:
a residual table cannot tell you a body was placed 180° out, and this picture
can.

### The optical frame

Without one, the solve reports where a camera's *shell* is — which is not what
anyone wants to know. **edit optical** declares where the lens sits on the model
and which way it faces, and the layout then comes out as real camera extrinsics:
entrance pupil, optical axis, yaw and pitch.

Both are reported pairwise — baselines and relative angles — because that is how
a rig is specified, and because pairwise numbers do not depend on where the
arbitrary world origin ended up.

### Locking an object

Tick **lock** on one object and it stops being something to solve and becomes the
world frame: origin, orientation and scale all come from it. Re-solving after
adding photos then leaves the coordinates where they were, which matters once you
are iterating.

The catch is that the whole scene now hangs off that one body's orientation, and
orientation is the weak axis. On the demo, half a degree of frame rotation is
8 mm of error 900 mm away. Lock the best-determined body, and prefer a large one.

### What gets checked

Because the shape is known in advance, the solve can tell you things a free point
cloud cannot:

| Reading | Meaning |
|---|---|
| **shape error** | How far the marked anchors sit from the model's own geometry. A mis-clicked corner shows up here as a contradiction, instead of hiding inside an otherwise plausible reconstruction. |
| **solved size** | What this body came out as, relative to its model. Should be 1.000×. Anything else means an anchor is on the wrong feature, or this is not the object the model describes. |
| **scale** | Now reads `known object geometry (mm)` rather than `arbitrary`. |
| **stated vs object scale** | Only when you have *also* stated a real distance. Two independent measurements of the same thing, so their ratio is a genuine check rather than a restatement. 1.000× means they agree. |

### The units trap, and how it gets caught

A model declared in the wrong units is the one error that leaves every residual
looking perfect while the whole answer is out by a round factor. State **one**
real distance anywhere in the scene and the solve catches it:

```
the figures you stated and the known object geometry disagree about scale by a
factor of 10x (the stated figures want the scene smaller). The bundle adjustment
has split the difference, so neither is now exactly right. A near-round factor
almost always means the model's declared units are wrong.
```

Fixing it is the **size** row on the model's card in Known objects: the dropdown
sets the scale from a named unit, and the box beside it takes the model's longest
side as a plain number, in your nominal units, for the models the dropdown cannot
help with — a fixture whose file is in some drawing unit that is none of the five,
or a mesh somebody already scaled on the way out of CAD. Either one is also
`Shift`+left-drag while posing, when all you have is the silhouette against the
photograph. Whichever you use, every copy of the model keeps its own centre, so a
resize does not also fling the bodies across the scene — which it otherwise would,
since model coordinates are scaled about the CAD file's origin and a tooling
origin is routinely further from the body than the body is wide.

That comparison is made at **initialisation**, before the refinement runs, which
is the only point it means anything. Afterwards the bundle adjustment has already
traded the two sources of scale against each other, so a disagreement shows up as
a little of each being wrong and nothing reads as the culprit.

One mis-marked body does **not** drag the scene's scale: the per-object sizes are
combined by median with outlier rejection, so a body that disagrees with the
others is left out of the scale estimate and named, rather than turning one local
mistake into a global one.

---

## Reading the diagnostics

| Reading | Meaning |
|---|---|
| **reprojection RMS** | How well tie points fit. Should land near your clicking accuracy — 1–2 px. Above ~5 px, something is mis-clicked. |
| **control-line RMS** | Perpendicular pixel slip of control lines. Similar scale. If this is large while reprojection RMS is small, a control line is on a non-parallel edge. |
| **focal certainty** | Predicted 1-sigma uncertainty on the focal length, combining every calibrated image. Available before the solve, from the Focal check panel. Above ±8% everything downstream — camera positions, point depths — inherits that error. |
| **init conditioning** | Eigenvalue ratio of the linear stage. Below 10 means some image pair hangs on too few tie points, or your points are too coplanar. |
| **focal spread warning** | Per-image focal estimates disagreeing wildly. Expected to some degree — that is the fragility this tool exists to average out — but above ~100% go and check the control lines. |
| **scale** | `arbitrary` until you add a distance constraint or place a known object. The solve panel names which one set it. |
| **constraint error** | Per constraint, what the solve delivered against what you asked for, in world units. The only reading here that compares against something you asserted rather than clicked — and a datum missed by a millimetre is invisible in every other number on this list. |

Turn on **Grid** after solving, from the **show** menu in the toolbar. It draws the world XY plane anchored to the
floor of the point cloud, plus an axis gizmo. If the grid lies flat along the
real ground in the photo and converges correctly, the solve is right. (Draw your
vertical edges as **Z** for this to land on the ground.)

Turn on **Origin** once your points carry nominals. Where the grid is anchored
to the floor of the point cloud, this is anchored to the part: it draws the
drawing's own zero and axes through the solved camera, labelled with the axis
length in your nominal units. If that triad lands on the datum corner of the
real part, the frame the nominals defined is right; if the labelled 50 mm axis
spans half the photograph, your units are not what you think they are. It draws
nothing until three nominal'd points, not in a straight line, pin the frame —
before that the origin is a gauge choice and pointing at a pixel would be a
guess. When the origin falls outside the photo it says so at the border rather
than drawing a triad into the margin.

The same toggle governs the **3D** view, where the triad replaces the plain
origin gizmo and is drawn at the same dimension, measured from the points
rather than from the camera spread — so the length you read while orbiting is
the length you read on a photograph. Turn it off and the gizmo comes back: that
one marks the solved frame, which is worth seeing in a project with no nominals
and worth nothing more than that.

### How well is the origin actually known?

The origin is not a feature anyone clicked. It is wherever the nominal'd points
say it is, which makes it an *extrapolation* from the datum triangle — and
extrapolations get worse with distance. Three datum points spanning a metre,
carrying an origin a metre and a half beyond them, hand any angular error in
that triangle straight to the origin with a lever on it.

No number already in the solve report says so. The reprojection RMS does not,
because the marks are as well fitted as ever. The asked-versus-achieved datum
table does not either, because the fit puts the origin at (0,0,0) in its own
frame *by construction*, however poor the data is. A project can show a perfect
datum table and still not know where its origin is to a centimetre.

So press **measure** in the Points panel, beside the "In part space" note. It
re-solves the project a few dozen times with every mark and control-line
endpoint jittered by its stated sigma, and watches where the origin lands
relative to the scene. What comes back is a sphere you can see: on the
photographs as a circle around the triad, in the 3D view as three great
circles, labelled with its radius. The origin is inside it 95% of the time.

| | |
|---|---|
| **green / amber / red** | Against `origin_good` and `origin_warn`, set in the same panel, in your nominal units. The defaults — 5 and 20 — are a guess, not a standard. Set them to the tolerance the part is actually inspected to, or read the figure and ignore the colour. |
| **samples** | Each is a full re-solve, so this is the cost knob. 32 lands within about ±10% run to run, 64 within ±3%. Fewer than 8 refuses to grade at all. |
| **the ellipsoid** | Printed when the scatter is more than 2:1, which it usually is: error along the lever is far worse than across it. The sphere is the 95% ball, not a ball drawn around the long axis, so a very elongated scatter makes it a loose summary and the per-axis sigmas are the better read. |
| **(stale)** | The marks or nominals have changed since the measurement. The circle goes dashed and the figure describes the old data until you measure again. |

Two honest limits. It measures *click noise only* — a nominal that is simply
wrong, or a datum point clicked on the wrong feature in every photograph, moves
the origin without widening this sphere at all. And nominals derived from the
solve they then anchor are self-consistent by construction, so they inflate
nothing here while validating nothing either. Read it as a floor on the
uncertainty, never a ceiling.

### One tolerance per axis, because a single-axis feature has one

A third limit, now fixed, and worth reading before anybody changes it.

The defect was that **one nominal sigma was applied to all three axes.**
`nominal_constraints` expands a point's nominal into three axis locks, and they
all used to carry the same tolerance — so a single-axis feature, a Perceptron
`Range` or `OneDimensionMinRange` row which states `Disabled / Enabled /
Disabled` for exactly this reason, had its two uninspected axes pinned to the
tolerance of the one axis that *was* inspected. On PRM1 that was p5 at 0.5 mm
and p14 at 3.0 mm, across four axes between them that no sensor measured.

`TiePoint.nominal_sigma_axes` now carries a tolerance per axis. A `None` entry
falls back to `nominal_sigma` and then to `sigma_nominal`, so a coordinate
somebody typed off a drawing still wants one number and nothing about it has
changed. The calibration import reads the `Enabled`/`Disabled` columns and
offers the uninspected axes at `sensors.UNINSPECTED_AXIS_FACTOR` — ten times the
tolerance governing the inspected axis. **That factor is a judgement, not a
measurement**, and it says so where it is defined: the file states a tolerance
for the axes it inspects and nothing at all for the ones it does not, so there
is no number to read and this one was chosen. The pick-list says which axes the
cell actually inspects, and the Points panel shows the three tolerances in
force.

**Widened, not released**, and that distinction is the measured part. Releasing
the uninspected axes — a `None` coordinate, no lock at all — is worse than
holding them too tightly, and widening lands between the two. All three read off
PRM1 itself, 48 samples at seed 11, differing only in the sigma on p5's and
p14's X and Z (the axes the file marks `Disabled`):

| reading | origin radius | reproj RMS | weak direction |
|---|---|---|---|
| tight — the defect, as the axes were pinned | ±31.06 mm | 2.03 px | (0.72, −0.38, 0.59) |
| widened — ten times, on X and Z only | ±38.90 mm | 1.98 px | (0.32, −0.10, 0.94) |
| released — no lock on those axes at all | **±68.99 mm** | 1.94 px | (−0.85, −0.52, −0.05) |

Those are one consistent run, so read them against each other rather than
against the ±27.61 mm and ±60.67 mm quoted earlier in this section, which were
32 samples at a different seed. The conclusion is the same either way and the
ordering is what matters: releasing roughly doubles the radius and swings the
weak direction 64°. The photographs are indifferent to a 30 mm shift of p5 along
X, because every camera that marks it is looking down that axis — so those locks
are weak evidence and the only evidence those axes have.

Read the middle row honestly in both directions. Widening costs about a quarter
of the radius and swings the weak direction 35°, so it is not a free change:
part of the old ±31 mm was bought with four axis locks nobody ever measured, and
the viewpoint panel's advice moves with the direction. What it does not cost is
pixels. The reprojection RMS *falls* monotonically as those locks loosen — 2.03,
1.98, 1.94 — which is the clue to what the tight reading was doing.

**A tight tolerance on an uninspected axis does not report a miss. It wins.**
This came out the opposite way round from expected and is the more useful
finding for it. Measured on a synthetic scene with two single-axis features
whose uninspected X contradict each other by 25 mm each way:

| reading | worst datum miss | reproj RMS |
|---|---|---|
| honest nominals | 0.02 mm | 1.06 px |
| tight — 0.5 mm on the uninspected axis | 0.44 mm | 2.02 px |
| widened — 5.0 mm | 18.76 mm | 1.20 px |

The tight reading is the one with the clean asked-versus-achieved table, and it
got there by bending the scene until the reprojection RMS doubled. A reader is
told the part met its nominals and left to guess why the photographs got worse.
Widening does not hide a disagreement behind a looser tolerance; it stops one
being hidden behind a tighter one. Same mechanism as the warning in
`nominalcheck.py`, arrived at from the other end.

Two related things, since they are easy to trip over. `in_part_space` counts
only *fully* specified nominals — it used to count partial ones, and raised
`LinAlgError: SVD did not converge` on them, because `np.array([None, 527.1,
None], dtype=float)` is an object array. And `align_to_nominals` requires three
fully specified nominals, which is correct and was left alone: of PRM1's four
links only p15 and p16 state all three axes, so making either single-axis
feature partial would take the project out of part space with four points still
labelled "nominal" in the panel. The solve report now says so in a sentence,
with the count, rather than leaving it to an absent flag. All of this is
defended in `tests/test_nominalaxes.py`.

### Which photograph is holding the origin up, and where should the next one stand?

The sphere above says how big the uncertainty is, and its per-axis sigmas say
which direction is worst. Neither says what to do about it. Press **read
viewpoints** in the Viewpoints panel: it apportions the information along the
origin's weakest direction across the photographs that supply it. No re-solve,
so it is instant.

It exists partly to head off a reasonable instinct that turns out to be a trap.
A camera aimed down the part's X axis cannot see depth along X, so it looks like
a camera that should count for less when X is the weak direction. **Do not weight
it down.** A reprojection residual is in pixels, and the derivative of a pixel
with respect to a world displacement *along* the viewing ray is already nearly
zero — the Jacobian is a per-observation 3×3 information matrix and it has been
doing this the whole time. On the PRM1 cell the effect is stark: per mark, a
camera looking across the weak direction is worth **27×** one staring down it,
with no weighting applied anywhere. Discounting that second camera by its aim
would double-count the X it had already lost *and* throw away the Y and Z it
sees better than anything else in the set, because a scalar cannot tell the
axes apart.

| | |
|---|---|
| **\|cos\|** | How squarely a camera looks down the weak direction. Near 1 and it can say almost nothing about that direction; near 0 and it is looking straight across it. Exactly 1 contributes exactly nothing, however close it stands. |
| **share** | How much of the information along that direction the camera actually supplies. Sums to 100% across the set. |
| **/mark** | `share` divided by its mark count — what one *more* mark there is worth, which is the figure to compare cameras on. A dense photo can hold a large share while each mark in it is nearly worthless. |
| **left / +if marked** | Positioned points inside the frame that are not marked in it, and what marking them all would add. Usually the cheapest gain available, since it needs no new photograph. Occlusion is not modelled, so some may sit behind the part. |
| **a missing vantage** | A line of sight perpendicular to the weak direction — that is what makes it worth taking — and, among those, the one furthest from the lines of sight you already hold. Two positions are offered because a camera and its opposite pin a point identically; take whichever is not a wall. |
| **verdict** | `viewpoint-limited` when the error ellipsoid is lopsided enough that a new vantage can take some of it out. `evidence-limited` when it is nearly spherical, where no viewpoint is badly missing and the shortage is marks and nominals instead. |

Read the verdict before the gain. A gain is a ratio of information along one
direction, and sigma goes as its inverse square root, so half as much again is
about 18% off one axis — and the containment radius answers to the whole
ellipsoid, which barely moves. PRM1 reports a 48% gain from a camera looking
*down* at the cell, which nothing in its nine photographs does, and that is
still only 12.5 → 10.3 mm on one axis of a 27.6 mm radius. The panel says so
where it would otherwise flatter itself.

A reading, not a setting: it is cleared by any edit, because every figure is
read off the poses and marks as they stood when it ran.

### Is each nominal on the feature it names?

The section above ends on a limit: a datum point clicked on the wrong feature in
every photograph moves the origin without widening that sphere at all. This is
the check for that one.

A nominal is two claims stapled together — *this coordinate is right*, which
comes off the drawing, and *this clicked point is that feature*, which comes off
somebody's judgement at a screen. Only the first is ever examined. On hardware
that repeats, the second is the one that goes wrong: a frame near-symmetric left
to right, identical clamps down a tube, three sensor bodies across fifteen
stations. Pick `LH_CP_58` where `RH_CP_58` was meant and every number in the
file is still a real number off a real drawing. Only the pairing is wrong.

And the usual readouts go quiet when it is. `align_to_nominals` fits an origin
and a scale to a mislabelled set as readily as to a good one; the
asked-versus-achieved table then shows each nominal met, because the solve bent
itself to meet it; and a point whose feature carries no stated tolerance falls
back to `sigma_nominal`, tight enough to act as a rigid anchor — so the wrong
point is reported as the perfect one while the correctly toleranced points
absorb its error. Measured on a nine-photograph truck-frame cell, two
mislabelled points pinned at the 0.1 mm default held their own deviation at
0.0 mm, pushed the two honest nominals 97 mm and 210 mm out, took the
reprojection RMS from 2.4 px to 20 px, drove the shared focal to 2.37× its EXIF
value, and stopped the bundle adjustment converging. Every one of those says
something is wrong. Not one says what.

So this asks a question the fit cannot answer about itself: forget the frame,
forget the units, and compare **shapes**. Distance is unchanged by rotation and
translation, and one distance over another cancels the scale — so a correctly
linked set agrees with itself on a single scale, pair after pair, and a swapped
one does not. The two features the drawing puts closest together stop being the
two the photographs put closest together, and no rigid motion repairs it.

**It re-solves before it compares, and that is the whole method.** Run the
comparison against a solve the nominals are driving and it passes trivially —
they are what put those points there, so they agree with themselves by
construction, the mislabelled ones most tightly of all. The project is copied,
every nominal stripped from the copy, and the copy re-solved from the
photographs alone. Scale goes arbitrary when the nominals go, which costs this
nothing, and it means the check works on a project that has never reached part
space — which is exactly when you want it.

| | |
|---|---|
| **spread** | Largest pair ratio over smallest. 1.0 is a set that agrees with itself. This is the headline because it owes nothing to a fit: no rotation and no origin were chosen before computing it, so nothing can absorb the error. |
| **rms**, **worst sigma** | The similarity fit's residual, per point, against the tolerance in force. In sigmas because one calibration file's tolerances run 0.5 mm to 8 mm, and a fixed millimetre threshold would excuse the tight features or condemn the loose ones. |
| **precision** | The finest agreement these photographs could show at that point — each view constrains it across its own ray by about `rms_px / focal` radians at that view's depth, summed over the views that see it. The tolerance in force is the **larger** of this and the drawing's. Without it, any feature toleranced finer than the camera can resolve reads as contradicted for ever: `RH_CP_58` is held to 0.5 mm, nine phone photographs place a point to about 2 mm, and a perfectly correct link came back at 20 sigma. It takes the baseline into account and not just the pixel error, because a point in two nearly-parallel views is barely placed along the viewing direction however sharp its marks are — the first version of this floor assumed otherwise and convicted a two-view point for weak geometry. |
| **suspect** | The one point whose removal turns a failing set consistent, when there is exactly one. A similarity fit splits one bad link's error across every point, so the residuals name nobody; refitting without each in turn does. Silent when nothing single explains it — two bad links, or a swap, which is two by definition — and silent at three points, where dropping one leaves two and any two points fit each other. |
| **consistent / suspect / contradicted** | Worst point within 3× its tolerance, within 5×, or beyond. |
| **untestable** | A real answer, not a failure. Fewer than three nominal'd points, a set too near collinear, too few of them placeable from the photographs alone, or a solve that failed. |
| **reproj** | The nominal-free re-solve's own RMS. Read it first: if the photographs do not agree with each other, their disagreement with the drawing says nothing yet about the links. |

Why both readings are kept: a swap along a plate's 20 mm thickness barely moves
the spread — about 6% — while the shape residual goes to forty times tolerance.
A swap across its 120 mm length moves the spread by a factor of 37. Either
number alone has a blind spot the other covers.

Three limits worth stating. It cannot see an error that **preserves shape**: a
whole set shifted together, or nominals taken from the solve they then anchor.
Below three points it declines rather than passes, because two points and a free
scale fit each other whatever they are, and a pass there would be the pass
somebody quotes. And naming a suspect is a place to look, not a verdict — one
link wrong badly enough drags its neighbours, so confirm it against the
photographs before acting on it.

Read the **magnitude**, not only the colour. A swapped feature puts the residual
a hundredfold or more above the precision figure; a few times above it is a
marking or a geometry problem instead — a point in too few views, or in views
too close together — and the report says which of the two it is looking at.

### Is the sensor calibration file telling the truth?

A measurement cell's calibration file states, per sensor, where it sits, which
way it faces, how far it stands off the part and how big its field of view is.
Those numbers would make superb nominals — the sensors are spread across the
whole cell, so they shorten the lever arm that makes an origin uncertain. The
catch is that calibration files lie, and the ones that lie worst look exactly
like the ones that do not.

So the file is read as a set of claims and checked before any of it is believed.
The check needs nothing else — no reconstruction, no photographs, no CAD —
because **the file falsifies itself**. It states each sensor's field of view
*and* the nominal of the feature that sensor is pointed at. If the feature lies
outside the field of view, the sensor cannot see the thing it claims to be
measuring, and that is a contradiction inside a single row.

Point it at an export and it reports, per sensor:

| | |
|---|---|
| **consistent** | Every feature it measures sits inside the field of view it declares. Read this narrowly: the box is 200–300 mm across and the tolerance you inspect to is a few millimetres, so it rules out a gross fault and says nothing finer. |
| **edge** | A feature sits just outside the box, by less than 1.3×. Marginal rather than damning — a feature nominal slightly out, or a sensor that was always tight on that feature. |
| **explained** | Wrong, and one catalogued fault accounts for it: the mirrored feature across the cell, a pose reflected through the centreline, a body turned about an axis. The useful verdict, because it comes with the correction. |
| **suspect** | Wrong, and nothing in the catalogue explains it. It still says whether a single translation would fix it — displaced rather than misoriented — and which axis the overrun is on, since an overrun along the sight line is a standoff error and one across it means the sensor is pointed somewhere else. |

**The most useful line in the report is about a day, not a sensor.** Faults in
these files are made by people in sittings: the part is set up once, the nominals
are loaded once, and every sensor calibrated that day inherits whatever was wrong
with the setup. So sensors are grouped by `Calibration Date`. On the cell this
was built against — 15 sensors, 22 features — three sensors are wrong by 2.3×,
4.8× and 7.5× of their own field of view, all three were calibrated on the same
morning, their corrections agree on X to within 1 mm at +411, +411 and +412, and
no sensor from any other session is wrong at all. That is one mistake in one
setup, fixed in one action. Three separate sensor verdicts would have been three
things to go and check.

What it will not do is tell you whether the sensors moved or the nominals they
were calibrated against did. A shared offset looks identical either way from the
file alone, and both readings are named rather than one being chosen.

The range axis is not assumed either. A triangulation sensor trades depth for
resolution, so its depth half-range is smaller than its lateral extents, and
every sensor votes from its own frustum. A sensor whose field of view is square
in cross-section abstains rather than voting for whichever axis came first.

Three limits, and the third is the one that matters most:

- **A 180° turn about the aim axis is invisible here.** The wrong sweep face
  moves neither the sight line nor the field of view, so it comes back
  *consistent* — correctly, since nothing in the file can see it. Every verdict,
  including the passing ones, carries the record that this fault was never
  tested. Only a photograph can catch it, and only a close one: these bodies are
  70 × 168 mm in section, so the turn maps the outline onto itself and leaves
  only off-centre detail to tell them apart.
- **A translation along the sight line is nearly invisible**, for the same
  reason. The depth half-range gives the test some power there, far less than
  across the beam.
- **A fault shared by most of the file reads as the truth.** This measures
  agreement, never correctness. The independent referees are the control-line
  axes, the clicked tie points, the mounting structure the sensors physically sit
  on, and the photographs.

Importing is read-only and does not touch the project. Turning imported poses
into nominals and then judging the fit by how well those nominals were met would
be circular — the same trap the confidence sphere above is built to avoid — so
reading the file and believing it stay separate actions. `sensor_tol` and
`sensor_warn` (10 and 50 in your nominal units) grade the corrections; they do
not set the conviction threshold, which comes from each sensor's own declared
optics.

### Asking the photographs whether the sensor is really there

The check above can only make the calibration file contradict itself, and two
things are beyond it by construction: a 180° turn about the aim axis moves
neither the sight line nor the field of view, and a transform wrong by less than
one field of view is wrong by less than the only ruler that check has. For both,
the photographs are the last referee.

The cameras are already solved, so a body at a known pose projects into every
photograph with no search in it — one matrix multiply per vertex. Put each
sensor where the file says it is, draw its wireframe into every photo, and look.

**The score is never the verdict.** `edgefit.py` documents measuring a chamfer
cost converge confidently onto a bookshelf, and the photographs this was written
for are of a tube frame where every surface is an edge. So two referees the
score does not get to influence decide instead:

| | |
|---|---|
| **agreement across views** | A body that is really there scores well in *every* photograph that can see it. A coincidental alignment with a piece of tube does not repeat from sixty degrees away, because the coincidence belonged to that one projection. The strong referee, and free once the geometry has run. |
| **a null distribution per photo** | The same wireframe is scored at a spread of deliberately wrong nearby offsets, which measures what "nothing there" scores *in this photograph's own clutter*. The figure reported is a z-score against that, because a raw cost of 3 px is excellent against a wall and unremarkable against a tube frame. |

Verdicts are **found**, **absent**, **displaced** and **undecided**. Measured on
a synthetic scene with ground truth — three claimed bodies, one rendered where
it is claimed, one rendered 40 mm away, one not rendered at all — the found
body came back at z 5.2 with every view agreeing, the absent one at z 0.3 with
none, and the displaced one was located at **44 mm against the 40 mm injected**.
That last case is the useful one: a body found strongly a few centimetres off is
a measured correction to the calibration, not a failed check.

**Marking is faster with the file open.** A cell this size names a dozen or more
features with part-space coordinates already in it, and the import offers them in
the Points panel as a list to pick from — choose `LH_CP_76` and its coordinate,
its own tolerance from the file's spec limits, and a record of where it came from
all arrive together. Forty-two numbers typed by hand is forty-two chances to
transpose a digit, and a transposed digit in a nominal does not read as a mistake:
it reads as a part that is out of tolerance. Features whose sensor was convicted
are listed too, disabled, with the reason — that verdict says one of the
transform and the nominal is wrong without saying which, so neither is safe to
mark until it is resolved, and hiding the feature would only invite someone to
type the number in anyway.

**Where this contradicts the file, that is the output.** A sensor the file calls
consistent that no photograph can find, or one the file convicts that every
photograph finds exactly where it claims — meaning the feature nominal is wrong
rather than the sensor — is worth more than any number of agreeing verdicts.
They are collected separately rather than averaged into a summary.

Four limits, all of them real on this cell:

- **Small in frame defeats it.** A 70 × 168 mm body at four metres is a few
  dozen pixels, which is not enough silhouette to tell from a bracket. Under
  25 px it reports undecided, never absent.
- **Sensors look alike.** Every sensor of one model is the same body, so a
  wireframe landing on the *neighbouring* sensor scores beautifully in every
  view. Cross-view agreement cannot catch that one — the neighbour is in every
  photo too — and only the displacement figure can.
- **One photograph is never a verdict.** A sensor seen once is reported as seen
  once.
- **A confident score is not proof.** All of this is evidence that edges are
  present where a body would put them, which is a narrower claim than the body
  being there.

**Residuals** draws observed → reprojected vectors at **×10**, since a good fit
is otherwise invisible.

---

## Exports

**Export** writes, next to the project:

| Output | Use |
|---|---|
| `export/sparse/` | COLMAP sparse model (text). Read by COLMAP, GLOMAP, Meshroom, `ns-process-data`. |
| `export/import_cameras.py` | Blender script. Builds the cameras with each photo wired up as that camera's background image, and imports every placed object's CAD model at its solved pose, with a camera object for each declared optical frame. Run via Scripting → Open → Run Script. |
| `export/transforms.json` | nerfstudio / instant-ngp. |
| `export/tiepoints.ply` | Tie points, camera centres and object anchors, each in its own colour, for a quick look in MeshLab or CloudCompare. |
| `export/scene.step` | **The whole scene as one STEP assembly** — see below. |
| `export/rig.json` | Every placed object's pose and optical frame, plus the pairwise layout. Written only when something is placed. |
| `export/rig.csv` | The same layout as a flat table — position, optical axis, yaw and pitch per object — for a spreadsheet or a spec sheet. |

The COLMAP export is the useful one for this workspace: a hand-marked solve can
feed Meshroom's dense stage, or be handed to COLMAP's own bundle adjuster to
refine further.

Verified end to end — `colmap model_analyzer` on the demo export reports 6
registered images, 17 points, 70 observations, 0.487 px mean reprojection error.

### The STEP scene

**STEP** in the toolbar writes `export/scene.step` on its own — the one export
a CAD seat wants, without the COLMAP model and the Blender script beside it.
(In the browser build it downloads the file directly.) **Shift-click** for the
wireframe, which is the escape hatch for a CAD system that will not read AP242;
see the table below. The **Export** button still writes everything, `scene.step`
included.

Everything the solve knows, in one file and one coordinate system, each a named
node in the assembly tree:

| Node | What it is |
|---|---|
| `cam_<photo>` | One camera per solved photo, drawn as its frustum at the solved pose. The notch on the frustum marks the top of the image, so roll is visible rather than implied. |
| `P4`, `p7`, … | One node per tie point, named as you named it, at its solved position. |
| `P4_nominal`, … | The same points where the drawing states them, in green. The gap between the two *is* the residual, measurable with the CAD system's own dimension tool. |
| `<part> (part)` | The part, at the origin — which is where it is by definition once the nominals have put the scene into part space. |
| placed bodies | Every placed object at its solved pose, with its anchors marked and a frustum for any declared optical frame. |
| `<name> (unplaced)` | A CAD template nothing has placed yet, parked at its own model origin. Pass `unplaced=False` for a file that states only what was measured. |
| reference planes | Your datum surfaces — see below. |
| `MS_Origin` | The world axis triad. |

#### Each body is its own part, with one transform above it

A placed object is a node holding the measurement, and the geometry hangs
underneath it:

```
MultiSpy_prm1                  ← part space: the frame the nominals define
 └─ Tooling                    ← local → part-space transform lives HERE
     ├─ Tooling_body           ← the mesh, at its own model origin
     ├─ Tooling_anchors        ← the anchors marked on it
     └─ Tooling_optical        ← frustum, if the template declares one
```

Nothing is ever baked into world coordinates. `Tooling_body` sits at the
identity in the coordinates its OBJ or STL was authored in, and the only thing
standing between it and part space is one rigid transform on the node above.

That is what makes a model swappable: a better CAD file of the same part, built
on the same origin, drops straight into the `_body` node, and the transform —
which is the measurement, and the expensive part — never moves. The same
structure is why ten identical cameras in an array cost one mesh and ten
transforms rather than ten copies of the mesh.

#### Reference planes

Datum surfaces you type in, under **Reference planes** in the panel: a name, an
axis and an offset, in the project's nominal units and in part space — the same
numbers you type for a point nominal, so a plane at `Z = 995` lands where a
nominal of 995 does. They save with the project and go into every export.

They are the one thing in the file that is not evidence. Nothing here enters
the solve, and a plane has no residual to report, because it states where a
datum *is* rather than claiming to have found one. It exists to be dimensioned
against once the scene is open in CAD.

Each one comes through as a genuine bounded planar surface — a `FACE_SURFACE`
on a `PLANE` — so CAD can select it, offset from it, project onto it and
measure to it. Not four lines in the shape of a rectangle, which can do none of
those things. They are sized to the scene unless you give one a size.

Written in **millimetres** by default, because that is what a drawing is
dimensioned in, while the scene is solved in metres; the file's unit context
says which, so a CAD system converts rather than guesses. Colour separates the
kinds: cameras red, points white, nominals green, anchors blue, the part grey.

How CAD bodies are carried is a choice, because no single answer is read by
everything:

| `--step-shape` | Geometry | Schema | Read by |
|---|---|---|---|
| `tessellated` *(default)* | Every triangle, as one AP242 tessellated shell — two entities for a whole body, so a 84,000-triangle casting is a few megabytes | AP242 | OpenCASCADE 7.5+ (FreeCAD, most viewers), NX, CATIA, Creo |
| `wireframe` | The body's feature edges as polylines — the same edges MultiSpy draws over a photo | AP214 | everything, back to AP203 |
| `brep` | A real faceted solid, one planar face per triangle — about nine entities per triangle, so it is capped at 40,000 | AP214 | everything, and properly solid |
| `none` | Placements only: the tree, the poses, the cameras and the points | AP214 | everything |

If a CAD seat opens the default with the cameras present and the bodies
missing, it does not do AP242 tessellated geometry — shift-click **STEP**, or
re-export with `--step-shape wireframe`:

```
.venv\Scripts\python.exe -m multispy export projects\mypart\project.multispy --step-shape wireframe
```

`--step-units mm|cm|m|in|ft` picks the file's units.

Verified by reading the output back with OpenCASCADE 7.9: zero schema errors on
a full check, every camera centre within 5 nm and every camera axis within
4e-10 of the solve, and the same scene written in mm, cm, m and inches reading
back to identical coordinates. The writer is pure Python — no geometry kernel,
no new dependency.

---

## Keyboard

| Key | |
|---|---|
| `L` `P` `O` `G` `S` | Lines / Points / Objects / Pose / Select mode |
| `1` `2` `3` | X / Y / Z axis, for control lines |
| `4` | Step through the direction groups |
| `N` | New tie point |
| `Tab` | Next photo missing the active point, or the active object's anchors |
| `←` `→` or `[` `]` | Previous / next photo |
| `V` `R` `E` `M` `H` | VP rays / residuals / expected / models / shaded, from the **show** menu. Grid and Origin are in that menu too and have no key — `G` is pose mode. |
| `D` | 3D view (orbit the scene, or a model to pick anchors off) |
| `F` | Fit view — the 3D view when it is open, otherwise the photo |
| `Enter` | Solve |
| `Ctrl+S` | Save |
| `?` | Open the tutorial (also the **?** button) |
| `Delete` | Remove whatever is under the cursor |
| wheel | Zoom at cursor |
| space-drag, middle-drag, alt-drag | Pan |
| right-click | Delete a marker or line |
| left-click a drawn line | Reverse its direction — the sign of that axis |

**In the 3D view**, which claims these keys while it is open:

| Key | |
|---|---|
| `X` `Y` `Z` | Put the eye on that axis; `Shift` for the far side, so `Shift+Z` looks straight up the +Z axis at the underside of inverted tooling |
| `Shift+F` | Frame the selection only — the one to use when a fixture, or a CAD origin far from its own geometry, makes the whole-scene box mostly empty |
| `T` | Turntable (horizontal drag turns about world Z) ↔ free tumble (about whatever is up on screen) |
| `Q` `E` | Roll the view, `Shift` for bigger steps |
| `R` | Level the horizon |
| drag · ctrl-drag · shift-drag · wheel | Orbit · roll · pan · zoom. The orbit is not clamped: it passes through straight-up and straight-down and carries on. |

**Posing a body** (`G`), where the drags act on the object rather than the view:

| | |
|---|---|
| left-drag · right-drag | Move it in the image plane · turn it |
| hold `S` + drag | Its distance along the view ray — this placement only, not the model's size |
| hold `X`/`Y`/`Z` + drag, or `←` `→` | Turn about one world axis; the arrows step exactly 15°, with `Shift` exactly 90° |
| `Shift`+left-drag | Resize the **model** — every copy of it, in every photo. Each copy keeps its own centre, so nothing jumps, and the gesture is the same from any view. Type an exact figure in the **size** box in Known objects when you know one. |
| `Shift`+right-drag | Roll it in the picture plane |
| `T` | Switches the free turn between world Z and a free tumble, the same switch the 3D view uses |
| `H` | Bring a body that has drifted off this photo back into frame |

---

## Headless

```bat
.venv\Scripts\python.exe -m multispy solve demo\project.multispy
.venv\Scripts\python.exe -m multispy export demo\project.multispy out\
.venv\Scripts\python.exe -m multispy exif demo\project.multispy
.venv\Scripts\python.exe -m multispy rig demo_rig\project.multispy
```

`solve` prints the full per-image residual table, including each image's
predicted focal uncertainty, plus a per-object table and the layout. `exif`
prints what the photo files claim, per image and per camera group. `rig` prints
the solved object layout on its own, or the whole payload with `--json`.

## Tests

```bat
.venv\Scripts\python.exe tests\test_synthetic.py
.venv\Scripts\python.exe tests\test_rig.py
.venv\Scripts\python.exe tests\test_exif.py
```

`test_synthetic` builds a Manhattan-world scene with known ground truth,
generates only what a user could plausibly have clicked (control lines on
visible box edges, tie points on visible corners, with realistic noise), and
checks recovery of focal length, rotations, camera positions and metric scale.
It also checks that the focal-observability prediction brackets the error the
solver actually makes, in both directions — a diagnostic that is over-confident
is worse than none.

`test_rig` builds the camera-array scene — a frame with five toed-in camera
bodies — and checks the layout comes back: baselines, relative angles, metric
scale from object geometry alone, and the world frame a locked object defines.
It also checks the *reporting*: that swapping two anchors on one body surfaces as
a shape error on that body and does not contaminate the others, and that two
anchors are refused as a placement rather than silently resolved.

`test_exif` writes JPEGs carrying known EXIF tags and checks each extraction
route, including the cases that must be *refused*: a cropped frame and a group
whose lens was zoomed between shots.

All three are pytest-compatible and all three run standalone.

---

## Where this fits, and where it does not

Use MultiSpy when the scene has **man-made parallel structure** and too little
overlap for automatic SfM.

**Every image still needs control lines.** A known object fixes scale and places
itself, but the camera *rotations* still come from vanishing points, so each
photo needs two axes with two control lines each. Recovering a camera's rotation
from a placed object instead — ordinary pose estimation, once three anchors are
marked — would lift that requirement and is the obvious next thing to add; it is
not there yet. In practice the rig or frame an array is mounted on supplies the
parallel edges, which is why the demo scene has one.

It will not help with organic scenes — trees, terrain, rock — because there are
no parallel edges to trace. For those, the alternatives worth reaching for are
hloc with LightGlue or LoFTR (keeps your existing COLMAP pipeline, just replaces
the matcher), or a pose-free learned reconstructor like VGGT or MASt3R-SfM.

It also does not need to be all-or-nothing. A reasonable hybrid: solve the
cameras here, export to COLMAP format, then hand that to COLMAP's triangulator
and bundle adjuster to densify with automatic matches that now have a known pose
prior to work against.

## Layout

```
multispy/
  geometry.py   vanishing points, focal and rotation from control lines, the
                focal-observability error propagation, and the similarity fit
  mesh.py       OBJ / STL / PLY loading, welding, feature edges and corners
  exif.py       focal length from photo metadata, with confidence
  solve.py      calibration, linear initialisation, known-object placement,
                bundle adjustment
  project.py    data model and JSON persistence
  stepfile.py   a small ISO 10303-21 writer: assembly tree, placements,
                wireframe, tessellated meshes, faceted solids, colour
  export.py     COLMAP, Blender, nerfstudio, PLY, STEP scene, rig layout
  webapp.py     every operation behind the UI, independent of transport --
                shared by the local server and the browser build
  server.py     local HTTP server: transport only, JSON in and out
  docpage.py    renders these docs as pages the app links to (/docs/...)
  __main__.py   CLI
web/
  app.js        marking UI (canvas, vanilla JS)
  viewer3d.js   3D view, a software renderer on a 2D canvas
  backend.js    sends the UI's requests to the local server, or to Pyodide
  config.js     which of the two this page is talking to
tools/
  build_static.py  assembles the browser build into dist/
tests/          synthetic ground-truth validation
demo/           synthetic scene generators (box scene, camera array)
projects/       your projects, one folder each (created on first run)
```

`webapp.py` is the reason the browser build is the same application rather than a
copy of it: both front ends call the same `Workspace`, and only the transport
differs.

A project folder holds `project.multispy`, its `photos/`, and `models/` for any
CAD models added to it, so it stays self-contained and portable.
